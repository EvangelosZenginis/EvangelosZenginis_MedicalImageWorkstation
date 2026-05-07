import io
import base64
import math
from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon, iradon, iradon_sart
from matplotlib.backends.backend_pdf import PdfPages

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
except Exception:
    pydicom = None

st.set_page_config(
    page_title="BME Medical Imaging Workstation ",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root { --line: rgba(255,255,255,.12); --muted: #aeb4c2; }
    .stApp { background: radial-gradient(circle at top left, rgba(124,58,237,.22), transparent 32%), radial-gradient(circle at 80% 0%, rgba(14,165,233,.16), transparent 28%), #0b0f17; color: #f7f7fb; }
    .block-container { padding-top: 1.4rem; max-width: 1500px; }
    h1 { font-size: 2.65rem !important; font-weight: 900 !important; letter-spacing: -0.04em; }
    h2, h3 { font-weight: 850 !important; letter-spacing: -0.025em; }
    .hero, .panel { border: 1px solid var(--line); background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.025)); border-radius: 1.1rem; padding: 1rem 1.15rem; margin-bottom: .8rem; }
    .hero p, .muted { color: var(--muted); }
    div[data-testid="stImage"] img { border-radius: .75rem; border: 1px solid rgba(255,255,255,.08); }
    .stDownloadButton button, .stButton button { border-radius: .7rem; font-weight: 750; }
</style>
""",
    unsafe_allow_html=True,
)

TONES = 256
IMAGE_DEPTH = 255

import base64
import streamlit.components.v1 as components

def _image_to_data_url(im):
    import io
    from PIL import Image
    arr = to_uint8_image(im)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def live_before_after_viewer(original, processed, start=50, height=700):
    before_url = _image_to_data_url(original)
    after_url = _image_to_data_url(processed)

    html = f"""
    <style>
      .frame {{
        position: relative;
        width: 100%;
        max-width: none;
        margin: auto;
        overflow: hidden;
    }}
    .frame img {{
        width: 100%;
        height: auto;
        display: block;
    }}
    .after {{
        position: absolute;
        top:0;
        left:0;
        height:100%;
        clip-path: inset(0 0 0 {start}%);
    }}
    .divider {{
        position:absolute;
        top:0;
        bottom:0;
        left:{start}%;
        width:3px;
        background:red;
        transform:translateX(-50%);
    }}
    </style>

    <div class="frame" id="frame">
        <img src="{before_url}">
        <img src="{after_url}" class="after" id="after">
        <div class="divider" id="divider"></div>
    </div>

    <script>
    const frame = document.getElementById("frame");
    const after = document.getElementById("after");
    const divider = document.getElementById("divider");

    frame.onmousemove = function(e){{
        const rect = frame.getBoundingClientRect();
        let x = (e.clientX - rect.left) / rect.width * 100;
        x = Math.max(0, Math.min(100, x));

        after.style.clipPath = `inset(0 0 0 ${{x}}%)`;
        divider.style.left = `${{x}}%`;
    }}
    </script>
    """

    components.html(html, height=height)
# ---------------------------- state ----------------------------
st.session_state.setdefault("outputs", {})
st.session_state.setdefault("output_meta", {})
st.session_state.setdefault("output_inputs", {})
st.session_state.setdefault("history", [])
st.session_state.setdefault("dicom_meta", {})
st.session_state.setdefault("pipeline_input", None)
st.session_state.setdefault("pipeline_source", "Original / noisy input")

# ---------------------------- utilities ----------------------------
def normalize(im: np.ndarray, tones: int = TONES) -> np.ndarray:
    im = np.asarray(im, dtype=float)
    mn, mx = np.nanmin(im), np.nanmax(im)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(im, dtype=float)
    return np.clip(np.round((tones - 1) * (im - mn) / (mx - mn)), 0, tones - 1)

def ensure_2d_image(arr):
    arr = np.asarray(arr)

    # Remove empty/singleton dimensions
    arr = np.squeeze(arr)

    # If DICOM has multiple frames, keep the first frame
    if arr.ndim == 3:
        # Case RGB/RGBA image: rows x cols x channels
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
            arr = (
                0.299 * arr[..., 0] +
                0.587 * arr[..., 1] +
                0.114 * arr[..., 2]
            )
        else:
            # Case frames x rows x cols
            arr = arr[0]

    # If still more than 2D, keep first available slice
    while arr.ndim > 2:
        arr = arr[0]

    return arr.astype(float)

def resize_like(im, reference):
    im8 = to_uint8_image(im)
    ref8 = to_uint8_image(reference)

    if im8.shape == ref8.shape:
        return im8

    resized = Image.fromarray(im8).resize(
        (ref8.shape[1], ref8.shape[0]),
        resample=Image.Resampling.BILINEAR
    )

    return np.asarray(resized, dtype=np.uint8)

def crop_nonblack_area(im, threshold=5, padding=5):
    arr = to_uint8_image(im)

    # Mask για να κρατήσουμε μόνο τα μη μαύρα pixels
    mask = arr > threshold

    # Αν δεν βρεθεί τίποτα, επέστρεψε την εικόνα όπως είναι
    if not np.any(mask):
        return arr

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    r1, r2 = rows[0], rows[-1]
    c1, c2 = cols[0], cols[-1]

    # Μικρό padding για να μην κοπεί η εικόνα πολύ κοντά
    r1 = max(r1 - padding, 0)
    r2 = min(r2 + padding, arr.shape[0] - 1)
    c1 = max(c1 - padding, 0)
    c2 = min(c2 + padding, arr.shape[1] - 1)

    cropped = arr[r1:r2 + 1, c1:c2 + 1]

    return cropped

def to_uint8_image(im: np.ndarray) -> np.ndarray:
    im = ensure_2d_image(im)
    return np.asarray(np.clip(normalize(im), 0, 255), dtype=np.uint8)


def load_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None, {}
    data = uploaded_file.read()
    uploaded_file.seek(0)
    meta = {}
    if uploaded_file.name.lower().endswith(".dcm"):
        if pydicom is None:
            st.error("Για DICOM χρειάζεται: pip install pydicom")
            return None, {}
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ensure_2d_image(ds.pixel_array)
        # Apply rescale slope/intercept when present.
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + intercept
        keys = ["PatientName", "PatientID", "Modality", "StudyDate", "SeriesDescription", "Rows", "Columns", "PixelSpacing", "SliceThickness", "BitsAllocated", "BitsStored", "RescaleSlope", "RescaleIntercept"]
        for k in keys:
            if hasattr(ds, k):
                meta[k] = str(getattr(ds, k))
        return normalize(arr), meta
    img = Image.open(io.BytesIO(data)).convert("L")
    return normalize(np.asarray(img, dtype=float)), meta


def image_download_bytes(im: np.ndarray, fmt: str):
    im8 = to_uint8_image(im)
    fmt_l = fmt.lower()
    if fmt_l == "dicom":
        if pydicom is None:
            raise RuntimeError("pydicom is not installed. Run: pip install pydicom")
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Anonymous"
        ds.PatientID = "000000"
        ds.Modality = "OT"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.ContentDate = datetime.now().strftime("%Y%m%d")
        ds.ContentTime = datetime.now().strftime("%H%M%S")
        ds.Rows, ds.Columns = im8.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = im8.tobytes()
        buf = io.BytesIO()
        ds.save_as(buf, write_like_original=False)
        return buf.getvalue(), "processed_image.dcm", "application/dicom"
    pil_fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "bmp": "BMP"}[fmt_l]
    buf = io.BytesIO()
    Image.fromarray(im8, mode="L").save(buf, format=pil_fmt)
    mime = "image/jpeg" if fmt_l in ("jpg", "jpeg") else f"image/{fmt_l}"
    return buf.getvalue(), f"processed_image.{fmt_l}", mime


def record_output(name, image, operation, params, input_image=None, input_source=None):
    """Save a processed result plus the exact input image used to create it."""
    st.session_state["outputs"][name] = image
    if input_image is not None:
        st.session_state.setdefault("output_inputs", {})[name] = np.asarray(input_image).copy()
    st.session_state["output_meta"][name] = {
        "operation": operation,
        "params": params,
        "time": datetime.now().strftime("%H:%M:%S"),
        "input_source": input_source or st.session_state.get("pipeline_source", "Original / noisy input"),
    }
    entry = f"{datetime.now().strftime('%H:%M:%S')} | {name}: {operation} | {params}"
    if not st.session_state["history"] or st.session_state["history"][-1] != entry:
        st.session_state["history"].append(entry)
        st.session_state["history"] = st.session_state["history"][-30:]


def plot_histogram_pair(original, processed, title="Histogram comparison"):
    a = to_uint8_image(original)
    b = to_uint8_image(processed)

    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0])))

    # Ignore very dark background pixels
    mask = (a > 5) | (b > 5)

    a_vals = a[mask]
    b_vals = b[mask]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.hist(a_vals.ravel(), bins=256, range=(0, 255), alpha=0.55, label="Original")
    ax.hist(b_vals.ravel(), bins=256, range=(0, 255), alpha=0.55, label="Processed")

    ax.set_title(title + " - background excluded")
    ax.set_xlabel("Gray value")
    ax.set_ylabel("Pixels")
    ax.grid(True, alpha=.25)
    ax.legend()

    return fig


def before_after_split(original, processed, split_percent=50, scale=0.65):
    """Create a lightweight before/after preview for smoother Streamlit slider updates."""
    a = to_uint8_image(original)
    b = to_uint8_image(processed)
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0])))

    # Downscale only the preview, not the actual processed/exported image.
    if 0 < scale < 1:
        new_w = max(1, int(a.shape[1] * scale))
        new_h = max(1, int(a.shape[0] * scale))
        a = np.array(Image.fromarray(a).resize((new_w, new_h)))
        b = np.array(Image.fromarray(b).resize((new_w, new_h)))

    x = int(a.shape[1] * split_percent / 100)
    out = a.copy()
    out[:, x:] = b[:, x:]
    out[:, max(0, x - 1):min(a.shape[1], x + 1)] = 255
    return out


def add_noise(im, mode, amount, seed=7):
    rng = np.random.default_rng(seed)
    base = to_uint8_image(im).astype(float)
    if mode == "None":
        return base
    if mode == "Gaussian":
        return normalize(base + rng.normal(0, amount, base.shape))
    if mode == "Salt & Pepper":
        out = base.copy()
        p = amount / 100.0
        r = rng.random(base.shape)
        out[r < p / 2] = 0
        out[(r >= p / 2) & (r < p)] = 255
        return out
    if mode == "Speckle":
        return normalize(base + base * rng.normal(0, amount / 100.0, base.shape))
    if mode == "Poisson":
        vals = 2 ** np.ceil(np.log2(len(np.unique(base))))
        return normalize(rng.poisson(base * vals / 255.0) * 255.0 / vals)
    return base


def quality_metrics(original, processed):
    a = to_uint8_image(original).astype(float)
    b = to_uint8_image(processed).astype(float)
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]))).astype(float)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    try:
        ssim_val = float(ssim(a, b, data_range=255))
    except Exception:
        ssim_val = float("nan")
    mae = float(np.mean(np.abs(a - b)))
    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim_val}


def image_statistics(im):
    arr = to_uint8_image(im)
    return {
        "Mean": float(np.mean(arr)),
        "Std": float(np.std(arr)),
        "Min": int(np.min(arr)),
        "Max": int(np.max(arr)),
        "Total pixels": int(arr.size),
    }


def simple_window_controls(prefix, default_wc=128, default_ww=255):
    """Controls for applying a Simple Window to a processed/reconstructed image."""
    st.markdown("**Simple Window Tool**")
    enabled = st.checkbox("Enable Simple Window", value=False, key=f"{prefix}_sw_enabled")
    wc = st.slider("Window center", 0, 255, int(default_wc), key=f"{prefix}_sw_wc")
    ww = st.slider("Window width", 1, 255, int(default_ww), key=f"{prefix}_sw_ww")
    return enabled, wc, ww


def apply_optional_simple_window(im, enabled, wc, ww):
    if enabled:
        return simple_window(im, wc=wc, ww=ww)
    return im   

def make_pdf_report(original, processed_results, history, dicom_meta=None, output_meta=None, result_inputs=None):
    """Create a multi-page PDF report with every saved chapter/result.

    For every result, the report stores and displays the exact input image used
    when that result was generated, then the final processed image. This matters
    when the input came from another chapter instead of the original upload.
    The visual result page is intentionally image-dominant so both images are
    much larger in the exported PDF.
    """
    buf = io.BytesIO()
    output_meta = output_meta or {}
    result_inputs = result_inputs or {}

    # Keep the four chapter outputs first, then any extra outputs such as Noise tool result.
    ordered_names = [name for name in [
        "Chapter 1 result",
        "Chapter 2 result",
        "Chapter 3 result",
        "Chapter 4 result",
    ] if name in processed_results]
    ordered_names += [name for name in processed_results.keys() if name not in ordered_names]

    with PdfPages(buf) as pdf:
        # Cover / summary page
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.suptitle("BME Medical Imaging Workstation - Full Auto Report", fontsize=16, fontweight="bold")

        gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.25], width_ratios=[1.15, 1.0])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(to_uint8_image(original), cmap="gray", vmin=0, vmax=255)
        ax0.set_title("Current active image at export time")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.axis("off")
        summary_lines = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Current image size: {original.shape[0]} rows x {original.shape[1]} columns",
            f"Saved results included: {len(ordered_names)}",
            "",
            "Included results:",
        ]
        for i, name in enumerate(ordered_names, start=1):
            meta = output_meta.get(name, {})
            operation = meta.get("operation", "-")
            result_time = meta.get("time", "-")
            input_source = meta.get("input_source", "Original / noisy input")
            summary_lines.append(f"{i}. {name} | {operation} | input: {input_source} | {result_time}")

        if dicom_meta:
            summary_lines += ["", "DICOM metadata:"]
            summary_lines += [f"{k}: {v}" for k, v in list(dicom_meta.items())[:12]]

        ax1.text(0.01, 0.98, "\n".join(summary_lines), va="top", fontsize=8.5, family="monospace")

        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis("off")
        hist_text = "Processing history:\n" + ("\n".join(history[-18:]) if history else "No recorded steps.")
        ax2.text(0.01, 0.98, hist_text, va="top", fontsize=8.0, family="monospace")

        fig.tight_layout(rect=[0, 0, 1, .94])
        pdf.savefig(fig)
        plt.close(fig)

        for result_name in ordered_names:
            processed = processed_results[result_name]
            input_image = result_inputs.get(result_name, original)
            meta = output_meta.get(result_name, {})
            input_source = meta.get("input_source", "Original / noisy input")

            # Page 1: very large before/after images.
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            fig.suptitle(f"{result_name}: input image -> final image", fontsize=15, fontweight="bold")
            gs = fig.add_gridspec(2, 2, height_ratios=[18, 1.1], hspace=0.10, wspace=0.03)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(to_uint8_image(input_image), cmap="gray", vmin=0, vmax=255)
            ax0.set_title(f"Initial image used for this result\nSource: {input_source}", fontsize=11)
            ax0.axis("off")

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(to_uint8_image(processed), cmap="gray", vmin=0, vmax=255)
            ax1.set_title("Final processed image", fontsize=11)
            ax1.axis("off")

            ax2 = fig.add_subplot(gs[1, :])
            ax2.axis("off")
            footer = (
                f"Operation: {meta.get('operation', '-')} | "
                f"Parameters: {meta.get('params', {})} | "
                f"Saved: {meta.get('time', '-')}"
            )
            ax2.text(0.01, 0.60, footer, va="center", fontsize=8.2, family="monospace")

            fig.subplots_adjust(left=0.015, right=0.985, bottom=0.035, top=0.88)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: supporting histogram, metrics and statistics.
            metrics = quality_metrics(input_image, processed)
            input_stats = image_statistics(input_image)
            proc_stats = image_statistics(processed)

            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle(f"Analysis: {result_name}", fontsize=15, fontweight="bold")
            gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.2])

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.hist(to_uint8_image(input_image).ravel(), bins=256, alpha=.55, label="Initial input")
            ax0.hist(to_uint8_image(processed).ravel(), bins=256, alpha=.55, label="Final processed")
            ax0.set_title("Histogram comparison")
            ax0.set_xlabel("Gray value")
            ax0.set_ylabel("Pixels")
            ax0.legend()
            ax0.grid(True, alpha=.25)

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.axis("off")
            metric_lines = [f"{k}: {v:.4f}" if np.isfinite(v) else f"{k}: inf" for k, v in metrics.items()]
            input_stat_lines = [f"Input {k}: {v:.4f}" if isinstance(v, float) else f"Input {k}: {v}" for k, v in input_stats.items()]
            proc_stat_lines = [f"Final {k}: {v:.4f}" if isinstance(v, float) else f"Final {k}: {v}" for k, v in proc_stats.items()]
            info_lines = [
                f"Initial image source: {input_source}",
                f"Operation: {meta.get('operation', '-')}",
                f"Parameters: {meta.get('params', {})}",
                f"Time: {meta.get('time', '-')}",
                "Quality metrics: " + ", ".join(metric_lines),
                "Image statistics: " + ", ".join(input_stat_lines + proc_stat_lines),
                "Note: Metrics compare the exact input image used for this result with the final processed image.",
            ]
            ax1.text(0.01, 0.98, "\n".join(info_lines), va="top", fontsize=8.5, family="monospace")

            fig.tight_layout(rect=[0, 0, 1, .94])
            pdf.savefig(fig)
            plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

# ---------------------- processing functions ----------------------
def simple_display(im, image_depth=IMAGE_DEPTH, tones=TONES):
    return np.clip(np.round(((tones - 1) / max(image_depth, 1)) * np.asarray(im, dtype=float)), 0, tones - 1)

def optimal_display(im): return normalize(im)

def simple_window(im, wc=50, ww=250, image_depth=IMAGE_DEPTH, tones=TONES):
    vb = min((2.0 * wc + ww) / 2.0, image_depth)
    va = max(vb - ww, 0)
    if vb == va: return normalize(im)
    return np.clip(np.round((tones - 1) * (im - va) / (vb - va)), 0, tones - 1)

def broken_window(im, gray_val=128, im_val=70, image_depth=IMAGE_DEPTH, tones=TONES):
    im = np.asarray(im, dtype=float)
    out = np.where(im <= im_val, (gray_val / max(im_val, 1)) * im, (((tones - 1) - (gray_val + 1)) / max(image_depth - (im_val + 1), 1)) * (im - (im_val + 1)) + (gray_val + 1))
    return np.clip(np.round(out), 0, tones - 1)

def double_window(im, ww1=100, wl1=50, ww2=100, wl2=150, image_depth=IMAGE_DEPTH, tones=TONES):
    im = np.asarray(im, dtype=float)
    half = tones / 2 - 1
    ve1 = round((2.0 * wl1 + ww1) / 2.0); vs1 = ve1 - ww1
    ve2 = round((2.0 * wl2 + ww2) / 2.0); vs2 = ve2 - ww2
    if vs2 < ve1:
        ve1 = round((vs2 + ve1) / 2.0); vs2 = ve1
    vs1 = max(vs1, 0); ve2 = min(ve2, image_depth)
    out = np.zeros_like(im, dtype=float)
    m1 = (im >= vs1) & (im <= ve1); m2 = (im > ve1) & (im < vs2); m3 = (im >= vs2) & (im <= ve2)
    out[m1] = half * (im[m1] - vs1) / max(ve1 - vs1, 1)
    out[m2] = half + 1
    out[m3] = (tones - 1 - (half + 1)) * (im[m3] - vs2) / max(ve2 - vs2, 1) + half + 1
    out[im > ve2] = tones - 1
    return np.clip(np.round(out), 0, tones - 1)

def nonlinear_window(im, kind):
    imn = normalize(im).astype(int)
    x = np.arange(TONES, dtype=float)
    if kind == "Inverse": w = TONES - 1 - x
    elif kind == "Logarithmic": w = np.log1p(0.05 * x)
    elif kind == "Inverse logarithmic": w = np.exp(x / 128.0) - 1
    elif kind == "Power": w = x ** 0.55
    elif kind == "Sine-window": w = np.sin(2 * np.pi * x / (4 * (TONES - 1)))
    elif kind == "Exp-window": w = 1 - np.exp(-x / 90.0)
    elif kind == "Sigmoid": w = 1 / (1 + np.exp(-(x - 128) / 25.0))
    elif kind == "Cosine window": w = np.cos(2 * np.pi * x / (4 * (TONES - 1)))
    elif kind == "Inverse square root": w = 1 / np.sqrt(x + 1e-6)
    else: w = x
    return normalize(w).astype(np.uint8)[imn]

def histogram_equalization(im):
    im8 = to_uint8_image(im)
    if cv2 is not None: return cv2.equalizeHist(im8)
    return normalize(exposure.equalize_hist(im8) * 255)

def cdf_equalization(im):
    im8 = to_uint8_image(im)
    hist, _ = np.histogram(im8.flatten(), 256, [0, 256])
    cdf = hist.cumsum().astype(float)
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    return np.ma.filled(cdf_masked, 0).astype(np.uint8)[im8]

def clahe_equalization(im, clip_limit=2.0, tile_grid_size=8):
    im8 = to_uint8_image(im)
    if cv2 is not None:
        return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size))).apply(im8)
    return normalize(exposure.equalize_adapthist(im8 / 255.0, clip_limit=float(clip_limit) / 40.0) * 255)

def transfer_curve(method, **kwargs):
    x = np.arange(256, dtype=float)
    if method == "Simple Display": y = simple_display(x)
    elif method == "Optimal Display": y = optimal_display(x)
    elif method == "Simple Window": y = simple_window(x, kwargs.get("wc", 50), kwargs.get("ww", 250))
    elif method == "Broken Window": y = broken_window(x, kwargs.get("gray_val", 128), kwargs.get("im_val", 70))
    elif method == "Double Window": y = double_window(x, kwargs.get("ww1",100), kwargs.get("wl1",50), kwargs.get("ww2",100), kwargs.get("wl2",150))
    else: y = x if method in ["Histogram Equalization", "CDF Equalization", "CLAHE"] else nonlinear_window(x.reshape(1, -1), method).ravel()
    return x, y

def apply_convolution(im, kernel):
    k = np.asarray(kernel, dtype=float)
    if k.sum() > 0: k = k / k.sum()
    return normalize(signal.convolve2d(im, k, mode="same", boundary="symm"))

def radial_filter(shape, family, mode, cutoff=0.20, order=2, bandwidth=0.10):
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows / 2.0, cols / 2.0
    D = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    Dmax = max(np.sqrt(cy ** 2 + cx ** 2), 1.0)
    D0 = max(float(cutoff) * Dmax, 1.0)
    W = max(float(bandwidth) * Dmax, 1.0)
    n = max(int(order), 1); eps = 1e-6
    if family == "Butterworth":
        lp = 1.0 / (1.0 + (D / D0) ** (2 * n)); hp = 1.0 / (1.0 + (D0 / (D + eps)) ** (2 * n)); br = 1.0 / (1.0 + ((D * W) / (np.abs(D ** 2 - D0 ** 2) + eps)) ** (2 * n)); bp = 1.0 - br
    elif family == "Gaussian":
        lp = np.exp(-((D ** 2) / (2.0 * D0 ** 2)) ** n); hp = 1.0 - lp; br = 1.0 - np.exp(-(((D ** 2 - D0 ** 2) / (D * W + eps)) ** 2) ** n); bp = 1.0 - br
    else:
        lp = np.exp(-np.log(2.0) * (D / D0) ** n); hp = np.exp(-np.log(2.0) * (D0 / (D + eps)) ** n); br = np.exp(-np.log(2.0) * (D0 / (np.abs(D - D0) + eps)) ** n); bp = np.exp(-np.log(2.0) * (np.abs(D - D0) / D0) ** n)
    return np.clip({"Low Pass": lp, "High Pass": hp, "Band Reject": br, "Band Pass": bp}[mode], 0, 1)

def frequency_filter(im, H):
    F = np.fft.fftshift(np.fft.fft2(im)); G = F * H
    return normalize(np.real(np.fft.ifft2(np.fft.ifftshift(G))))

def tomography(A, degrees, algorithm, fbp_filter=None, sart_iterations=1):
    A = normalize(A)
    theta = np.linspace(0.0, float(degrees), int(degrees), endpoint=False)
    sino = radon(A, theta=theta, circle=False)
    if algorithm == "SART":
        recon = iradon_sart(sino, theta=theta)
        for _ in range(max(0, int(sart_iterations) - 1)):
            recon = iradon_sart(sino, theta=theta, image=recon)
    else:
        recon = iradon(sino, theta=theta, filter_name=fbp_filter, output_size=A.shape[0], circle=False)
    return sino, normalize(recon)

# ---------------------------- UI ----------------------------
st.markdown('<div class="hero"><h1>🩻 BME Medical Imaging Workstation PLUS</h1><p>Provides DICOM viewer, noise tools, quality metrics, before/after slider, processing history και PDF report.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader("Upload biomedical image", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff", "dcm"])
    st.header("2. Noise tools")
    noise_mode = st.selectbox("Noise type", ["None", "Gaussian", "Salt & Pepper", "Speckle", "Poisson"])
    noise_amount = st.slider("Noise amount", 1, 100, 10, disabled=(noise_mode == "None"))
    apply_noise = st.checkbox("Use noisy image as input", value=False)

im, dicom_meta = load_uploaded_image(uploaded)
st.session_state["dicom_meta"] = dicom_meta
if im is None:
    st.info("Upload an image to activate the workstation.")
    st.stop()

noisy_im = add_noise(im, noise_mode, noise_amount) if apply_noise and noise_mode != "None" else im
if apply_noise and noise_mode != "None":
    record_output("Noise tool result", noisy_im, f"Noise: {noise_mode}", {"amount": noise_amount}, input_image=im, input_source="Original upload")

base_input = noisy_im
if st.session_state.get("pipeline_input") is not None:
    active_im = st.session_state["pipeline_input"]
else:
    active_im = base_input

# Top info + DICOM viewer panel
info1, info2 = st.columns([1.1, 2.3])
with info1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.metric("Rows", active_im.shape[0]); st.metric("Columns", active_im.shape[1]); st.metric("Gray levels", "0–255")
    st.caption(f"Active processing input: {st.session_state.get('pipeline_source', 'Original / noisy input')}")
    if st.session_state.get("pipeline_input") is not None:
        if st.button("↩️ Use original / noisy input", key="reset_pipeline_input"):
            st.session_state["pipeline_input"] = None
            st.session_state["pipeline_source"] = "Original / noisy input"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with info2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🧾 DICOM viewer panel")
    if dicom_meta:
        cols = st.columns(3)
        for i, (k, v) in enumerate(dicom_meta.items()):
            cols[i % 3].caption(k); cols[i % 3].write(v)
    else:
        st.caption("There is no DICOM file.")
    st.markdown('</div>', unsafe_allow_html=True)

if apply_noise and noise_mode != "None":
    c1, c2 = st.columns(2)
    c1.image(to_uint8_image(im), caption="Original", use_container_width=True)
    c2.image(to_uint8_image(active_im), caption=f"Noisy input: {noise_mode}", use_container_width=True)

chapter1, chapter2, chapter3, chapter4 = st.tabs(["📊 Display & Histograms", "🔬 Spatial Domain", "🌊 Frequency Domain", "☢️ Tomography"])

with chapter1:
    st.header("Chapter 1: Display & Histogram Processing")
    st.caption("Choose parameters and press Apply. The chapter does not save/process automatically on image load.")
    controls, view = st.columns([.95, 2.25])
    with controls:
        method = st.selectbox("Processing method", ["Simple Display", "Optimal Display", "Simple Window", "Broken Window", "Double Window", "Inverse", "Logarithmic", "Inverse logarithmic", "Power", "Sine-window", "Exp-window", "Sigmoid", "Cosine window", "Inverse square root", "Histogram Equalization", "CDF Equalization", "CLAHE"])
        params = {}
        if method == "Simple Window":
            params["wc"] = st.slider("Window center", 0, 255, 50)
            params["ww"] = st.slider("Window width", 1, 255, 250)
        elif method == "Broken Window":
            params["gray_val"] = st.slider("Gray value", 0, 255, 128)
            params["im_val"] = st.slider("Image threshold", 1, 255, 70)
        elif method == "Double Window":
            params["ww1"] = st.slider("Width 1", 1, 255, 100)
            params["wl1"] = st.slider("Level 1", 0, 255, 50)
            params["ww2"] = st.slider("Width 2", 1, 255, 100)
            params["wl2"] = st.slider("Level 2", 0, 255, 150)
        elif method == "CLAHE":
            params["clip_limit"] = st.slider("CLAHE clip limit", 0.5, 8.0, 2.0, 0.1)
            params["tile_grid_size"] = st.select_slider("CLAHE tile grid", options=[4, 6, 8, 10, 12, 16], value=8)

        apply1 = st.button("Apply Chapter 1 processing", key="apply_ch1", type="primary")

    if apply1:
        if method == "Simple Window":
            processed1 = simple_window(active_im, params["wc"], params["ww"])
        elif method == "Broken Window":
            processed1 = broken_window(active_im, params["gray_val"], params["im_val"])
        elif method == "Double Window":
            processed1 = double_window(active_im, **params)
        elif method == "Optimal Display":
            processed1 = optimal_display(active_im)
        elif method == "Simple Display":
            processed1 = simple_display(active_im)
        elif method == "Histogram Equalization":
            processed1 = histogram_equalization(active_im)
        elif method == "CDF Equalization":
            processed1 = cdf_equalization(active_im)
        elif method == "CLAHE":
            processed1 = clahe_equalization(active_im, params["clip_limit"], params["tile_grid_size"])
        else:
            processed1 = nonlinear_window(active_im, method)
        record_output("Chapter 1 result", processed1, method, params, input_image=active_im, input_source=st.session_state.get("pipeline_source", "Original / noisy input"))
        st.session_state["chapter1_last_method"] = method
        st.success("Chapter 1 result saved.")

    with view:
        saved1 = st.session_state.get("outputs", {}).get("Chapter 1 result")
        c1, c2, c3 = st.columns(3)
        c1.image(to_uint8_image(active_im), caption="Input Image", use_container_width=True)
        if method in ["Histogram Equalization", "CDF Equalization", "CLAHE"]:
            c2.info("This method is image-dependent; a fixed transfer curve is not shown.")
        else:
            x, y = transfer_curve(method, **params)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(x, y)
            ax.set_title(method)
            ax.set_xlabel("Input gray value")
            ax.set_ylabel("Output gray value")
            ax.grid(True, alpha=.28)
            c2.pyplot(fig, use_container_width=True)
        if saved1 is not None:
            c3.image(to_uint8_image(saved1), caption=f"Saved result: {st.session_state.get('chapter1_last_method', 'Chapter 1')}", use_container_width=True)
            if st.button("➡️ Send Chapter 1 result to processing input", key="pipe_ch1"):
                st.session_state["pipeline_input"] = saved1.copy()
                st.session_state["pipeline_source"] = "Chapter 1 result"
                st.rerun()
            st.pyplot(plot_histogram_pair(active_im, saved1), use_container_width=True)
        else:
            c3.info("No saved Chapter 1 result yet. Press Apply to create one.")

with chapter2:
    st.header("Chapter 2: Spatial Domain Processing")
    st.caption("Choose parameters and press Apply. This chapter does not run automatically on image load.")
    controls, view = st.columns([.95, 2.25])

    with controls:
        method2 = st.radio("Method", ["Kernel Convolution", "Median Filter"])
        params2 = {"method": method2}
        kernel = None

        if method2 == "Kernel Convolution":
            category = st.selectbox("Mask category", ["Smoothing", "Laplacian", "High Emphasis"])

            if category == "Smoothing":
                masks = [
                    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], float),
                    np.ones((3, 3), float),
                    np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], float),
                    np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float),
                ]
            elif category == "Laplacian":
                masks = [
                    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
                    np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], float),
                ]
            else:
                masks = [
                    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
                    np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], float),
                    np.array([[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]], float),
                ]

            mask_i = st.selectbox(
                "Active mask",
                list(range(1, len(masks) + 1)),
                format_func=lambda x: f"Mask {x}",
            ) - 1

            kernel = masks[mask_i]
            st.code(str(kernel.astype(int)))
            params2.update({"category": category, "mask": mask_i + 1})

        else:
            ksize = st.select_slider("Median kernel size", options=[3, 5, 7, 9], value=5)
            params2["kernel_size"] = ksize

        sw_enabled2, sw_wc2, sw_ww2 = simple_window_controls("chapter2")
        if sw_enabled2:
            params2.update({"simple_window": True, "wc": sw_wc2, "ww": sw_ww2})

        apply2 = st.button("Apply Chapter 2 processing", key="apply_ch2", type="primary")

    if apply2:
        if method2 == "Kernel Convolution":
            processed2 = apply_convolution(active_im, kernel)
        else:
            processed2 = normalize(signal.medfilt2d(active_im, kernel_size=params2["kernel_size"]))
        processed2_display = apply_optional_simple_window(processed2, sw_enabled2, sw_wc2, sw_ww2)
        record_output("Chapter 2 result", processed2_display, "Spatial filtering", params2, input_image=active_im, input_source=st.session_state.get("pipeline_source", "Original / noisy input"))
        st.session_state["chapter2_last_caption"] = "Processed Result" + (f" + Simple Window (WC={sw_wc2}, WW={sw_ww2})" if sw_enabled2 else "")
        st.success("Chapter 2 result saved.")

    with view:
        saved2 = st.session_state.get("outputs", {}).get("Chapter 2 result")
        c1, c2 = st.columns(2)
        c1.image(to_uint8_image(active_im), caption="Input Image", use_container_width=True)
        if saved2 is not None:
            c2.image(to_uint8_image(saved2), caption=st.session_state.get("chapter2_last_caption", "Saved Chapter 2 result"), use_container_width=True)
            if st.button("➡️ Send Chapter 2 result to processing input", key="pipe_ch2"):
                st.session_state["pipeline_input"] = saved2.copy()
                st.session_state["pipeline_source"] = "Chapter 2 result"
                st.rerun()
        else:
            c2.info("No saved Chapter 2 result yet. Press Apply to create one.")

with chapter3:
    st.header("Chapter 3: Frequency Domain Processing")
    st.caption("Choose parameters and press Apply. FFT/filtering does not run automatically on image load.")
    controls, view = st.columns([.95, 2.25])

    with controls:
        family = st.selectbox("Filter family", ["Butterworth", "Gaussian", "Exponential"])
        mode = st.selectbox("Filter type", ["Low Pass", "High Pass", "Band Reject", "Band Pass"])
        cutoff = st.slider("Cutoff frequency", 0.01, 0.80, 0.20, 0.01)
        order = st.slider("Order / degree", 1, 8, 2)
        bandwidth = st.slider("Bandwidth", 0.01, 0.50, 0.10, 0.01)

        sw_enabled3, sw_wc3, sw_ww3 = simple_window_controls("chapter3")
        params3 = {"cutoff": cutoff, "order": order, "bandwidth": bandwidth}
        if sw_enabled3:
            params3.update({"simple_window": True, "wc": sw_wc3, "ww": sw_ww3})

        apply3 = st.button("Apply Chapter 3 processing", key="apply_ch3", type="primary")

    if apply3:
        H = radial_filter(active_im.shape, family, mode, cutoff, order, bandwidth)
        processed3 = frequency_filter(active_im, H)
        processed3_display = apply_optional_simple_window(processed3, sw_enabled3, sw_wc3, sw_ww3)
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(active_im))))
        st.session_state["chapter3_spectrum"] = spectrum
        st.session_state["chapter3_filter_mask"] = H
        st.session_state["chapter3_last_caption"] = f"{family} {mode}" + (f" + Simple Window (WC={sw_wc3}, WW={sw_ww3})" if sw_enabled3 else "")
        record_output("Chapter 3 result", processed3_display, f"{family} {mode}", params3, input_image=active_im, input_source=st.session_state.get("pipeline_source", "Original / noisy input"))
        st.success("Chapter 3 result saved.")

    with view:
        saved3 = st.session_state.get("outputs", {}).get("Chapter 3 result")
        c1, c2, c3 = st.columns(3)
        c1.image(to_uint8_image(active_im), caption="Input Image", use_container_width=True)
        if saved3 is not None:
            spectrum = st.session_state.get("chapter3_spectrum")
            H = st.session_state.get("chapter3_filter_mask")
            if spectrum is not None:
                c2.image(to_uint8_image(spectrum), caption="Magnitude Spectrum", use_container_width=True)
            else:
                c2.info("Magnitude spectrum will appear after Apply.")
            c3.image(to_uint8_image(saved3), caption=st.session_state.get("chapter3_last_caption", "Saved Chapter 3 result"), use_container_width=True)
            if st.button("➡️ Send Chapter 3 result to processing input", key="pipe_ch3"):
                st.session_state["pipeline_input"] = saved3.copy()
                st.session_state["pipeline_source"] = "Chapter 3 result"
                st.rerun()
            if H is not None:
                st.image(to_uint8_image(H * 255), caption="2-D filter mask", use_container_width=False)
        else:
            c2.info("Magnitude spectrum will appear after Apply.")
            c3.info("No saved Chapter 3 result yet. Press Apply to create one.")

with chapter4:
    st.header("Chapter 4: Tomographic Image Reconstruction")
    st.caption("This chapter already runs only when you press Run Reconstruction.")
    controls, view = st.columns([.95, 2.25])

    with controls:
        degrees = st.slider("Total degrees of camera revolution", 30, 360, 180, 5)
        algorithm = st.radio("Tomographic algorithm", ["FBP", "SART"])

        fbp_filter = None
        iterations = 1

        if algorithm == "FBP":
            fbp_filter = st.selectbox(
                "Reconstruction filter",
                ["ramp", "shepp-logan", "cosine", "hamming", "hann"],
            )

        if algorithm == "SART":
            iterations = st.slider("SART iterations", 1, 100, 2)

        sw_enabled4, sw_wc4, sw_ww4 = simple_window_controls("chapter4")

        run = st.button("Run Reconstruction", type="primary")

    if run:
        sino, processed4_raw = tomography(active_im, degrees, algorithm, fbp_filter, iterations)

# Crop για να φύγει το μαύρο περιθώριο της ανακατασκευής
        processed4_raw = crop_nonblack_area(
           processed4_raw,
           threshold=5,
           padding=5
        )

# Resize για να έχει ίδιο μέγεθος με την αρχική εικόνα
        processed4_raw = resize_like(processed4_raw, active_im)

        processed4 = apply_optional_simple_window(
            processed4_raw,
            sw_enabled4,
            sw_wc4,
            sw_ww4,
        )

        st.session_state["tomo_result"] = (
            sino,
            processed4,
            algorithm,
            sw_enabled4,
            sw_wc4,
            sw_ww4,
        )

        params4 = {"degrees": degrees, "filter": fbp_filter, "iterations": iterations}
        if sw_enabled4:
            params4.update({"simple_window": True, "wc": sw_wc4, "ww": sw_ww4})

        record_output("Chapter 4 result", processed4, f"Tomography {algorithm}", params4, input_image=active_im, input_source=st.session_state.get("pipeline_source", "Original / noisy input"))
        st.success("Chapter 4 result saved.")

    with view:
        if "tomo_result" in st.session_state:
            tomo_result = st.session_state["tomo_result"]

            # Compatibility with old saved session_state that had only 3 values.
            if len(tomo_result) == 3:
                sino, processed4, alg = tomo_result
                saved_sw_enabled4 = False
                saved_sw_wc4 = 128
                saved_sw_ww4 = 255
            else:
                sino, processed4, alg, saved_sw_enabled4, saved_sw_wc4, saved_sw_ww4 = tomo_result

            c1, c2 = st.columns(2)
            c1.image(
                to_uint8_image(active_im),
                caption="Image to produce sinograms",
                use_container_width=True,
            )

            cap4 = f"Reconstructed by {alg}"
            if saved_sw_enabled4:
                cap4 += f" + Simple Window (WC={saved_sw_wc4}, WW={saved_sw_ww4})"

            c2.image(to_uint8_image(processed4), caption=cap4, use_container_width=True)
            if st.button("➡️ Send Chapter 4 result to processing input", key="pipe_ch4"):
                st.session_state["pipeline_input"] = processed4.copy()
                st.session_state["pipeline_source"] = "Chapter 4 result"
                st.rerun()

            st.markdown("### Sinogram")
            fig_sino, ax_sino = plt.subplots(figsize=(12, 4))
            ax_sino.imshow(to_uint8_image(sino), cmap="gray", aspect="auto")
            ax_sino.set_xlabel("Projection angle")
            ax_sino.set_ylabel("Detector position")
            ax_sino.set_title("Sinogram display")
            st.pyplot(fig_sino, use_container_width=True)
        else:
            st.info("Set scan parameters and press Run Reconstruction.")
# ---------------------------- Advanced panels ----------------------------
st.divider()
st.header("🧪 Analysis tools")

if st.button("Reset all outputs", type="secondary"):
    st.session_state["outputs"] = {}
    st.session_state["output_meta"] = {}
    st.session_state["output_inputs"] = {}
    st.session_state["history"] = []
    st.session_state["pipeline_input"] = None
    st.session_state["pipeline_source"] = "Original / noisy input"
    st.session_state.pop("tomo_result", None)
    st.session_state.pop("chapter3_spectrum", None)
    st.session_state.pop("chapter3_filter_mask", None)
    st.rerun()

outputs = st.session_state.get("outputs", {})
if outputs:
    result_name = st.selectbox("Active processed result", list(outputs.keys()))
    result = outputs[result_name]

    if st.button("➡️ Use selected result as input for another chapter", key="pipe_selected_result", type="primary"):
        st.session_state["pipeline_input"] = result.copy()
        st.session_state["pipeline_source"] = result_name
        st.rerun()

    # Use the exact input image that created this saved result.
    # This keeps the Before/After slider stable even after sending a chapter
    # result back into the processing pipeline as the new active input.
    output_inputs = st.session_state.get("output_inputs", {})
    result_before = output_inputs.get(result_name, active_im)
    result_input_source = st.session_state.get("output_meta", {}).get(result_name, {}).get(
        "input_source",
        "Original / noisy input",
    )

    with st.expander("Before/After live viewer", expanded=True):
        st.caption(
            f"Drag the red bar to compare the saved input for this result ({result_input_source}) "
            f"with {result_name}. The left/before image stays fixed even if this result is sent back as processing input."
        )
        live_before_after_viewer(result_before, result, start=50, height=1000)

    with st.expander("Quality metrics", expanded=True):
        metrics = quality_metrics(result_before, result)
        orig_stats = image_statistics(result_before)
        proc_stats = image_statistics(result)

        st.caption("Comparison metrics between the saved input image for this result and the processed image")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MSE", f"{metrics['MSE']:.3f}")
        m2.metric("MAE", f"{metrics['MAE']:.3f}")
        m3.metric("PSNR", "∞" if not np.isfinite(metrics['PSNR']) else f"{metrics['PSNR']:.2f} dB")
        m4.metric("SSIM", f"{metrics['SSIM']:.4f}")

        st.caption("Image statistics")
        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.markdown("**Saved input image**")
            st.metric("Mean", f"{orig_stats['Mean']:.3f}")
            st.metric("Std", f"{orig_stats['Std']:.3f}")
            st.metric("Min", f"{orig_stats['Min']}")
            st.metric("Max", f"{orig_stats['Max']}")
            st.metric("Total pixels", f"{orig_stats['Total pixels']}")
        with stat_cols[1]:
            st.markdown("**Processed image**")
            st.metric("Mean", f"{proc_stats['Mean']:.3f}")
            st.metric("Std", f"{proc_stats['Std']:.3f}")
            st.metric("Min", f"{proc_stats['Min']}")
            st.metric("Max", f"{proc_stats['Max']}")
            st.metric("Total pixels", f"{proc_stats['Total pixels']}")

    with st.expander("Processing history", expanded=True):
        if st.session_state["history"]:
            st.code("\n".join(st.session_state["history"]), language="text")
        else:
            st.caption("No steps recorded yet.")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

    st.subheader("💾 Export processed image")
    fmt = st.radio("Save format", ["png", "jpg", "bmp", "dicom"])
    data, fname, mime = image_download_bytes(result, fmt)
    base = uploaded.name.rsplit('.', 1)[0] if uploaded is not None else "processed_image"
    ext = "dcm" if fmt == "dicom" else fmt
    st.download_button(f"Download {result_name} as {fmt.upper()}", data=data, file_name=f"{base}_{result_name.lower().replace(' ', '_')}.{ext}", mime=mime, type="primary")

    st.subheader("📄 Auto report PDF")
    st.caption("The PDF includes all saved results from the chapters, not only the active selected result.")
    pdf_bytes = make_pdf_report(
        active_im,
        outputs,
        st.session_state["history"],
        st.session_state.get("dicom_meta", {}),
        st.session_state.get("output_meta", {}),
        st.session_state.get("output_inputs", {}),
    )
    pdf_filename = st.text_input(
    "PDF filename",
    value=f"{base}_full_report.pdf"
     )

    if not pdf_filename.lower().endswith(".pdf"):
         pdf_filename += ".pdf"

    st.download_button(
        "Download full PDF report",
         data=pdf_bytes,
         file_name=pdf_filename,
         mime="application/pdf"
     )
else:
    st.caption("First, create a processed result from some chapter.")
