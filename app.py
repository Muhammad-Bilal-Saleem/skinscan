import streamlit as st
import numpy as np
import yaml
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan — Disease Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}
.stApp { background-color: #0a0a0a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222222;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600;
     letter-spacing: -0.02em; color: #ffffff; margin-bottom: 0; }
h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 500;
     color: #aaaaaa; letter-spacing: 0.05em; text-transform: uppercase; }
h3 { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 500;
     color: #777777; letter-spacing: 0.08em; text-transform: uppercase; }
hr { border: none; border-top: 1px solid #1e1e1e; margin: 1.2rem 0; }

[data-testid="stFileUploader"] {
    background-color: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 1rem;
}

[data-testid="stRadio"] > div { display: flex; flex-direction: row; gap: 0.5rem; }
[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important; color: #aaaaaa !important;
    background: #161616; border: 1px solid #2a2a2a;
    border-radius: 3px; padding: 0.3rem 0.7rem; cursor: pointer;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #222222; border-color: #555555; color: #ffffff !important;
}

.metric-card {
    background-color: #111111; border: 1px solid #1e1e1e;
    border-radius: 4px; padding: 1.1rem 1.2rem; text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: #555555; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;
    font-weight: 600; color: #ffffff; line-height: 1;
}
.metric-unit { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #555555; margin-top: 0.2rem; }

.det-header-row {
    display: flex; justify-content: space-between;
    padding: 0.45rem 0.9rem; background: #161616;
    border: 1px solid #1e1e1e; border-bottom: none;
    border-radius: 4px 4px 0 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #444444;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.det-wrap { border: 1px solid #1e1e1e; border-radius: 0 0 4px 4px; overflow: hidden; }
.det-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 0.9rem; border-bottom: 1px solid #161616;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}
.det-item:last-child { border-bottom: none; }
.det-cls  { color: #e8e8e8; font-weight: 500; min-width: 140px; }
.det-conf { font-weight: 500; min-width: 52px; text-align: right; }
.det-bar-bg { flex: 1; height: 3px; background: #1e1e1e; border-radius: 2px; margin: 0 0.8rem; overflow: hidden; }
.det-bar-fg { height: 100%; border-radius: 2px; }

.badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.25rem 0.6rem; border-radius: 2px; font-weight: 500;
}
.badge-ok   { background:#1a2e1a; color:#5a9e5a; border:1px solid #2a4a2a; }
.badge-warn { background:#2e2a1a; color:#9e8a5a; border:1px solid #4a3a2a; }
.badge-none { background:#1e1e1e; color:#666666; border:1px solid #2a2a2a; }

.info-box {
    background:#111111; border:1px solid #1e1e1e;
    border-left:3px solid #333333; border-radius:4px;
    padding:0.8rem 1rem; font-family:'IBM Plex Mono',monospace;
    font-size:0.78rem; color:#666666; line-height:1.6;
}
.logo-block {
    font-family:'IBM Plex Mono',monospace; font-size:1.05rem;
    font-weight:600; color:#ffffff; padding:0.2rem 0 1rem 0;
    border-bottom:1px solid #1e1e1e; margin-bottom:1.2rem;
}
.logo-sub {
    font-size:0.68rem; color:#444444; font-weight:400;
    letter-spacing:0.08em; text-transform:uppercase; margin-top:0.15rem;
}
.empty-state {
    background:#0e0e0e; border:1px dashed #222222; border-radius:4px;
    padding:4rem 2rem; text-align:center; color:#333333;
    font-family:'IBM Plex Mono',monospace; font-size:0.8rem; letter-spacing:0.05em;
}
[data-testid="stSlider"] label {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.78rem !important; color:#666666 !important;
}
[data-testid="stImage"] img { border-radius:4px; border:1px solid #1e1e1e; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0a0a; }
::-webkit-scrollbar-thumb { background:#2a2a2a; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
PT_PATH    = MODELS_DIR / "best.pt"
ONNX_PATH  = MODELS_DIR / "best.onnx"
YAML_PATH  = MODELS_DIR / "data.yaml"

def conf_color(conf: float) -> str:
    if conf >= 0.70: return "#5a9e5a"
    if conf >= 0.45: return "#9e8a5a"
    return "#9e5a5a"

# ── Load class names ──────────────────────────────────────────────────────────
@st.cache_resource
def load_class_names():
    if not YAML_PATH.exists():
        return []
    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(backend: str):
    if backend == "PyTorch (.pt)":
        from ultralytics import YOLO
        return YOLO(str(PT_PATH)), "pt"
    else:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers  = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        return sess, "onnx"

# ── Preprocessing — Pillow only, zero cv2 ────────────────────────────────────
def letterbox_pil(img: Image.Image, target: int = 640):
    w, h    = img.size
    scale   = target / max(w, h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas  = Image.new("RGB", (target, target), (0, 0, 0))
    pad_x   = (target - new_w) // 2
    pad_y   = (target - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y

def preprocess_onnx(img: Image.Image, imgsz: int = 640):
    lb, scale, pad_x, pad_y = letterbox_pil(img, imgsz)
    blob = np.array(lb, dtype=np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]   # NCHW
    return blob, scale, pad_x, pad_y

# ── Pure-numpy NMS ────────────────────────────────────────────────────────────
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1   = np.maximum(x1[i], x1[order[1:]])
        iy1   = np.maximum(y1[i], y1[order[1:]])
        ix2   = np.minimum(x2[i], x2[order[1:]])
        iy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep

def postprocess_onnx(output, scale, pad_x, pad_y, conf_thresh, iou_thresh, class_names):
    preds      = output[0].T
    boxes_xywh = preds[:, :4]
    scores     = preds[:, 4:]
    class_ids  = np.argmax(scores, axis=1)
    confs      = scores[np.arange(len(scores)), class_ids]

    mask       = confs >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    class_ids  = class_ids[mask]

    if len(confs) == 0:
        return []

    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / scale
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / scale
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_x) / scale
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_y) / scale
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep    = nms(boxes_xyxy, confs, iou_thresh)
    results = []
    for i in keep:
        cid = int(class_ids[i])
        results.append({
            "class_id":   cid,
            "class_name": class_names[cid] if cid < len(class_names) else f"class_{cid}",
            "conf":       float(confs[i]),
            "box":        [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
        })
    results.sort(key=lambda x: x["conf"], reverse=True)
    return results

def run_inference_pt(model, img: Image.Image, conf, iou, class_names):
    arr     = np.array(img)
    results = model.predict(arr, conf=conf, iou=iou, verbose=False)
    result  = results[0]
    dets    = []
    for box in result.boxes:
        cid  = int(box.cls.item())
        name = result.names.get(cid, class_names[cid] if cid < len(class_names) else f"class_{cid}")
        dets.append({
            "class_id":   cid,
            "class_name": name,
            "conf":       float(box.conf.item()),
            "box":        box.xyxy[0].tolist(),
        })
    dets.sort(key=lambda x: x["conf"], reverse=True)
    return dets

# ── Drawing — Pillow only ─────────────────────────────────────────────────────
def draw_detections_pil(img: Image.Image, detections: list) -> Image.Image:
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']}  {det['conf']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(220, 220, 220), width=1)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=(220, 220, 220))
        draw.text((x1 + 3, y1 - th - 3), label, fill=(10, 10, 10), font=font)
    return out

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        SKINSCAN
        <div class="logo-sub">Disease Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Model Backend</h3>", unsafe_allow_html=True)
    available_backends = []
    if PT_PATH.exists():   available_backends.append("PyTorch (.pt)")
    if ONNX_PATH.exists(): available_backends.append("ONNX (.onnx)")
    if not available_backends:
        st.error("No model files found in models/")
        st.stop()

    backend = st.radio("backend", available_backends,
                       label_visibility="collapsed", horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Detection Thresholds</h3>", unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", 0.10, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU (NMS)",  0.10, 0.95, 0.45, 0.05)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Display</h3>", unsafe_allow_html=True)
    show_original = st.checkbox("Show original alongside", value=False)

    class_names = load_class_names()
    if class_names:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3>Classes</h3>", unsafe_allow_html=True)
        classes_html = "".join(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
            f'color:#555555;padding:0.2rem 0;border-bottom:1px solid #1a1a1a;">'
            f'<span style="color:#333333;">{i:02d}</span>&nbsp;&nbsp;{name}</div>'
            for i, name in enumerate(class_names)
        )
        st.markdown(
            f'<div style="background:#0e0e0e;border:1px solid #1e1e1e;'
            f'border-radius:4px;padding:0.5rem 0.8rem;">{classes_html}</div>',
            unsafe_allow_html=True,
        )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:0.3rem;">
    <h1>SKINSCAN</h1>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                 color:#444444;letter-spacing:0.08em;">YOLO11s · 11 CLASSES</span>
</div>
<div style="font-size:0.82rem;color:#555555;margin-bottom:1.5rem;
            font-family:'IBM Plex Sans',sans-serif;">
    Upload a skin image for automated disease detection and classification.
</div>
<hr>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, mode = load_model(backend)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div class="empty-state">
        NO IMAGE LOADED<br>
        <span style="font-size:0.7rem;color:#222222;">JPG · PNG · BMP · WEBP</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load image ────────────────────────────────────────────────────────────────
img_pil        = Image.open(uploaded).convert("RGB")
w_orig, h_orig = img_pil.size

# ── Inference ─────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
if mode == "pt":
    detections = run_inference_pt(model, img_pil, conf_thresh, iou_thresh, class_names)
else:
    blob, scale, pad_x, pad_y = preprocess_onnx(img_pil)
    raw        = model.run(None, {model.get_inputs()[0].name: blob})
    detections = postprocess_onnx(raw[0], scale, pad_x, pad_y,
                                  conf_thresh, iou_thresh, class_names)
latency_ms = (time.perf_counter() - t0) * 1000

annotated = draw_detections_pil(img_pil, detections)

# ── Layout ────────────────────────────────────────────────────────────────────
col_img, col_res = st.columns([3, 2], gap="large")

with col_img:
    st.markdown("<h2>Detection Output</h2>", unsafe_allow_html=True)
    if show_original:
        t1, t2 = st.tabs(["Annotated", "Original"])
        with t1: st.image(annotated, width="stretch")
        with t2: st.image(img_pil,   width="stretch")
    else:
        st.image(annotated, width="stretch")

    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
        f'color:#3a3a3a;margin-top:0.5rem;">'
        f'{uploaded.name}&nbsp;·&nbsp;{w_orig}×{h_orig}px'
        f'&nbsp;·&nbsp;{uploaded.size/1024:.1f} KB</div>',
        unsafe_allow_html=True,
    )

with col_res:
    st.markdown("<h2>Results</h2>", unsafe_allow_html=True)

    n_det    = len(detections)
    top_conf = detections[0]["conf"] if detections else 0.0
    top_cls  = detections[0]["class_name"] if detections else "—"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detections</div>
            <div class="metric-value">{n_det}</div>
            <div class="metric-unit">objects</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latency</div>
            <div class="metric-value">{latency_ms:.0f}</div>
            <div class="metric-unit">ms</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Conf</div>
            <div class="metric-value">{top_conf:.2f}</div>
            <div class="metric-unit">{top_cls[:12] if top_cls != '—' else '—'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if n_det == 0:
        st.markdown('<span class="badge badge-none">NO DETECTIONS</span>', unsafe_allow_html=True)
    elif top_conf >= 0.60:
        st.markdown('<span class="badge badge-ok">CONFIDENT DETECTION</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-warn">LOW CONFIDENCE</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Detections</h3>", unsafe_allow_html=True)

    if n_det == 0:
        st.markdown("""
        <div class="info-box">
            No detections above the current threshold.<br>
            Try lowering the confidence slider in the sidebar.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="det-header-row">
            <span>Class</span><span>Conf</span>
        </div>""", unsafe_allow_html=True)
        rows = ""
        for det in detections:
            bar_w = int(det["conf"] * 100)
            color = conf_color(det["conf"])
            rows += (
                f'<div class="det-item">'
                f'<span class="det-cls">{det["class_name"]}</span>'
                f'<div class="det-bar-bg">'
                f'<div class="det-bar-fg" style="width:{bar_w}%;background:{color};"></div>'
                f'</div>'
                f'<span class="det-conf" style="color:{color};">{det["conf"]:.3f}</span>'
                f'</div>'
            )
        st.markdown(f'<div class="det-wrap">{rows}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        For research and educational purposes only.<br>
        Not a substitute for professional medical diagnosis.
    </div>""", unsafe_allow_html=True)
