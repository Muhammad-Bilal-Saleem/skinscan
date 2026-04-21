import streamlit as st
import cv2
import numpy as np
import yaml
import time
from pathlib import Path

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

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 1rem;
}

/* Radio buttons — clean pill style */
[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
}
[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #aaaaaa !important;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 0.3rem 0.7rem;
    cursor: pointer;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #222222;
    border-color: #555555;
    color: #ffffff !important;
}

/* Metric cards */
.metric-card {
    background-color: #111111;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 1.1rem 1.2rem;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #555555;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem; font-weight: 600;
    color: #ffffff; line-height: 1;
}
.metric-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #555555; margin-top: 0.2rem;
}

/* Detection row */
.det-header-row {
    display: flex; justify-content: space-between;
    padding: 0.45rem 0.9rem;
    background: #161616;
    border: 1px solid #1e1e1e;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #444444;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.det-wrap {
    border: 1px solid #1e1e1e;
    border-radius: 0 0 4px 4px;
    overflow: hidden;
}
.det-item {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid #161616;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.det-item:last-child { border-bottom: none; }
.det-cls  { color: #e8e8e8; font-weight: 500; min-width: 140px; }
.det-conf { font-weight: 500; min-width: 52px; text-align: right; }
.det-bar-bg {
    flex: 1; height: 3px; background: #1e1e1e;
    border-radius: 2px; margin: 0 0.8rem; overflow: hidden;
}
.det-bar-fg { height: 100%; border-radius: 2px; }

/* Status badge */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.25rem 0.6rem;
    border-radius: 2px; font-weight: 500;
}
.badge-ok   { background:#1a2e1a; color:#5a9e5a; border:1px solid #2a4a2a; }
.badge-warn { background:#2e2a1a; color:#9e8a5a; border:1px solid #4a3a2a; }
.badge-none { background:#1e1e1e; color:#666666; border:1px solid #2a2a2a; }

/* Info box */
.info-box {
    background:#111111; border:1px solid #1e1e1e;
    border-left:3px solid #333333; border-radius:4px;
    padding:0.8rem 1rem;
    font-family:'IBM Plex Mono',monospace;
    font-size:0.78rem; color:#666666; line-height:1.6;
}

/* Logo */
.logo-block {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.05rem; font-weight:600; color:#ffffff;
    padding:0.2rem 0 1rem 0;
    border-bottom:1px solid #1e1e1e; margin-bottom:1.2rem;
}
.logo-sub {
    font-size:0.68rem; color:#444444; font-weight:400;
    letter-spacing:0.08em; text-transform:uppercase; margin-top:0.15rem;
}

/* Empty state */
.empty-state {
    background:#0e0e0e; border:1px dashed #222222;
    border-radius:4px; padding:4rem 2rem; text-align:center;
    color:#333333; font-family:'IBM Plex Mono',monospace;
    font-size:0.8rem; letter-spacing:0.05em;
}

/* Slider label */
[data-testid="stSlider"] label {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.78rem !important; color:#666666 !important;
}

/* Image border */
[data-testid="stImage"] img {
    border-radius:4px; border:1px solid #1e1e1e;
}

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
    # Roboflow sometimes stores as dict {0: 'Acne', 1: ...}
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

# ── Preprocessing helpers ─────────────────────────────────────────────────────
def letterbox(img: np.ndarray, target: int = 640):
    h, w    = img.shape[:2]
    scale   = target / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((target, target, 3), dtype=np.uint8)
    pad_x   = (target - new_w) // 2
    pad_y   = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y

def preprocess_onnx(img_bgr: np.ndarray, imgsz: int = 640):
    img_rgb              = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb, scale, px, py   = letterbox(img_rgb, imgsz)
    blob                 = lb.astype(np.float32) / 255.0
    blob                 = np.transpose(blob, (2, 0, 1))[np.newaxis]
    return blob, scale, px, py

def postprocess_onnx(output, scale, pad_x, pad_y, conf_thresh, iou_thresh, class_names):
    preds      = output[0].T                          # [N, 4+nc]
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

    boxes_nms = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    indices   = cv2.dnn.NMSBoxes(boxes_nms.tolist(), confs.tolist(), conf_thresh, iou_thresh)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            cid = int(class_ids[i])
            results.append({
                "class_id":   cid,
                "class_name": class_names[cid] if cid < len(class_names) else f"class_{cid}",
                "conf":       float(confs[i]),
                "box":        [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            })
    results.sort(key=lambda x: x["conf"], reverse=True)
    return results

def run_inference_pt(model, img_bgr, conf, iou, class_names):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results  = model.predict(img_rgb, conf=conf, iou=iou, verbose=False)
    result   = results[0]
    dets = []
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

def draw_detections(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    img = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']}  {det['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (220, 220, 220), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    return img

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

    backend = st.radio(
        "backend",
        available_backends,
        label_visibility="collapsed",
        horizontal=True,
    )

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

# ── Decode ────────────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not decode image.")
    st.stop()

h_orig, w_orig = img_bgr.shape[:2]

# ── Inference ─────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
if mode == "pt":
    detections = run_inference_pt(model, img_bgr, conf_thresh, iou_thresh, class_names)
else:
    blob, scale, pad_x, pad_y = preprocess_onnx(img_bgr)
    raw = model.run(None, {model.get_inputs()[0].name: blob})
    detections = postprocess_onnx(raw[0], scale, pad_x, pad_y,
                                  conf_thresh, iou_thresh, class_names)
latency_ms = (time.perf_counter() - t0) * 1000

# ── Annotate ──────────────────────────────────────────────────────────────────
annotated_rgb = cv2.cvtColor(draw_detections(img_bgr, detections), cv2.COLOR_BGR2RGB)
original_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── Layout ────────────────────────────────────────────────────────────────────
col_img, col_res = st.columns([3, 2], gap="large")

with col_img:
    st.markdown("<h2>Detection Output</h2>", unsafe_allow_html=True)
    if show_original:
        t1, t2 = st.tabs(["Annotated", "Original"])
        with t1: st.image(annotated_rgb, width="stretch")
        with t2: st.image(original_rgb,  width="stretch")
    else:
        st.image(annotated_rgb, width="stretch")

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

    # Metric cards
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

    # Status badge
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
        # Header
        st.markdown("""
        <div class="det-header-row">
            <span>Class</span><span>Conf</span>
        </div>""", unsafe_allow_html=True)

        # Rows — built as one HTML block so it renders reliably
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
