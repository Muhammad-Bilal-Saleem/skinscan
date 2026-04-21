# SkinScan — Skin Disease Detection

> Real-time skin disease detection and classification using YOLO11s, trained on a multi-class dermatology dataset with a clean Streamlit interface.

---

## Overview

SkinScan is an end-to-end computer vision pipeline for automated skin disease detection. It uses a fine-tuned YOLO11s object detection model trained on 1,590 annotated dermatology images across 11 disease classes. The project includes the full training notebook (Google Colab), a local inference app built with Streamlit, and support for both PyTorch and ONNX backends.

---

## Classes

| ID | Class |
|----|-------|
| 0  | Acne |
| 1  | Chickenpox |
| 2  | Eczema |
| 3  | Monkeypox |
| 4  | Psoriasis |
| 5  | Ringworm |
| 6  | Basal Cell Carcinoma |
| 7  | Tinea Versicolor |
| 8  | Vitiligo |
| 9  | Warts |
| 10 | Chickenpox |

---

## Model Performance

Evaluated on held-out test split after 60 epochs of training on a Tesla T4 GPU.

| Metric | Score |
|--------|-------|
| mAP@0.5 | 49.4% |
| mAP@0.5-0.95 | 27.3% |
| Precision | 53.8% |
| Recall | 52.3% |
| F1-Score | 53.0% |

**Per-class highlights:**

| Class | mAP@0.5 |
|-------|---------|
| Psoriasis | 98.3% |
| Tinea Versicolor | 83.1% |
| Ringworm | 50.9% |
| Acne | 51.9% |
| Chickenpox | 18.0% |

Strong performance on visually distinct conditions (Psoriasis, Tinea Versicolor). Lower scores on visually similar or data-sparse classes (Chickenpox, Warts) reflect dataset imbalance rather than model failure.

---

## Project Structure

```
skinscan/
├── skin_disease_detection_yolo11.ipynb   # Training notebook (Colab)
├── app.py                                # Streamlit inference app
├── requirements.txt
├── .streamlit/
│   └── config.toml                       # Dark theme config
└── models/                               # Place trained weights here
    ├── best.pt
    ├── best.onnx
    └── data.yaml
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/yourusername/skinscan.git
cd skinscan
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If running on a CUDA GPU, replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`.

### 3. Add model weights

Download `best.pt`, `best.onnx`, and `data.yaml` from the [releases page](https://github.com/yourusername/skinscan/releases) and place them in the `models/` directory.

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Training

The full training pipeline is in `skin_disease_detection_yolo11.ipynb`, designed to run on Google Colab with a T4 GPU.

**Requirements before running:**
- A [Roboflow](https://roboflow.com) account with API key
- GPU runtime enabled in Colab (`Runtime > Change runtime type > T4 GPU`)
- API key stored in Colab Secrets as `ROBOFLOW_API_KEY`

**Training config:**

| Parameter | Value |
|-----------|-------|
| Model | YOLO11s |
| Epochs | 60 |
| Image size | 640×640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Early stopping patience | 15 |
| Augmentation | Mosaic, MixUp, Flip, HSV |

**Dataset:** [Skin Disease Detection — Roboflow Universe](https://universe.roboflow.com/workshop-yg2yt/skin-3n2jd)
— 1,590 images, 11 classes, YOLOv8 format.

---

## Inference Details

The app supports two backends:

**PyTorch (.pt)** — uses the Ultralytics library directly. Handles all preprocessing internally. Recommended for simplicity.

**ONNX (.onnx)** — uses ONNX Runtime. Manual preprocessing pipeline: letterbox resize → RGB conversion → normalization → NCHW transpose → NMS post-processing. Faster and more portable for deployment. Automatically selects CUDA or CPU provider based on what's available.

---

## Stack

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Roboflow](https://roboflow.com/)

---

## Disclaimer

This project is for research and educational purposes only. It is not a substitute for professional medical diagnosis. Do not use model outputs to make clinical decisions.

---

## License

MIT