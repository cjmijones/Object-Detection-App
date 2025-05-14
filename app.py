# streamlit_app.py ────────────────────────────────────────────────────────────
# Real-Time Hand Detection (default YOLOv8) + ASL Letter Classification
# ----------------------------------------------------------------------------
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from collections import Counter
from torchvision import models
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Page configuration & shared containers
# ----------------------------------------------------------------------------
st.set_page_config(page_title="🤟 ASL Letter & Hand Detection", layout="centered")
st.title("🤟 Real-Time Hand & ASL Letter Detection")

FRAME_WINDOW   = st.empty()   # live video feed
prediction_txt = st.empty()   # predicted letter headline

# ----------------------------------------------------------------------------
# Model loading (cached once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_default_yolo():
    """Stock YOLOv8m pretrained on COCO-hand subset."""
    return YOLO("yolov8m.pt")

@st.cache_resource
def load_finetuned_yolo():
    """Your hand-finetuned YOLO checkpoint."""
    return YOLO("yolov8_medium_model/best_medium.pt")

@st.cache_resource
def load_letter_classifier():
    net = models.resnet18()
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 512),
        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 26)
    )
    net.load_state_dict(torch.load("letter_classifier_v1.pt", map_location="cpu"))
    net.eval()
    return net

det_default  = load_default_yolo()   # for Section A
det_finetune = load_finetuned_yolo() # for Section B
cls_model    = load_letter_classifier()
class_map    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------------------------
# Session flags
# ----------------------------------------------------------------------------
for flag in ("run_hand", "run_letter"):
    st.session_state.setdefault(flag, False)

# ----------------------------------------------------------------------------
# Webcam loop
# ----------------------------------------------------------------------------
def webcam_loop(detector: YOLO, with_letters: bool = False):
    """Stream webcam; draw boxes; optionally classify letters."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        return

    crop_buf, frame_id = [], 0
    running_flag = "run_letter" if with_letters else "run_hand"

    with st.spinner("Running detection…  Click **Stop** to quit."):
        while st.session_state.get(running_flag):
            ok, frame = cap.read()
            if not ok:
                st.warning("⚠️ Failed to grab frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector(rgb)
            annotated = res[0].plot()

            if with_letters:
                boxes = res[0].boxes
                if boxes and len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
                    hand_crop = rgb[y1:y2, x1:x2]
                    pil_crop  = Image.fromarray(hand_crop)

                    if frame_id % 5 == 0:
                        crop_buf.append(transform(pil_crop))

                if len(crop_buf) >= 5:
                    batch = torch.stack(crop_buf)
                    with torch.no_grad():
                        preds = torch.argmax(cls_model(batch), 1)
                        letter = class_map[Counter(preds.tolist()).most_common(1)[0][0]]
                    prediction_txt.markdown(f"## 🧠 Predicted Letter: **{letter}**")
                    crop_buf.clear()

                frame_id += 1

            FRAME_WINDOW.image(annotated, channels="RGB")

    cap.release()
    FRAME_WINDOW.empty()
    prediction_txt.empty()

# ----------------------------------------------------------------------------
# Section A – Hand detection with default YOLO
# ----------------------------------------------------------------------------
st.header("✋ Section A – Default YOLO Detection)")
with st.expander("ℹ️ What this does"):
    st.markdown("""
    * Uses the YOLO v8m model to draw bounding boxes 100 different classes of common objects.
    * Run on your individual webcam and use to detect chairs, people, tvs, cell phones, and a variety of other common objects.
    """)

cA1, cA2 = st.columns([1, 1])
if cA1.button("▶️ Start Object Detection", key="start_hand"):
    st.session_state.run_hand = True
    webcam_loop(detector=det_default, with_letters=False)

if cA2.button("⏹️ Stop Objeect Detection", key="stop_hand"):
    st.session_state.run_hand = False

st.markdown("---")   # horizontal divider

# ----------------------------------------------------------------------------
# Section B – Full ASL letter pipeline (finetuned YOLO + ResNet-18)
# ----------------------------------------------------------------------------
st.header("🔤 Section B – ASL Letter Detection Pipeline")
with st.expander("ℹ️ What this does"):
    st.markdown("""
    1. Click the **Start Webcam Detection** button.
    2. Show a hand sign within the camera view.
    3. The model will detect your hand and classify the ASL letter using a buffer of cropped images.
    4. The predicted letter will be shown below the video feed.
    """)

cB1, cB2 = st.columns([1, 1])
if cB1.button("▶️ Start Letter Detection", key="start_letter"):
    st.session_state.run_letter = True
    webcam_loop(detector=det_finetune, with_letters=True)

if cB2.button("⏹️ Stop Letter Detection", key="stop_letter"):
    st.session_state.run_letter = False
