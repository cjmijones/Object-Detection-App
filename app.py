import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from collections import Counter
from torchvision import models
import torch.nn as nn

# ----------------------
# Streamlit UI Setup
# ----------------------
st.set_page_config(page_title="ASL Letter Detection", layout="centered")
st.title("🤟 Real-Time ASL Letter Detection")

with st.expander("ℹ️ Instructions"):
    st.markdown("""
    1. Click the **Start Webcam Detection** button.
    2. Show a hand sign within the camera view.
    3. The model will detect your hand and classify the ASL letter using a buffer of cropped images.
    4. The predicted letter will be shown below the video feed.
    """)

FRAME_WINDOW = st.image([])
prediction_text = st.empty()

# ----------------------
# Load Models
# ----------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8_medium_model/best_medium.pt")

@st.cache_resource
def load_letter_classifier():
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(256, 26)
    )
    model.load_state_dict(torch.load("letter_classifier_v1.pt", map_location="cpu"))
    model.eval()
    return model

model = load_yolo_model()
letter_model = load_letter_classifier()

class_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# ----------------------
# Detection Logic
# ----------------------
crop_buffer = []
frame_count = 0


def detect_and_display():
    global frame_count, crop_buffer
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        return

    with st.spinner("Running detection... Press STOP to quit."):
        while st.session_state.get("running", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame)
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            if boxes and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
                hand_crop = rgb_frame[y1:y2, x1:x2]
                pil_crop = Image.fromarray(hand_crop)

                if frame_count % 5 == 0:
                    crop_buffer.append(transform(pil_crop))

            frame_count += 1

            if len(crop_buffer) >= 5:
                batch = torch.stack(crop_buffer)
                with torch.no_grad():
                    outputs = letter_model(batch)
                    preds = torch.argmax(outputs, dim=1)
                    letter_counts = Counter(preds.tolist())
                    top_letter = letter_counts.most_common(1)[0][0]
                    predicted_char = class_map[top_letter]

                prediction_text.markdown(f"## 🧠 Predicted Letter: **{predicted_char}**")
                crop_buffer = []

            FRAME_WINDOW.image(annotated_frame, channels="RGB")

    cap.release()
    FRAME_WINDOW.image([])
    prediction_text.markdown("")


# ----------------------
# Start/Stop Control
# ----------------------
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([1, 1])

if col1.button("▶️ Start Webcam Detection"):
    st.session_state.running = True
    detect_and_display()

if col2.button("⏹️ Stop Detection"):
    st.session_state.running = False
