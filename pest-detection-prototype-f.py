import streamlit as st
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
import numpy as np
import warnings
import logging
import torch
from tempfile import NamedTemporaryFile

# ==============================
# Suppress Warnings & Logging
# ==============================
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="Pest Detection", layout="wide")
st.title("🪲 Pest Detection System (Grasshopper & Snail)")

# ==============================
# Device Detection
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥 Running on: **{'GPU (CUDA)' if DEVICE=='cuda' else 'CPU'}**")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Avyl/snail-grasshopper_model",
        filename="yolov9.pt"
    )
    model = YOLO(model_path)
    model.to(DEVICE)
    return model

model = load_model()

# ==============================
# Initialize DeepSORT Tracker
# ==============================
@st.cache_resource
def init_tracker():
    return DeepSort(max_age=60, n_init=3, max_iou_distance=0.8)

tracker = init_tracker()

# ==============================
# Sidebar Mode Selection
# ==============================
mode = st.sidebar.radio(
    "Select Mode",
    ["🎥 Live Camera Detection", "🎞 Video Inference"]
)

# ==============================
# Session States
# ==============================
if "cumulative_counts" not in st.session_state:
    st.session_state.cumulative_counts = {"Grasshopper": 0, "Snail": 0}
if "unique_ids" not in st.session_state:
    st.session_state.unique_ids = set()
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

# ==============================
# Helper Function
# ==============================
def boxes_count(results_item):
    try:
        return len(results_item.boxes)
    except Exception:
        try:
            return results_item.boxes.xyxy.shape[0]
        except Exception:
            return 0


# ==========================================================
# MODE 1: 🎥 LIVE CAMERA DETECTION
# ==========================================================
if mode == "🎥 Live Camera Detection":
    st.sidebar.header("🎥 Controls")
    stream_url = "rtsp://yolov9:yolo69@192.168.1.6:8080/h264_pcm.sdp"

    if st.sidebar.button("🔄 Reset Counts"):
        st.session_state.cumulative_counts = {"Grasshopper": 0, "Snail": 0}
        st.session_state.unique_ids.clear()
        st.session_state.detection_log.clear()
        st.sidebar.success("✅ Counts reset!")

    frame_placeholder = st.empty()
    counts_placeholder = st.empty()
    logs_placeholder = st.empty()

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error("⚠ Cannot open video stream. Check your URL.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.45, device=DEVICE)
            detections = []

            if results and boxes_count(results[0]) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.int().cpu().numpy()
                for box, conf, cls_idx in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = box
                    label = str(model.names[int(cls_idx)]).lower()
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

            tracks = tracker.update_tracks(detections, frame=frame)

            frame_counts = {"Grasshopper": 0, "Snail": 0}

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                label = track.det_class.lower()
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                color = (0, 255, 0) if "grass" in label else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                frame_counts["Grasshopper" if "grass" in label else "Snail"] += 1

                if track_id not in st.session_state.unique_ids:
                    st.session_state.unique_ids.add(track_id)
                    st.session_state.cumulative_counts["Grasshopper" if "grass" in label else "Snail"] += 1
                    st.session_state.detection_log.append({
                        "id": track_id,
                        "label": label.capitalize(),
                        "time": datetime.now().strftime("%H:%M:%S")
                    })

            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            counts_placeholder.markdown(f"""
            ### 🧮 Detection Summary
            **Realtime Frame Counts:**
            - 🦗 Grasshoppers: **{frame_counts['Grasshopper']}**
            - 🐌 Snails: **{frame_counts['Snail']}**

            **Cumulative Detections (Unique IDs):**
            - 🦗 Grasshoppers: **{st.session_state.cumulative_counts['Grasshopper']}**
            - 🐌 Snails: **{st.session_state.cumulative_counts['Snail']}**
            """)

            if st.session_state.detection_log:
                logs_placeholder.markdown("### 🧾 Detection Log (Last 10)")
                for log in reversed(st.session_state.detection_log[-10:]):
                    logs_placeholder.write(f"🕒 {log['time']} — {log['label']} (ID: {log['id']})")

            time.sleep(0.02)

        cap.release()


# ==========================================================
# MODE 2: 🎞 VIDEO INFERENCE
# ==========================================================
elif mode == "🎞 Video Inference":
    st.header("🎞 Upload a Video for Pest Detection")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        output_path = "output_inference.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        total_counts = {"Grasshopper": 0, "Snail": 0}
        unique_ids_video = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.45, device=DEVICE)
            detections = []
            if results and boxes_count(results[0]) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.int().cpu().numpy()
                for box, conf, cls_idx in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = box
                    label = str(model.names[int(cls_idx)]).lower()
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                label = track.det_class.lower()
                if track_id not in unique_ids_video:
                    unique_ids_video.add(track_id)
                    total_counts["Grasshopper" if "grass" in label else "Snail"] += 1

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                color = (0, 255, 0) if "grass" in label else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()
        st.success(f"✅ Inference Completed!\n\n🦗 Grasshoppers: {total_counts['Grasshopper']} | 🐌 Snails: {total_counts['Snail']}")
        with open(output_path, "rb") as file:
            st.download_button("⬇ Download Processed Video", data=file, file_name="pest_detection_output.avi")
