# 🪲 Pest Detection System (Grasshopper & Snail)

A real-time pest detection and tracking application built with **YOLOv9**, **DeepSORT**, and **Streamlit**. The system can detect and track grasshoppers and snails either through a live RTSP camera stream or by running inference on uploaded videos.

---

## 📽️ Prototype Demo

> Watch the recorded prototype demonstration below:

<a href="https://drive.google.com/file/d/1biwlppbYeGulbKaMIOXd-xxYIhp2RuOw/view?usp=sharing" target="_blank">▶ Watch Demo on Google Drive</a>

---

## 📋 Table of Contents

- [Demo](#️-prototype-demo)
- [Features](#features)
- [Demo Modes](#demo-modes)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training](#training)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Output & Logging](#output--logging)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Author](#-author)
- [Date Created](#-date-created)

---

## ✨ Features

- 🔴 **Live RTSP Camera Detection** — Real-time inference from an IP camera stream
- 🎞 **Video Inference** — Upload and process pre-recorded video files (MP4, AVI, MOV)
- 🧮 **Per-frame & Cumulative Counts** — Track pest counts per frame and across the entire session
- 🆔 **Unique ID Tracking** — DeepSORT assigns persistent IDs to each pest to avoid double-counting
- 🧾 **Detection Log** — Time-stamped log of every newly detected pest
- ⬇️ **Video Download** — Export the annotated inference video as an AVI file
- 🖥 **GPU / CPU Auto-detection** — Automatically uses CUDA if available, falls back to CPU
- 🔄 **Reset Controls** — One-click reset for counts, IDs, and logs in live mode

---

## 🎬 Demo Modes

### 🎥 Mode 1 — Live Camera Detection

Connects to an RTSP stream and runs continuous detection. Displays:
- Annotated live feed with bounding boxes and track IDs
- Real-time frame-level counts for grasshoppers and snails
- Cumulative unique pest counts across the session
- Rolling detection log (last 10 entries)

### 🎞 Mode 2 — Video Inference

Upload a local video file and run batch inference. Features:
- Frame-by-frame progress bar
- Live preview of annotated frames during processing
- Final count summary on completion
- Download button for the processed output video

---

## 🛠 Tech Stack

| Component | Library / Tool |
|---|---|
| Object Detection | [Ultralytics YOLOv9](https://github.com/ultralytics/ultralytics) |
| Multi-Object Tracking | [Deep SORT Realtime](https://github.com/levan92/deep_sort_realtime) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Model Hosting | [Hugging Face Hub](https://huggingface.co/) |
| Computer Vision | [OpenCV](https://opencv.org/) |
| Deep Learning Backend | [PyTorch](https://pytorch.org/) |
| Numerical Computing | [NumPy](https://numpy.org/) |

---

## 📦 Prerequisites

- Python **3.8+**
- pip
- *(Optional but recommended)* NVIDIA GPU with CUDA support for faster inference
- An RTSP-compatible IP camera (for live detection mode)

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/pest-detection-system.git
cd pest-detection-system
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install streamlit opencv-python ultralytics deep-sort-realtime \
            huggingface_hub numpy torch torchvision
```

> 💡 For GPU support, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

**4. Run the application**

```bash
streamlit run pest-detection-prototype-f.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🏋️ Training

The model was trained (and can be retrained) using either **YOLOv9** or **YOLOv8** via the provided shell scripts. Both scripts target the same snail-grasshopper dataset.

### 📁 Dataset

Both training scripts expect a dataset configuration file at:

```
/home/avela/miniconda3/envs/yolov9detection/yolov9/snail-grasshopper_dataset/data.yaml
```

Make sure your `data.yaml` defines the correct paths to your `train`, `val`, and `test` image sets, as well as the class names (`Grasshopper`, `Snail`).

---

### 🔵 Option A — YOLOv9 Fine-tuning (`finetune.sh`)

Fine-tunes a pre-trained YOLOv9 checkpoint (`best.pt`) on the pest dataset.

**Prerequisites**

```bash
pip install ultralytics
```

**Run**

```bash
bash finetune.sh
```

**Key Parameters**

| Parameter | Value | Description |
|---|---|---|
| `MODEL_PATH` | `/home/avela/best.pt` | Pre-trained YOLOv9 checkpoint to fine-tune |
| `EPOCHS` | `100` | Maximum training epochs |
| `BATCH_SIZE` | `16` | Batch size per step |
| `IMG_SIZE` | `640` | Input image resolution |
| `DEVICE` | `0` | GPU device index (`cpu` for CPU-only) |
| `PATIENCE` | `50` | Early stopping patience (epochs without improvement) |
| `SAVE_PERIOD` | `10` | Save a checkpoint every N epochs |
| `optimizer` | `AdamW` | Optimizer |
| `lr0` | `0.001` | Initial learning rate |
| `lrf` | `0.01` | Final learning rate (fraction of `lr0`) |
| `warmup_epochs` | `3` | Gradual LR warmup at the start of training |
| `amp` | `True` | Automatic Mixed Precision (faster GPU training) |
| `mosaic` | `1.0` | Mosaic augmentation probability |

**Output**

```
snail_grasshopper_detection/
└── yolov9_finetune/
    └── weights/
        ├── best.pt    ← Best checkpoint (by validation mAP)
        └── last.pt    ← Final epoch checkpoint
```

---

### 🟢 Option B — YOLOv8 Training (`yolov8training.sh`)

Trains a YOLOv8 medium model (`yolov8m.pt`) from scratch on the pest dataset. Suitable as a lighter-weight alternative or baseline comparison.

**Prerequisites**

The script activates the conda environment and installs dependencies automatically:

```bash
conda activate /home/avela/miniconda3/envs/yolov9detection
pip install ultralytics
```

**Run**

```bash
bash yolov8training.sh
```

**Key Parameters**

| Parameter | Value | Description |
|---|---|---|
| `model` | `yolov8m.pt` | YOLOv8 medium pretrained weights |
| `epochs` | `100` | Maximum training epochs |
| `imgsz` | `640` | Input image resolution |
| `batch` | `8` | Batch size per step |
| `project` | `/home/avela/yolov8_runs` | Output directory |
| `name` | `snail_grasshopper_yolov8m` | Experiment subfolder name |

**Output**

```
/home/avela/yolov8_runs/
└── snail_grasshopper_yolov8m/
    └── weights/
        ├── best.pt
        └── last.pt
```

---

### ⚖️ YOLOv9 vs YOLOv8 Comparison

| | YOLOv9 (`finetune.sh`) | YOLOv8 (`yolov8training.sh`) |
|---|---|---|
| Base model | Custom `best.pt` checkpoint | `yolov8m.pt` (pretrained) |
| Batch size | 16 | 8 |
| Optimizer | AdamW | Default (SGD) |
| AMP | ✅ Yes | Default |
| Early stopping | ✅ Yes (patience=50) | Default |
| Augmentation | Fully configured | Default |
| Use case | Fine-tuning existing model | Training new baseline |

---

### 💡 Tips for Retraining

- **GPU strongly recommended** — both scripts default to `device=0` (first GPU). Change to `device=cpu` if no GPU is available, but expect significantly longer training times.
- **Adjust batch size** for your GPU VRAM — reduce to `8` or `4` if you encounter out-of-memory errors on the YOLOv9 script.
- **Monitor training** — both scripts save plots and metrics inside the output project folder. Open `results.png` to review loss and mAP curves.
- **Using the trained model** — copy `best.pt` to Hugging Face or point `MODEL_PATH` directly to it in `finetune.sh` for further fine-tuning rounds.

---

## ⚙️ Configuration

### RTSP Stream URL

The live camera stream URL is hardcoded in the script. To change it, edit this line:

```python
stream_url = "rtsp://yolov9:yolo69@192.168.1.6:8080/h264_pcm.sdp"
```

Replace with your camera's RTSP address in the format:

```
rtsp://<username>:<password>@<ip_address>:<port>/<stream_path>
```

### Detection Confidence Threshold

The minimum confidence score for a detection to be considered valid is set to `0.45`. Adjust it in both modes by changing the `conf` parameter:

```python
results = model(frame, conf=0.45, device=DEVICE)
```

Lower values (e.g., `0.3`) detect more objects but may increase false positives. Higher values (e.g., `0.6`) reduce false positives but may miss detections.

### DeepSORT Tracker Parameters

```python
DeepSort(max_age=60, n_init=3, max_iou_distance=0.8)
```

| Parameter | Default | Description |
|---|---|---|
| `max_age` | `60` | Frames to keep a track alive without a new detection |
| `n_init` | `3` | Detections required before a track is confirmed |
| `max_iou_distance` | `0.8` | Maximum IoU distance for track association |

---

## 📖 Usage

### Live Camera Mode

1. Select **🎥 Live Camera Detection** in the sidebar.
2. Ensure your IP camera is accessible on the network and the RTSP URL is correctly configured.
3. The stream will start automatically and display annotated frames, counts, and logs.
4. Use the **🔄 Reset Counts** button in the sidebar to clear all counts and the detection log.
5. Close the browser tab or stop the Streamlit server to end the session.

### Video Inference Mode

1. Select **🎞 Video Inference** in the sidebar.
2. Click **Browse files** and upload a video (`.mp4`, `.avi`, or `.mov`).
3. Inference runs automatically frame by frame with a live progress bar.
4. Once complete, a summary shows total grasshopper and snail counts.
5. Click **⬇ Download Processed Video** to save the annotated output.

---

## 📁 Project Structure

```
pest-detection-system/
├── pest-detection-prototype-f.py   # Main application entry point
├── finetune.sh                     # YOLOv9 fine-tuning script
├── yolov8training.sh               # YOLOv8 training script
├── output_inference.avi            # Generated after video inference (auto-created)
└── README.md
```

> The YOLOv9 model weights are downloaded automatically from Hugging Face on first run and cached locally by the `huggingface_hub` library.

---

## 🤖 Model Information

The detection model is hosted on Hugging Face:

- **Repository:** [`Avyl/snail-grasshopper_model`](https://huggingface.co/Avyl/snail-grasshopper_model)
- **File:** `yolov9.pt`
- **Architecture:** YOLOv9 (via Ultralytics)
- **Classes:** Grasshopper, Snail

The model is downloaded automatically on the first run using `hf_hub_download` and cached for subsequent sessions. An internet connection is required on the first launch.

---

## 📊 Output & Logging

### Live Mode — Detection Summary

Displayed in real time below the video feed:

```
### 🧮 Detection Summary
Realtime Frame Counts:
  🦗 Grasshoppers: 2
  🐌 Snails: 1

Cumulative Detections (Unique IDs):
  🦗 Grasshoppers: 7
  🐌 Snails: 3
```

### Live Mode — Detection Log

A time-stamped log entry is created every time a new unique pest is detected:

```
🕒 14:32:05 — Grasshopper (ID: 4)
🕒 14:32:08 — Snail (ID: 5)
```

### Video Inference — Output File

The processed video is saved as `output_inference.avi` in the working directory using the **XVID** codec. The frame rate and resolution match the input video.

---

## ⚠️ Known Limitations

- **Live mode runs indefinitely** — there is no built-in stop button; the Streamlit app must be restarted or the browser tab closed to end the stream.
- **Single camera support** — live mode supports only one RTSP stream at a time.
- **RTSP URL is hardcoded** — the camera URL must be changed directly in the source file.
- **Tracker state is shared** — unique IDs in video inference mode use a per-run local set; live mode uses Streamlit session state which persists until reset or page refresh.
- **Output format is AVI** — the inference output video uses the XVID codec and `.avi` container. Some players may require a codec pack.

---

## 🔧 Troubleshooting

**Cannot open video stream**

> `⚠ Cannot open video stream. Check your URL.`

- Verify the camera is powered on and connected to the same network.
- Confirm the RTSP URL, username, and password are correct.
- Test the stream URL with VLC: `Media → Open Network Stream`.

**Model download fails on first run**

- Ensure you have an active internet connection.
- Check that the Hugging Face repository `Avyl/snail-grasshopper_model` is public and accessible.
- If behind a proxy, set the `HTTPS_PROXY` environment variable before running.

**Running slowly on CPU**

- Inference on CPU is significantly slower than on GPU.
- Consider reducing input resolution or increasing the `time.sleep` interval in live mode.
- For production use, a CUDA-capable GPU is strongly recommended.

**`AttributeError` on `track.det_class`**

- Ensure you are using a compatible version of `deep-sort-realtime`. The code relies on `track.det_class` being populated by the tracker.
- Install the recommended version: `pip install deep-sort-realtime==1.3.2`

**Out-of-memory error during training**

- Reduce `BATCH_SIZE` in `finetune.sh` (try `8` or `4`).
- Reduce `batch` in `yolov8training.sh` (try `4`).
- Set `amp=True` to enable Automatic Mixed Precision if not already enabled.

---

## 📄 License

---

This project is for research and project purposes.

---

## 👩🏻‍💻 Author

Jacques Nico Belmonte - AI Developer

---

## 📆 Date Created

January 2026

---
