# VisionAI • Advanced Face Detection Studio

A high-performance, full-featured face detection application developed in the `adhi` workspace. Powered by OpenCV's state-of-the-art **YuNet Deep Learning ONNX** model and classical **Haar Cascade Classifiers**, featuring interactive Web, Desktop, and CLI interfaces.

---

## ✨ Key Features

- **Dual Detection Engines**:
  - 🧠 **YuNet Deep Learning (ONNX)**: Modern, ultra-fast CNN-based detector with facial landmarks (eyes, nose, mouth corners) and high detection recall across diverse poses, lighting, and scales (~18 ms latency).
  - ⚡ **Haar Cascade (Classical CV)**: Fast, zero-dependency traditional classifier with eye and smile sub-feature detection.
- **Multiple User Interfaces**:
  - 🌐 **Web Studio (Streamlit)**: Interactive browser interface with webcam capture, drag-and-drop file upload, real-time sliders, face avatar gallery, and JSON export.
  - 🖥️ **Desktop Studio (Tkinter + OpenCV)**: Native Windows GUI supporting file loading, live webcam feed, real-time parameters, and instant image saving.
  - 💻 **CLI Utility (`main.py`)**: Powerful scriptable command-line interface for batch processing, image analysis, and video processing.
- **Visual Effects & HUD Overlays**:
  - Futuristic **Corner HUD** bounding boxes, rounded rectangles, or classic boxes.
  - **Color Themes**: Cyber Neon, Electric Cyan, Sunset Orange, Ultra Violet, Alert Red, Clean White.
  - Facial landmarks display with geometry wireframe lines.
  - Top HUD status bar with real-time inference latency (ms), detected face count, and FPS.
- **Privacy & Anonymization Mode**:
  - Built-in **Gaussian Blur** filter (adjustable radius).
  - Built-in **Mosaic / Pixelation** filter (adjustable block size).
- **Face Avatar Gallery**:
  - Automatically crops and extracts each detected face with proportional margins for instant portrait avatar downloads.
- **Data Export**:
  - Full detection metadata export (bounding boxes, confidence scores, landmark coordinates) to structured JSON.

---

## 📁 Project Structure

```
adhi/
│
├── models/
│   └── face_detection_yunet.onnx     # Pre-trained deep learning detector (ONNX)
│
├── samples/
│   ├── sample_portrait.jpg           # Standard portrait sample image
│   ├── sample_sports.jpg             # Sports portrait sample image
│   └── sample_people.png             # Multi-person sample image
│
├── core/
│   ├── __init__.py                   # Package exports
│   ├── detector.py                   # Unified FaceDetector engine (YuNet & Haar)
│   ├── visualizer.py                 # HUD, landmarks, and privacy filter renderer
│   └── utils.py                      # I/O, cropping, video processing, JSON export
│
├── app_streamlit.py                  # Streamlit Web Application
├── app_desktop.py                    # Native Windows Tkinter GUI Application
├── main.py                           # CLI tool and application launcher
│
├── run_web.bat                       # One-click launcher for Web Studio
├── run_desktop.bat                   # One-click launcher for Desktop Studio
├── requirements.txt                  # Dependencies list
└── README.md                         # Documentation
```

---

## 🚀 Quickstart

### 1. Launch the Interactive Web Studio
Run either by double-clicking `run_web.bat` or executing in PowerShell:
```powershell
python -m streamlit run app_streamlit.py
```
This opens the web browser interface where you can:
- Snap photos directly via your webcam.
- Upload any image or choose preset sample images.
- Process video files frame-by-frame.
- Download annotated images, cropped face avatars, or detection JSON.

---

### 2. Launch the Native Desktop GUI
Run either by double-clicking `run_desktop.bat` or executing:
```powershell
python app_desktop.py
```
- Real-time webcam streaming toggle with instant face bounding boxes.
- Open image files from any folder on your computer.
- Customize color themes, bounding styles, confidence thresholds, and blur filters.
- Save annotated results with a single click.

---

### 3. Command-Line Interface (CLI)

#### Run on a Single Image:
```powershell
python main.py --image samples/sample_portrait.jpg --output result.jpg
```

#### Run with Privacy Blur / Anonymization:
```powershell
python main.py --image samples/sample_portrait.jpg --privacy blur --blur-strength 45 --output blurred.jpg
```

#### Automatically Crop Detected Faces & Export JSON:
```powershell
python main.py --image samples/sample_sports.jpg --crop --export-json detection.json
```

#### Process a Video File:
```powershell
python main.py --video input.mp4 --output annotated_output.mp4
```

#### Choose Detector Algorithm:
```powershell
# Using Deep Learning YuNet (Default)
python main.py --image samples/sample_portrait.jpg --detector yunet --conf 0.65

# Using Classical Haar Cascade
python main.py --image samples/sample_portrait.jpg --detector haar
```

---

## 📊 Detection Engines Comparison

| Feature | YuNet Deep Learning (ONNX) | Haar Cascade (OpenCV) |
| :--- | :--- | :--- |
| **Model Type** | Deep Convolutional Neural Network | Feature-based Cascade Classifier |
| **Landmarks** | 5 points (Eyes, Nose, Mouth Corners) | Sub-cascade for eyes and smiles |
| **Inference Speed** | ~18 ms / frame | ~30 - 80 ms / frame |
| **Accuracy / Recall** | High (Robust to angle, shadows, scale) | Moderate (Frontal view optimized) |
| **Confidence Scores** | Continuous probabilities (0.0 - 1.0) | Discrete cascade stage matches |

---

## 🛠️ Requirements & Installation

Dependencies are minimal and standard:
```powershell
pip install -r requirements.txt
```
Installed packages:
- `opencv-python` / `opencv-python-headless` (>= 4.8.0)
- `numpy` (>= 1.24.0)
- `pillow` (>= 9.5.0)
- `streamlit` (>= 1.28.0)

