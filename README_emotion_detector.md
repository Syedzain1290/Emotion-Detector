# 😊 Real-Time Facial Emotion Detector

> Detects 7 human emotions from images and live webcam using deep learning — with business action recommendations for each emotion.

!\[Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square\&logo=python)
!\[OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat-square)
!\[DeepFace](https://img.shields.io/badge/DeepFace-Deep%20Learning-orange?style=flat-square)
!\[Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=flat-square)

\---

## 📌 Overview

A real-time facial emotion detection system that analyzes human faces from uploaded photos or live webcam feed. Detects 7 emotions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral) with confidence scores. Built using Haar Cascade for face detection and DeepFace's neural network for emotion classification.

\---

## ✨ Features

* **7 Emotion Classes** — Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral
* **Two Detection Modes** — Image upload OR live webcam capture
* **Confidence Bar Charts** — Shows probability score for all 7 emotions
* **Multi-Face Detection** — Detects and analyzes multiple faces in one image
* **Bounding Boxes** — Draws colored boxes around each detected face
* **Session History** — Tracks all detections with confidence scores
* **Action Recommendations** — Maps each emotion to a recommended response

\---

## 🧠 How It Works

```
Input Image / Webcam Frame
        ↓
Haar Cascade Face Detection
(finds face locations in image)
        ↓
Crop Each Face Region
        ↓
DeepFace Neural Network
(classifies emotion from face pixels)
        ↓
Display Result + Confidence Score
```

### Key Concepts

|Concept|Explanation|
|-|-|
|**Haar Cascade**|Pre-trained XML classifier that detects frontal faces using pixel patterns|
|**DeepFace**|Facebook's deep learning library — uses VGG-Face architecture for emotion|
|**ROI**|Region of Interest — the cropped face area fed into the emotion model|
|**Confidence Score**|Probability (0-100%) that the detected emotion is correct|

\---

## 🛠️ Tech Stack

|Tool|Purpose|
|-|-|
|`Python`|Core programming language|
|`OpenCV`|Face detection and image processing|
|`DeepFace`|Deep learning emotion classification|
|`Streamlit`|Web application interface|
|`NumPy`|Image array operations|
|`Pillow`|Image format conversion|

\---

## 🚀 Getting Started

### 1\. Clone the Repository

```bash
git clone https://github.com/YOUR\_USERNAME/emotion-detector.git
cd emotion-detector
```

### 2\. Install Dependencies

```bash
pip install deepface opencv-python streamlit numpy pillow tf\_keras matplotlib
```

### 3\. Run the App

```bash
streamlit run emotion\_detector.py
```

> ⚠️ First run downloads DeepFace models (\~500MB). Ensure stable internet connection.

\---

## 📁 Project Structure

```
emotion-detector/
├── emotion\_detector.py    # Main Streamlit app
├── preview.png            # App screenshot
└── README.md              # This file
```

\---

## 🔮 Future Improvements

* \[ ] Add real-time video stream (continuous detection)
* \[ ] Add emotion tracking over time (mood timeline)
* \[ ] Train custom model on specific demographic data
* \[ ] Add voice emotion detection
* \[ ] Deploy as REST API

\---

## 👤 Author

**Syed Zain ul Abideen** · [GitHub](https://github.com/your-username)

