# ============================================================
# 😊 Customer Emotion Detector
# ============================================================
# Detects 7 emotions from face images:
#   Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
#
# Two modes:
#   1. Upload an image → analyze emotions
#   2. Live webcam → real-time emotion detection
# detecting if customers are frustrated or happy during
# support interactions!
#
# ── SETUP ──────────────────────────────────────────────────
# pip install deepface opencv-python streamlit numpy pillow tf_keras matplotlib
#
# Run:
#   streamlit run emotion_detector.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@700&display=swap');

    .stApp { background-color: #f8fafc; color: #1e293b; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    #MainMenu, footer, header { visibility: hidden; }

    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(59,130,246,0.08);
    }

    .header-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563eb;
    }

    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #2563eb;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .emotion-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .emotion-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }

    .emotion-label {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .emotion-confidence {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #64748b;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .metric-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #94a3b8;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.3rem;
    }

    .bar-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.7rem;
        gap: 0.8rem;
    }

    .bar-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #475569;
        width: 80px;
        flex-shrink: 0;
    }

    .bar-track {
        flex: 1;
        background: #f1f5f9;
        border-radius: 99px;
        height: 10px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.5s ease;
    }

    .bar-pct {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        width: 45px;
        text-align: right;
        flex-shrink: 0;
    }

    .history-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# EMOTION CONFIG
# ============================================================

EMOTION_CONFIG = {
    "happy":    {"emoji": "😊", "color": "#22c55e", "label": "Happy",    "jazz_note": "Customer is satisfied"},
    "sad":      {"emoji": "😢", "color": "#3b82f6", "label": "Sad",      "jazz_note": "Customer may need extra support"},
    "angry":    {"emoji": "😡", "color": "#ef4444", "label": "Angry",    "jazz_note": "Escalate to senior agent immediately"},
    "surprise": {"emoji": "😲", "color": "#f59e0b", "label": "Surprised","jazz_note": "Customer is confused — clarify"},
    "fear":     {"emoji": "😨", "color": "#8b5cf6", "label": "Fearful",  "jazz_note": "Reassure the customer"},
    "disgust":  {"emoji": "🤢", "color": "#f97316", "label": "Disgusted","jazz_note": "Customer is frustrated — apologize"},
    "neutral":  {"emoji": "😐", "color": "#64748b", "label": "Neutral",  "jazz_note": "Normal interaction"},
}

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_face_detector():
    """Load OpenCV Haar cascade face detector."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_faces(image_array, face_cascade):
    """
    Detect faces in an image using Haar cascade.
    Returns list of (x, y, w, h) tuples for each face found.
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much image is reduced at each scale
        minNeighbors=5,    # how many neighbors each rectangle should have
        minSize=(30, 30)   # minimum face size to detect
    )
    return faces

def analyze_emotion(face_roi):
    """
    Analyze emotion of a face region using DeepFace.
    Returns dominant emotion and all emotion scores.
    """
    try:
        result = DeepFace.analyze(
            face_roi,
            actions=["emotion"],
            enforce_detection=False,  # don't crash if face not detected
            silent=True
        )
        emotions   = result[0]["emotion"]
        dominant   = result[0]["dominant_emotion"]
        return dominant, emotions
    except Exception as e:
        return "neutral", {e: 0 for e in EMOTION_CONFIG}

def draw_emotion_on_frame(frame, faces, emotions_list):
    """
    Draw bounding boxes and emotion labels on the image.
    """
    annotated = frame.copy()

    for i, (x, y, w, h) in enumerate(faces):
        if i >= len(emotions_list):
            break

        dominant, _ = emotions_list[i]
        config = EMOTION_CONFIG.get(dominant, EMOTION_CONFIG["neutral"])

        # Convert hex color to BGR for OpenCV
        hex_color = config["color"].lstrip("#")
        r, g, b   = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
        bgr_color = (b, g, r)

        # Draw rectangle around face
        cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_color, 3)

        # Draw emotion label background
        label     = f"{config['emoji']} {config['label']}"
        label_bgr = f"{dominant.upper()}"
        (tw, th), _ = cv2.getTextSize(label_bgr, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x, y - th - 15), (x + tw + 10, y), bgr_color, -1)

        # Draw emotion text
        cv2.putText(
            annotated, label_bgr,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

    return annotated

def process_image(image_array):
    """
    Full pipeline:
    1. Detect faces
    2. Analyze each face's emotion
    3. Draw annotations
    """
    face_cascade = load_face_detector()
    faces        = detect_faces(image_array, face_cascade)
    emotions_list = []

    if len(faces) == 0:
        # Try DeepFace directly if Haar cascade misses the face
        try:
            dominant, emotions = analyze_emotion(image_array)
            emotions_list      = [(dominant, emotions)]
            faces              = np.array([[50, 50, image_array.shape[1]-100, image_array.shape[0]-100]])
        except:
            pass
    else:
        for (x, y, w, h) in faces:
            face_roi            = image_array[y:y+h, x:x+w]
            dominant, emotions  = analyze_emotion(face_roi)
            emotions_list.append((dominant, emotions))

    annotated = draw_emotion_on_frame(image_array, faces, emotions_list)
    return faces, emotions_list, annotated

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <div class="header-title">😊 Customer Emotion Detector</div>
    <div class="header-subtitle">
        Real-time facial emotion analysis ·
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 😊 Emotion Detector")
    st.markdown("---")

    mode = st.radio(
        "Select Mode",
        ["📸 Image Upload", "🎥 Live Webcam"],
        help="Image upload analyzes a photo. Live webcam does real-time detection."
    )

    st.markdown("---")
    st.markdown("### 😶 Detectable Emotions")
    for key, config in EMOTION_CONFIG.items():
        st.markdown(
            f"<div style='font-family:Inter;font-size:0.85rem;padding:0.3rem 0'>"
            f"{config['emoji']} <b>{config['label']}</b> — {config['jazz_note']}"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SCANS</div>
            <div class="metric-value">{st.session_state.total_scans}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        happy_count = sum(1 for h in st.session_state.history if h["emotion"] == "happy")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HAPPY</div>
            <div class="metric-value" style="color:#22c55e">{happy_count}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Clear History"):
        st.session_state.history   = []
        st.session_state.total_scans = 0
        st.rerun()

# ============================================================
# MODE 1: IMAGE UPLOAD
# ============================================================

if "📸 Image Upload" in mode:

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">📤 Upload Face Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a photo with a face",
            type=["jpg", "jpeg", "png"],
            help="Works best with clear, well-lit face photos"
        )

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 DETECT EMOTIONS", use_container_width=True):
                with st.spinner("Analyzing emotions... Please wait"):
                    faces, emotions_list, annotated = process_image(image_rgb)

                    st.session_state["last_faces"]      = faces
                    st.session_state["last_emotions"]   = emotions_list
                    st.session_state["last_annotated"]  = annotated

                    # Update stats
                    st.session_state.total_scans += 1
                    for dominant, emotions in emotions_list:
                        st.session_state.history.insert(0, {
                            "emotion":    dominant,
                            "confidence": round(emotions.get(dominant, 0), 1),
                            "faces":      len(faces)
                        })

        # Show annotated image
        if "last_annotated" in st.session_state:
            st.markdown('<div class="section-title" style="margin-top:1rem">🎯 Detection Result</div>', unsafe_allow_html=True)
            st.image(
                st.session_state["last_annotated"],
                caption=f"{len(st.session_state['last_faces'])} face(s) detected",
                use_column_width=True
            )

    with col_right:
        st.markdown('<div class="section-title">📊 Emotion Analysis</div>', unsafe_allow_html=True)

        if "last_emotions" not in st.session_state or len(st.session_state["last_emotions"]) == 0:
            st.info("Upload an image and click DETECT EMOTIONS to see results.")
        else:
            emotions_list = st.session_state["last_emotions"]

            for face_idx, (dominant, emotions) in enumerate(emotions_list):
                config = EMOTION_CONFIG.get(dominant, EMOTION_CONFIG["neutral"])

                st.markdown(f"**Face {face_idx + 1}**" if len(emotions_list) > 1 else "")

                # Main emotion card
                st.markdown(f"""
                <div class="emotion-card">
                    <div class="emotion-emoji">{config['emoji']}</div>
                    <div class="emotion-label" style="color:{config['color']}">{config['label'].upper()}</div>
                    <div class="emotion-confidence">{emotions.get(dominant, 0):.1f}% confidence</div>
                    <div style="margin-top:0.8rem;background:{config['color']}22;border:1px solid {config['color']}44;border-radius:8px;padding:0.5rem;font-family:'Inter',sans-serif;font-size:0.85rem;color:{config['color']}">
                        💼 Jazz Action: {config['jazz_note']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # All emotions bar chart
                st.markdown('<div class="bar-container">', unsafe_allow_html=True)
                st.markdown(
                    "<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#94a3b8;letter-spacing:2px;margin-bottom:0.8rem'>ALL EMOTIONS</div>",
                    unsafe_allow_html=True
                )

                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                for emotion_name, score in sorted_emotions:
                    cfg   = EMOTION_CONFIG.get(emotion_name, EMOTION_CONFIG["neutral"])
                    width = min(score, 100)
                    st.markdown(f"""
                    <div class="bar-row">
                        <div class="bar-label">{cfg['emoji']} {cfg['label']}</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{width}%;background:{cfg['color']}"></div>
                        </div>
                        <div class="bar-pct">{score:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MODE 2: LIVE WEBCAM
# ============================================================

else:
    st.markdown('<div class="section-title">🎥 Live Webcam Detection</div>', unsafe_allow_html=True)

    st.info("💡 The webcam mode captures a frame from your camera and analyzes it. Click the button to capture and analyze.")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        capture_btn = st.button("📸 CAPTURE & ANALYZE", use_container_width=True)

    if capture_btn:
        with st.spinner("Opening webcam and capturing frame..."):
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not open webcam. Make sure your camera is connected and not in use by another app.")
            else:
                time.sleep(0.5)  # warm up camera
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.markdown("**📸 Captured Frame**")
                        st.image(frame_rgb, use_column_width=True)

                    with st.spinner("Analyzing emotions..."):
                        faces, emotions_list, annotated = process_image(frame_rgb)
                        st.session_state.total_scans += 1

                    with col_left:
                        st.markdown("**🎯 Annotated Result**")
                        st.image(annotated, caption=f"{len(faces)} face(s) detected", use_column_width=True)

                    with col_right:
                        st.markdown('<div class="section-title">📊 Results</div>', unsafe_allow_html=True)

                        if len(emotions_list) == 0:
                            st.warning("No face detected. Please try again with better lighting.")
                        else:
                            for face_idx, (dominant, emotions) in enumerate(emotions_list):
                                config = EMOTION_CONFIG.get(dominant, EMOTION_CONFIG["neutral"])
                                st.session_state.history.insert(0, {
                                    "emotion":    dominant,
                                    "confidence": round(emotions.get(dominant, 0), 1),
                                    "faces":      len(faces)
                                })

                                st.markdown(f"""
                                <div class="emotion-card">
                                    <div class="emotion-emoji">{config['emoji']}</div>
                                    <div class="emotion-label" style="color:{config['color']}">{config['label'].upper()}</div>
                                    <div class="emotion-confidence">{emotions.get(dominant, 0):.1f}% confidence</div>
                                    <div style="margin-top:0.8rem;background:{config['color']}22;border:1px solid {config['color']}44;border-radius:8px;padding:0.5rem;font-family:'Inter',sans-serif;font-size:0.85rem;color:{config['color']}">
                                        💼 Jazz Action: {config['jazz_note']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                                for emotion_name, score in sorted_emotions:
                                    cfg   = EMOTION_CONFIG.get(emotion_name, EMOTION_CONFIG["neutral"])
                                    width = min(score, 100)
                                    st.markdown(f"""
                                    <div class="bar-row">
                                        <div class="bar-label">{cfg['emoji']} {cfg['label']}</div>
                                        <div class="bar-track">
                                            <div class="bar-fill" style="width:{width}%;background:{cfg['color']}"></div>
                                        </div>
                                        <div class="bar-pct">{score:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.error("Failed to capture frame. Please try again.")

# ============================================================
# DETECTION HISTORY
# ============================================================

if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Detection History</div>', unsafe_allow_html=True)

    for item in st.session_state.history[:10]:
        config = EMOTION_CONFIG.get(item["emotion"], EMOTION_CONFIG["neutral"])
        st.markdown(f"""
        <div class="history-item">
            <div>{config['emoji']} <b>{config['label']}</b></div>
            <div style="color:#64748b">{item['faces']} face(s) detected</div>
            <div style="color:{config['color']};font-family:'Space Mono',monospace;font-size:0.8rem">
                {item['confidence']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER — What this demonstrates
# ============================================================

st.markdown("---")
st.markdown('<div class="section-title">💼 What This Project Demonstrates</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
skills = [
    ("🔲", "Face Detection",     "Haar Cascade classifier detects face regions in any image"),
    ("🧠", "Deep Learning",      "DeepFace neural network classifies 7 emotions with high accuracy"),
    ("🎨", "Image Annotation",   "OpenCV draws bounding boxes and labels on detected faces"),
    ("💼", "Business Logic",     "Emotion mapped to customer service action recommendations"),
]
for col, (emoji, title, desc) in zip([col1, col2, col3, col4], skills):
    with col:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:1.8rem">{emoji}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#2563eb;margin:0.5rem 0">{title}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#64748b">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Inter',sans-serif;font-size:0.75rem;color:#cbd5e1">
    Customer Emotion Detector · Built with DeepFace & OpenCV ·
</div>
""", unsafe_allow_html=True)
