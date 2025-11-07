# taskoptimizer_streamlit.py
import streamlit as st
from transformers import pipeline
from deepface import DeepFace
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
import io
import os
import tempfile

import streamlit as st

# ---------------------------
# 🌈 Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Emotion-Based Task Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 🎨 Light Pastel Theme Styling
# ---------------------------
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    font-family: "Poppins", sans-serif;
}

/* Main title */
h1 {
    text-align: center;
    color: #003566;
    font-weight: 700;
    font-size: 2.6em;
}

/* Subtitles */
h2, h3 {
    color: #0077b6;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #0096c7, #00b4d8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.3em;
    font-size: 16px;
    font-weight: 600;
    transition: 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #48cae4, #90e0ef);
    transform: scale(1.05);
}

/* Result cards */
.result-card {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 25px;
    margin: 25px 0;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}

/* Task list */
.result-card ul {
    list-style-type: none;
    padding: 0;
}
.result-card li {
    background: #caf0f8;
    color: #023e8a;
    padding: 10px;
    margin: 6px 0;
    border-radius: 8px;
    font-size: 16px;
    transition: 0.2s;
}
.result-card li:hover {
    background: #ade8f4;
    transform: translateX(5px);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.8);
    backdrop-filter: blur(8px);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 1rem;
    color: #0077b6;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# audio
import librosa
import soundfile as sf

# --------------------------
# Page config & session state
# --------------------------
st.set_page_config(page_title="AI Task Optimizer", layout="wide")
if "text_model" not in st.session_state:
    st.session_state.text_model = None
if "deepface_ok" not in st.session_state:
    st.session_state.deepface_ok = True

DB_PATH = "predictions.db"

# --------------------------
# Database helpers
# --------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            input_text TEXT,
            face_emotion TEXT,
            voice_emotion TEXT,
            text_emotion TEXT,
            confidence REAL,
            details TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn

conn = init_db()

def log_prediction(source, input_text, face_emotion, voice_emotion, text_emotion, confidence, details):
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO predictions (source, input_text, face_emotion, voice_emotion, text_emotion, confidence, details, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (source, input_text, face_emotion, voice_emotion, text_emotion, confidence, details, ts),
    )
    conn.commit()

# --------------------------
# Model loading (cached)
# --------------------------
@st.cache_resource
def load_text_model():
    try:
        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        # fallback to default pipeline if model name fails
        model = pipeline("sentiment-analysis")
    return model

# --------------------------
# Text analysis
# --------------------------
def analyze_text(text: str):
    if not st.session_state.text_model:
        st.session_state.text_model = load_text_model()
    result = st.session_state.text_model(text)[0]
    label = result.get("label", "NEUTRAL")
    score = float(result.get("score", 0.0))
    if label.upper() in ("POSITIVE", "LABEL_1"):
        emotion = "positive"
    elif label.upper() in ("NEGATIVE", "LABEL_0"):
        emotion = "negative"
    else:
        emotion = "neutral"
    return emotion, score, str(result)

# --------------------------
# Face analysis (DeepFace)
# --------------------------
def analyze_face(image: Image.Image):
    try:
        arr = np.array(image.convert("RGB"))
        analysis = DeepFace.analyze(arr, actions=["emotion"], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        dominant = analysis.get("dominant_emotion", None)
        emotion_scores = analysis.get("emotion", {})
        top_score = max(emotion_scores.values()) / (sum(emotion_scores.values()) + 1e-8) if emotion_scores else 0.0
        return dominant, float(top_score), str(emotion_scores)
    except Exception as e:
        st.session_state.deepface_ok = False
        return None, 0.0, f"DeepFace error: {e}"
#voice analysis
from pydub import AudioSegment
import io

import io
import numpy as np
import librosa
import streamlit as st
from pydub import AudioSegment

import io
import numpy as np
import librosa
import soundfile as sf
import streamlit as st

import io
import numpy as np
import librosa
import streamlit as st
import soundfile as sf
import io
import numpy as np
import librosa
import streamlit as st
import soundfile as sf
import base64
from io import BytesIO

def analyze_voice(audio_bytes):
    try:
        # 🔹 STEP 1: Some versions of mic_recorder give base64 strings — handle both
        if isinstance(audio_bytes, str):
            if "base64," in audio_bytes:
                audio_bytes = audio_bytes.split("base64,")[-1]
            audio_bytes = base64.b64decode(audio_bytes)

        # 🔹 STEP 2: Try reading as a proper WAV/OGG/FLAC stream
        try:
            data, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            # 🔹 STEP 3: If fails, assume it's a WebM blob — convert to PCM
            import av  # from PyAV (install once: pip install av)
            container = av.open(io.BytesIO(audio_bytes))
            stream = container.streams.audio[0]
            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray().flatten())
            y = np.concatenate(frames)
            sr = stream.rate
        else:
            y = np.mean(data, axis=1) if len(data.shape) > 1 else data

        # --- Handle silent or too-short audio ---
        if y is None or len(y) < 2000:
            st.warning("⚠️ Audio too short or silent.")
            return "Neutral", 0.6, "Audio too short to analyze properly."

        # --- Compute features ---
        rms = float(np.mean(librosa.feature.rms(y=y)))
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0]) if len(y) > 5000 else 0
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        energy = float(np.mean(np.abs(y)))
        duration = len(y) / sr

        st.write("📊 Voice Features:")
        st.json({
            "RMS": round(rms, 5),
            "Tempo": round(tempo, 2),
            "ZCR": round(zcr, 5),
            "Energy": round(energy, 5),
            "Duration": round(duration, 2)
        })

        # --- Emotion Heuristic ---
        if rms > 0.03 and tempo > 130:
            emotion = "Angry / Excited"
            confidence = min(0.95, rms * 10)
        elif rms > 0.02 and tempo > 110:
            emotion = "Happy / Energetic"
            confidence = min(0.9, rms * 8)
        elif rms < 0.01 and tempo < 100:
            emotion = "Sad / Calm"
            confidence = min(0.85, 1 - rms * 50)
        else:
            emotion = "Neutral / Relaxed"
            confidence = 0.7

        return emotion, confidence, (
            f"RMS={rms:.4f}, Tempo={tempo:.2f}, "
            f"ZCR={zcr:.4f}, Energy={energy:.4f}, Duration={duration:.2f}s"
        )

    except Exception as e:
        st.error(f"⚠️ Voice analysis failed: {e}")
        return "Neutral", 0.5, f"Error occurred: {e}"

# --------------------------
# Task recommendations
import random
from transformers import pipeline

# Optional: small text-generation model (you can replace with a local or HuggingFace model)
try:
    task_gen = pipeline("text-generation", model="gpt2")
except Exception:
    task_gen = None

def recommend_tasks(text_emotion=None, face_emotion=None, voice_emotion=None, user_text=None):
    # Pick whichever emotion is available
    dominant_emotion = text_emotion or face_emotion or voice_emotion
    if not dominant_emotion:
        return ["Take a short break, then try analyzing again."], "No emotion detected"

    dominant_emotion = dominant_emotion.lower()

    # Fallback static task suggestions (used if GPT fails)
    task_map = {
        "happy": [
            "Plan your next creative project 🎨",
            "Write a gratitude journal entry ✨",
            "Tidy up your workspace while listening to music 🎧",
            "Reach out to a friend and share good news 💬",
            "Take on a challenging or exciting task 🚀",
            "Set new personal goals — you’re in a great mindset!"
        ],
        "sad": [
            "Take a short walk or go outside 🌳",
            "Write about what’s bothering you 📝",
            "Listen to calming music or meditate 🎵",
            "Focus on small, achievable tasks ✅",
            "Call a close friend or family member 💬",
            "Do something comforting — watch a favorite show 🍵"
        ],
        "angry": [
            "Pause and take deep breaths 🧘",
            "Go for a run or quick workout 🏃‍♂️",
            "Work on a task that requires physical energy 💪",
            "Avoid making major decisions right now ⚠️",
            "Write out what made you angry — then delete it 🧾",
            "Switch to a neutral or repetitive task to cool off 🔄"
        ],
        "neutral": [
            "Organize your schedule for the week 📅",
            "Respond to pending emails or messages 📬",
            "Review your goals and progress 📈",
            "Learn something new — watch a short tutorial 🎓",
            "Do light chores or digital cleanup 🧹",
            "Read or listen to something inspirational 📚"
        ],
        "fear": [
            "List out what’s worrying you and why 🕵️",
            "Do grounding exercises or deep breathing 🌬️",
            "Focus on one small, controllable task ✅",
            "Seek reassurance or talk to someone you trust 🤝",
            "Distract yourself with a calm, routine activity 🧩",
            "Avoid overloading yourself — keep tasks light today 🌤️"
        ],
        "surprise": [
            "Reflect on what surprised you — write your thoughts 💭",
            "Take advantage of this energy — brainstorm new ideas 💡",
            "Capture your reaction in a journal or note 📝",
            "Do something spontaneous but fun 🎢",
            "Try exploring new topics or hobbies 🧭",
            "Talk about your experience with a friend 🤗"
        ],
        "disgust": [
            "Take a break from what’s bothering you ☕",
            "Do a reset — clean up your environment 🧼",
            "Switch to a completely different kind of task 🔄",
            "Watch or listen to something positive 🌈",
            "Write down what caused your disgust — and reframe it 🧠",
            "Focus on self-care or a refreshing task 🪞"
        ]
    }

    # Get tasks for the detected emotion (default: neutral)
    tasks = task_map.get(dominant_emotion, ["Take a short break", "Stay hydrated"])

    # Return tasks and the emotion that triggered them
    return tasks, dominant_emotion


# --------------------------
# UI layout
# --------------------------
st.title("🤖 AI Task Optimizer — Streamlit (Integrated)")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Inputs")
    st.subheader("1) Text Input")
    user_text = st.text_area("Enter text to analyze (chat, status, diary...)", height=120)

    st.subheader("2) Face Input (Camera or Upload)")
    st.write("Use webcam or upload an image for face emotion detection")
    face_mode = st.radio("Face input method", ("Skip", "Camera snapshot", "Upload image"))
    face_image = None
    if face_mode == "Camera snapshot":
        face_file = st.camera_input("Take a picture")
        if face_file:
            face_image = Image.open(face_file)
            st.image(face_image, caption="Captured", use_column_width=True)
    elif face_mode == "Upload image":
        uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"], key="face_up")
        if uploaded:
            face_image = Image.open(uploaded)
            st.image(face_image, caption="Uploaded", use_column_width=True)

    # --- Voice Input ---
# --- Voice Input ---
# --- Voice Input (Mic Recording) ---
# --- Ensure audio_file variable exists ---
audio_file = globals().get("audio_file", None)
st.subheader("🎤 Voice Input (Record and Analyze)")

try:
    from streamlit_mic_recorder import mic_recorder
    import io

    # Step 1: Record the voice
    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="🛑 Stop Recording",
        key="voice_recorder",
        just_once=False,
    )

    if audio:
        st.success("✅ Voice recorded successfully!")

        # Step 2: Play the recorded audio
        st.audio(audio["bytes"], format="audio/wav")

        # Step 3: Analyze the recorded audio
        st.info("🔍 Analyzing your voice input... please wait.")
        try:
            ve, vc, vd = analyze_voice(audio["bytes"])
            if ve:
                st.subheader("🎧 Voice Analysis Result")
                st.write(f"Emotion: **{ve}**, Confidence: **{vc:.2f}**")
                st.write("Details:", vd)

                # Step 4: Suggest tasks based on emotion
                tasks, dom_src = recommend_tasks(None, None, ve)
                st.subheader("🧠 Recommended Tasks (Based on Your Emotion)")
                for t in tasks:
                    st.markdown(f"- {t}")
            else:
                st.warning("Could not detect emotion. Try again with a clearer voice.")

        except Exception as e:
            st.error(f"Voice analysis failed: {e}")

except Exception as e:
    st.warning(f"Microphone recording not available: {e}")
# --- Manual Analysis Button (for text or combined inputs) ---
st.markdown("---")
analyze_btn = st.button("🔍 Analyze Inputs", key="analyze_btn")

if analyze_btn:
    st.info("Analyzing all available inputs (text + voice + emotion)...")
    try:
        # Replace this logic with how you process text inputs
        text_input = st.session_state.get("text_input", "")
        ve = None

        # Example: If you already have analyze_voice or analyze_text logic
        if text_input:
            st.write("🧠 Text Input Found! Running analysis...")
            # Add your text analysis logic here
            # Example: sentiment, emotion, etc.

        st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"Analysis failed: {e}")

st.markdown("---")
st.header("Results")

if analyze_btn:
    # initialize result holders
    text_emotion, text_conf, text_det = None, 0.0, ""
    face_emotion, face_conf, face_det = None, 0.0, ""
    voice_emotion, voice_conf, voice_det = None, 0.0, ""

    # Text
    if user_text and user_text.strip():
        with st.spinner("Analyzing text..."):
            try:
                text_emotion, text_conf, text_det = analyze_text(user_text)
                st.subheader("Text Analysis")
                st.write(f"Emotion: **{text_emotion}**, Confidence: **{text_conf:.2f}**")
                st.write("Model details:", text_det)
            except Exception as e:
                st.error(f"Text analysis error: {e}")
    else:
        st.info("No text provided — skipping text analysis.")

    # Face
    if face_image is not None:
        if not st.session_state.deepface_ok:
            st.warning("Face analysis disabled (DeepFace initialization failed previously).")
        else:
            with st.spinner("Analyzing face..."):
                try:
                    fe, fc, fd = analyze_face(face_image)
                    face_emotion, face_conf, face_det = fe, fc, fd
                    if fe:
                        st.subheader("Face Analysis (DeepFace)")
                        st.write(f"Dominant emotion: **{fe}**, Confidence (rel): **{fc:.2f}**")
                        st.write("Emotion scores:", fd)
                    else:
                        st.warning("No face detected or analysis failed.")
                except Exception as e:
                    st.error(f"Face analysis error: {e}")
    else:
        st.info("No face input.")

    # Voice
    if audio_file is not None:
        with st.spinner("Analyzing voice..."):
            try:
                b = audio_file.read()
                ve, vc, vd = analyze_voice(b)
                voice_emotion, voice_conf, voice_det = ve, vc, vd
                if ve:
                    st.subheader("Voice Analysis (heuristic)")
                    st.write(f"Emotion: **{ve}**, Confidence: **{vc:.2f}**")
                    st.write("Details:", vd)
                else:
                    st.warning("Voice analysis returned no result.")
            except Exception as e:
                st.error(f"Voice analysis error: {e}")
    else:
        st.info("No audio input.")

    # Recommendations
    tasks, dom_src = recommend_tasks(text_emotion, face_emotion, voice_emotion)
    st.subheader("Recommended Tasks")
    st.write(f"Based on: **{dom_src}**")
    for t in tasks:
        st.markdown(f"- {t}")

    # Log to DB
    combined_conf = max(text_conf or 0.0, face_conf or 0.0, voice_conf or 0.0)
    details = f"face_details={face_det} | voice_details={voice_det} | text_details={text_det}"
    try:
        log_prediction(dom_src, user_text, face_emotion, voice_emotion, text_emotion, combined_conf, details)
        st.success("Logged prediction to local SQLite DB.")
    except Exception as e:
        st.error(f"Failed to log: {e}")
# ---------------------------
# 📜 Move Recent Logs to End + Add Delete Option
# ---------------------------
st.markdown("---")
st.header("📜 Recent Logs")

try:
    rows = conn.cursor().execute(
        "SELECT id, source, text_emotion, face_emotion, voice_emotion, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 20"
    ).fetchall()

    import pandas as pd
    if rows:
        df = pd.DataFrame(rows, columns=["id", "source", "text_emotion", "face_emotion", "voice_emotion", "confidence", "timestamp"])
        st.dataframe(df)

        # Add delete option
        st.warning("⚠️ You can delete all logs if needed (this cannot be undone).")
        if st.button("🗑️ Delete All Logs"):
            conn.cursor().execute("DELETE FROM predictions")
            conn.commit()
            st.success("✅ All logs deleted successfully.")
    else:
        st.info("No prediction logs yet.")

except Exception as e:
    st.error(f"DB read error: {e}")

st.markdown("---")
st.write("END:")
st.write(
    """
-THANK YOU HAVE A NICE DAY
"""
)