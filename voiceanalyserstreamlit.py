import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa

st.set_page_config(page_title="AI Task Optimizer - Voice Emotion", page_icon="🎙️", layout="centered")
st.title("🎙️ AI Task Optimizer - Voice Emotion Analyzer")

# ===== Step 1: Record =====
duration = st.slider("Select recording duration (seconds)", 3, 15, 5)
fs = 44100  # Sampling rate

if st.button("🎤 Record Voice"):
    st.info("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    st.success("✅ Recording complete!")

    # Save voice to WAV file
    wav_path = "recorded_voice.wav"
    sf.write(wav_path, recording, fs)

    # Step 2: Playback
    st.audio(wav_path)

    # Step 3: Analyze using Librosa
    y, sr = librosa.load(wav_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    pitch = np.mean(librosa.yin(y, fmin=80, fmax=400))

    # Basic mood logic
    if rms < 0.02 and tempo < 70:
        emotion = "😔 Calm / Sad"
    elif rms > 0.05 and tempo > 100:
        emotion = "😄 Energetic / Happy"
    else:
        emotion = "😐 Neutral"

    st.subheader("🧠 Voice Analysis Results")
    st.write(f"- **Average Pitch:** {pitch:.2f} Hz")
    st.write(f"- **Energy (RMS):** {rms:.4f}")
    st.write(f"- **Tempo:** {tempo:.2f} BPM")
    st.write(f"- **Detected Emotion:** {emotion}")

    st.success(f"✅ Emotion detected as: {emotion}")
