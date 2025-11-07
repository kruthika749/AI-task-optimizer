import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.title("🎤 Microphone Test")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True,
    key="test"
)

if audio:
    st.audio(audio["bytes"], format="audio/wav")
    st.write(f"Audio bytes length: {len(audio['bytes'])}")
