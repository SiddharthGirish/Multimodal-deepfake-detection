"""
Deepfake Audio-Video Detector Web App.

A Streamlit application that accepts video uploads, extracts frame-level and
audio features, and predicts whether the video is a deepfake based on a user-defined threshold.
"""

import os
import streamlit as st
from src.model_utils import predict_fake_score, is_ffmpeg_available

# Streamlit Page configuration
st.set_page_config(
    page_title="Deepfake Audio-Video Detector",
    page_icon="🎭",
    layout="centered"
)

# Title and Description
st.title("🎭 Deepfake Detection System")
st.markdown(
    "Upload a short video file (MP4, AVI, or MOV) to analyze if it contains signs of synthetic manipulation.\n\n"
    "Use the slider below to set the decision threshold for labeling a video as 'Probably Fake'."
)

# Check for system dependencies (FFmpeg is required for audio feature extraction)
if not is_ffmpeg_available():
    st.warning(
        "⚠️ **System Warning**: FFmpeg command-line utility is not found in your system PATH. "
        "Audio analysis is currently disabled; predictions will rely solely on visual frame features."
    )

# 1. Slider to configure decision threshold
st.subheader("Configuration")
threshold = st.slider("Set probability threshold (%)", min_value=0, max_value=100, value=60) / 100.0

# 2. File uploader for videos
st.subheader("Video Analysis")
uploaded = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

if uploaded:
    # Get file extension and prepare temporary upload path
    video_ext = uploaded.name.split('.')[-1]
    video_path = f"uploaded_video.{video_ext}"
    
    # Save uploaded file to disk
    with open(video_path, "wb") as f:
        f.write(uploaded.read())

    # Preview the uploaded video
    st.video(video_path)
    
    st.info("🔄 Analyzing video and audio signals... please wait (this might take 30–60 seconds).")

    try:
        # Run prediction
        fake_score = predict_fake_score(video_path)

        # Display analysis results
        st.subheader("Analysis Results")
        st.write(f"**Calculated Fake Probability:** {fake_score * 100:.2f}%")

        if fake_score >= threshold:
            st.error(f"⚠️ **Result:** This video is **Probably FAKE**! (Probability exceeds {threshold * 100:.1f}%)")
        else:
            st.success(f"✅ **Result:** This video is **Probably ORIGINAL**. (Probability is below {threshold * 100:.1f}%)")
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        
    finally:
        # Clean up the temporary video file to avoid polluting the workspace
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass