"""
Model utility functions for the Deepfake Detection System.

This module provides helper functions to extract visual features using a pretrained
EfficientNet-B0 model and audio features using a Wav2Vec2 model. These features
can be used to evaluate whether a video is a deepfake.
"""

import os
import shutil
import subprocess
import cv2
import librosa
import numpy as np
import timm
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Device configuration (Use GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pretrained feature extraction models
# 1. Visual Feature Extractor: EfficientNet-B0 (pretrained on ImageNet)
visual_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0).to(device).eval()

# 2. Audio Feature Extractor: Wav2Vec2 (pretrained base model from Hugging Face)
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()


def is_ffmpeg_available() -> bool:
    """
    Check if ffmpeg command-line utility is installed and available in system PATH.

    Returns:
        bool: True if ffmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def extract_frame(video_path: str) -> np.ndarray:
    """
    Extract the first frame from a video file and generate its visual embedding.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        np.ndarray: Image feature embeddings representing the first frame.

    Raises:
        ValueError: If the video frame cannot be read successfully.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame from video: {video_path}")
        
    # Preprocess image frame for EfficientNet (RGB, resize to 224x224, normalize, convert to tensor)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    tensor = torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        v_emb = visual_model(tensor).cpu().numpy()
        
    return v_emb


def extract_audio_features(video_path: str) -> np.ndarray | None:
    """
    Extract audio track from the video file using FFmpeg, preprocess it,
    and generate the audio embeddings using Wav2Vec2.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        np.ndarray | None: Audio embeddings representing the audio channel,
                           or None if audio extraction fails or no audio is present.
    """
    if not is_ffmpeg_available():
        print("Warning: FFmpeg is not installed or not found in system PATH. Audio feature extraction is skipped.")
        return None

    os.makedirs("temp", exist_ok=True)
    wav_path = os.path.join("temp", "temp_audio.wav")

    # Command to extract mono audio at 16kHz sample rate without video
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print(f"Warning: Could not extract audio from {video_path}.")
        return None

    # Verify audio file was created and is non-empty
    if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
        print(f"Warning: No audio found or empty audio channel in {video_path}.")
        return None

    # Load audio signal (resampled to 16kHz)
    wav, sr = librosa.load(wav_path, sr=16000)

    # Clean up temporary audio file
    try:
        os.remove(wav_path)
    except OSError:
        pass

    # Preprocess and obtain Wav2Vec2 embeddings
    inputs = audio_processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = audio_model(**{k: v.to(device) for k, v in inputs.items()})
    
    # Pool output (mean pooling over time dimension)
    a_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return a_emb


def predict_fake_score(video_path: str, threshold: float = 0.5) -> float:
    """
    Compute a fake probability score for a given video file.
    
    Currently extracts visual frame embeddings and audio embeddings,
    and returns a placeholder probability score.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold to classify a video as fake (unused in placeholder).

    Returns:
        float: Fake probability score (range 0.0 to 1.0).
    """
    try:
        v_emb = extract_frame(video_path)
    except Exception as e:
        print(f"Error during visual extraction: {e}")
        v_emb = None

    a_emb = extract_audio_features(video_path)

    if a_emb is None:
        print("No audio found or audio processing failed. Using visual features only.")
        return 0.0  # Assumes original or fallback
    else:
        # Placeholder probability calculation (e.g. constant small probability for demonstration)
        fake_prob = 0.1
        return fake_prob


if __name__ == "__main__":
    # Quick self-test script
    test_video = "your_video.mp4"
    if os.path.exists(test_video):
        print(f"Testing deepfake prediction on: {test_video}")
        score = predict_fake_score(test_video)
        print(f"Prediction result: {'FAKE' if score >= 0.5 else 'ORIGINAL'} (Score: {score:.2f})")
    else:
        print("No default test video found. Run python inside the root directory with a valid video path.")
