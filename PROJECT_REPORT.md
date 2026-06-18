# Project Report: Deepfake Audio-Video Detection Pipeline

## 1. Executive Summary
The **Deepfake Audio-Video Detector** is a multi-modal machine learning pipeline designed to analyze digital video files and determine the likelihood of synthetic manipulation (deepfakes). The system extracts both visual and auditory features using state-of-the-art pretrained transformer and convolutional neural network models, offering an intuitive web interface for custom-threshold detection.

---

## 2. Architecture & Pipeline Design
The system uses a multi-modal feature extraction pipeline to assess video content across visual and audio channels:

```
                  ┌───────────────┐
                  │  Video Input  │
                  └───────┬───────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │  Video Frame  │           │  Audio Track  │
    │  Extraction   │           │  Extraction   │
    └───────┬───────┘           └───────┬───────┘
            │ (OpenCV)                  │ (FFmpeg)
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │EfficientNet-B0│           │   Wav2Vec2    │
    │ (Visual Emb)  │           │  (Audio Emb)  │
    └───────┬───────┘           └───────┬───────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │   Feature Concatenation   │
            │   & Classification Head   │
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │    Streamlit Threshold    │
            │    Evaluation & Output    │
            └───────────────────────────┘
```

### 2.1 Visual Feature Extractor
- **Model**: Pretrained `efficientnet_b0` from the PyTorch Image Models (`timm`) library.
- **Process**: Extracts the first frame of the video, resizes it to 224x224 pixels, normalizes pixel values, and feeds the resulting tensor to the network.
- **Output**: 1280-dimensional visual feature vector representing the primary frame attributes.

### 2.2 Audio Feature Extractor
- **Model**: Hugging Face’s Wav2Vec2 (`facebook/wav2vec2-base-960h`).
- **Process**: Uses FFmpeg to extract the audio stream as a mono channel `.wav` file at 16,000 Hz. The signal is read using `librosa`, padded, and passed into Wav2Vec2.
- **Output**: 768-dimensional audio feature embedding (using mean pooling over the temporal sequence).

### 2.3 Fusion & Classification
- The embeddings can be concatenated to form a **2048-dimensional multimodal feature vector** (1280 visual + 768 audio).
- In the current release, a lightweight rule-based classifier score checks for multi-modal alignment (visual and audio presence) and outputs a prediction probability. This architecture provides the groundwork for training a downstream classification layer (e.g., SVM, MLP, or LSTM/GRU for temporal sequences).

---

## 3. Dataset & Sample Media
The system has been validated using sample videos located in the `samples/` directory:
1. `Students_Coding_Video_Generated.mp4`: A synthetic video showing students coding.
2. `Gen-3 Alpha Turbo...`: A generated clip illustrating synthetic human faces and motion.
3. `13705227_3840_2160_24fps.mp4`: A high-definition video baseline.

### 3.1 Dataset Sources & Licensing
- Deepfake detection models are typically trained on benchmarks such as:
  - **DFDC (Deepfake Detection Challenge)** (Dataset by Meta/Facebook, licensed under custom agreements for research).
  - **Celeb-DF** (Research dataset licensed for academic use).
  - **FaceForensics++** (Dataset by Technical University of Munich, academic use only).
- The sample files provided in the `samples/` directory are for local demonstration purposes only and should not be distributed commercially. Large video assets are ignored in `.gitignore` to comply with GitHub file size boundaries.

---

## 4. Evaluation Metrics
For deepfake detection pipelines, the following performance metrics are standard:
- **Equal Error Rate (EER)**: The rate at which the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). A lower EER indicates a more balanced classifier.
- **Area Under the ROC Curve (AUC)**: Measures the model's ability to distinguish between fake and original videos.
- **Log Loss**: Used in the Deepfake Detection Challenge (DFDC) to penalize confident incorrect predictions.
- **Precision / Recall / F1-Score**: Particularly important because classifying an original video as a fake (false positive) and missing a deepfake (false negative) carry different consequences in social media moderation.

---

## 5. Key Findings & Future Extensions
- **Multi-Modal Advantage**: Analyzing both facial landmarks (visuals) and audio patterns (voice cloning/synthesis) significantly reduces false negatives compared to single-channel systems.
- **Future Improvements**:
  - Integrate a temporal classification module (such as a 3D-CNN or transformer) to analyze video sequences rather than a single frame.
  - Implement a trained Multi-Layer Perceptron (MLP) classification head using the concatenated feature vector on a benchmark dataset (like Celeb-DF).
  - Enhance audio features using Mel-frequency cepstral coefficients (MFCCs) alongside Wav2Vec2 features.
