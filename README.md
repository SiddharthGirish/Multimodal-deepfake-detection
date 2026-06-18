# Deepfake Audio-Video Detector 🎭

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

A professional, multi-modal **Deepfake Audio-Video Detection System** that analyzes video uploads to detect signs of artificial manipulation. This project extracts visual features using a pretrained **EfficientNet-B0** convolutional neural network and audio features using a pretrained **Wav2Vec2** speech transformer, allowing users to configure classification thresholds through an interactive Streamlit UI.

---

## 🚀 Features
- **Multi-Modal Feature Extraction**: Combines state-of-the-art CNNs for video frames with transformer-based speech models for audio signals.
- **Interactive Decision Slider**: Dynamically adjust the classification probability threshold.
- **FFmpeg Integration**: Programmatically extracts audio tracks from video streams (with fallback safety warnings if FFmpeg is missing on the host system).
- **Clean Workspace Management**: Automated cleanup of temporary files and uploaded video files after analysis.
- **Jupyter Notebook Support**: Step-by-step pipeline exploration in `notebooks/deepfake_detection_demo.ipynb`.

---

## 🛠️ Technologies Used
- **Streamlit**: Web interface and interactive UI.
- **PyTorch**: Deep learning backend.
- **PyTorch Image Models (`timm`)**: Pretrained EfficientNet-B0 visual encoder.
- **Hugging Face Transformers**: Wav2Vec2 speech processor and embedding model.
- **OpenCV (`opencv-python`)**: Video capture and frame preprocessing.
- **Librosa**: Audio signal loading and digital signal processing.
- **FFmpeg**: CLI utility for audio stream extraction from video containers.
- **NumPy**: Matrix computations and embedding alignment.

---

## 📥 Installation Steps

Follow these instructions to set up the project locally:

### 1. Prerequisites
Ensure you have **Python 3.9+** and **FFmpeg** installed on your system.

- **FFmpeg Installation**:
  - *macOS*: `brew install ffmpeg`
  - *Windows*: Download from the [Official FFmpeg site](https://ffmpeg.org/download.html) and add the binary folders to your system's environment variable `PATH`.
  - *Linux*: `sudo apt-get install ffmpeg`

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/deepfake-audio-video-detector.git
cd deepfake-audio-video-detector
```

### 3. Create a Virtual Environment
```bash
python -m venv deepfake_env
```
- **Activate the environment**:
  - *Windows*: `deepfake_env\Scripts\activate`
  - *macOS/Linux*: `source deepfake_env/bin/activate`

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Usage Instructions

### Run the Web Interface
Start the Streamlit application from the root directory:
```bash
streamlit run app.py
```
Open the provided local URL (typically `http://localhost:8501`) in your web browser.

1. **Adjust Threshold**: Set the sliding bar to your desired probability threshold (default is 60%).
2. **Upload Video**: Upload a short `.mp4`, `.avi`, or `.mov` file.
3. **Analyze**: Review the calculated manipulation score and the final classification output.

### Run Notebook Demonstration
Explore how individual models work by opening the Jupyter Notebook:
```bash
jupyter notebook notebooks/deepfake_detection_demo.ipynb
```

---

## 📂 Project Structure
```
deepfake_audio_video_detector/
├── .env.example            # Template for system configurations
├── .gitignore              # Ignores venvs, cache, and large media files
├── app.py                  # Main Streamlit web application
├── CHANGELOG.md            # Release updates log
├── CONTRIBUTING.md         # Open-source contribution guidelines
├── LICENSE                 # MIT License file
├── PROJECT_REPORT.md       # Technical report on pipeline models & metrics
├── README.md               # User manual & project overview
├── requirements.txt        # Python package dependencies list
├── notebooks/
│   └── deepfake_detection_demo.ipynb  # Interactive pipeline walkthrough
├── samples/                # Sample video assets folder
│   └── (Sample videos are kept locally and ignored by Git LFS / .gitignore)
└── src/
    ├── __init__.py         # Package initialization
    └── model_utils.py      # Feature extraction and prediction functions
```

---

## 🖼️ Screenshots Section

Below is a visual layout of the Streamlit application in action:

*(Placeholder image for Streamlit application dashboard)*
![Deepfake Detection Dashboard Interface](https://via.placeholder.com/800x450.png?text=Streamlit+Deepfake+Detection+Dashboard+Placeholder)

---

## 🔮 Future Improvements
1. **Classifier Head Training**: Replace the current heuristic logic with a trained Multi-Layer Perceptron (MLP) or SVM head using deepfake dataset embeddings (e.g. Celeb-DF).
2. **Temporal Video Modeling**: Sample multiple frames throughout the video and process them with a Recurrent Neural Network (LSTM/GRU) or video transformer to identify changes across time.
3. **Voice Cloning Detection**: Fine-tune Wav2Vec2 on synthetic speech datasets to distinguish between real human voices and AI-generated ones.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✍️ Author Information
- **Name**: Siddharth Girish
- **LinkedIn**: [SIDDHARTH GIRISH](www.linkedin.com/in/siddharthgirish)
