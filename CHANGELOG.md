# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-18

### Added
- **Project Structure**: Created `src/` directory to modularize the codebase.
- **Dependency Tracking**: Added `requirements.txt` listing all necessary packages.
- **Environment Template**: Introduced `.env.example` for future environment configuration.
- **Ignored Files**: Configured `.gitignore` for Python, Streamlit, and Machine Learning workflows.
- **Interactive Notebook**: Added `notebooks/deepfake_detection_demo.ipynb` showing step-by-step feature extraction.
- **Documentation**: Added `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`, and `PROJECT_REPORT.md`.

### Changed
- **Folder Refactoring**: Renamed sample videos folder to `samples/`.
- **Package Relocation**: Moved and refactored `model_utils.py` to `src/model_utils.py` with enhanced checks for FFmpeg.
- **Streamlit App**: Refactored `app.py` to import from the modular source package, display warnings if FFmpeg is missing, and automatically clean up temporary files.

### Removed
- Removed temporary audio cache `temp/` folder.
