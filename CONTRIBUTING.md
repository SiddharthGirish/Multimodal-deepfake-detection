# Contributing to Deepfake Detection System

Thank you for your interest in contributing to our project! We welcome contributions to improve features, code quality, and documentation.

## How to Contribute

1. **Fork the Repository**: Create a personal copy of the project on GitHub.
2. **Clone the Fork**: Clone it locally using:
   ```bash
   git clone https://github.com/your-username/deepfake-detector.git
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Implement Your Changes**:
   - Ensure your code follows the [PEP 8](https://peps.python.org/pep-0008/) style guide.
   - Write comments and docstrings.
   - Document any UI enhancements or configuration parameters.
5. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**: Submit your pull request to the main branch for review.

## Guidelines for Code Submissions
- Ensure no API keys, secrets, or temporary files (`temp/`, `*.pyc`) are committed.
- If adding new libraries, append them to `requirements.txt`.
- Make sure to test your code locally by running the Streamlit app:
  ```bash
  streamlit run app.py
  ```

## Issues and Feedback
Please feel free to open an issue on GitHub to report bugs, suggest new models (e.g. replacing placeholder heads with trained classifier layers), or propose user interface updates.
