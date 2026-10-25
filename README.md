# CrimeVision AI — Intelligent Multimodal Forensic Analysis Platform

CrimeVision AI is a Python-based forensic analysis web application built with Streamlit. It helps users analyze image, video, and audio evidence using automated analysis tools and an AI-powered forensic assistant.

## Features

- **Image Analysis:** Upload images and detect objects with confidence scores.
- **Video Analysis:** Analyze video frames and review detected objects.
- **Audio Analysis:** Upload audio recordings and generate speech transcripts.
- **AI Forensic Assistant:** Ask questions about recorded evidence findings using the Groq API.
- **PDF Report Generation:** Export forensic analysis summaries as PDF reports.
- **Evidence History:** Review evidence and findings recorded during the session.

## Technologies Used

- Python
- Streamlit
- OpenCV
- Groq API
- Requests
- Faster-Whisper
- ReportLab

## How to Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/Ruchiiii001/c-c-.git
   ```

2. Open the project folder:

   ```bash
   cd c-c-
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your Groq API key in Streamlit secrets:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

5. Start the application:

   ```bash
   streamlit run app_new.py
   ```

## Live Demo

[Open CrimeVision AI](https://crimevision-ai-forensics.streamlit.app)

## Important Note

This application supports preliminary forensic analysis. Automated detections and AI-generated responses are not definitive forensic conclusions. Always verify results against the original evidence.

## Future Improvements

- Improve object detection accuracy.
- Add more advanced evidence correlation.
- Improve forensic report formatting.
- Expand testing across image, video, and audio inputs.
