# Fake News Detector

This project is a simple starter for building a fake news detector using Python.

## Structure
- `train.py` trains a text classifier from a sample dataset.
- `app.py` provides a small Streamlit web interface.
- `data/sample_news.csv` contains sample real/fake article examples.

## Setup
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Notes
This is a beginner-friendly baseline. A real-world detector would need a larger and more diverse dataset.
