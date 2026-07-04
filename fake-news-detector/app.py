"""Simple Streamlit UI for a baseline fake-news detector.

This file exposes a `fetch_article_text` helper used by the UI. Comments are added
to make it easier for contributors to understand where to modify behavior.
"""

import pickle
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
import streamlit as st

from train import MODEL_PATH, load_data, train_model, save_model


def fetch_article_text(url: str, timeout: int = 10) -> Optional[str]:
    """Fetches `url` and returns extracted article text or `None`.

    Strategy: prefer content inside an <article> element; otherwise join all
    <p> tags. Returns None for non-HTML responses or on errors.
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        content_type = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or "text/html" not in content_type:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Prefer <article> tags when present
        article = soup.find("article")
        if article:
            texts = [p.get_text(separator=" ", strip=True) for p in article.find_all("p")]
            return "\n\n".join(t for t in texts if t)
        # Fallback: collect main <p> tags
        paragraphs = soup.find_all("p")
        texts = [p.get_text(separator=" ", strip=True) for p in paragraphs]
        text = "\n\n".join(t for t in texts if t)
        return text if text.strip() else None
    except Exception:
        return None


# Streamlit UI
st.set_page_config(page_title="AI Fake News Detector", page_icon="📰")
st.title("Fake News Detector")
st.write("Paste a news article or provide a URL of one — the model will estimate whether the context of it is real or fake.")

# Load saved model if available; otherwise train on bundled sample dataset and save it.
if MODEL_PATH.exists():
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
else:
    df = load_data()
    model = train_model(df)
    save_model(model)

# Persist the last fetched/edited article in Streamlit session state so the UI
# doesn't lose content when buttons are clicked.
if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""


# URL input + fetch flow
url = st.text_input("Article URL")
# The user can click the button to fetch the URL, extract text, and get a
# prediction immediately. We separate this from the manual text area below so
# users can edit the fetched text before re-predicting.
if st.button("Fetch and Predict"):
    if url.strip():
        with st.spinner("Fetching article..."):
            text = fetch_article_text(url.strip())
        if not text:
            st.warning("Could not extract article text from the URL.")
        else:
            st.session_state["article_text"] = text
            prediction = model.predict([text])[0]
            probability = model.predict_proba([text])[0][1]
            label = "Likely real" if prediction == 1 else "Likely fake"
            st.success(f"{label} ({probability:.2%})")
    else:
        st.warning("Please enter a URL first.")


# Manual text area (read-only). Users fetch a URL and then edit outside the app
# if they need to change text; disabling prevents accidental edits in the UI.
article_text = st.text_area(
    "Article text",
    value=st.session_state["article_text"],
    height=250,
    disabled=True,
)


st.caption("Note: URL extraction is best-effort and may miss articles on some sites.")
