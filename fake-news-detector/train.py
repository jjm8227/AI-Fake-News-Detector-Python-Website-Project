import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_news.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].map({"real": 1, "fake": 0})
    return df


def train_model(df: pd.DataFrame):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model


def save_model(model, path: Path = MODEL_PATH) -> None:
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def main() -> None:
    df = load_data()
    model = train_model(df)
    save_model(model)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
