import json
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

DATA = Path("data/processed/dataset.csv")
MODEL = Path("models/model.joblib")

REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run first: python -m src.make_dataset")
    if not MODEL.exists():
        raise FileNotFoundError("Run first: python -m src.train")

    df = pd.read_csv(DATA)
    X = df["Resume_Details"].astype(str)
    y = df["Category"].astype(str)

    # same split settings as train.py
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = joblib.load(MODEL)
    pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "n_test": int(len(y_test)),
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS / "classification_report.txt").write_text(
        classification_report(y_test, pred), encoding="utf-8"
    )

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)

    for (i, j), v in np.ndenumerate(cm):
        if v != 0:
            plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[OK] Saved reports/metrics.json + reports/figures/confusion_matrix.png")
    print(metrics)

if __name__ == "__main__":
    main()
