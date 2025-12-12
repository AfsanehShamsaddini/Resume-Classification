import pandas as pd
from pathlib import Path

RAW = Path("data/raw/Cleaned_Resumes.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not RAW.exists():
        raise FileNotFoundError("Cleaned_Resumes.csv not found in data/raw/")

    df = pd.read_csv(RAW)

    # Expected: Category, Resume_Details
    if "Category" not in df.columns or "Resume_Details" not in df.columns:
        raise ValueError(f"Expected columns: Category, Resume_Details. Found: {df.columns.tolist()}")

    df = df.dropna(subset=["Category", "Resume_Details"]).copy()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Resume_Details"] = df["Resume_Details"].astype(str).str.strip()

    out = OUT_DIR / "dataset.csv"
    df.to_csv(out, index=False)

    print(f"[OK] Saved: {out} rows={len(df)} cols={len(df.columns)}")
    print("[INFO] Class counts:")
    print(df["Category"].value_counts())

if __name__ == "__main__":
    main()
