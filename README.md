
│   ├── make_dataset.py     # prepare processed dataset
│   ├── train.py            # train + save best model
│   ├── evaluate.py         # metrics + confusion matrix plot
│   └── predict.py          # CLI inference from raw text
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── model.joblib
├── reports/
│   ├── metrics.json
│   ├── classification_report.txt
│   └── figures/
│       └── confusion_matrix.png
│       └──model_accuracy.png
├── notebooks/
└── requirements.txt

````

---

## ⚙️ Setup
Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 🚀 How to Run (End-to-End)

### 1️⃣ Build processed dataset

Creates: `data/processed/dataset.csv`

```bash
python -m src.make_dataset
```

### 2️⃣ Train model

Creates: `models/model.joblib`

```bash
python -m src.train
```

### 3️⃣ Evaluate model + generate plots

Creates:

* `reports/metrics.json`
* `reports/classification_report.txt`
* `reports/figures/confusion_matrix.png`

```bash
python -m src.evaluate
```

---

## 📈 Metrics

Metrics are stored in:

* `reports/metrics.json`

Recommended metrics for multi-class text classification:

