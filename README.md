# Readmission-DL — City General Hospital 30-day Readmission Prediction

**Student name:**
**Student ID:**
**Submission date:**

---

## Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

---

## My model

**Architecture:**
Neural network (Keras MLP) with two hidden layers: 64 → 32 units, ReLU activations, and Dropout=0.30 after each hidden layer.

**Key preprocessing decisions:**
- Normalize `blood_pressure_systolic` by converting values < 50 as kPa to mmHg.
- Parse `admission_date` into year/month/day and drop the raw date.
- Median impute `glucose_level_mgdl` and scale numeric features.
- One-hot encode categorical fields, including coded categories (`admission_type`, `discharge_destination`).
- Cap numeric outliers at the 1st/99th percentiles (train-derived).

**How I handled class imbalance:**
I balanced the training data with random oversampling of the minority class (readmitted = 1) to match the majority class size, then trained the MLP on the balanced set.

---

## Results on validation set

| Metric | Value |
|--------|-------|
| AUROC | 0.920 |
| PR-AUC | 0.646 |
| F1 (minority class) | 0.605 |
| Precision (minority) | 0.548 |
| Recall (minority) | 0.676 |
| Decision threshold used | 0.65 |

---

## How to run

### 1. Install dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model (optional — run notebook)

```bash
jupyter notebook notebooks/solution.ipynb
```

### 3. Run inference on the test set

```bash
python src/predict.py --input data/test.csv --output predictions.csv
```

The output CSV will contain two columns: `patient_id` and `readmitted_30d` (0/1 labels).

### 4. Run the Streamlit app

```bash
streamlit run app.py
```



---

## Repository structure

```
readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb
├── src/
│   └── predict.py
├── DECISIONS.md
├── requirements.txt
└── README.md
```

---

## Limitations and honest assessment

- The MLP is still a relatively small network; more capacity might help but risks overfitting on 3,800 rows.
- Threshold tuning was done on a single validation split; a full cross-validated search could be more robust.
- No external clinical context or temporal leakage checks were available, so performance in production may differ.


---

## Local setup (quick)

```bash
# 1) Create and activate venv
python3 -m venv .venv
. .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run predictions
python src/predict.py --input data/test.csv --output predictions.csv

# 4) (Optional) Run Streamlit app
streamlit run app.py
```
