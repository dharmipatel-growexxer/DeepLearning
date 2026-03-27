# Decision log

This file documents three key decision points from your pipeline.
**Each entry is mandatory. Vague or generic answers will be penalised.**
The question to answer is not "what does this technique do" — it is "why did YOU make THIS choice given YOUR data."

---

## Decision 1: Data cleaning strategy
*Complete after Phase 1 (by approximately 1:30)*

**What I did:**
I normalized systolic blood pressure by converting values under 50 as kPa into mmHg, parsed mixed-format admission dates into year/month/day features, treated age > 120 as missing, capped numeric outliers at the 1st/99th percentiles, and imputed missing `glucose_level_mgdl` with the median. I also treated `admission_type` and `discharge_destination` as categorical codes rather than continuous numbers.

**Why I did it (not why the technique works — why THIS choice given what you observed in the data):**
The presence of values like 10.67 for systolic BP is not clinically plausible in mmHg, but it becomes plausible after kPa conversion. Dropping those rows would lose signal. Dates were mixed (`YYYY-MM-DD` and `DD/MM/YYYY`), so extracting components avoids parsing failures. Only glucose had missingness, so a simple median imputation keeps the distribution stable without introducing extra assumptions. Percentile capping reduces the influence of extreme values while preserving rank order.

**What I considered and rejected:**
I considered treating low BP values as invalid and imputing them, but that would discard meaningful measurements if they were recorded in kPa. I also considered dropping the date feature entirely due to format inconsistency, but it likely carries seasonality or operational patterns.

**What would happen if I was wrong here:**
If low BP values were not kPa, I would be inflating them and potentially biasing the model. If date components add noise, they could reduce generalization.

---

## Decision 2: Model architecture and handling class imbalance
*Complete after Phase 2 (by approximately 3:00)*

**What I did:**
I used a Keras MLP with two hidden layers (64 → 32, ReLU) plus dropout (0.30) and balanced the training data via random oversampling of the minority class.

**Why I did it (not why the technique works — why THIS choice given what you observed in the data):**
The dataset is small (3,800 rows) and has only ~9% positives. A small MLP adds nonlinear capacity without excessive complexity, and oversampling directly increases minority examples so the network sees enough positive cases during training. This avoids relying solely on loss weighting when the minority count is extremely low.

**What I considered and rejected:**
I considered a deeper network but it would likely overfit on this size. I also considered SMOTE or synthetic generation, but those can create unrealistic medical combinations; random oversampling keeps samples real.

**What would happen if I was wrong here:**
If oversampling causes overfitting to duplicated minority cases, model calibration and generalization could degrade.

---

## Decision 3: Evaluation metric and threshold selection
*Complete after Phase 3 (by approximately 4:00)*

**What I did:**
I reported PR-AUC and tuned the decision threshold to maximize F1 on the validation set, instead of using the default 0.5 cutoff.

**Why I did it (not why the technique works — why THIS choice given what you observed in the data):**
With a 9% positive rate, accuracy and a fixed 0.5 threshold would under-detect readmissions. PR-AUC reflects performance on the minority class, and F1 balances precision/recall for a practical operating point. The threshold tuning is lightweight and aligns the model to the task objective (catch readmissions without excessive false positives).

**What I considered and rejected:**
I considered optimizing accuracy or ROC-AUC alone. Accuracy is misleading in imbalanced data, and ROC-AUC can look good even when precision is poor. I also considered cost-sensitive thresholds but lacked explicit costs.

**What would happen if I was wrong here:**
If the chosen threshold overfits the validation split, performance on new data may degrade.

---

*Word count guidance: aim for 80–150 words per decision. More is not better — precision is.*
