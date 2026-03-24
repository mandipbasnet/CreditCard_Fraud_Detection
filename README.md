# Credit Card Fraud Detection

A practical machine learning project to detect fraudulent credit card transactions from Kaggle's anonymized dataset.

## The Problem

Banks process millions of transactions daily. Only 0.17% are frauds (about 1 in 580 transactions). If you build a naive model that predicts "everything is legitimate," you get 99.83% accuracy but catch **zero frauds**. That's useless.

The real challenge: **Catch frauds without annoying legitimate customers with false alarms.**

## Dataset

- **284,807 transactions** from European cardholders (Sept 2013)
- **492 frauds** (0.17%), 283,315 legitimate (99.83%)
- **Features**: V1-V28 (anonymized PCA components), Amount, Time, Class
- **Source**: Kaggle (European Credit Card Fraud Detection)

## The Approach

### Step 1: Understand the Imbalance
The dataset is **severely imbalanced**. Standard classifiers will just learn "always predict legitimate." We need to handle this upfront.

### Step 2: Prepare Data
- Drop `Time` (just a sequence, not predictive)
- Scale `Amount` (ranges 0-25,691) to match V1-V28 scaling (mean ≈ 0, std ≈ 1)
- Separate features (X) from target (y)

**Why scaling matters:** If Amount isn't scaled, the model thinks it's 100x more important than other features. We want all features to compete equally.

### Step 3: Train/Test Split (Stratified)
- 70% training (199,364 samples), 30% test (85,443 samples)
- **Stratified split** keeps fraud % the same in both sets (0.17% each)
- Random split might accidentally put all frauds in one set

### Step 4: Handle Imbalance with SMOTE
Training on imbalanced data = model learns nothing. Solution: **SMOTE** (Synthetic Minority Oversampling).

SMOTE creates **synthetic frauds** by interpolating between real fraud samples:
- Fraud A: [V1=2.1, V2=1.5]
- Fraud B: [V1=2.3, V2=1.7]
- **Synthetic**: [V1=2.2, V2=1.6] (between A & B)

Result: Training data goes from 344 frauds / 199,020 legitimate → **199,020 frauds / 199,020 legitimate (50/50)**.

**Critical:** Only apply SMOTE to training data, never test data. Test data is ground truth.

### Step 5: Train Logistic Regression (Baseline)
Quick, interpretable model to set a baseline.

**Results:**
```
Accuracy:  97.68%
Precision: 6.20%
Recall:    87.84%
F1-Score:  0.1158
ROC-AUC:   0.9676
```

**Analysis:**
- ✓ Catches 87.84% of frauds
- ✗ 93.8% of flagged transactions are false alarms (15 false alarms per fraud caught)
- ✓ ROC-AUC is great (0.97), but precision kills this model

**Verdict:** Good baseline, but not production-ready. Too many false alarms will annoy customers.

### Step 6: Train XGBoost (Production Model)
Gradient boosting with class weighting to handle imbalance natively.

**Results:**
```
Accuracy:  99.86%
Precision: 56.60%
Recall:    81.08%
F1-Score:  0.6667
ROC-AUC:   0.9688
```

**Comparison:**
| Metric | Logistic Reg | XGBoost | Improvement |
|--------|--------------|---------|------------|
| Precision | 6.20% | 56.60% | **9x better** |
| Recall | 87.84% | 81.08% | -6.7% (acceptable) |
| F1-Score | 0.1158 | 0.6667 | **5.7x better** |
| ROC-AUC | 0.9676 | 0.9688 | Same |

**Real-world impact:**
- **Logistic Reg:** For every fraud caught, 15 legitimate transactions flagged as suspicious
- **XGBoost:** For every fraud caught, <1 legitimate transaction flagged

XGBoost is **production-ready**. The 6.7% drop in recall is worth the 50% improvement in precision.

### Step 7: Feature Importance
Top 10 features driving fraud predictions:

```
V10       : 0.1523  (most important)
V4        : 0.1201
V14       : 0.1089
V12       : 0.0956
V8        : 0.0845
V17       : 0.0742
Amount    : 0.0689  (transaction size matters, but less than V10-V17)
V21       : 0.0612
V3        : 0.0534
V27       : 0.0498
```

**Note:** V features are anonymized (PCA components), so we can't interpret them as "merchant type" or "location." But we can rank them by importance.

## Key Learnings

### 1. Imbalance is the Real Problem
Accuracy is misleading on imbalanced data. A model predicting "always legitimate" gets 99.83% accuracy. Use **Precision, Recall, F1-Score, ROC-AUC** instead.

### 2. Precision vs Recall Trade-off
- **High Recall (87.84%):** Catch more frauds, more false alarms (annoyed customers)
- **High Precision (56.60%):** Few false alarms, some frauds slip through
- **Sweet spot:** 81% recall + 57% precision (XGBoost)

### 3. Scaling Matters
Amount ranges 0-25,691. V1-V28 range -60 to +3. Without scaling, Amount dominates. With scaling, all features compete fairly.

### 4. SMOTE Works, But Carefully
SMOTE balanced training data 50/50. But we tested on **real, imbalanced data** (0.17% fraud). This gap ensures honest evaluation.

### 5. Model Choice Matters
Logistic Regression: simple, baseline
XGBoost: **10x better precision for 6.7% lower recall**. Easy choice for production.

## How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib
```

### Step-by-step
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import xgboost as xgb

# 1. Load & prepare
df = pd.read_csv('creditcard.csv')
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

# 2. Scale Amount
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 5. Train XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 6. Evaluate
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
```

## Business Value

**Scenario:** Process 1 million transactions/day
- **No model:** Miss 1,000 frauds/day
- **Logistic Reg:** Catch 878 frauds, but flag 19,500 false alarms (customers hate it)
- **XGBoost:** Catch 811 frauds, flag ~1,440 false alarms (acceptable)

**Cost analysis:**
- Each fraud costs bank: $500 (chargeback + investigation)
- Each false alarm costs: $2 (customer support call)
- **XGBoost saves:** (811 × $500) - (1,440 × $2) = **$397,220/day**

## Things I'd Improve (Next Steps)

1. **Threshold tuning:** Currently using 0.5 (50% probability = fraud). Could adjust to 0.3 or 0.7 based on business cost of false positives vs false negatives.

2. **Feature engineering:** V features are anonymized, but could analyze their distributions by fraud/legitimate to find patterns.

3. **Hyperparameter tuning:** Used default XGBoost params. Could grid search `max_depth`, `learning_rate`, `n_estimators`.

4. **Ensemble:** Combine XGBoost + LightGBM + CatBoost for even better performance.

5. **Real-time deployment:** Put model behind REST API. Stream transactions through it. Log decisions for monitoring drift.

6. **Monitoring:** Track precision/recall over time. Fraudsters adapt. Retrain monthly or quarterly.

## Evaluation Metrics Explained

**Confusion Matrix:**
```
                 Predicted
           Legitimate  Fraud
Actual
Legitimate    TN         FP  (false alarms)
Fraud         FN         TP  (caught)
```

**Precision:** TP / (TP + FP)
- "Of the transactions we flagged as fraud, what % are actually fraud?"
- High precision = few false alarms
- **XGBoost: 56.6%** (when we flag, 56.6% are real fraud)

**Recall:** TP / (TP + FN)
- "Of all the frauds, what % did we catch?"
- High recall = catch most frauds
- **XGBoost: 81.08%** (catch 81% of all frauds)

**F1-Score:** Harmonic mean of precision & recall
- Balances both metrics (neither too low)
- Scale 0-1, higher is better
- **XGBoost: 0.6667** (good balance)

**ROC-AUC:** Area under the ROC curve
- Measures discrimination ability at all thresholds
- Scale 0-1, higher is better (0.5 = random guessing, 1.0 = perfect)
- **XGBoost: 0.9688** (near-perfect)

## Files

- `creditcard.csv` - Original dataset from Kaggle
- `fraud_detection.py` - Full pipeline (all 7 steps)
- `feature_importance.png` - Top 10 features chart
- `fraud_detection_results.png` - ROC curves, confusion matrix, precision-recall


**Q: Why does XGBoost have lower recall but I prefer it?**
A: Precision improved 9x (6.2% → 56.6%) at cost of 6.7% recall drop. In production, fewer false alarms > catching every single fraud. Banks will accept 81% recall if it means customers aren't annoyed by false flags.

## Takeaway

Don't just trust accuracy. Understand your business tradeoff:
- Catch more fraud → annoyed customers (high recall, low precision)
- Avoid false alarms → miss some fraud (low recall, high precision)

XGBoost nailed the balance for this problem.

---

**Built on:** Kaggle European Credit Card Fraud Detection dataset
**Tools:** Python, scikit-learn, XGBoost, SMOTE, pandas
**Model:** XGBoost (Gradient Boosting)
**Best Result:** 81% recall, 57% precision, 0.97 ROC-AUC
-----------------------------
**Aurthor** Mandip Basnet
**Batchlore In Computer Application** 
