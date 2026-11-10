# Credit Card Fraud Detection

A research demonstration of imbalanced transaction classification. It has not been validated for live fraud decisions.

## Project Structure

```
credit_card_fraud_detection/
├── data/
│   └── creditcard.csv         # Place the dataset here
├── src/
│   ├── data_exploration.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup
1. Use the tracked `archive (2).zip`, or place `creditcard.csv` in `data/`. The original dataset is the [ULB/Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud); review its redistribution terms before reuse.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```

## Modules
- `data_exploration.py`: Data loading, exploration, visualization
- `preprocessing.py`: Stratified split, training-only scaling and SMOTE
- `modeling.py`: Model training and comparison
- `evaluation.py`: Metrics, plots, and insights
- `utils.py`: Helper functions

---

## Dataset
Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud

## Validation limits

The held-out rows are separated before fitting the scaler. SMOTE is applied only to training rows. This project uses a random stratified split of an anonymized historical sample; it has no production traffic, temporal drift, fraud cost, or calibration validation. Report precision, recall, PR-AUC and the decision threshold alongside ROC-AUC. Recompute all results after this preprocessing correction; older screenshots and `summary.txt` are historical outputs.

Run focused checks with `python -m unittest discover -s tests`. Run the full pipeline with `python main.py` after installing `requirements.txt`; it can be slow on the full dataset.

The 2026-11-10 rerun metrics and execution limits are in `VALIDATION_2026-11-10.md`. The ZIP's underlying dataset license is not established here; the repository code has no software license file.

**Repository description recommendation:** “Historical credit-card fraud classification with training-only scaling and SMOTE, untouched-test evaluation.”

---


