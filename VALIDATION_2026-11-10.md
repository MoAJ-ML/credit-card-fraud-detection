# Fraud validation rerun — 10 November 2026

**PASS:** `../../.venv/bin/python -m unittest discover -s tests -v` (1 focused preprocessing test).

**PASS:** `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/plx_mpl_cache OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../../.venv/bin/python main.py` using the tracked ZIP, random stratified 80/20 split, scaler fitted only on training rows, and SMOTE applied only to the training partition. The untouched test has 56,962 transactions, including 98 labeled fraud. Decision threshold is 0.5. Full machine-readable metrics are in the locally generated `validated_metrics.json` (ignored by Git).

| Model | Precision | Recall | Average precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression, unbalanced | 0.829 | 0.643 | 0.737 | 0.955 |
| Logistic regression, SMOTE | 0.054 | 0.918 | 0.732 | 0.973 |
| Random forest, unbalanced | 0.963 | 0.806 | 0.867 | 0.953 |
| Random forest, SMOTE | 0.863 | 0.837 | 0.869 | 0.982 |

**NOT TESTED:** XGBoost. The installed wheel could not load `libomp.dylib` on this Mac. No cross-validation tuning, temporal drift, calibration, cost-sensitive threshold selection or production fraud performance was tested. The preceding metrics describe this dataset and random holdout only.
