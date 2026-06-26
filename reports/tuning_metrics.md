# SBA Credit-Trust — E-3 XGBoost Hyperparameter Tuning (Optuna)

Addresses AUDIT M-3. 30 TPE trials. Best val PR-AUC = 0.9213. Test n = 179,434. Runtime 279.7s.

| Model | PR-AUC | ROC-AUC | F1(def) | Brier | ECE |
|---|---|---|---|---|---|
| XGBoost (default) | 0.9139 | 0.9785 | 0.8200 | 0.0533 | 0.0741 |
| **XGBoost (tuned)** | 0.9265 | 0.9816 | 0.8444 | 0.0458 | 0.0583 |

**Best params:** `{'n_estimators': 700, 'max_depth': 8, 'learning_rate': 0.05116500998797686, 'subsample': 0.8249502727149073, 'colsample_bytree': 0.8124819686258051, 'min_child_weight': 4, 'reg_lambda': 0.054683267586723545}`

> Tuning Δ PR-AUC = **+0.0126**. The tuned XGBoost is the fair tree-side baseline for the tree-vs-DL comparison; FT-Transformer (PR-AUC ≈ 0.90) remains below it.
