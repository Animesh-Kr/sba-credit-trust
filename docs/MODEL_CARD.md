# Model Card — SBA Credit-Trust (draft, Phase A)

> Living document. Metrics are filled in from real runs as each phase lands. No invented numbers.

## Model details
- **Task:** Binary prediction of small-business loan **default** (SBA 7(a)/504 program loans).
- **Positive/event class:** `default` (MIS_Status = CHGOFF). Prevalence ≈ **17.56%**.
- **Models:** Phase A — Logistic Regression (class-weighted), XGBoost (scale_pos_weight).
  Phase B — MLP+embeddings and FT-Transformer (focal loss α=0.75 γ=2). Trustworthy-ML layer in Phase C.
- **Inputs:** approval-time features only — loan terms/amounts (`Term`, `GrAppv`, `SBA_Appv`,
  `sba_portion`), business attributes (`NoEmp`, `NewExist`, `is_franchise`, `naics_sector`),
  geography (`State`, `BankState`, `UrbanRural`), program flags (`RevLineCr`, `LowDoc`,
  `real_estate`), and `ApprovalFY`.

## Intended use & out-of-scope
- **Intended:** research/education on trustworthy credit-risk ML; a decision-*support* tool with
  calibrated probabilities, uncertainty, and an abstention option.
- **Out of scope:** real lending decisions. Not fair-lending audited; trained on 1987–2014 U.S. SBA
  loans and will not transfer to other portfolios/eras without recalibration.

## Data
- Public Kaggle dataset `mirbektoktogaraev/should-this-loan-be-approved-or-denied`
  (`SBAnational.csv`, 899,164 rows). 1,997 unlabeled rows dropped.
- **Leakage handling:** `ChgOffDate`, `ChgOffPrinGr`, `BalanceGross` are post-outcome and excluded
  (presence ⇒ ~97% default). `DisbursementGross`/`DisbursementDate` are post-decision and excluded
  from the primary feature set (configurable).

## Metrics — Phase A baselines (stratified hold-out, test n=179,434; from `reports/baseline_metrics.md`)

| Model | PR-AUC | ROC-AUC | F1(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.563 | 0.846 | 0.534 | 0.821 | 0.166 | 0.216 |
| XGBoost | **0.914** | **0.978** | 0.820 | 0.933 | 0.053 | 0.074 |

- Accuracy is intentionally **not** a headline metric (uninformative at 18% prevalence).
- **Leakage-corrected benchmark:** reproducing the reference demo's oversample-*before*-split flaw
  inflates XGBoost PR-AUC to 0.978 — a **+0.064** over-report from pipeline order alone.
- **Honesty caveat:** the strong random-split XGBoost numbers partly reflect `ApprovalFY` letting the
  model exploit each cohort's default-rate trend (e.g. the 2007 spike). A deployed model cannot see
  future cohorts, so the **temporal split (Phase D)** is the honest deployment estimate — expect a
  drop. ECE 0.074 (uncalibrated) motivates the Phase-C calibration layer.

### Phase B — Tabular DL (same hold-out; from `reports/deep_metrics.md`)

| Model | PR-AUC | ROC-AUC | F1(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|
| FT-Transformer | 0.901 | 0.973 | 0.818 | 0.878 | 0.070 | 0.145 |
| MLP + embeddings | 0.867 | 0.958 | 0.791 | 0.837 | 0.093 | 0.185 |

- DL does **not** beat XGBoost on tabular accuracy (consistent with Grinsztajn et al. 2022); FT-Transformer
  is closest (0.901 vs 0.914). DL is also **worse calibrated** (ECE 0.145/0.185 vs 0.074) — focal loss
  distorts probabilities. Both facts motivate the Phase-C trust layer rather than undermining it.

### Phase C — Trustworthy-ML layer (three-way split; from `reports/trust_metrics.md`)

- **Calibration (MLP):** ECE 0.193 → 0.052 (temperature) → **0.0031** (isotonic); Brier 0.099 → 0.054.
  Ranking unchanged (PR-AUC ≈ 0.857). XGBoost already near-calibrated (ECE 0.074 → 0.069).
- **Uncertainty:** MC-Dropout entropy higher on errors (0.66 vs 0.53 nats). Selective accuracy
  **0.988** (confident half) vs **0.848** (uncertain half). Deep ensemble PR-AUC 0.861.
- **OOD:** Mahalanobis AUROC **0.796** vs feature-permuted OOD (synthetic; see AUDIT M-4).
- **Conformal selective prediction:** at α=0.05, coverage 90.6%, realized default among auto-approved
  **5.0%** (risk control holds; formal caveat in AUDIT M-1); at α=0.10, coverage 96.7%, abstain 3.3%,
  default among approved 7.3%. *(Numbers vary ±0.01 across reruns — CUDA nondeterminism, AUDIT R-1.)*
- **SHAP (XGBoost):** top drivers `Term`, `ApprovalFY`, `sba_portion`, `SBA_Appv` — the time+money
  signals the reference repo dropped. Many `BankState` one-hots → geographic signal; see Phase-D
  fairness audit.

### Phase D — Novelty experiments (from `reports/novelty_metrics.md`)

- **Temporal shift (train ≤2006, test >2006):** ranking holds (PR-AUC 0.916) but calibration degrades
  (ECE 0.069→0.105, Brier ~2×); per-year PR-AUC falls 0.95 (2007) → 0.23 (2013). **The conformal
  guarantee breaks under shift** — default-among-approved 0.049 in-era → 0.106 future. This is the
  honest deployment estimate; exchangeability is violated by temporal data.
- **Cost-sensitive decisioning (empirical LGD ≈ 64% of gross):** EV-optimal cut-off `t* = m/(m+LGD)`.
  At a 3% margin the dollar-aware policy beats the 0.5 threshold on net $ (+$2.4M) and approves 87%
  fewer defaulters; at higher margins it trades profit for large reductions in approved defaults.
- **Slice/fairness:** PR-AUC 0.79 (large-loan quartile) – 0.94 (mid); ECE 0.06–0.09 across NAICS
  sectors. Large loans are the hardest slice; geographic (`BankState`) signal warrants fair-lending
  scrutiny before any real use.

### Post-audit enhancements (from the respective reports)

- **Metric stability (E-2):** XGBoost PR-AUC **0.9132 ± 0.0022** over 5 seeds; bootstrap 95% CI
  [0.911, 0.917]. Headline numbers are stable.
- **Tuning (E-3):** Optuna-tuned XGBoost PR-AUC **0.927** (ECE 0.058) — trees remain ≥ DL after tuning.
- **Best-model trust stack (E-8):** FT-Transformer + isotonic gives ECE **0.002**; conformal holds
  (α=0.05 → 4.97% default among auto-approved).
- **Adaptive conformal under shift (E-9):** static conformal's long-run default among approved drifts
  to **0.111** on post-2006 cohorts; Adaptive Conformal Inference restores it to **0.047** (target 0.05).
- **Fairness (E-10):** equal-opportunity gaps 0.02–0.07, within-group ECE 0.01–0.05; raw approval-rate
  gaps 0.18–0.20 reflect real base-rate differences and warrant mitigation before real use.
- **OOD honesty (E-4):** Mahalanobis flags *synthetic* permutation OOD (AUROC 0.81) but **fails on a
  real held-out industry (AUROC 0.47)** — synthetic OOD over-states performance; real semantic OOD
  detection is unsolved here.

## Ethical considerations & limitations
- Tree ensembles often match DL on tabular accuracy; the project's value is **trustworthiness**
  (calibration, uncertainty, conformal selective risk, cost-awareness), not raw accuracy.
- Geographic/industry features risk encoding socioeconomic bias → slice/fairness analysis in Phase D.

## Deployment (Phase E)

- **Served model:** Optuna-tuned XGBoost + temperature calibration (T≈0.96) + conformal
  approve/reject thresholds, persisted as a single `ServingBundle` (`models/serving_bundle.joblib`).
- **Validated served behaviour (held-out test):** at α=0.05, auto-approves 85.6% of loans with a
  realised **4.9%** default rate among approved; at α=0.10, 4.8% approved-rate threshold holds too.
- **Streamlit app** (`src/app/streamlit_app.py`): per-loan calibrated probability + auto-approve /
  auto-reject / **refer-to-human** decision + SHAP explanation.
- **ONNX:** the gradient-boosted model is exported to ONNX (preprocessing kept in the serving layer);
  parity vs sklearn max|Δ| ≈ 6e-7; single-row latency 2.26 ms → **0.04 ms** (≈56× faster).
- **Hugging Face Space** scaffold in `hf_space/` (streamlit SDK; no GPU/torch needed for serving).

## Provenance
- Reference repo (Bagja Satiaraharja) used only to understand features/prior work; no code/results
  copied. All metrics from this repo's own runs.
