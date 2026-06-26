# SBA Loan Default Prediction — Advanced Deep-Learning Project
## Master Plan & Handoff Document
*Prepared 2026-06-26 for Animesh Kumar. This is a PLAN ONLY — no code has been written. Carry this whole file into the new chat.*

---

## 0. HANDOFF CONTEXT (read this first in the new chat)

**Who:** Animesh Kumar — MSc Advanced Computer Science, Newcastle University (ML 92/100, DL 83/100). Research identity: deep learning, **calibration / uncertainty / clinical-safety methodology** (OCT imaging), coding theory & DNA storage (dissertation). Applying for research PhDs (DE/NL/SE). Established deployment stack: **PyTorch, ONNX, TFLite, Streamlit, Hugging Face Spaces, Zenodo preprints**. GPU: local RTX 4060 + Colab (A100). Python via Anaconda.

**What we are building:** An original, research-grade **deep-learning** project on the **public** U.S. SBA loan dataset (Kaggle: "Should this loan be approved or denied"), that:
1. Is genuinely **his own work** (the downloaded repo is reference-only — see Integrity Rules).
2. Is genuinely **deep learning** (the source repo is Gradient Boosting + Flask — classical ML, not DL).
3. Showcases his **signature trustworthy-ML methodology** (calibration, uncertainty, OOD, conformal risk control, explainability) now applied to **high-stakes credit decisions** — a coherent extension of his medical-imaging "clinical safety" work.
4. Is CV/PhD-worthy and ideally becomes a **4th preprint**.

**Why it exists:** Animesh holds a **deep-learning certificate whose capstone was SBA loan approval prediction**. This project is the genuine, advanced version that backs that certificate — replacing/augmenting the weaker "cloud" entry on his CV with a DL project that aligns with his ML focus.

### 0.1 INTEGRITY RULES (non-negotiable — same truth-rule used on his CV)
- The dataset is **public** → doing an original project on it is completely legitimate.
- The downloaded repo (`Final_Project_SBA_Loan_Approval-master`, **author: Bagja Satiaraharja**, github.com/bagjasatia) is **REFERENCE ONLY** — use it to understand the data/features and what's been done before. **Do NOT copy its code, models, or results.** Every line of code and every metric on the CV must be from Animesh's own runs.
- **No invented metrics.** Report real numbers from real runs. Accuracy alone is misleading on this imbalanced data — report PR-AUC, ROC-AUC, and minority-class recall/F1.
- Cite the source repo and any papers honestly.

---

## 1. THE DATASET (what we're working with)

- **File:** `SBAnational.csv` (~899,000 loans). Too big for GitHub — host on Google Drive / Kaggle, pull via script.
- **Target:** `MIS_Status` → binary: **Paid In Full (PIF)** vs **Charged Off (CHGOFF = default)**.
- **Imbalance:** ~17–18% default. This is the central modelling challenge.
- **Key features:** `Term`, `GrAppv` (gross approved), `SBA_Appv` (SBA-guaranteed amount), `NewExist` (new vs existing business), `RevLineCr`, `LowDoc`, `NAICS` (industry code → rich categorical), `ApprovalFY` (approval year → enables **temporal validation**), plus state, employees, jobs created, etc.
- **Two underused signals the source repo ignores:**
  - **Time** (`ApprovalFY`) → distribution shift across the 2007–2009 financial crisis.
  - **Money** (`GrAppv`, `SBA_Appv`, `ChgOffPrinGr`) → real, asymmetric **cost** of a wrong decision (a missed default costs far more than a rejected good loan) → enables **cost-sensitive / expected-loss** decisioning rather than plain accuracy.

### 1.1 Baseline to beat (from the source repo, honestly)
- Models used there: Logistic Regression, Random Forest, **Gradient Boosting** (deployed model = `gbc_SBA_Loan_oversampling`).
- Reported ~0.84–0.96 F1 **with oversampling** — but there is a **likely data-leakage flaw**: oversampling appears applied before the train/test split, which inflates the 0.96. **Our version fixes this** (resample inside CV folds only) and reports honest numbers. Re-establishing a *clean* GBM/XGBoost/LightGBM baseline is Step 1, and is itself a legitimate contribution ("the published demo over-reports due to leakage; here is the corrected baseline").

---

## 2. POSITIONING / NOVELTY (the "something new")

**Headline framing:** *"Trustworthy, calibrated, uncertainty-aware deep learning for small-business loan-default prediction, with conformal risk control and cost-sensitive decisioning."*

This mirrors his OCT clinical-safety work — the same thesis (high-stakes decisions need calibration + uncertainty + provenance, whether retina or credit) transferred to a new domain. That cross-domain transfer is the coherent, defensible novelty.

**Ranked novelty candidates (all honest, doable, genuinely under-explored on this dataset):**
1. **Conformal risk control / selective prediction** — give the model the option to **abstain** ("refer to a human") and guarantee a chosen ceiling on the default rate among auto-approved loans, with finite-sample coverage guarantees (split-conformal). Almost no public SBA project does this. *Strongest novelty + most practical.*
2. **Cross-domain methodology transfer** — port his clinical-safety stack (temperature-scaling calibration / ECE, Mahalanobis OOD, MC-Dropout) from medical imaging to credit risk. *Strongest narrative fit with his profile.*
3. **Cost-sensitive, expected-loss-optimal decisioning** — use the dataset's actual loan amounts to optimise an **expected-monetary-loss** policy, not accuracy. Report dollars saved vs the GBM baseline.
4. **Temporal distribution-shift study** — train pre-2006, evaluate through the 2008 crisis; measure how calibration and AUC degrade under shift, and whether DL + conformal degrades gracefully.
5. **Fairness/robustness slice analysis** — performance and calibration across NAICS industries and loan-size bands.

Pick **1 + 2 as the core**, with 3/4 as strong extensions. That combination is workshop-paper / strong-portfolio grade.

> **Honesty caveat to state up front (and in the writeup):** On tabular data, **well-tuned tree ensembles often match or beat deep nets** (Grinsztajn et al. 2022). So the contribution must NOT be "DL beats GBM on accuracy." It is **trustworthiness**: calibration, uncertainty, selective-risk guarantees, and cost-optimal decisions — where the DL + conformal + uncertainty stack adds value a raw GBM does not. Frame it that way and it's both novel and defensible at a viva.

---

## 3. TECHNICAL ROADMAP (phased)

### Phase A — Foundations & honest baselines
- Reproducible data pipeline: download → schema validation → cleaning → leakage-safe split (stratified + a **temporal** split variant).
- EDA (his own, not the repo's): target balance, missingness, leakage audit (e.g., `ChgOffDate`/`ChgOffPrinGr` are **post-outcome** → must be dropped from features), categorical cardinality (NAICS).
- Clean baselines **with correct pipeline order** (resample/encode inside CV only): Logistic Regression, **XGBoost / LightGBM / CatBoost**. Record PR-AUC, ROC-AUC, F1(minority), Brier, ECE. *This is the number to beat and the leakage-corrected reference.*

### Phase B — Tabular deep learning
- **MLP with learned embeddings** for high-cardinality categoricals (NAICS, state) — the proper DL baseline.
- **TabNet** (Arik & Pfister 2021) — attentive, interpretable feature selection.
- **FT-Transformer** (Gorishniy et al. 2021) — transformer for tabular; usually the strongest DL option.
- (Optional) **SAINT** (Somepalli et al. 2021), **NODE**, or **TabTransformer**.
- Imbalance handling done **right**: class weights, **focal loss**, threshold moving, SMOTE-NC *inside folds only* (and a critique of naive oversampling — the leakage point above).

### Phase C — Trustworthy-ML layer (his signature, the differentiator)
- **Calibration:** temperature scaling + isotonic; report **ECE** and **Brier** before/after (his hallmark metric).
- **Uncertainty:** MC-Dropout + **deep ensembles**; predictive entropy.
- **OOD detection:** **Mahalanobis** distance in the penultimate layer (his exact OCT method) → flag out-of-distribution loan applications.
- **Conformal prediction:** split-conformal for **risk-controlled selective classification** (Angelopoulos & Bates 2021 tutorial) → guaranteed coverage / bounded default rate among auto-decisions, with an abstention option.
- **Explainability:** SHAP (global + local), TabNet/FT-Transformer attention masks, a few counterfactuals.

### Phase D — Novelty experiments
- Cost-sensitive expected-loss decision policy (Section 2.3) with $-saved reporting.
- Temporal shift experiment (Section 2.4): calibration & AUC across years; does conformal coverage hold under shift?
- Slice/fairness analysis across NAICS and loan-size bands.

### Phase E — Deployment & reproducibility (maxed-out engineering)
- **Streamlit app** (his stack) — input a loan, get probability **+ calibrated confidence + abstain/refer flag + SHAP explanation**. Big upgrade over the source's bare yes/no.
- **ONNX export** + latency benchmark (matches his CV pattern).
- **Hugging Face Space + model card**; **Zenodo DOI** for the preprint.

### Phase F — Writeup
- Short **preprint** (his pattern): "Calibrated, selective deep learning for high-stakes small-business credit decisions." Target arXiv cs.LG / a fintech-ML or trustworthy-ML workshop. This becomes his **4th** research output.

---

## 4. ENGINEERING STANDARDS ("maxed-out level of coding")

- **Repo layout:** `src/` modular package (`data/`, `features/`, `models/`, `calibration/`, `conformal/`, `eval/`, `app/`), `configs/` (Hydra/YAML), `tests/` (pytest, ≥ the data + pipeline + metric paths), `notebooks/` (EDA only), `models/` (artifacts, git-ignored), `reports/`.
- **Reproducibility:** fixed seeds, `requirements.txt`/`environment.yml`, **DVC** or a data-pull script for `SBAnational.csv`, deterministic splits saved to disk.
- **Experiment tracking:** **MLflow or Weights & Biases** — every run logged (params, metrics, calibration curves, conformal coverage).
- **Quality gates:** type hints, `ruff`/`black`, `pre-commit`, GitHub Actions CI (lint + tests), a `Dockerfile`, and a **model card** documenting data, metrics, limitations, and intended use.
- **TDD discipline** (matches his FRIDAY habit): write the failing test for each metric/pipeline step before implementing.

---

## 5. PAPERS TO READ (grounding — VERIFY each citation before quoting)

> These are real, well-known works, but the next chat / Animesh should confirm exact authors/years/venues before citing in a preprint.

1. **Gorishniy, Rubachev, Khrulkov, Babenko — "Revisiting Deep Learning Models for Tabular Data"** (NeurIPS 2021). FT-Transformer + honest tabular DL baselines.
2. **Arik & Pfister — "TabNet: Attentive Interpretable Tabular Learning"** (AAAI 2021).
3. **Somepalli et al. — "SAINT"** (2021). Self-attention + contrastive for tabular.
4. **Grinsztajn, Oyallon, Varoquaux — "Why do tree-based models still outperform deep learning on tabular data?"** (NeurIPS 2022). The honesty anchor — read FIRST.
5. **Guo, Pleiss, Sun, Weinberger — "On Calibration of Modern Neural Networks"** (ICML 2017). Temperature scaling, ECE.
6. **Angelopoulos & Bates — "A Gentle Introduction to Conformal Prediction…"** (2021). The conformal/selective-risk toolkit.
7. **Lakshminarayanan, Pritzel, Blundell — "Deep Ensembles"** (NeurIPS 2017). Uncertainty.
8. **Gal & Ghahramani — "Dropout as Bayesian Approximation" (MC-Dropout)** (ICML 2016).
9. **Lee et al. — "A Simple Unified Framework for Detecting OOD… (Mahalanobis)"** (NeurIPS 2018). His OCT OOD method.
10. **Lin et al. — "Focal Loss"** (ICCV 2017). Imbalance.
11. (Domain) A recent **deep learning for credit scoring / SBA loan** survey or paper — search and add one or two for related-work framing.

---

## 6. ACCEPTANCE CRITERIA ("definition of done", maxed out)

- [ ] Leakage-corrected, properly-validated **GBM baseline** reported (and the source repo's over-reporting explained).
- [ ] At least **2 tabular DL models** (MLP+embeddings, FT-Transformer) trained, beating or matching the GBM baseline on PR-AUC.
- [ ] **Calibration**: ECE + Brier reported and **improved** post temperature/isotonic scaling.
- [ ] **Uncertainty**: MC-Dropout / ensemble predictive uncertainty quantified.
- [ ] **Conformal selective classification**: demonstrated coverage guarantee + abstention; show the bounded default rate among auto-approved loans.
- [ ] **Cost-sensitive policy**: expected-loss (\$) improvement vs baseline quantified.
- [ ] **Temporal shift** experiment run.
- [ ] **SHAP** + model interpretability delivered.
- [ ] **Deployed**: Streamlit app + ONNX + HF Space + model card.
- [ ] **Reproducible**: tests pass, seeds fixed, configs + tracking in place.
- [ ] **Preprint draft** written with honest claims and the tree-vs-DL caveat.

---

## 7. SUGGESTED BETTER PLAN (my recommendation)

Don't treat this as "a CV project." Treat it as a **mini research preprint** in the exact mold of his OCT papers — that is the highest-leverage use of the same effort:

> **"Calibrated, Selective Deep Learning for High-Stakes Small-Business Credit Decisions."**
> Contribution = not "DL beats GBM," but a **trustworthy decision system**: calibrated probabilities, quantified uncertainty, conformal risk guarantees, and cost-optimal (dollar-aware) decisions with an abstention option — plus a corrected, leakage-free benchmark on a widely-mis-modelled public dataset.

This (a) is genuinely novel-enough, (b) plays to his proven strengths (calibration/uncertainty/deployment), (c) yields a **4th preprint + Zenodo DOI + HF demo** for the PhD portfolio, and (d) is fully honest. The plain "improve the Flask app" version is a waste of his ceiling.

**CV-placement advice:** add this as a **Research Project** (backing the DL certificate). Do **not** delete the real IIT Kanpur internship to make room — they're different categories (experience vs project). On the 2-page CV, this can replace the *plant-disease* project if space is tight, since plant disease is the least aligned with his coding/medical research narrative.

---

## 8. READY-TO-PASTE PROMPT FOR THE NEW CHAT

> I'm building an original, research-grade **deep-learning** project on the **public** SBA loan dataset (Kaggle "Should this loan be approved or denied"). I have a downloaded reference repo (`Final_Project_SBA_Loan_Approval` by Bagja Satiaraharja) — **reference only, do not copy its code/results**; it's Gradient Boosting + Flask and over-reports due to oversampling-before-split leakage. My profile: MSc Adv CS (Newcastle), strong in PyTorch + calibration/uncertainty/OOD methodology (from OCT medical-imaging work), I deploy via Streamlit/ONNX/HuggingFace. GPU: RTX 4060 + Colab.
>
> **Goal:** a trustworthy, calibrated, uncertainty-aware DL credit-risk model with **conformal selective prediction** and **cost-sensitive decisioning**, leakage-free baselines, full reproducibility, deployment, and a short honest preprint. Tree models may match DL on accuracy — the contribution is **trustworthiness + selective risk**, not raw accuracy.
>
> I have a full master plan (this document — `SBA_Loan_DL_MasterPlan_and_Handoff.md`). **Start with Phase A:** set up the leakage-safe data pipeline + EDA + corrected GBM/XGBoost baseline (report PR-AUC, ROC-AUC, F1-minority, Brier, ECE), and confirm the exact target/leakage-prone columns to drop. Enforce: every metric from my own runs, no invented numbers, honest claims. Walk me through it step by step.

---

*End of plan. Nothing here has been executed — it is a blueprint for the next session.*
