# SBA Credit-Trust — Adversarial Audit (Phases A–D)

*Self-audit before deployment. Adversarial intent: assume every claim is wrong until the code
proves it. Severity: **C**ritical (invalidates a result) · **M**ethodological (weakens rigor) ·
**B**ug · **R**eproducibility · **N**ice-to-have. Status as of the A–D re-audit.*

## Summary

The project's **core integrity claim — no feature leakage — holds.** Every feature is derived from
approval-time columns; the three post-outcome columns (`ChgOffDate`, `ChgOffPrinGr`, `BalanceGross`)
are excluded from `X` and only `ChgOffPrinGr` is reused, for ex-post dollar accounting and a
train-only LGD estimate (both legitimate). Encoders/calibrators/conformal thresholds are all fit on
data disjoint from the test set. No critical leakage was found.

Four real issues were found and **fixed** in this pass; the rest are honest limitations to disclose
in the preprint or to add as enhancements.

---

## Fixed in this pass

| ID | Sev | Finding | Fix |
|---|---|---|---|
| F-1 | B | `SelectiveRiskController.decide` applied approve then reject; when thresholds crossed (`t_lo ≥ t_hi`, seen at α=0.10) loans in the overlap were silently forced to *reject*. | Approve/reject made mutually exclusive; overlap now **abstains** (ambiguous → refer). |
| F-2 | C | Phase-D "in-era" conformal coverage was **resubstitution** — evaluated on the very slice its threshold was fit on, so the in-era ✅ was optimistically biased. | Calibration era split into a fit half and a **held-out** eval half; in-era number now honest (0.049, still ✅). |
| F-3 | B | ECE dropped predictions of exactly `0.0` (first bin was half-open `(lo,hi]`); isotonic can emit 0, so a few samples were silently excluded. | First bin now closed on the left `[0, hi]`. |
| F-4 | M | Conformal results were described as a "finite-sample guarantee" without stating the conditions. | Wording softened to "risk control (empirically validated)" with a pointer to M-1; caveat added to the module docstring. |

---

## Open — methodological (disclose in preprint)

**M-1 — Conformal guarantee is empirical, not formally airtight.**
The approve/reject threshold is chosen by scanning all candidate cut-offs and keeping the largest
whose Clopper–Pearson upper bound ≤ α. Two gaps vs a textbook RCPS/Learn-then-Test guarantee:
(a) the per-threshold bound uses `delta` with **no multiplicity correction** for the many thresholds
tried; (b) the risk among the selected set is **not strictly monotone** in the threshold, which
RCPS assumes. The realised risk holds in every experiment (5.1% at α=0.05), but the *formal* claim
needs a monotonised risk + a single calibrated cut-off (Bates et al. 2021; Angelopoulos et al. 2021).
→ **Backlog E-1.**

**M-2 — Single split, no confidence intervals.** Every headline metric is a point estimate from one
train/test split and one seed. No bootstrap CIs, no repeated-seed mean±std. Differences like
FT-Transformer 0.901 vs XGBoost 0.914 are reported without uncertainty. → **Backlog E-2.**

**M-3 — No hyperparameter tuning.** XGBoost and both DL models use fixed, hand-set hyperparameters.
The headline "DL does not beat XGBoost" is fair directionally but not a tuned-vs-tuned comparison;
state this explicitly or run a modest search. → **Backlog E-3.**

**M-4 — Synthetic OOD only → RESOLVED with a cautionary finding (E-4).** The Phase-C Mahalanobis
AUROC of 0.81 was against *feature-permuted* inputs. The real leave-one-sector-out benchmark
(`ood_metrics.md`) shows the detector **fails on genuine semantic shift**: a held-out industry scores
AUROC **0.47** (≈ chance), because finance-sector loans are not actually distant in the penultimate
feature space. Lesson: synthetic permutation OOD massively over-states OOD performance; the model
card and preprint now report the real (failing) number. Better OOD (sector-embedding distance,
density models) is future work.

**M-5 — Profit model is a flat margin.** The dollar policy credits `m·GrAppv` on every paid loan with
no time-discounting or term dependence, and a single portfolio-wide LGD (64%). Honest as a
first-order model; a per-loan/term-structured LGD would be stronger. Assumptions are stated in-report.

**M-6 — Threshold-dependent metrics at 0.5.** F1/precision/recall use a fixed 0.5 cut-off, which is
not meaningful for the `scale_pos_weight`/focal-loss models (their raw scores are not centred at 0.5).
PR-AUC (threshold-free) is the headline and is unaffected; consider reporting F1 at a tuned operating
point instead. → minor.

---

## Open — reproducibility

**R-1 — CUDA nondeterminism.** Seeds are set for Python/NumPy/Torch, but cuDNN nondeterministic
kernels are not disabled and `torch.use_deterministic_algorithms` is off, so DL runs are not
bit-reproducible. Document the seed and report tolerance, or pin determinism. → **Backlog E-5.**

**R-2 — No dataset checksum / versioning.** The Kaggle dataset is pulled by name; if the upstream
file changes, results drift silently. Add a SHA-256 of `SBAnational.csv` and assert it. → **Backlog E-6.**

**R-3 — No CI / lint gate enforced.** `pyproject` configures ruff/black/pytest but there is no
GitHub Actions workflow running them. → **Backlog E-7.**

---

## Enhancement backlog (the "what extra can be added")

Ranked by value for a portfolio/preprint:

1. **E-2 — Repeated-seed CIs.** Run each headline experiment over ≥5 seeds; report mean±std and
   bootstrap CIs on test metrics. *Single highest-credibility add.*
2. **E-1 — Rigorous conformal.** Replace the scan with a proper RCPS / split-conformal selective
   classifier (monotonised risk, single calibrated λ, union-bounded δ). Turns M-1 from caveat into
   a real theorem.
3. **E-8 — Calibrated DL + abstention end-to-end.** Combine isotonic-calibrated FT-Transformer with
   the conformal selector (currently conformal runs on the MLP) so the *best* model carries the
   guarantee.
4. **E-3 — Light hyperparameter search** (Optuna) for XGBoost and FT-Transformer, so the tree-vs-DL
   claim is tuned-vs-tuned.
5. **E-9 — Adaptive conformal under shift (ACI).** Since the guarantee breaks temporally (Phase D),
   add Gibbs–Candès adaptive conformal inference and show it restores coverage on future cohorts —
   this would be a genuine novelty beat, directly motivated by our own negative result.
6. **E-4 — Real OOD benchmark** (leave-one-sector-out; pre/post-crisis as OOD).
7. **E-10 — Fairness deep-dive.** Demographic-parity / equalised-odds style gaps across states and
   loan-size bands, plus calibration-within-group (we have ECE per slice; add disparity metrics).
8. **E-6 / E-5 / E-7 — Repro hardening:** dataset checksum, deterministic flags, CI workflow.
9. **E-11 — SHAP for the DL model** (DeepExplainer/GradientExplainer) to complement the XGBoost SHAP.
10. **E-12 — Recovery-aware cost** (LGD as a function of term/sector) and a full cost-vs-coverage
    operating curve.

---

## Enhancement status (implemented 2026-06-26)

| ID | Status | Result |
|---|---|---|
| E-2 repeated-seed CIs | ✅ done | PR-AUC 0.9132 ± 0.0022; bootstrap 95% CI [0.911, 0.917] (`robustness_metrics.md`) |
| E-1 rigorous conformal | ✅ done | Bonferroni union-bounded threshold (`fit_rigorous`); reported in `adaptive_conformal_metrics.md` |
| E-9 adaptive conformal | ✅ done | ACI restores long-run risk control under shift: static 0.111 → **adaptive 0.047** (target 0.05) |
| E-3 Optuna tuning | ✅ done | Tuned XGBoost PR-AUC 0.914 → **0.927**; trees still ≥ DL (`tuning_metrics.md`) |
| E-10 fairness disparity | ✅ done | Equal-opportunity gaps 0.02–0.07; within-group ECE 0.01–0.05 (`fairness_metrics.md`) |
| E-6 dataset checksum | ✅ done | Pinned SHA-256, verified on load (`download.py`) |
| E-5 determinism | ✅ done | `set_determinism()` (cuDNN deterministic); residual ±0.01 documented |
| E-7 CI workflow | ✅ done | `.github/workflows/ci.yml` (ruff + pytest) |
| E-8 best-model conformal | ✅ done | FT-Transformer isotonic ECE 0.159→**0.002**; conformal holds (α=0.05, 4.97% default@approved) (`best_model_conformal.md`) |
| E-4 real OOD | ✅ done | **Negative finding** — held-out sector AUROC **0.47** (chance) vs synthetic 0.81; confirms M-4 (`ood_metrics.md`) |

Remaining backlog (post-deployment / preprint polish): E-11 SHAP for the DL model, E-12 recovery-aware
cost curve, deeper fairness mitigation.

## Verdict

Methodology is sound and the integrity story is clean. The headline upgrade — **E-9 adaptive
conformal turning the Phase-D negative result into long-run risk control under shift** — is done and
validated. None of the enhancements blocked deployment; they materially sharpen the eventual writeup.
