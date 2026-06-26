"""Phase-A EDA + leakage audit -> reports/eda_report.md.

This is our own analysis (not the reference repo's): target balance, missingness, categorical
cardinality, the explicit leakage audit, and the temporal default-rate trend that motivates the
distribution-shift study.
"""
from __future__ import annotations

import os

import pandas as pd

from ..data import clean, schema

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "eda_report.md")


def _leakage_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Show, per leakage column, how strongly it reveals the label (the reason to drop it)."""
    y = clean.make_target(raw)
    mask = y.notna()
    y = y[mask]
    rows = []
    for col in schema.POST_OUTCOME_LEAKAGE + schema.POST_DECISION:
        s = raw.loc[mask, col]
        if col in schema.CURRENCY_COLS:
            val = clean.parse_currency(s)
            nonzero = (val.fillna(0) > 0)
            # P(default | value>0) vs base rate -> leakage signal
            p_def_given_pos = float(y[nonzero].mean()) if nonzero.any() else float("nan")
            rows.append(
                {
                    "column": col,
                    "kind": "post-outcome" if col in schema.POST_OUTCOME_LEAKAGE else "post-decision",
                    "signal": f"P(default | {col}>0) = {p_def_given_pos:.3f} vs base {y.mean():.3f}",
                }
            )
        else:
            present = s.notna() & (s.astype(str).str.strip() != "")
            p_def_given_present = float(y[present].mean()) if present.any() else float("nan")
            rows.append(
                {
                    "column": col,
                    "kind": "post-outcome" if col in schema.POST_OUTCOME_LEAKAGE else "post-decision",
                    "signal": f"P(default | {col} present) = {p_def_given_present:.3f} vs base {y.mean():.3f}",
                }
            )
    return pd.DataFrame(rows)


def run(raw_csv: str, nrows: int | None = None) -> str:
    raw = pd.read_csv(raw_csv, nrows=nrows, low_memory=False)
    y = clean.make_target(raw)
    n_total = len(raw)
    n_labeled = int(y.notna().sum())
    prevalence = float(y[y.notna()].mean())

    audit = _leakage_audit(raw)

    # temporal trend of default rate
    tmp = pd.DataFrame({"FY": pd.to_numeric(raw["ApprovalFY"], errors="coerce"), "y": y})
    tmp = tmp[tmp["y"].notna() & tmp["FY"].notna()]
    trend = tmp.groupby("FY")["y"].agg(["mean", "count"]).reset_index()

    # cardinality of engineered categoricals
    X = clean.engineer(raw.loc[y.notna()])
    card = {c: int(X[c].nunique(dropna=True)) for c in schema.CATEGORICAL_FEATURES}
    miss = {c: float(X[c].isna().mean()) for c in X.columns}

    lines = []
    lines.append("# SBA Credit-Trust — Phase A EDA & Leakage Audit\n")
    lines.append(f"- Rows (total): **{n_total:,}**")
    lines.append(f"- Rows with usable label (PIF/CHGOFF): **{n_labeled:,}**")
    lines.append(f"- Default (CHGOFF) prevalence: **{prevalence:.4f}** (~{prevalence*100:.1f}%)\n")

    lines.append("## Leakage audit (why these columns are dropped from features)\n")
    lines.append("| Column | Kind | Leakage signal |")
    lines.append("|---|---|---|")
    for _, r in audit.iterrows():
        lines.append(f"| `{r['column']}` | {r['kind']} | {r['signal']} |")
    lines.append("")

    lines.append("## Categorical cardinality (post-engineering)\n")
    lines.append("| Feature | # categories |")
    lines.append("|---|---|")
    for k, v in card.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## Missingness (feature frame)\n")
    lines.append("| Feature | % missing |")
    lines.append("|---|---|")
    for k, v in sorted(miss.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v*100:.2f}% |")
    lines.append("")

    lines.append("## Temporal default-rate trend (motivates the shift study)\n")
    lines.append("| ApprovalFY | default rate | n |")
    lines.append("|---|---|---|")
    for _, r in trend.iterrows():
        lines.append(f"| {int(r['FY'])} | {r['mean']:.3f} | {int(r['count']):,} |")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[eda] wrote {out}")
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv")
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(os.path.abspath(csv), nrows=nrows)
