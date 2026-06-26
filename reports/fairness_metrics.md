# SBA Credit-Trust — E-10 Fairness / Disparity Audit

Temperature-calibrated XGBoost, random-split test. Approve iff P(default) < 0.5. Runtime 38.5s.

Favourable outcome = approved. `*_gap` = max−min across groups (0 = perfectly fair).

## By State

| Group | n | base default | approval rate | approval rate (good) | default-catch | ECE |
|---|---|---|---|---|---|---|
| CA | 26,032 | 0.184 | 0.783 | 0.950 | 0.953 | 0.050 |
| TX | 14,056 | 0.188 | 0.750 | 0.907 | 0.929 | 0.079 |
| NY | 11,620 | 0.196 | 0.757 | 0.928 | 0.941 | 0.071 |
| FL | 8,302 | 0.275 | 0.681 | 0.926 | 0.965 | 0.057 |
| PA | 6,975 | 0.150 | 0.791 | 0.912 | 0.893 | 0.083 |
| OH | 6,582 | 0.160 | 0.784 | 0.920 | 0.927 | 0.081 |
| IL | 5,954 | 0.227 | 0.726 | 0.922 | 0.944 | 0.067 |
| MA | 4,968 | 0.124 | 0.834 | 0.946 | 0.951 | 0.066 |
| NJ | 4,817 | 0.206 | 0.723 | 0.895 | 0.941 | 0.088 |
| MN | 4,811 | 0.123 | 0.837 | 0.940 | 0.895 | 0.065 |
| WA | 4,695 | 0.145 | 0.812 | 0.941 | 0.944 | 0.058 |
| GA | 4,397 | 0.242 | 0.713 | 0.922 | 0.944 | 0.058 |

**Disparity gaps** — approval 0.182, equal-opportunity (approval|good) 0.072, default-catch 0.096, ECE 0.051.

## By Loan-size band

| Group | n | base default | approval rate | approval rate (good) | default-catch | ECE |
|---|---|---|---|---|---|---|
| Q2 | 45,988 | 0.205 | 0.754 | 0.935 | 0.947 | 0.062 |
| Q4>$225,000 | 45,480 | 0.095 | 0.856 | 0.930 | 0.852 | 0.071 |
| Q3 | 45,125 | 0.133 | 0.817 | 0.927 | 0.896 | 0.072 |
| Q1<=$35,000 | 42,841 | 0.274 | 0.674 | 0.918 | 0.972 | 0.070 |

**Disparity gaps** — approval 0.182, equal-opportunity (approval|good) 0.017, default-catch 0.120, ECE 0.010.

## By NAICS sector

| Group | n | base default | approval rate | approval rate (good) | default-catch | ECE |
|---|---|---|---|---|---|---|
| 44 | 16,899 | 0.227 | 0.710 | 0.905 | 0.953 | 0.077 |
| 81 | 14,380 | 0.199 | 0.754 | 0.926 | 0.938 | 0.067 |
| 54 | 13,613 | 0.188 | 0.770 | 0.937 | 0.952 | 0.061 |
| 23 | 13,404 | 0.233 | 0.712 | 0.914 | 0.955 | 0.073 |
| 72 | 13,397 | 0.218 | 0.704 | 0.884 | 0.942 | 0.089 |
| 62 | 11,009 | 0.105 | 0.864 | 0.951 | 0.887 | 0.058 |
| 42 | 9,717 | 0.197 | 0.747 | 0.919 | 0.957 | 0.070 |
| 45 | 8,601 | 0.231 | 0.715 | 0.913 | 0.943 | 0.072 |
| 33 | 7,524 | 0.137 | 0.808 | 0.923 | 0.915 | 0.075 |
| 56 | 6,518 | 0.238 | 0.702 | 0.910 | 0.963 | 0.076 |
| 48 | 4,118 | 0.271 | 0.672 | 0.909 | 0.968 | 0.081 |
| 32 | 3,545 | 0.162 | 0.777 | 0.914 | 0.927 | 0.075 |

**Disparity gaps** — approval 0.200, equal-opportunity (approval|good) 0.067, default-catch 0.080, ECE 0.033.

> Disparity gaps quantify how unevenly the decision and its calibration fall across groups. Geographic/industry gaps would require mitigation before any real deployment.
