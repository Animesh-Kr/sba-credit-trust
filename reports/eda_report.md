# SBA Credit-Trust — Phase A EDA & Leakage Audit

- Rows (total): **899,164**
- Rows with usable label (PIF/CHGOFF): **897,167**
- Default (CHGOFF) prevalence: **0.1756** (~17.6%)

## Leakage audit (why these columns are dropped from features)

| Column | Kind | Leakage signal |
|---|---|---|
| `ChgOffDate` | post-outcome | P(default | ChgOffDate present) = 0.970 vs base 0.176 |
| `ChgOffPrinGr` | post-outcome | P(default | ChgOffPrinGr>0) = 0.970 vs base 0.176 |
| `BalanceGross` | post-outcome | P(default | BalanceGross>0) = 0.000 vs base 0.176 |
| `DisbursementGross` | post-decision | P(default | DisbursementGross>0) = 0.176 vs base 0.176 |
| `DisbursementDate` | post-decision | P(default | DisbursementDate present) = 0.176 vs base 0.176 |

## Categorical cardinality (post-engineering)

| Feature | # categories |
|---|---|
| `State` | 51 |
| `BankState` | 56 |
| `NewExist` | 2 |
| `UrbanRural` | 3 |
| `RevLineCr` | 2 |
| `LowDoc` | 2 |
| `naics_sector` | 24 |
| `is_franchise` | 2 |

## Missingness (feature frame)

| Feature | % missing |
|---|---|
| `RevLineCr` | 30.90% |
| `naics_sector` | 22.48% |
| `LowDoc` | 0.67% |
| `BankState` | 0.17% |
| `NewExist` | 0.13% |
| `ApprovalFY` | 0.00% |
| `State` | 0.00% |
| `Term` | 0.00% |
| `NoEmp` | 0.00% |
| `CreateJob` | 0.00% |
| `RetainedJob` | 0.00% |
| `GrAppv` | 0.00% |
| `SBA_Appv` | 0.00% |
| `sba_portion` | 0.00% |
| `real_estate` | 0.00% |
| `UrbanRural` | 0.00% |
| `is_franchise` | 0.00% |

## Temporal default-rate trend (motivates the shift study)

| ApprovalFY | default rate | n |
|---|---|---|
| 1966 | 1.000 | 1 |
| 1968 | 1.000 | 1 |
| 1969 | 0.667 | 3 |
| 1970 | 0.875 | 8 |
| 1971 | 1.000 | 18 |
| 1972 | 0.840 | 25 |
| 1973 | 0.918 | 49 |
| 1974 | 0.952 | 42 |
| 1975 | 0.897 | 29 |
| 1976 | 0.954 | 65 |
| 1977 | 0.934 | 137 |
| 1978 | 0.958 | 239 |
| 1979 | 0.951 | 349 |
| 1980 | 0.962 | 453 |
| 1981 | 0.708 | 602 |
| 1982 | 0.424 | 719 |
| 1983 | 0.317 | 1,682 |
| 1984 | 0.369 | 2,019 |
| 1985 | 0.404 | 1,941 |
| 1986 | 0.408 | 2,118 |
| 1987 | 0.435 | 2,218 |
| 1988 | 0.494 | 1,898 |
| 1989 | 0.065 | 13,245 |
| 1990 | 0.045 | 14,859 |
| 1991 | 0.028 | 15,660 |
| 1992 | 0.022 | 20,875 |
| 1993 | 0.019 | 23,299 |
| 1994 | 0.022 | 31,584 |
| 1995 | 0.028 | 45,688 |
| 1996 | 0.041 | 40,021 |
| 1997 | 0.060 | 37,718 |
| 1998 | 0.082 | 36,005 |
| 1999 | 0.099 | 37,348 |
| 2000 | 0.114 | 37,352 |
| 2001 | 0.119 | 37,317 |
| 2002 | 0.117 | 44,307 |
| 2003 | 0.145 | 58,000 |
| 2004 | 0.180 | 68,195 |
| 2005 | 0.253 | 76,958 |
| 2006 | 0.350 | 75,756 |
| 2007 | 0.428 | 71,649 |
| 2008 | 0.412 | 39,458 |
| 2009 | 0.208 | 19,103 |
| 2010 | 0.137 | 16,828 |
| 2011 | 0.079 | 12,593 |
| 2012 | 0.057 | 5,992 |
| 2013 | 0.029 | 2,455 |
| 2014 | 0.019 | 268 |
