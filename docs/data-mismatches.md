# Data Mismatches — Challenge PDF vs Provided CSV

Resolution rule for every mismatch: **the data wins.** The PDF is a narrative description;
the CSV in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is the actual dataset the
classifier consumes. Where they disagree, the pipeline follows the CSV and this
document records both values.

## Named column/value deltas

| # | Topic | Challenge PDF says | CSV contains | Resolution |
|---|-------|--------------------|--------------|------------|
| 1 | Payment method | "Postal check" (postal/bank mail checks) | `Payment_Method` = `Mailed check` | Data wins: the value is `Mailed check`; no postal-check category exists |
| 2 | Column naming | Describes features conversationally (senior status, married status, etc.) | Header names `Senior_Citizen`, `Is_Married`, `Phone_Service`, `Dual`, `Internet_Service`, … | Data wins: canonical model vocabulary is the CSV header (trimmed of whitespace) |
| 3 | Senior encoding | Describes senior flag as yes/no | `Senior_Citizen` encoded 0/1 | Data wins: mapped to boolean during cleaning (2.2) |
| 4 | Identifier | Implies per-customer row identity | `customerID` present but unused for prediction | Data wins: client row key; dropped at load per leakage guardrail |

## Numeric/format observations

| # | Observation | Handling |
|---|-------------|----------|
| 5 | 11 rows have blank `Total_Charges` — all `tenure` = 0, all `Churn` = No | Documented rule: `Total_Charges := 0.0` ("no billing cycle completed"); asserted pre-split; cohort is tiny (0.16% of rows) |
| 6 | `Total_Charges` parses cleanly as numeric once the 11 blanks are handled | Coerced to Float64 before any split (leakage guardrail) |
| 7 | `avg_monthly` is not a source column | Engineered as `Total_Charges / tenure` where tenure > 0; null for the 11 tenure-0 rows |

## Statistics pinned by tests

- Rows: 7,043 × 21 (ingest asserts exactly; 20 after `customerID` drop)
- Churn rate: 26.54% ("Yes" = 1,869)
- Blank `Total_Charges` rows: exactly 11 (unit-tested)
- Payment categories present: `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` — no "Postal check"

Reviewed against the real CSV on this branch; the unit tests
(`tests/unit/test_data_pipeline.py`) pin every documented rule above.
