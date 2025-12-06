# EasyVisa

EasyVisa is a machine learning project that predicts US work visa case outcomes (Certified vs. Denied) using the 2016 OFLC dataset. The goal is to help triage applications with high precision so reviewers can focus on the most likely approvals first.

## Overview
- Problem: Growing visa volumes make manual review difficult and slow.
- Approach: Supervised classification with engineered features, calibrated probabilities, and precision-focused thresholding.
- Outcome: A precision-first pipeline to surface candidates most likely to be certified.

## Repository Contents
- `EasyVisa.csv`: Source dataset with applicant and employer attributes.
- `EasyVisa_Full_Code_Notebook.ipynb`: End-to-end exploration, modeling, and evaluation.
- `EasyVisa_Full_Code_Notebook.html`: Rendered HTML version of the full notebook (no runtime needed).
- `EasyVisa_Precision_Optimized.ipynb`: Concise, precision-optimized modeling pipeline.

## Data Summary
Key columns include:
- `case_status` (target): Certified/Denied
- `education_of_employee`, `has_job_experience`, `requires_job_training`
- `no_of_employees`, `yr_of_estab`, `region_of_employment`, `continent`
- `prevailing_wage`, `unit_of_wage`, `full_time_position`

The precision pipeline engineers features such as:
- Annualized wage (`prevailing_wage` normalized by `unit_of_wage`)
- Log transforms for skewed numeric fields
- Employer age derived from `yr_of_estab`
- Rare-category smoothing for high-cardinality categoricals

## Modeling Highlights
- Preprocessing: `OneHotEncoder` + passthrough numerics via `ColumnTransformer`
- Model: `GradientBoostingClassifier`
- Calibration: `CalibratedClassifierCV` (isotonic) for well-formed probabilities
- Decision: Threshold tuning targeting a desired precision (e.g., ≥ 0.85)

## Quickstart
1. Environment (Python 3.10+ recommended):
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn xgboost imbalanced-learn
   ```
2. Ensure `EasyVisa.csv` is in the repository root (as provided).
3. Open a notebook tool (e.g., JupyterLab/VS Code) and run:
   - `EasyVisa_Precision_Optimized.ipynb` for the precision-first pipeline, or
   - `EasyVisa_Full_Code_Notebook.ipynb` for the complete EDA and modeling flow.

## How To Use
- Adjust the `target_precision` in `EasyVisa_Precision_Optimized.ipynb` to fit your review tolerance.
- Re-run cells to produce validation/test metrics (precision, recall, confusion matrix).
- Inspect feature engineering steps to adapt to new time periods or schema changes.

## Notes
- The notebooks assume the year 2016 when deriving employer age; adjust as needed for other years.
- Columns must match the provided schema; if upstream data changes, revisit preprocessing/encodings.

## Attribution
This project is inspired by the OFLC visa certification workflow and uses a derived dataset (`EasyVisa.csv`) for educational and prototyping purposes.

