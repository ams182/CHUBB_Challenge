# CHUBB MLOps Engineer Technical Challenge

## System Overview
This project establishes the MLOps infrastructure for a Property & Casualty insurance portfolio. It features a complete Databricks Asset Bundles (DABs) configuration deploying two primary pipelines:
1. **Scheduled Scoring & Drift Monitor**: Checks incoming data distributions against baseline training metrics.
2. **On-Demand Retraining & Promotion**: Evaluates newly trained models against the current Production model on a holdout set, promoting deterministically based on RMSE.

## Repository Setup
1. Clone this repository to your local machine.
2. Place `portfolio_mexico.csv` on your Desktop at `C:\Users\amunguia\Desktop\portfolio_mexico.csv`.
3. Configure your Databricks CLI authentication (`databricks auth login`).
4. Deploy the infrastructure using: `databricks bundle deploy`

## MLflow Monitoring
* **Training Experiment**: Found at `/Shared/chubb_baseline_training`. Contains hyperparameter tuning history and initial baseline stats.
* **Drift Experiment**: Found at `/Shared/chubb_drift_monitoring`. Logs drift metrics and alerts.
* **Retraining Experiment**: Found at `/Shared/chubb_retraining`. Contains auditable traces of RMSE comparisons and promotion decisions.

---

## Architectural Write-Up & Conclusions

### ETL and Preprocessing
A custom robust ETL framework was built to manage data irregularities (e.g., unexpected characters, missing variables, Spanish accents). Utilizing the `unicodedata` library ensures feature space consistency before applying ML algorithms, which prevents silent pipeline failures during inference. 

### Model Selection & Tuning
While a basic linear regression was suggested, XGBoost provides a superior capability for capturing nonlinear relationships in insurance premiums (e.g., interactions between coverage limits and geographic locations). Hyperparameter tuning (`RandomizedSearchCV`) is implemented to establish a competitive baseline, ensuring the deployed architecture handles modern ML artifacts effectively.

### Drift Detection
Drift is monitored by calculating the distributional shift of incoming feature means against the baseline statistics logged during training. If the relative shift exceeds a defined threshold (0.05), a deterministic alert is raised and logged into MLflow. This simple, lightweight statistical check guarantees that downstream applications are insulated from silent data deterioration.

### Promotion Logic
The promotion pipeline relies on a strict, deterministic evaluation: candidate models are compared to the active Production model on a static validation set. The decision metric is RMSE. 
* **Edge Cases:** If the MLflow registry lacks a Production model (e.g., first run scenario), the candidate is automatically promoted.
* **Auditability:** Every metric (Candidate RMSE vs Prod RMSE) and the final decision rule is hard-logged as parameters in the MLflow run, providing total visibility to data governance teams. 

### Future Work
Given more time, I would expand the drift detection to utilize full multidimensional distance metrics (like the Wasserstein distance or PSI) rather than point estimates. I would also integrate Delta Live Tables (DLT) for more rigorous data quality expectations prior to the scoring phase.
