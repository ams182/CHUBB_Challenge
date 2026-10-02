# MLOps Architecture and Conclusions

## 1. Architectural & Infrastructure Decisions
The deployment relies on Databricks Asset Bundles (DABs) to cleanly decouple infrastructure declarations (`databricks.yml`) from core Python orchestration logic.
* **Compute & Execution:** Leverages serverless environments to eliminate cluster management overhead, ensuring fast, automated scaling for scheduled or event-driven runs.
* **Model Persistence & Registry Alignment:** To work around workspace-level policy constraints (such as disabled legacy registries and strict UC-managed S3 permission restrictions in trial environments), the architecture utilizes direct model serialization and Unity Catalog Volumes (`/Volumes/...`). This guarantees reliable read/write persistence without hitting unexpected cloud permission blocks.
* **Separation of Concerns:** ETL and exploratory data analysis (EDA) live upstream to establish baseline references, while the DABs pipelines focus strictly on production-grade execution: batch scoring, distribution drift monitoring, and conditional model retraining.

## 2. Statistical Drift Monitoring
The **Kolmogorov-Smirnov (KS) Test** is implemented for real-time feature drift detection.
* **Why:** It’s a robust, non-parametric test that evaluates both the shape and location of empirical cumulative distribution functions without making strict assumptions about the underlying distribution of insurance premiums or portfolio features.
* **Implementation Mechanics:** A baseline reference dataset is captured and versioned during initial training. The scoring pipeline evaluates incoming production batches against this baseline. Any feature yielding a p-value below the $0.05$ threshold triggers an alert tag (`DRIFT_EXCEEDED_THRESHOLD`) in MLflow for auditing and operational visibility.

## 3. Automated Promotion & Governance Logic
The retraining and promotion workflow is fully deterministic, safe, and built to handle production edge cases gracefully:
* **Evaluation Metric:** Root Mean Squared Error (RMSE) serves as the primary evaluation metric because it heavily penalizes large pricing errors—a critical requirement in insurance underwriting where underestimating severe risk directly translates to financial exposure.
* **Decision Rule:** Candidate models are trained on recent data and evaluated against a held-out validation set. The reigning production model is loaded and benchmarked against the exact same validation subset. Promotion occurs strictly if the candidate achieves a lower RMSE.
* **Edge Case Handling:**
    * *First-Run Bootstrap:* If no production model exists yet, the framework bypasses the comparison check and automatically promotes the initial candidate.
    * *Empty Validation Guard:* Includes explicit exception handling to catch empty validation batches (`ValueError`), aborting the pipeline cleanly instead of pushing unvalidated code to production.

## 4. Roadmap & Future Enhancements
Given extended timelines, production hardening would incorporate:
1. **Databricks Feature Store:** Centralizing feature definitions to guarantee zero train-serve skew across training and scoring pipelines.
2. **Enhanced Imputation Strategies:** Moving from basic median/constant filling to iterative or KNN-based imputers during ETL.
3. **CI/CD Automation:** Wiring GitHub Actions directly to `databricks bundle deploy` for seamless trunk-based deployments.
