import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json

from etl_eda import run_etl, prep_data

# LOCAL_DATA_PATH = "https://github.com/ams182/CHUBB_Challenge/blob/f9471cbb09a041f26d35e6c9d64cd055ad63fdb5/portfolio_mexico.csv"
LOCAL_DATA_PATH = r"C:\Users\amunguia\Desktop\portfolio_mexico.csv"
MODEL_NAME = "chubb_premium_xgboost"
DRIFT_THRESHOLD = 0.05

def run_scoring_and_drift():
    df = run_etl(LOCAL_DATA_PATH)
    _, X_new, _, _ = prep_data(df) # Simulating new batch data
    
    mlflow.set_experiment("/Shared/chubb_drift_monitoring")
    
    client = MlflowClient()
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            raise ValueError("No Production model found.")
        prod_version = prod_versions[0]
        model_uri = f"models:/{MODEL_NAME}/Production"
        
        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    predictions = model.predict(X_new)
    
    # Retrieve baseline stats from the original run
    run_id = prod_version.run_id
    local_path = client.download_artifacts(run_id, "baseline_stats.json")
    with open(local_path, 'r') as f:
        baseline_stats = json.load(f)

    with mlflow.start_run():
        drift_detected = False
        # Calculate drift using Kolmogorov-Smirnov test on a key feature (e.g., sum_insured)
        # For demonstration, checking against mean distribution shifts
        for col in X_new.columns:
            if col in baseline_stats:
                base_mean = baseline_stats[col]['mean']
                new_mean = X_new[col].mean()
                shift = abs((new_mean - base_mean) / (base_mean + 1e-9))
                mlflow.log_metric(f"{col}_drift_score", shift)
                
                if shift > DRIFT_THRESHOLD:
                    drift_detected = True
                    print(f"DRIFT ALERT: Feature {col} has shifted by {shift*100:.2f}%")
        
        mlflow.log_param("drift_detected", str(drift_detected))
        if drift_detected:
            # Surface alert via MLflow tags / delta table entry
            mlflow.set_tag("alert", "DRIFT_EXCEEDED_THRESHOLD")
            print("Drift exceeds threshold. Consider triggering Retraining Pipeline.")

if __name__ == "__main__":
    run_scoring_and_drift()
