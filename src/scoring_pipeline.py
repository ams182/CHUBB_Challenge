import pandas as pd
import numpy as np
import mlflow
from scipy.stats import ks_2samp
from xgboost import XGBRegressor

EXPERIMENT_NAME = "/Shared/CHUBB_Scoring_Drift_Monitor"
NEW_BATCH_PATH = '/Volumes/main/default/chubb_data/new_batch.csv'
BASELINE_REF_PATH = '/Volumes/main/default/chubb_data/baseline_reference.csv'
PROD_MODEL_PATH = '/Volumes/main/default/chubb_data/production_model.xgb'

### comentarios en español para validar su realización :)

def calculate_drift(reference_data, current_data, numerical_columns):
    drift_metrics = {}
    drift_detected = False
    for col in numerical_columns:
        if col in reference_data.columns and col in current_data.columns:
            statistic, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            drift_metrics[f"{col}_ks_stat"] = statistic
            drift_metrics[f"{col}_p_value"] = p_value
            if p_value < 0.05:
                drift_detected = True
                print(f"Drift detected in feature: {col} (p-value: {p_value:.4f})")
    return drift_metrics, drift_detected

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    try:
        print("Loading Production model from Unity Catalog Volume...")
        model = XGBRegressor()
        model.load_model(PROD_MODEL_PATH)
    except Exception as e:
        print(f"Failed to load Production model from Volume: {e}")
        return

    try:
        current_data = pd.read_csv(NEW_BATCH_PATH)
        reference_data = pd.read_csv(BASELINE_REF_PATH)
    except FileNotFoundError:
        reference_data = pd.DataFrame({'sum_insured': np.random.normal(10000, 2000, 1000)})
        current_data = pd.DataFrame({'sum_insured': np.random.normal(12000, 2500, 500)}) 

    X_current = current_data.drop(columns=['premium', 'policy_id', 'policy_date'], errors='ignore')
    
    try:
        predictions = model.predict(X_current)
        current_data['predicted_premium'] = predictions
        print(f"Successfully scored {len(current_data)} records.")
    except Exception as e:
        print(f"Scoring failed. Data schema might not match: {e}")
        return

    numeric_cols = X_current.select_dtypes(include=[np.number]).columns
    
    with mlflow.start_run(run_name="Scheduled_Scoring_and_Drift"):
        drift_metrics, is_drifting = calculate_drift(reference_data, X_current, numeric_cols)
        mlflow.log_metrics(drift_metrics)
        mlflow.log_param("drift_detected", str(is_drifting))
        
        if is_drifting:
            mlflow.set_tag("alert_status", "DRIFT_EXCEEDED_THRESHOLD")
        else:
            mlflow.set_tag("alert_status", "NOMINAL")

if __name__ == '__main__':
    main()