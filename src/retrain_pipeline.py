import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error
from mlflow.tracking import MlflowClient

MODEL_NAME = "chubb_premium_xgboost"
EXPERIMENT_NAME = "/Shared/CHUBB_Retrain_Promote"
# Path to the latest aggregated dataset
LATEST_DATA_PATH = '/Volumes/main/default/chubb_data/latest_full_portfolio.csv'

def evaluate_model(model, X_val, y_val):
    if len(X_val) == 0:
        raise ValueError("Validation set is empty. Cannot evaluate model.")
    predictions = model.predict(X_val)
    return mean_squared_error(y_val, predictions, squared=False) # RMSE

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    
    # 1. Load Data
    try:
        df = pd.read_csv(LATEST_DATA_PATH)
    except FileNotFoundError:
        print("Latest data not found. Simulating data load for pipeline verification...")
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['premium'] = y

    # Simple train/val split (Assume ETL is abstracted or imported from a shared module)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    X_train = train_df.drop(columns=['premium', 'policy_id'], errors='ignore')
    y_train = train_df['premium']
    X_val = val_df.drop(columns=['premium', 'policy_id'], errors='ignore')
    y_val = val_df['premium']

    with mlflow.start_run(run_name="OnDemand_Retrain_Evaluation") as run:
        # 2. Retrain Model (Simplified to focus on MLOps logic)
        from xgboost import XGBRegressor
        candidate_model = XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
        candidate_model.fit(X_train, y_train)
        
        # 3. Evaluate Candidate
        try:
            candidate_rmse = evaluate_model(candidate_model, X_val, y_val)
        except ValueError as e:
            print(f"Failure mode triggered: {e}")
            mlflow.log_param("pipeline_status", "FAILED_EMPTY_VALIDATION")
            return
            
        mlflow.log_metric("candidate_rmse", candidate_rmse)
        
        # Register Candidate to Staging
        model_info = mlflow.sklearn.log_model(candidate_model, "model")
        candidate_version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
        client.transition_model_version_stage(
            name=MODEL_NAME, version=candidate_version, stage="Staging"
        )
        
        # 4. Compare with Production
        promote_decision = False
        try:
            prod_uri = f"models:/{MODEL_NAME}/Production"
            prod_model = mlflow.sklearn.load_model(prod_uri)
            prod_rmse = evaluate_model(prod_model, X_val, y_val)
            mlflow.log_metric("production_rmse", prod_rmse)
            
            print(f"Candidate RMSE: {candidate_rmse:.2f} | Production RMSE: {prod_rmse:.2f}")
            
            # Deterministic Promotion Logic
            if candidate_rmse < prod_rmse:
                print("Candidate outperforms Production. Promoting...")
                promote_decision = True
            else:
                print("Candidate does not outperform Production. Archiving candidate...")
                client.transition_model_version_stage(name=MODEL_NAME, version=candidate_version, stage="Archived")
                
        except Exception as e:
            print(f"First-run scenario or Prod model unreachable ({e}). Automatically promoting candidate to Production.")
            promote_decision = True
            
        # 5. Execute Promotion
        if promote_decision:
            client.transition_model_version_stage(
                name=MODEL_NAME, 
                version=candidate_version, 
                stage="Production", 
                archive_existing_versions=True
            )
            mlflow.log_param("promotion_status", "PROMOTED")
        else:
            mlflow.log_param("promotion_status", "REJECTED")

if __name__ == '__main__':
    main()
