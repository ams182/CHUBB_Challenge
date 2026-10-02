import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd

from etl_eda import run_etl, prep_data

# LOCAL_DATA_PATH = "https://github.com/ams182/CHUBB_Challenge/blob/f9471cbb09a041f26d35e6c9d64cd055ad63fdb5/portfolio_mexico.csv"
LOCAL_DATA_PATH = r"C:\Users\amunguia\Desktop\portfolio_mexico.csv"
MODEL_NAME = "chubb_premium_xgboost"

def run_retrain_and_promote():
    df = run_etl(LOCAL_DATA_PATH)
    X_train, X_val, y_train, y_val = prep_data(df)
    
    mlflow.set_experiment("/Shared/chubb_retraining")
    client = MlflowClient()
    
    with mlflow.start_run() as run:
        # Retrain candidate model
        print("Retraining candidate model...")
        candidate_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, max_depth=6, learning_rate=0.1)
        candidate_model.fit(X_train, y_train)
        
        candidate_preds = candidate_model.predict(X_val)
        candidate_rmse = mean_squared_error(y_val, candidate_preds, squared=False)
        mlflow.log_metric("candidate_rmse", candidate_rmse)
        
        # Log to MLflow
        model_info = mlflow.xgboost.log_model(candidate_model, "model", registered_model_name=MODEL_NAME)
        candidate_version = model_info.registered_model_version
        
        # Transition candidate to Staging
        client.transition_model_version_stage(name=MODEL_NAME, version=candidate_version, stage="Staging")
        
        # Evaluate against Production
        try:
            prod_model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/Production")
            prod_preds = prod_model.predict(X_val)
            prod_rmse = mean_squared_error(y_val, prod_preds, squared=False)
            mlflow.log_metric("production_rmse", prod_rmse)
        except Exception:
            # Edge case: First run or missing Prod model
            print("No Production model found. Defaulting Prod RMSE to infinity.")
            prod_rmse = float('inf')
            
        print(f"Candidate RMSE: {candidate_rmse} | Production RMSE: {prod_rmse}")
        
        # Deterministic Promotion Logic
        if candidate_rmse < prod_rmse:
            print("Candidate outperforms Production. Promoting...")
            client.transition_model_version_stage(name=MODEL_NAME, version=candidate_version, stage="Production", archive_existing_versions=True)
            mlflow.log_param("promotion_decision", "PROMOTED")
        else:
            print("Candidate did not outperform. Archiving candidate...")
            client.transition_model_version_stage(name=MODEL_NAME, version=candidate_version, stage="Archived")
            mlflow.log_param("promotion_decision", "ARCHIVED")

if __name__ == "__main__":
    run_retrain_and_promote()
