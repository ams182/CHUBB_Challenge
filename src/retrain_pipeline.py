import pandas as pd
import numpy as np
import mlflow
import os
import joblib
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature
from xgboost import XGBRegressor

### comentarios en español para validar su realización :)

EXPERIMENT_NAME = "/Shared/CHUBB_Retrain_Promote"

# Definir ruta segura (si el volumen de UC no existe, usa una ruta local temporal)
UC_VOLUME_DIR = '/Volumes/main/default/chubb_data'
if os.path.exists(UC_VOLUME_DIR):
    LATEST_DATA_PATH = f'{UC_VOLUME_DIR}/latest_full_portfolio.csv'
    PROD_MODEL_PATH = f'{UC_VOLUME_DIR}/production_model.xgb'
else:

    os.makedirs('temp_data', exist_ok=True)
    LATEST_DATA_PATH = 'temp_data/latest_full_portfolio.csv'
    PROD_MODEL_PATH = 'temp_data/production_model.xgb'

def evaluate_model(model, X_val, y_val):
    if len(X_val) == 0:
        raise ValueError("Validation set is empty. Cannot evaluate model.")
    predictions = model.predict(X_val)
    return mean_squared_error(y_val, predictions, squared=False) 

def main():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    try:
        df = pd.read_csv(LATEST_DATA_PATH)
    except FileNotFoundError:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['premium'] = y

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    X_train = train_df.drop(columns=['premium', 'policy_id'], errors='ignore')
    y_train = train_df['premium']
    X_val = val_df.drop(columns=['premium', 'policy_id'], errors='ignore')
    y_val = val_df['premium']

    with mlflow.start_run(run_name="OnDemand_Retrain_Evaluation") as run:
        candidate_model = XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
        candidate_model.fit(X_train, y_train)
        
        try:
            candidate_rmse = evaluate_model(candidate_model, X_val, y_val)
        except ValueError as e:
            mlflow.log_param("pipeline_status", "FAILED_EMPTY_VALIDATION")
            return
            
        mlflow.log_metric("candidate_rmse", candidate_rmse)
        
        signature = infer_signature(X_train, candidate_model.predict(X_train))
        mlflow.xgboost.log_model(candidate_model, "model", signature=signature)
        
        promote_decision = False
        try:
            prod_model = joblib.load(PROD_MODEL_PATH)
            prod_predictions = prod_model.predict(X_val)
            prod_rmse = mean_squared_error(y_val, prod_predictions, squared=False)
            mlflow.log_metric("production_rmse", prod_rmse)
            
            if candidate_rmse < prod_rmse:
                promote_decision = True
            else:
                mlflow.log_param("promotion_status", "REJECTED")
        except Exception as e:
            promote_decision = True
            
        if promote_decision:
            joblib.dump(candidate_model, PROD_MODEL_PATH)
            mlflow.log_param("promotion_status", "PROMOTED")
            print("Model into prod")
        else:
            mlflow.log_param("promotion_status", "REJECTED")

if __name__ == '__main__':
    main()