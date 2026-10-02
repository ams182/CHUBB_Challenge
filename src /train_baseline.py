import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import json

### comentarios en español para validar su realización :)


from etl_eda import run_etl, perform_eda, prep_data

# Config
# LOCAL_DATA_PATH = "https://github.com/ams182/CHUBB_Challenge/blob/f9471cbb09a041f26d35e6c9d64cd055ad63fdb5/portfolio_mexico.csv"
LOCAL_DATA_PATH = r"C:\Users\amunguia\Desktop\portfolio_mexico.csv"
MODEL_NAME = "chubb_premium_xgboost"

def train_and_register():
    df = run_etl(LOCAL_DATA_PATH)
    perform_eda(df)
    X_train, X_test, y_train, y_test = prep_data(df)
    
    mlflow.set_experiment("/Shared/chubb_baseline_training")
    
    with mlflow.start_run() as run:
        # Tuneo
        print("Tuning XGBoost Hyperparameters...")
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200]
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=5, scoring='neg_mean_squared_error', cv=3, random_state=42)
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # monitoreo y sus stats
        baseline_stats = X_train.describe().to_dict()
        with open("baseline_stats.json", "w") as f:
            json.dump(baseline_stats, f)
        mlflow.log_artifact("baseline_stats.json")
        
        # modelo
        mlflow.xgboost.log_model(best_model, "model", registered_model_name=MODEL_NAME)
        print(f"Baseline model registered. RMSE: {rmse}")

    # a produccion
    client = MlflowClient()
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if latest_versions:
        version = latest_versions[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Version {version} promoted to Production.")

if __name__ == "__main__":
    train_and_register()
