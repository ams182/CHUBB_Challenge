import pandas as pd
import numpy as np
import unicodedata
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from mlflow.tracking import MlflowClient

### comentarios en español para validar su realización :)

# config databricks
os.environ["DATABRICKS_HOST"] = "https://dbc-b679398c-700f.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapidab3e0ebf0c075abf4f96f587b330151"
mlflow.set_tracking_uri("databricks")
# forzar mi catalogo personalizado
mlflow.set_registry_uri("databricks-uc")

DATA_PATH = r'C:\Users\amunguia\Desktop\portfolio_mexico.csv'
EXPERIMENT_NAME = "/Shared/CHUBB_Baseline_Training"
MODEL_NAME = "main.default.chubb_premium_xgboost"

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text).lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

def perform_etl(df):
    print("Performing ETL: Cleaning accents, standardizing text, handling blanks...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].apply(clean_text)
        df[col] = df[col].fillna('unknown')
    
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def perform_eda(df):
    print("Generating EDA visualizations...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['premium'], bins=50, kde=True)
    plt.title('Distribution of Target Variable: Premium')
    plt.savefig(r'C:\Users\amunguia\Desktop\CHUBB_Challenge\docs\premium_distribution.png')
    plt.close()
    
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Numeric Feature Correlation Matrix')
    plt.savefig(r'C:\Users\amunguia\Desktop\CHUBB_Challenge\docs\correlation_matrix.png')
    plt.close()

def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"File not found at {DATA_PATH}.")
        return

    df = perform_etl(df)
    perform_eda(df)

    target = 'premium'
    X = df.drop(columns=[target, 'policy_id', 'policy_date'], errors='ignore')
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(r'C:\Users\amunguia\Desktop\CHUBB_Challenge\src\baseline_reference.csv', index=False)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', XGBRegressor(random_state=42))])

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5],
        'regressor__learning_rate': [0.05, 0.1]
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="Baseline_XGBoost_Tuning"):
        print("Starting Hyperparameter Tuning...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_rmse = np.sqrt(-grid_search.best_score_)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_rmse", best_rmse)
        
        test_predictions = best_model.predict(X_test)
        test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))
        mlflow.log_metric("test_rmse", test_rmse)
        

        print("Inferring model signature for Unity Catalog...")
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        # registrando el modelo
        model_info = mlflow.sklearn.log_model(
            best_model, 
            "model", 
            registered_model_name=MODEL_NAME, 
            serialization_format="cloudpickle",
            signature=signature
        )
        print(f"Model registered. URI: {model_info.model_uri}")
        

        client = MlflowClient()
        version = model_info.registered_model_version
        client.set_registered_model_alias(name=MODEL_NAME, alias="Production", version=version)
        print("✅ Baseline model promoted to Production (via Unity Catalog Alias).")


    print("Actualizando scripts locales para compatibilidad con Databricks Unity Catalog...")
    
    scoring_script = """import pandas as pd
import numpy as np
import mlflow
from scipy.stats import ks_2samp
from mlflow.tracking import MlflowClient

MODEL_NAME = "main.default.chubb_premium_xgboost"
EXPERIMENT_NAME = "/Shared/CHUBB_Scoring_Drift_Monitor"
NEW_BATCH_PATH = '/Volumes/main/default/chubb_data/new_batch.csv'
BASELINE_REF_PATH = '/Volumes/main/default/chubb_data/baseline_reference.csv'

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
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    try:
        print("Loading Production model via Unity Catalog Alias...")
        model_uri = f"models:/{MODEL_NAME}@Production"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Failed to load Production model: {e}")
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
"""

    retrain_script = """import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

MODEL_NAME = "main.default.chubb_premium_xgboost"
EXPERIMENT_NAME = "/Shared/CHUBB_Retrain_Promote"
LATEST_DATA_PATH = '/Volumes/main/default/chubb_data/latest_full_portfolio.csv'

def evaluate_model(model, X_val, y_val):
    if len(X_val) == 0:
        raise ValueError("Validation set is empty. Cannot evaluate model.")
    predictions = model.predict(X_val)
    return mean_squared_error(y_val, predictions, squared=False) 

def main():
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    
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
        from xgboost import XGBRegressor
        candidate_model = XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
        candidate_model.fit(X_train, y_train)
        
        try:
            candidate_rmse = evaluate_model(candidate_model, X_val, y_val)
        except ValueError as e:
            mlflow.log_param("pipeline_status", "FAILED_EMPTY_VALIDATION")
            return
            
        mlflow.log_metric("candidate_rmse", candidate_rmse)
        
        # Guardar con firma obligatoria
        signature = infer_signature(X_train, candidate_model.predict(X_train))
        model_info = mlflow.sklearn.log_model(candidate_model, "model", signature=signature)
        candidate_version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
        
        client.set_registered_model_alias(name=MODEL_NAME, alias="Staging", version=candidate_version)
        
        promote_decision = False
        try:
            prod_uri = f"models:/{MODEL_NAME}@Production"
            prod_model = mlflow.sklearn.load_model(prod_uri)
            prod_rmse = evaluate_model(prod_model, X_val, y_val)
            mlflow.log_metric("production_rmse", prod_rmse)
            
            if candidate_rmse < prod_rmse:
                promote_decision = True
            else:
                client.set_registered_model_alias(name=MODEL_NAME, alias="Archived", version=candidate_version)
                
        except Exception as e:
            promote_decision = True
            
        if promote_decision:
            client.set_registered_model_alias(name=MODEL_NAME, alias="Production", version=candidate_version)
            mlflow.log_param("promotion_status", "PROMOTED")
        else:
            mlflow.log_param("promotion_status", "REJECTED")

if __name__ == '__main__':
    main()
"""

    with open(r"C:\Users\amunguia\Desktop\CHUBB_Challenge\src\scoring_pipeline.py", "w", encoding="utf-8") as f:
        f.write(scoring_script)
    with open(r"C:\Users\amunguia\Desktop\CHUBB_Challenge\src\retrain_pipeline.py", "w", encoding="utf-8") as f:
        f.write(retrain_script)
        
    print("✅ ¡Todo actualizado con éxito!")

main()