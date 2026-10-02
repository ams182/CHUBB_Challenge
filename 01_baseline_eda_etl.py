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
# trabaje local pero existe la opcion online con el path
# DATA_PATH = 'https://raw.githubusercontent.com/ams182/CHUBB_Challenge/f9471cbb09a041f26d35e6c9d64cd055ad63fdb5/portfolio_mexico.csv'
DATA_PATH = r'C:\Users\amunguia\Desktop\portfolio_mexico.csv'
EXPERIMENT_NAME = "/Shared/CHUBB_Baseline_Training"
MODEL_NAME = "chubb_premium_xgboost"

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
        # llenando vacios con 'unknown'
        df[col] = df[col].fillna('unknown')
    
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    for col in numerical_cols:
        # usando la media
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
    # carga
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"File not found at {DATA_PATH}. Please ensure the CSV is on your desktop.")
        return

    # ETL & EDA
    df = perform_etl(df)
    perform_eda(df)

    # caracteristicas y objetivo
    target = 'premium'
    X = df.drop(columns=[target, 'policy_id', 'policy_date'], errors='ignore')
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train.to_csv(r'C:\Users\amunguia\Desktop\CHUBB_Challenge\src\baseline_reference.csv', index=False)

    # Pipeline
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

    # Modelo Setup (XGBoost)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', XGBRegressor(random_state=42))])

    # tuneo
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
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation RMSE: {best_rmse}")
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_rmse", best_rmse)
        
        test_predictions = best_model.predict(X_test)
        test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))
        mlflow.log_metric("test_rmse", test_rmse)
        
### aegurar prvio congi de databricks
        model_info = mlflow.sklearn.log_model(best_model, "model", registered_model_name=MODEL_NAME, serialization_format="cloudpickle")
        print(f"Model registered. URI: {model_info.model_uri}")
        
        client = MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print("Baseline model promoted to Production.")

main()