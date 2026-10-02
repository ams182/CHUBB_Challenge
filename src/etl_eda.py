import pandas as pd
import numpy as np
import unicodedata
from sklearn.model_selection import train_test_split

def clean_string(text):
    if pd.isna(text):
        return text
    # Remove accents and special characters
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8').strip()

def run_etl(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Handle missing values: fill numeric with median, categorical with 'Unknown'
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    # Clean string columns (accents, weird characters, blanks)
    for col in cat_cols:
        df[col] = df[col].apply(clean_string)
        
    return df

def perform_eda(df):
    print("--- Exploratory Data Analysis ---")
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values after ETL:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    # Note: In a notebook, we would plot distributions here using matplotlib/seaborn.
    print("---------------------------------")

def prep_data(df, target_col='premium'):
    # One-hot encoding for simplicity in this baseline
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
