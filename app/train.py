import pandas as pd
import numpy as np
import mlflow
import time
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mlflow.models.signature import infer_signature


# Load data from S3
def load_data_from_s3(bucket_name, key):
    """
    Load dataset from the given S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        key (str): The S3 key (file path in the bucket).

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    s3_client = boto3.client('s3')

    # Download the file from S3 into memory
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    data = obj['Body'].read().decode('utf-8')
    
    # Read the data into a pandas DataFrame
    df = pd.read_csv(StringIO(data), sep=';')
    
    return df

# Preprocess data
def preprocess_data(df):
    """
    Perform data cleaning and preprocessing.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        tuple: Preprocessed data (X_train, X_test, y_train, y_test).
    """
    # Delete rows with missing values
    df = df.dropna()

    # Drop ID column
    df = df.drop(columns=["id"])

    # Drop duplicates
    df = df.drop_duplicates()

    # Transform height to meters
    df['height'] = df['height'].apply(lambda x: x / 100)

    # Remove heights and weights that fall below 2.5% or above 97.5% of a given range
    df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index, inplace=True)
    df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index, inplace=True)

    # Remove negative numbers in blood pressure
    df = df[(df['ap_hi'] >= 0) & (df['ap_lo'] >= 30)]

    # Remove records where diastolic pressure is higher than systolic pressure
    df = df[df['ap_lo'] < df['ap_hi']]

    # Calculate BMI
    df['bmi'] = round(df['weight'] / (df['height'] ** 2), 2)

    # Drop weight and height columns
    df = df.drop(columns=['height', 'weight'])

    # Convert age to years if needed
    #if df["age"].mean() > 100:
    #    df["age"] = (df["age"] / 365).round()

    # Modify gender: 1 -> 0, 2 -> 1
    #df["gender"] = df["gender"].replace({1: 0, 2: 1})

    # Split the dataframe into X (features) and y (target)
    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the pipeline
def create_pipeline():
    # Preprocessing
    # Categorical variables pipeline
    categorical_features = ['gluc', 'cholesterol']
    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(drop='first'))]  # OneHotEncoder for categorical features
    )

    # Quantitative variables pipeline
    numeric_features = ['age', 'ap_hi', 'ap_lo', 'bmi']
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]  # StandardScaler for numeric features
    )

    # Pipelines combination
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Machine learning pipeline with RandomForestClassifier
    return Pipeline(steps=[
        ("Preprocessing", preprocessor),
        ("Random_Forest", RandomForestClassifier()) 
    ])

# Train model
def train_model(pipe, X_train, y_train, param_grid, cv=2, n_jobs=-1, verbose=3):
    """
    Train the model using GridSearchCV.

    Args:
        pipe (Pipeline): The pipeline to use for training.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        param_grid (dict): The hyperparameter grid to search over.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of jobs to run in parallel.
        verbose (int): Verbosity level.

    Returns:
        GridSearchCV: Trained GridSearchCV object.
    """
    model = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    return model

# Log metrics and model to MLflow
def log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    """
    Log training and test metrics, and the model to MLflow.

    Args:
        model (GridSearchCV): The trained model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    mlflow.log_metric("Train Score", model.score(X_train, y_train))
    mlflow.log_metric("Test Score", model.score(X_test, y_test))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )

# Main function to execute the workflow
def run_experiment(experiment_name, bucket_name, key, param_grid, artifact_path, registered_model_name):
    """
    Run the entire ML experiment pipeline.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        bucket_name (str): S3 bucket name where the dataset is stored.
        key (str): S3 key (file path in the bucket).
        param_grid (dict): The hyperparameter grid for GridSearchCV.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    # Start timing
    start_time = time.time()

    # Load and preprocess data
    df = load_data_from_s3(bucket_name, key)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Create pipeline
    pipe = create_pipeline()

    # Set experiment's info 
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        train_model(pipe, X_train, y_train, param_grid)

    # Print timing
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

# Entry point for the script
if __name__ == "__main__":
    # Define experiment parameters
    experiment_name = "cardio-detect"
    bucket_name = "projet-cardiodetect" 
    key = "cardio_train.csv"

    param_grid = {
        "Random_Forest__max_depth": [2, 4, 6, 8, 10],
        "Random_Forest__min_samples_leaf": [1, 2, 5],
        "Random_Forest__min_samples_split": [2, 4, 8],
        "Random_Forest__n_estimators": [10, 20, 40, 60, 80, 100],
    }
    artifact_path = "modeling_cardiodetect"
    registered_model_name = "random_forest"

    # Run the experiment
    run_experiment(experiment_name, bucket_name, key, param_grid, artifact_path, registered_model_name)
