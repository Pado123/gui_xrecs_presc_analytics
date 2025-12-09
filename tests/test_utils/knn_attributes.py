import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from typing import Union, List, Tuple, Any


def apply_knn_prediction(
    dataset: pd.DataFrame,
    input_columns: List[str],
    output_variable: str,
    n_neighbors: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    task_type: str = 'auto'
) -> dict:
    """
    Applies KNN algorithm on specified input columns to predict the output variable.
    
    Parameters:ff
    -----------
    dataset : pd.DataFrame
        The input dataset containing all variables
    input_columns : List[str]
        List of column names to use as features for prediction
    output_variable : str
        Name of the column to predict (target variable)
    n_neighbors : int, default=5
        Number of neighbors to use for KNN
    test_size : float, default=0.2
        Proportion of dataset to use for testing
    random_state : int, default=42
        Random state for reproducibility
    scale_features : bool, default=True
        Whether to standardize the input features
    task_type : str, default='auto'
        Type of task: 'classification', 'regression', or 'auto' (auto-detect)
    
    Returns:
    --------
    dict
        Dictionary containing model, predictions, metrics, and other results
    """
    
    # Validate inputs
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame")
    
    if not all(col in dataset.columns for col in input_columns):
        missing_cols = [col for col in input_columns if col not in dataset.columns]
        raise ValueError(f"Input columns not found in dataset: {missing_cols}")
    
    if output_variable not in dataset.columns:
        raise ValueError(f"Output variable '{output_variable}' not found in dataset")
    
    # Extract features and target, ignoring other columns
    X = dataset[input_columns].copy()
    y = dataset[output_variable].copy()
    
    # Handle missing values
    if X.isnull().any().any():
        print("Warning: Missing values detected in input columns. Dropping rows with missing values.")
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
    
    if len(X) == 0:
        raise ValueError("No valid data remaining after handling missing values")
    
    # Determine task type automatically if not specified
    if task_type == 'auto':
        # Check if target variable is categorical or continuous
        if y.dtype == 'object' or len(y.unique()) <= 10:
            task_type = 'classification'
        else:
            task_type = 'regression'
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == 'classification' else None
    )
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
    
    # Initialize and train KNN model
    if task_type == 'classification':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)
    
    # Calculate metrics
    metrics = {}
    if task_type == 'classification':
        metrics['test_accuracy'] = accuracy_score(y_test, y_pred)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    else:
        metrics['test_mse'] = mean_squared_error(y_test, y_pred)
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        metrics['test_r2'] = r2_score(y_test, y_pred)
        metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
    
    # Prepare results
    results = {
        'model': model,
        'scaler': scaler,
        'task_type': task_type,
        'input_columns': input_columns,
        'output_variable': output_variable,
        'n_neighbors': n_neighbors,
        'predictions': {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred
        },
        'metrics': metrics,
        'data_splits': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    }
    
    return results


def predict_new_data(
    results: dict,
    new_data: pd.DataFrame
) -> np.ndarray:
    """
    Use trained KNN model to predict on new data.
    
    Parameters:
    -----------
    results : dict
        Results dictionary returned by apply_knn_prediction
    new_data : pd.DataFrame
        New data to make predictions on
    
    Returns:
    --------
    np.ndarray
        Predictions for the new data
    """
    model = results['model']
    scaler = results['scaler']
    input_columns = results['input_columns']
    
    # Extract features
    X_new = new_data[input_columns].copy()
    
    # Handle missing values
    if X_new.isnull().any().any():
        print("Warning: Missing values detected in new data. These will be ignored.")
        X_new = X_new.dropna()
    
    # Scale features if scaler was used
    if scaler is not None:
        X_new_scaled = scaler.transform(X_new)
    else:
        X_new_scaled = X_new.values
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    
    return predictions


# Example usage function
def example_usage():
    """
    Example of how to use the KNN prediction function.
    """
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'irrelevant_col1': np.random.random(n_samples),
        'irrelevant_col2': np.random.choice(['A', 'B', 'C'], n_samples),
    }
    
    # Create target variable (classification example)
    data['target_class'] = (
        (data['feature1'] + data['feature2'] + data['feature3']) > 0
    ).astype(int)
    
    # Create target variable (regression example)
    data['target_continuous'] = (
        data['feature1'] * 2 + data['feature2'] * 1.5 + data['feature3'] * 0.5 + 
        np.random.normal(0, 0.1, n_samples)
    )
    
    df = pd.DataFrame(data)
    
    # Define input columns (ignoring irrelevant columns)
    input_cols = ['feature1', 'feature2', 'feature3']
    
    # Example 1: Classification
    print("=== Classification Example ===")
    results_clf = apply_knn_prediction(
        dataset=df,
        input_columns=input_cols,
        output_variable='target_class',
        n_neighbors=5,
        task_type='classification'
    )
    
    print(f"Classification Metrics: {results_clf['metrics']}")
    
    # Example 2: Regression
    print("\n=== Regression Example ===")
    results_reg = apply_knn_prediction(
        dataset=df,
        input_columns=input_cols,
        output_variable='target_continuous',
        n_neighbors=5,
        task_type='regression'
    )
    
    print(f"Regression Metrics: {results_reg['metrics']}")
    
    # Example 3: Predict on new data
    print("\n=== Predicting on New Data ===")
    new_data = pd.DataFrame({
        'feature1': [0.5, -0.5, 1.0],
        'feature2': [1.0, 0.0, -1.0],
        'feature3': [-0.5, 0.5, 0.0]
    })
    
    new_predictions_clf = predict_new_data(results_clf, new_data)
    new_predictions_reg = predict_new_data(results_reg, new_data)
    
    print(f"New Classification Predictions: {new_predictions_clf}")
    print(f"New Regression Predictions: {new_predictions_reg}")


if __name__ == "__main__":
    example_usage()
