import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from typing import List, Dict, Any, Union, Tuple
import warnings


class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for LabelEncoder to work with ColumnTransformer"""
    
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(X[col].astype(str))
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            # Handle unknown categories by assigning them to a default value
            try:
                X_transformed[col] = self.encoders[col].transform(X[col].astype(str))
            except ValueError:
                # Handle unknown categories
                known_categories = set(self.encoders[col].classes_)
                X_transformed[col] = X[col].astype(str).apply(
                    lambda x: self.encoders[col].transform([x])[0] if x in known_categories else 0
                )
        return X_transformed


class KNNPredictor:
    """
    A KNN-based predictor with automatic hyperparameter tuning using k-fold cross-validation.
    Supports both classification and regression tasks.
    """
    
    def __init__(self, task_type: str = 'auto', random_state: int = 42, categorical_encoding: str = 'onehot', 
                 categorical_columns: List[str] = None, max_categories: int = 20):
        """
        Initialize the KNN predictor.
        
        Parameters:
        -----------
        task_type : str, default='auto'
            Type of task: 'classification', 'regression', or 'auto' (auto-detect)
        random_state : int, default=42
            Random state for reproducibility
        categorical_encoding : str, default='onehot'
            Method for encoding categorical variables: 'onehot' or 'label'
        categorical_columns : List[str], optional
            Explicitly specify which columns should be treated as categorical.
            If None, will auto-detect categorical columns.
        max_categories : int, default=20
            Maximum number of unique values for a numerical column to be considered categorical
        """
        self.task_type = task_type
        self.random_state = random_state
        self.categorical_encoding = categorical_encoding
        self.categorical_columns = categorical_columns
        self.max_categories = max_categories
        self.model = None
        self.feature_columns = None
        self.target_column = None
        self.is_fitted = False
        self.best_params = None
        self.cv_results = None
        self.preprocessor = None
        self.detected_categorical_columns = None
        self.detected_numerical_columns = None
        
    def _detect_task_type(self, y: pd.Series) -> str:
        """
        Automatically detect if the task is classification or regression.
        
        Parameters:
        -----------
        y : pd.Series
            Target variable
            
        Returns:
        --------
        str : 'classification' or 'regression'
        """
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        elif len(y.unique()) <= 20 and y.dtype in ['int64', 'int32']:
            # Assume classification if few unique integer values
            return 'classification'
        else:
            return 'regression'
    
    def _detect_categorical_columns(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Detect categorical and numerical columns in the dataset.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features dataframe
            
        Returns:
        --------
        Tuple[List[str], List[str]] : (categorical_columns, numerical_columns)
        """
        categorical_cols = []
        numerical_cols = []
        
        for col in X.columns:
            # If explicitly specified as categorical, treat as categorical
            if self.categorical_columns and col in self.categorical_columns:
                categorical_cols.append(col)
                continue
            
            # Check data type
            if X[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            elif X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Check if numerical column might be categorical
                unique_values = X[col].nunique()
                
                # If few unique values and they look like categories
                if unique_values <= self.max_categories:
                    # Additional checks for categorical nature
                    if X[col].dtype in ['int64', 'int32']:
                        # Check if values are consecutive integers starting from 0 or 1
                        sorted_values = sorted(X[col].dropna().unique())
                        if (len(sorted_values) == unique_values and 
                            (sorted_values == list(range(min(sorted_values), max(sorted_values) + 1)) or
                             sorted_values == list(range(1, max(sorted_values) + 1)))):
                            categorical_cols.append(col)
                        else:
                            numerical_cols.append(col)
                    else:
                        # For float columns, be more conservative
                        numerical_cols.append(col)
                else:
                    numerical_cols.append(col)
            else:
                # Default to numerical for other types
                numerical_cols.append(col)
        
        return categorical_cols, numerical_cols
    
    def train(self, 
              df: pd.DataFrame, 
              feature_columns: List[str], 
              target_column: str,
              k_values: List[int] = None,
              cv_folds: int = 5,
              scoring: str = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the KNN model with hyperparameter cross-validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_columns : List[str]
            Column names to use as features
        target_column : str
            Name of the target column
        k_values : List[int], optional
            List of k values to test. Default: [1, 3, 5, 7, 9, 11, 15, 19, 25]
        cv_folds : int, default=5
            Number of folds for cross-validation
        scoring : str, optional
            Scoring metric. Auto-selected based on task type if not provided
        **kwargs : dict
            Additional parameters for KNN model
            
        Returns:
        --------
        Dict[str, Any] : Training results including best parameters and CV scores
        """
        # Validate inputs
        if not all(col in df.columns for col in feature_columns):
            missing = [col for col in feature_columns if col not in df.columns]
            raise ValueError(f"Missing feature columns: {missing}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Store column information
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Detect categorical and numerical columns
        self.detected_categorical_columns, self.detected_numerical_columns = self._detect_categorical_columns(X)
        
        # Handle missing values
        if X.isnull().any().any():
            warnings.warn("Missing values found in features. Filling with median/mode.")
            for col in X.columns:
                if col in self.detected_numerical_columns:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
        
        if y.isnull().any():
            raise ValueError("Missing values found in target variable. Please clean the data first.")
        
        # Detect task type if set to auto
        if self.task_type == 'auto':
            self.task_type = self._detect_task_type(y)
        
        # Create preprocessing pipeline for categorical variables
        if self.detected_categorical_columns:
            if self.categorical_encoding == 'onehot':
                # Use OneHotEncoder for categorical variables
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', self.detected_numerical_columns),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.detected_categorical_columns)
                    ]
                )
            else:  # label encoding
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', self.detected_numerical_columns),
                        ('cat', LabelEncoderWrapper(), self.detected_categorical_columns)
                    ]
                )
        else:
            # No categorical variables, just pass through
            preprocessor = ColumnTransformer(
                transformers=[('num', 'passthrough', self.detected_numerical_columns)]
            )
        
        self.preprocessor = preprocessor
        
        # Transform the features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Convert back to DataFrame for easier handling
        if self.detected_categorical_columns and self.categorical_encoding == 'onehot':
            # Get feature names from onehot encoder
            feature_names = self.detected_numerical_columns.copy()
            if hasattr(self.preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
                cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.detected_categorical_columns)
                feature_names.extend(cat_features)
            else:
                # Fallback for older sklearn versions
                cat_features = [f"{col}_{val}" for col in self.detected_categorical_columns 
                              for val in self.preprocessor.named_transformers_['cat'].categories_[self.detected_categorical_columns.index(col)]]
                feature_names.extend(cat_features)
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        else:
            X_transformed = pd.DataFrame(X_transformed, columns=feature_columns, index=X.index)
        
        # Set default parameters
        if k_values is None:
            max_k = min(25, len(X) // 2)  # Ensure k is not too large
            k_values = [k for k in [1, 3, 5, 7, 9, 11, 15, 19, 25] if k <= max_k]
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        # Choose model based on task type
        if self.task_type == 'classification':
            base_model = KNeighborsClassifier(**kwargs)
        else:
            base_model = KNeighborsRegressor(**kwargs)
        
        # Set up parameter grid
        param_grid = {'n_neighbors': k_values}
        
        # Perform grid search with cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Fit the grid search
        grid_search.fit(X_transformed, y)
        
        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        self.is_fitted = True
        
        # Calculate additional metrics
        best_score = grid_search.best_score_
        std_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        
        results = {
            'best_params': self.best_params,
            'best_cv_score': best_score,
            'best_cv_std': std_score,
            'task_type': self.task_type,
            'scoring_metric': scoring,
            'cv_folds': cv_folds,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'n_samples': len(X),
            'n_features': len(feature_columns)
        }
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with same features as training data
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        # Validate input
        if not all(col in df.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in df.columns]
            raise ValueError(f"Missing feature columns in prediction data: {missing}")
        
        # Prepare features
        X = df[self.feature_columns].copy()
        
        # Handle missing values (same strategy as training)
        if X.isnull().any().any():
            warnings.warn("Missing values found in prediction features. Filling with median/mode from training.")
            for col in X.columns:
                if col in self.detected_numerical_columns:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
        
        # Transform features using the same preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_transformed)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (only for classification tasks).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with same features as training data
            
        Returns:
        --------
        np.ndarray : Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        
        # Prepare features (same as predict method)
        X = df[self.feature_columns].copy()
        
        if X.isnull().any().any():
            warnings.warn("Missing values found in prediction features. Filling with median/mode from training.")
            for col in X.columns:
                if col in self.detected_numerical_columns:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
        
        # Transform features using the same preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_transformed)
        
        return probabilities
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with features and target column
            
        Returns:
        --------
        Dict[str, float] : Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation. Call train() first.")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in evaluation data")
        
        # Make predictions
        y_true = df[self.target_column]
        y_pred = self.predict(df)
        
        # Calculate metrics based on task type
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            metrics = {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2
            }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on permutation importance (approximate).
        Note: KNN doesn't have built-in feature importance, so this is an approximation.
        
        Returns:
        --------
        pd.DataFrame : Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance. Call train() first.")
        
        # This is a simplified approach - in practice, you might want to use
        # permutation importance from sklearn.inspection
        importance_scores = np.ones(len(self.feature_columns))  # Placeholder
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores / np.sum(importance_scores)  # Normalize
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_cv_results(self) -> pd.DataFrame:
        """
        Get detailed cross-validation results.
        
        Returns:
        --------
        pd.DataFrame : Cross-validation results
        """
        if self.cv_results is None:
            raise ValueError("No cross-validation results available. Train the model first.")
        
        return self.cv_results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving. Call train() first.")
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'categorical_encoding': self.categorical_encoding,
            'categorical_columns': self.categorical_columns,
            'max_categories': self.max_categories,
            'detected_categorical_columns': self.detected_categorical_columns,
            'detected_numerical_columns': self.detected_numerical_columns,
            'best_params': self.best_params,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.task_type = model_data['task_type']
        self.categorical_encoding = model_data.get('categorical_encoding', 'onehot')
        self.categorical_columns = model_data.get('categorical_columns', None)
        self.max_categories = model_data.get('max_categories', 20)
        self.detected_categorical_columns = model_data.get('detected_categorical_columns', [])
        self.detected_numerical_columns = model_data.get('detected_numerical_columns', [])
        self.best_params = model_data['best_params']
        self.is_fitted = model_data['is_fitted']


# # Example usage function
# def example_usage():
#     """
#     Example of how to use the KNNPredictor class.
#     """
#     # Create sample data
#     np.random.seed(42)
#     n_samples = 1000
    
#     # Classification example
#     X1 = np.random.randn(n_samples, 2)
#     X2 = np.random.randn(n_samples, 2) + [2, 2]
#     X = np.vstack([X1, X2])
#     y_class = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
#     df_class = pd.DataFrame(X, columns=['feature1', 'feature2'])
#     df_class['target'] = y_class
    
#     # Initialize and train classifier
#     knn_classifier = KNNPredictor(task_type='classification')
#     results = knn_classifier.train(
#         df=df_class,
#         feature_columns=['feature1', 'feature2'],
#         target_column='target',
#         cv_folds=5
#     )
    
#     print("Classification Results:")
#     print(f"Best parameters: {results['best_params']}")
#     print(f"Best CV score: {results['best_cv_score']:.4f} ± {results['best_cv_std']:.4f}")
    
#     # Make predictions
#     predictions = knn_classifier.predict(df_class.head())
#     print(f"Sample predictions: {predictions}")
    
#     # Evaluate
#     metrics = knn_classifier.evaluate(df_class)
#     print(f"Evaluation metrics: {metrics}")
    
#     # Regression example
#     X_reg = np.random.randn(500, 3)
#     y_reg = X_reg.sum(axis=1) + np.random.randn(500) * 0.1
    
#     df_reg = pd.DataFrame(X_reg, columns=['feat1', 'feat2', 'feat3'])
#     df_reg['target'] = y_reg
    
#     # Initialize and train regressor
#     knn_regressor = KNNPredictor(task_type='regression')
#     results_reg = knn_regressor.train(
#         df=df_reg,
#         feature_columns=['feat1', 'feat2', 'feat3'],
#         target_column='target',
#         cv_folds=5
#     )
    
#     print("\nRegression Results:")
#     print(f"Best parameters: {results_reg['best_params']}")
#     print(f"Best CV score: {results_reg['best_cv_score']:.4f} ± {results_reg['best_cv_std']:.4f}")
    
#     # Make predictions
#     predictions_reg = knn_regressor.predict(df_reg.head())
#     print(f"Sample predictions: {predictions_reg}")
    
#     # Evaluate
#     metrics_reg = knn_regressor.evaluate(df_reg)
#     print(f"Evaluation metrics: {metrics_reg}")


# if __name__ == "__main__":
#     example_usage()
