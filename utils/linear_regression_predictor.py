import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
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


class LinearRegressionPredictor:
    """
    A Linear Regression-based predictor with automatic preprocessing.
    Supports both classification and regression tasks.
    """
    
    def __init__(self, task_type: str = 'auto', random_state: int = 42, categorical_encoding: str = 'onehot', 
                 categorical_columns: List[str] = None, max_categories: int = 20):
        """
        Initialize the Linear Regression predictor.
        
        Parameters:
        -----------
        task_type : str
            Type of task: 'regression', 'classification', or 'auto'
        random_state : int
            Random state for reproducibility
        categorical_encoding : str
            Method for encoding categorical variables: 'onehot' or 'label'
        categorical_columns : List[str]
            List of categorical column names. If None, will be auto-detected
        max_categories : int
            Maximum number of categories for a column to be considered categorical
        """
        self.task_type = task_type
        self.random_state = random_state
        self.categorical_encoding = categorical_encoding
        self.categorical_columns = categorical_columns
        self.max_categories = max_categories
        
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = None
        self.is_fitted = False
        
    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns in the dataframe"""
        categorical_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (few unique values)
                unique_vals = df[col].nunique()
                if unique_vals <= self.max_categories and unique_vals < len(df) * 0.5:
                    categorical_cols.append(col)
        
        return categorical_cols
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        if self.categorical_columns is None:
            self.categorical_columns = self._detect_categorical_columns(X)
        
        # Get numerical columns
        numerical_columns = [col for col in X.columns if col not in self.categorical_columns]
        
        # Create transformers
        transformers = []
        
        if numerical_columns:
            transformers.append(('num', StandardScaler(), numerical_columns))
        
        if self.categorical_columns:
            if self.categorical_encoding == 'onehot':
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_columns))
            else:  # label encoding
                transformers.append(('cat', LabelEncoderWrapper(), self.categorical_columns))
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine task type based on target variable"""
        if self.task_type == 'auto':
            if y.dtype == 'object' or y.dtype.name == 'category':
                return 'classification'
            elif y.nunique() <= 20 and y.dtype in ['int64', 'float64']:
                # Could be classification with numeric labels
                return 'classification'
            else:
                return 'regression'
        return self.task_type
    
    def train(self, df: pd.DataFrame, feature_columns: List[str], target_column: str, 
              cv_folds: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Train the Linear Regression model with cross-validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe
        feature_columns : List[str]
            List of feature column names
        target_column : str
            Name of the target column
        cv_folds : int
            Number of cross-validation folds
        **kwargs
            Additional parameters for the model
            
        Returns:
        --------
        Dict[str, Any]
            Training results including CV scores
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Determine task type
        task_type = self._determine_task_type(y)
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)
        
        # Initialize model based on task type
        if task_type == 'regression':
            self.model = LinearRegression(**kwargs)
        else:  # classification
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000, **kwargs)
        
        # Fit preprocessor
        X_processed = self.preprocessor.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_processed, y, cv=cv_folds, 
                                   scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy')
        
        # Train final model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        # Calculate training score
        train_score = self.model.score(X_processed, y)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_score': train_score,
            'task_type': task_type,
            'n_features': X_processed.shape[1]
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure we have the same columns as training
        X = X[self.feature_columns].copy()
        
        # Transform features
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions (for classification only).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
            
        Returns:
        --------
        np.ndarray
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Probability predictions only available for classification models")
        
        # Ensure we have the same columns as training
        X = X[self.feature_columns].copy()
        
        # Transform features
        X_processed = self.preprocessor.transform(X)
        
        # Make probability predictions
        return self.model.predict_proba(X_processed)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients).
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'coef_'):
            # Get feature names after preprocessing
            feature_names = []
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback for older sklearn versions
                feature_names = [f'feature_{i}' for i in range(len(self.model.coef_.flatten()))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': self.model.coef_.flatten()
            })
            
            # Sort by absolute coefficient value
            importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            
            return importance_df
        else:
            raise ValueError("Feature importance not available for this model type")
