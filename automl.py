"""
AutoML Core Engine - Supervised Learning
This is the core ML logic that can be used standalone or with any UI framework.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            r2_score, mean_squared_error, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import pickle


class AutoMLEngine:
    """
    Core AutoML Engine for Supervised Learning.
    Can be used programmatically without any user interaction.
    """
    
    def __init__(self, dataframe=None, csv_path=None):
        """
        Initialize with either a DataFrame or CSV path.
        
        Args:
            dataframe: pandas DataFrame (optional)
            csv_path: path to CSV file (optional)
        """
        if dataframe is not None:
            self.df = dataframe
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None
        self.feature_columns = None
        self.problem_type = None
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
    
    def detect_target_column(self):
        """
        Intelligently detect the most likely target column.
        
        Returns:
            tuple: (best_target, list of top 3 candidates with scores)
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        target_scores = {}
        
        for col in self.df.columns:
            score = 0
            
            # Common target keywords
            target_keywords = ['target', 'label', 'class', 'output', 'y', 'result', 
                              'outcome', 'prediction', 'category', 'type', 'species',
                              'diagnosis', 'status', 'price', 'value', 'sales', 'revenue',
                              'survived', 'fraud', 'churn']
            
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword in col_lower:
                    score += 10
                    break
            
            # Position bonus
            if col == self.df.columns[-1]:
                score += 5
            
            # Numeric analysis
            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.05 and self.df[col].nunique() < 20:
                    score += 8
                elif unique_ratio > 0.5:
                    score += 6
            else:
                unique_count = self.df[col].nunique()
                if 2 <= unique_count <= 20:
                    score += 7
                if unique_count == 2:
                    score += 3
            
            target_scores[col] = score
        
        best_target = max(target_scores, key=target_scores.get)
        sorted_scores = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [(col, score) for col, score in sorted_scores[:3]]
        
        return best_target, candidates
    
    def auto_select_features(self, target_column):
        """
        Automatically select useful features by removing ID/Name columns.
        
        Args:
            target_column: name of target column to exclude
            
        Returns:
            tuple: (selected_features, removed_features)
        """
        available_features = [col for col in self.df.columns if col != target_column]
        
        selected = []
        removed = []
        
        for col in available_features:
            unique_ratio = self.df[col].nunique() / len(self.df)
            # Remove if >50% unique and categorical (likely ID/Name)
            if unique_ratio > 0.5 and self.df[col].dtype == 'object':
                removed.append(col)
            else:
                selected.append(col)
        
        return selected, removed
    
    def prepare_data(self, target_column, feature_columns):
        """
        Prepare data for training.
        
        Args:
            target_column: name of target column
            feature_columns: list of feature column names
        """
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        # Separate features and target
        self.y = self.df[target_column].copy()
        self.X = self.df[feature_columns].copy()
        
        # Detect problem type
        self._detect_problem_type()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Encode categorical features
        self._encode_features()
        
        # Encode target if classification
        if 'Classification' in self.problem_type:
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)
        
        # Split data
        if 'Classification' in self.problem_type:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def _detect_problem_type(self):
        """Detect if problem is regression or classification."""
        unique_values = self.y.nunique()
        total_samples = len(self.y)
        
        if pd.api.types.is_numeric_dtype(self.y):
            if unique_values < 20 and unique_values / total_samples < 0.05:
                self.problem_type = 'Binary Classification' if unique_values == 2 else 'Multiclass Classification'
            else:
                self.problem_type = 'Regression'
        else:
            self.problem_type = 'Binary Classification' if unique_values == 2 else 'Multiclass Classification'
    
    def _handle_missing_values(self):
        """Handle missing values in dataset."""
        for col in self.X.columns:
            if self.X[col].isnull().sum() > 0:
                if self.X[col].dtype in ['int64', 'float64']:
                    self.X[col].fillna(self.X[col].median(), inplace=True)
                else:
                    self.X[col].fillna(self.X[col].mode()[0], inplace=True)
    
    def _encode_features(self):
        """Encode categorical features."""
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le
    
    def train_all_models(self, progress_callback=None):
        """
        Train all supervised ML models.
        
        Args:
            progress_callback: optional function(current, total, model_name) for progress updates
            
        Returns:
            list of results dictionaries
        """
        models = self._get_models()
        self.results = []
        
        total = len(models)
        
        for i, (name, model) in enumerate(models.items(), 1):
            try:
                if progress_callback:
                    progress_callback(i, total, name)
                
                # Train
                model.fit(self.X_train, self.y_train)
                
                # Predict
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                if 'Classification' in self.problem_type:
                    metrics = {
                        'Model': name,
                        'Accuracy': accuracy_score(self.y_test, y_pred),
                        'Precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'F1_Score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'Model_Object': model
                    }
                else:
                    metrics = {
                        'Model': name,
                        'R2_Score': r2_score(self.y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        'MAE': mean_absolute_error(self.y_test, y_pred),
                        'Model_Object': model
                    }
                
                self.results.append(metrics)
                
            except Exception as e:
                print(f"Warning: {name} failed - {str(e)}")
        
        # Sort by primary metric
        if 'Classification' in self.problem_type:
            self.results.sort(key=lambda x: x['Accuracy'], reverse=True)
        else:
            self.results.sort(key=lambda x: x['R2_Score'], reverse=True)
        
        # Set best model
        if self.results:
            self.best_model = self.results[0]['Model_Object']
            self.best_model_name = self.results[0]['Model']
        
        return self.results
    
    def _get_models(self):
        """Get models based on problem type."""
        if 'Classification' in self.problem_type:
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
            }
        else:
            return {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'Neural Network': MLPRegressor(max_iter=1000, random_state=42)
            }
    
    def predict(self, input_data):
        """
        Make prediction on new data.
        
        Args:
            input_data: dict with feature values
            
        Returns:
            prediction value
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        # Convert to dataframe
        input_df = pd.DataFrame([input_data])
        
        # Ensure correct columns
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[self.feature_columns]
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                input_value = str(input_df[col].iloc[0])
                if input_value not in encoder.classes_:
                    # Use most common class if value not seen
                    input_df[col] = encoder.transform([encoder.classes_[0]])[0]
                else:
                    input_df[col] = encoder.transform([input_value])[0]
        
        # Scale
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.best_model.predict(input_scaled)
        
        # Decode if classification
        if self.target_encoder:
            prediction = self.target_encoder.inverse_transform(prediction.astype(int))
        
        return prediction[0]
    
    def get_results_dataframe(self):
        """
        Get results as a formatted DataFrame.
        
        Returns:
            pandas DataFrame with model comparison
        """
        if not self.results:
            return None
        
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'Model_Object'}
            for r in self.results
        ])
        
        return results_df
    
    def save_model(self, filepath='best_model.pkl'):
        """
        Save the best model and all necessary data.
        
        Args:
            filepath: path to save the model
        """
        if self.best_model is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns,
            'problem_type': self.problem_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(filepath):
        """
        Load a saved model.
        
        Args:
            filepath: path to the saved model
            
        Returns:
            AutoMLEngine instance with loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        engine = AutoMLEngine()
        engine.best_model = model_data['model']
        engine.best_model_name = model_data['model_name']
        engine.scaler = model_data['scaler']
        engine.label_encoders = model_data['label_encoders']
        engine.target_encoder = model_data['target_encoder']
        engine.feature_columns = model_data['feature_columns']
        engine.problem_type = model_data['problem_type']
        
        return engine


# Example standalone usage
if __name__ == "__main__":
    print("AutoML Engine - Standalone Mode")
    print("=" * 70)
    
    # Example usage
    print("\nThis is a library file. Import it in your code:")
    print("\nfrom automl import AutoMLEngine")
    print("\n# Load data")
    print("engine = AutoMLEngine(csv_path='your_data.csv')")
    print("\n# Detect target")
    print("target, candidates = engine.detect_target_column()")
    print("\n# Auto-select features")
    print("features, removed = engine.auto_select_features(target)")
    print("\n# Prepare data")
    print("engine.prepare_data(target, features)")
    print("\n# Train models")
    print("results = engine.train_all_models()")
    print("\n# Make predictions")
    print("prediction = engine.predict({'feature1': value1, ...})")








































































































































































