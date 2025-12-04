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









































































































































































# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
# import warnings
# warnings.filterwarnings('ignore')

# # Classification Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier

# # Regression Models
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor

# import pickle


# class SupervisedMLAutomation:
#     """
#     Complete Supervised ML Automation System:
#     - Auto-detects target column
#     - Trains all supervised ML algorithms
#     - Compares and recommends best model
#     - Provides prediction interface
#     """
    
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path
#         self.df = None
#         self.X = None
#         self.y = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.target_column = None
#         self.problem_type = None
#         self.feature_columns = None
#         self.results = []
#         self.best_model = None
#         self.best_model_name = None
#         self.scaler = StandardScaler()
#         self.label_encoders = {}
#         self.target_encoder = None
        
#     def load_and_prepare_data(self):
#         """Load dataset and auto-detect target column."""
#         print("="*70)
#         print("STEP 1: LOADING AND PREPARING DATA")
#         print("="*70)
        
#         # Load data
#         self.df = pd.read_csv(self.dataset_path)
#         print(f"\n✓ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
#         # Show all columns to user
#         print("\n📋 Available columns in your dataset:")
#         for i, col in enumerate(self.df.columns, 1):
#             print(f"  {i}. {col}")
        
#         # Auto-detect target column intelligently
#         detected_target = self._detect_target_column()
#         print(f"\n🎯 Auto-detected target column: '{detected_target}'")
        
#         # Ask user for confirmation and selection
#         self.target_column = self._confirm_target_column(detected_target)
        
#         # Ask user which features to use
#         self._select_feature_columns()
        
#         # Separate features and target
#         self.y = self.df[self.target_column]
#         self.X = self.df[self.feature_columns]
        
#         print(f"\n✓ Using {len(self.feature_columns)} features for training")
        
#         # Detect problem type
#         self._detect_problem_type()
        
#         # Handle missing values
#         self._handle_missing_values()
        
#         # Encode categorical features
#         self._encode_features()
        
#         # Encode target if classification
#         if self.problem_type in ['binary_classification', 'multiclass_classification']:
#             self.target_encoder = LabelEncoder()
#             self.y = self.target_encoder.fit_transform(self.y)
        
#         # Split data
#         test_size = 0.2
#         if self.problem_type in ['binary_classification', 'multiclass_classification']:
#             self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#                 self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
#             )
#         else:
#             self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#                 self.X, self.y, test_size=test_size, random_state=42
#             )
        
#         # Scale features
#         self.X_train = self.scaler.fit_transform(self.X_train)
#         self.X_test = self.scaler.transform(self.X_test)
        
#         print(f"✓ Train set: {self.X_train.shape[0]} samples")
#         print(f"✓ Test set: {self.X_test.shape[0]} samples")
    
#     def _detect_target_column(self):
#         """
#         Intelligently detect the target column based on dataset characteristics.
#         Uses multiple heuristics to identify the most likely target column.
#         """
#         print("\n🔍 Analyzing columns to detect target...")
        
#         target_scores = {}
        
#         for col in self.df.columns:
#             score = 0
            
#             # Check 1: Common target column names
#             target_keywords = ['target', 'label', 'class', 'output', 'y', 'result', 
#                               'outcome', 'prediction', 'category', 'type', 'species',
#                               'diagnosis', 'status', 'price', 'value', 'sales', 'revenue']
            
#             col_lower = col.lower()
#             for keyword in target_keywords:
#                 if keyword in col_lower:
#                     score += 10
#                     break
            
#             # Check 2: Position (last column is common for target)
#             if col == self.df.columns[-1]:
#                 score += 5
            
#             # Check 3: For numeric columns - check if it's the dependent variable
#             if pd.api.types.is_numeric_dtype(self.df[col]):
#                 # Check uniqueness ratio
#                 unique_ratio = self.df[col].nunique() / len(self.df)
                
#                 # Classification targets typically have low unique values
#                 if unique_ratio < 0.05 and self.df[col].nunique() < 20:
#                     score += 8
                
#                 # Regression targets typically have high unique values
#                 elif unique_ratio > 0.5:
#                     score += 6
            
#             # Check 4: For categorical columns
#             else:
#                 unique_count = self.df[col].nunique()
                
#                 # Good targets have reasonable number of classes (2-20)
#                 if 2 <= unique_count <= 20:
#                     score += 7
                
#                 # Binary targets are very common
#                 if unique_count == 2:
#                     score += 3
            
#             # Check 5: Correlation with other features (for numeric)
#             if pd.api.types.is_numeric_dtype(self.df[col]):
#                 numeric_cols = self.df.select_dtypes(include=[np.number]).columns
#                 if len(numeric_cols) > 1:
#                     try:
#                         # Target often has moderate correlation with features
#                         correlations = self.df[numeric_cols].corr()[col].drop(col)
#                         avg_correlation = abs(correlations).mean()
                        
#                         if 0.2 < avg_correlation < 0.8:
#                             score += 4
#                     except:
#                         pass
            
#             target_scores[col] = score
        
#         # Get column with highest score
#         best_target = max(target_scores, key=target_scores.get)
        
#         # Show top 3 candidates
#         sorted_scores = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
#         print(f"   Top candidates:")
#         for col, score in sorted_scores[:3]:
#             print(f"   - {col}: score = {score}")
        
#         return best_target
    
#     def _confirm_target_column(self, detected_target):
#         """Ask user to confirm or change the target column."""
#         print("\n" + "="*70)
#         print("TARGET COLUMN SELECTION")
#         print("="*70)
        
#         while True:
#             print(f"\n👉 Is '{detected_target}' the correct target column?")
#             choice = input("   Enter 'y' for YES or 'n' to select different column: ").lower().strip()
            
#             if choice == 'y' or choice == 'yes':
#                 print(f"✓ Confirmed target column: '{detected_target}'")
#                 return detected_target
            
#             elif choice == 'n' or choice == 'no':
#                 print("\n📝 Please select the correct target column:")
#                 for i, col in enumerate(self.df.columns, 1):
#                     print(f"  {i}. {col}")
                
#                 try:
#                     col_num = int(input("\nEnter column number: ").strip())
#                     if 1 <= col_num <= len(self.df.columns):
#                         selected_col = self.df.columns[col_num - 1]
#                         print(f"✓ Target column set to: '{selected_col}'")
#                         return selected_col
#                     else:
#                         print(f"❌ Invalid number. Please enter between 1 and {len(self.df.columns)}")
#                 except ValueError:
#                     print("❌ Invalid input. Please enter a number.")
            
#             else:
#                 print("❌ Please enter 'y' or 'n'")
    
#     def _select_feature_columns(self):
#         """Ask user to select which columns to use as features."""
#         print("\n" + "="*70)
#         print("FEATURE COLUMN SELECTION")
#         print("="*70)
        
#         # Get all columns except target
#         available_features = [col for col in self.df.columns if col != self.target_column]
        
#         print(f"\nAvailable feature columns (excluding target '{self.target_column}'):")
#         for i, col in enumerate(available_features, 1):
#             # Show some info about each column
#             dtype = self.df[col].dtype
#             unique_count = self.df[col].nunique()
#             print(f"  {i}. {col:<20} (type: {dtype}, unique: {unique_count})")
        
#         print("\n" + "-"*70)
#         print("OPTIONS:")
#         print("  1. Type 'all' to use ALL features")
#         print("  2. Type 'auto' to auto-remove ID/Name columns")
#         print("  3. Type column numbers separated by commas (e.g., 1,3,5,7)")
#         print("-"*70)
        
#         while True:
#             choice = input("\nYour choice: ").strip().lower()
            
#             if choice == 'all':
#                 self.feature_columns = available_features
#                 print(f"✓ Using ALL {len(self.feature_columns)} features")
#                 break
            
#             elif choice == 'auto':
#                 # Auto-remove high-cardinality columns
#                 selected = []
#                 removed = []
#                 for col in available_features:
#                     unique_ratio = self.df[col].nunique() / len(self.df)
#                     # Remove if >50% unique and categorical (likely ID/Name)
#                     if unique_ratio > 0.5 and self.df[col].dtype == 'object':
#                         removed.append(col)
#                     else:
#                         selected.append(col)
                
#                 self.feature_columns = selected
#                 print(f"\n✓ Auto-selected {len(self.feature_columns)} features")
#                 if removed:
#                     print(f"🗑️  Auto-removed: {removed}")
#                 break
            
#             else:
#                 # Manual selection by numbers
#                 try:
#                     numbers = [int(x.strip()) for x in choice.split(',')]
#                     selected = []
#                     for num in numbers:
#                         if 1 <= num <= len(available_features):
#                             selected.append(available_features[num - 1])
#                         else:
#                             print(f"❌ Invalid number: {num}")
#                             break
#                     else:
#                         self.feature_columns = selected
#                         print(f"✓ Selected {len(self.feature_columns)} features:")
#                         for col in self.feature_columns:
#                             print(f"   - {col}")
#                         break
#                 except ValueError:
#                     print("❌ Invalid input. Try 'all', 'auto', or column numbers like 1,2,3")
    
#     def _detect_problem_type(self):
#         """Automatically detect if problem is regression or classification."""
#         unique_values = self.y.nunique()
#         total_samples = len(self.y)
        
#         if pd.api.types.is_numeric_dtype(self.y):
#             if unique_values < 20 and unique_values / total_samples < 0.05:
#                 self.problem_type = 'binary_classification' if unique_values == 2 else 'multiclass_classification'
#             else:
#                 self.problem_type = 'regression'
#         else:
#             self.problem_type = 'binary_classification' if unique_values == 2 else 'multiclass_classification'
        
#         print(f"\n🎯 Problem Type: {self.problem_type.replace('_', ' ').title()}")
#         print(f"   Target unique values: {unique_values}")
    
#     def _handle_missing_values(self):
#         """Handle missing values in dataset."""
#         missing_count = self.X.isnull().sum().sum()
#         if missing_count > 0:
#             print(f"\n⚠️  Found {missing_count} missing values - filling with median/mode")
#             for col in self.X.columns:
#                 if self.X[col].dtype in ['int64', 'float64']:
#                     self.X[col].fillna(self.X[col].median(), inplace=True)
#                 else:
#                     self.X[col].fillna(self.X[col].mode()[0], inplace=True)
    
#     def _encode_features(self):
#         """Encode categorical features."""
#         categorical_cols = self.X.select_dtypes(include=['object']).columns
#         if len(categorical_cols) > 0:
#             print(f"\n🔄 Encoding {len(categorical_cols)} categorical features")
#             for col in categorical_cols:
#                 le = LabelEncoder()
#                 self.X[col] = le.fit_transform(self.X[col].astype(str))
#                 self.label_encoders[col] = le
    
#     def train_all_models(self):
#         """Train all supervised ML algorithms and compare performance."""
#         print("\n" + "="*70)
#         print("STEP 2: TRAINING ALL SUPERVISED ML MODELS")
#         print("="*70)
        
#         if self.problem_type == 'regression':
#             models = self._get_regression_models()
#         else:
#             models = self._get_classification_models()
        
#         print(f"\n🚀 Training {len(models)} different models...\n")
        
#         for name, model in models.items():
#             try:
#                 print(f"   Training {name}...", end=" ")
                
#                 # Train model
#                 model.fit(self.X_train, self.y_train)
                
#                 # Predict
#                 y_pred = model.predict(self.X_test)
                
#                 # Calculate metrics
#                 if self.problem_type == 'regression':
#                     metrics = self._calculate_regression_metrics(y_pred)
#                 else:
#                     metrics = self._calculate_classification_metrics(y_pred)
                
#                 # Store results
#                 self.results.append({
#                     'Model': name,
#                     'Model_Object': model,
#                     **metrics
#                 })
                
#                 print("✓")
                
#             except Exception as e:
#                 print(f"✗ (Error: {str(e)[:50]})")
        
#         # Sort by primary metric
#         if self.problem_type == 'regression':
#             self.results.sort(key=lambda x: x['R2_Score'], reverse=True)
#         else:
#             self.results.sort(key=lambda x: x['Accuracy'], reverse=True)
        
#         # Set best model
#         if self.results:
#             self.best_model = self.results[0]['Model_Object']
#             self.best_model_name = self.results[0]['Model']
    
#     def _get_classification_models(self):
#         """Get all classification models."""
#         return {
#             'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#             'Decision Tree': DecisionTreeClassifier(random_state=42),
#             'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#             'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
#             'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
#             'Support Vector Machine': SVC(random_state=42),
#             'K-Nearest Neighbors': KNeighborsClassifier(),
#             'Naive Bayes': GaussianNB(),
#             'Neural Network (MLP)': MLPClassifier(max_iter=1000, random_state=42)
#         }
    
#     def _get_regression_models(self):
#         """Get all regression models."""
#         return {
#             'Linear Regression': LinearRegression(),
#             'Ridge Regression': Ridge(random_state=42),
#             'Lasso Regression': Lasso(random_state=42),
#             'ElasticNet': ElasticNet(random_state=42),
#             'Decision Tree': DecisionTreeRegressor(random_state=42),
#             'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#             'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#             'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
#             'Support Vector Machine': SVR(),
#             'K-Nearest Neighbors': KNeighborsRegressor(),
#             'Neural Network (MLP)': MLPRegressor(max_iter=1000, random_state=42)
#         }
    
#     def _calculate_classification_metrics(self, y_pred):
#         """Calculate classification metrics."""
#         return {
#             'Accuracy': accuracy_score(self.y_test, y_pred),
#             'Precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
#             'Recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
#             'F1_Score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
#         }
    
#     def _calculate_regression_metrics(self, y_pred):
#         """Calculate regression metrics."""
#         return {
#             'R2_Score': r2_score(self.y_test, y_pred),
#             'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
#             'MAE': mean_absolute_error(self.y_test, y_pred)
#         }
    
#     def show_results(self):
#         """Display comparison of all models."""
#         print("\n" + "="*70)
#         print("STEP 3: MODEL COMPARISON & RESULTS")
#         print("="*70)
        
#         if not self.results:
#             print("No results available.")
#             return
        
#         # Create results dataframe
#         results_df = pd.DataFrame(self.results)
#         results_df = results_df.drop(columns=['Model_Object'])
        
#         # Format numeric columns
#         for col in results_df.columns:
#             if col != 'Model':
#                 results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}")
        
#         print("\n📊 Model Performance Comparison:\n")
#         print(results_df.to_string(index=False))
        
#         print("\n" + "="*70)
#         print(f"🏆 BEST MODEL: {self.best_model_name}")
#         print("="*70)
        
#         best_result = self.results[0]
#         print(f"\n✓ Model: {best_result['Model']}")
        
#         if self.problem_type == 'regression':
#             print(f"✓ R² Score: {best_result['R2_Score']:.4f}")
#             print(f"✓ RMSE: {best_result['RMSE']:.4f}")
#             print(f"✓ MAE: {best_result['MAE']:.4f}")
#         else:
#             print(f"✓ Accuracy: {best_result['Accuracy']:.4f}")
#             print(f"✓ Precision: {best_result['Precision']:.4f}")
#             print(f"✓ Recall: {best_result['Recall']:.4f}")
#             print(f"✓ F1 Score: {best_result['F1_Score']:.4f}")
    
#     def save_best_model(self, filename='best_model.pkl'):
#         """Save the best model to disk."""
#         if self.best_model is None:
#             print("No model to save. Train models first.")
#             return
        
#         model_data = {
#             'model': self.best_model,
#             'model_name': self.best_model_name,
#             'scaler': self.scaler,
#             'label_encoders': self.label_encoders,
#             'target_encoder': self.target_encoder,
#             'feature_columns': self.feature_columns,
#             'problem_type': self.problem_type
#         }
        
#         with open(filename, 'wb') as f:
#             pickle.dump(model_data, f)
        
#         print(f"\n✓ Best model saved as '{filename}'")
    
#     def predict_new_data(self, input_data):
#         """
#         Make predictions on new data.
        
#         Args:
#             input_data: dict or DataFrame with feature values
        
#         Returns:
#             prediction result
#         """
#         print("\n" + "="*70)
#         print("MAKING PREDICTION")
#         print("="*70)
        
#         if self.best_model is None:
#             print("No trained model available. Train models first.")
#             return None
        
#         # Convert dict to dataframe if needed
#         if isinstance(input_data, dict):
#             input_df = pd.DataFrame([input_data])
#         else:
#             input_df = input_data.copy()
        
#         # Keep only the columns that were used in training
#         available_cols = [col for col in self.feature_columns if col in input_df.columns]
#         input_df = input_df[available_cols]
        
#         # Handle missing columns
#         for col in self.feature_columns:
#             if col not in input_df.columns:
#                 print(f"⚠️  Warning: Column '{col}' not provided, using default value")
#                 # Use median for numeric, mode for categorical
#                 if col in self.X.columns:
#                     if pd.api.types.is_numeric_dtype(self.X[col]):
#                         input_df[col] = self.X[col].median()
#                     else:
#                         input_df[col] = self.X[col].mode()[0]
        
#         # Reorder columns to match training
#         input_df = input_df[self.feature_columns]
        
#         # Encode categorical features (handle unknown values)
#         for col, encoder in self.label_encoders.items():
#             if col in input_df.columns:
#                 # Check if value is in encoder's classes
#                 input_value = str(input_df[col].iloc[0])
#                 if input_value not in encoder.classes_:
#                     print(f"⚠️  Warning: '{input_value}' in column '{col}' not seen during training. Using most common value.")
#                     # Use the most common class
#                     input_df[col] = encoder.classes_[0]
#                 else:
#                     input_df[col] = encoder.transform(input_df[col].astype(str))
        
#         # Scale features
#         input_scaled = self.scaler.transform(input_df)
        
#         # Make prediction
#         prediction = self.best_model.predict(input_scaled)
        
#         # Decode prediction if classification
#         if self.target_encoder:
#             prediction = self.target_encoder.inverse_transform(prediction.astype(int))
        
#         print(f"\n✓ Using model: {self.best_model_name}")
#         print(f"✓ Prediction: {prediction[0]}")
        
#         return prediction[0]
    
#     def run_complete_pipeline(self):
#         """Run the complete ML automation pipeline."""
#         print("\n" + "🤖 SUPERVISED ML AUTOMATION SYSTEM 🤖".center(70))
        
#         # Step 1: Load and prepare data
#         self.load_and_prepare_data()
        
#         # Step 2: Train all models
#         self.train_all_models()
        
#         # Step 3: Show results
#         self.show_results()
        
#         # Step 4: Save best model
#         self.save_best_model()
        
#         print("\n" + "="*70)
#         print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
#         print("="*70)
#         print("\nYou can now use predict_new_data() to make predictions!")


# # =============================================================================
# # EXAMPLE USAGE
# # =============================================================================

# if __name__ == "__main__":
    
#     # Initialize the system
#     ml_system = SupervisedMLAutomation(r'C:\Users\HP\Documents\VSC\Python\titanic.csv')
    
#     # Run complete pipeline (load, train, compare, save)
#     ml_system.run_complete_pipeline()
    
#     # Make predictions on new data
#     print("\n" + "="*70)
#     print("PREDICTION EXAMPLE")
#     print("="*70)
    
#     # Example: Create sample input data (adjust based on your dataset features)
#     new_data = {
#     'Pclass': 3,           # Ticket class
#     'Sex': 'male',         # Gender  
#     'Age': 25,
#     'Siblings/Spouses Aboard': 0,
#     'Parents/Children Aboard': 0,   
#     'Fare': 7.25
                      
    
#     }
    
#     prediction = ml_system.predict_new_data(new_data)
    
#     print("\n" + "="*70)
#     print("💡 TIP: To make predictions, create a dictionary with your feature")
#     print("    values and call ml_system.predict_new_data(your_data)")
#     print("="*70)