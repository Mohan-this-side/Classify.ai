"""
Comprehensive ML Sandbox Tests

This module tests the sandbox executor with realistic machine learning operations
to ensure it can handle the full range of tasks required for our classification system.
"""

import unittest
import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.sandbox_executor import SandboxExecutor


class TestMLSandbox(unittest.TestCase):
    """Test cases for the ML Sandbox with realistic ML operations."""
    
    def setUp(self):
        """Set up the test environment."""
        self.executor = SandboxExecutor(timeout=180)  # Longer timeout for complex ML tasks
        
        # Create a test dataset for classification
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        self.X = np.random.randn(n_samples, 10)
        
        # Generate synthetic target (binary classification)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        
        # Create a more complex dataset with categorical features
        self.df = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, n_samples),
            'numeric2': np.random.normal(5, 2, n_samples),
            'numeric3': np.random.exponential(2, n_samples),
            'category1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'category2': np.random.choice(['low', 'medium', 'high'], n_samples),
            'binary1': np.random.choice([0, 1], n_samples),
            'binary2': np.random.choice([True, False], n_samples),
            'date1': pd.date_range('2020-01-01', periods=n_samples),
        })
        
        # Target variable (influenced by several features)
        self.df['target'] = (
            (self.df['numeric1'] > 0) & 
            (self.df['numeric2'] > 5) | 
            (self.df['category1'] == 'A') | 
            (self.df['binary1'] == 1)
        ).astype(int)
    
    def test_data_cleaning(self):
        """Test data cleaning operations."""
        # Convert DataFrame to CSV string
        df_csv = self.df.to_csv(index=False)
        
        code = f"""
import pandas as pd
import numpy as np
from io import StringIO

# Load the dataset
csv_data = '''{df_csv}'''
df = pd.read_csv(StringIO(csv_data))

# Print initial info
print(f"Initial dataset shape: {df.shape}")
print(f"Initial columns: {df.columns.tolist()}")

# Add some missing values
df.loc[10:20, 'numeric1'] = np.nan
df.loc[30:40, 'numeric2'] = np.nan
df.loc[50:60, 'category1'] = np.nan

# Data cleaning operations
print("\\nPerforming data cleaning...")

# 1. Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values[missing_values > 0]}")

# 2. Impute missing values
df['numeric1'].fillna(df['numeric1'].median(), inplace=True)
df['numeric2'].fillna(df['numeric2'].mean(), inplace=True)
df['category1'].fillna(df['category1'].mode()[0], inplace=True)

# 3. Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# 4. Handle outliers using IQR method for numeric columns
for col in ['numeric1', 'numeric2', 'numeric3']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    print(f"Outliers in {col}: {outliers}")
    
    # Cap outliers
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# 5. Convert date column to datetime features
df['year'] = df['date1'].dt.year
df['month'] = df['date1'].dt.month
df['day'] = df['date1'].dt.day
df.drop('date1', axis=1, inplace=True)

# Print final info
print("\\nCleaned dataset:")
print(f"Final shape: {df.shape}")
print(f"Final columns: {df.columns.tolist()}")
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
"""
        
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Performing data cleaning", result["output"])
        self.assertIn("Missing values after cleaning: 0", result["output"])
    
    def test_exploratory_data_analysis(self):
        """Test EDA operations."""
        # Convert DataFrame to CSV string
        df_csv = self.df.to_csv(index=False)
        
        code = f"""
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_data = '''{df_csv}'''
df = pd.read_csv(StringIO(csv_data))

# Basic EDA
print("Dataset Shape:", df.shape)
print("\\nData Types:\\n", df.dtypes)
print("\\nSummary Statistics:\\n", df.describe())

# Correlation analysis
numeric_cols = df.select_dtypes(include=['number']).columns
corr_matrix = df[numeric_cols].corr()
print("\\nCorrelation Matrix:\\n", corr_matrix)

# Target distribution
target_counts = df['target'].value_counts()
print("\\nTarget Distribution:\\n", target_counts)
print(f"Target Balance: {target_counts[1] / target_counts[0]:.2f}")

# Feature importance based on correlation with target
feature_importance = corr_matrix['target'].sort_values(ascending=False)
print("\\nFeature Importance (Correlation with Target):\\n", feature_importance)

# Categorical feature analysis
for col in ['category1', 'category2']:
    print(f"\n{col} Distribution:")
    print(df[col].value_counts())
    print(f"\n{col} vs Target:")
    print(pd.crosstab(df[col], df['target'], normalize='index'))

# Generate some plots (these won't be displayed but we can check if the code runs)
try:
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('/tmp/correlation_heatmap.png')
    print("Successfully generated correlation heatmap")
    
    # Distribution plots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols[:5]):  # First 5 numeric columns
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('/tmp/distribution_plots.png')
    print("Successfully generated distribution plots")
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Target Distribution')
    plt.savefig('/tmp/target_distribution.png')
    print("Successfully generated target distribution plot")
    
except Exception as e:
    print(f"Error generating plots: {e}")
"""
        
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Dataset Shape:", result["output"])
        self.assertIn("Correlation Matrix:", result["output"])
        self.assertIn("Successfully generated", result["output"])
    
    def test_feature_engineering(self):
        """Test feature engineering operations."""
        # Convert DataFrame to CSV string
        df_csv = self.df.to_csv(index=False)
        
        code = f"""
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
csv_data = '''{df_csv}'''
df = pd.read_csv(StringIO(csv_data))

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Original dataset shape: {X.shape}")

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
date_features = X.select_dtypes(include=['datetime64']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")
print(f"Date features: {date_features}")

# Feature Engineering Pipeline
print("\\nPerforming feature engineering...")

# 1. Create new features
print("Creating new features...")

# Interaction terms for numeric features
if len(numeric_features) >= 2:
    X['numeric1_times_numeric2'] = X['numeric1'] * X['numeric2']
    print("Added interaction term: numeric1_times_numeric2")

# Binning numeric features
if 'numeric1' in numeric_features:
    X['numeric1_binned'] = pd.qcut(X['numeric1'], 4, labels=['q1', 'q2', 'q3', 'q4'])
    print("Added binned feature: numeric1_binned")

# 2. Handle categorical features
print("\\nHandling categorical features...")

# Convert date to year, month, day if date columns exist
if 'date1' in X.columns:
    X['date1'] = pd.to_datetime(X['date1'])
    X['year'] = X['date1'].dt.year
    X['month'] = X['date1'].dt.month
    X['day'] = X['date1'].dt.day
    X['dayofweek'] = X['date1'].dt.dayofweek
    X.drop('date1', axis=1, inplace=True)
    print("Extracted date features: year, month, day, dayofweek")

# 3. Define preprocessing pipeline
print("\\nDefining preprocessing pipeline...")

# Update column lists after feature engineering
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

# Define preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing pipeline
feature_engineering_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=10))  # Select top 10 features
])

# Fit and transform the data
X_transformed = feature_engineering_pipeline.fit_transform(X, y)

print(f"\nTransformed dataset shape: {X_transformed.shape}")
print(f"Number of features after engineering: {X_transformed.shape[1]}")

# Get feature importance scores
if hasattr(feature_engineering_pipeline.named_steps['selector'], 'scores_'):
    scores = feature_engineering_pipeline.named_steps['selector'].scores_
    print("\\nFeature importance scores:")
    for i, score in enumerate(scores):
        print(f"Feature {i}: {score:.4f}")

print("\\nFeature engineering completed successfully!")
"""
        
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Performing feature engineering", result["output"])
        self.assertIn("Feature engineering completed successfully", result["output"])
    
    def test_model_training(self):
        """Test model training operations."""
        # Convert arrays to CSV string
        X_df = pd.DataFrame(self.X)
        y_df = pd.DataFrame(self.y, columns=['target'])
        data_df = pd.concat([X_df, y_df], axis=1)
        data_csv = data_df.to_csv(index=False)
        
        code = f"""
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time

# Load the dataset
csv_data = '''{data_csv}'''
df = pd.read_csv(StringIO(csv_data))

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define models to try
models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
}

# Train and evaluate models
results = {}

print("\\nTraining models:")
for name, pipeline in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    except:
        roc_auc = "N/A"
    
    # Store results
    train_time = time.time() - start_time
    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'training_time': train_time
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc}, Training time: {train_time:.2f}s")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Find the best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
accuracy = best_model[1]['accuracy']
print(f"\nBest model: {best_model[0]} with accuracy: {accuracy:.4f}")

# Hyperparameter tuning for the best model
print(f"\nPerforming hyperparameter tuning for {best_model[0]}...")

if best_model[0] == 'LogisticRegression':
    param_grid = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__solver': ['liblinear', 'lbfgs']
    }
elif best_model[0] == 'RandomForest':
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10, 20]
    }
else:  # GradientBoosting
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__learning_rate': [0.01, 0.1]
    }

# Create grid search
grid_search = GridSearchCV(
    models[best_model[0]],
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Train with grid search
grid_search.fit(X_train, y_train)

# Get best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred = grid_search.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final test accuracy: {final_accuracy:.4f}")

print("\\nModel training completed successfully!")
"""
        
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Training models", result["output"])
        self.assertIn("Model training completed successfully", result["output"])
    
    def test_model_evaluation(self):
        """Test model evaluation operations."""
        # Convert arrays to CSV string
        X_df = pd.DataFrame(self.X)
        y_df = pd.DataFrame(self.y, columns=['target'])
        data_df = pd.concat([X_df, y_df], axis=1)
        data_csv = data_df.to_csv(index=False)
        
        code = f"""
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_data = '''{data_csv}'''
df = pd.read_csv(StringIO(csv_data))

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\\nConfusion Matrix:")
print(cm)

# Classification Report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
    importances = pipeline.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\\nFeature Ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Generate evaluation plots
try:
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('/tmp/roc_curve.png')
    print("\\nROC curve plot saved")
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('/tmp/confusion_matrix.png')
    print("Confusion matrix plot saved")
    
    # Feature Importance Plot
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.savefig('/tmp/feature_importance.png')
        print("Feature importance plot saved")
    
except Exception as e:
    print(f"Error generating plots: {e}")

print("\\nModel evaluation completed successfully!")
"""
        
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Model Evaluation Metrics", result["output"])
        self.assertIn("Model evaluation completed successfully", result["output"])
    
    def test_resource_intensive_operations(self):
        """Test resource-intensive operations to verify resource limits."""
        code = f"""
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import psutil

# Monitor resources
print(f"CPU count: {psutil.cpu_count()}")
print(f"Available memory: {psutil.virtual_memory().available / (1024 * 1024):.2f} MB")

# Generate a large dataset
print("\\nGenerating large dataset...")
X, y = make_classification(
    n_samples=50000,
    n_features=100,
    n_informative=20,
    n_redundant=10,
    n_classes=2,
    random_state=42
)
print(f"Dataset shape: {X.shape}")

# Train a complex model
print("\\nTraining complex model...")
start_time = time.time()

# Create and train a large random forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,  # Use all available cores
    random_state=42
)

rf.fit(X, y)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")
print(f"Model accuracy on training data: {rf.score(X, y):.4f}")

# Check memory usage after training
print(f"Memory usage after training: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# Perform a memory-intensive operation
print("\\nPerforming memory-intensive operation...")
start_time = time.time()

# Create large matrices and perform operations
matrix_size = 5000
A = np.random.rand(matrix_size, matrix_size)
B = np.random.rand(matrix_size, matrix_size)
C = np.dot(A, B)  # Matrix multiplication

operation_time = time.time() - start_time
print(f"Operation completed in {operation_time:.2f} seconds")
print(f"Result matrix shape: {C.shape}")
print(f"Memory usage after operation: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

print("\\nResource-intensive operations completed successfully!")
"""
        
        result = self.executor.execute_code(code)
        # The test might pass or fail depending on the resource limits
        # We're mainly checking if the sandbox can handle these operations
        status = result['status']
        output = result['output']
        print(f"Resource test status: {status}")
        print(f"Resource test output: {output}")
        
        # If it succeeds, check the output
        if result["status"] == "SUCCESS":
            self.assertIn("Resource-intensive operations completed successfully", result["output"])
        # If it fails due to resource limits, that's also acceptable
        else:
            self.assertIn(("FAILED", "TIMEOUT"), result["status"])


if __name__ == "__main__":
    unittest.main()
