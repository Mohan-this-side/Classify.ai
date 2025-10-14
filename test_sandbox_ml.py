#!/usr/bin/env python3
"""
Simple ML sandbox test to verify functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.sandbox_executor import SandboxExecutor

def test_basic_ml_operations():
    """Test basic ML operations in the sandbox"""
    print("🧪 Testing ML Sandbox with Basic Operations")
    print("=" * 50)
    
    executor = SandboxExecutor()
    
    # Test 1: Basic Python and NumPy
    print("\n1. Testing basic Python and NumPy...")
    code1 = """
import numpy as np
import pandas as pd

# Create sample data
data = np.random.randn(100, 5)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

print(f"Data shape: {data.shape}")
print(f"DataFrame shape: {df.shape}")
print(f"Mean of column A: {df['A'].mean():.4f}")
print("✅ Basic operations successful")
"""
    
    result1 = executor.execute_code(code1)
    print(f"Status: {result1['status']}")
    if result1['status'] == 'SUCCESS':
        print("✅ Basic Python/NumPy test passed")
        print(f"Output: {result1['output'][:200]}...")
    else:
        print(f"❌ Basic test failed: {result1['error']}")
        return False
    
    # Test 2: Scikit-learn operations
    print("\n2. Testing scikit-learn operations...")
    code2 = """
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Model accuracy: {accuracy:.4f}")
print("✅ Scikit-learn operations successful")
"""
    
    result2 = executor.execute_code(code2)
    print(f"Status: {result2['status']}")
    if result2['status'] == 'SUCCESS':
        print("✅ Scikit-learn test passed")
        print(f"Output: {result2['output'][:200]}...")
    else:
        print(f"❌ Scikit-learn test failed: {result2['error']}")
        return False
    
    # Test 3: Data processing with pandas
    print("\n3. Testing data processing...")
    code3 = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create sample dataset
np.random.seed(42)
data = {
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
    'feature3': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.choice([0, 1], 1000)
}

df = pd.DataFrame(data)

# Data preprocessing
print(f"Original shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Handle categorical data
df_encoded = pd.get_dummies(df, columns=['feature3'])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['feature1', 'feature2']
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

print(f"After preprocessing: {df_encoded.shape}")
print(f"Feature names: {list(df_encoded.columns)}")
print("✅ Data processing successful")
"""
    
    result3 = executor.execute_code(code3)
    print(f"Status: {result3['status']}")
    if result3['status'] == 'SUCCESS':
        print("✅ Data processing test passed")
        print(f"Output: {result3['output'][:200]}...")
    else:
        print(f"❌ Data processing test failed: {result3['error']}")
        return False
    
    # Test 4: Memory and resource usage
    print("\n4. Testing resource usage...")
    code4 = """
import numpy as np
import psutil
import time

# Monitor initial resources
initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
print(f"Initial memory usage: {initial_memory:.2f} MB")

# Create large array
large_array = np.random.randn(10000, 100)
print(f"Large array shape: {large_array.shape}")

# Perform computation
result = np.dot(large_array.T, large_array)
print(f"Computation result shape: {result.shape}")

# Check memory after computation
final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
print(f"Final memory usage: {final_memory:.2f} MB")
print(f"Memory increase: {final_memory - initial_memory:.2f} MB")

print("✅ Resource usage test successful")
"""
    
    result4 = executor.execute_code(code4)
    print(f"Status: {result4['status']}")
    if result4['status'] == 'SUCCESS':
        print("✅ Resource usage test passed")
        print(f"Output: {result4['output'][:200]}...")
    else:
        print(f"❌ Resource usage test failed: {result4['error']}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All ML sandbox tests passed!")
    print("✅ The sandbox is ready for production use")
    return True

if __name__ == "__main__":
    success = test_basic_ml_operations()
    if success:
        print("\n✅ Sandbox testing completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
