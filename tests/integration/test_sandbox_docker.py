#!/usr/bin/env python3
"""
Test Docker Sandbox Execution

Tests the sandbox executor with sample LLM-generated code to verify Docker integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.sandbox_executor import SandboxExecutor

def test_sandbox_execution():
    """Test basic sandbox execution"""
    print("=" * 60)
    print("TESTING DOCKER SANDBOX EXECUTION")
    print("=" * 60)
    
    # Test code (simulating LLM-generated code)
    test_code = """
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.4f}")
print(f"Classification Report:")
print(classification_report(y_test, y_pred))
"""
    
    print("\n1. Initializing Sandbox Executor...")
    try:
        executor = SandboxExecutor()
        print("   ✅ Sandbox executor initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    print("\n2. Executing test code in Docker sandbox...")
    try:
        result = executor.execute_code(test_code)
        
        print(f"\n   Status: {result.get('status')}")
        print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if result.get('status') == 'SUCCESS':
            print("\n   ✅ Code executed successfully!")
            print(f"\n   Output:\n{result.get('output', '')[:500]}")
            return True
        else:
            print(f"\n   ❌ Execution failed")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            if result.get('output'):
                print(f"   Output: {result.get('output')[:500]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sandbox_execution()
    sys.exit(0 if success else 1)

