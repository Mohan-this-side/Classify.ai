#!/usr/bin/env python3
"""
Test script for the AI Data Cleaning Agent
Run this to verify that all components are working correctly
"""

import sys
import pandas as pd
import numpy as np
from utils import create_sample_datasets, get_dataset_summary
from code_executor import SafeCodeExecutor

def test_utilities():
    """Test utility functions"""
    print("🧪 Testing utility functions...")
    
    # Test sample dataset creation
    try:
        datasets = create_sample_datasets()
        assert 'customers' in datasets
        assert 'sales' in datasets
        assert isinstance(datasets['customers'], pd.DataFrame)
        assert isinstance(datasets['sales'], pd.DataFrame)
        print("✅ Sample dataset creation: PASSED")
    except Exception as e:
        print(f"❌ Sample dataset creation: FAILED - {e}")
        return False
    
    # Test dataset summary
    try:
        df = datasets['customers']
        summary = get_dataset_summary(df)
        assert 'basic_info' in summary
        assert 'data_types' in summary
        assert 'missing_values' in summary
        print("✅ Dataset summary generation: PASSED")
    except Exception as e:
        print(f"❌ Dataset summary generation: FAILED - {e}")
        return False
    
    return True

def test_code_executor():
    """Test code execution environment"""
    print("\n🧪 Testing code executor...")
    
    executor = SafeCodeExecutor()
    
    # Test safe code validation
    try:
        safe_code = "df['new_col'] = df['age'] * 2"
        is_safe, msg = executor.validate_code_safety(safe_code)
        assert is_safe == True
        print("✅ Safe code validation: PASSED")
    except Exception as e:
        print(f"❌ Safe code validation: FAILED - {e}")
        return False
    
    # Test unsafe code detection
    try:
        unsafe_code = "import os; os.system('rm -rf /')"
        is_safe, msg = executor.validate_code_safety(unsafe_code)
        assert is_safe == False
        print("✅ Unsafe code detection: PASSED")
    except Exception as e:
        print(f"❌ Unsafe code detection: FAILED - {e}")
        return False
    
    # Test code execution
    try:
        datasets = create_sample_datasets()
        test_df = datasets['customers'].head(10)
        
        test_code = """
# Simple cleaning code
cleaned_df = df.copy()
cleaned_df = cleaned_df.dropna()
cleaned_df = cleaned_df.drop_duplicates()
print(f"Cleaned {len(df)} rows to {len(cleaned_df)} rows")
"""
        
        success, output, result_df = executor.execute_with_timeout(test_code, test_df)
        assert success == True
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)
        print("✅ Code execution: PASSED")
    except Exception as e:
        print(f"❌ Code execution: FAILED - {e}")
        return False
    
    # Test quality tests
    try:
        original_df = datasets['customers'].head(20)
        cleaned_df = original_df.dropna().drop_duplicates()
        
        quality_tests = executor.run_data_quality_tests(original_df, cleaned_df)
        assert 'data_integrity' in quality_tests
        assert 'overall_passed' in quality_tests
        print("✅ Quality testing: PASSED")
    except Exception as e:
        print(f"❌ Quality testing: FAILED - {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test agent initialization without API key"""
    print("\n🧪 Testing agent initialization...")
    
    try:
        # Test with dummy API key (won't make actual calls)
        from data_cleaning_agent import DataCleaningAgent
        
        # Test initialization
        agent = DataCleaningAgent("AIzaSyDummy_Key_For_Testing_Only_1234567890")
        assert agent is not None
        print("✅ Agent initialization: PASSED")
        
        # Test dataset loading
        datasets = create_sample_datasets()
        success = agent.load_dataset(dataframe=datasets['customers'])
        assert success == True
        assert agent.df is not None
        print("✅ Dataset loading: PASSED")
        
    except Exception as e:
        print(f"❌ Agent initialization: FAILED - {e}")
        return False
    
    return True

def test_streamlit_imports():
    """Test that all Streamlit dependencies are available"""
    print("\n🧪 Testing Streamlit dependencies...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import: PASSED")
    except ImportError as e:
        print(f"❌ Streamlit import: FAILED - {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI import: PASSED")
    except ImportError as e:
        print(f"❌ Google Generative AI import: FAILED - {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting AI Data Cleaning Agent System Tests")
    print("=" * 60)
    
    tests = [
        test_streamlit_imports,
        test_utilities,
        test_code_executor,
        test_agent_initialization
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        if test():
            passed_tests += 1
        else:
            break  # Stop on first failure
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. Get your Google Gemini API key from https://ai.google.dev/")
        print("2. Run the application: streamlit run app.py")
        print("3. Upload a dataset and start cleaning!")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)