import pandas as pd
import numpy as np
import tempfile
import subprocess
import sys
import os
import io
import traceback
from typing import Dict, List, Tuple, Any
import json

class SafeCodeExecutor:
    """
    Safe code execution environment for data cleaning scripts
    """
    
    def __init__(self):
        self.allowed_imports = [
            'pandas', 'numpy', 'datetime', 'math', 'statistics',
            'collections', 're', 'string', 'itertools', 'sklearn',
            'sklearn.preprocessing', 'sklearn.impute', 'sklearn.ensemble',
            'sklearn.cluster', 'sklearn.decomposition', 'sklearn.feature_selection'
        ]
        self.forbidden_functions = [
            'exec', 'eval', 'open', '__import__', 'compile',
            'input', 'raw_input', 'file', 'execfile'
        ]
    
    def validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute"""
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Check for import statements (forbidden in our environment)
            if (stripped_line.startswith('import ') or stripped_line.startswith('from ')) and not stripped_line.startswith('#'):
                return False, f"Import statement forbidden at line {line_num}: '{stripped_line}'. All libraries are pre-imported!"
            
            # Check for forbidden functions
            for forbidden in self.forbidden_functions:
                if forbidden in line and not stripped_line.startswith('#'):
                    return False, f"Forbidden function '{forbidden}' found at line {line_num}"
            
            # Check for file operations
            if any(keyword in line for keyword in ['open(', 'file(', 'with open']) and not stripped_line.startswith('#'):
                return False, f"File operation detected at line {line_num}"
            
            # Check for network operations
            if any(keyword in line for keyword in ['urllib', 'requests', 'socket', 'http']) and not stripped_line.startswith('#'):
                return False, f"Network operation detected at line {line_num}"
            
            # Check for system operations
            if any(keyword in line for keyword in ['os.system', 'subprocess', 'sys.exit']) and not stripped_line.startswith('#'):
                return False, f"System operation detected at line {line_num}"
        
        return True, "Code appears safe"
    
    def execute_with_timeout(self, code: str, dataframe: pd.DataFrame, timeout: int = 30) -> Tuple[bool, str, Any]:
        """Execute code with timeout and capture output"""
        try:
            # Validate code safety first
            is_safe, safety_msg = self.validate_code_safety(code)
            if not is_safe:
                return False, f"Code safety check failed: {safety_msg}", None
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # Import sklearn modules safely
            try:
                from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
                from sklearn.impute import SimpleImputer, KNNImputer
                from sklearn.ensemble import IsolationForest
                from sklearn.cluster import KMeans
                sklearn_available = True
            except ImportError:
                sklearn_available = False
            
            # Create execution environment with safer approach
            # Use standard builtins but remove dangerous functions
            safe_builtins = __builtins__.copy() if isinstance(__builtins__, dict) else __builtins__.__dict__.copy()
            
            # Remove dangerous functions (but keep __import__ for pandas/numpy internal use)
            dangerous_functions = ['exec', 'eval', 'compile', 'open', 'input', 'raw_input']
            for func in dangerous_functions:
                safe_builtins.pop(func, None)
            
            exec_globals = {
                'pd': pd,
                'np': np,
                'df': dataframe.copy(),
                'print': print,
                '__builtins__': safe_builtins
            }
            
            # Add sklearn to environment if available
            if sklearn_available:
                exec_globals.update({
                    'StandardScaler': StandardScaler,
                    'LabelEncoder': LabelEncoder,
                    'MinMaxScaler': MinMaxScaler,
                    'SimpleImputer': SimpleImputer,
                    'KNNImputer': KNNImputer,
                    'IsolationForest': IsolationForest,
                    'KMeans': KMeans
                })
            
            # Execute the code
            exec(code, exec_globals)
            
            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Get the result
            if 'cleaned_df' in exec_globals:
                cleaned_df = exec_globals['cleaned_df']
                
                # Critical validation: Check for duplicate creation bug
                original_duplicates = dataframe.duplicated().sum()
                final_duplicates = cleaned_df.duplicated().sum()
                
                if final_duplicates > original_duplicates:
                    validation_error = (
                        f"CRITICAL ERROR: Code CREATED duplicates instead of removing them!\n"
                        f"Original duplicates: {original_duplicates}\n"
                        f"Final duplicates: {final_duplicates}\n"
                        f"This indicates the code used pd.concat() or similar operations incorrectly.\n"
                        f"The code must be fixed to use drop_duplicates() instead."
                    )
                    return False, validation_error, dataframe
                
                # Critical validation: Check for missing values (data scientist requirement)
                final_missing = cleaned_df.isnull().sum().sum()
                if final_missing > 0:
                    missing_details = cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0].to_dict()
                    validation_error = (
                        f"DATA SCIENCE ERROR: Missing values still present after cleaning!\n"
                        f"Missing values found: {missing_details}\n"
                        f"A professional data scientist NEVER leaves missing values.\n"
                        f"All missing values must be imputed using:\n"
                        f"- Numeric columns: mean (normal dist) or median (skewed dist)\n"
                        f"- Categorical columns: mode or domain knowledge\n"
                        f"The code must be enhanced with comprehensive imputation."
                    )
                    return False, validation_error, dataframe
                
                return True, output, cleaned_df
            else:
                return False, "No 'cleaned_df' variable found in executed code", None
                
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            
            # Enhanced error reporting with context
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'dataframe_shape': dataframe.shape,
                'dataframe_columns': list(dataframe.columns),
                'dataframe_dtypes': dict(dataframe.dtypes)
            }
            
            # Create comprehensive error message
            error_msg = (
                f"EXECUTION ERROR DETAILS:\n"
                f"Error Type: {error_details['error_type']}\n"
                f"Error Message: {error_details['error_message']}\n"
                f"DataFrame Context: Shape={error_details['dataframe_shape']}, "
                f"Columns={error_details['dataframe_columns']}\n"
                f"Data Types: {error_details['dataframe_dtypes']}\n"
                f"Full Traceback:\n{error_details['traceback']}"
            )
            
            return False, error_msg, None
    
    def run_data_quality_tests(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive data quality tests"""
        tests = {}
        
        try:
            # Test 1: Data integrity
            tests['data_integrity'] = {
                'passed': True,
                'issues': []
            }
            
            # Check if we lost too much data
            data_loss_ratio = (original_df.shape[0] - cleaned_df.shape[0]) / original_df.shape[0]
            if data_loss_ratio > 0.5:
                tests['data_integrity']['passed'] = False
                tests['data_integrity']['issues'].append(f"High data loss: {data_loss_ratio:.2%}")
            
            # Test 2: Column preservation
            tests['column_preservation'] = {
                'passed': True,
                'issues': []
            }
            
            important_columns_lost = set(original_df.columns) - set(cleaned_df.columns)
            if important_columns_lost:
                tests['column_preservation']['passed'] = False
                tests['column_preservation']['issues'].append(f"Lost columns: {list(important_columns_lost)}")
            
            # Test 3: Data type consistency
            tests['data_type_consistency'] = {
                'passed': True,
                'issues': []
            }
            
            for col in cleaned_df.columns:
                if col in original_df.columns:
                    # Check if numeric columns are still numeric
                    if pd.api.types.is_numeric_dtype(original_df[col]) and not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        tests['data_type_consistency']['passed'] = False
                        tests['data_type_consistency']['issues'].append(f"Column '{col}' lost numeric type")
            
            # Test 4: Missing value handling
            tests['missing_value_handling'] = {
                'passed': True,
                'issues': []
            }
            
            original_missing = original_df.isnull().sum().sum()
            cleaned_missing = cleaned_df.isnull().sum().sum()
            
            if cleaned_missing > original_missing:
                tests['missing_value_handling']['passed'] = False
                tests['missing_value_handling']['issues'].append("Cleaning increased missing values")
            
            # Test 5: Duplicate handling
            tests['duplicate_handling'] = {
                'passed': True,
                'issues': []
            }
            
            original_duplicates = original_df.duplicated().sum()
            cleaned_duplicates = cleaned_df.duplicated().sum()
            
            if cleaned_duplicates > original_duplicates:
                tests['duplicate_handling']['passed'] = False
                tests['duplicate_handling']['issues'].append("Cleaning increased duplicates")
            
            # Overall test result
            tests['overall_passed'] = all(test['passed'] for test in tests.values() if isinstance(test, dict) and 'passed' in test)
            
        except Exception as e:
            tests['error'] = f"Testing failed: {str(e)}"
            tests['overall_passed'] = False
        
        return tests
    
    def generate_quality_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, execution_output: str) -> str:
        """Generate a comprehensive quality report"""
        
        report = []
        report.append("=" * 60)
        report.append("DATA CLEANING QUALITY REPORT")
        report.append("=" * 60)
        
        # Basic statistics
        report.append(f"\n📊 BASIC STATISTICS:")
        report.append(f"Original dataset shape: {original_df.shape}")
        report.append(f"Cleaned dataset shape: {cleaned_df.shape}")
        report.append(f"Rows removed: {original_df.shape[0] - cleaned_df.shape[0]}")
        report.append(f"Columns removed: {original_df.shape[1] - cleaned_df.shape[1]}")
        
        # Missing values
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        report.append(f"\n🔍 MISSING VALUES:")
        report.append(f"Before cleaning: {original_missing}")
        report.append(f"After cleaning: {cleaned_missing}")
        report.append(f"Missing values removed: {original_missing - cleaned_missing}")
        
        # Duplicates
        original_duplicates = original_df.duplicated().sum()
        cleaned_duplicates = cleaned_df.duplicated().sum()
        report.append(f"\n🔄 DUPLICATES:")
        report.append(f"Before cleaning: {original_duplicates}")
        report.append(f"After cleaning: {cleaned_duplicates}")
        report.append(f"Duplicates removed: {original_duplicates - cleaned_duplicates}")
        
        # Data types
        report.append(f"\n📋 DATA TYPES:")
        type_changes = []
        for col in original_df.columns:
            if col in cleaned_df.columns:
                if str(original_df[col].dtype) != str(cleaned_df[col].dtype):
                    type_changes.append(f"  {col}: {original_df[col].dtype} → {cleaned_df[col].dtype}")
        
        if type_changes:
            report.append("Type changes made:")
            report.extend(type_changes)
        else:
            report.append("No data type changes made")
        
        # Quality tests
        tests = self.run_data_quality_tests(original_df, cleaned_df)
        report.append(f"\n✅ QUALITY TESTS:")
        
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict) and 'passed' in test_result:
                status = "✅ PASSED" if test_result['passed'] else "❌ FAILED"
                report.append(f"  {test_name.replace('_', ' ').title()}: {status}")
                if test_result['issues']:
                    for issue in test_result['issues']:
                        report.append(f"    - {issue}")
        
        # Execution output
        if execution_output.strip():
            report.append(f"\n📝 EXECUTION OUTPUT:")
            report.append(execution_output)
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)