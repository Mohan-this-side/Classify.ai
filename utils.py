import pandas as pd
import numpy as np
import io
import tempfile
import os
from typing import Dict, List, Tuple, Any

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding for better CSV reading"""
    import chardet
    
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

def smart_read_csv(file_path: str) -> pd.DataFrame:
    """Smart CSV reader with automatic encoding detection and error handling"""
    try:
        # Try with default encoding first
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        # Try with detected encoding
        encoding = detect_file_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        # Try with common encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except:
                continue
        raise e

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset summary"""
    summary = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
        },
        'data_types': {
            'numeric': list(df.select_dtypes(include=[np.number]).columns),
            'categorical': list(df.select_dtypes(include=['object']).columns),
            'datetime': list(df.select_dtypes(include=['datetime64']).columns),
            'boolean': list(df.select_dtypes(include=['bool']).columns)
        },
        'missing_values': {
            'total': df.isnull().sum().sum(),
            'per_column': df.isnull().sum().to_dict(),
            'percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        },
        'duplicates': {
            'count': df.duplicated().sum(),
            'percentage': df.duplicated().sum() / len(df) * 100
        }
    }
    
    # Add statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Add info about categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary['categorical_info'] = {}
        for col in categorical_cols:
            summary['categorical_info'][col] = {
                'unique_count': df[col].nunique(),
                'most_common': df[col].value_counts().head(5).to_dict()
            }
    
    return summary

def create_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Create sample datasets for testing"""
    datasets = {}
    
    # Dataset 1: Customer data with common issues
    np.random.seed(42)
    n_customers = 1000
    
    customers_data = {
        'customer_id': range(1, n_customers + 1),
        'name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
        'age': np.random.randint(18, 80, n_customers),
        'email': [f'customer{i}@email.com' for i in range(1, n_customers + 1)],
        'signup_date': pd.date_range('2020-01-01', periods=n_customers, freq='D'),
        'annual_income': np.random.normal(50000, 15000, n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers)
    }
    
    customers_df = pd.DataFrame(customers_data)
    
    # Introduce data quality issues
    # Missing values
    customers_df.loc[np.random.choice(customers_df.index, 50), 'email'] = None
    customers_df.loc[np.random.choice(customers_df.index, 30), 'annual_income'] = None
    
    # Duplicates - Create realistic duplicates without teaching wrong patterns
    # Select random rows to duplicate (not bulk concatenation)
    duplicate_indices = np.random.choice(customers_df.index, 10, replace=False)
    duplicate_rows = customers_df.iloc[duplicate_indices].copy()
    customers_df = pd.concat([customers_df, duplicate_rows], ignore_index=True)
    
    # Data type issues
    customers_df['age'] = customers_df['age'].astype(str)  # Age as string
    customers_df.loc[np.random.choice(customers_df.index, 10), 'age'] = 'unknown'
    
    # Outliers
    customers_df.loc[np.random.choice(customers_df.index, 5), 'annual_income'] = 1000000
    
    datasets['customers'] = customers_df
    
    # Dataset 2: Sales data
    sales_data = {
        'order_id': range(1, 500),
        'customer_id': np.random.randint(1, 100, 499),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 499),
        'quantity': np.random.randint(1, 10, 499),
        'price': np.random.uniform(10, 100, 499),
        'order_date': pd.date_range('2023-01-01', periods=499, freq='h')
    }
    
    sales_df = pd.DataFrame(sales_data)
    
    # Introduce issues
    sales_df.loc[np.random.choice(sales_df.index, 20), 'price'] = None
    sales_df.loc[np.random.choice(sales_df.index, 10), 'quantity'] = 0  # Invalid quantity
    
    datasets['sales'] = sales_df
    
    return datasets

def save_dataframe(df: pd.DataFrame, filename: str, format: str = 'csv') -> str:
    """Save dataframe to file and return the file path"""
    temp_dir = tempfile.gettempdir()
    
    if format.lower() == 'csv':
        file_path = os.path.join(temp_dir, f"{filename}.csv")
        df.to_csv(file_path, index=False)
    elif format.lower() == 'excel':
        file_path = os.path.join(temp_dir, f"{filename}.xlsx")
        df.to_excel(file_path, index=False)
    elif format.lower() == 'json':
        file_path = os.path.join(temp_dir, f"{filename}.json")
        df.to_json(file_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return file_path

def create_download_link(df: pd.DataFrame, filename: str, format: str = 'csv') -> bytes:
    """Create downloadable content for dataframe"""
    if format.lower() == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format.lower() == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        return output.getvalue()
    elif format.lower() == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8')
    else:
        raise ValueError(f"Unsupported format: {format}")

def validate_api_key(api_key: str) -> bool:
    """Validate Google Gemini API key format"""
    if not api_key:
        return False
    
    # Basic validation - Gemini API keys typically start with 'AIza'
    if not api_key.startswith('AIza') or len(api_key) < 30:
        return False
    
    return True

def format_code_for_download(code: str, dataset_name: str = "dataset") -> str:
    """Format the cleaning code for download with proper structure"""
    formatted_code = f'''"""
Data Cleaning Code
Generated by AI Data Cleaning Agent
Dataset: {dataset_name}
Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np

def clean_data(input_file_path, output_file_path=None):
    """
    Clean the dataset using AI-generated cleaning logic
    
    Parameters:
    input_file_path (str): Path to the input dataset file
    output_file_path (str, optional): Path to save cleaned dataset
    
    Returns:
    pd.DataFrame: Cleaned dataset
    """
    
    # Load the dataset
    if input_file_path.endswith('.csv'):
        df = pd.read_csv(input_file_path)
    elif input_file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    print(f"Original dataset shape: {{df.shape}}")
    
    # AI-Generated Cleaning Code
    # ========================
{code}
    
    print(f"Cleaned dataset shape: {{cleaned_df.shape}}")
    
    # Save cleaned dataset if output path is provided
    if output_file_path:
        if output_file_path.endswith('.csv'):
            cleaned_df.to_csv(output_file_path, index=False)
        elif output_file_path.endswith(('.xlsx', '.xls')):
            cleaned_df.to_excel(output_file_path, index=False)
        print(f"Cleaned dataset saved to: {{output_file_path}}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    # cleaned_data = clean_data('your_dataset.csv', 'cleaned_dataset.csv')
    pass
'''
    
    return formatted_code