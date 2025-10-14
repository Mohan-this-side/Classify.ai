#!/usr/bin/env python3
"""
🎵 Simple Spotify Churn Analysis Test

This script tests our enhanced data cleaning agent on the real Spotify churn dataset
to demonstrate the capabilities without the full workflow complexity.
"""

import pandas as pd
import numpy as np
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the backend directory to the Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Mock the required modules to avoid configuration issues
class MockSettings:
    def __init__(self):
        pass

class MockStateManager:
    def __init__(self):
        self.storage = {}
    
    def store_dataset(self, state, dataset, dataset_type):
        key = f"{state.get('session_id', 'test')}_{dataset_type}"
        self.storage[key] = dataset
    
    def get_dataset(self, state, dataset_type):
        key = f"{state.get('session_id', 'test')}_{dataset_type}"
        return self.storage.get(key)

# Mock the modules
sys.modules['app.config'] = type('MockConfig', (), {'settings': MockSettings()})()
sys.modules['app.workflows.state_management'] = type('MockStateManagement', (), {
    'ClassificationState': dict,
    'AgentStatus': str,
    'state_manager': MockStateManager()
})()

# Import our enhanced data cleaning agent
from app.agents.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent

def load_spotify_dataset():
    """Load the Spotify churn dataset"""
    dataset_path = "test_data/spotify_churn_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Loaded Spotify dataset: {dataset_path} - Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Analyze the dataset characteristics"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': dict(df.dtypes),
        'missing_values': dict(df.isnull().sum()),
        'duplicate_rows': df.duplicated().sum(),
        'target_distribution': df['is_churned'].value_counts().to_dict() if 'is_churned' in df.columns else {},
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return analysis

async def test_data_cleaning_on_spotify():
    """Test the enhanced data cleaning agent on Spotify dataset"""
    
    print("🎵 Testing Enhanced Data Cleaning Agent on Spotify Churn Dataset")
    print("=" * 70)
    
    # Load dataset
    df = load_spotify_dataset()
    if df is None:
        return None
    
    # Analyze original dataset
    print("\n📊 Original Dataset Analysis:")
    analysis = analyze_dataset(df)
    print(f"  Shape: {analysis['shape']}")
    print(f"  Columns: {len(analysis['columns'])}")
    print(f"  Missing values: {sum(analysis['missing_values'].values())}")
    print(f"  Duplicates: {analysis['duplicate_rows']}")
    print(f"  Target distribution: {analysis['target_distribution']}")
    print(f"  Numeric columns: {len(analysis['numeric_columns'])}")
    print(f"  Categorical columns: {len(analysis['categorical_columns'])}")
    
    # Show sample data
    print(f"\n📋 Sample data (first 5 rows):")
    print(df.head().to_string())
    
    # Show data types
    print(f"\n🔧 Data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Show missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ Missing values by column:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count}")
    else:
        print(f"\n✅ No missing values found")
    
    # Create test state
    state = {
        "session_id": f"spotify_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "dataset_id": "spotify_churn",
        "target_column": "is_churned",
        "user_description": "Spotify user churn prediction - predict which users will cancel their subscription",
        "api_key": "test_key",
        "workflow_status": "running",
        "agent_statuses": {},
        "completed_agents": [],
        "failed_agents": [],
        "workflow_progress": 0.0,
        "progress": 0.0,
        "errors": [],
        "warnings": [],
        "retry_count": 0,
        "max_retries": 3,
        "error_count": 0,
        "last_error": None,
        "start_time": datetime.now(),
        "end_time": None,
        "total_execution_time": None,
        "agent_execution_times": {},
        "memory_usage": {},
        "cpu_usage": {},
        "requires_human_input": False,
        "human_input_required": None,
        "human_feedback": None,
        "user_approvals": {},
        "output_artifacts": {},
        "downloadable_files": [],
        "notebook_path": None,
        "model_path": None,
        "report_path": None
    }
    
    # Store dataset
    from app.workflows.state_management import state_manager
    state_manager.store_dataset(state, df, "original")
    
    try:
        # Initialize enhanced data cleaning agent
        print(f"\n🧹 Initializing Enhanced Data Cleaning Agent...")
        agent = EnhancedDataCleaningAgent()
        
        # Execute data cleaning
        print(f"\n⚡ Running data cleaning...")
        start_time = datetime.now()
        
        # Run the data cleaning agent
        result = await agent.execute(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Data cleaning completed in {execution_time:.2f} seconds")
        
        # Get cleaned dataset
        cleaned_df = state_manager.get_dataset(result, "cleaned")
        
        if cleaned_df is not None:
            # Analyze cleaned dataset
            print(f"\n📊 Cleaned Dataset Analysis:")
            cleaned_analysis = analyze_dataset(cleaned_df)
            print(f"  Shape: {cleaned_analysis['shape']}")
            print(f"  Missing values: {sum(cleaned_analysis['missing_values'].values())}")
            print(f"  Duplicates: {cleaned_analysis['duplicate_rows']}")
            print(f"  Target distribution: {cleaned_analysis['target_distribution']}")
            
            # Show improvements
            print(f"\n📈 Data Quality Improvements:")
            missing_before = sum(analysis['missing_values'].values())
            missing_after = sum(cleaned_analysis['missing_values'].values())
            duplicates_before = analysis['duplicate_rows']
            duplicates_after = cleaned_analysis['duplicate_rows']
            
            print(f"  Missing values: {missing_before} → {missing_after} ({missing_before - missing_after} fixed)")
            print(f"  Duplicates: {duplicates_before} → {duplicates_after} ({duplicates_before - duplicates_after} removed)")
            print(f"  Quality score: {result.get('data_quality_score', 0):.3f}")
            
            # Show cleaning actions
            actions = result.get('cleaning_actions_taken', [])
            print(f"\n⚡ Cleaning Actions Taken ({len(actions)}):")
            for i, action in enumerate(actions[:10], 1):  # Show first 10 actions
                print(f"  {i}. {action}")
            if len(actions) > 10:
                print(f"  ... and {len(actions) - 10} more actions")
            
            # Show sample cleaned data
            print(f"\n📋 Cleaned data (first 5 rows):")
            print(cleaned_df.head().to_string())
            
            # Show data type changes
            print(f"\n🔧 Data type changes:")
            for col in df.columns:
                if col in cleaned_df.columns:
                    orig_type = str(df[col].dtype)
                    clean_type = str(cleaned_df[col].dtype)
                    if orig_type != clean_type:
                        print(f"  {col}: {orig_type} → {clean_type}")
            
            return {
                'success': True,
                'execution_time': execution_time,
                'quality_score': result.get('data_quality_score', 0),
                'actions_count': len(actions),
                'missing_fixed': missing_before - missing_after,
                'duplicates_removed': duplicates_before - duplicates_after,
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape
            }
        else:
            print("❌ No cleaned dataset returned")
            return None
        
    except Exception as e:
        print(f"\n❌ Data cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

async def main():
    """Main test function"""
    print("🎵 Spotify Churn Analysis - Enhanced Data Cleaning Test")
    print("=" * 70)
    
    # Run the test
    results = await test_data_cleaning_on_spotify()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if results and results['success']:
        print("✅ Data cleaning test completed successfully!")
        print(f"⏱️ Execution time: {results['execution_time']:.2f} seconds")
        print(f"📈 Quality score: {results['quality_score']:.3f}")
        print(f"⚡ Actions taken: {results['actions_count']}")
        print(f"🔧 Missing values fixed: {results['missing_fixed']}")
        print(f"🔧 Duplicates removed: {results['duplicates_removed']}")
        print(f"📊 Shape change: {results['original_shape']} → {results['cleaned_shape']}")
    else:
        print("❌ Data cleaning test failed!")
        if results:
            print(f"Error: {results['error']}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
