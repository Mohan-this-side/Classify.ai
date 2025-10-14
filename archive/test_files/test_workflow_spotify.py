#!/usr/bin/env python3
"""
🎵 Spotify Churn Analysis Workflow Test

This script tests our complete agentic workflow on the Spotify churn analysis dataset
to evaluate the performance and accuracy of our multi-agent system.
"""

import pandas as pd
import numpy as np
import os
import sys
import asyncio
import json
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
        self.debug = True
        self.max_retries = 3
        self.timeout_seconds = 300

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

# Import our workflow and agents
from app.workflows.classification_workflow import ClassificationWorkflow

def load_spotify_dataset(file_path: str) -> pd.DataFrame:
    """Load the Spotify churn dataset"""
    try:
        # Try different possible file names
        possible_files = [
            file_path,
            file_path.replace('.csv', ''),
            'spotify_churn_dataset.csv',
            'spotify_dataset_for_churn_analysis.csv',
            'spotify_churn.csv',
            'churn_data.csv'
        ]
        
        for file in possible_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                print(f"✅ Loaded Spotify dataset: {file} - Shape: {df.shape}")
                return df
        
        # If no file found, create a sample dataset
        print("⚠️ Spotify dataset not found, creating sample dataset...")
        return create_sample_spotify_dataset()
        
    except Exception as e:
        print(f"❌ Error loading Spotify dataset: {e}")
        print("Creating sample dataset instead...")
        return create_sample_spotify_dataset()

def create_sample_spotify_dataset() -> pd.DataFrame:
    """Create a sample Spotify churn dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data based on typical Spotify churn features
    data = {
        'user_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'country': np.random.choice(['US', 'UK', 'Canada', 'Germany', 'France', 'Japan'], n_samples),
        'premium_subscriber': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'total_playtime_minutes': np.random.exponential(1000, n_samples),
        'sessions_per_week': np.random.poisson(15, n_samples),
        'avg_session_duration': np.random.normal(25, 10, n_samples),
        'songs_played_per_session': np.random.poisson(8, n_samples),
        'skips_per_session': np.random.poisson(3, n_samples),
        'likes_per_session': np.random.poisson(2, n_samples),
        'playlist_creations': np.random.poisson(2, n_samples),
        'followed_artists': np.random.poisson(5, n_samples),
        'days_since_last_login': np.random.exponential(7, n_samples),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Web'], n_samples),
        'subscription_length_months': np.random.exponential(12, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it more realistic
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'avg_session_duration'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[outlier_indices, 'total_playtime_minutes'] *= 10
    
    print(f"✅ Created sample Spotify dataset: {df.shape}")
    return df

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Analyze the dataset characteristics"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': dict(df.dtypes),
        'missing_values': dict(df.isnull().sum()),
        'duplicate_rows': df.duplicated().sum(),
        'target_distribution': df['churn'].value_counts().to_dict() if 'churn' in df.columns else {},
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return analysis

async def test_workflow_on_spotify():
    """Test the complete workflow on Spotify churn dataset"""
    
    print("🎵 Testing Multi-Agent Workflow on Spotify Churn Dataset")
    print("=" * 70)
    
    # Load dataset
    dataset_path = "test_data/spotify_churn_dataset.csv"
    df = load_spotify_dataset(dataset_path)
    
    # Analyze dataset
    print("\n📊 Dataset Analysis:")
    analysis = analyze_dataset(df)
    print(f"  Shape: {analysis['shape']}")
    print(f"  Columns: {len(analysis['columns'])}")
    print(f"  Missing values: {sum(analysis['missing_values'].values())}")
    print(f"  Duplicates: {analysis['duplicate_rows']}")
    print(f"  Target distribution: {analysis['target_distribution']}")
    
    # Show sample data
    print(f"\n📋 Sample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Create workflow state
    state = {
        "session_id": f"spotify_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "dataset_id": "spotify_churn",
        "target_column": "churn",
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
        # Initialize workflow
        print(f"\n🚀 Initializing Multi-Agent Workflow...")
        workflow = ClassificationWorkflow()
        
        # Execute workflow
        print(f"\n⚡ Executing workflow...")
        start_time = datetime.now()
        
        # Run the workflow
        result = await workflow.execute_workflow(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Workflow completed in {execution_time:.2f} seconds")
        
        # Analyze results
        print(f"\n📈 Workflow Results:")
        print(f"  Final status: {result.get('workflow_status', 'unknown')}")
        print(f"  Completed agents: {len(result.get('completed_agents', []))}")
        print(f"  Failed agents: {len(result.get('failed_agents', []))}")
        print(f"  Final progress: {result.get('workflow_progress', 0):.1%}")
        
        # Get cleaned dataset
        cleaned_df = state_manager.get_dataset(result, "cleaned")
        if cleaned_df is not None:
            print(f"\n🧹 Data Cleaning Results:")
            print(f"  Original shape: {df.shape}")
            print(f"  Cleaned shape: {cleaned_df.shape}")
            print(f"  Missing values removed: {df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()}")
            print(f"  Duplicates removed: {df.duplicated().sum() - cleaned_df.duplicated().sum()}")
        
        # Check for model results
        if 'model_performance' in result:
            performance = result['model_performance']
            print(f"\n🎯 Model Performance:")
            print(f"  Best model: {performance.get('best_model', 'N/A')}")
            print(f"  Accuracy: {performance.get('accuracy', 0):.3f}")
            print(f"  Precision: {performance.get('precision', 0):.3f}")
            print(f"  Recall: {performance.get('recall', 0):.3f}")
            print(f"  F1-Score: {performance.get('f1_score', 0):.3f}")
        
        # Check for artifacts
        artifacts = result.get('output_artifacts', {})
        if artifacts:
            print(f"\n📁 Generated Artifacts:")
            for artifact_type, artifact_info in artifacts.items():
                print(f"  {artifact_type}: {artifact_info}")
        
        # Check for errors
        errors = result.get('errors', [])
        if errors:
            print(f"\n⚠️ Errors encountered:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        # Save results
        results_file = f"test_data/spotify_workflow_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_result[key] = value.item()
                elif isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    json_result[key] = "DataFrame stored externally"
                else:
                    json_result[key] = str(value)
            
            json.dump(json_result, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'final_status': result.get('workflow_status'),
            'completed_agents': len(result.get('completed_agents', [])),
            'failed_agents': len(result.get('failed_agents', [])),
            'model_performance': result.get('model_performance', {}),
            'artifacts': artifacts,
            'errors': errors
        }
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

async def main():
    """Main test function"""
    print("🎵 Spotify Churn Analysis - Multi-Agent Workflow Test")
    print("=" * 70)
    
    # Check if dataset exists
    dataset_files = [
        "test_data/spotify_dataset_for_churn_analysis.csv",
        "test_data/spotify-dataset-for-churn-analysis.csv",
        "test_data/spotify_churn.csv"
    ]
    
    dataset_found = any(os.path.exists(f) for f in dataset_files)
    
    if not dataset_found:
        print("⚠️ Spotify dataset not found in test_data folder.")
        print("Please follow the instructions in kaggle_setup_instructions.md")
        print("or place the dataset file in the test_data folder.")
        print("\nProceeding with sample dataset for testing...")
    
    # Run the test
    results = await test_workflow_on_spotify()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if results['success']:
        print("✅ Workflow test completed successfully!")
        print(f"⏱️ Execution time: {results['execution_time']:.2f} seconds")
        print(f"🤖 Completed agents: {results['completed_agents']}")
        print(f"❌ Failed agents: {results['failed_agents']}")
        
        if results['model_performance']:
            perf = results['model_performance']
            print(f"🎯 Model accuracy: {perf.get('accuracy', 0):.3f}")
            print(f"🎯 Model F1-score: {perf.get('f1_score', 0):.3f}")
        
        if results['artifacts']:
            print(f"📁 Generated artifacts: {len(results['artifacts'])}")
        
        if results['errors']:
            print(f"⚠️ Errors: {len(results['errors'])}")
    else:
        print("❌ Workflow test failed!")
        print(f"Error: {results['error']}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
