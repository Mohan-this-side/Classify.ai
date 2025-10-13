#!/usr/bin/env python3
"""
🎵 Complete Multi-Agent Workflow Test on Spotify Churn Dataset

This script tests the complete multi-agent workflow on the real Spotify churn dataset
to evaluate the performance of all agents working together.
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

# Import all agents
from app.agents.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.agents.data_discovery_agent import DataDiscoveryAgent
from app.agents.eda_agent import EDAAgent
from app.agents.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_builder_agent import MLBuilderAgent
from app.agents.model_evaluation_agent import ModelEvaluationAgent
from app.agents.technical_reporter_agent import TechnicalReporterAgent

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

async def test_agent(agent, agent_name, state, dataset_type=None):
    """Test a single agent"""
    print(f"\n🤖 Testing {agent_name}...")
    print("-" * 50)
    
    try:
        start_time = datetime.now()
        
        # Execute the agent
        result = await agent.execute(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"✅ {agent_name} completed in {execution_time:.2f} seconds")
        
        # Get the processed dataset if applicable
        if dataset_type:
            processed_df = state_manager.get_dataset(result, dataset_type)
            if processed_df is not None:
                print(f"📊 Output dataset shape: {processed_df.shape}")
                print(f"📊 Output dataset columns: {list(processed_df.columns)}")
        
        # Show any artifacts or results
        artifacts = result.get('output_artifacts', {})
        if artifacts:
            print(f"📁 Generated artifacts: {list(artifacts.keys())}")
        
        # Show any errors or warnings
        errors = result.get('errors', [])
        warnings = result.get('warnings', [])
        
        if errors:
            print(f"⚠️ Errors: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        if warnings:
            print(f"⚠️ Warnings: {len(warnings)}")
            for warning in warnings[:3]:  # Show first 3 warnings
                print(f"  - {warning}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'result': result,
            'artifacts': artifacts,
            'errors': errors,
            'warnings': warnings
        }
        
    except Exception as e:
        print(f"❌ {agent_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'execution_time': 0
        }

async def test_complete_workflow():
    """Test the complete multi-agent workflow on Spotify dataset"""
    
    print("🎵 Testing Complete Multi-Agent Workflow on Spotify Churn Dataset")
    print("=" * 80)
    
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
    print(f"\n📋 Sample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Create initial state
    state = {
        "session_id": f"spotify_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "dataset_id": "spotify_churn",
        "target_column": "is_churned",
        "user_description": "To predict whether a Spotify user will churn (cancel subscription) or remain active.",
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
    
    # Store original dataset
    from app.workflows.state_management import state_manager
    state_manager.store_dataset(state, df, "original")
    
    # Initialize all agents
    print(f"\n🚀 Initializing Multi-Agent System...")
    agents = {
        "Enhanced Data Cleaning Agent": EnhancedDataCleaningAgent(),
        "Data Discovery Agent": DataDiscoveryAgent(),
        "EDA Analysis Agent": EDAAgent(),
        "Feature Engineering Agent": FeatureEngineeringAgent(),
        "ML Builder Agent": MLBuilderAgent(),
        "Model Evaluation Agent": ModelEvaluationAgent(),
        "Technical Reporter Agent": TechnicalReporterAgent()
    }
    
    # Test each agent in sequence
    results = {}
    current_state = state.copy()
    
    # 1. Data Cleaning Agent
    agent_result = await test_agent(
        agents["Enhanced Data Cleaning Agent"], 
        "Enhanced Data Cleaning Agent", 
        current_state, 
        "cleaned"
    )
    results["data_cleaning"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # 2. Data Discovery Agent
    agent_result = await test_agent(
        agents["Data Discovery Agent"], 
        "Data Discovery Agent", 
        current_state
    )
    results["data_discovery"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # 3. EDA Analysis Agent
    agent_result = await test_agent(
        agents["EDA Analysis Agent"], 
        "EDA Analysis Agent", 
        current_state
    )
    results["eda_analysis"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # 4. Feature Engineering Agent
    agent_result = await test_agent(
        agents["Feature Engineering Agent"], 
        "Feature Engineering Agent", 
        current_state, 
        "feature_engineered"
    )
    results["feature_engineering"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # 5. ML Builder Agent
    agent_result = await test_agent(
        agents["ML Builder Agent"], 
        "ML Builder Agent", 
        current_state
    )
    results["ml_building"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # 6. Model Evaluation Agent
    agent_result = await test_agent(
        agents["Model Evaluation Agent"], 
        "Model Evaluation Agent", 
        current_state
    )
    results["model_evaluation"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # 7. Technical Reporter Agent
    agent_result = await test_agent(
        agents["Technical Reporter Agent"], 
        "Technical Reporter Agent", 
        current_state
    )
    results["technical_reporting"] = agent_result
    if agent_result['success']:
        current_state = agent_result['result']
    
    # Final analysis
    print("\n" + "=" * 80)
    print("📊 COMPLETE WORKFLOW RESULTS")
    print("=" * 80)
    
    successful_agents = [name for name, result in results.items() if result['success']]
    failed_agents = [name for name, result in results.items() if not result['success']]
    
    print(f"✅ Successful agents: {len(successful_agents)}/{len(results)}")
    print(f"❌ Failed agents: {len(failed_agents)}/{len(results)}")
    
    total_time = sum(result['execution_time'] for result in results.values())
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    
    print(f"\n📋 Individual Agent Results:")
    for agent_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        time = result['execution_time']
        print(f"  {status} {agent_name}: {time:.2f}s")
        
        if not result['success']:
            print(f"    Error: {result['error']}")
    
    # Check for final artifacts
    final_artifacts = current_state.get('output_artifacts', {})
    if final_artifacts:
        print(f"\n📁 Final Artifacts Generated:")
        for artifact_type, artifact_info in final_artifacts.items():
            print(f"  - {artifact_type}: {artifact_info}")
    
    # Check for model performance
    if 'model_performance' in current_state:
        performance = current_state['model_performance']
        print(f"\n🎯 Final Model Performance:")
        print(f"  Best model: {performance.get('best_model', 'N/A')}")
        print(f"  Accuracy: {performance.get('accuracy', 0):.3f}")
        print(f"  Precision: {performance.get('precision', 0):.3f}")
        print(f"  Recall: {performance.get('recall', 0):.3f}")
        print(f"  F1-Score: {performance.get('f1_score', 0):.3f}")
    
    # Save results
    results_file = f"test_data/spotify_complete_workflow_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Prepare results for JSON serialization
    json_results = {}
    for agent_name, result in results.items():
        json_results[agent_name] = {
            'success': result['success'],
            'execution_time': result['execution_time'],
            'artifacts_count': len(result.get('artifacts', {})),
            'errors_count': len(result.get('errors', [])),
            'warnings_count': len(result.get('warnings', []))
        }
        if not result['success']:
            json_results[agent_name]['error'] = str(result['error'])
    
    json_results['summary'] = {
        'total_agents': len(results),
        'successful_agents': len(successful_agents),
        'failed_agents': len(failed_agents),
        'total_execution_time': total_time,
        'success_rate': len(successful_agents) / len(results)
    }
    
    with open(results_file, 'w') as f:
        import json
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return {
        'success': len(successful_agents) > 0,
        'successful_agents': len(successful_agents),
        'failed_agents': len(failed_agents),
        'total_time': total_time,
        'results': results
    }

async def main():
    """Main test function"""
    print("🎵 Spotify Churn Analysis - Complete Multi-Agent Workflow Test")
    print("=" * 80)
    
    # Run the complete workflow test
    results = await test_complete_workflow()
    
    if results:
        print(f"\n🎉 Complete workflow test finished!")
        print(f"Success rate: {results['successful_agents']}/{results['successful_agents'] + results['failed_agents']}")
        print(f"Total time: {results['total_time']:.2f} seconds")
    else:
        print("❌ Workflow test failed!")

if __name__ == "__main__":
    asyncio.run(main())
