#!/usr/bin/env python3
"""
Complete System Smoke Test

This script performs a comprehensive smoke test of the entire system
to ensure all components are working correctly.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add backend to path
sys.path.append('backend')

from backend.app.workflows.classification_workflow import ClassificationWorkflow
from backend.app.workflows.state_management import ClassificationState, WorkflowStatus, AgentStatus
from backend.app.services.storage import ResultsStorageService


def create_test_dataset():
    """Create a test dataset for smoke testing"""
    np.random.seed(42)
    n_samples = 50
    
    # Create features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    
    # Create target with some relationship to features
    target = (feature1 * 0.5 + feature2 * 0.3 + feature3 * 0.2 + 
             np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'target': target
    })
    
    # Add some missing values for testing
    df.loc[5:10, 'feature1'] = np.nan
    df.loc[15:20, 'feature2'] = np.nan
    
    return df


def create_test_state(dataset, temp_dir):
    """Create a test state for workflow execution"""
    return {
        "session_id": "smoke_test_session",
        "dataset": dataset,
        "target_column": "target",
        "workflow_status": WorkflowStatus.PENDING,
        "agent_statuses": {
            "data_cleaning": AgentStatus.PENDING,
            "data_discovery": AgentStatus.PENDING,
            "eda_analysis": AgentStatus.PENDING,
            "feature_engineering": AgentStatus.PENDING,
            "ml_building": AgentStatus.PENDING,
            "model_evaluation": AgentStatus.PENDING,
            "technical_reporter": AgentStatus.PENDING,
            "project_manager": AgentStatus.PENDING
        },
        "completed_agents": [],
        "failed_agents": [],
        "errors": [],
        "error_count": 0,
        "progress": 0,
        "current_agent": None,
        "dataset_shape": dataset.shape,
        "upload_dir": temp_dir,
        "results_dir": temp_dir
    }


async def test_workflow_execution():
    """Test complete workflow execution"""
    print("🧪 Starting Complete System Smoke Test...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Create test dataset
        print("📊 Creating test dataset...")
        dataset = create_test_dataset()
        print(f"   Dataset shape: {dataset.shape}")
        print(f"   Target distribution: {dataset['target'].value_counts().to_dict()}")
        
        # Create workflow
        print("🔄 Initializing workflow...")
        workflow = ClassificationWorkflow()
        
        # Create test state
        print("📝 Creating test state...")
        state = create_test_state(dataset, temp_dir)
        
        # Execute workflow
        print("🚀 Executing workflow...")
        result_state = await workflow.execute_workflow(state)
        
        # Check results
        print("✅ Workflow execution completed!")
        print(f"   Final status: {result_state['workflow_status']}")
        print(f"   Completed agents: {len(result_state['completed_agents'])}")
        print(f"   Failed agents: {len(result_state['failed_agents'])}")
        print(f"   Errors: {len(result_state['errors'])}")
        
        # Verify deliverables
        print("📋 Checking deliverables...")
        storage_service = ResultsStorageService(base_results_dir=temp_dir)
        files = storage_service.get_workflow_files("smoke_test_session")
        
        deliverables = {
            "cleaned_dataset": "CSV file with cleaned data",
            "model": "Trained ML model",
            "notebook": "Jupyter notebook with analysis",
            "report": "Technical report"
        }
        
        for file_type, description in deliverables.items():
            if file_type in files:
                print(f"   ✅ {description}: {files[file_type]}")
            else:
                print(f"   ❌ Missing {description}")
        
        # Check for plots
        if "plots" in files:
            print(f"   ✅ Generated {len(files['plots'])} plots")
        else:
            print("   ⚠️  No plots generated")
        
        # Summary
        if result_state['workflow_status'] == WorkflowStatus.COMPLETED:
            print("\n🎉 SMOKE TEST PASSED! System is working correctly.")
            return True
        else:
            print(f"\n❌ SMOKE TEST FAILED! Workflow status: {result_state['workflow_status']}")
            if result_state['errors']:
                print("Errors:")
                for error in result_state['errors']:
                    print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"\n💥 SMOKE TEST FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_individual_components():
    """Test individual components"""
    print("\n🔧 Testing individual components...")
    
    # Test storage service
    print("📦 Testing storage service...")
    temp_dir = tempfile.mkdtemp()
    try:
        storage_service = ResultsStorageService(base_results_dir=temp_dir)
        
        # Test directory creation
        workflow_dir = storage_service.create_workflow_directory("test")
        assert workflow_dir.exists()
        print("   ✅ Directory creation works")
        
        # Test dataset storage
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        dataset_path = storage_service.store_cleaned_dataset("test", test_df)
        assert Path(dataset_path).exists()
        print("   ✅ Dataset storage works")
        
        # Test file retrieval
        files = storage_service.get_workflow_files("test")
        assert "cleaned_dataset" in files
        print("   ✅ File retrieval works")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("   ✅ Storage service test passed")
    
    # Test workflow initialization
    print("🔄 Testing workflow initialization...")
    try:
        workflow = ClassificationWorkflow()
        assert workflow is not None
        print("   ✅ Workflow initialization works")
    except Exception as e:
        print(f"   ❌ Workflow initialization failed: {e}")
        return False
    
    print("   ✅ Individual components test passed")
    return True


async def main():
    """Main test function"""
    print("🚀 DS Capstone Multi-Agent System - Complete Smoke Test")
    print("=" * 60)
    
    # Test individual components first
    components_ok = await test_individual_components()
    if not components_ok:
        print("\n❌ Component tests failed. Aborting full test.")
        return False
    
    # Test complete workflow
    workflow_ok = await test_workflow_execution()
    
    if workflow_ok:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        return True
    else:
        print("\n❌ TESTS FAILED! Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
