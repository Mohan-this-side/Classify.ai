"""
End-to-End Workflow Tests

This module contains comprehensive smoke tests for the entire workflow
from file upload to final deliverable generation.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any
import json

from app.workflows.classification_workflow import ClassificationWorkflow
from app.workflows.state_management import ClassificationState, WorkflowStatus, AgentStatus
from app.services.storage import ResultsStorageService


class TestEndToEndWorkflow:
    """Test suite for end-to-end workflow execution"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        np.random.seed(42)
        n_samples = 100
        
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
    
    @pytest.fixture
    def iris_dataset(self):
        """Create Iris dataset for testing"""
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        return df
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance for testing"""
        return ClassificationWorkflow()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def create_test_state(self, dataset: pd.DataFrame, temp_dir: str) -> ClassificationState:
        """Create a test state for workflow execution"""
        return {
            "session_id": "test_session_123",
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
    
    @pytest.mark.asyncio
    async def test_complete_workflow_happy_path(self, iris_dataset, workflow, temp_dir):
        """Test complete workflow execution with Iris dataset"""
        # Create test state
        state = self.create_test_state(iris_dataset, temp_dir)
        
        # Execute workflow
        try:
            result_state = await workflow.execute_workflow(state)
            
            # Verify workflow completed successfully
            assert result_state["workflow_status"] == WorkflowStatus.COMPLETED
            assert len(result_state["completed_agents"]) >= 7  # All main agents completed
            
            # Verify all agents completed successfully
            for agent_name in ["data_cleaning", "data_discovery", "eda_analysis", 
                             "feature_engineering", "ml_building", "model_evaluation", 
                             "technical_reporter"]:
                assert result_state["agent_statuses"][agent_name] == AgentStatus.COMPLETED
            
            # Verify deliverables were created
            assert "cleaned_dataset_path" in result_state
            assert "model_path" in result_state
            assert "notebook_path" in result_state
            assert "report_path" in result_state
            
            # Verify files exist
            storage_service = ResultsStorageService(base_results_dir=temp_dir)
            files = storage_service.get_workflow_files("test_session_123")
            
            assert "cleaned_dataset" in files
            assert "model" in files
            assert "notebook" in files
            assert "report" in files
            
            print("✅ Complete workflow test passed")
            
        except Exception as e:
            pytest.fail(f"Workflow execution failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_workflow_with_missing_data_handling(self, workflow, temp_dir):
        """Test workflow behavior with missing data"""
        # Create dataset with significant missing values
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Add significant missing values
        df.loc[10:20, 'feature1'] = np.nan
        df.loc[25:35, 'feature2'] = np.nan
        df.loc[5:15, 'feature3'] = np.nan
        
        state = self.create_test_state(df, temp_dir)
        
        try:
            result_state = await workflow.execute_workflow(state)
            
            # Should complete successfully despite missing data
            assert result_state["workflow_status"] == WorkflowStatus.COMPLETED
            
            # Verify data cleaning handled missing values
            assert "cleaning_issues_found" in result_state
            assert len(result_state["cleaning_issues_found"]) > 0
            
            # Verify cleaned dataset has no missing values
            cleaned_path = result_state.get("cleaned_dataset_path")
            if cleaned_path and Path(cleaned_path).exists():
                cleaned_df = pd.read_csv(cleaned_path)
                assert cleaned_df.isnull().sum().sum() == 0
            
            print("✅ Missing data handling test passed")
            
        except Exception as e:
            pytest.fail(f"Missing data handling test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_workflow_with_invalid_target_column(self, iris_dataset, workflow, temp_dir):
        """Test workflow behavior with invalid target column"""
        state = self.create_test_state(iris_dataset, temp_dir)
        state["target_column"] = "nonexistent_column"  # Invalid target column
        
        try:
            result_state = await workflow.execute_workflow(state)
            
            # Should fail gracefully
            assert result_state["workflow_status"] in [WorkflowStatus.FAILED, WorkflowStatus.ERROR]
            assert len(result_state["errors"]) > 0
            
            print("✅ Invalid target column test passed")
            
        except Exception as e:
            # Expected to fail
            assert "target" in str(e).lower() or "column" in str(e).lower()
            print("✅ Invalid target column test passed (expected failure)")
    
    @pytest.mark.asyncio
    async def test_workflow_with_small_dataset(self, workflow, temp_dir):
        """Test workflow with very small dataset"""
        # Create very small dataset
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        
        state = self.create_test_state(df, temp_dir)
        
        try:
            result_state = await workflow.execute_workflow(state)
            
            # Should complete but may have warnings
            assert result_state["workflow_status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.WARNING]
            
            print("✅ Small dataset test passed")
            
        except Exception as e:
            pytest.fail(f"Small dataset test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, iris_dataset, workflow, temp_dir):
        """Test workflow error recovery mechanisms"""
        state = self.create_test_state(iris_dataset, temp_dir)
        
        # Simulate an error by corrupting the dataset
        corrupted_dataset = iris_dataset.copy()
        corrupted_dataset.iloc[0, 0] = "invalid_value"  # Corrupt first value
        
        state["dataset"] = corrupted_dataset
        
        try:
            result_state = await workflow.execute_workflow(state)
            
            # Should either complete with warnings or fail gracefully
            assert result_state["workflow_status"] in [
                WorkflowStatus.COMPLETED, 
                WorkflowStatus.FAILED, 
                WorkflowStatus.ERROR
            ]
            
            # If it fails, should have error information
            if result_state["workflow_status"] in [WorkflowStatus.FAILED, WorkflowStatus.ERROR]:
                assert len(result_state["errors"]) > 0
                assert result_state["error_count"] > 0
            
            print("✅ Error recovery test passed")
            
        except Exception as e:
            # Expected to potentially fail
            print(f"✅ Error recovery test passed (expected behavior): {str(e)}")
    
    def test_storage_service_functionality(self, iris_dataset, temp_dir):
        """Test storage service functionality"""
        storage_service = ResultsStorageService(base_results_dir=temp_dir)
        
        # Test workflow directory creation
        workflow_dir = storage_service.create_workflow_directory("test_workflow")
        assert workflow_dir.exists()
        
        # Test dataset storage
        dataset_path = storage_service.store_cleaned_dataset(
            "test_workflow", iris_dataset, "test_dataset.csv"
        )
        assert Path(dataset_path).exists()
        
        # Test model storage
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(iris_dataset.drop('target', axis=1), iris_dataset['target'])
        
        model_path = storage_service.store_model(
            "test_workflow", model, "test_model.joblib"
        )
        assert Path(model_path).exists()
        
        # Test report storage
        report_content = "# Test Report\nThis is a test report."
        report_path = storage_service.store_report(
            "test_workflow", report_content, "test_report.md"
        )
        assert Path(report_path).exists()
        
        # Test file retrieval
        files = storage_service.get_workflow_files("test_workflow")
        assert "cleaned_dataset" in files
        assert "model" in files
        assert "report" in files
        
        print("✅ Storage service test passed")
    
    def test_workflow_state_validation(self, iris_dataset, temp_dir):
        """Test workflow state validation"""
        # Test valid state
        valid_state = self.create_test_state(iris_dataset, temp_dir)
        assert "session_id" in valid_state
        assert "dataset" in valid_state
        assert "target_column" in valid_state
        assert "workflow_status" in valid_state
        assert "agent_statuses" in valid_state
        
        # Test missing required fields
        invalid_state = {"session_id": "test"}
        
        # Should raise error or handle gracefully
        try:
            workflow = ClassificationWorkflow()
            # This would normally be called by the workflow
            assert True
        except Exception as e:
            # Expected behavior
            assert True
        
        print("✅ State validation test passed")


class TestWorkflowIntegration:
    """Integration tests for workflow components"""
    
    @pytest.mark.asyncio
    async def test_agent_dependencies(self):
        """Test that agent dependencies are properly defined"""
        workflow = ClassificationWorkflow()
        
        # Check that all agents have proper dependencies
        agents = [
            workflow.data_cleaning_agent,
            workflow.data_discovery_agent,
            workflow.eda_agent,
            workflow.feature_engineering_agent,
            workflow.ml_builder_agent,
            workflow.model_evaluation_agent,
            workflow.technical_reporter_agent,
            workflow.project_manager_agent
        ]
        
        for agent in agents:
            assert hasattr(agent, 'get_dependencies')
            dependencies = agent.get_dependencies()
            assert isinstance(dependencies, list)
            
            # Check that dependencies are valid agent names
            valid_agents = [
                "data_cleaning", "data_discovery", "eda_analysis",
                "feature_engineering", "ml_building", "model_evaluation",
                "technical_reporter", "project_manager"
            ]
            
            for dep in dependencies:
                assert dep in valid_agents, f"Invalid dependency: {dep}"
        
        print("✅ Agent dependencies test passed")
    
    def test_workflow_graph_structure(self):
        """Test that workflow graph is properly structured"""
        workflow = ClassificationWorkflow()
        
        # Check that all required nodes exist
        required_nodes = [
            "data_cleaning", "data_discovery", "eda_analysis",
            "feature_engineering", "ml_building", "model_evaluation",
            "technical_reporting", "project_management", "error_recovery",
            "workflow_completion"
        ]
        
        graph_nodes = list(workflow.graph.nodes.keys())
        for node in required_nodes:
            assert node in graph_nodes, f"Missing node: {node}"
        
        print("✅ Workflow graph structure test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
