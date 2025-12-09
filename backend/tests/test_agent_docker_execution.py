"""
Comprehensive Agent Docker Execution Test Suite

This test suite validates that each agent:
1. Generates valid, executable code
2. Successfully executes in Docker sandbox
3. Produces meaningful results
4. Handles errors gracefully
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.data_discovery.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
from app.agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
from app.services.llm_service import LLMService
from app.services.sandbox_executor import SandboxExecutor
from app.services.storage import StorageService


# Test dataset - Heart Disease dataset (similar to what user uploaded)
def create_test_dataset():
    """Create a realistic test dataset"""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Age': np.random.randint(29, 80, n_samples),
        'Sex': np.random.choice(['M', 'F'], n_samples),
        'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
        'RestingBP': np.random.randint(90, 200, n_samples),
        'Cholesterol': np.random.randint(100, 600, n_samples),
        'FastingBS': np.random.choice([0, 1], n_samples),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples),
        'MaxHR': np.random.randint(60, 210, n_samples),
        'ExerciseAngina': np.random.choice(['N', 'Y'], n_samples),
        'Oldpeak': np.random.uniform(0, 6, n_samples),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples),
        'HeartDisease': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values (realistic scenario)
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'Cholesterol'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_indices, 'RestingBP'] = np.random.randint(250, 300, len(outlier_indices))
    
    return df


class AgentDockerTestSuite:
    """Comprehensive test suite for agent Docker execution"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', 'dummy_key')
        self.llm_service = LLMService(user_api_key=self.api_key)
        self.sandbox_executor = SandboxExecutor()
        self.storage_service = StorageService()
        self.test_dataset = create_test_dataset()
        self.workflow_id = "test_workflow_12345"
        
    def create_test_state(self, target_column: str = 'HeartDisease') -> dict:
        """Create a test workflow state"""
        # Save test dataset
        temp_dir = tempfile.mkdtemp()
        dataset_path = os.path.join(temp_dir, 'test_dataset.csv')
        self.test_dataset.to_csv(dataset_path, index=False)
        
        # Copy dataset to sandbox
        self.sandbox_executor.prepare_dataset(dataset_path, self.workflow_id)
        
        return {
            'dataset_id': self.workflow_id,
            'target_column': target_column,
            'user_description': 'Heart disease prediction dataset',
            'api_key': self.api_key,
            'dataset_shape': self.test_dataset.shape,
            'dataset': self.test_dataset,
            'processed_dataset': self.test_dataset.copy(),
            'columns': list(self.test_dataset.columns),
            'data_types': {col: str(dtype) for col, dtype in self.test_dataset.dtypes.items()}
        }
    
    async def test_agent_layer2_execution(self, agent, state: dict, agent_name: str):
        """Test Layer 2 execution for a specific agent"""
        print(f"\n{'='*80}")
        print(f"Testing {agent_name} Layer 2 Execution")
        print(f"{'='*80}")
        
        try:
            # Step 1: Execute Layer 1
            print(f"\n[1/5] Executing Layer 1...")
            layer1_results = await agent.perform_layer1_analysis(state)
            print(f"✅ Layer 1 completed: {len(layer1_results)} keys")
            
            # Step 2: Generate Layer 2 code prompt
            print(f"\n[2/5] Generating Layer 2 code prompt...")
            prompt = agent.generate_layer2_code(layer1_results, state)
            print(f"✅ Prompt generated: {len(prompt)} characters")
            print(f"   Preview: {prompt[:200]}...")
            
            # Step 3: Generate code with LLM
            print(f"\n[3/5] Calling LLM to generate code...")
            llm_response = await self.llm_service.generate_code(
                prompt=prompt,
                context=layer1_results,
                code_type=agent_name
            )
            
            generated_code = llm_response.get("code")
            if not generated_code:
                raise ValueError("LLM did not generate any code")
            
            print(f"✅ Code generated: {len(generated_code)} characters")
            
            # Step 4: Validate code syntax
            print(f"\n[4/5] Validating code syntax...")
            from app.services.code_validator import CodeValidator
            validator = CodeValidator()
            validation_result = validator.validate_code(generated_code)
            
            if not validation_result.is_valid:
                print(f"❌ Code validation failed:")
                for error in validation_result.errors:
                    print(f"   - {error}")
                return False
            
            print(f"✅ Code validation passed")
            if validation_result.warnings:
                print(f"   Warnings: {len(validation_result.warnings)}")
            
            # Step 5: Execute in Docker sandbox
            print(f"\n[5/5] Executing code in Docker sandbox...")
            sandbox_result = await agent.execute_layer2_in_sandbox(
                generated_code,
                layer1_results,
                state
            )
            
            status = sandbox_result.get('status', 'UNKNOWN')
            print(f"✅ Sandbox execution completed: {status}")
            
            if status == 'SUCCESS':
                output = sandbox_result.get('output', '')
                print(f"   Output length: {len(output)} characters")
                print(f"   Execution time: {sandbox_result.get('execution_time', 0):.2f}s")
                return True
            else:
                error = sandbox_result.get('error', 'Unknown error')
                print(f"❌ Sandbox execution failed: {error}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run tests for all agents"""
        print("\n" + "="*80)
        print("COMPREHENSIVE AGENT DOCKER EXECUTION TEST SUITE")
        print("="*80)
        
        state = self.create_test_state()
        
        agents_to_test = [
            (DataDiscoveryAgent(), "data_discovery"),
            (EDAAgent(), "eda"),
            (EnhancedDataCleaningAgent(), "data_cleaning"),
            (FeatureEngineeringAgent(), "feature_engineering"),
            (MLBuilderAgent(), "ml_builder"),
            (ModelEvaluationAgent(), "model_evaluation"),
        ]
        
        results = {}
        
        for agent, agent_name in agents_to_test:
            agent.llm_service = self.llm_service
            agent.sandbox_executor = self.sandbox_executor
            
            success = await self.test_agent_layer2_execution(agent, state, agent_name)
            results[agent_name] = success
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        for agent_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{agent_name:30s} {status}")
        
        all_passed = all(results.values())
        print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return results


async def main():
    """Main test runner"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set. Using dummy key (tests will fail at LLM step)")
        print("   Set GEMINI_API_KEY environment variable to run full tests")
    
    test_suite = AgentDockerTestSuite(api_key=api_key)
    results = await test_suite.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

