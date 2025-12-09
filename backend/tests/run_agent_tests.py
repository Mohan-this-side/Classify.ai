#!/usr/bin/env python3
"""
Quick Agent Docker Execution Test Runner

Tests each agent's Layer 2 execution to verify Docker fixes are working.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from app.agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
from app.agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
from app.services.llm_service import LLMService
from app.services.sandbox_executor import SandboxExecutor


def create_test_dataset():
    """Create a realistic test dataset"""
    np.random.seed(42)
    n_samples = 100  # Smaller for faster testing
    
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
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'Cholesterol'] = np.nan
    
    return df


async def test_agent(agent, agent_name: str, state: dict, api_key: str):
    """Test a single agent"""
    print(f"\n{'='*80}")
    print(f"Testing: {agent_name}")
    print(f"{'='*80}")
    
    results = {
        'agent': agent_name,
        'layer1': False,
        'prompt_generation': False,
        'code_generation': False,
        'code_validation': False,
        'docker_execution': False,
        'errors': []
    }
    
    try:
        # Initialize agent services
        agent.llm_service = LLMService(api_key=api_key)
        agent.sandbox_executor = SandboxExecutor()
        
        # Step 1: Layer 1
        print("\n[1/5] Testing Layer 1 execution...")
        try:
            layer1_results = await agent.perform_layer1_analysis(state)
            print(f"   ✅ Layer 1 completed: {len(layer1_results)} result keys")
            results['layer1'] = True
        except Exception as e:
            print(f"   ❌ Layer 1 failed: {e}")
            results['errors'].append(f"Layer 1: {str(e)}")
            return results
        
        # Step 2: Prompt generation
        print("\n[2/5] Testing prompt generation...")
        try:
            prompt = agent.generate_layer2_code(layer1_results, state)
            if prompt and len(prompt) > 100:
                print(f"   ✅ Prompt generated: {len(prompt)} characters")
                results['prompt_generation'] = True
            else:
                print(f"   ❌ Prompt too short or empty")
                results['errors'].append("Prompt generation: Empty or too short")
                return results
        except Exception as e:
            print(f"   ❌ Prompt generation failed: {e}")
            results['errors'].append(f"Prompt generation: {str(e)}")
            return results
        
        # Step 3: Code generation (skip if no API key)
        print("\n[3/5] Testing code generation...")
        if api_key == 'dummy_key' or not api_key:
            print("   ⚠️  Skipping (no API key provided)")
            results['code_generation'] = None
        else:
            try:
                llm_response = await agent.llm_service.generate_code(
                    prompt=prompt,
                    context=layer1_results,
                    code_type=agent_name
                )
                generated_code = llm_response.get("code")
                if generated_code and len(generated_code) > 100:
                    print(f"   ✅ Code generated: {len(generated_code)} characters")
                    results['code_generation'] = True
                else:
                    print(f"   ❌ Code generation failed: No code returned")
                    results['errors'].append("Code generation: No code returned")
                    return results
            except Exception as e:
                print(f"   ❌ Code generation failed: {e}")
                results['errors'].append(f"Code generation: {str(e)}")
                return results
        
        # Step 4: Code validation (skip if no code generated)
        if results['code_generation']:
            print("\n[4/5] Testing code validation...")
            try:
                from app.services.code_validator import CodeValidator
                validator = CodeValidator()
                validation_result = validator.validate_code(generated_code)
                
                if validation_result.is_valid:
                    print(f"   ✅ Code validation passed")
                    if validation_result.warnings:
                        print(f"   ⚠️  Warnings: {len(validation_result.warnings)}")
                    results['code_validation'] = True
                else:
                    print(f"   ❌ Code validation failed:")
                    for error in validation_result.errors[:3]:  # Show first 3 errors
                        print(f"      - {error}")
                    results['errors'].append(f"Validation: {validation_result.errors[0] if validation_result.errors else 'Unknown'}")
                    return results
            except Exception as e:
                print(f"   ❌ Code validation error: {e}")
                results['errors'].append(f"Code validation: {str(e)}")
                return results
        
        # Step 5: Docker execution (skip if no code generated)
        if results['code_generation'] and results['code_validation']:
            print("\n[5/5] Testing Docker sandbox execution...")
            try:
                sandbox_result = await agent.execute_layer2_in_sandbox(
                    generated_code,
                    layer1_results,
                    state
                )
                
                status = sandbox_result.get('status', 'UNKNOWN')
                if status == 'SUCCESS':
                    print(f"   ✅ Docker execution successful")
                    print(f"      Execution time: {sandbox_result.get('execution_time', 0):.2f}s")
                    results['docker_execution'] = True
                else:
                    error = sandbox_result.get('error', 'Unknown error')
                    print(f"   ❌ Docker execution failed: {error}")
                    results['errors'].append(f"Docker execution: {error}")
            except Exception as e:
                print(f"   ❌ Docker execution error: {e}")
                results['errors'].append(f"Docker execution: {str(e)}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        results['errors'].append(f"Unexpected: {str(e)}")
    
    return results


async def main():
    """Run all agent tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE AGENT DOCKER EXECUTION TEST SUITE")
    print("="*80)
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY', 'dummy_key')
    if api_key == 'dummy_key':
        print("\n⚠️  WARNING: GEMINI_API_KEY not set. Code generation and Docker tests will be skipped.")
        print("   Set GEMINI_API_KEY environment variable to run full tests.\n")
    
    # Create test dataset and state
    test_df = create_test_dataset()
    workflow_id = "test_workflow_12345"
    
    # Prepare dataset for sandbox
    import tempfile
    temp_dir = tempfile.mkdtemp()
    dataset_path = os.path.join(temp_dir, 'test_dataset.csv')
    test_df.to_csv(dataset_path, index=False)
    
    # Initialize sandbox executor and prepare dataset
    sandbox_executor = SandboxExecutor()
    try:
        sandbox_executor.prepare_dataset(dataset_path, workflow_id)
    except Exception as e:
        print(f"⚠️  Warning: Could not prepare dataset in sandbox: {e}")
    
    state = {
        'dataset_id': workflow_id,
        'session_id': workflow_id,
        'target_column': 'HeartDisease',
        'user_description': 'Heart disease prediction dataset',
        'api_key': api_key,
        'dataset_shape': test_df.shape,
        'dataset': test_df,  # Original dataset
        'processed_dataset': test_df.copy(),  # Processed dataset (same for now)
        'original_dataset': test_df.copy(),  # Some agents need this
        'columns': list(test_df.columns),
        'data_types': {col: str(dtype) for col, dtype in test_df.dtypes.items()},
        'dataset_path': dataset_path,  # Path to dataset file
        'cleaned_dataset': test_df.copy(),  # For agents that need cleaned data
        'feature_engineered_dataset': test_df.copy(),  # For ML agents
    }
    
    # Test agents
    agents_to_test = [
        (DataDiscoveryAgent(), "data_discovery"),
        (EDAAgent(), "eda"),
        (EnhancedDataCleaningAgent(), "data_cleaning"),
        (FeatureEngineeringAgent(), "feature_engineering"),
        (MLBuilderAgent(), "ml_builder"),
        (ModelEvaluationAgent(), "model_evaluation"),
    ]
    
    all_results = []
    
    for agent, agent_name in agents_to_test:
        result = await test_agent(agent, agent_name, state, api_key)
        all_results.append(result)
        await asyncio.sleep(1)  # Small delay between tests
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Agent':<25} {'L1':<4} {'Prompt':<7} {'Code':<6} {'Valid':<6} {'Docker':<7} {'Status'}")
    print("-" * 80)
    
    for result in all_results:
        agent = result['agent']
        l1 = "✅" if result['layer1'] else "❌"
        prompt = "✅" if result['prompt_generation'] else ("⚠️" if result['prompt_generation'] is None else "❌")
        code = "✅" if result['code_generation'] else ("⚠️" if result['code_generation'] is None else "❌")
        valid = "✅" if result['code_validation'] else ("⚠️" if result['code_validation'] is None else "❌")
        docker = "✅" if result['docker_execution'] else ("⚠️" if result['docker_execution'] is None else "❌")
        
        if result['errors']:
            status = f"❌ {result['errors'][0][:30]}"
        elif result['docker_execution']:
            status = "✅ PASSED"
        elif result['code_validation']:
            status = "⚠️  No Docker test"
        elif result['code_generation']:
            status = "⚠️  Validation failed"
        elif result['prompt_generation']:
            status = "⚠️  No API key"
        else:
            status = "❌ FAILED"
        
        print(f"{agent:<25} {l1:<4} {prompt:<7} {code:<6} {valid:<6} {docker:<7} {status}")
    
    # Overall status
    print("\n" + "="*80)
    passed = sum(1 for r in all_results if r['layer1'] and r['prompt_generation'])
    total = len(all_results)
    
    print(f"\nOverall: {passed}/{total} agents passed Layer 1 and Prompt Generation")
    
    if api_key != 'dummy_key':
        docker_tested = sum(1 for r in all_results if r['docker_execution'] is not None)
        docker_passed = sum(1 for r in all_results if r['docker_execution'])
        print(f"Docker Execution: {docker_passed}/{docker_tested} agents passed")
    
    print("="*80 + "\n")
    
    # Return exit code
    if passed == total:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

