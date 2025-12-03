#!/usr/bin/env python3
"""
Comprehensive End-to-End Workflow Test

Tests the complete classification workflow:
1. All agents execute correctly
2. Layer 1 and Layer 2 execution
3. Data passing between agents
4. Plot generation and rendering
5. ML model building and evaluation
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
from app.agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
from app.workflows.state_management import ClassificationState, StateManager
from app.services.llm_service import LLMService
from app.services.sandbox_executor import SandboxExecutor
# Storage service not needed for testing - we'll use state manager directly


def create_realistic_test_dataset():
    """Create a realistic test dataset similar to heart disease dataset"""
    np.random.seed(42)
    n_samples = 200  # Reasonable size for testing
    
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
    
    # Add some missing values (5% missing)
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'Cholesterol'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'RestingBP'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_indices, 'Cholesterol'] = np.random.randint(600, 800, len(outlier_indices))
    
    return df


class FullWorkflowTester:
    """Comprehensive workflow tester"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', 'dummy_key')
        self.state_manager = StateManager()
        self.workflow_id = f"test_workflow_{int(datetime.now().timestamp())}"
        
        # Initialize agents
        self.data_discovery_agent = DataDiscoveryAgent()
        self.eda_agent = EDAAgent()
        self.data_cleaning_agent = EnhancedDataCleaningAgent()
        self.feature_engineering_agent = FeatureEngineeringAgent()
        self.ml_builder_agent = MLBuilderAgent()
        self.model_evaluation_agent = ModelEvaluationAgent()
        
        # Initialize services
        self.llm_service = LLMService(api_key=self.api_key)
        self.sandbox_executor = SandboxExecutor()
        
        # Set services on agents
        for agent in [self.data_discovery_agent, self.eda_agent, self.data_cleaning_agent,
                     self.feature_engineering_agent, self.ml_builder_agent, self.model_evaluation_agent]:
            agent.llm_service = self.llm_service
            agent.sandbox_executor = self.sandbox_executor
        
        self.results = {
            'workflow_id': self.workflow_id,
            'agents_tested': [],
            'layer1_results': {},
            'layer2_results': {},
            'data_passing': {},
            'plot_generation': {},
            'errors': [],
            'warnings': []
        }
    
    def log_result(self, agent_name: str, test_name: str, status: str, details: dict = None):
        """Log test result"""
        result = {
            'agent': agent_name,
            'test': test_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        if status == 'ERROR':
            self.results['errors'].append(result)
        elif status == 'WARNING':
            self.results['warnings'].append(result)
        
        print(f"\n{'='*80}")
        print(f"{'✅' if status == 'PASS' else '❌' if status == 'ERROR' else '⚠️'} {agent_name} - {test_name}: {status}")
        print(f"{'='*80}")
        if details:
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    async def test_agent_layer1(self, agent, agent_name: str, state: ClassificationState):
        """Test Layer 1 execution for an agent"""
        try:
            print(f"\n[Layer 1] Testing {agent_name}...")
            layer1_results = await agent.perform_layer1_analysis(state)
            
            if layer1_results and len(layer1_results) > 0:
                self.results['layer1_results'][agent_name] = {
                    'status': 'PASS',
                    'result_keys': list(layer1_results.keys()),
                    'result_count': len(layer1_results)
                }
                self.log_result(agent_name, 'Layer 1', 'PASS', {
                    'result_keys': len(layer1_results),
                    'keys': list(layer1_results.keys())[:5]
                })
                return layer1_results, True
            else:
                self.log_result(agent_name, 'Layer 1', 'ERROR', {'error': 'No results returned'})
                return None, False
        except Exception as e:
            self.log_result(agent_name, 'Layer 1', 'ERROR', {'error': str(e)})
            return None, False
    
    async def test_agent_layer2(self, agent, agent_name: str, state: ClassificationState, layer1_results: dict):
        """Test Layer 2 execution for an agent"""
        if self.api_key == 'dummy_key':
            self.log_result(agent_name, 'Layer 2', 'SKIPPED', {'reason': 'No API key'})
            return None, False
        
        try:
            print(f"\n[Layer 2] Testing {agent_name}...")
            
            # Generate prompt
            prompt = agent.generate_layer2_code(layer1_results, state)
            if not prompt or len(prompt) < 100:
                self.log_result(agent_name, 'Layer 2 Prompt', 'ERROR', {'error': 'Prompt too short'})
                return None, False
            
            # Generate code
            llm_response = await self.llm_service.generate_code(
                prompt=prompt,
                context=layer1_results,
                code_type=agent_name
            )
            
            generated_code = llm_response.get("code")
            if not generated_code or len(generated_code) < 100:
                self.log_result(agent_name, 'Layer 2 Code Generation', 'ERROR', {'error': 'No code generated'})
                return None, False
            
            # Validate code
            from app.services.code_validator import CodeValidator
            validator = CodeValidator()
            validation_result = validator.validate(generated_code)
            
            if not validation_result.is_valid:
                self.log_result(agent_name, 'Layer 2 Validation', 'ERROR', {
                    'error': 'Code validation failed',
                    'errors': validation_result.errors[:3]
                })
                return None, False
            
            # Execute in sandbox
            sandbox_result = await agent.execute_layer2_in_sandbox(
                generated_code,
                layer1_results,
                state
            )
            
            if sandbox_result and sandbox_result.get('status') == 'SUCCESS':
                self.results['layer2_results'][agent_name] = {
                    'status': 'PASS',
                    'execution_time': sandbox_result.get('execution_time', 0)
                }
                self.log_result(agent_name, 'Layer 2 Execution', 'PASS', {
                    'execution_time': sandbox_result.get('execution_time', 0),
                    'output_length': len(sandbox_result.get('output', ''))
                })
                return sandbox_result, True
            else:
                error = sandbox_result.get('error', 'Unknown error') if sandbox_result else 'No result'
                self.log_result(agent_name, 'Layer 2 Execution', 'ERROR', {'error': error})
                return None, False
                
        except Exception as e:
            self.log_result(agent_name, 'Layer 2', 'ERROR', {'error': str(e)})
            return None, False
    
    async def test_data_passing(self, from_agent: str, to_agent: str, state: ClassificationState):
        """Test data passing between agents"""
        try:
            # Check if data from previous agent is available
            data_checks = {
                'data_discovery': ['discovery_results'],
                'data_cleaning': ['cleaned_dataset', 'cleaning_summary'],
                'eda': ['eda_plots', 'statistical_summary'],
                'feature_engineering': ['engineered_features', 'feature_transformations'],
                'ml_builder': ['best_model', 'model_selection_results'],
                'model_evaluation': ['evaluation_metrics', 'confusion_matrix']
            }
            
            required_keys = data_checks.get(from_agent, [])
            missing_keys = []
            
            for key in required_keys:
                if key not in state or state[key] is None:
                    missing_keys.append(key)
            
            if missing_keys:
                self.log_result(f"{from_agent}->{to_agent}", 'Data Passing', 'WARNING', {
                    'missing_keys': missing_keys
                })
                return False
            else:
                self.log_result(f"{from_agent}->{to_agent}", 'Data Passing', 'PASS', {
                    'keys_passed': required_keys
                })
                return True
        except Exception as e:
            self.log_result(f"{from_agent}->{to_agent}", 'Data Passing', 'ERROR', {'error': str(e)})
            return False
    
    async def test_plot_generation(self, agent_name: str, state: ClassificationState):
        """Test plot generation"""
        try:
            plot_keys = {
                'eda': 'eda_plots',
                'model_evaluation': 'evaluation_plots'
            }
            
            plot_key = plot_keys.get(agent_name)
            if not plot_key:
                return True  # Agent doesn't generate plots
            
            plots = state.get(plot_key, [])
            if isinstance(plots, list) and len(plots) > 0:
                # Check if plots are accessible
                accessible_plots = []
                for plot_path in plots:
                    if isinstance(plot_path, str) and (plot_path.endswith('.png') or '/api/workflow/plot/' in plot_path):
                        accessible_plots.append(plot_path)
                
                if accessible_plots:
                    self.results['plot_generation'][agent_name] = {
                        'status': 'PASS',
                        'plot_count': len(accessible_plots),
                        'plots': accessible_plots[:3]  # First 3 plots
                    }
                    self.log_result(agent_name, 'Plot Generation', 'PASS', {
                        'plot_count': len(accessible_plots),
                        'sample_plots': accessible_plots[:2]
                    })
                    return True
                else:
                    self.log_result(agent_name, 'Plot Generation', 'WARNING', {
                        'error': 'Plots generated but not accessible'
                    })
                    return False
            else:
                self.log_result(agent_name, 'Plot Generation', 'WARNING', {
                    'error': 'No plots generated'
                })
                return False
        except Exception as e:
            self.log_result(agent_name, 'Plot Generation', 'ERROR', {'error': str(e)})
            return False
    
    async def run_full_workflow_test(self):
        """Run complete workflow test"""
        print("\n" + "="*80)
        print("COMPREHENSIVE END-TO-END WORKFLOW TEST")
        print("="*80)
        
        # Create test dataset
        test_df = create_realistic_test_dataset()
        
        # Save dataset to temp file
        temp_dir = tempfile.mkdtemp()
        dataset_path = os.path.join(temp_dir, 'test_dataset.csv')
        test_df.to_csv(dataset_path, index=False)
        
        # Prepare dataset in sandbox
        try:
            self.sandbox_executor.prepare_dataset(dataset_path, self.workflow_id)
        except Exception as e:
            print(f"⚠️ Warning: Could not prepare dataset in sandbox: {e}")
        
        # Initialize state
        state = self.state_manager.initialize_state(
            session_id=self.workflow_id,
            dataset_id=self.workflow_id,
            target_column='HeartDisease',
            user_description='Heart disease prediction dataset for testing',
            api_key=self.api_key,
            original_dataset=test_df
        )
        
        # Store dataset in state_manager's external storage (agents access via state_manager.get_dataset)
        from app.workflows.state_management import state_manager
        state_manager.external_storage[self.workflow_id] = {
            'original': test_df,
            'cleaned': test_df.copy(),
            'processed': test_df.copy()
        }
        
        # Also set in state directly for agents that access directly
        state['dataset'] = test_df
        state['processed_dataset'] = test_df.copy()
        state['original_dataset'] = test_df.copy()
        state['cleaned_dataset'] = test_df.copy()
        
        # Test agents in sequence
        agents_to_test = [
            ('data_discovery', self.data_discovery_agent),
            ('data_cleaning', self.data_cleaning_agent),
            ('eda', self.eda_agent),
            ('feature_engineering', self.feature_engineering_agent),
            ('ml_builder', self.ml_builder_agent),
            ('model_evaluation', self.model_evaluation_agent),
        ]
        
        previous_agent = None
        
        for agent_name, agent in agents_to_test:
            print(f"\n{'#'*80}")
            print(f"Testing Agent: {agent_name}")
            print(f"{'#'*80}")
            
            # Test Layer 1
            layer1_results, layer1_pass = await self.test_agent_layer1(agent, agent_name, state)
            
            if not layer1_pass:
                print(f"❌ {agent_name} Layer 1 failed, skipping Layer 2")
                continue
            
            # Update state with Layer 1 results
            if layer1_results:
                # Update state based on agent type
                if agent_name == 'data_discovery':
                    state['discovery_results'] = layer1_results
                elif agent_name == 'data_cleaning':
                    if 'cleaned_dataset' in layer1_results:
                        state['cleaned_dataset'] = layer1_results['cleaned_dataset']
                        state['processed_dataset'] = layer1_results['cleaned_dataset']
                    state['cleaning_summary'] = layer1_results.get('cleaning_summary')
                elif agent_name == 'eda':
                    state['eda_plots'] = layer1_results.get('plot_paths', [])
                    state['statistical_summary'] = layer1_results.get('statistical_summary')
                elif agent_name == 'feature_engineering':
                    state['engineered_features'] = layer1_results.get('engineered_features', [])
                    state['feature_transformations'] = layer1_results.get('feature_transformations', {})
                elif agent_name == 'ml_builder':
                    state['model_selection_results'] = layer1_results.get('model_selection_results')
                    state['best_model'] = layer1_results.get('best_model')
                elif agent_name == 'model_evaluation':
                    state['evaluation_metrics'] = layer1_results.get('evaluation_metrics')
                    state['confusion_matrix'] = layer1_results.get('confusion_matrix')
            
            # Test Layer 2
            layer2_results, layer2_pass = await self.test_agent_layer2(agent, agent_name, state, layer1_results)
            
            # Test data passing
            if previous_agent:
                await self.test_data_passing(previous_agent, agent_name, state)
            
            # Test plot generation
            if agent_name in ['eda', 'model_evaluation']:
                await self.test_plot_generation(agent_name, state)
            
            self.results['agents_tested'].append(agent_name)
            previous_agent = agent_name
            
            # Small delay between agents
            await asyncio.sleep(1)
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\nAgents Tested: {len(self.results['agents_tested'])}")
        print(f"  {', '.join(self.results['agents_tested'])}")
        
        print(f"\nLayer 1 Results:")
        for agent, result in self.results['layer1_results'].items():
            status = result['status']
            print(f"  {agent}: {status} ({result['result_count']} keys)")
        
        print(f"\nLayer 2 Results:")
        for agent, result in self.results['layer2_results'].items():
            status = result['status']
            exec_time = result.get('execution_time', 0)
            print(f"  {agent}: {status} ({exec_time:.2f}s)")
        
        print(f"\nPlot Generation:")
        for agent, result in self.results['plot_generation'].items():
            plot_count = result.get('plot_count', 0)
            print(f"  {agent}: {plot_count} plots")
        
        print(f"\nErrors: {len(self.results['errors'])}")
        if self.results['errors']:
            for error in self.results['errors'][:5]:  # Show first 5
                print(f"  - {error['agent']}: {error['test']} - {error['details'].get('error', 'Unknown')}")
        
        print(f"\nWarnings: {len(self.results['warnings'])}")
        if self.results['warnings']:
            for warning in self.results['warnings'][:5]:  # Show first 5
                print(f"  - {warning['agent']}: {warning['test']}")
        
        print("="*80)


async def main():
    """Main test runner"""
    api_key = os.getenv('GEMINI_API_KEY', 'dummy_key')
    
    if api_key == 'dummy_key':
        print("\n⚠️  WARNING: GEMINI_API_KEY not set.")
        print("   Layer 2 tests will be skipped.")
        print("   Set GEMINI_API_KEY environment variable to run full tests.\n")
    
    tester = FullWorkflowTester(api_key=api_key)
    results = await tester.run_full_workflow_test()
    
    # Return exit code
    if len(tester.results['errors']) == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {len(tester.results['errors'])} ERRORS FOUND")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

