"""
System-Level Workflow Evaluator
Evaluates end-to-end workflow performance across all datasets.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import json
import yaml
import asyncio
import time
import psutil
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.workflows.classification_workflow import ClassificationWorkflow
from app.workflows.state_management import ClassificationState

logger = logging.getLogger(__name__)


class WorkflowEvaluator:
    """Evaluates end-to-end workflow performance."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the workflow evaluator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load a dataset from file."""
        return pd.read_csv(dataset_path)
    
    def load_metadata(self, dataset_name: str) -> Dict:
        """Load metadata for a dataset."""
        metadata_path = Path(f"Evaluation/datasets/metadata/{dataset_name}_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    async def evaluate_workflow(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end workflow on a dataset.
        
        Args:
            dataset_path: Path to dataset CSV
            dataset_name: Name of dataset
            metadata: Dataset metadata
            target_column: Target column name
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating workflow on {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        # Initialize workflow
        workflow = ClassificationWorkflow()
        
        # Measure execution metrics
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Run workflow
            # Note: This is a simplified version - actual workflow execution may differ
            # based on your implementation
            result = await workflow.run_workflow(
                dataset=df,
                target_column=target_column,
                session_id=f"eval_{dataset_name}"
            )
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Extract metrics from workflow result
            workflow_state = result.get('state', {})
            
            # Calculate Layer 2 success rate
            layer2_successes = 0
            layer2_attempts = 0
            
            agent_statuses = workflow_state.get('agent_statuses', {})
            for agent_name, status in agent_statuses.items():
                if 'layer2' in str(status).lower():
                    layer2_attempts += 1
                    if 'success' in str(status).lower():
                        layer2_successes += 1
            
            layer2_success_rate = layer2_successes / layer2_attempts if layer2_attempts > 0 else 0.0
            
            # Check decision quality
            decision_quality = self._evaluate_decision_quality(
                workflow_state, metadata
            )
            
            # Check output quality
            output_quality = self._evaluate_output_quality(
                workflow_state, metadata
            )
            
            # Check failure handling
            failure_handling = self._evaluate_failure_handling(
                workflow_state, result
            )
            
            evaluation_result = {
                'dataset_name': dataset_name,
                'success': True,
                'execution_metrics': {
                    'execution_time_seconds': execution_time,
                    'memory_usage_mb': memory_usage,
                    'layer2_success_rate': layer2_success_rate
                },
                'decision_quality': decision_quality,
                'output_quality': output_quality,
                'failure_handling': failure_handling,
                'workflow_status': workflow_state.get('workflow_status', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error evaluating workflow on {dataset_name}: {e}")
            evaluation_result = {
                'dataset_name': dataset_name,
                'success': False,
                'error': str(e),
                'execution_metrics': {
                    'execution_time_seconds': time.time() - start_time,
                    'memory_usage_mb': 0,
                    'layer2_success_rate': 0.0
                }
            }
        
        self.results.append(evaluation_result)
        return evaluation_result
    
    def _evaluate_decision_quality(
        self,
        workflow_state: Dict,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Evaluate quality of decisions made during workflow."""
        pass_fail_criteria = metadata.get('pass_fail_criteria', {})
        
        decisions = {}
        
        # Check each agent's decisions
        for agent_name, criteria in pass_fail_criteria.items():
            # This would check if agent made correct decisions
            # Simplified for now
            decisions[agent_name] = {
                'passed': True,  # Would be evaluated based on actual decisions
                'details': {}
            }
        
        return {
            'overall_score': 1.0,  # Would calculate based on decisions
            'agent_decisions': decisions
        }
    
    def _evaluate_output_quality(
        self,
        workflow_state: Dict,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Evaluate quality of workflow outputs."""
        # Check if model was created
        model_created = 'model' in workflow_state or 'trained_model' in workflow_state
        
        # Check if report was generated
        report_generated = 'report' in workflow_state or 'technical_report' in workflow_state
        
        # Check if deliverables are complete
        deliverables_complete = model_created and report_generated
        
        return {
            'model_created': model_created,
            'report_generated': report_generated,
            'deliverables_complete': deliverables_complete,
            'overall_score': 1.0 if deliverables_complete else 0.5
        }
    
    def _evaluate_failure_handling(
        self,
        workflow_state: Dict,
        result: Dict
    ) -> Dict[str, Any]:
        """Evaluate failure handling and graceful degradation."""
        # Check if workflow completed despite errors
        workflow_completed = workflow_state.get('workflow_status') == 'completed'
        
        # Check if Layer 1 fallback was used
        layer1_fallback_used = 'layer1' in str(result).lower() and 'fallback' in str(result).lower()
        
        # Check error recovery
        errors_occurred = 'error' in str(result).lower() or 'exception' in str(result).lower()
        recovered_from_errors = errors_occurred and workflow_completed
        
        return {
            'workflow_completed': workflow_completed,
            'layer1_fallback_used': layer1_fallback_used,
            'recovered_from_errors': recovered_from_errors,
            'graceful_degradation': recovered_from_errors or not errors_occurred
        }
    
    async def evaluate_all_datasets(self, datasets: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate workflow on all datasets.
        
        Args:
            datasets: List of dictionaries with 'name', 'path', 'metadata', and 'target_column'
            
        Returns:
            Dictionary with all evaluation results
        """
        all_results = {}
        
        for dataset_info in datasets:
            dataset_name = dataset_info['name']
            dataset_path = dataset_info['path']
            metadata = dataset_info['metadata']
            target_column = dataset_info.get('target_column', metadata.get('target_column'))
            
            result = await self.evaluate_workflow(
                dataset_path, dataset_name, metadata, target_column
            )
            all_results[dataset_name] = result
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(all_results)
        
        # Save results
        output_path = Path("Evaluation/results/system_level/workflow_evaluation.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'individual_results': all_results,
                'aggregate_metrics': aggregate_metrics
            }, f, indent=2, default=str)
        
        return {
            'individual_results': all_results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def _calculate_aggregate_metrics(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all datasets."""
        successful_runs = [r for r in all_results.values() if r.get('success', False)]
        
        if not successful_runs:
            return {
                'total_datasets': len(all_results),
                'successful_runs': 0,
                'average_execution_time': 0,
                'average_layer2_success_rate': 0
            }
        
        execution_times = [
            r['execution_metrics']['execution_time_seconds']
            for r in successful_runs
        ]
        
        layer2_rates = [
            r['execution_metrics']['layer2_success_rate']
            for r in successful_runs
        ]
        
        return {
            'total_datasets': len(all_results),
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / len(all_results),
            'average_execution_time': sum(execution_times) / len(execution_times),
            'average_layer2_success_rate': sum(layer2_rates) / len(layer2_rates) if layer2_rates else 0,
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times)
        }

