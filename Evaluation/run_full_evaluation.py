"""
Main Evaluation Orchestration Script
Runs the complete evaluation framework.
"""

import sys
import asyncio
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from datasets.kaggle_downloader import KaggleDownloader
from datasets.synthetic_generator import SyntheticDatasetGenerator
from datasets.metadata_generator import MetadataGenerator
from test_cases.agent_tests import AgentTestSuite
from orchestration.workflow_evaluator import WorkflowEvaluator
from visualization.plot_generator import PlotGenerator
from visualization.flowchart_generator import FlowchartGenerator
from reports.report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/evaluation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates the complete evaluation process."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the orchestrator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def _load_config(self) -> Dict:
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run_full_evaluation(self):
        """Run the complete evaluation process."""
        logger.info("="*80)
        logger.info("Starting Comprehensive Evaluation Framework")
        logger.info("="*80)
        
        try:
            # Step 1: Download/generate datasets
            logger.info("\n[Step 1/7] Downloading and generating datasets...")
            datasets = await self._prepare_datasets()
            
            # Step 2: Generate metadata
            logger.info("\n[Step 2/7] Generating dataset metadata...")
            metadata = await self._generate_metadata(datasets)
            
            # Step 3: Run agent-level tests
            logger.info("\n[Step 3/7] Running agent-level tests...")
            agent_results = await self._run_agent_tests(datasets, metadata)
            
            # Step 4: Run system-level tests
            logger.info("\n[Step 4/7] Running system-level workflow tests...")
            workflow_results = await self._run_workflow_tests(datasets, metadata)
            
            # Step 5: Generate visualizations
            logger.info("\n[Step 5/7] Generating visualizations...")
            await self._generate_visualizations(agent_results, workflow_results)
            
            # Step 6: Generate reports
            logger.info("\n[Step 6/7] Generating reports...")
            await self._generate_reports(agent_results, workflow_results, metadata)
            
            # Step 7: Summary
            logger.info("\n[Step 7/7] Evaluation complete!")
            self._print_summary(agent_results, workflow_results)
            
            logger.info("\n" + "="*80)
            logger.info("Evaluation completed successfully!")
            logger.info("="*80)
            
            return {
                'success': True,
                'agent_results': agent_results,
                'workflow_results': workflow_results,
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': self.timestamp
            }
    
    async def _prepare_datasets(self) -> List[Dict[str, str]]:
        """Download and generate all datasets."""
        datasets = []
        
        # Download real-world datasets
        try:
            downloader = KaggleDownloader(self.config_path)
            kaggle_results = downloader.download_all_datasets()
            
            for name, path in kaggle_results.items():
                if path:
                    datasets.append({
                        'name': name,
                        'path': path,
                        'type': 'real_world'
                    })
        except Exception as e:
            logger.warning(f"Error downloading Kaggle datasets: {e}")
        
        # Generate synthetic datasets
        try:
            generator = SyntheticDatasetGenerator(self.config_path)
            synthetic_results = generator.generate_all_synthetic_datasets()
            
            for name, path in synthetic_results.items():
                datasets.append({
                    'name': name,
                    'path': path,
                    'type': 'synthetic'
                })
        except Exception as e:
            logger.warning(f"Error generating synthetic datasets: {e}")
        
        logger.info(f"Prepared {len(datasets)} datasets")
        return datasets
    
    async def _generate_metadata(self, datasets: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Generate metadata for all datasets."""
        generator = MetadataGenerator(self.config_path)
        all_metadata = generator.generate_all_metadata()
        return all_metadata
    
    async def _run_agent_tests(
        self,
        datasets: List[Dict[str, str]],
        metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run agent-level tests."""
        test_suite = AgentTestSuite(self.config_path)
        
        # Prepare dataset info for tests
        dataset_info_list = []
        for dataset in datasets:
            dataset_name = dataset['name']
            dataset_info_list.append({
                'name': dataset_name,
                'path': dataset['path'],
                'metadata': metadata.get(dataset_name, {})
            })
        
        # Run all tests
        results = await test_suite.run_all_agent_tests(dataset_info_list)
        
        return results
    
    async def _run_workflow_tests(
        self,
        datasets: List[Dict[str, str]],
        metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run system-level workflow tests."""
        evaluator = WorkflowEvaluator(self.config_path)
        
        # Prepare dataset info
        dataset_info_list = []
        for dataset in datasets:
            dataset_name = dataset['name']
            meta = metadata.get(dataset_name, {})
            dataset_info_list.append({
                'name': dataset_name,
                'path': dataset['path'],
                'metadata': meta,
                'target_column': meta.get('target_column')
            })
        
        # Run evaluation
        results = await evaluator.evaluate_all_datasets(dataset_info_list)
        
        return results
    
    async def _generate_visualizations(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        workflow_results: Optional[Dict[str, Any]]
    ):
        """Generate all visualizations."""
        # Generate plots
        plot_gen = PlotGenerator()
        plot_gen.generate_all_plots(agent_results, workflow_results)
        
        # Generate flowcharts
        flowchart_gen = FlowchartGenerator()
        flowchart_gen.generate_all_diagrams()
    
    async def _generate_reports(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        workflow_results: Optional[Dict[str, Any]],
        metadata: Dict[str, Dict[str, Any]]
    ):
        """Generate comprehensive reports."""
        report_gen = ReportGenerator()
        report_gen.generate_full_report(
            agent_results,
            workflow_results,
            metadata,
            self.config
        )
    
    def _print_summary(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        workflow_results: Optional[Dict[str, Any]]
    ):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Agent summary
        print("\nAgent-Level Results:")
        for agent_name, results in agent_results.items():
            total = len(results)
            passed = sum(1 for r in results.values() if isinstance(r, dict) and r.get('passed', False))
            pass_rate = passed / total if total > 0 else 0.0
            status = "✓" if pass_rate >= 0.80 else "✗"
            print(f"  {status} {agent_name}: {passed}/{total} ({pass_rate:.1%})")
        
        # System summary
        if workflow_results:
            aggregate = workflow_results.get('aggregate_metrics', {})
            print(f"\nSystem-Level Results:")
            print(f"  Total Datasets: {aggregate.get('total_datasets', 0)}")
            print(f"  Success Rate: {aggregate.get('success_rate', 0):.1%}")
            print(f"  Avg Layer 2 Success: {aggregate.get('average_layer2_success_rate', 0):.1%}")
        
        print("\n" + "="*80)
        print("Results saved to:")
        print("  - Evaluation/results/agent_level/all_agent_tests.json")
        print("  - Evaluation/results/system_level/workflow_evaluation.json")
        print("  - Evaluation/reports/evaluation_report.md")
        print("  - Evaluation/results/visualization/")
        print("="*80)


async def main():
    """Main entry point."""
    orchestrator = EvaluationOrchestrator()
    result = await orchestrator.run_full_evaluation()
    
    # Exit with appropriate code
    exit_code = 0 if result.get('success') else 1
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

