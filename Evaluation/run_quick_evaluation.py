"""
Quick Evaluation Script
Runs evaluation on a subset of datasets with shorter timeouts for faster results.
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
from visualization.plot_generator import PlotGenerator
from visualization.flowchart_generator import FlowchartGenerator
from reports.report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/evaluation_quick.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def run_quick_evaluation():
    """Run a quick evaluation on subset of datasets."""
    logger.info("="*80)
    logger.info("Starting Quick Evaluation (Subset)")
    logger.info("="*80)
    
    try:
        # Step 1: Use existing datasets (don't download)
        logger.info("\n[Step 1/6] Checking existing datasets...")
        datasets = []
        
        real_world_dir = Path("Evaluation/datasets/real_world")
        synthetic_dir = Path("Evaluation/datasets/synthetic")
        
        # Use first 3 real-world datasets that exist
        for csv_file in list(real_world_dir.glob("*.csv"))[:3]:
            datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'real_world'
            })
        
        # Use all synthetic datasets
        for csv_file in synthetic_dir.glob("*.csv"):
            datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'synthetic'
            })
        
        logger.info(f"Using {len(datasets)} datasets: {[d['name'] for d in datasets]}")
        
        # Step 2: Load metadata
        logger.info("\n[Step 2/6] Loading dataset metadata...")
        metadata = {}
        for dataset in datasets:
            metadata_path = Path(f"Evaluation/datasets/metadata/{dataset['name']}_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata[dataset['name']] = json.load(f)
            else:
                metadata[dataset['name']] = {'target_column': 'target'}
        
        # Step 3: Run agent tests with shorter timeouts
        logger.info("\n[Step 3/6] Running agent-level tests (with timeouts)...")
        test_suite = AgentTestSuite()
        
        # Prepare dataset info
        dataset_info_list = []
        for dataset in datasets:
            dataset_info_list.append({
                'name': dataset['name'],
                'path': dataset['path'],
                'metadata': metadata.get(dataset['name'], {})
            })
        
        # Run tests with timeout wrapper
        agent_results = {}
        for dataset_info in dataset_info_list:
            dataset_name = dataset_info['name']
            logger.info(f"\nTesting agents on {dataset_name}...")
            
            results = {}
            
            # Test each agent with timeout
            agents_to_test = [
                ('data_discovery', test_suite.test_data_discovery_agent),
                ('eda_analysis', test_suite.test_eda_agent),
                ('data_cleaning', test_suite.test_data_cleaning_agent),
            ]
            
            for agent_name, test_func in agents_to_test:
                try:
                    result = await asyncio.wait_for(
                        test_func(
                            dataset_info['path'],
                            dataset_name,
                            dataset_info['metadata']
                        ),
                        timeout=60  # 60 second timeout per agent
                    )
                    results[agent_name] = result
                except asyncio.TimeoutError:
                    logger.warning(f"{agent_name} timed out on {dataset_name}")
                    results[agent_name] = {
                        'test_name': agent_name,
                        'dataset_name': dataset_name,
                        'passed': False,
                        'metrics': {'error': 'Timeout'}
                    }
                except Exception as e:
                    logger.error(f"{agent_name} failed on {dataset_name}: {e}")
                    results[agent_name] = {
                        'test_name': agent_name,
                        'dataset_name': dataset_name,
                        'passed': False,
                        'metrics': {'error': str(e)}
                    }
            
            agent_results[dataset_name] = results
        
        # Step 4: Generate visualizations
        logger.info("\n[Step 4/6] Generating visualizations...")
        plot_gen = PlotGenerator()
        plot_gen.generate_all_plots(agent_results, None)
        
        # Step 5: Generate reports
        logger.info("\n[Step 5/6] Generating reports...")
        report_gen = ReportGenerator()
        
        # Load config
        with open("Evaluation/config/evaluation_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        report_gen.generate_full_report(
            agent_results,
            None,
            metadata,
            config
        )
        
        # Step 6: Summary
        logger.info("\n[Step 6/6] Quick evaluation complete!")
        
        # Print summary
        total_tests = sum(len(results) for results in agent_results.values())
        passed_tests = sum(
            sum(1 for r in results.values() if isinstance(r, dict) and r.get('passed', False))
            for results in agent_results.values()
        )
        
        print("\n" + "="*80)
        print("QUICK EVALUATION SUMMARY")
        print("="*80)
        print(f"Datasets tested: {len(datasets)}")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Pass rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        print("\nResults saved to:")
        print("  - Evaluation/results/agent_level/all_agent_tests.json")
        print("  - Evaluation/reports/evaluation_report.md")
        print("  - Evaluation/results/visualization/")
        print("="*80)
        
        return {
            'success': True,
            'agent_results': agent_results,
            'total_tests': total_tests,
            'passed_tests': passed_tests
        }
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    result = asyncio.run(run_quick_evaluation())
    sys.exit(0 if result.get('success') else 1)

