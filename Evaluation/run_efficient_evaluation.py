"""
Efficient Evaluation Script
Runs evaluation with dataset sampling for faster results while maintaining meaningful evaluation.
"""

import sys
import asyncio
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

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
        logging.FileHandler('Evaluation/evaluation_efficient.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def sample_dataset(df: pd.DataFrame, max_samples: int = 5000) -> pd.DataFrame:
    """Sample dataset if it's too large for faster evaluation."""
    if len(df) > max_samples:
        # Stratified sampling if target exists
        target_col = None
        for col in df.columns:
            if col.lower() in ['target', 'class', 'label', 'y']:
                target_col = col
                break
        
        if target_col and target_col in df.columns:
            # Stratified sample
            sampled = df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max_samples // df[target_col].nunique()))
            ).reset_index(drop=True)
            if len(sampled) > max_samples:
                sampled = sampled.sample(n=max_samples, random_state=42)
            return sampled
        else:
            # Simple random sample
            return df.sample(n=min(max_samples, len(df)), random_state=42)
    return df


async def run_efficient_evaluation():
    """Run efficient evaluation with dataset sampling."""
    logger.info("="*80)
    logger.info("Starting Efficient Evaluation (with sampling)")
    logger.info("="*80)
    
    try:
        # Step 1: Load datasets with sampling
        logger.info("\n[Step 1/6] Loading and sampling datasets...")
        datasets = []
        
        real_world_dir = Path("Evaluation/datasets/real_world")
        synthetic_dir = Path("Evaluation/datasets/synthetic")
        
        # Use available datasets
        for csv_file in list(real_world_dir.glob("*.csv"))[:4]:  # Limit to 4 real-world
            datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'real_world'
            })
        
        for csv_file in synthetic_dir.glob("*.csv"):
            datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'synthetic'
            })
        
        logger.info(f"Using {len(datasets)} datasets")
        
        # Step 2: Load metadata
        logger.info("\n[Step 2/6] Loading dataset metadata...")
        metadata = {}
        for dataset in datasets:
            metadata_path = Path(f"Evaluation/datasets/metadata/{dataset['name']}_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata[dataset['name']] = json.load(f)
            else:
                # Create basic metadata
                df_sample = pd.read_csv(dataset['path'], nrows=100)
                target_col = None
                for col in df_sample.columns:
                    if col.lower() in ['target', 'class', 'label', 'y', 'income', 'survived', 'quality']:
                        target_col = col
                        break
                metadata[dataset['name']] = {
                    'target_column': target_col or 'target',
                    'analysis': {'data_types': {'all': {col: str(dtype) for col, dtype in df_sample.dtypes.items()}}}
                }
        
        # Step 3: Run agent tests
        logger.info("\n[Step 3/6] Running agent-level tests...")
        test_suite = AgentTestSuite()
        
        # Modify test suite to use sampled datasets
        original_load = test_suite.base_test.load_dataset
        
        def load_dataset_sampled(path: str) -> pd.DataFrame:
            df = original_load(path)
            return sample_dataset(df, max_samples=5000)
        
        test_suite.base_test.load_dataset = load_dataset_sampled
        
        # Prepare dataset info
        dataset_info_list = []
        for dataset in datasets:
            dataset_info_list.append({
                'name': dataset['name'],
                'path': dataset['path'],
                'metadata': metadata.get(dataset['name'], {})
            })
        
        # Run all agent tests
        agent_results = await test_suite.run_all_agent_tests(dataset_info_list)
        
        # Step 4: Generate visualizations
        logger.info("\n[Step 4/6] Generating visualizations...")
        plot_gen = PlotGenerator()
        plot_gen.generate_all_plots(agent_results, None)
        
        # Step 5: Generate reports
        logger.info("\n[Step 5/6] Generating reports...")
        report_gen = ReportGenerator()
        
        with open("Evaluation/config/evaluation_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        report_gen.generate_full_report(
            agent_results,
            None,
            metadata,
            config
        )
        
        # Step 6: Summary
        logger.info("\n[Step 6/6] Evaluation complete!")
        
        # Calculate summary
        total_tests = 0
        passed_tests = 0
        
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                if isinstance(result, dict):
                    total_tests += 1
                    if result.get('passed', False):
                        passed_tests += 1
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
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
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    result = asyncio.run(run_efficient_evaluation())
    sys.exit(0 if result.get('success') else 1)

