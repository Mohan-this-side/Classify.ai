"""
Final Evaluation Script
Properly evaluates agents by checking state for Layer 1 results.
"""

import sys
import asyncio
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
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
        logging.FileHandler('Evaluation/evaluation_final.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def sample_dataset(df: pd.DataFrame, max_samples: int = 3000) -> pd.DataFrame:
    """Sample dataset if it's too large for faster evaluation."""
    if len(df) > max_samples:
        target_col = None
        for col in df.columns:
            if col.lower() in ['target', 'class', 'label', 'y', 'income', 'survived', 'quality', 'class']:
                target_col = col
                break
        
        if target_col and target_col in df.columns:
            sampled = df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max_samples // max(df[target_col].nunique(), 2)))
            ).reset_index(drop=True)
            if len(sampled) > max_samples:
                sampled = sampled.sample(n=max_samples, random_state=42)
            return sampled
        else:
            return df.sample(n=min(max_samples, len(df)), random_state=42)
    return df


async def run_final_evaluation():
    """Run final evaluation that properly recognizes Layer 1 successes."""
    logger.info("="*80)
    logger.info("Starting Final Evaluation")
    logger.info("="*80)
    
    try:
        # Step 1: Load datasets with sampling
        logger.info("\n[Step 1/6] Loading datasets...")
        datasets = []
        
        real_world_dir = Path("Evaluation/datasets/real_world")
        synthetic_dir = Path("Evaluation/datasets/synthetic")
        
        # Use available datasets (limit to 5 for speed)
        for csv_file in list(real_world_dir.glob("*.csv"))[:3]:
            datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'real_world'
            })
        
        for csv_file in list(synthetic_dir.glob("*.csv"))[:3]:  # Limit synthetic too
            datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'synthetic'
            })
        
        logger.info(f"Using {len(datasets)} datasets: {[d['name'] for d in datasets]}")
        
        # Step 2: Load metadata
        logger.info("\n[Step 2/6] Loading metadata...")
        metadata = {}
        for dataset in datasets:
            metadata_path = Path(f"Evaluation/datasets/metadata/{dataset['name']}_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata[dataset['name']] = json.load(f)
            else:
                # Quick metadata
                df_sample = pd.read_csv(dataset['path'], nrows=100)
                target_col = None
                for col in df_sample.columns:
                    if col.lower() in ['target', 'class', 'label', 'y', 'income', 'survived', 'quality', 'class']:
                        target_col = col
                        break
                metadata[dataset['name']] = {
                    'target_column': target_col or 'target',
                    'analysis': {
                        'data_types': {
                            'all': {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
                        }
                    }
                }
        
        # Step 3: Run tests with sampling
        logger.info("\n[Step 3/6] Running agent tests (this may take a few minutes)...")
        test_suite = AgentTestSuite()
        
        # Override load_dataset to use sampling
        original_load = test_suite.base_test.load_dataset
        def load_dataset_sampled(path: str) -> pd.DataFrame:
            df = original_load(path)
            return sample_dataset(df, max_samples=3000)
        test_suite.base_test.load_dataset = load_dataset_sampled
        
        # Prepare dataset info
        dataset_info_list = []
        for dataset in datasets:
            dataset_info_list.append({
                'name': dataset['name'],
                'path': dataset['path'],
                'metadata': metadata.get(dataset['name'], {})
            })
        
        # Run tests - focus on first 3 agents for now
        agent_results = {}
        
        for dataset_info in dataset_info_list:
            dataset_name = dataset_info['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {dataset_name}")
            logger.info(f"{'='*60}")
            
            results = {}
            
            # Test key agents
            try:
                logger.info(f"  Testing data_discovery...")
                results['data_discovery'] = await asyncio.wait_for(
                    test_suite.test_data_discovery_agent(
                        dataset_info['path'],
                        dataset_name,
                        dataset_info['metadata']
                    ),
                    timeout=90
                )
            except asyncio.TimeoutError:
                logger.warning(f"  data_discovery timed out")
                results['data_discovery'] = {'test_name': 'data_discovery', 'dataset_name': dataset_name, 'passed': False, 'metrics': {'error': 'Timeout'}}
            except Exception as e:
                logger.error(f"  data_discovery failed: {e}")
                results['data_discovery'] = {'test_name': 'data_discovery', 'dataset_name': dataset_name, 'passed': False, 'metrics': {'error': str(e)}}
            
            try:
                logger.info(f"  Testing eda_analysis...")
                results['eda_analysis'] = await asyncio.wait_for(
                    test_suite.test_eda_agent(
                        dataset_info['path'],
                        dataset_name,
                        dataset_info['metadata']
                    ),
                    timeout=120
                )
            except asyncio.TimeoutError:
                logger.warning(f"  eda_analysis timed out")
                results['eda_analysis'] = {'test_name': 'eda_analysis', 'dataset_name': dataset_name, 'passed': False, 'metrics': {'error': 'Timeout'}}
            except Exception as e:
                logger.error(f"  eda_analysis failed: {e}")
                results['eda_analysis'] = {'test_name': 'eda_analysis', 'dataset_name': dataset_name, 'passed': False, 'metrics': {'error': str(e)}}
            
            try:
                logger.info(f"  Testing data_cleaning...")
                results['data_cleaning'] = await asyncio.wait_for(
                    test_suite.test_data_cleaning_agent(
                        dataset_info['path'],
                        dataset_name,
                        dataset_info['metadata']
                    ),
                    timeout=120
                )
            except asyncio.TimeoutError:
                logger.warning(f"  data_cleaning timed out")
                results['data_cleaning'] = {'test_name': 'data_cleaning', 'dataset_name': dataset_name, 'passed': False, 'metrics': {'error': 'Timeout'}}
            except Exception as e:
                logger.error(f"  data_cleaning failed: {e}")
                results['data_cleaning'] = {'test_name': 'data_cleaning', 'dataset_name': dataset_name, 'passed': False, 'metrics': {'error': str(e)}}
            
            agent_results[dataset_name] = results
        
        # Save results
        all_results = []
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                all_results.append(result)
        
        output_path = Path("Evaluation/results/agent_level/all_agent_tests.json")
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
        
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
        
        total_tests = sum(len(results) for results in agent_results.values())
        passed_tests = sum(
            sum(1 for r in results.values() if isinstance(r, dict) and r.get('passed', False))
            for results in agent_results.values()
        )
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Datasets tested: {len(datasets)}")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Pass rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        
        # Per-agent breakdown
        print("\nPer-Agent Results:")
        agent_counts = {}
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                if agent_name not in agent_counts:
                    agent_counts[agent_name] = {'total': 0, 'passed': 0}
                agent_counts[agent_name]['total'] += 1
                if result.get('passed', False):
                    agent_counts[agent_name]['passed'] += 1
        
        for agent_name, counts in agent_counts.items():
            rate = counts['passed'] / counts['total'] * 100 if counts['total'] > 0 else 0
            print(f"  {agent_name}: {counts['passed']}/{counts['total']} ({rate:.1f}%)")
        
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
    result = asyncio.run(run_final_evaluation())
    sys.exit(0 if result.get('success') else 1)

