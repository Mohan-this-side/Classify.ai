"""
Single Dataset Evaluation Script
Tests ALL 8 agents on ONE dataset for quick feedback and iteration.
"""

import sys
import asyncio
import logging
import json
import yaml
import pandas as pd
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
from reports.table_generator import TableGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/single_dataset_eval.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def sample_dataset(df: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame:
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


async def run_single_dataset_evaluation(dataset_name: str = None):
    """Run evaluation on a single dataset testing all 8 agents."""
    logger.info("="*80)
    logger.info("Single Dataset Evaluation - ALL 8 AGENTS")
    logger.info("="*80)
    
    try:
        # Step 1: Select dataset
        logger.info("\n[Step 1/6] Selecting dataset...")
        
        real_world_dir = Path("Evaluation/datasets/real_world")
        synthetic_dir = Path("Evaluation/datasets/synthetic")
        
        # Get available datasets
        available_datasets = []
        for csv_file in list(real_world_dir.glob("*.csv")):
            available_datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'real_world'
            })
        for csv_file in synthetic_dir.glob("*.csv"):
            available_datasets.append({
                'name': csv_file.stem,
                'path': str(csv_file),
                'type': 'synthetic'
            })
        
        if not available_datasets:
            logger.error("No datasets found!")
            return {'success': False, 'error': 'No datasets found'}
        
        # Select dataset
        if dataset_name:
            selected = next((d for d in available_datasets if d['name'] == dataset_name), None)
            if not selected:
                logger.warning(f"Dataset '{dataset_name}' not found. Available: {[d['name'] for d in available_datasets]}")
                selected = available_datasets[0]
        else:
            # Use first available dataset
            selected = available_datasets[0]
        
        logger.info(f"Selected dataset: {selected['name']}")
        logger.info(f"  Path: {selected['path']}")
        logger.info(f"  Type: {selected['type']}")
        
        # Step 2: Load metadata
        logger.info("\n[Step 2/6] Loading metadata...")
        metadata_path = Path(f"Evaluation/datasets/metadata/{selected['name']}_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Quick metadata
            try:
                df_sample = pd.read_csv(selected['path'], nrows=100)
                target_col = None
                for col in df_sample.columns:
                    if col.lower() in ['target', 'class', 'label', 'y', 'income', 'survived', 'quality', 'class']:
                        target_col = col
                        break
                metadata = {
                    'target_column': target_col or 'target',
                    'analysis': {
                        'data_types': {
                            'all': {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
                        }
                    }
                }
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                metadata = {'target_column': 'target'}
        
        logger.info(f"Target column: {metadata.get('target_column', 'target')}")
        
        # Step 3: Initialize test suite
        logger.info("\n[Step 3/6] Initializing test suite...")
        test_suite = AgentTestSuite()
        
        # Override load_dataset to use sampling
        original_load = test_suite.base_test.load_dataset
        def load_dataset_sampled(path: str) -> pd.DataFrame:
            df = original_load(path)
            return sample_dataset(df, max_samples=2000)
        test_suite.base_test.load_dataset = load_dataset_sampled
        
        dataset_info = {
            'name': selected['name'],
            'path': selected['path'],
            'metadata': metadata
        }
        
        # Step 4: Run ALL 8 agent tests
        logger.info("\n[Step 4/6] Running ALL 8 agent tests...")
        logger.info("="*60)
        
        agent_tests = [
            ('data_discovery', 'Data Discovery', test_suite.test_data_discovery_agent, 90),
            ('eda_analysis', 'EDA Analysis', test_suite.test_eda_agent, 120),
            ('data_cleaning', 'Data Cleaning', test_suite.test_data_cleaning_agent, 120),
            ('feature_engineering', 'Feature Engineering', test_suite.test_feature_engineering_agent, 120),
            ('ml_builder', 'ML Builder', test_suite.test_ml_builder_agent, 180),
            ('model_evaluation', 'Model Evaluation', test_suite.test_model_evaluation_agent, 120),
            ('technical_reporter', 'Technical Reporter', test_suite.test_technical_reporter_agent, 120),
            ('project_manager', 'Project Manager', test_suite.test_project_manager_agent, 120),
        ]
        
        results = {}
        start_time = datetime.now()
        
        for agent_key, agent_name, test_func, timeout in agent_tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {agent_name} ({agent_key})")
            logger.info(f"{'='*60}")
            
            try:
                result = await asyncio.wait_for(
                    test_func(
                        dataset_info['path'],
                        dataset_info['name'],
                        dataset_info['metadata']
                    ),
                    timeout=timeout
                )
                results[agent_key] = result
                
                # Print quick summary
                passed = result.get('passed', False)
                status = "✓ PASS" if passed else "✗ FAIL"
                logger.info(f"\n{status} - {agent_name}")
                
                # Show key metrics
                metrics = result.get('metrics', {})
                if metrics:
                    key_metrics = {k: v for k, v in metrics.items() if k not in ['error']}
                    if key_metrics:
                        logger.info(f"  Metrics: {key_metrics}")
                if 'error' in metrics:
                    logger.warning(f"  Error: {metrics['error']}")
                
            except asyncio.TimeoutError:
                logger.warning(f"  ✗ TIMEOUT after {timeout}s")
                results[agent_key] = {
                    'test_name': agent_key,
                    'dataset_name': dataset_info['name'],
                    'passed': False,
                    'metrics': {'error': f'Timeout after {timeout}s'}
                }
            except Exception as e:
                logger.error(f"  ✗ FAILED: {e}")
                results[agent_key] = {
                    'test_name': agent_key,
                    'dataset_name': dataset_info['name'],
                    'passed': False,
                    'metrics': {'error': str(e)}
                }
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*60}")
        logger.info(f"All tests completed in {elapsed_time:.1f} seconds")
        logger.info(f"{'='*60}")
        
        # Step 5: Generate summary
        logger.info("\n[Step 5/6] Generating summary...")
        
        agent_results = {dataset_info['name']: results}
        
        # Save results
        all_results = [result for result in results.values()]
        output_path = Path("Evaluation/results/agent_level/single_dataset_tests.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
        
        # Step 6: Generate tables and summary
        logger.info("\n[Step 6/6] Generating tables and summary...")
        
        # Generate tables
        table_gen = TableGenerator()
        datasets_list = [selected]
        tables_dir = Path("Evaluation/results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Table 1: Dataset info
        table1 = table_gen.generate_dataset_table(datasets_list, {selected['name']: metadata}, format="markdown")
        with open(tables_dir / "single_dataset_table1.md", 'w') as f:
            f.write(table1)
        
        # Table 2: Agent performance
        table2 = table_gen.generate_agent_performance_table(agent_results, format="markdown")
        with open(tables_dir / "single_dataset_table2.md", 'w') as f:
            f.write(table2)
        
        # Table 3: Model performance (if available)
        table3 = table_gen.generate_model_performance_table(agent_results, format="markdown")
        with open(tables_dir / "single_dataset_table3.md", 'w') as f:
            f.write(table3)
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY - SINGLE DATASET")
        print("="*80)
        print(f"Dataset: {selected['name']}")
        print(f"Type: {selected['type']}")
        print(f"Execution time: {elapsed_time:.1f} seconds")
        print(f"\nAgent Results:")
        print("-" * 80)
        
        agent_names = {
            'data_discovery': 'Data Discovery',
            'eda_analysis': 'EDA Analysis',
            'data_cleaning': 'Data Cleaning',
            'feature_engineering': 'Feature Engineering',
            'ml_builder': 'ML Builder',
            'model_evaluation': 'Model Evaluation',
            'technical_reporter': 'Technical Reporter',
            'project_manager': 'Project Manager'
        }
        
        total_passed = 0
        total_tests = len(results)
        
        for agent_key, agent_name in agent_names.items():
            if agent_key in results:
                result = results[agent_key]
                passed = result.get('passed', False)
                status = "✓ PASS" if passed else "✗ FAIL"
                
                metrics = result.get('metrics', {})
                layer1 = "Layer 1 ✓" if metrics.get('layer1_success') or metrics.get('layer1_completed') else ""
                
                print(f"{status:8s} {agent_name:25s} {layer1}")
                
                # Show key metrics
                if metrics and passed:
                    key_metrics = []
                    for k, v in metrics.items():
                        if k not in ['error', 'layer1_success', 'layer1_completed'] and isinstance(v, (int, float)):
                            if isinstance(v, float):
                                key_metrics.append(f"{k}={v:.3f}")
                            else:
                                key_metrics.append(f"{k}={v}")
                    if key_metrics:
                        print(f"         {'':25s} {', '.join(key_metrics[:3])}")
                
                if passed:
                    total_passed += 1
        
        print("-" * 80)
        print(f"Total: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
        print("\n" + "="*80)
        print("Results saved to:")
        print(f"  - {output_path}")
        print(f"  - {tables_dir}/single_dataset_table*.md")
        print("="*80)
        
        return {
            'success': True,
            'dataset': selected['name'],
            'agent_results': results,
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    import sys
    
    # Allow dataset name as command line argument
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    result = asyncio.run(run_single_dataset_evaluation(dataset_name))
    
    if result.get('success'):
        print(f"\n✓ Evaluation completed successfully!")
        print(f"  Dataset: {result.get('dataset')}")
        print(f"  Passed: {result.get('passed_tests')}/{result.get('total_tests')}")
    else:
        print(f"\n✗ Evaluation failed: {result.get('error')}")
        sys.exit(1)

