"""
Comprehensive Evaluation Script
Tests ALL 8 agents on all datasets and generates comprehensive result tables.
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
from visualization.flowchart_generator import FlowchartGenerator
from reports.report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/evaluation_comprehensive.log'),
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


def generate_dataset_table(datasets: List[Dict], metadata: Dict) -> str:
    """Generate Table 1: Evaluation Datasets and Modification Strategy."""
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Evaluation Datasets and Modification Strategy}\n"
    table += "\\label{tab:datasets}\n"
    table += "\\begin{tabular}{|l|l|r|r|r|l|l|l|}\n"
    table += "\\hline\n"
    table += "\\textbf{Dataset} & \\textbf{Source} & \\textbf{Samples} & \\textbf{Features} & \\textbf{Classes} & \\textbf{Condition} & \\textbf{Modifications Applied} & \\textbf{Purpose} \\\\\n"
    table += "\\hline\n"
    
    for dataset in datasets:
        name = dataset['name']
        meta = metadata.get(name, {})
        
        # Load dataset to get actual stats
        try:
            df = pd.read_csv(dataset['path'], nrows=1000)
            samples = len(df) if len(df) < 1000 else f"{len(df):,}"
            features = len(df.columns) - 1  # Exclude target
        except:
            samples = "N/A"
            features = "N/A"
        
        # Determine classes
        target_col = meta.get('target_column', 'target')
        try:
            df = pd.read_csv(dataset['path'], nrows=1000)
            if target_col in df.columns:
                classes = df[target_col].nunique()
            else:
                classes = 2  # Default
        except:
            classes = 2
        
        # Source
        source = "Kaggle" if dataset['type'] == 'real_world' else "Synthetic"
        
        # Condition and modifications
        condition = "Clean"
        modifications = "None"
        purpose = ""
        
        if 'leakage' in name.lower():
            condition = "Modified"
            modifications = "Perfect target leakage"
            purpose = "Anti-cheating test"
        elif 'imbalance' in name.lower():
            condition = "Modified"
            modifications = "99:1 class imbalance"
            purpose = "Imbalance detection"
        elif 'multicollinearity' in name.lower():
            condition = "Modified"
            modifications = "High correlation features"
            purpose = "Feature engineering test"
        elif 'dimensionality' in name.lower():
            condition = "Modified"
            modifications = "High dimensionality"
            purpose = "Scalability test"
        else:
            purpose = "Baseline evaluation"
        
        table += f"{name.replace('_', ' ').title()} & {source} & {samples} & {features} & {classes} & {condition} & {modifications} & {purpose} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    
    return table


def generate_agent_performance_table(agent_results: Dict[str, Dict[str, Any]], datasets: List[Dict]) -> str:
    """Generate Table 2: Agent Performance Summary."""
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Agent Performance Summary Across All Datasets}\n"
    table += "\\label{tab:agent_performance}\n"
    table += "\\begin{tabular}{|l|r|r|r|r|}\n"
    table += "\\hline\n"
    table += "\\textbf{Agent} & \\textbf{Tests Run} & \\textbf{Passed} & \\textbf{Pass Rate (\\%)} & \\textbf{Layer 1 Success Rate} \\\\\n"
    table += "\\hline\n"
    
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
    
    for agent_key, agent_name in agent_names.items():
        total = 0
        passed = 0
        layer1_success = 0
        
        for dataset_name, results in agent_results.items():
            if agent_key in results:
                total += 1
                result = results[agent_key]
                if isinstance(result, dict):
                    if result.get('passed', False):
                        passed += 1
                    if result.get('metrics', {}).get('layer1_success') or result.get('metrics', {}).get('layer1_completed'):
                        layer1_success += 1
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        layer1_rate = (layer1_success / total * 100) if total > 0 else 0
        
        table += f"{agent_name} & {total} & {passed} & {pass_rate:.1f} & {layer1_rate:.1f} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    
    return table


def generate_model_performance_table(agent_results: Dict[str, Dict[str, Any]], datasets: List[Dict]) -> str:
    """Generate Table 3: Model Building and Evaluation Results."""
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Model Building and Evaluation Results}\n"
    table += "\\label{tab:model_performance}\n"
    table += "\\begin{tabular}{|l|l|r|r|}\n"
    table += "\\hline\n"
    table += "\\textbf{Dataset} & \\textbf{Best Model} & \\textbf{Test Acc.} & \\textbf{CV Mean ($\\pm$Std)} \\\\\n"
    table += "\\hline\n"
    
    for dataset in datasets:
        dataset_name = dataset['name']
        results = agent_results.get(dataset_name, {})
        
        ml_result = results.get('ml_builder', {})
        eval_result = results.get('model_evaluation', {})
        
        best_model = "N/A"
        test_acc = "N/A"
        cv_mean = "N/A"
        
        if isinstance(ml_result, dict):
            metrics = ml_result.get('metrics', {})
            best_model = metrics.get('best_model', 'N/A')
            if best_model == 'N/A':
                best_model = metrics.get('model_selected', 'N/A')
        
        if isinstance(eval_result, dict):
            metrics = eval_result.get('metrics', {})
            test_acc = metrics.get('test_accuracy', 'N/A')
            cv_mean = metrics.get('cv_mean', 'N/A')
            cv_std = metrics.get('cv_std', 'N/A')
            if cv_mean != 'N/A' and cv_std != 'N/A':
                cv_mean = f"{cv_mean:.3f} ($\\pm${cv_std:.3f})"
        
        table += f"{dataset_name.replace('_', ' ').title()} & {best_model} & {test_acc} & {cv_mean} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    
    return table


async def run_comprehensive_evaluation():
    """Run comprehensive evaluation testing all 8 agents."""
    logger.info("="*80)
    logger.info("Starting Comprehensive Evaluation - ALL 8 AGENTS")
    logger.info("="*80)
    
    try:
        # Step 1: Load datasets
        logger.info("\n[Step 1/7] Loading datasets...")
        datasets = []
        
        real_world_dir = Path("Evaluation/datasets/real_world")
        synthetic_dir = Path("Evaluation/datasets/synthetic")
        
        # Use all available datasets
        for csv_file in list(real_world_dir.glob("*.csv"))[:6]:  # Up to 6 real-world
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
        
        logger.info(f"Using {len(datasets)} datasets: {[d['name'] for d in datasets]}")
        
        # Step 2: Load metadata
        logger.info("\n[Step 2/7] Loading metadata...")
        metadata = {}
        for dataset in datasets:
            metadata_path = Path(f"Evaluation/datasets/metadata/{dataset['name']}_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata[dataset['name']] = json.load(f)
            else:
                # Quick metadata
                try:
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
                except Exception as e:
                    logger.warning(f"Could not load metadata for {dataset['name']}: {e}")
                    metadata[dataset['name']] = {'target_column': 'target'}
        
        # Step 3: Run ALL agent tests
        logger.info("\n[Step 3/7] Running ALL agent tests (this will take time)...")
        test_suite = AgentTestSuite()
        
        # Override load_dataset to use sampling
        original_load = test_suite.base_test.load_dataset
        def load_dataset_sampled(path: str) -> pd.DataFrame:
            df = original_load(path)
            return sample_dataset(df, max_samples=2000)
        test_suite.base_test.load_dataset = load_dataset_sampled
        
        # Prepare dataset info
        dataset_info_list = []
        for dataset in datasets:
            dataset_info_list.append({
                'name': dataset['name'],
                'path': dataset['path'],
                'metadata': metadata.get(dataset['name'], {})
            })
        
        # Test all 8 agents on all datasets
        agent_results = {}
        agent_tests = [
            ('data_discovery', test_suite.test_data_discovery_agent, 90),
            ('eda_analysis', test_suite.test_eda_agent, 120),
            ('data_cleaning', test_suite.test_data_cleaning_agent, 120),
            ('feature_engineering', test_suite.test_feature_engineering_agent, 120),
            ('ml_builder', test_suite.test_ml_builder_agent, 180),
            ('model_evaluation', test_suite.test_model_evaluation_agent, 120),
            ('technical_reporter', test_suite.test_technical_reporter_agent, 120),
            ('project_manager', test_suite.test_project_manager_agent, 120),
        ]
        
        for dataset_info in dataset_info_list:
            dataset_name = dataset_info['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing ALL agents on {dataset_name}")
            logger.info(f"{'='*60}")
            
            results = {}
            
            for agent_name, test_func, timeout in agent_tests:
                try:
                    logger.info(f"  Testing {agent_name}...")
                    results[agent_name] = await asyncio.wait_for(
                        test_func(
                            dataset_info['path'],
                            dataset_name,
                            dataset_info['metadata']
                        ),
                        timeout=timeout
                    )
                    logger.info(f"    ✓ {agent_name} completed")
                except asyncio.TimeoutError:
                    logger.warning(f"    ✗ {agent_name} timed out after {timeout}s")
                    results[agent_name] = {
                        'test_name': agent_name,
                        'dataset_name': dataset_name,
                        'passed': False,
                        'metrics': {'error': f'Timeout after {timeout}s'}
                    }
                except Exception as e:
                    logger.error(f"    ✗ {agent_name} failed: {e}")
                    results[agent_name] = {
                        'test_name': agent_name,
                        'dataset_name': dataset_name,
                        'passed': False,
                        'metrics': {'error': str(e)}
                    }
            
            agent_results[dataset_name] = results
        
        # Step 4: Save results
        logger.info("\n[Step 4/7] Saving results...")
        all_results = []
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                all_results.append(result)
        
        output_path = Path("Evaluation/results/agent_level/all_agent_tests.json")
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
        
        # Step 5: Generate tables
        logger.info("\n[Step 5/7] Generating result tables...")
        tables_dir = Path("Evaluation/results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Table 1: Datasets
        table1 = generate_dataset_table(datasets, metadata)
        with open(tables_dir / "table1_datasets.tex", 'w') as f:
            f.write(table1)
        
        # Table 2: Agent Performance
        table2 = generate_agent_performance_table(agent_results, datasets)
        with open(tables_dir / "table2_agent_performance.tex", 'w') as f:
            f.write(table2)
        
        # Table 3: Model Performance
        table3 = generate_model_performance_table(agent_results, datasets)
        with open(tables_dir / "table3_model_performance.tex", 'w') as f:
            f.write(table3)
        
        logger.info(f"Tables saved to {tables_dir}/")
        
        # Step 6: Generate visualizations
        logger.info("\n[Step 6/7] Generating visualizations...")
        plot_gen = PlotGenerator()
        plot_gen.generate_all_plots(agent_results, None)
        
        # Step 7: Generate reports
        logger.info("\n[Step 7/7] Generating reports...")
        report_gen = ReportGenerator()
        
        with open("Evaluation/config/evaluation_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        report_gen.generate_full_report(
            agent_results,
            None,
            metadata,
            config
        )
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EVALUATION COMPLETE")
        logger.info("="*80)
        
        total_tests = sum(len(results) for results in agent_results.values())
        passed_tests = sum(
            sum(1 for r in results.values() if isinstance(r, dict) and r.get('passed', False))
            for results in agent_results.values()
        )
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Datasets tested: {len(datasets)}")
        print(f"Agents tested: 8")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Pass rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        
        # Per-agent breakdown
        print("\nPer-Agent Results:")
        agent_counts = {}
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                if agent_name not in agent_counts:
                    agent_counts[agent_name] = {'total': 0, 'passed': 0, 'layer1': 0}
                agent_counts[agent_name]['total'] += 1
                if isinstance(result, dict):
                    if result.get('passed', False):
                        agent_counts[agent_name]['passed'] += 1
                    if result.get('metrics', {}).get('layer1_success') or result.get('metrics', {}).get('layer1_completed'):
                        agent_counts[agent_name]['layer1'] += 1
        
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
        
        for agent_key, agent_name in agent_names.items():
            if agent_key in agent_counts:
                counts = agent_counts[agent_key]
                rate = counts['passed'] / counts['total'] * 100 if counts['total'] > 0 else 0
                layer1_rate = counts['layer1'] / counts['total'] * 100 if counts['total'] > 0 else 0
                print(f"  {agent_name}: {counts['passed']}/{counts['total']} ({rate:.1f}%) | Layer 1: {layer1_rate:.1f}%")
        
        print("\nResults saved to:")
        print("  - Evaluation/results/agent_level/all_agent_tests.json")
        print("  - Evaluation/results/tables/ (LaTeX tables)")
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
    result = asyncio.run(run_comprehensive_evaluation())
    sys.exit(0 if result.get('success') else 1)

