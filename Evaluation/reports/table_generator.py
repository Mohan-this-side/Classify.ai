"""
Table Generator for Evaluation Results
Generates LaTeX and Markdown tables for evaluation reports.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class TableGenerator:
    """Generate comprehensive result tables for evaluation reports."""
    
    def __init__(self, results_path: str = "Evaluation/results/agent_level/all_agent_tests.json"):
        self.results_path = Path(results_path)
        self.results = self._load_results()
    
    def _load_results(self) -> List[Dict]:
        """Load evaluation results."""
        if self.results_path.exists():
            with open(self.results_path, 'r') as f:
                return json.load(f)
        return []
    
    def generate_dataset_table(self, datasets: List[Dict], metadata: Dict, format: str = "latex") -> str:
        """Generate Table 1: Evaluation Datasets and Modification Strategy."""
        if format == "latex":
            return self._generate_dataset_table_latex(datasets, metadata)
        else:
            return self._generate_dataset_table_markdown(datasets, metadata)
    
    def _generate_dataset_table_latex(self, datasets: List[Dict], metadata: Dict) -> str:
        """Generate LaTeX table."""
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
                    classes = 2
            except:
                classes = 2
            
            source = "Kaggle" if dataset['type'] == 'real_world' else "Synthetic"
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
    
    def _generate_dataset_table_markdown(self, datasets: List[Dict], metadata: Dict) -> str:
        """Generate Markdown table."""
        table = "## Table 1: Evaluation Datasets and Modification Strategy\n\n"
        table += "| Dataset | Source | Samples | Features | Classes | Condition | Modifications Applied | Purpose |\n"
        table += "|---------|--------|---------|----------|---------|-----------|----------------------|----------|\n"
        
        for dataset in datasets:
            name = dataset['name']
            meta = metadata.get(name, {})
            
            try:
                df = pd.read_csv(dataset['path'], nrows=1000)
                samples = len(df) if len(df) < 1000 else f"{len(df):,}"
                features = len(df.columns) - 1
            except:
                samples = "N/A"
                features = "N/A"
            
            target_col = meta.get('target_column', 'target')
            try:
                df = pd.read_csv(dataset['path'], nrows=1000)
                if target_col in df.columns:
                    classes = df[target_col].nunique()
                else:
                    classes = 2
            except:
                classes = 2
            
            source = "Kaggle" if dataset['type'] == 'real_world' else "Synthetic"
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
            
            table += f"| {name.replace('_', ' ').title()} | {source} | {samples} | {features} | {classes} | {condition} | {modifications} | {purpose} |\n"
        
        return table
    
    def generate_agent_performance_table(self, agent_results: Dict[str, Dict[str, Any]], format: str = "latex") -> str:
        """Generate Table 2: Agent Performance Summary."""
        if format == "latex":
            return self._generate_agent_performance_latex(agent_results)
        else:
            return self._generate_agent_performance_markdown(agent_results)
    
    def _generate_agent_performance_latex(self, agent_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate LaTeX table."""
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
    
    def _generate_agent_performance_markdown(self, agent_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate Markdown table."""
        table = "## Table 2: Agent Performance Summary\n\n"
        table += "| Agent | Tests Run | Passed | Pass Rate (%) | Layer 1 Success Rate |\n"
        table += "|-------|-----------|--------|---------------|----------------------|\n"
        
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
            
            table += f"| {agent_name} | {total} | {passed} | {pass_rate:.1f} | {layer1_rate:.1f} |\n"
        
        return table
    
    def generate_model_performance_table(self, agent_results: Dict[str, Dict[str, Any]], format: str = "latex") -> str:
        """Generate Table 3: Model Building and Evaluation Results."""
        if format == "latex":
            return self._generate_model_performance_latex(agent_results)
        else:
            return self._generate_model_performance_markdown(agent_results)
    
    def _generate_model_performance_latex(self, agent_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate LaTeX table."""
        table = "\\begin{table}[h]\n"
        table += "\\centering\n"
        table += "\\caption{Model Building and Evaluation Results}\n"
        table += "\\label{tab:model_performance}\n"
        table += "\\begin{tabular}{|l|l|r|r|}\n"
        table += "\\hline\n"
        table += "\\textbf{Dataset} & \\textbf{Best Model} & \\textbf{Test Acc.} & \\textbf{CV Mean ($\\pm$Std)} \\\\\n"
        table += "\\hline\n"
        
        for dataset_name, results in agent_results.items():
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
                if test_acc != 'N/A':
                    test_acc = f"{test_acc:.3f}"
                cv_mean_val = metrics.get('cv_mean', 'N/A')
                cv_std_val = metrics.get('cv_std', 'N/A')
                if cv_mean_val != 'N/A' and cv_std_val != 'N/A':
                    cv_mean = f"{cv_mean_val:.3f} ($\\pm${cv_std_val:.3f})"
            
            table += f"{dataset_name.replace('_', ' ').title()} & {best_model} & {test_acc} & {cv_mean} \\\\\n"
        
        table += "\\hline\n"
        table += "\\end{tabular}\n"
        table += "\\end{table}\n"
        return table
    
    def _generate_model_performance_markdown(self, agent_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate Markdown table."""
        table = "## Table 3: Model Building and Evaluation Results\n\n"
        table += "| Dataset | Best Model | Test Acc. | CV Mean (±Std) |\n"
        table += "|---------|------------|-----------|----------------|\n"
        
        for dataset_name, results in agent_results.items():
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
                if test_acc != 'N/A':
                    test_acc = f"{test_acc:.3f}"
                cv_mean_val = metrics.get('cv_mean', 'N/A')
                cv_std_val = metrics.get('cv_std', 'N/A')
                if cv_mean_val != 'N/A' and cv_std_val != 'N/A':
                    cv_mean = f"{cv_mean_val:.3f} (±{cv_std_val:.3f})"
            
            table += f"| {dataset_name.replace('_', ' ').title()} | {best_model} | {test_acc} | {cv_mean} |\n"
        
        return table
    
    def generate_all_tables(self, datasets: List[Dict], metadata: Dict, agent_results: Dict[str, Dict[str, Any]], output_dir: str = "Evaluation/results/tables"):
        """Generate all tables in both LaTeX and Markdown formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Table 1: Datasets
        table1_latex = self.generate_dataset_table(datasets, metadata, format="latex")
        table1_md = self.generate_dataset_table(datasets, metadata, format="markdown")
        
        with open(output_path / "table1_datasets.tex", 'w') as f:
            f.write(table1_latex)
        with open(output_path / "table1_datasets.md", 'w') as f:
            f.write(table1_md)
        
        # Table 2: Agent Performance
        table2_latex = self.generate_agent_performance_table(agent_results, format="latex")
        table2_md = self.generate_agent_performance_table(agent_results, format="markdown")
        
        with open(output_path / "table2_agent_performance.tex", 'w') as f:
            f.write(table2_latex)
        with open(output_path / "table2_agent_performance.md", 'w') as f:
            f.write(table2_md)
        
        # Table 3: Model Performance
        table3_latex = self.generate_model_performance_table(agent_results, format="latex")
        table3_md = self.generate_model_performance_table(agent_results, format="markdown")
        
        with open(output_path / "table3_model_performance.tex", 'w') as f:
            f.write(table3_latex)
        with open(output_path / "table3_model_performance.md", 'w') as f:
            f.write(table3_md)
        
        print(f"All tables generated and saved to {output_path}/")

