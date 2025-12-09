"""
Differential Evaluation Comparison Tool
Compares current evaluation results with previous baseline.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EvaluationComparator:
    """Compares evaluation results over time."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the comparator."""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_evaluation_results(self, results_path: str) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def compare_agent_results(
        self,
        current: Dict[str, Dict[str, Any]],
        baseline: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare agent-level results."""
        comparison = {
            'improvements': [],
            'regressions': [],
            'unchanged': [],
            'new_tests': [],
            'removed_tests': []
        }
        
        # Compare each agent
        all_agents = set(current.keys()) | set(baseline.keys())
        
        for agent_name in all_agents:
            current_results = current.get(agent_name, {})
            baseline_results = baseline.get(agent_name, {})
            
            # Calculate pass rates
            current_passed = sum(1 for r in current_results.values() 
                               if isinstance(r, dict) and r.get('passed', False))
            current_total = len(current_results)
            current_rate = current_passed / current_total if current_total > 0 else 0.0
            
            baseline_passed = sum(1 for r in baseline_results.values() 
                                if isinstance(r, dict) and r.get('passed', False))
            baseline_total = len(baseline_results)
            baseline_rate = baseline_passed / baseline_total if baseline_total > 0 else 0.0
            
            # Determine change
            change = current_rate - baseline_rate
            
            if change > 0.05:  # 5% improvement threshold
                comparison['improvements'].append({
                    'agent': agent_name,
                    'baseline_rate': baseline_rate,
                    'current_rate': current_rate,
                    'improvement': change
                })
            elif change < -0.05:  # 5% regression threshold
                comparison['regressions'].append({
                    'agent': agent_name,
                    'baseline_rate': baseline_rate,
                    'current_rate': current_rate,
                    'regression': abs(change)
                })
            else:
                comparison['unchanged'].append({
                    'agent': agent_name,
                    'rate': current_rate
                })
        
        return comparison
    
    def compare_workflow_results(
        self,
        current: Dict[str, Any],
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare system-level workflow results."""
        current_agg = current.get('aggregate_metrics', {})
        baseline_agg = baseline.get('aggregate_metrics', {})
        
        comparison = {}
        
        # Compare key metrics
        metrics_to_compare = [
            'success_rate',
            'average_layer2_success_rate',
            'average_execution_time'
        ]
        
        for metric in metrics_to_compare:
            current_val = current_agg.get(metric, 0)
            baseline_val = baseline_agg.get(metric, 0)
            
            if metric == 'average_execution_time':
                # Lower is better
                change = baseline_val - current_val
                improvement = change > 0
            else:
                # Higher is better
                change = current_val - baseline_val
                improvement = change > 0
            
            comparison[metric] = {
                'baseline': baseline_val,
                'current': current_val,
                'change': change,
                'improvement': improvement,
                'change_pct': (change / baseline_val * 100) if baseline_val > 0 else 0
            }
        
        return comparison
    
    def generate_comparison_report(
        self,
        current_results_path: str,
        baseline_results_path: str,
        output_path: str = "Evaluation/reports/comparison_report.md"
    ) -> str:
        """Generate comparison report."""
        # Load results
        current_agent = self.load_evaluation_results(
            "Evaluation/results/agent_level/all_agent_tests.json"
        )
        baseline_agent = self.load_evaluation_results(baseline_results_path.replace(
            'workflow_evaluation.json', 'all_agent_tests.json'
        )) if Path(baseline_results_path.replace(
            'workflow_evaluation.json', 'all_agent_tests.json'
        )).exists() else {}
        
        current_workflow = self.load_evaluation_results(
            "Evaluation/results/system_level/workflow_evaluation.json"
        )
        baseline_workflow = self.load_evaluation_results(baseline_results_path) \
            if Path(baseline_results_path).exists() else {}
        
        # Compare
        agent_comparison = self.compare_agent_results(current_agent, baseline_agent)
        workflow_comparison = self.compare_workflow_results(current_workflow, baseline_workflow)
        
        # Generate report
        lines = []
        lines.append("# Evaluation Comparison Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Agent improvements
        if agent_comparison['improvements']:
            lines.append("## Improvements")
            lines.append("")
            for imp in agent_comparison['improvements']:
                lines.append(f"- **{imp['agent']}:** {imp['baseline_rate']:.1%} → {imp['current_rate']:.1%} "
                           f"(+{imp['improvement']:.1%})")
            lines.append("")
        
        # Regressions
        if agent_comparison['regressions']:
            lines.append("## Regressions")
            lines.append("")
            for reg in agent_comparison['regressions']:
                lines.append(f"- **{reg['agent']}:** {reg['baseline_rate']:.1%} → {reg['current_rate']:.1%} "
                           f"(-{reg['regression']:.1%})")
            lines.append("")
        
        # Workflow comparison
        lines.append("## System-Level Changes")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Change |")
        lines.append("|--------|----------|---------|--------|")
        for metric, comp in workflow_comparison.items():
            change_str = f"{comp['change']:+.3f}" if comp['improvement'] else f"{comp['change']:.3f}"
            lines.append(f"| {metric} | {comp['baseline']:.3f} | {comp['current']:.3f} | {change_str} |")
        lines.append("")
        
        # Save report
        report_content = "\n".join(lines)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comparison report saved to {output_path}")
        
        return report_content


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_evaluations.py <baseline_results_path>")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    
    comparator = EvaluationComparator()
    comparator.generate_comparison_report(
        "Evaluation/results/system_level/workflow_evaluation.json",
        baseline_path
    )

