"""
Agent Scorecard Generator
Generates scorecards for each agent showing performance metrics.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ScorecardGenerator:
    """Generates agent scorecards."""
    
    def __init__(self):
        """Initialize the scorecard generator."""
        pass
    
    def generate_agent_scorecard(
        self,
        agent_name: str,
        test_results: List[Dict[str, Any]],
        thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a scorecard for a single agent.
        
        Args:
            agent_name: Name of the agent
            test_results: List of test results for this agent
            thresholds: Quality thresholds for this agent
            
        Returns:
            Dictionary with scorecard data
        """
        # Calculate aggregate metrics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.get('passed', False))
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Extract all metrics
        all_metrics = {}
        for result in test_results:
            metrics = result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
        
        # Calculate averages
        average_metrics = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in all_metrics.items()
        }
        
        # Determine overall grade
        grade = self._calculate_grade(pass_rate, average_metrics, thresholds)
        
        scorecard = {
            'agent_name': agent_name,
            'overall_grade': grade,
            'pass_rate': pass_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'average_metrics': average_metrics,
            'thresholds': thresholds,
            'test_results': test_results
        }
        
        return scorecard
    
    def _calculate_grade(
        self,
        pass_rate: float,
        average_metrics: Dict[str, float],
        thresholds: Dict[str, Any]
    ) -> str:
        """Calculate overall grade (A/B/C/D/F)."""
        # Weight pass rate heavily
        if pass_rate >= 0.95:
            base_grade = 'A'
        elif pass_rate >= 0.85:
            base_grade = 'B'
        elif pass_rate >= 0.75:
            base_grade = 'C'
        elif pass_rate >= 0.65:
            base_grade = 'D'
        else:
            base_grade = 'F'
        
        # Check if key metrics meet thresholds
        for metric_name, threshold_value in thresholds.items():
            if metric_name in average_metrics:
                metric_value = average_metrics[metric_name]
                if isinstance(threshold_value, (int, float)):
                    if metric_value < threshold_value:
                        # Downgrade grade
                        if base_grade == 'A':
                            base_grade = 'B'
                        elif base_grade == 'B':
                            base_grade = 'C'
                        elif base_grade == 'C':
                            base_grade = 'D'
                        else:
                            base_grade = 'F'
        
        return base_grade
    
    def generate_all_scorecards(
        self,
        all_agent_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate scorecards for all agents.
        
        Args:
            all_agent_results: Dictionary mapping agent names to test results
            config: Configuration with thresholds
            
        Returns:
            Dictionary mapping agent names to scorecards
        """
        scorecards = {}
        thresholds = config.get('quality_thresholds', {})
        
        for agent_name, results in all_agent_results.items():
            # Get test results for this agent across all datasets
            test_results = []
            for dataset_name, dataset_results in results.items():
                if isinstance(dataset_results, dict) and 'passed' in dataset_results:
                    test_results.append(dataset_results)
            
            agent_thresholds = thresholds.get(agent_name, {})
            scorecard = self.generate_agent_scorecard(
                agent_name, test_results, agent_thresholds
            )
            scorecards[agent_name] = scorecard
        
        return scorecards
    
    def format_scorecard_markdown(self, scorecard: Dict[str, Any]) -> str:
        """Format a scorecard as markdown."""
        lines = []
        lines.append(f"## {scorecard['agent_name']} Scorecard")
        lines.append("")
        lines.append(f"**Overall Grade: {scorecard['overall_grade']}**")
        lines.append("")
        lines.append(f"- Pass Rate: {scorecard['pass_rate']:.1%}")
        lines.append(f"- Total Tests: {scorecard['total_tests']}")
        lines.append(f"- Passed: {scorecard['passed_tests']}")
        lines.append(f"- Failed: {scorecard['failed_tests']}")
        lines.append("")
        
        if scorecard['average_metrics']:
            lines.append("### Average Metrics")
            lines.append("")
            for metric_name, metric_value in scorecard['average_metrics'].items():
                threshold = scorecard['thresholds'].get(metric_name, 'N/A')
                status = "✓" if isinstance(threshold, (int, float)) and metric_value >= threshold else "✗"
                lines.append(f"- {metric_name}: {metric_value:.3f} {status} (threshold: {threshold})")
            lines.append("")
        
        return "\n".join(lines)

