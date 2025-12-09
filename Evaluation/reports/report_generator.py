"""
Comprehensive Report Generator
Generates markdown reports with all evaluation results.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

from reports.scorecard_generator import ScorecardGenerator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "Evaluation/reports"):
        """Initialize the report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorecard_gen = ScorecardGenerator()
        
    def generate_executive_summary(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        workflow_results: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate executive summary section."""
        lines = []
        lines.append("# Executive Summary")
        lines.append("")
        lines.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Calculate overall pass/fail for each agent
        agent_summary = {}
        for agent_name, results in agent_results.items():
            total = 0
            passed = 0
            for dataset_name, result in results.items():
                if isinstance(result, dict) and 'passed' in result:
                    total += 1
                    if result['passed']:
                        passed += 1
            
            pass_rate = passed / total if total > 0 else 0.0
            agent_summary[agent_name] = {
                'pass_rate': pass_rate,
                'status': 'PASS' if pass_rate >= 0.80 else 'FAIL'
            }
        
        lines.append("## Agent-Level Performance")
        lines.append("")
        lines.append("| Agent | Pass Rate | Status |")
        lines.append("|-------|-----------|--------|")
        for agent_name, summary in agent_summary.items():
            lines.append(f"| {agent_name} | {summary['pass_rate']:.1%} | {summary['status']} |")
        lines.append("")
        
        # System-level summary
        if workflow_results:
            aggregate = workflow_results.get('aggregate_metrics', {})
            lines.append("## System-Level Performance")
            lines.append("")
            lines.append(f"- Total Datasets Evaluated: {aggregate.get('total_datasets', 0)}")
            lines.append(f"- Successful Workflows: {aggregate.get('successful_runs', 0)}")
            lines.append(f"- Success Rate: {aggregate.get('success_rate', 0):.1%}")
            lines.append(f"- Average Layer 2 Success Rate: {aggregate.get('average_layer2_success_rate', 0):.1%}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_agent_scorecards_section(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> str:
        """Generate agent scorecards section."""
        lines = []
        lines.append("# Agent-by-Agent Detailed Results")
        lines.append("")
        
        # Generate scorecards
        scorecards = self.scorecard_gen.generate_all_scorecards(agent_results, config)
        
        for agent_name, scorecard in scorecards.items():
            lines.append(self.scorecard_gen.format_scorecard_markdown(scorecard))
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_system_performance_section(
        self,
        workflow_results: Dict[str, Any]
    ) -> str:
        """Generate system-level performance section."""
        lines = []
        lines.append("# System-Level Performance Summary")
        lines.append("")
        
        aggregate = workflow_results.get('aggregate_metrics', {})
        individual = workflow_results.get('individual_results', {})
        
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append(f"- **Total Datasets:** {aggregate.get('total_datasets', 0)}")
        lines.append(f"- **Successful Runs:** {aggregate.get('successful_runs', 0)}")
        lines.append(f"- **Success Rate:** {aggregate.get('success_rate', 0):.1%}")
        lines.append(f"- **Average Execution Time:** {aggregate.get('average_execution_time', 0):.2f} seconds")
        lines.append(f"- **Average Layer 2 Success Rate:** {aggregate.get('average_layer2_success_rate', 0):.1%}")
        lines.append("")
        
        lines.append("## Per-Dataset Results")
        lines.append("")
        lines.append("| Dataset | Success | Execution Time (s) | Layer 2 Success Rate |")
        lines.append("|---------|---------|-------------------|---------------------|")
        
        for dataset_name, result in individual.items():
            success = "✓" if result.get('success') else "✗"
            exec_time = result.get('execution_metrics', {}).get('execution_time_seconds', 0)
            layer2_rate = result.get('execution_metrics', {}).get('layer2_success_rate', 0)
            lines.append(f"| {dataset_name} | {success} | {exec_time:.2f} | {layer2_rate:.1%} |")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_edge_case_results_section(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate edge case test results section."""
        lines = []
        lines.append("# Edge Case Test Results")
        lines.append("")
        
        # Identify edge case datasets
        edge_case_datasets = []
        for dataset_name, meta in metadata.items():
            expected_issues = meta.get('expected_issues', [])
            if any('leakage' in str(issue).lower() or 'imbalance' in str(issue).lower() 
                   or 'multicollinearity' in str(issue).lower() 
                   or 'dimensionality' in str(issue).lower() 
                   for issue in expected_issues):
                edge_case_datasets.append(dataset_name)
        
        for dataset_name in edge_case_datasets:
            lines.append(f"## {dataset_name}")
            lines.append("")
            
            meta = metadata.get(dataset_name, {})
            expected_issues = meta.get('expected_issues', [])
            lines.append("**Expected Issues:**")
            for issue in expected_issues:
                lines.append(f"- {issue}")
            lines.append("")
            
            # Check agent results
            lines.append("**Agent Performance:**")
            for agent_name, results in agent_results.items():
                if dataset_name in results:
                    result = results[dataset_name]
                    status = "✓ PASS" if result.get('passed') else "✗ FAIL"
                    lines.append(f"- {agent_name}: {status}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_failure_analysis_section(
        self,
        agent_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate failure analysis section."""
        lines = []
        lines.append("# Failure Analysis")
        lines.append("")
        
        # Count failures by agent
        failure_counts = {}
        failure_details = {}
        
        for agent_name, results in agent_results.items():
            failures = []
            for dataset_name, result in results.items():
                if isinstance(result, dict) and not result.get('passed', True):
                    failures.append(dataset_name)
                    if agent_name not in failure_details:
                        failure_details[agent_name] = []
                    failure_details[agent_name].append({
                        'dataset': dataset_name,
                        'error': result.get('metrics', {}).get('error', 'Unknown error')
                    })
            failure_counts[agent_name] = len(failures)
        
        lines.append("## Failure Counts by Agent")
        lines.append("")
        lines.append("| Agent | Failure Count |")
        lines.append("|-------|--------------|")
        for agent_name, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {agent_name} | {count} |")
        lines.append("")
        
        lines.append("## Failure Details")
        lines.append("")
        for agent_name, failures in failure_details.items():
            lines.append(f"### {agent_name}")
            lines.append("")
            for failure in failures:
                lines.append(f"- **{failure['dataset']}:** {failure['error']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_recommendations_section(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> str:
        """Generate recommendations for improvement."""
        lines = []
        lines.append("# Recommendations for Improvement")
        lines.append("")
        
        # Analyze failures and generate recommendations
        recommendations = []
        
        # Check each agent
        for agent_name, results in agent_results.items():
            failures = [r for r in results.values() if isinstance(r, dict) and not r.get('passed', True)]
            if failures:
                recommendations.append({
                    'agent': agent_name,
                    'issue': f"{len(failures)} test failures",
                    'recommendation': f"Review {agent_name} implementation and address identified issues"
                })
        
        # Check Layer 2 success rate
        # This would require workflow results
        
        if recommendations:
            lines.append("## Priority Recommendations")
            lines.append("")
            for idx, rec in enumerate(recommendations[:5], 1):  # Top 5
                lines.append(f"{idx}. **{rec['agent']}:** {rec['recommendation']}")
            lines.append("")
        else:
            lines.append("No critical issues identified. System is performing well.")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_full_report(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        workflow_results: Optional[Dict[str, Any]],
        metadata: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
        output_filename: str = "evaluation_report.md"
    ) -> str:
        """Generate full comprehensive report."""
        logger.info("Generating comprehensive evaluation report...")
        
        sections = []
        
        # Executive Summary
        sections.append(self.generate_executive_summary(agent_results, workflow_results, config))
        
        # Agent Scorecards
        sections.append(self.generate_agent_scorecards_section(agent_results, config))
        
        # System Performance
        if workflow_results:
            sections.append(self.generate_system_performance_section(workflow_results))
        
        # Edge Case Results
        sections.append(self.generate_edge_case_results_section(agent_results, metadata))
        
        # Failure Analysis
        sections.append(self.generate_failure_analysis_section(agent_results))
        
        # Recommendations
        sections.append(self.generate_recommendations_section(agent_results, config))
        
        # Combine all sections
        full_report = "\n\n".join(sections)
        
        # Save report
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(full_report)
        
        logger.info(f"Report saved to {output_path}")
        
        return full_report

