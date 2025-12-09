"""
Plot Generator for Evaluation Results
Generates comprehensive visualizations for evaluation reports.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

# Set style with better colors
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLORS = {
    'pass': '#2ecc71',  # Green
    'fail': '#e74c3c',  # Red
    'warning': '#f39c12',  # Orange
    'info': '#3498db',  # Blue
    'primary': '#9b59b6'  # Purple
}


class PlotGenerator:
    """Generates visualizations for evaluation results."""
    
    def __init__(self, output_dir: str = "Evaluation/results/visualization/plots"):
        """Initialize the plot generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_agent_scorecard_heatmap(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        output_filename: str = "agent_scorecard_heatmap.png"
    ):
        """Generate heatmap showing pass/fail for each agent per dataset."""
        # Prepare data
        datasets = []
        agents = []
        scores = []
        
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                datasets.append(dataset_name)
                agents.append(agent_name)
                scores.append(1 if result.get('passed', False) else 0)
        
        df = pd.DataFrame({
            'dataset': datasets,
            'agent': agents,
            'passed': scores
        })
        
        # Create pivot table
        pivot = df.pivot_table(
            index='dataset',
            columns='agent',
            values='passed',
            aggfunc='mean',
            fill_value=0
        )
        
        # Create heatmap with better styling
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(
            pivot,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Pass Rate (1=Pass, 0=Fail)', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='gray',
            square=False
        )
        plt.title('Agent Scorecard: Pass/Fail Performance by Dataset', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Agent', fontsize=13, fontweight='bold')
        plt.ylabel('Dataset', fontsize=13, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45)
        plt.setp(plt.gca().get_xticklabels(), ha='right')
        plt.yticks(rotation=0)
        
        # Add grid lines
        ax.set_xticks(np.arange(len(pivot.columns)) + 0.5, minor=False)
        ax.set_yticks(np.arange(len(pivot.index)) + 0.5, minor=False)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved agent scorecard heatmap to {output_path}")
    
    def generate_metric_dashboard(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        output_filename: str = "metric_dashboard.png"
    ):
        """Generate dashboard with key metrics."""
        # Extract metrics
        metrics_data = []
        
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                metrics = result.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics_data.append({
                            'dataset': dataset_name,
                            'agent': agent_name,
                            'metric': metric_name,
                            'value': metric_value
                        })
        
        if not metrics_data:
            logger.warning("No metrics data available for dashboard")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Select key metrics to visualize
        key_metrics = ['type_detection_accuracy', 'imbalance_detection_rate',
                      'layer2_success_rate', 'pm_accuracy', 'anti_cheating_score']
        
        available_metrics = [m for m in key_metrics if m in df['metric'].values]
        
        if not available_metrics:
            available_metrics = df['metric'].unique()[:5]  # Use first 5 available
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            metric_df = df[df['metric'] == metric]
            if len(metric_df) > 0:
                # Calculate average for threshold line
                avg_value = metric_df['value'].mean()
                
                # Create barplot with colors
                bars = sns.barplot(
                    data=metric_df,
                    x='agent',
                    y='value',
                    ax=axes[idx],
                    palette='viridis',
                    alpha=0.8
                )
                
                # Add average line
                axes[idx].axhline(y=avg_value, color='red', linestyle='--', 
                                linewidth=2, label=f'Average: {avg_value:.2f}')
                
                # Add value labels on bars
                for container in bars.containers:
                    bars.bar_label(container, fmt='%.2f', padding=3)
                
                axes[idx].set_title(metric.replace('_', ' ').title(), 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('Score', fontsize=10)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), ha='right')
                axes[idx].legend(loc='upper right', fontsize=8)
                axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Key Metrics Dashboard Across All Agents', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved metric dashboard to {output_path}")
    
    def generate_layer_comparison(
        self,
        workflow_results: Dict[str, Any],
        output_filename: str = "layer_comparison.png"
    ):
        """Generate comparison of Layer 1 vs Layer 2 success rates."""
        # Extract layer 2 success rates
        datasets = []
        success_rates = []
        
        individual_results = workflow_results.get('individual_results', {})
        for dataset_name, result in individual_results.items():
            if result.get('success'):
                datasets.append(dataset_name)
                success_rates.append(
                    result['execution_metrics'].get('layer2_success_rate', 0)
                )
        
        if not datasets:
            logger.warning("No workflow results available for layer comparison")
            return
        
        # Create bar chart with gradient colors
        plt.figure(figsize=(14, 7))
        
        # Color bars based on success rate
        colors = [COLORS['pass'] if rate >= 0.80 else COLORS['warning'] if rate >= 0.60 else COLORS['fail'] 
                 for rate in success_rates]
        
        bars = plt.bar(datasets, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add threshold line
        threshold = 0.80
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Target Threshold ({threshold:.0%})', zorder=0)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            color = 'white' if height < 0.5 else 'black'
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontweight='bold', color=color, fontsize=10)
        
        plt.title('Layer 2 (LLM-Generated Code) Success Rate by Dataset', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontsize=13, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=13, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
        plt.xticks(rotation=45)
        plt.setp(plt.gca().get_xticklabels(), ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved layer comparison to {output_path}")
    
    def generate_failure_analysis(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        output_filename: str = "failure_analysis.png"
    ):
        """Generate Pareto chart of failure modes."""
        # Count failures by agent
        failure_counts = {}
        
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                if not result.get('passed', True):
                    failure_counts[agent_name] = failure_counts.get(agent_name, 0) + 1
        
        if not failure_counts:
            logger.info("No failures to analyze")
            return
        
        # Sort by count
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        agents = [item[0] for item in sorted_failures]
        counts = [item[1] for item in sorted_failures]
        
        # Create Pareto chart with better styling
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Bar chart with gradient colors
        colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, len(agents)))
        bars = ax1.bar(agents, counts, color=colors_bar, alpha=0.8, edgecolor='darkred', linewidth=1.5)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(count)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Agent', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Failure Count', fontsize=13, fontweight='bold', color='darkred')
        ax1.tick_params(axis='y', labelcolor='darkred')
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Cumulative line
        cumulative = np.cumsum(counts)
        cumulative_pct = cumulative / cumulative[-1] * 100 if cumulative[-1] > 0 else np.zeros_like(cumulative)
        
        ax2 = ax1.twinx()
        line = ax2.plot(agents, cumulative_pct, color='blue', marker='o', 
                       linewidth=3, markersize=10, label='Cumulative %', zorder=5)
        
        # Add percentage labels
        for i, (agent, pct) in enumerate(zip(agents, cumulative_pct)):
            ax2.text(i, pct + 2, f'{pct:.0f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9, color='blue')
        
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=13, fontweight='bold', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 110)
        
        plt.title('Failure Analysis: Pareto Chart (80/20 Rule)', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Add 80% line
        ax2.axhline(y=80, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7, label='80% Threshold')
        ax2.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved failure analysis to {output_path}")
    
    def generate_execution_timeline(
        self,
        workflow_results: Dict[str, Any],
        output_filename: str = "execution_timeline.png"
    ):
        """Generate Gantt-style visualization of workflow execution."""
        # This would require timing data for each stage
        # Simplified version for now
        individual_results = workflow_results.get('individual_results', {})
        
        datasets = []
        execution_times = []
        
        for dataset_name, result in individual_results.items():
            if result.get('success'):
                datasets.append(dataset_name)
                execution_times.append(
                    result['execution_metrics'].get('execution_time_seconds', 0)
                )
        
        if not datasets:
            return
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.barh(datasets, execution_times, color='teal', alpha=0.7)
        
        plt.title('Workflow Execution Time by Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Execution Time (seconds)', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved execution timeline to {output_path}")
    
    def generate_quality_score_distribution(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        output_filename: str = "quality_score_distribution.png"
    ):
        """Generate histogram of quality scores."""
        # Extract all quality scores
        scores = []
        
        for dataset_name, results in agent_results.items():
            for agent_name, result in results.items():
                metrics = result.get('metrics', {})
                # Look for score-like metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                        if 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
                            scores.append(metric_value)
        
        if not scores:
            logger.warning("No quality scores found")
            return
        
        # Create enhanced histogram
        plt.figure(figsize=(12, 7))
        n, bins, patches = plt.hist(scores, bins=20, color='skyblue', edgecolor='black', 
                                   alpha=0.7, linewidth=1.5)
        
        # Color bars based on score ranges
        for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
            if bin_val >= 0.8:
                patch.set_facecolor(COLORS['pass'])
            elif bin_val >= 0.6:
                patch.set_facecolor(COLORS['warning'])
            else:
                patch.set_facecolor(COLORS['fail'])
        
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        
        plt.axvline(x=mean_score, color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean: {mean_score:.3f}')
        plt.axvline(x=median_score, color='blue', linestyle='--', linewidth=2.5,
                   label=f'Median: {median_score:.3f}')
        plt.axvline(x=0.85, color='green', linestyle=':', linewidth=2,
                   label='Target (0.85)', alpha=0.7)
        
        plt.title('Quality Score Distribution Across All Agents', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Quality Score', fontsize=13, fontweight='bold')
        plt.ylabel('Frequency', fontsize=13, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved quality score distribution to {output_path}")
    
    def generate_agent_performance_radar(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        output_filename: str = "agent_performance_radar.png"
    ):
        """Generate radar chart comparing agent performance."""
        try:
            from math import pi
            
            # Aggregate metrics per agent
            agent_metrics = {}
            
            for dataset_name, results in agent_results.items():
                for agent_name, result in results.items():
                    if agent_name not in agent_metrics:
                        agent_metrics[agent_name] = {'scores': [], 'pass_count': 0, 'total': 0}
                    
                    agent_metrics[agent_name]['total'] += 1
                    if result.get('passed', False):
                        agent_metrics[agent_name]['pass_count'] += 1
                    
                    # Collect numeric metrics
                    metrics = result.get('metrics', {})
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                            agent_metrics[agent_name]['scores'].append(metric_value)
            
            if not agent_metrics:
                logger.warning("No agent metrics available for radar chart")
                return
            
            # Calculate average scores per agent
            agent_avg_scores = {}
            for agent_name, data in agent_metrics.items():
                pass_rate = data['pass_count'] / data['total'] if data['total'] > 0 else 0
                avg_score = np.mean(data['scores']) if data['scores'] else 0
                agent_avg_scores[agent_name] = {
                    'pass_rate': pass_rate,
                    'avg_score': avg_score,
                    'overall': (pass_rate + avg_score) / 2
                }
            
            # Select top agents for clarity
            sorted_agents = sorted(agent_avg_scores.items(), 
                                 key=lambda x: x[1]['overall'], reverse=True)[:6]
            
            if len(sorted_agents) < 2:
                logger.warning("Not enough agents for radar chart")
                return
            
            # Create radar chart
            categories = ['Pass Rate', 'Avg Score', 'Overall']
            N = len(categories)
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
            
            colors_list = plt.cm.Set3(np.linspace(0, 1, len(sorted_agents)))
            
            for idx, (agent_name, scores) in enumerate(sorted_agents):
                values = [scores['pass_rate'], scores['avg_score'], scores['overall']]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2.5, label=agent_name,
                       color=colors_list[idx], alpha=0.7)
                ax.fill(angles, values, alpha=0.15, color=colors_list[idx])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            plt.title('Agent Performance Radar Chart (Top Agents)', 
                     fontsize=18, fontweight='bold', pad=30)
            plt.tight_layout()
            
            output_path = self.output_dir / output_filename
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved agent performance radar chart to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate radar chart: {e}")
    
    def generate_all_plots(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        workflow_results: Optional[Dict[str, Any]] = None
    ):
        """Generate all plots."""
        logger.info("Generating all evaluation plots...")
        
        self.generate_agent_scorecard_heatmap(agent_results)
        self.generate_metric_dashboard(agent_results)
        self.generate_failure_analysis(agent_results)
        self.generate_quality_score_distribution(agent_results)
        self.generate_agent_performance_radar(agent_results)
        
        if workflow_results:
            self.generate_layer_comparison(workflow_results)
            self.generate_execution_timeline(workflow_results)
        
        logger.info("All plots generated successfully")

