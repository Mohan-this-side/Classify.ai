"""
Generate agent-level plots directly from the full_workflow_weatherAUS_report.json file
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('default')
sns.set_palette("husl")


def load_report(report_path: Path) -> Dict[str, Any]:
    """Load the workflow report JSON"""
    with open(report_path, 'r') as f:
        return json.load(f)


def extract_agent_data(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract agent execution data from report"""
    agent_data = {}
    
    # Extract from steps
    for step in report.get('steps', []):
        if 'Agent Complete:' in step.get('step', ''):
            agent_name = step['step'].replace('Agent Complete: ', '')
            details = step.get('details', {})
            
            if agent_name not in agent_data:
                agent_data[agent_name] = {
                    'execution_time': details.get('execution_time', 0),
                    'layer1': details.get('layer1', False),
                    'layer2': details.get('layer2', False),
                    'docker_success': details.get('docker_success', False),
                    'state_keys_added': details.get('new_state_keys', 0),
                    'timestamp': step.get('timestamp', '')
                }
    
    # Also check agents section if it exists
    if 'agents' in report:
        for agent_name, agent_info in report['agents'].items():
            if agent_name not in agent_data:
                agent_data[agent_name] = {}
            agent_data[agent_name].update({
                'layer1_executed': agent_info.get('layer1_executed', False),
                'layer2_executed': agent_info.get('layer2_executed', False),
                'layer2_docker_success': agent_info.get('layer2_docker_success', False),
                'state_keys_added': agent_info.get('state_keys_added', [])
            })
    
    return agent_data


def extract_problem_solution_data(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract problem-solving data"""
    problem_data = {
        'problems_detected': [],
        'problems_solved': [],
        'solutions_applied': []
    }
    
    if 'problem_analysis' in report:
        problems = report['problem_analysis'].get('problems', [])
        problem_data['problems_detected'] = [p.get('type', 'Unknown') for p in problems]
    
    if 'solution_summary' in report:
        solutions = report['solution_summary'].get('solutions', [])
        problem_data['solutions_applied'] = solutions
    
    return problem_data


def generate_agent_execution_time_plot(agent_data: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate execution time plot"""
    agents = list(agent_data.keys())
    times = [agent_data[agent].get('execution_time', 0) for agent in agents]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = ['#3498db' if t < 30 else '#f39c12' if t < 60 else '#e74c3c' for t in times]
    bars = ax1.barh(agents, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Agent Execution Time Analysis', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add time labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax1.text(bar.get_width() + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                f'{time:.2f}s', ha='left', va='center',
                fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Pie chart
    ax2.pie(times, labels=agents, autopct='%1.1f%%', startangle=90,
           colors=colors, textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax2.set_title('Time Distribution Across Agents', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / "agent_execution_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: agent_execution_time.png")


def generate_agent_methods_plot(agent_data: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate methods used by each agent"""
    agents = list(agent_data.keys())
    
    methods_data = {}
    for agent_name, data in agent_data.items():
        methods = []
        
        if data.get('layer1', False):
            methods.append("Layer 1 (Hardcoded)")
        if data.get('layer2', False):
            methods.append("Layer 2 (LLM)")
        if data.get('docker_success', False):
            methods.append("Docker Sandbox")
        
        # Agent-specific methods
        state_keys = data.get('state_keys_added', 0)
        if isinstance(state_keys, list):
            state_keys = len(state_keys)
        
        if 'Data Cleaning' in agent_name:
            if state_keys > 0:
                methods.extend(["Missing Value Handling", "Outlier Treatment", "Data Imputation"])
        elif 'Feature Engineering' in agent_name:
            if state_keys > 0:
                methods.extend(["Feature Creation", "Multicollinearity Removal", "Encoding"])
        elif 'ML Building' in agent_name or 'ML Builder' in agent_name:
            if state_keys > 0:
                methods.extend(["Model Training", "Hyperparameter Tuning", "Model Selection"])
        elif 'EDA' in agent_name:
            if state_keys > 0:
                methods.extend(["Statistical Analysis", "Visualization", "Problem Detection"])
        elif 'Data Discovery' in agent_name:
            if state_keys > 0:
                methods.extend(["Data Profiling", "Type Detection", "Summary Statistics"])
        
        methods_data[agent_name] = methods
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_methods = set()
    for methods in methods_data.values():
        all_methods.update(methods)
    all_methods = sorted(list(all_methods))
    
    method_counts = {method: [] for method in all_methods}
    for agent in agents:
        for method in all_methods:
            method_counts[method].append(1 if method in methods_data[agent] else 0)
    
    x = np.arange(len(agents))
    width = 0.8 / len(all_methods) if len(all_methods) > 0 else 0.8
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(all_methods))) if len(all_methods) > 0 else ['#3498db']
    
    bottom = np.zeros(len(agents))
    for i, method in enumerate(all_methods):
        ax.bar(x, method_counts[method], width, label=method, bottom=bottom, 
              color=colors_map[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += np.array(method_counts[method])
    
    ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Methods Used (Count)', fontsize=12, fontweight='bold')
    ax.set_title('Methods and Techniques Used by Each Agent', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "agent_methods_used.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: agent_methods_used.png")


def generate_detailed_agent_evaluation(agent_data: Dict[str, Dict[str, Any]], output_dir: Path):
    """Create comprehensive agent evaluation dashboard"""
    agents = list(agent_data.keys())
    
    # Extract metrics
    scores = []
    times = []
    layer1_success = []
    layer2_success = []
    docker_success = []
    state_keys = []
    
    for agent in agents:
        data = agent_data[agent]
        times.append(data.get('execution_time', 0))
        layer1_success.append(1 if data.get('layer1', False) else 0)
        layer2_success.append(1 if data.get('layer2', False) else 0)
        docker_success.append(1 if data.get('docker_success', False) else 0)
        state_keys_val = data.get('state_keys_added', 0)
        if isinstance(state_keys_val, list):
            state_keys_val = len(state_keys_val)
        state_keys.append(state_keys_val)
        
        # Calculate score based on success rates
        score = (layer1_success[-1] * 0.3 + layer2_success[-1] * 0.3 + 
                docker_success[-1] * 0.2 + min(state_keys[-1] / 10, 1) * 0.2)
        scores.append(score)
    
    # Determine grades
    grades = ['A' if s >= 0.9 else 'B' if s >= 0.8 else 'C' if s >= 0.7 else 'D' if s >= 0.6 else 'F' for s in scores]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Performance Scores
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#2ecc71' if g == 'A' else '#f39c12' if g == 'B' else '#e74c3c' for g in grades]
    bars = ax1.barh(agents, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, score, grade) in enumerate(zip(bars, scores, grades)):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'{score:.2f} ({grade})', ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.set_xlabel('Performance Score', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Performance Scores', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 2. Execution Time
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(agents, times, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, time) in enumerate(zip(bars2, times)):
        ax2.text(time + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                f'{time:.1f}s', ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Execution Time', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 3. Layer Success Rates
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(agents))
    width = 0.25
    ax3.bar(x - width, layer1_success, width, label='Layer 1', color='#3498db', alpha=0.7, edgecolor='black')
    ax3.bar(x, layer2_success, width, label='Layer 2', color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.bar(x + width, docker_success, width, label='Docker', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Success (1=Yes, 0=No)', fontsize=11, fontweight='bold')
    ax3.set_title('Layer Execution Success', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(agents, rotation=45, ha='right', fontsize=8, fontweight='bold')
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.set_ylim(0, 1.2)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. State Keys Added
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.barh(agents, state_keys, color='#16a085', alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, keys) in enumerate(zip(bars4, state_keys)):
        ax4.text(keys + max(state_keys)*0.05 if max(state_keys) > 0 else 0.5, bar.get_y() + bar.get_height()/2,
                str(keys), ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax4.set_xlabel('State Keys Added', fontsize=11, fontweight='bold')
    ax4.set_title('State Keys Added by Each Agent', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 5. Success Rate Breakdown
    ax5 = fig.add_subplot(gs[1, 1])
    success_rates = []
    for i in range(len(agents)):
        rate = (layer1_success[i] + layer2_success[i] + docker_success[i]) / 3.0
        success_rates.append(rate)
    bars5 = ax5.barh(agents, success_rates, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, rate) in enumerate(zip(bars5, success_rates)):
        ax5.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'{rate:.2f}', ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax5.set_xlabel('Success Rate', fontsize=11, fontweight='bold')
    ax5.set_title('Overall Success Rate', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 1.1)
    ax5.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 6. Efficiency Score (State Keys per Second)
    ax6 = fig.add_subplot(gs[1, 2])
    efficiency = []
    for i in range(len(agents)):
        if times[i] > 0:
            eff = state_keys[i] / times[i]
        else:
            eff = state_keys[i] if state_keys[i] > 0 else 0
        efficiency.append(eff)
    bars6 = ax6.barh(agents, efficiency, color='#f39c12', alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, eff) in enumerate(zip(bars6, efficiency)):
        ax6.text(bar.get_width() + max(efficiency)*0.05 if max(efficiency) > 0 else 0.1, bar.get_y() + bar.get_height()/2,
                f'{eff:.2f}', ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax6.set_xlabel('Efficiency (Keys/Second)', fontsize=11, fontweight='bold')
    ax6.set_title('Agent Efficiency', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 7. Comprehensive Heatmap
    ax7 = fig.add_subplot(gs[2, :])
    metrics_data = {
        'Performance Score': scores,
        'Execution Time (norm)': [(max(times) - t) / max(times) if max(times) > 0 else 0 for t in times],
        'Layer 1 Success': layer1_success,
        'Layer 2 Success': layer2_success,
        'Docker Success': docker_success,
        'State Keys (norm)': [s / max(state_keys) if max(state_keys) > 0 else 0 for s in state_keys],
        'Success Rate': success_rates,
        'Efficiency (norm)': [e / max(efficiency) if max(efficiency) > 0 else 0 for e in efficiency]
    }
    heatmap_data = pd.DataFrame(metrics_data, index=agents).T
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax7, 
               cbar_kws={'label': 'Normalized Score'}, vmin=0, vmax=1,
               linewidths=1, linecolor='black', square=False)
    ax7.set_title('Comprehensive Agent Performance Heatmap', fontsize=13, fontweight='bold', pad=15)
    ax7.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.suptitle('Comprehensive Agent Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_dir / "detailed_agent_evaluation_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: detailed_agent_evaluation_dashboard.png")


def generate_agent_workflow_timeline(agent_data: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate workflow timeline showing agent execution order and duration"""
    agents = list(agent_data.keys())
    times = [agent_data[agent].get('execution_time', 0) for agent in agents]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create timeline
    y_pos = np.arange(len(agents))
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#e74c3c', '#16a085', '#34495e']
    
    bars = ax.barh(y_pos, times, color=colors[:len(agents)], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add time labels
    cumulative_time = 0
    for i, (bar, time, agent) in enumerate(zip(bars, times, agents)):
        # Start position
        start_x = cumulative_time
        # End position
        end_x = cumulative_time + time
        
        # Add label with start and end time
        ax.text(bar.get_width() / 2 + start_x, bar.get_y() + bar.get_height()/2,
               f'{time:.2f}s', ha='center', va='center',
               fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Add agent name with status indicators
        status = []
        if agent_data[agent].get('layer1', False):
            status.append('L1✓')
        if agent_data[agent].get('layer2', False):
            status.append('L2✓')
        if agent_data[agent].get('docker_success', False):
            status.append('D✓')
        status_str = ' '.join(status)
        
        ax.text(-max(times)*0.1, bar.get_y() + bar.get_height()/2,
               f'{agent}\n{status_str}', ha='right', va='center',
               fontweight='bold', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        cumulative_time = end_x
    
    ax.set_xlabel('Cumulative Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Agent Workflow Timeline', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(-max(times)*0.3, cumulative_time * 1.1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='Layer 1'),
        Patch(facecolor='#9b59b6', alpha=0.7, label='Layer 2'),
        Patch(facecolor='#2ecc71', alpha=0.7, label='Docker')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "agent_workflow_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: agent_workflow_timeline.png")


def main():
    """Main function to generate all plots"""
    import sys
    
    # Find latest results directory
    eval_dir = Path(__file__).parent.parent
    results_dirs = sorted(eval_dir.glob("results/20*"), reverse=True)
    
    if not results_dirs:
        print("❌ No results directories found!")
        return
    
    latest_dir = results_dirs[0]
    report_path = latest_dir / "reports" / "full_workflow_weatherAUS_report.json"
    plots_dir = latest_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating agent plots from: {report_path}")
    print(f"📁 Output directory: {plots_dir}")
    
    if not report_path.exists():
        print(f"❌ Report file not found: {report_path}")
        return
    
    # Load report
    report = load_report(report_path)
    print(f"✅ Loaded report with {len(report.get('steps', []))} steps")
    
    # Extract agent data
    agent_data = extract_agent_data(report)
    print(f"✅ Extracted data for {len(agent_data)} agents")
    
    if not agent_data:
        print("❌ No agent data found in report!")
        return
    
    # Generate plots
    print("\n🎨 Generating plots...")
    generate_agent_execution_time_plot(agent_data, plots_dir)
    generate_agent_methods_plot(agent_data, plots_dir)
    generate_detailed_agent_evaluation(agent_data, plots_dir)
    generate_agent_workflow_timeline(agent_data, plots_dir)
    generate_model_performance_plot(report, plots_dir)
    
    print(f"\n✅ All plots generated successfully in: {plots_dir}")


def generate_model_performance_plot(report: Dict[str, Any], output_dir: Path):
    """Generate model performance visualization"""
    metrics = report.get('steps', [])
    model_metrics = None
    training_metrics = None
    feature_importance = None
    metrics_step = None
    
    for step in metrics:
        if step.get('step') == 'Metrics Extraction' and step.get('status') == 'COMPLETED':
            metrics_step = step
            details = step.get('details', {})
            model_metrics = details.get('model_metrics', {})
            training_metrics = details.get('training_metrics', {})
            feature_importance = details.get('feature_importance', {})
            break
    
    if not model_metrics:
        print("⚠️ No model metrics found in report")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Main metrics bar chart
    ax1 = axes[0, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [
        model_metrics.get('accuracy', 0),
        model_metrics.get('precision', 0),
        model_metrics.get('recall', 0),
        model_metrics.get('f1_score', 0)
    ]
    bars = ax1.bar(metric_names, metric_values, color=['#3498db', '#9b59b6', '#2ecc71', '#f39c12'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Training vs Test accuracy
    ax2 = axes[0, 1]
    train_acc = training_metrics.get('train_accuracy', 0) if training_metrics else 0
    test_acc = training_metrics.get('test_accuracy', 0) if training_metrics else 0
    ax2.bar(['Train', 'Test'], [train_acc, test_acc], color=['#3498db', '#2ecc71'], 
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Train vs Test Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for i, val in enumerate([train_acc, test_acc]):
        ax2.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Feature importance (top 10)
    ax3 = axes[1, 0]
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        ax3.barh(features, importances, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1)
        ax3.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 4. Cross-validation scores
    ax4 = axes[1, 1]
    cv_mean = training_metrics.get('cv_mean', 0) if training_metrics else 0
    cv_std = training_metrics.get('cv_std', 0) if training_metrics else 0
    ax4.bar(['CV Mean'], [cv_mean], color='#16a085', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.errorbar(['CV Mean'], [cv_mean], yerr=[cv_std], fmt='none', color='black', capsize=10, capthick=2)
    ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax4.set_title(f'Cross-Validation\nMean: {cv_mean:.3f} ± {cv_std:.3f}', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: model_performance_analysis.png")


if __name__ == "__main__":
    main()

