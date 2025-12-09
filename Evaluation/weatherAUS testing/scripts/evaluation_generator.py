"""
Comprehensive Evaluation Generator

Generates agent-wise and system-level evaluation reports with comparisons and visualizations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class EvaluationGenerator:
    """Generate comprehensive agent-wise and system-level evaluations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.EvaluationGenerator")
        
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_comprehensive_evaluation(
        self,
        tracker: Any,
        state: Dict[str, Any],
        problem_analysis: Dict[str, Any],
        reasoning_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation with agent-wise and system-level metrics"""
        
        # 1. Agent-wise evaluation
        agent_evaluation = self._evaluate_agents(tracker, state, problem_analysis, reasoning_data)
        
        # 2. System-level evaluation
        system_evaluation = self._evaluate_system(tracker, state, problem_analysis)
        
        # 3. Problem-solving evaluation
        problem_solving_evaluation = self._evaluate_problem_solving(
            problem_analysis, reasoning_data, tracker
        )
        
        # 4. Generate visualizations
        self._generate_evaluation_plots(
            agent_evaluation, system_evaluation, problem_solving_evaluation
        )
        
        # 5. Generate tables
        self._generate_evaluation_tables(
            agent_evaluation, system_evaluation, problem_solving_evaluation
        )
        
        # 6. Generate markdown report
        report_path = self._generate_markdown_report(
            agent_evaluation, system_evaluation, problem_solving_evaluation
        )
        
        return {
            "agent_evaluation": agent_evaluation,
            "system_evaluation": system_evaluation,
            "problem_solving_evaluation": problem_solving_evaluation,
            "report_path": str(report_path)
        }
    
    def _evaluate_agents(
        self,
        tracker: Any,
        state: Dict[str, Any],
        problem_analysis: Dict[str, Any],
        reasoning_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate each agent individually"""
        agent_scores = {}
        
        agent_roles = {
            "Data Discovery": {
                "expected_outputs": ["dataset_summary", "data_types", "basic_statistics"],
                "success_criteria": ["layer1_executed", "state_keys_added > 0"]
            },
            "EDA Analysis": {
                "expected_outputs": ["eda_plots", "correlations", "target_relationships", "problem_detection"],
                "success_criteria": ["layer1_executed", "problem_detection in state"]
            },
            "Data Cleaning": {
                "expected_outputs": ["cleaned_dataset", "cleaning_actions_taken", "cleaning_reasoning"],
                "success_criteria": ["layer1_executed", "cleaned_dataset in state", "missing_values reduced"]
            },
            "Feature Engineering": {
                "expected_outputs": ["engineered_features", "feature_reasoning", "removed_features"],
                "success_criteria": ["layer1_executed", "engineered_features in state", "features_created > 0"]
            },
            "ML Building": {
                "expected_outputs": ["best_model", "model_selection_results", "imbalance_handling", "temporal_split_info"],
                "success_criteria": ["layer1_executed", "best_model in state", "model trained successfully"]
            },
            "Model Evaluation": {
                "expected_outputs": ["evaluation_metrics", "cross_validation_scores"],
                "success_criteria": ["layer1_executed", "evaluation_metrics in state", "metrics calculated"]
            },
            "Technical Reporter": {
                "expected_outputs": ["final_report", "technical_documentation"],
                "success_criteria": ["layer1_executed", "final_report in state"]
            }
        }
        
        for agent_name, result in tracker.agent_results.items():
            role_info = agent_roles.get(agent_name, {})
            expected_outputs = role_info.get("expected_outputs", [])
            success_criteria = role_info.get("success_criteria", [])
            
            # Check outputs
            outputs_present = []
            for output in expected_outputs:
                if output in state or any(output in str(k) for k in state.keys()):
                    outputs_present.append(output)
            
            # Check success criteria
            criteria_met = []
            for criterion in success_criteria:
                if "layer1_executed" in criterion:
                    criteria_met.append(result.get("layer1_executed", False))
                elif "layer2_executed" in criterion:
                    criteria_met.append(result.get("layer2_executed", False))
                elif "docker_success" in criterion:
                    criteria_met.append(result.get("layer2_docker_success", False))
                elif "state_keys_added" in criterion:
                    criteria_met.append(len(result.get("state_keys_added", [])) > 0)
                elif " in state" in criterion:
                    key = criterion.split(" in state")[0]
                    criteria_met.append(key in state)
                elif ">" in criterion:
                    # Parse comparison like "features_created > 0"
                    parts = criterion.split(" > ")
                    key = parts[0].strip()
                    threshold = float(parts[1].strip())
                    value = state.get(key, 0)
                    criteria_met.append(value > threshold)
            
            # Calculate scores
            output_score = len(outputs_present) / len(expected_outputs) if expected_outputs else 1.0
            criteria_score = sum(criteria_met) / len(criteria_met) if criteria_met else 1.0
            layer1_score = 1.0 if result.get("layer1_executed") else 0.0
            layer2_score = 1.0 if result.get("layer2_executed") and result.get("layer2_docker_success") else 0.5 if result.get("layer2_executed") else 0.0
            
            # Overall agent score (weighted)
            overall_score = (
                output_score * 0.3 +
                criteria_score * 0.3 +
                layer1_score * 0.2 +
                layer2_score * 0.2
            )
            
            # Grade assignment
            if overall_score >= 0.9:
                grade = "A"
            elif overall_score >= 0.8:
                grade = "B"
            elif overall_score >= 0.7:
                grade = "C"
            elif overall_score >= 0.6:
                grade = "D"
            else:
                grade = "F"
            
            agent_scores[agent_name] = {
                "overall_score": overall_score,
                "grade": grade,
                "output_score": output_score,
                "criteria_score": criteria_score,
                "layer1_score": layer1_score,
                "layer2_score": layer2_score,
                "outputs_present": outputs_present,
                "outputs_missing": [o for o in expected_outputs if o not in outputs_present],
                "criteria_met": sum(criteria_met),
                "criteria_total": len(criteria_met),
                "execution_time": result.get("execution_time", 0),
                "layer1_executed": result.get("layer1_executed", False),
                "layer2_executed": result.get("layer2_executed", False),
                "docker_success": result.get("layer2_docker_success", False),
                "state_keys_added": len(result.get("state_keys_added", []))
            }
        
        return agent_scores
    
    def _evaluate_system(
        self,
        tracker: Any,
        state: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate system-level performance"""
        
        # Workflow completion
        agents_executed = len(tracker.agent_results) if hasattr(tracker, 'agent_results') else 0
        agents_total = 7  # Expected agents
        completion_rate = agents_executed / agents_total if agents_total > 0 else 0
        
        # Success rates
        agent_results = tracker.agent_results if hasattr(tracker, 'agent_results') else {}
        layer1_success_rate = sum(
            1 for r in agent_results.values() if r.get("layer1_executed")
        ) / agents_executed if agents_executed > 0 else 0
        
        layer2_success_rate = sum(
            1 for r in agent_results.values() if r.get("layer2_executed")
        ) / agents_executed if agents_executed > 0 else 0
        
        docker_success_rate = sum(
            1 for r in agent_results.values() if r.get("layer2_docker_success")
        ) / agents_executed if agents_executed > 0 else 0
        
        # Total execution time
        total_time = sum(r.get("execution_time", 0) for r in agent_results.values())
        avg_time_per_agent = total_time / agents_executed if agents_executed > 0 else 0
        
        # Model performance (if available)
        model_metrics = state.get("evaluation_metrics", {}) if state else {}
        model_performance = {
            "accuracy": model_metrics.get("accuracy", 0) if model_metrics else 0,
            "precision": model_metrics.get("precision", 0) if model_metrics else 0,
            "recall": model_metrics.get("recall", 0) if model_metrics else 0,
            "f1_score": model_metrics.get("f1_score", 0) if model_metrics else 0,
            "roc_auc": model_metrics.get("roc_auc", 0) if model_metrics and "roc_auc" in model_metrics else None
        }
        
        # Data quality improvement
        problem_metrics = problem_analysis.get("metrics", {}) if problem_analysis else {}
        initial_missing = problem_metrics.get("problems_detected", 0) if problem_metrics else 0
        final_missing = state.get("missing_values", {}) if state else {}
        if isinstance(final_missing, dict):
            final_missing_count = sum(final_missing.values())
        else:
            final_missing_count = 0
        
        # Problem resolution
        problems_detected = problem_metrics.get("problems_detected", 0) if problem_metrics else 0
        problems_resolved = self._count_resolved_problems(state, problem_analysis) if problem_analysis else 0
        
        # System score
        system_score = (
            completion_rate * 0.25 +
            layer1_success_rate * 0.20 +
            layer2_success_rate * 0.15 +
            docker_success_rate * 0.15 +
            (problems_resolved / problems_detected if problems_detected > 0 else 1.0) * 0.15 +
            (model_performance.get("f1_score", 0) if model_performance.get("f1_score") else 0.5) * 0.10
        )
        
        return {
            "completion_rate": completion_rate,
            "layer1_success_rate": layer1_success_rate,
            "layer2_success_rate": layer2_success_rate,
            "docker_success_rate": docker_success_rate,
            "total_execution_time": total_time,
            "avg_time_per_agent": avg_time_per_agent,
            "model_performance": model_performance,
            "data_quality_improvement": {
                "initial_missing": initial_missing,
                "final_missing": final_missing_count,
                "improvement": (initial_missing - final_missing_count) / initial_missing if initial_missing > 0 else 0
            },
            "problem_resolution": {
                "problems_detected": problems_detected,
                "problems_resolved": problems_resolved,
                "resolution_rate": problems_resolved / problems_detected if problems_detected > 0 else 0
            },
            "system_score": system_score,
            "system_grade": self._score_to_grade(system_score)
        }
    
    def _count_resolved_problems(
        self,
        state: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> int:
        """Count how many problems were resolved"""
        resolved = 0
        
        if not problem_analysis:
            return 0
        
        problems = problem_analysis.get("problems", []) if problem_analysis else []
        
        for problem in problems:
            problem_type = problem.get("type", "")
            
            if problem_type == "Missing Values":
                # Check if cleaning was done
                if "cleaning_actions_taken" in state or "cleaned_dataset" in state:
                    resolved += 1
            elif problem_type == "Class Imbalance":
                # Check if imbalance handling was applied
                if "imbalance_handling" in state or "model_selection_reasoning" in state:
                    resolved += 1
            elif problem_type == "Multicollinearity":
                # Check if features were removed
                if "feature_reasoning" in state:
                    fe_reasoning = state.get("feature_reasoning", {})
                    if fe_reasoning.get("removed"):
                        resolved += 1
            elif problem_type == "High Cardinality":
                # Check if encoding was applied
                if "feature_reasoning" in state:
                    fe_reasoning = state.get("feature_reasoning", {})
                    if fe_reasoning.get("high_cardinality_encoding"):
                        resolved += 1
            elif problem_type == "Outliers":
                # Check if outliers were handled
                if "cleaning_reasoning" in state:
                    cleaning_reasoning = state.get("cleaning_reasoning", {})
                    if any("outliers" in str(k).lower() for k in cleaning_reasoning.keys()):
                        resolved += 1
        
        return resolved
    
    def _evaluate_problem_solving(
        self,
        problem_analysis: Dict[str, Any],
        reasoning_data: Dict[str, Any],
        tracker: Any
    ) -> Dict[str, Any]:
        """Evaluate how well agents solved problems"""
        
        if not problem_analysis:
            return {
                "problem_solving_scores": {},
                "overall_score": 0.0,
                "grade": "F"
            }
        
        problems = problem_analysis.get("problems", []) if problem_analysis else []
        all_problems = reasoning_data.get("all_problems", {}) if reasoning_data else {}
        
        problem_solving_scores = {}
        
        for problem in problems:
            problem_type = problem.get("type", "")
            severity = problem.get("severity", "low")
            
            # Check if problem was detected
            detected = False
            if problem_type == "Class Imbalance":
                detected = all_problems.get("class_imbalance", {}).get("detected", False)
            elif problem_type == "Missing Values":
                detected = all_problems.get("missing_data", {}).get("detected", False)
            elif problem_type == "Multicollinearity":
                detected = all_problems.get("multicollinearity", {}).get("detected", False)
            elif problem_type == "Outliers":
                detected = all_problems.get("outliers", {}).get("detected", False)
            
            # Check if solution was applied
            solution_applied = False
            solution_details = {}
            
            if problem_type == "Class Imbalance":
                imbalance_reasoning = reasoning_data.get("imbalance", {})
                if imbalance_reasoning.get("handling"):
                    solution_applied = True
                    solution_details = imbalance_reasoning.get("handling", {})
            elif problem_type == "Missing Values":
                cleaning_reasoning = reasoning_data.get("cleaning", {})
                if cleaning_reasoning:
                    solution_applied = True
                    solution_details = {"actions": len(cleaning_reasoning)}
            elif problem_type == "Multicollinearity":
                multicollinearity_reasoning = reasoning_data.get("multicollinearity", {})
                if multicollinearity_reasoning.get("removal"):
                    solution_applied = True
                    solution_details = multicollinearity_reasoning.get("removal", {})
            elif problem_type == "Outliers":
                cleaning_reasoning = reasoning_data.get("cleaning", {})
                if any("outlier" in str(k).lower() for k in cleaning_reasoning.keys()):
                    solution_applied = True
            
            # Score: detection (0.4) + solution (0.6)
            score = (0.4 if detected else 0.0) + (0.6 if solution_applied else 0.0)
            
            problem_solving_scores[problem_type] = {
                "detected": detected,
                "solution_applied": solution_applied,
                "solution_details": solution_details,
                "score": score,
                "severity": severity
            }
        
        # Overall problem-solving score
        if problem_solving_scores:
            overall_score = sum(ps.get("score", 0) for ps in problem_solving_scores.values()) / len(problem_solving_scores)
        else:
            overall_score = 1.0
        
        return {
            "problem_solving_scores": problem_solving_scores,
            "overall_score": overall_score,
            "grade": self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_evaluation_plots(
        self,
        agent_evaluation: Dict[str, Any],
        system_evaluation: Dict[str, Any],
        problem_solving_evaluation: Dict[str, Any]
    ):
        """Generate evaluation visualizations"""
        
        # 1. Agent Performance Scores
        self._plot_agent_scores(agent_evaluation)
        
        # 2. System Metrics Dashboard
        self._plot_system_metrics(system_evaluation)
        
        # 3. Problem-Solving Effectiveness
        self._plot_problem_solving(problem_solving_evaluation)
        
        # 4. Agent Comparison
        self._plot_agent_comparison(agent_evaluation)
        
        # 5. Agent Execution Time Analysis
        self._plot_agent_execution_time(agent_evaluation)
        
        # 6. Methods Used by Each Agent
        self._plot_agent_methods(agent_evaluation, system_evaluation)
        
        # 7. Detailed Agent Evaluation Dashboard
        self._plot_detailed_agent_evaluation(agent_evaluation)
    
    def _plot_agent_scores(self, agent_evaluation: Dict[str, Any]):
        """Plot agent performance scores"""
        agents = list(agent_evaluation.keys())
        scores = [agent_evaluation[agent]["overall_score"] for agent in agents]
        grades = [agent_evaluation[agent]["grade"] for agent in agents]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(agents, scores, color=['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.6 else '#e74c3c' for s in scores])
        
        # Add grade labels
        for i, (bar, grade) in enumerate(zip(bars, grades)):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                   f'{scores[i]:.2f} ({grade})', ha='left', va='center',
                   fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Performance Score', fontsize=12, fontweight='bold')
        ax.set_title('Agent Performance Scores', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "agent_performance_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_system_metrics(self, system_evaluation: Dict[str, Any]):
        """Plot system-level metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Success rates
        metrics = ["Layer 1", "Layer 2", "Docker"]
        rates = [
            system_evaluation["layer1_success_rate"],
            system_evaluation["layer2_success_rate"],
            system_evaluation["docker_success_rate"]
        ]
        
        axes[0, 0].bar(metrics, rates, color=['#3498db', '#9b59b6', '#2ecc71'], alpha=0.7)
        axes[0, 0].set_ylabel('Success Rate', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Layer Success Rates', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, rate in enumerate(rates):
            axes[0, 0].text(i, rate + 0.05, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Model performance
        model_perf = system_evaluation["model_performance"]
        metrics_model = ["Accuracy", "Precision", "Recall", "F1"]
        values = [
            model_perf.get("accuracy", 0),
            model_perf.get("precision", 0),
            model_perf.get("recall", 0),
            model_perf.get("f1_score", 0)
        ]
        
        axes[0, 1].bar(metrics_model, values, color='#e74c3c', alpha=0.7)
        axes[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, val in enumerate(values):
            axes[0, 1].text(i, val + 0.05, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Problem resolution
        prob_res = system_evaluation["problem_resolution"]
        axes[1, 0].bar(['Detected', 'Resolved'], 
                      [prob_res["problems_detected"], prob_res["problems_resolved"]],
                      color=['#f39c12', '#2ecc71'], alpha=0.7)
        axes[1, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Problem Resolution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].text(1, prob_res["problems_resolved"] + 0.5,
                       f'{prob_res["resolution_rate"]:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # System score
        system_score = system_evaluation["system_score"]
        system_grade = system_evaluation["system_grade"]
        axes[1, 1].barh(['System'], [system_score], color='#3498db', alpha=0.7)
        axes[1, 1].set_xlabel('Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(f'Overall System Score: {system_score:.2f} ({system_grade})', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlim(0, 1.1)
        axes[1, 1].grid(axis='x', alpha=0.3)
        axes[1, 1].text(system_score, 0, f'{system_score:.2f}', ha='left', va='center', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "system_metrics_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_problem_solving(self, problem_solving_evaluation: Dict[str, Any]):
        """Plot problem-solving effectiveness"""
        problem_scores = problem_solving_evaluation.get("problem_solving_scores", {})
        
        if not problem_scores:
            return
        
        problems = list(problem_scores.keys())
        scores = [problem_scores[p]["score"] for p in problems]
        detected = [1 if problem_scores[p]["detected"] else 0 for p in problems]
        solved = [1 if problem_scores[p]["solution_applied"] else 0 for p in problems]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(problems))
        width = 0.25
        
        bars1 = ax.bar(x - width, detected, width, label='Detected', color='#3498db', alpha=0.7)
        bars2 = ax.bar(x, solved, width, label='Solved', color='#2ecc71', alpha=0.7)
        bars3 = ax.bar(x + width, scores, width, label='Score', color='#f39c12', alpha=0.7)
        
        ax.set_xlabel('Problem Type', fontsize=11, fontweight='bold')
        ax.set_ylabel('Status (1=Yes, 0=No) / Score', fontsize=11, fontweight='bold')
        ax.set_title('Problem-Solving Effectiveness', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(problems, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "problem_solving_effectiveness.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_comparison(self, agent_evaluation: Dict[str, Any]):
        """Plot multi-metric agent comparison"""
        agents = list(agent_evaluation.keys())
        
        metrics = {
            "Output Score": [agent_evaluation[a]["output_score"] for a in agents],
            "Criteria Score": [agent_evaluation[a]["criteria_score"] for a in agents],
            "Layer 1 Score": [agent_evaluation[a]["layer1_score"] for a in agents],
            "Layer 2 Score": [agent_evaluation[a]["layer2_score"] for a in agents]
        }
        
        x = np.arange(len(agents))
        width = 0.2
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#f39c12']
        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=metric_name, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Agent Performance Comparison (Multi-Metric)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "agent_comparison_multimetric.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_execution_time(self, agent_evaluation: Dict[str, Any]):
        """Plot execution time for each agent"""
        agents = list(agent_evaluation.keys())
        times = [agent_evaluation[agent]["execution_time"] for agent in agents]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        colors = ['#3498db' if t < 30 else '#f39c12' if t < 60 else '#e74c3c' for t in times]
        bars = ax1.barh(agents, times, color=colors, alpha=0.7)
        ax1.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Agent Execution Time', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add time labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{time:.2f}s', ha='left', va='center',
                    fontweight='bold', fontsize=10)
        
        # Pie chart showing time distribution
        ax2.pie(times, labels=agents, autopct='%1.1f%%', startangle=90,
               colors=colors, textprops={'fontsize': 9})
        ax2.set_title('Time Distribution Across Agents', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "agent_execution_time.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_methods(self, agent_evaluation: Dict[str, Any], system_evaluation: Dict[str, Any]):
        """Plot methods/techniques used by each agent"""
        agents = list(agent_evaluation.keys())
        
        # Extract methods from agent evaluation data
        methods_data = {}
        for agent_name, eval_data in agent_evaluation.items():
            methods = []
            
            # Layer 1 methods
            if eval_data.get("layer1_executed"):
                methods.append("Layer 1 (Hardcoded)")
            
            # Layer 2 methods
            if eval_data.get("layer2_executed"):
                methods.append("Layer 2 (LLM)")
            
            # Docker execution
            if eval_data.get("docker_success"):
                methods.append("Docker Sandbox")
            
            # Agent-specific methods based on outputs
            if agent_name == "Data Cleaning":
                if eval_data.get("state_keys_added", 0) > 0:
                    methods.extend(["Missing Value Handling", "Outlier Treatment", "Data Imputation"])
            elif agent_name == "Feature Engineering":
                if eval_data.get("state_keys_added", 0) > 0:
                    methods.extend(["Feature Creation", "Multicollinearity Removal", "Encoding"])
            elif agent_name == "ML Building":
                if eval_data.get("state_keys_added", 0) > 0:
                    methods.extend(["Model Training", "Hyperparameter Tuning", "Model Selection"])
            elif agent_name == "EDA Analysis":
                if eval_data.get("state_keys_added", 0) > 0:
                    methods.extend(["Statistical Analysis", "Visualization", "Problem Detection"])
            
            methods_data[agent_name] = methods
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for stacked bar chart
        all_methods = set()
        for methods in methods_data.values():
            all_methods.update(methods)
        all_methods = sorted(list(all_methods))
        
        method_counts = {method: [] for method in all_methods}
        for agent in agents:
            for method in all_methods:
                method_counts[method].append(1 if method in methods_data[agent] else 0)
        
        x = np.arange(len(agents))
        width = 0.8 / len(all_methods)
        colors_map = plt.cm.Set3(np.linspace(0, 1, len(all_methods)))
        
        bottom = np.zeros(len(agents))
        for i, method in enumerate(all_methods):
            ax.bar(x, method_counts[method], width, label=method, bottom=bottom, 
                  color=colors_map[i], alpha=0.8)
            bottom += np.array(method_counts[method])
        
        ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Methods Used', fontsize=12, fontweight='bold')
        ax.set_title('Methods and Techniques Used by Each Agent', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "agent_methods_used.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_agent_evaluation(self, agent_evaluation: Dict[str, Any]):
        """Create a detailed evaluation dashboard for all agents"""
        agents = list(agent_evaluation.keys())
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance Scores
        ax1 = fig.add_subplot(gs[0, 0])
        scores = [agent_evaluation[agent]["overall_score"] for agent in agents]
        grades = [agent_evaluation[agent]["grade"] for agent in agents]
        colors = ['#2ecc71' if g == 'A' else '#f39c12' if g == 'B' else '#e74c3c' for g in grades]
        bars = ax1.barh(agents, scores, color=colors, alpha=0.7)
        for i, (bar, score, grade) in enumerate(zip(bars, scores, grades)):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{score:.2f} ({grade})', ha='left', va='center', fontweight='bold', fontsize=9)
        ax1.set_xlabel('Score', fontsize=10, fontweight='bold')
        ax1.set_title('Overall Performance Scores', fontsize=11, fontweight='bold')
        ax1.set_xlim(0, 1.1)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Execution Time
        ax2 = fig.add_subplot(gs[0, 1])
        times = [agent_evaluation[agent]["execution_time"] for agent in agents]
        ax2.barh(agents, times, color='#3498db', alpha=0.7)
        for i, (agent, time) in enumerate(zip(agents, times)):
            ax2.text(time, i, f'{time:.1f}s', ha='left', va='center', fontweight='bold', fontsize=9)
        ax2.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax2.set_title('Execution Time', fontsize=11, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Layer Success Rates
        ax3 = fig.add_subplot(gs[0, 2])
        layer1_success = [1 if agent_evaluation[agent]["layer1_executed"] else 0 for agent in agents]
        layer2_success = [1 if agent_evaluation[agent]["layer2_executed"] else 0 for agent in agents]
        docker_success = [1 if agent_evaluation[agent]["docker_success"] else 0 for agent in agents]
        x = np.arange(len(agents))
        width = 0.25
        ax3.bar(x - width, layer1_success, width, label='Layer 1', color='#3498db', alpha=0.7)
        ax3.bar(x, layer2_success, width, label='Layer 2', color='#9b59b6', alpha=0.7)
        ax3.bar(x + width, docker_success, width, label='Docker', color='#2ecc71', alpha=0.7)
        ax3.set_ylabel('Success (1=Yes)', fontsize=10, fontweight='bold')
        ax3.set_title('Layer Execution Success', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=8)
        ax3.set_ylim(0, 1.2)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Output Quality Scores
        ax4 = fig.add_subplot(gs[1, 0])
        output_scores = [agent_evaluation[agent]["output_score"] for agent in agents]
        criteria_scores = [agent_evaluation[agent]["criteria_score"] for agent in agents]
        x = np.arange(len(agents))
        width = 0.35
        ax4.bar(x - width/2, output_scores, width, label='Output Score', color='#e74c3c', alpha=0.7)
        ax4.bar(x + width/2, criteria_scores, width, label='Criteria Score', color='#f39c12', alpha=0.7)
        ax4.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax4.set_title('Output & Criteria Scores', fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1.2)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. State Keys Added
        ax5 = fig.add_subplot(gs[1, 1])
        state_keys = [agent_evaluation[agent]["state_keys_added"] for agent in agents]
        ax5.barh(agents, state_keys, color='#16a085', alpha=0.7)
        for i, (agent, keys) in enumerate(zip(agents, state_keys)):
            ax5.text(keys, i, str(keys), ha='left', va='center', fontweight='bold', fontsize=9)
        ax5.set_xlabel('State Keys Added', fontsize=10, fontweight='bold')
        ax5.set_title('State Keys Added', fontsize=11, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. Layer 1 vs Layer 2 Scores
        ax6 = fig.add_subplot(gs[1, 2])
        layer1_scores = [agent_evaluation[agent]["layer1_score"] for agent in agents]
        layer2_scores = [agent_evaluation[agent]["layer2_score"] for agent in agents]
        x = np.arange(len(agents))
        width = 0.35
        ax6.bar(x - width/2, layer1_scores, width, label='Layer 1', color='#3498db', alpha=0.7)
        ax6.bar(x + width/2, layer2_scores, width, label='Layer 2', color='#9b59b6', alpha=0.7)
        ax6.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax6.set_title('Layer 1 vs Layer 2 Scores', fontsize=11, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax6.legend(fontsize=8)
        ax6.set_ylim(0, 1.2)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Performance Heatmap
        ax7 = fig.add_subplot(gs[2, :])
        metrics_data = {
            'Overall Score': scores,
            'Output Score': output_scores,
            'Criteria Score': criteria_scores,
            'Layer 1 Score': layer1_scores,
            'Layer 2 Score': layer2_scores,
            'Execution Time (norm)': [(max(times) - t) / max(times) if max(times) > 0 else 0 for t in times],  # Normalized (inverted)
            'State Keys': [s / max(state_keys) if max(state_keys) > 0 else 0 for s in state_keys]  # Normalized
        }
        heatmap_data = pd.DataFrame(metrics_data, index=agents).T
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax7, 
                   cbar_kws={'label': 'Normalized Score'}, vmin=0, vmax=1)
        ax7.set_title('Agent Performance Heatmap (All Metrics)', fontsize=12, fontweight='bold', pad=15)
        ax7.set_xlabel('Agent', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Metric', fontsize=11, fontweight='bold')
        
        plt.suptitle('Comprehensive Agent Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(self.output_dir / "detailed_agent_evaluation_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_evaluation_tables(
        self,
        agent_evaluation: Dict[str, Any],
        system_evaluation: Dict[str, Any],
        problem_solving_evaluation: Dict[str, Any]
    ):
        """Generate evaluation tables"""
        
        # Agent evaluation table
        agent_table_data = []
        for agent_name, eval_data in agent_evaluation.items():
            agent_table_data.append({
                "Agent": agent_name,
                "Overall Score": f"{eval_data['overall_score']:.3f}",
                "Grade": eval_data['grade'],
                "Output Score": f"{eval_data['output_score']:.3f}",
                "Criteria Score": f"{eval_data['criteria_score']:.3f}",
                "Layer 1": "✓" if eval_data['layer1_executed'] else "✗",
                "Layer 2": "✓" if eval_data['layer2_executed'] else "✗",
                "Docker": "✓" if eval_data['docker_success'] else "✗",
                "Time (s)": f"{eval_data['execution_time']:.2f}",
                "State Keys": eval_data['state_keys_added']
            })
        
        df_agent = pd.DataFrame(agent_table_data)
        
        # System evaluation table
        system_table_data = [{
            "Metric": "Completion Rate",
            "Value": f"{system_evaluation['completion_rate']:.1%}"
        }, {
            "Metric": "Layer 1 Success Rate",
            "Value": f"{system_evaluation['layer1_success_rate']:.1%}"
        }, {
            "Metric": "Layer 2 Success Rate",
            "Value": f"{system_evaluation['layer2_success_rate']:.1%}"
        }, {
            "Metric": "Docker Success Rate",
            "Value": f"{system_evaluation['docker_success_rate']:.1%}"
        }, {
            "Metric": "Total Execution Time",
            "Value": f"{system_evaluation['total_execution_time']:.2f}s"
        }, {
            "Metric": "Problem Resolution Rate",
            "Value": f"{system_evaluation['problem_resolution']['resolution_rate']:.1%}"
        }, {
            "Metric": "System Score",
            "Value": f"{system_evaluation['system_score']:.3f} ({system_evaluation['system_grade']})"
        }]
        
        df_system = pd.DataFrame(system_table_data)
        
        # Save tables
        with open(self.output_dir / "agent_evaluation_table.md", 'w') as f:
            f.write("# Agent Evaluation Table\n\n")
            f.write(df_agent.to_markdown(index=False))
        
        with open(self.output_dir / "system_evaluation_table.md", 'w') as f:
            f.write("# System Evaluation Table\n\n")
            f.write(df_system.to_markdown(index=False))
    
    def _generate_markdown_report(
        self,
        agent_evaluation: Dict[str, Any],
        system_evaluation: Dict[str, Any],
        problem_solving_evaluation: Dict[str, Any]
    ) -> Path:
        """Generate comprehensive markdown evaluation report"""
        
        report_lines = []
        report_lines.append("# Comprehensive Evaluation Report")
        report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report_lines.append("## Executive Summary\n")
        report_lines.append(f"- **System Score:** {system_evaluation['system_score']:.3f} ({system_evaluation['system_grade']})")
        report_lines.append(f"- **Completion Rate:** {system_evaluation['completion_rate']:.1%}")
        report_lines.append(f"- **Problem Resolution Rate:** {system_evaluation['problem_resolution']['resolution_rate']:.1%}")
        report_lines.append(f"- **Model Performance (F1):** {system_evaluation['model_performance'].get('f1_score', 0):.3f}\n")
        
        # Agent Evaluation
        report_lines.append("## Agent-Wise Evaluation\n")
        report_lines.append("| Agent | Score | Grade | Layer 1 | Layer 2 | Docker | Time (s) |")
        report_lines.append("|-------|-------|-------|---------|---------|--------|----------|")
        for agent_name, eval_data in agent_evaluation.items():
            report_lines.append(
                f"| {agent_name} | {eval_data['overall_score']:.3f} | {eval_data['grade']} | "
                f"{'✓' if eval_data['layer1_executed'] else '✗'} | "
                f"{'✓' if eval_data['layer2_executed'] else '✗'} | "
                f"{'✓' if eval_data['docker_success'] else '✗'} | "
                f"{eval_data['execution_time']:.2f} |"
            )
        report_lines.append("")
        
        # System Evaluation
        report_lines.append("## System-Level Evaluation\n")
        report_lines.append(f"- **Layer 1 Success Rate:** {system_evaluation['layer1_success_rate']:.1%}")
        report_lines.append(f"- **Layer 2 Success Rate:** {system_evaluation['layer2_success_rate']:.1%}")
        report_lines.append(f"- **Docker Success Rate:** {system_evaluation['docker_success_rate']:.1%}")
        report_lines.append(f"- **Total Execution Time:** {system_evaluation['total_execution_time']:.2f}s")
        report_lines.append(f"- **Average Time per Agent:** {system_evaluation['avg_time_per_agent']:.2f}s\n")
        
        # Model Performance
        report_lines.append("### Model Performance\n")
        model_perf = system_evaluation['model_performance']
        report_lines.append(f"- **Accuracy:** {model_perf.get('accuracy', 0):.3f}")
        report_lines.append(f"- **Precision:** {model_perf.get('precision', 0):.3f}")
        report_lines.append(f"- **Recall:** {model_perf.get('recall', 0):.3f}")
        report_lines.append(f"- **F1 Score:** {model_perf.get('f1_score', 0):.3f}")
        if model_perf.get('roc_auc'):
            report_lines.append(f"- **ROC-AUC:** {model_perf['roc_auc']:.3f}")
        report_lines.append("")
        
        # Problem-Solving Evaluation
        report_lines.append("## Problem-Solving Evaluation\n")
        problem_scores = problem_solving_evaluation.get('problem_solving_scores', {})
        report_lines.append(f"- **Overall Score:** {problem_solving_evaluation['overall_score']:.3f} ({problem_solving_evaluation['grade']})\n")
        report_lines.append("| Problem Type | Detected | Solved | Score |")
        report_lines.append("|--------------|----------|--------|-------|")
        for problem_type, ps_data in problem_scores.items():
            report_lines.append(
                f"| {problem_type} | {'✓' if ps_data['detected'] else '✗'} | "
                f"{'✓' if ps_data['solution_applied'] else '✗'} | {ps_data['score']:.3f} |"
            )
        report_lines.append("")
        
        # Save report
        report_path = self.output_dir / "comprehensive_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        return report_path

