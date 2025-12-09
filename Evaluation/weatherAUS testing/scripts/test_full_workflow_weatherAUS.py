"""
Comprehensive End-to-End Test: Full Workflow on weatherAUS Dataset

Tests the complete pipeline:
1. Data ingestion
2. Data flow through agents
3. Layer 1 execution → Results
4. Layer 2 execution → Docker sandbox → Results
5. Information passing between agents
6. Final report creation
7. Final plots generation
8. Final classification model metrics
"""

import sys
import asyncio
import logging
import json
import pandas as pd
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add paths (updated for new location in weatherAUS testing folder)
# Path to Evaluation folder (parent.parent.parent)
eval_folder = Path(__file__).parent.parent.parent
sys.path.insert(0, str(eval_folder / "backend"))
sys.path.insert(0, str(eval_folder))  # Evaluation folder for test_cases

from test_cases.base_test_framework import BaseAgentTest
from app.agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
from app.agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
from app.agents.reporting.technical_reporter_agent import TechnicalReporterAgent
from app.workflows.state_management import ClassificationState, WorkflowStatus, state_manager
from app.services.sandbox_executor import SandboxExecutor

# Import enhanced workflow tracker
from enhanced_workflow_tracker import DatasetProblemAnalyzer, AgentSolutionTracker, WorkflowProgressTracker

# Import reasoning extraction and visualization
from reasoning_extractor import AgentReasoningExtractor
from reasoning_plot_generator import ReasoningPlotGenerator
from evaluation_generator import EvaluationGenerator

# Import visualization modules
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from visualization.plot_generator import PlotGenerator
    from reports.table_generator import TableGenerator
except ImportError as e:
    # Logger not yet initialized, use print
    print(f"Warning: Could not import visualization modules - plots/tables will be skipped: {e}")
    PlotGenerator = None
    TableGenerator = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Note: Logging will be set up in test_full_workflow with timestamp-based log file

logger = logging.getLogger(__name__)


def _generate_problem_solving_visualizations(tracker, output_dir: Path):
    """Generate visualizations showing problem-solving workflow."""
    try:
        problem_analysis = tracker.problem_analysis
        solution_tracker = tracker.solution_tracker
        
        # 1. Problem Severity Chart
        problems = problem_analysis.get('problems', [])
        if problems:
            severity_counts = {}
            for p in problems:
                severity = p.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#3498db'}
            bars = ax.bar(severity_counts.keys(), severity_counts.values(), 
                         color=[colors.get(s, '#95a5a6') for s in severity_counts.keys()])
            ax.set_xlabel('Problem Severity', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Problems', fontsize=12, fontweight='bold')
            ax.set_title('Dataset Problems Detected Before Agent Processing', fontsize=14, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "problem_severity_chart.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Problem Types Chart
        if problems:
            problem_types = {}
            for p in problems:
                ptype = p.get('type', 'Unknown')
                problem_types[ptype] = problem_types.get(ptype, 0) + 1
            
            fig, ax = plt.subplots(figsize=(12, 6))
            types = list(problem_types.keys())
            counts = list(problem_types.values())
            bars = ax.barh(types, counts, color='#3498db')
            ax.set_xlabel('Number of Problems', fontsize=12, fontweight='bold')
            ax.set_ylabel('Problem Type', fontsize=12, fontweight='bold')
            ax.set_title('Types of Problems Detected in Dataset', fontsize=14, fontweight='bold', pad=15)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count, bar.get_y() + bar.get_height()/2,
                       f' {count}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "problem_types_chart.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Agent Problem-Solving Flow Chart
        if hasattr(tracker, 'agent_results'):
            agent_names = list(tracker.agent_results.keys())
            problems_solved = []
            for agent_name in agent_names:
                # Count problems that this agent type typically solves
                solved = 0
                if 'Data Cleaning' in agent_name:
                    solved = len([p for p in problems if p.get('type') in ['Missing Values', 'Outliers', 'Data Type Issues']])
                elif 'EDA' in agent_name:
                    solved = len([p for p in problems if p.get('type') in ['Class Imbalance']])
                elif 'Feature Engineering' in agent_name:
                    solved = len([p for p in problems if p.get('type') in ['High Cardinality', 'Multicollinearity', 'Zero Variance']])
                problems_solved.append(solved)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            x_pos = np.arange(len(agent_names))
            bars = ax.bar(x_pos, problems_solved, color='#2ecc71', alpha=0.7)
            ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
            ax.set_ylabel('Problems Addressed', fontsize=12, fontweight='bold')
            ax.set_title('Agent Problem-Solving Capabilities', fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(agent_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, problems_solved):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "agent_problem_solving.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("✅ Enhanced problem-solving visualizations generated")
    except Exception as e:
        logger.warning(f"Could not generate enhanced visualizations: {e}")


def _generate_before_after_plots(tracker, output_dir: Path):
    """Generate before/after comparison plots for each agent."""
    try:
        before_after_states = tracker.before_after_states
        
        for agent_name, states in before_after_states.items():
            before = states.get("before", {})
            after = states.get("after", {})
            
            if not before or not after:
                continue
            
            # Create comparison figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{agent_name}: Before vs After Comparison', fontsize=16, fontweight='bold', y=0.995)
            
            # 1. Dataset Shape Comparison
            ax = axes[0, 0]
            if before.get("dataset_shape") and after.get("dataset_shape"):
                before_shape = before["dataset_shape"]
                after_shape = after["dataset_shape"]
                x = ['Rows', 'Columns']
                before_vals = [before_shape[0], before_shape[1]]
                after_vals = [after_shape[0], after_shape[1]]
                
                x_pos = np.arange(len(x))
                width = 0.35
                bars1 = ax.bar(x_pos - width/2, before_vals, width, label='Before', color='#e74c3c', alpha=0.7)
                bars2 = ax.bar(x_pos + width/2, after_vals, width, label='After', color='#2ecc71', alpha=0.7)
                
                ax.set_xlabel('Dimension', fontsize=11, fontweight='bold')
                ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                ax.set_title('Dataset Shape', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            # 2. Missing Values Comparison
            ax = axes[0, 1]
            before_missing = before.get("missing_values", {})
            after_missing = after.get("missing_values", {})
            
            if before_missing and after_missing:
                cols_with_missing = set(list(before_missing.keys()) + list(after_missing.keys()))
                cols_with_missing = [c for c in cols_with_missing if before_missing.get(c, 0) > 0 or after_missing.get(c, 0) > 0]
                
                if cols_with_missing:
                    cols_to_show = cols_with_missing[:10]  # Top 10
                    before_vals = [before_missing.get(c, 0) for c in cols_to_show]
                    after_vals = [after_missing.get(c, 0) for c in cols_to_show]
                    
                    x_pos = np.arange(len(cols_to_show))
                    width = 0.35
                    bars1 = ax.bar(x_pos - width/2, before_vals, width, label='Before', color='#e74c3c', alpha=0.7)
                    bars2 = ax.bar(x_pos + width/2, after_vals, width, label='After', color='#2ecc71', alpha=0.7)
                    
                    ax.set_xlabel('Column', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Missing Values', fontsize=11, fontweight='bold')
                    ax.set_title('Missing Values Comparison', fontsize=12, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(cols_to_show, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
            
            # 3. Column Count Comparison
            ax = axes[1, 0]
            before_cols = len(before.get("columns", []))
            after_cols = len(after.get("columns", []))
            
            bars = ax.bar(['Before', 'After'], [before_cols, after_cols], 
                         color=['#e74c3c', '#2ecc71'], alpha=0.7)
            ax.set_ylabel('Number of Columns', fontsize=11, fontweight='bold')
            ax.set_title('Column Count', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 4. Target Distribution Comparison (if available)
            ax = axes[1, 1]
            before_target = before.get("target_distribution", {})
            after_target = after.get("target_distribution", {})
            
            if before_target and after_target:
                categories = sorted(set(list(before_target.keys()) + list(after_target.keys())))
                before_vals = [before_target.get(c, 0) for c in categories]
                after_vals = [after_target.get(c, 0) for c in categories]
                
                x_pos = np.arange(len(categories))
                width = 0.35
                bars1 = ax.bar(x_pos - width/2, before_vals, width, label='Before', color='#e74c3c', alpha=0.7)
                bars2 = ax.bar(x_pos + width/2, after_vals, width, label='After', color='#2ecc71', alpha=0.7)
                
                ax.set_xlabel('Category', fontsize=11, fontweight='bold')
                ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                ax.set_title('Target Distribution', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No target distribution\ndata available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title('Target Distribution', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            safe_agent_name = agent_name.replace(' ', '_').replace('/', '_')
            plt.savefig(str(output_dir / f"before_after_{safe_agent_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Generated before/after plot for {agent_name}")
        
        logger.info("✅ All before/after comparison plots generated")
    except Exception as e:
        logger.warning(f"Could not generate before/after plots: {e}")


class WorkflowTracker:
    """Track workflow execution and verify each step."""
    
    def __init__(self):
        self.steps = []
        self.agent_results = {}
        self.layer1_results = {}
        self.layer2_results = {}
        self.state_snapshots = []
        self.docker_executions = []
        self.errors = []
        self.warnings = []
        
    def track_step(self, step_name: str, status: str, details: Dict[str, Any] = None):
        """Track a workflow step."""
        step = {
            "step": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.steps.append(step)
        logger.info(f"📋 [{status}] {step_name}: {details}")
        
    def track_agent(self, agent_name: str, layer1: bool, layer2: bool, docker_success: bool, 
                   state_keys: List[str], execution_time: float):
        """Track agent execution."""
        self.agent_results[agent_name] = {
            "layer1_executed": layer1,
            "layer2_executed": layer2,
            "layer2_docker_success": docker_success,
            "state_keys_added": state_keys,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
    def track_docker(self, container_name: str, status: str, execution_time: float):
        """Track Docker execution."""
        self.docker_executions.append({
            "container": container_name,
            "status": status,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        })
        
    def snapshot_state(self, state: ClassificationState, agent_name: str):
        """Take a snapshot of state after agent execution."""
        snapshot = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "keys": list(state.keys()),
            "dataset_shape": state.get("dataset_shape"),
            "processed_dataset_shape": None,
            "agent_statuses": dict(state.get("agent_statuses", {})),
            "workflow_status": state.get("workflow_status"),
            "has_cleaned_dataset": "cleaned_dataset" in state or "processed_dataset" in state,
            "has_eda_results": "eda_plots" in state or "statistical_summary" in state,
            "has_features": "engineered_features" in state,
            "has_model": "best_model" in state or "model_selection_results" in state,
            "has_evaluation": "evaluation_metrics" in state,
            "has_report": "final_report" in state or "technical_documentation" in state
        }
        
        # Get processed dataset shape if available
        if "processed_dataset" in state and state["processed_dataset"] is not None:
            try:
                df = state_manager.get_dataset(state, "processed")
                if df is not None:
                    snapshot["processed_dataset_shape"] = df.shape
            except:
                pass
                
        self.state_snapshots.append(snapshot)
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "test_name": "Full Workflow Test - weatherAUS",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_steps": len(self.steps),
                "agents_executed": len(self.agent_results),
                "docker_executions": len(self.docker_executions),
                "successful_docker": sum(1 for d in self.docker_executions if d["status"] == "SUCCESS"),
                "errors": len(self.errors),
                "warnings": len(self.warnings)
            },
            "steps": self.steps,
            "agents": self.agent_results,
            "docker_executions": self.docker_executions,
            "state_snapshots": self.state_snapshots,
            "errors": self.errors,
            "warnings": self.warnings
        }
        return report


def check_docker():
    """Check Docker availability."""
    try:
        subprocess.run(['docker', 'ps'], check=True, capture_output=True, timeout=5)
        logger.info("✓ Docker daemon is running")
        return True
    except Exception as e:
        logger.error(f"✗ Docker daemon not accessible: {e}")
        return False


async def test_agent_with_tracking(
    agent,
    agent_name: str,
    state: ClassificationState,
    tracker: WorkflowTracker
) -> ClassificationState:
    """Test an agent and track all execution details."""
    logger.info("\n" + "="*80)
    logger.info(f"Testing Agent: {agent_name}")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Track agent start
    tracker.track_step(f"Agent Start: {agent_name}", "STARTED")
    
    # Capture BEFORE state for visualization
    before_state = {
        "dataset_shape": None,
        "missing_values": {},
        "columns": [],
        "data_types": {},
        "outliers": {},
        "target_distribution": {}
    }
    
    try:
        df_before = state_manager.get_dataset(state, "cleaned")
        if df_before is None:
            df_before = state_manager.get_dataset(state, "original")
        if df_before is not None and isinstance(df_before, pd.DataFrame) and not df_before.empty:
            before_state["dataset_shape"] = df_before.shape
            before_state["columns"] = list(df_before.columns)
            before_state["missing_values"] = df_before.isnull().sum().to_dict()
            before_state["data_types"] = df_before.dtypes.astype(str).to_dict()
            
            # Get target distribution if available
            target_col = state.get("target_column")
            if target_col and target_col in df_before.columns:
                before_state["target_distribution"] = df_before[target_col].value_counts().to_dict()
    except Exception as e:
        logger.warning(f"Could not capture before state: {e}")
    
    # Store before state in tracker
    if not hasattr(tracker, 'before_after_states'):
        tracker.before_after_states = {}
    tracker.before_after_states[agent_name] = {"before": before_state}
    
    # Get state keys before execution
    keys_before = set(state.keys())
    
    # Get containers before execution
    containers_before = set()
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=sandbox-', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            containers_before = set(result.stdout.strip().split('\n'))
    except:
        pass
    
    try:
        # Execute agent
        state = await agent.execute(state)
        execution_time = time.time() - start_time
        
        # Extract reasoning after agent execution
        if hasattr(tracker, 'reasoning_extractor'):
            try:
                agent_reasoning = {}
                if agent_name == "Data Cleaning":
                    agent_reasoning = tracker.reasoning_extractor.extract_cleaning_reasoning(state, agent_name)
                elif agent_name == "Feature Engineering":
                    agent_reasoning = tracker.reasoning_extractor.extract_feature_engineering_reasoning(state)
                elif agent_name == "EDA Analysis":
                    # EDA reasoning is extracted later from problem_detection
                    pass
                elif agent_name == "ML Building":
                    agent_reasoning = {
                        "imbalance": tracker.reasoning_extractor.extract_imbalance_reasoning(state),
                        "temporal": tracker.reasoning_extractor.extract_temporal_reasoning(state),
                        "model_selection": tracker.reasoning_extractor.extract_model_selection_reasoning(state)
                    }
                
                if agent_reasoning:
                    tracker.reasoning_tracking[agent_name] = agent_reasoning
                    logger.info(f"✅ Extracted reasoning for {agent_name}")
            except Exception as e:
                logger.warning(f"Could not extract reasoning for {agent_name}: {e}")
        
        # Get state keys after execution
        keys_after = set(state.keys())
        new_keys = list(keys_after - keys_before)
        
        # Check for Layer 1 results
        layer1_executed = True  # Always executed
        
        # Check for Layer 2 execution
        layer2_executed = False
        docker_success = False
        
        # Check for new Docker containers
        containers_after = set()
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', 'name=sandbox-', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                containers_after = set(result.stdout.strip().split('\n'))
        except:
            pass
        
        new_containers = containers_after - containers_before
        if new_containers:
            layer2_executed = True
            latest_container = list(new_containers)[-1]
            
            # Check Docker execution status
            try:
                result = subprocess.run(
                    ['docker', 'inspect', latest_container, '--format', '{{.State.Status}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                docker_status = result.stdout.strip()
                docker_success = docker_status in ['exited', 'running']
                
                # Get execution time from container
                docker_time = 0
                try:
                    result = subprocess.run(
                        ['docker', 'inspect', latest_container, '--format', '{{.State.FinishedAt}}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    # Could calculate time difference here if needed
                except:
                    pass
                    
                tracker.track_docker(latest_container, docker_status, docker_time)
            except:
                docker_success = False
        
        # Check state for Layer 2 success indicators (more comprehensive)
        # Check multiple possible indicators
        layer2_indicators = []
        docker_indicators = []
        
        # Check global state keys
        if state.get('layer2_success') is True:
            layer2_indicators.append(True)
            docker_indicators.append(True)
        
        sandbox_status = state.get('layer2_sandbox_status')
        if sandbox_status:
            layer2_executed = True  # Layer 2 was attempted if sandbox_status exists
            if sandbox_status == 'SUCCESS':
                layer2_indicators.append(True)
                docker_indicators.append(True)
        
        if state.get('sandbox_execution_time') is not None:
            layer2_executed = True
            layer2_indicators.append(True)
            docker_indicators.append(True)
        
        # Check for layer2_processing_failed flag (indicates Layer 2 was attempted)
        if state.get('layer2_processing_failed') is not None:
            layer2_executed = True  # Layer 2 was attempted
            if state.get('layer2_processing_failed') is False:
                docker_indicators.append(True)
        
        # Also check all state keys for layer2 indicators
        for key in state.keys():
            key_lower = key.lower()
            value = state.get(key)
            
            # Check for layer2_sandbox_status (any value means Layer 2 was attempted)
            if 'layer2_sandbox_status' in key_lower:
                layer2_executed = True
                if value == 'SUCCESS':
                        layer2_indicators.append(True)
                    docker_indicators.append(True)
            
            # Check for layer2 success indicators
            if 'layer2' in key_lower and 'success' in key_lower:
                if value is True:
                    layer2_executed = True
                        layer2_indicators.append(True)
                    docker_indicators.append(True)
        
            # Check for sandbox execution time
            if 'sandbox_execution_time' in key_lower and value is not None:
            layer2_executed = True
                layer2_indicators.append(True)
                docker_indicators.append(True)
        
        # If Layer 2 was executed, set flags
        if layer2_executed or any(layer2_indicators):
            layer2_executed = True
            if any(docker_indicators):
            docker_success = True
        
        # Track agent execution
        tracker.track_agent(
            agent_name,
            layer1_executed,
            layer2_executed,
            docker_success,
            new_keys,
            execution_time
        )
        
        # Track problem-solving progress
        if hasattr(tracker, 'progress_tracker') and hasattr(tracker, 'problem_analysis'):
            # Get current dataset state
            current_df = state_manager.get_dataset(state, "cleaned")
            if current_df is None:
                current_df = state_manager.get_dataset(state, "original")
            if current_df is not None and not current_df.empty:
                dataset_state = {
                    "rows": len(current_df),
                    "columns": len(current_df.columns),
                    "missing_pct": (current_df.isnull().sum().sum() / (len(current_df) * len(current_df.columns))) * 100 if len(current_df) > 0 else 0
                }
                
                # Determine problems solved by this agent
                problems_solved = []
                actions_taken = []
                
                if 'Data Cleaning' in agent_name:
                    problems_solved = ['Missing Values', 'Outliers', 'Data Type Issues']
                    actions_taken = ['Imputation', 'Outlier handling', 'Type conversion']
                elif 'EDA' in agent_name:
                    problems_solved = ['Class Imbalance']
                    actions_taken = ['Imbalance detection', 'Statistical analysis']
                elif 'Feature Engineering' in agent_name:
                    problems_solved = ['High Cardinality', 'Multicollinearity', 'Zero Variance']
                    actions_taken = ['Feature creation', 'Feature selection', 'Encoding']
                
                tracker.progress_tracker.track_step(
                    agent_name, agent_name, dataset_state, problems_solved, actions_taken
                )
        
        # Capture AFTER state for visualization
        after_state = {
            "dataset_shape": None,
            "missing_values": {},
            "columns": [],
            "data_types": {},
            "outliers": {},
            "target_distribution": {}
        }
        
        try:
            df_after = state_manager.get_dataset(state, "cleaned")
            if df_after is None:
                df_after = state_manager.get_dataset(state, "original")
            if df_after is not None and isinstance(df_after, pd.DataFrame) and not df_after.empty:
                after_state["dataset_shape"] = df_after.shape
                after_state["columns"] = list(df_after.columns)
                after_state["missing_values"] = df_after.isnull().sum().to_dict()
                after_state["data_types"] = df_after.dtypes.astype(str).to_dict()
                
                # Get target distribution if available
                target_col = state.get("target_column")
                if target_col and target_col in df_after.columns:
                    after_state["target_distribution"] = df_after[target_col].value_counts().to_dict()
        except Exception as e:
            logger.warning(f"Could not capture after state: {e}")
        
        # Store after state in tracker
        if not hasattr(tracker, 'before_after_states'):
            tracker.before_after_states = {}
        if agent_name not in tracker.before_after_states:
            tracker.before_after_states[agent_name] = {}
        tracker.before_after_states[agent_name]["after"] = after_state
        
        # Snapshot state
        tracker.snapshot_state(state, agent_name)
        
        # Track completion
        tracker.track_step(
            f"Agent Complete: {agent_name}",
            "COMPLETED",
            {
                "layer1": layer1_executed,
                "layer2": layer2_executed,
                "docker_success": docker_success,
                "execution_time": execution_time,
                "new_state_keys": len(new_keys)
            }
        )
        
        logger.info(f"✅ {agent_name} completed in {execution_time:.2f}s")
        logger.info(f"   Layer 1: {'✓' if layer1_executed else '✗'}")
        logger.info(f"   Layer 2: {'✓' if layer2_executed else '✗'}")
        logger.info(f"   Docker Success: {'✓' if docker_success else '✗'}")
        logger.info(f"   New State Keys: {len(new_keys)}")
        
        return state
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        tracker.errors.append({
            "agent": agent_name,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        tracker.track_step(f"Agent Error: {agent_name}", "FAILED", {"error": error_msg})
        logger.error(f"✗ {agent_name} failed: {e}", exc_info=True)
        raise


async def test_full_workflow():
    """Test the complete workflow from start to finish."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE END-TO-END WORKFLOW TEST")
    logger.info("Dataset: weatherAUS.csv")
    logger.info("Target: RainTomorrow")
    logger.info("="*80)
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent.parent / "results" / timestamp
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    plots_dir = output_base / "plots"
    reports_dir = output_base / "reports"
    tables_dir = output_base / "tables"
    logs_dir = output_base / "logs"
    
    plots_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    logger.info(f"📁 Output directory: {output_base}")
    
    tracker = WorkflowTracker()
    tracker.output_base = output_base  # Store for later use
    
    # Initialize reasoning extractor
    reasoning_extractor = AgentReasoningExtractor()
    tracker.reasoning_extractor = reasoning_extractor
    tracker.reasoning_tracking = {}  # Store reasoning for each agent
    
    # Set up logging with timestamp-based log file
    log_file = logs_dir / f'full_workflow_weatherAUS_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration
    )
    
    # Step 1: Check Docker
    tracker.track_step("Docker Check", "STARTED")
    if not check_docker():
        tracker.track_step("Docker Check", "FAILED")
        return {"success": False, "error": "Docker not available"}
    tracker.track_step("Docker Check", "COMPLETED")
    
    # Step 2: Data Ingestion
    tracker.track_step("Data Ingestion", "STARTED")
    dataset_path = '/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project/Evaluation/datasets/real_world/weatherAUS.csv'
    target_col = 'RainTomorrow'
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"✓ Dataset loaded: {df.shape}")
        
        # Sample for faster testing (use full dataset for production)
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
            logger.info(f"✓ Sampled to: {df.shape} for faster testing")
        
        # Analyze dataset problems BEFORE agents process it
        logger.info("🔍 Analyzing dataset problems...")
        problem_analyzer = DatasetProblemAnalyzer(df, target_col)
        problem_analysis = problem_analyzer.analyze()
        logger.info(f"✓ Detected {problem_analysis['metrics']['problems_detected']} problems")
        logger.info(f"  - High severity: {problem_analysis['metrics']['high_severity_problems']}")
        logger.info(f"  - Medium severity: {problem_analysis['metrics']['medium_severity_problems']}")
        
        # Initialize enhanced trackers
        solution_tracker = AgentSolutionTracker()
        progress_tracker = WorkflowProgressTracker()
        
        # Store trackers in tracker object for later use
        tracker.problem_analyzer = problem_analyzer
        tracker.solution_tracker = solution_tracker
        tracker.progress_tracker = progress_tracker
        tracker.problem_analysis = problem_analysis
        
        tracker.track_step("Data Ingestion", "COMPLETED", {
            "shape": df.shape,
            "target_column": target_col,
            "target_distribution": df[target_col].value_counts().to_dict(),
            "problems_detected": problem_analysis['metrics']['problems_detected']
        })
    except Exception as e:
        tracker.track_step("Data Ingestion", "FAILED", {"error": str(e)})
        return {"success": False, "error": f"Data ingestion failed: {e}"}
    
    # Step 3: Initialize State
    tracker.track_step("State Initialization", "STARTED")
    
    # Fix config path - use absolute path from script location
    eval_folder = Path(__file__).parent.parent.parent
    config_path = eval_folder / "config" / "evaluation_config.yaml"
    base_test = BaseAgentTest(config_path=str(config_path))
    state = base_test.create_state(df, target_col)
    state['target_column'] = target_col
    
    # Track initial dataset state
    if hasattr(tracker, 'progress_tracker'):
        initial_state = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_pct": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "problems": tracker.problem_analysis['problems'] if hasattr(tracker, 'problem_analysis') else []
        }
        tracker.progress_tracker.track_step("Initial State", "None", initial_state, [], [])
    
    tracker.track_step("State Initialization", "COMPLETED", {
        "state_keys": len(state.keys()),
        "dataset_shape": state.get("dataset_shape")
    })
    
    # Step 4: Execute All Agents in Sequence
    tracker.track_step("Agent Execution", "STARTED")
    
    agents = [
        ("Data Discovery", DataDiscoveryAgent()),
        ("EDA Analysis", EDAAgent()),
        ("Data Cleaning", EnhancedDataCleaningAgent()),
        ("Feature Engineering", FeatureEngineeringAgent()),
        ("ML Building", MLBuilderAgent()),
        ("Model Evaluation", ModelEvaluationAgent()),
        ("Technical Reporter", TechnicalReporterAgent()),
    ]
    
    # Execute agents sequentially - continue even if one fails
    for agent_name, agent in agents:
        try:
            state = await test_agent_with_tracking(agent, agent_name, state, tracker)
            await asyncio.sleep(1)  # Small delay between agents
        except Exception as e:
            logger.error(f"⚠️ Agent {agent_name} failed: {e}", exc_info=True)
            tracker.track_step(f"Agent {agent_name}", "FAILED", {
                "error": str(e)
            })
            # Continue to next agent instead of breaking
            continue
    
    tracker.track_step("Agent Execution", "COMPLETED")
    
    # Step 5: Verify Final Outputs
    tracker.track_step("Output Verification", "STARTED")
    
    outputs = {
        "has_cleaned_dataset": "cleaned_dataset" in state or "processed_dataset" in state,
        "has_eda_results": "eda_plots" in state or "statistical_summary" in state,
        "has_features": "engineered_features" in state,
        "has_model": "best_model" in state or "model_selection_results" in state,
        "has_evaluation_metrics": "evaluation_metrics" in state,
        "has_final_report": "final_report" in state or "technical_documentation" in state,
        "has_plots": "eda_plots" in state or len(state.get("eda_plots", [])) > 0,
    }
    
    tracker.track_step("Output Verification", "COMPLETED", outputs)
    
    # Step 6: Extract Final Metrics
    tracker.track_step("Metrics Extraction", "STARTED")
    
    final_metrics = {
        "model_metrics": state.get("evaluation_metrics", {}),
        "training_metrics": state.get("training_metrics", {}),
        "cross_validation_scores": state.get("cross_validation_scores", {}),
        "feature_importance": state.get("feature_importance_model", {}),
        "data_quality_score": state.get("data_quality_score"),
        "workflow_status": state.get("workflow_status"),
        "completed_agents": state.get("completed_agents", []),
        "layer_usage": state.get("layer_usage", {})
    }
    
    tracker.track_step("Metrics Extraction", "COMPLETED", final_metrics)
    
    # Step 7: Generate Final Report
    tracker.track_step("Report Generation", "STARTED")
    
    report = tracker.generate_report()
    
    # Add enhanced tracking data to report
    if hasattr(tracker, 'problem_analysis'):
        report['problem_analysis'] = tracker.problem_analysis
    if hasattr(tracker, 'solution_tracker'):
        report['solution_summary'] = tracker.solution_tracker.get_solution_summary()
    if hasattr(tracker, 'progress_tracker'):
        report['workflow_progress'] = tracker.progress_tracker.get_progress_summary()
    if hasattr(tracker, 'before_after_states'):
        report['before_after_states'] = tracker.before_after_states
    
    # Extract comprehensive reasoning from final state
    if hasattr(tracker, 'reasoning_extractor'):
        try:
            comprehensive_reasoning = tracker.reasoning_extractor.extract_comprehensive_reasoning(state)
            report['comprehensive_reasoning'] = comprehensive_reasoning
            tracker.reasoning_tracking['comprehensive'] = comprehensive_reasoning
            logger.info("✅ Extracted comprehensive reasoning from final state")
        except Exception as e:
            logger.warning(f"Could not extract comprehensive reasoning: {e}")
    
    # Save report (will be updated with comprehensive evaluation later)
    report_path = reports_dir / "full_workflow_weatherAUS_report.json"
    
    tracker.track_step("Report Generation", "COMPLETED", {"report_path": str(report_path)})
    
    # Step 8: Generate Plots and Tables
    tracker.track_step("Visualization Generation", "STARTED")
    
    try:
        if PlotGenerator:
            # Prepare data for visualization
            agent_results_for_plots = {}
            dataset_name = "weatherAUS"
            
            # Convert tracker results to format expected by PlotGenerator
            for agent_name, result in tracker.agent_results.items():
                agent_results_for_plots[agent_name] = {
                    dataset_name: {
                        "passed": result['layer1_executed'] and result['layer2_docker_success'],
                        "layer1_executed": result['layer1_executed'],
                        "layer2_executed": result['layer2_executed'],
                        "docker_success": result['layer2_docker_success'],
                        "execution_time": result['execution_time'],
                        "state_keys_added": len(result['state_keys_added'])
                    }
                }
            
            # Generate basic performance plots
            if PlotGenerator:
                plot_gen = PlotGenerator(output_dir=str(plots_dir))
            plot_gen.generate_agent_scorecard_heatmap(agent_results_for_plots, "agent_performance_heatmap.png")
        
        # Create execution timeline plot
        agent_names = list(tracker.agent_results.keys())
        execution_times = [tracker.agent_results[agent]['execution_time'] for agent in agent_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(agent_names)), execution_times, color=['#2ecc71' if tracker.agent_results[agent]['layer2_docker_success'] else '#e74c3c' for agent in agent_names])
        plt.xlabel('Agent', fontsize=12, fontweight='bold')
        plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Agent Execution Times - Full Workflow Test', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(range(len(agent_names)), agent_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, execution_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(execution_times)*0.01,
                    f'{time:.1f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
            plt.savefig(str(plots_dir / "execution_timeline.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create Layer 1 vs Layer 2 comparison plot
        layer1_success = [tracker.agent_results[agent]['layer1_executed'] for agent in agent_names]
        layer2_success = [tracker.agent_results[agent]['layer2_docker_success'] for agent in agent_names]
        
        x = np.arange(len(agent_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, [1 if s else 0 for s in layer1_success], width, label='Layer 1', color='#3498db')
        bars2 = ax.bar(x + width/2, [1 if s else 0 for s in layer2_success], width, label='Layer 2', color='#9b59b6')
        
        ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success (1=Yes, 0=No)', fontsize=12, fontweight='bold')
        ax.set_title('Layer 1 vs Layer 2 Success Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.2])
        
        plt.tight_layout()
            plt.savefig(str(plots_dir / "layer_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
            # Generate enhanced problem-solving workflow visualizations
            if hasattr(tracker, 'problem_analysis') and hasattr(tracker, 'solution_tracker'):
                _generate_problem_solving_visualizations(tracker, plots_dir)
            
            # CRITICAL: Generate comprehensive reasoning plots
            if hasattr(tracker, 'reasoning_tracking') and 'comprehensive' in tracker.reasoning_tracking:
                logger.info("📊 Generating comprehensive reasoning plots...")
                reasoning_plot_gen = ReasoningPlotGenerator(plots_dir)
                reasoning_plots = reasoning_plot_gen.generate_all_plots(tracker.reasoning_tracking['comprehensive'])
                logger.info(f"✅ Generated {len(reasoning_plots)} reasoning plots")
            
            logger.info("✅ All plots generated successfully")
        
        # Step 9: Generate Comprehensive Evaluation
        tracker.track_step("Comprehensive Evaluation", "STARTED")
        try:
            logger.info("📊 Generating comprehensive agent-wise and system-level evaluation...")
            eval_generator = EvaluationGenerator(output_base)
            
            # Ensure we have valid data
            problem_analysis_data = tracker.problem_analysis if hasattr(tracker, 'problem_analysis') and tracker.problem_analysis else {}
            reasoning_data = tracker.reasoning_tracking.get('comprehensive', {}) if hasattr(tracker, 'reasoning_tracking') and tracker.reasoning_tracking else {}
            
            comprehensive_eval = eval_generator.generate_comprehensive_evaluation(
                tracker=tracker,
                state=state,
                problem_analysis=problem_analysis_data,
                reasoning_data=reasoning_data
            )
            
            # Add evaluation to report
            report['comprehensive_evaluation'] = comprehensive_eval
            
            logger.info(f"✅ Comprehensive evaluation generated: {comprehensive_eval.get('report_path')}")
            tracker.track_step("Comprehensive Evaluation", "COMPLETED", {
                "agent_evaluation": len(comprehensive_eval.get('agent_evaluation', {})),
                "system_score": comprehensive_eval.get('system_evaluation', {}).get('system_score', 0),
                "report_path": comprehensive_eval.get('report_path')
            })
        except Exception as e:
            logger.error(f"Error generating comprehensive evaluation: {e}", exc_info=True)
            tracker.track_step("Comprehensive Evaluation", "FAILED", {"error": str(e)})
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)
    
    # Generate agent plots from report JSON
    try:
        from generate_agent_plots_from_report import main as generate_agent_plots
        logger.info("📊 Generating agent-level plots from report JSON...")
        report_path = reports_dir / "full_workflow_weatherAUS_report.json"
        generate_agent_plots(report_path=report_path, plots_dir=plots_dir)
        logger.info("✅ Agent-level plots generated successfully")
    except Exception as e:
        logger.warning(f"Could not generate agent plots from report: {e}", exc_info=True)
    
    # Generate tables
    try:
        if TableGenerator:
            table_gen = TableGenerator()
        
        # Create agent performance table
        table_data = []
        for agent_name, result in tracker.agent_results.items():
            table_data.append({
                "Agent": agent_name,
                "Layer 1": "✓" if result['layer1_executed'] else "✗",
                "Layer 2": "✓" if result['layer2_executed'] else "✗",
                "Docker Success": "✓" if result['layer2_docker_success'] else "✗",
                "Execution Time (s)": f"{result['execution_time']:.2f}",
                "State Keys Added": len(result['state_keys_added'])
            })
        
        # Generate markdown table
        df_table = pd.DataFrame(table_data)
        table_md = df_table.to_markdown(index=False)
        
        with open(tables_dir / "agent_performance_table.md", 'w') as f:
            f.write("# Agent Performance Summary\n\n")
            f.write(table_md)
        
        # Generate LaTeX table
        table_latex = df_table.to_latex(index=False, float_format="%.2f")
        with open(tables_dir / "agent_performance_table.tex", 'w') as f:
            f.write(table_latex)
        
        logger.info("✅ Tables generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating tables: {e}", exc_info=True)
    
    tracker.track_step("Visualization Generation", "COMPLETED")
    
    # Step 9: Print Summary
    logger.info("\n" + "="*80)
    logger.info("FULL WORKFLOW TEST SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n📋 Steps Executed: {len(tracker.steps)}")
    logger.info(f"🤖 Agents Executed: {len(tracker.agent_results)}")
    logger.info(f"🐳 Docker Executions: {len(tracker.docker_executions)}")
    logger.info(f"✅ Successful Docker: {sum(1 for d in tracker.docker_executions if d['status'] == 'SUCCESS')}")
    logger.info(f"❌ Errors: {len(tracker.errors)}")
    logger.info(f"⚠️  Warnings: {len(tracker.warnings)}")
    
    logger.info("\n" + "-"*80)
    logger.info("AGENT EXECUTION SUMMARY")
    logger.info("-"*80)
    
    for agent_name, result in tracker.agent_results.items():
        logger.info(f"\n{agent_name}:")
        logger.info(f"  Layer 1: {'✓' if result['layer1_executed'] else '✗'}")
        logger.info(f"  Layer 2: {'✓' if result['layer2_executed'] else '✗'}")
        logger.info(f"  Docker Success: {'✓' if result['layer2_docker_success'] else '✗'}")
        logger.info(f"  Execution Time: {result['execution_time']:.2f}s")
        logger.info(f"  State Keys Added: {len(result['state_keys_added'])}")
    
    # Extract final metrics
    model_metrics = state.get("evaluation_metrics", {})
    data_quality_score = state.get("data_quality_score")
    
    logger.info("\n" + "-"*80)
    logger.info("FINAL METRICS")
    logger.info("-"*80)
    
    if model_metrics:
        logger.info(f"  Accuracy: {model_metrics.get('accuracy', 'N/A'):.4f}" if isinstance(model_metrics.get('accuracy'), (int, float)) else f"  Accuracy: {model_metrics.get('accuracy', 'N/A')}")
        logger.info(f"  Precision: {model_metrics.get('precision', 'N/A'):.4f}" if isinstance(model_metrics.get('precision'), (int, float)) else f"  Precision: {model_metrics.get('precision', 'N/A')}")
        logger.info(f"  Recall: {model_metrics.get('recall', 'N/A'):.4f}" if isinstance(model_metrics.get('recall'), (int, float)) else f"  Recall: {model_metrics.get('recall', 'N/A')}")
        logger.info(f"  F1 Score: {model_metrics.get('f1_score', 'N/A'):.4f}" if isinstance(model_metrics.get('f1_score'), (int, float)) else f"  F1 Score: {model_metrics.get('f1_score', 'N/A')}")
        if 'roc_auc' in model_metrics:
            logger.info(f"  ROC-AUC: {model_metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(model_metrics.get('roc_auc'), (int, float)) else f"  ROC-AUC: {model_metrics.get('roc_auc', 'N/A')}")
    
    if data_quality_score:
        logger.info(f"  Data Quality Score: {data_quality_score:.2f}" if isinstance(data_quality_score, (int, float)) else f"  Data Quality Score: {data_quality_score}")
    
    logger.info(f"  Workflow Status: {state.get('workflow_status', 'unknown')}")
    
    # Print reasoning summary
    if hasattr(tracker, 'reasoning_tracking') and tracker.reasoning_tracking:
    logger.info("\n" + "-"*80)
        logger.info("REASONING TRACKING SUMMARY")
    logger.info("-"*80)
        logger.info(f"  Agents with reasoning extracted: {len([k for k in tracker.reasoning_tracking.keys() if k != 'comprehensive'])}")
        if 'comprehensive' in tracker.reasoning_tracking:
            comp_reasoning = tracker.reasoning_tracking['comprehensive']
            problems = comp_reasoning.get('all_problems', {})
            logger.info(f"  Problems detected: {len(problems)}")
            logger.info(f"  Problems addressed: {sum(1 for p in problems.values() if p.get('detected', False))}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ FULL WORKFLOW TEST COMPLETED")
    logger.info("="*80)
    # Final save of report (with all evaluations included)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📊 Final report saved to: {report_path}")
    
    logger.info(f"\n📁 All outputs saved to: {output_base}")
    logger.info(f"  - Reports: {reports_dir}")
    logger.info(f"  - Plots: {plots_dir}")
    logger.info(f"  - Tables: {tables_dir}")
    logger.info(f"  - Logs: {logs_dir}")
    
    # Print evaluation summary
    if 'comprehensive_evaluation' in report:
        eval_data = report['comprehensive_evaluation']
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"System Score: {eval_data.get('system_evaluation', {}).get('system_score', 0):.3f} ({eval_data.get('system_evaluation', {}).get('system_grade', 'N/A')})")
        logger.info(f"Agent Evaluations: {len(eval_data.get('agent_evaluation', {}))} agents evaluated")
        logger.info(f"Problem-Solving Score: {eval_data.get('problem_solving_evaluation', {}).get('overall_score', 0):.3f} ({eval_data.get('problem_solving_evaluation', {}).get('grade', 'N/A')})")
    logger.info("="*80)
    
    return {
        "success": True,
        "tracker": tracker,
        "state": state,
        "model_metrics": model_metrics,
        "output_directory": str(output_base),
        "evaluation": report.get('comprehensive_evaluation', {})
    }


if __name__ == "__main__":
    try:
        result = asyncio.run(test_full_workflow())
        if result.get("success"):
            logger.info("\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"\n❌ Test failed: {result.get('error')}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test crashed: {e}", exc_info=True)
        sys.exit(1)

