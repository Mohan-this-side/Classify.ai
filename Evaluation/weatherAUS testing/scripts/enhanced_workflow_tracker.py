"""
Enhanced Workflow Tracker

Tracks dataset problems and how agents solve them, demonstrating data scientist workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class DatasetProblemAnalyzer:
    """Analyze dataset problems before agent processing."""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.problems = []
        self.metrics = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Comprehensive problem analysis."""
        problems = []
        
        # 1. Missing Values Analysis
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 5}
        if high_missing:
            problems.append({
                "type": "Missing Values",
                "severity": "high" if max(high_missing.values()) > 30 else "medium",
                "description": f"{len(high_missing)} columns have >5% missing values",
                "details": high_missing,
                "expected_action": "Data Cleaning agent should impute or drop"
            })
        
        # 2. Class Imbalance Analysis
        if self.target_col in self.df.columns:
            target_dist = self.df[self.target_col].value_counts(normalize=True)
            if len(target_dist) == 2:
                imbalance_ratio = min(target_dist) / max(target_dist)
                if imbalance_ratio < 0.3:
                    problems.append({
                        "type": "Class Imbalance",
                        "severity": "high" if imbalance_ratio < 0.1 else "medium",
                        "description": f"Class imbalance ratio: {imbalance_ratio:.3f}",
                        "details": target_dist.to_dict(),
                        "expected_action": "EDA agent should detect, ML Builder should use class weights/SMOTE"
                    })
        
        # 3. High Cardinality Categorical Features
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        high_cardinality = {}
        for col in categorical_cols:
            if col != self.target_col:
                cardinality = self.df[col].nunique()
                if cardinality > 50:
                    high_cardinality[col] = cardinality
        
        if high_cardinality:
            problems.append({
                "type": "High Cardinality",
                "severity": "medium",
                "description": f"{len(high_cardinality)} categorical columns have >50 unique values",
                "details": high_cardinality,
                "expected_action": "Feature Engineering agent should use target encoding or reduce categories"
            })
        
        # 4. Multicollinearity Detection
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append({
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                problems.append({
                    "type": "Multicollinearity",
                    "severity": "medium",
                    "description": f"{len(high_corr_pairs)} pairs have correlation >0.95",
                    "details": high_corr_pairs[:5],  # Top 5
                    "expected_action": "Feature Engineering agent should remove redundant features"
                })
        
        # 5. Outliers Detection
        outlier_counts = {}
        for col in numeric_cols:
            if col != self.target_col:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outliers / len(self.df)) * 100
                if outlier_pct > 10:
                    outlier_counts[col] = {"count": outliers, "percentage": outlier_pct}
        
        if outlier_counts:
            problems.append({
                "type": "Outliers",
                "severity": "medium",
                "description": f"{len(outlier_counts)} columns have >10% outliers",
                "details": outlier_counts,
                "expected_action": "Data Cleaning agent should cap or transform outliers"
            })
        
        # 6. Zero Variance Features
        zero_variance = []
        for col in numeric_cols:
            if self.df[col].nunique() <= 1:
                zero_variance.append(col)
        
        if zero_variance:
            problems.append({
                "type": "Zero Variance",
                "severity": "high",
                "description": f"{len(zero_variance)} columns have zero variance",
                "details": zero_variance,
                "expected_action": "Feature Engineering agent should remove these features"
            })
        
        # 7. Data Types Issues
        type_issues = []
        for col in self.df.columns:
            if col != self.target_col:
                # Check if numeric column has non-numeric values
                if self.df[col].dtype == 'object':
                    try:
                        pd.to_numeric(self.df[col], errors='raise')
                        type_issues.append({
                            "column": col,
                            "issue": "Numeric data stored as string"
                        })
                    except:
                        pass
        
        if type_issues:
            problems.append({
                "type": "Data Type Issues",
                "severity": "medium",
                "description": f"{len(type_issues)} columns have type issues",
                "details": type_issues,
                "expected_action": "Data Cleaning agent should convert types"
            })
        
        # Calculate overall metrics
        metrics = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_value_percentage": (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "problems_detected": len(problems),
            "high_severity_problems": len([p for p in problems if p["severity"] == "high"]),
            "medium_severity_problems": len([p for p in problems if p["severity"] == "medium"])
        }
        
        return {
            "problems": problems,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }


class AgentSolutionTracker:
    """Track how agents solve problems."""
    
    def __init__(self):
        self.solutions = {}
        self.problem_solutions = {}  # Maps problem type to solution
        
    def track_solution(self, agent_name: str, problem_type: str, solution: Dict[str, Any]):
        """Track how an agent solved a problem."""
        if agent_name not in self.solutions:
            self.solutions[agent_name] = []
        
        solution_entry = {
            "problem_type": problem_type,
            "solution": solution,
            "timestamp": datetime.now().isoformat()
        }
        self.solutions[agent_name].append(solution_entry)
        
        # Map problem to solution
        if problem_type not in self.problem_solutions:
            self.problem_solutions[problem_type] = []
        self.problem_solutions[problem_type].append({
            "agent": agent_name,
            "solution": solution
        })
    
    def get_solution_summary(self) -> Dict[str, Any]:
        """Get summary of all solutions."""
        return {
            "agents": self.solutions,
            "problem_solutions": self.problem_solutions,
            "total_solutions": sum(len(sols) for sols in self.solutions.values())
        }


class WorkflowProgressTracker:
    """Track workflow progress showing data scientist steps."""
    
    def __init__(self):
        self.steps = []
        self.dataset_state_history = []
        
    def track_step(self, step_name: str, agent_name: str, 
                   dataset_state: Dict[str, Any], 
                   problems_solved: List[str],
                   actions_taken: List[str]):
        """Track a workflow step."""
        step = {
            "step_name": step_name,
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
            "dataset_state": dataset_state,
            "problems_solved": problems_solved,
            "actions_taken": actions_taken
        }
        self.steps.append(step)
        self.dataset_state_history.append({
            "step": step_name,
            "state": dataset_state,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get workflow progress summary."""
        return {
            "total_steps": len(self.steps),
            "steps": self.steps,
            "dataset_state_history": self.dataset_state_history
        }

