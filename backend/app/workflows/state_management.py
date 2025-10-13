"""
🎛️ State Management for DS Capstone Project

This module defines the state schema and management utilities for the LangGraph workflow.
It handles the state transitions and data flow between different agents.
"""

from typing import TypedDict, List, Optional, Dict, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

class WorkflowStatus(str, Enum):
    """Workflow status enumeration"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

class AgentStatus(str, Enum):
    """Agent status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ClassificationState(TypedDict):
    """
    🎛️ Main State Schema for Classification Workflow
    
    This state object tracks the entire classification workflow from start to finish.
    It contains all the data and metadata needed for the multi-agent system.
    """
    
    # === INPUT DATA ===
    session_id: str
    dataset_id: str
    target_column: str
    user_description: str
    api_key: str
    
    # === DATASET INFORMATION ===
    original_dataset: Optional[pd.DataFrame]  # Will be stored externally
    dataset_shape: Optional[tuple]
    dataset_metadata: Dict[str, Any]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    duplicate_count: int
    
    # === AGENT EXECUTION STATUS ===
    current_agent: Optional[str]
    workflow_status: WorkflowStatus
    agent_statuses: Dict[str, AgentStatus]
    completed_agents: List[str]
    failed_agents: List[str]
    
    # === DATA CLEANING AGENT OUTPUTS ===
    cleaned_dataset: Optional[pd.DataFrame]  # Will be stored externally
    cleaning_summary: Optional[str]
    data_quality_score: Optional[float]
    cleaning_issues_found: List[str]
    cleaning_actions_taken: List[str]
    
    # === DATA DISCOVERY AGENT OUTPUTS ===
    discovery_results: Optional[Dict[str, Any]]
    similar_datasets_found: List[Dict[str, Any]]
    research_insights: List[str]
    recommended_approaches: List[str]
    domain_knowledge: Dict[str, Any]
    
    # === EDA AGENT OUTPUTS ===
    eda_plots: List[str]  # Paths to generated plots
    statistical_summary: Optional[Dict[str, Any]]
    correlation_matrix: Optional[np.ndarray]
    distribution_analysis: Optional[Dict[str, Any]]
    outlier_analysis: Optional[Dict[str, Any]]
    feature_importance_initial: Optional[Dict[str, float]]
    
    # === FEATURE ENGINEERING AGENT OUTPUTS ===
    engineered_features: List[str]
    feature_selection_results: Optional[Dict[str, Any]]
    feature_transformations: Dict[str, str]
    feature_importance_final: Optional[Dict[str, float]]
    feature_correlation_analysis: Optional[Dict[str, Any]]
    
    # === ML MODEL BUILDER AGENT OUTPUTS ===
    model_selection_results: Optional[Dict[str, Any]]
    best_model: Optional[Any]  # Will be stored externally
    model_hyperparameters: Optional[Dict[str, Any]]
    training_metrics: Optional[Dict[str, float]]
    cross_validation_scores: Optional[Dict[str, List[float]]]
    model_explanation: Optional[str]
    
    # === MODEL EVALUATION AGENT OUTPUTS ===
    evaluation_metrics: Optional[Dict[str, float]]
    confusion_matrix: Optional[np.ndarray]
    roc_curve_data: Optional[Dict[str, Any]]
    precision_recall_curve: Optional[Dict[str, Any]]
    feature_importance_model: Optional[Dict[str, float]]
    model_performance_analysis: Optional[str]
    
    # === TECHNICAL REPORTER AGENT OUTPUTS ===
    final_report: Optional[str]
    executive_summary: Optional[str]
    technical_documentation: Optional[str]
    recommendations: List[str]
    limitations: List[str]
    future_improvements: List[str]
    
    # === PROJECT MANAGER OUTPUTS ===
    workflow_progress: float
    progress: float  # Alias for workflow_progress
    estimated_completion_time: Optional[datetime]
    resource_usage: Dict[str, Any]
    quality_checks_passed: List[str]
    quality_checks_failed: List[str]
    
    # === ERROR HANDLING ===
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    error_count: int
    last_error: Optional[str]
    
    # === PERFORMANCE TRACKING ===
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_execution_time: Optional[float]
    agent_execution_times: Dict[str, float]
    memory_usage: Dict[str, float]
    cpu_usage: Dict[str, float]
    
    # === USER INTERACTION ===
    requires_human_input: bool
    human_input_required: Optional[str]
    human_feedback: Optional[Dict[str, Any]]
    user_approvals: Dict[str, bool]
    
    # === OUTPUT ARTIFACTS ===
    output_artifacts: Dict[str, str]  # artifact_type -> file_path
    downloadable_files: List[Dict[str, str]]  # name, path, type, size
    notebook_path: Optional[str]
    model_path: Optional[str]
    report_path: Optional[str]

class StateManager:
    """
    🎛️ State Management Utility Class
    
    Provides utilities for managing and updating the workflow state.
    Handles state transitions, validation, and persistence.
    """
    
    def __init__(self):
        self.state_history: List[ClassificationState] = []
        self.external_storage: Dict[str, Any] = {}  # For DataFrames and large objects
    
    def initialize_state(
        self,
        session_id: str,
        dataset_id: str,
        target_column: str,
        user_description: str,
        api_key: str,
        original_dataset: pd.DataFrame
    ) -> ClassificationState:
        """Initialize a new workflow state"""
        
        # Store large objects externally
        self.external_storage[dataset_id] = {
            'original': original_dataset,
            'cleaned': None,
            'model': None
        }
        
        state: ClassificationState = {
            # Input data
            "session_id": session_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "user_description": user_description,
            "api_key": api_key,
            
            # Dataset information
            "original_dataset": None,  # Stored externally
            "dataset_shape": original_dataset.shape,
            "dataset_metadata": {
                "columns": list(original_dataset.columns),
                "dtypes": dict(original_dataset.dtypes),
                "memory_usage": original_dataset.memory_usage(deep=True).sum(),
                "created_at": datetime.now().isoformat()
            },
            "data_types": {col: str(dtype) for col, dtype in dict(original_dataset.dtypes).items()},
            "missing_values": dict(original_dataset.isnull().sum()),
            "duplicate_count": int(original_dataset.duplicated().sum()),
            
            # Agent execution status
            "current_agent": None,
            "workflow_status": WorkflowStatus.INITIALIZED,
            "agent_statuses": {
                "data_cleaning": AgentStatus.PENDING,
                "data_discovery": AgentStatus.PENDING,
                "eda_analysis": AgentStatus.PENDING,
                "feature_engineering": AgentStatus.PENDING,
                "ml_building": AgentStatus.PENDING,
                "model_evaluation": AgentStatus.PENDING,
                "technical_reporting": AgentStatus.PENDING,
                "project_manager": AgentStatus.PENDING
            },
            "completed_agents": [],
            "failed_agents": [],
            
            # Initialize all outputs as None/empty
            "cleaned_dataset": None,
            "cleaning_summary": None,
            "data_quality_score": None,
            "cleaning_issues_found": [],
            "cleaning_actions_taken": [],
            
            "discovery_results": None,
            "similar_datasets_found": [],
            "research_insights": [],
            "recommended_approaches": [],
            "domain_knowledge": {},
            
            "eda_plots": [],
            "statistical_summary": None,
            "correlation_matrix": None,
            "distribution_analysis": None,
            "outlier_analysis": None,
            "feature_importance_initial": None,
            
            "engineered_features": [],
            "feature_selection_results": None,
            "feature_transformations": {},
            "feature_importance_final": None,
            "feature_correlation_analysis": None,
            
            "model_selection_results": None,
            "best_model": None,
            "model_hyperparameters": None,
            "training_metrics": None,
            "cross_validation_scores": None,
            "model_explanation": None,
            
            "evaluation_metrics": None,
            "confusion_matrix": None,
            "roc_curve_data": None,
            "precision_recall_curve": None,
            "feature_importance_model": None,
            "model_performance_analysis": None,
            
            "final_report": None,
            "executive_summary": None,
            "technical_documentation": None,
            "recommendations": [],
            "limitations": [],
            "future_improvements": [],
            
            "workflow_progress": 0.0,
            "progress": 0.0,  # Alias for workflow_progress
            "estimated_completion_time": None,
            "resource_usage": {},
            "quality_checks_passed": [],
            "quality_checks_failed": [],
            
            "errors": [],
            "warnings": [],
            "retry_count": 0,
            "max_retries": 3,
            "error_count": 0,
            "last_error": None,
            
            "start_time": datetime.now(),
            "end_time": None,
            "total_execution_time": None,
            "agent_execution_times": {},
            "memory_usage": {},
            "cpu_usage": {},
            
            "requires_human_input": False,
            "human_input_required": None,
            "human_feedback": None,
            "user_approvals": {},
            
            "output_artifacts": {},
            "downloadable_files": [],
            "notebook_path": None,
            "model_path": None,
            "report_path": None
        }
        
        self.state_history.append(state.copy())
        return state
    
    def update_agent_status(
        self,
        state: ClassificationState,
        agent_name: str,
        status: AgentStatus,
        execution_time: Optional[float] = None
    ) -> ClassificationState:
        """Update the status of a specific agent"""
        
        state["agent_statuses"][agent_name] = status
        
        if status == AgentStatus.COMPLETED:
            state["completed_agents"].append(agent_name)
            if agent_name in state["failed_agents"]:
                state["failed_agents"].remove(agent_name)
        elif status == AgentStatus.FAILED:
            state["failed_agents"].append(agent_name)
            if agent_name in state["completed_agents"]:
                state["completed_agents"].remove(agent_name)
        
        if execution_time:
            state["agent_execution_times"][agent_name] = execution_time
        
        # Update workflow progress
        total_agents = len(state["agent_statuses"])
        completed_agents = len(state["completed_agents"])
        state["workflow_progress"] = (completed_agents / total_agents) * 100
        
        return state
    
    def add_error(
        self,
        state: ClassificationState,
        agent_name: str,
        error_message: str,
        error_type: str = "execution_error"
    ) -> ClassificationState:
        """Add an error to the state"""
        
        error = {
            "agent": agent_name,
            "message": error_message,
            "type": error_type,
            "timestamp": datetime.now().isoformat()
        }
        
        state["errors"].append(error)
        return state
    
    def add_warning(
        self,
        state: ClassificationState,
        agent_name: str,
        warning_message: str,
        warning_type: str = "general_warning"
    ) -> ClassificationState:
        """Add a warning to the state"""
        
        warning = {
            "agent": agent_name,
            "message": warning_message,
            "type": warning_type,
            "timestamp": datetime.now().isoformat()
        }
        
        state["warnings"].append(warning)
        return state
    
    def get_dataset(self, state: ClassificationState, dataset_type: str = "original") -> Optional[pd.DataFrame]:
        """Get dataset from external storage"""
        dataset_id = state["dataset_id"]
        if dataset_id in self.external_storage:
            return self.external_storage[dataset_id].get(dataset_type)
        return None
    
    def store_dataset(self, state: ClassificationState, dataset: pd.DataFrame, dataset_type: str = "cleaned") -> None:
        """Store dataset in external storage"""
        dataset_id = state["dataset_id"]
        if dataset_id not in self.external_storage:
            self.external_storage[dataset_id] = {}
        self.external_storage[dataset_id][dataset_type] = dataset
    
    def get_serializable_state(self, state: ClassificationState) -> Dict[str, Any]:
        """Get a serializable version of the state (without DataFrames)"""
        serializable_state = state.copy()
        
        # Remove non-serializable objects
        serializable_state["original_dataset"] = None
        serializable_state["cleaned_dataset"] = None
        serializable_state["best_model"] = None
        serializable_state["correlation_matrix"] = None
        serializable_state["confusion_matrix"] = None
        
        # Convert numpy arrays to lists
        for key, value in serializable_state.items():
            if isinstance(value, np.ndarray):
                serializable_state[key] = value.tolist()
        
        return serializable_state
    
    def validate_state(self, state: ClassificationState) -> List[str]:
        """Validate the state and return any validation errors"""
        errors = []
        
        # Check required fields
        required_fields = ["session_id", "dataset_id", "target_column", "user_description", "api_key"]
        for field in required_fields:
            if not state.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Check workflow status consistency
        if state["workflow_status"] == WorkflowStatus.COMPLETED:
            if len(state["completed_agents"]) < len(state["agent_statuses"]):
                errors.append("Workflow marked as completed but not all agents are completed")
        
        # Check for critical errors
        if len(state["errors"]) > state["max_retries"]:
            errors.append("Too many errors, workflow should be failed")
        
        return errors
    
    def get_state_summary(self, state: ClassificationState) -> Dict[str, Any]:
        """Get a summary of the current state"""
        return {
            "session_id": state["session_id"],
            "workflow_status": state["workflow_status"],
            "current_agent": state["current_agent"],
            "progress": state["workflow_progress"],
            "completed_agents": state["completed_agents"],
            "failed_agents": state["failed_agents"],
            "error_count": len(state["errors"]),
            "warning_count": len(state["warnings"]),
            "execution_time": state["total_execution_time"],
            "requires_human_input": state["requires_human_input"]
        }

# Global state manager instance
state_manager = StateManager()
