"""
LangGraph Workflow for Multi-Agent Data Science Classification System

This module implements the main workflow that coordinates all agents in the
classification pipeline using LangGraph's state machine approach.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from ..agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from ..agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from ..agents.data_analysis.eda_agent import EDAAgent
from ..agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
from ..agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
from ..agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
from ..agents.reporting.technical_reporter_agent import TechnicalReporterAgent
from ..agents.coordination.project_manager_agent import ProjectManagerAgent
from ..workflows.state_management import ClassificationState, WorkflowStatus, AgentStatus, state_manager
from ..workflows.approval_gates import (
    ApprovalGateManager, 
    ApprovalGateType, 
    should_trigger_approval_gate,
    create_approval_proposal,
    generate_educational_explanation
)
from ..config import get_settings
from ..services import realtime

logger = logging.getLogger(__name__)
settings = get_settings()


class ClassificationWorkflow:
    """
    Main workflow orchestrator for the multi-agent classification system.
    
    This class manages the entire pipeline from data upload to final report generation,
    coordinating all agents and maintaining state throughout the process.
    """
    
    def __init__(self):
        """Initialize the workflow with all required agents and configuration."""
        self.settings = settings
        self.checkpointer = MemorySaver()
        self.workflow_id = str(uuid.uuid4())
        
        # Initialize all agents
        self.data_cleaning_agent = EnhancedDataCleaningAgent()
        self.data_discovery_agent = DataDiscoveryAgent()
        self.eda_agent = EDAAgent()
        self.feature_engineering_agent = FeatureEngineeringAgent()
        self.ml_builder_agent = MLBuilderAgent()
        self.model_evaluation_agent = ModelEvaluationAgent()
        self.technical_reporter_agent = TechnicalReporterAgent()
        self.project_manager_agent = ProjectManagerAgent()
        
        # Initialize approval gate manager
        self.approval_manager = ApprovalGateManager()
        
        # Build the workflow graph
        self.graph = self._build_workflow_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        logger.info(f"ClassificationWorkflow initialized with ID: {self.workflow_id}")
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with all nodes and edges.
        
        Returns:
            StateGraph: The compiled workflow graph
        """
        workflow = StateGraph(ClassificationState)
        
        # Add nodes for each agent
        workflow.add_node("data_cleaning", self._data_cleaning_node)
        workflow.add_node("data_discovery", self._data_discovery_node)
        workflow.add_node("eda_analysis", self._eda_analysis_node)
        workflow.add_node("feature_engineering", self._feature_engineering_node)
        workflow.add_node("ml_building", self._ml_building_node)
        workflow.add_node("model_evaluation", self._model_evaluation_node)
        workflow.add_node("technical_reporting", self._technical_reporting_node)
        workflow.add_node("project_management", self._project_management_node)
        
        # Add error handling nodes
        workflow.add_node("error_recovery", self._error_recovery_node)
        workflow.add_node("workflow_completion", self._workflow_completion_node)
        
        # Define the workflow flow
        workflow.set_entry_point("data_cleaning")
        
        # Main pipeline flow with project management coordination
        workflow.add_edge("data_cleaning", "project_management")
        workflow.add_edge("project_management", "data_discovery")
        workflow.add_edge("data_discovery", "project_management")
        workflow.add_edge("project_management", "eda_analysis")
        workflow.add_edge("eda_analysis", "project_management")
        workflow.add_edge("project_management", "feature_engineering")
        workflow.add_edge("feature_engineering", "project_management")
        workflow.add_edge("project_management", "ml_building")
        workflow.add_edge("ml_building", "project_management")
        workflow.add_edge("project_management", "model_evaluation")
        workflow.add_edge("model_evaluation", "project_management")
        workflow.add_edge("project_management", "technical_reporting")
        workflow.add_edge("technical_reporting", "workflow_completion")
        
        # Error handling flow
        workflow.add_edge("error_recovery", "workflow_completion")
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "data_cleaning",
            self._should_continue_after_cleaning,
            {
                "continue": "project_management",
                "error": "error_recovery",
                "end": END
            }
        )
        
        # Add conditional edges for project management
        workflow.add_conditional_edges(
            "project_management",
            self._should_continue_after_project_management,
            {
                "continue": "data_discovery",
                "error": "error_recovery",
                "end": END
            }
        )
        
        # Add similar conditional edges for other nodes
        for node in ["data_discovery", "eda_analysis", "feature_engineering", 
                    "ml_building", "model_evaluation", "technical_reporting"]:
            workflow.add_conditional_edges(
                node,
                self._should_continue_workflow,
                {
                    "continue": "project_management",
                    "error": "error_recovery",
                    "end": END
                }
            )
        
        return workflow
    
    async def _data_cleaning_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute data cleaning agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after data cleaning
        """
        try:
            logger.info("Starting data cleaning phase")
            # Build delta state to avoid duplicate writes of immutable keys
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "data_cleaning"
            # mark running
            running_status = state["agent_statuses"].copy()
            running_status["data_cleaning"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            delta["workflow_status"] = WorkflowStatus.RUNNING
            # Update progress
            delta["workflow_progress"] = 14.0  # 1/7 * 100
            delta["progress"] = 14.0
            
            # Check for required data
            if not state.get("dataset"):
                raise ValueError("No dataset provided for data cleaning")
            
            # Execute data cleaning via agent run (handles status updates internally)
            state_after = await self.data_cleaning_agent.run(state)
            
            # Extract results to delta
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["cleaning_summary"] = state_after.get("cleaning_summary")
            delta["data_quality_score"] = state_after.get("data_quality_score")
            delta["cleaning_actions_taken"] = state_after.get("cleaning_actions_taken", [])
            delta["cleaning_issues_found"] = state_after.get("cleaning_issues_found", [])
            
            logger.info("Data cleaning phase completed")
            # Realtime update
            await realtime.emit(state_after.get("session_id"), "agent_update", {
                "agent": "data_cleaning",
                "status": str(completed_status.get("data_cleaning")),
                "progress": 14.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            delta: Dict[str, Any] = {}
            delta["agent_statuses"] = {**state["agent_statuses"], "data_cleaning": AgentStatus.FAILED}
            delta["last_error"] = str(e)
            delta["error_count"] = state.get("error_count", 0) + 1
            delta["workflow_status"] = WorkflowStatus.FAILED
            return delta
    
    async def _data_discovery_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute data discovery agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after data discovery
        """
        try:
            logger.info("Starting data discovery phase")
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "data_discovery"
            running_status = state["agent_statuses"].copy()
            running_status["data_discovery"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            # Update progress
            delta["workflow_progress"] = 28.0  # 2/7 * 100
            delta["progress"] = 28.0
            
            # Execute discovery via agent
            state_after = await self.data_discovery_agent.run(state)
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["discovery_results"] = state_after.get("discovery_results")
            
            logger.info("Data discovery phase completed")
            
            # Check for approval gates after data discovery
            state_after_approval = await self._check_approval_gates(state_after, "data_discovery")
            if state_after_approval.get("workflow_status") == WorkflowStatus.PAUSED:
                delta["workflow_status"] = WorkflowStatus.PAUSED
                delta["pending_approval_gate"] = state_after_approval.get("pending_approval_gate")
                delta["approval_gate_type"] = state_after_approval.get("approval_gate_type")
            
            await realtime.emit(state.get("session_id"), "agent_update", {
                "agent": "data_discovery",
                "status": str(completed_status.get("data_discovery")),
                "progress": 28.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in data discovery: {str(e)}")
            state["agent_statuses"]["data_discovery"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state
    
    async def _eda_analysis_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute EDA analysis agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after EDA analysis
        """
        try:
            logger.info("Starting EDA analysis phase")
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "eda_analysis"
            running_status = state["agent_statuses"].copy()
            running_status["eda_analysis"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            # Update progress
            delta["workflow_progress"] = 42.0
            delta["progress"] = 42.0
            
            # Execute EDA via agent
            state_after = await self.eda_agent.run(state)
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["statistical_summary"] = state_after.get("statistical_summary")
            delta["eda_plots"] = state_after.get("eda_plots")
            delta["distribution_analysis"] = state_after.get("distribution_analysis")
            delta["outlier_analysis"] = state_after.get("outlier_analysis")
            
            logger.info("EDA analysis phase completed")
            
            # Check for approval gates after EDA analysis
            state_after_approval = await self._check_approval_gates(state_after, "eda_analysis")
            if state_after_approval.get("workflow_status") == WorkflowStatus.PAUSED:
                delta["workflow_status"] = WorkflowStatus.PAUSED
                delta["pending_approval_gate"] = state_after_approval.get("pending_approval_gate")
                delta["approval_gate_type"] = state_after_approval.get("approval_gate_type")
            
            await realtime.emit(state.get("session_id"), "agent_update", {
                "agent": "eda_analysis",
                "status": str(completed_status.get("eda_analysis")),
                "progress": 42.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in EDA analysis: {str(e)}")
            state["agent_statuses"]["eda_analysis"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state
    
    async def _feature_engineering_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute feature engineering agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after feature engineering
        """
        try:
            logger.info("Starting feature engineering phase")
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "feature_engineering"
            running_status = state["agent_statuses"].copy()
            running_status["feature_engineering"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            # Update progress
            delta["workflow_progress"] = 56.0
            delta["progress"] = 56.0
            
            # Execute Feature Engineering via agent
            state_after = await self.feature_engineering_agent.run(state)
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["engineered_features"] = state_after.get("engineered_features")
            delta["feature_selection_results"] = state_after.get("feature_selection_results")
            delta["feature_transformations"] = state_after.get("feature_transformations")
            
            logger.info("Feature engineering phase completed")
            
            # Check for approval gates after feature engineering
            state_after_approval = await self._check_approval_gates(state_after, "feature_engineering")
            if state_after_approval.get("workflow_status") == WorkflowStatus.PAUSED:
                delta["workflow_status"] = WorkflowStatus.PAUSED
                delta["pending_approval_gate"] = state_after_approval.get("pending_approval_gate")
                delta["approval_gate_type"] = state_after_approval.get("approval_gate_type")
            
            await realtime.emit(state.get("session_id"), "agent_update", {
                "agent": "feature_engineering",
                "status": str(completed_status.get("feature_engineering")),
                "progress": 56.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            state["agent_statuses"]["feature_engineering"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state
    
    async def _ml_building_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute ML model building agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after ML building
        """
        try:
            logger.info("Starting ML model building phase")
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "ml_building"
            running_status = state["agent_statuses"].copy()
            running_status["ml_building"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            
            # Update progress
            delta["workflow_progress"] = 70.0
            delta["progress"] = 70.0
            
            # Execute ML building via agent
            state_after = await self.ml_builder_agent.run(state)
            
            # Extract results to delta
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["model_selection_results"] = state_after.get("model_selection_results")
            delta["best_model"] = state_after.get("best_model")
            delta["model_hyperparameters"] = state_after.get("model_hyperparameters")
            delta["training_metrics"] = state_after.get("training_metrics")
            delta["cross_validation_scores"] = state_after.get("cross_validation_scores")
            delta["evaluation_metrics"] = state_after.get("evaluation_metrics")
            delta["model_explanation"] = state_after.get("model_explanation")
            
            logger.info("ML model building phase completed")
            
            # Check for approval gates after ML building
            state_after_approval = await self._check_approval_gates(state_after, "ml_building")
            if state_after_approval.get("workflow_status") == WorkflowStatus.PAUSED:
                delta["workflow_status"] = WorkflowStatus.PAUSED
                delta["pending_approval_gate"] = state_after_approval.get("pending_approval_gate")
                delta["approval_gate_type"] = state_after_approval.get("approval_gate_type")
            
            await realtime.emit(state.get("session_id"), "agent_update", {
                "agent": "ml_building",
                "status": str(completed_status.get("ml_building")),
                "progress": 70.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in ML building: {str(e)}")
            delta: Dict[str, Any] = {}
            delta["agent_statuses"] = {**state["agent_statuses"], "ml_building": AgentStatus.FAILED}
            delta["last_error"] = str(e)
            delta["error_count"] = state.get("error_count", 0) + 1
            return delta
    
    async def _model_evaluation_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute model evaluation agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after model evaluation
        """
        try:
            logger.info("Starting model evaluation phase")
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "model_evaluation"
            running_status = state["agent_statuses"].copy()
            running_status["model_evaluation"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            # Update progress
            delta["workflow_progress"] = 84.0
            delta["progress"] = 84.0
            
            # Execute model evaluation via agent
            state_after = await self.model_evaluation_agent.run(state)
            
            # Extract results to delta
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["evaluation_metrics"] = state_after.get("evaluation_metrics")
            delta["confusion_matrix"] = state_after.get("confusion_matrix")
            delta["roc_curve_data"] = state_after.get("roc_curve_data")
            delta["precision_recall_curve"] = state_after.get("precision_recall_curve")
            delta["feature_importance_model"] = state_after.get("feature_importance_model")
            delta["model_performance_analysis"] = state_after.get("model_performance_analysis")
            
            logger.info("Model evaluation phase completed")
            
            # Check for approval gates after model evaluation
            state_after_approval = await self._check_approval_gates(state_after, "model_evaluation")
            if state_after_approval.get("workflow_status") == WorkflowStatus.PAUSED:
                delta["workflow_status"] = WorkflowStatus.PAUSED
                delta["pending_approval_gate"] = state_after_approval.get("pending_approval_gate")
                delta["approval_gate_type"] = state_after_approval.get("approval_gate_type")
            
            await realtime.emit(state.get("session_id"), "agent_update", {
                "agent": "model_evaluation",
                "status": str(completed_status.get("model_evaluation")),
                "progress": 84.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            delta: Dict[str, Any] = {}
            delta["agent_statuses"] = {**state["agent_statuses"], "model_evaluation": AgentStatus.FAILED}
            delta["last_error"] = str(e)
            delta["error_count"] = state.get("error_count", 0) + 1
            return delta
    
    async def _technical_reporting_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute technical reporting agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after technical reporting
        """
        try:
            logger.info("Starting technical reporting phase")
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "technical_reporting"
            running_status = state["agent_statuses"].copy()
            running_status["technical_reporting"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            # Update progress
            delta["workflow_progress"] = 98.0
            delta["progress"] = 98.0
            
            # Execute technical reporting via agent
            state_after = await self.technical_reporter_agent.run(state)
            
            # Extract results to delta
            completed_status = state_after["agent_statuses"]
            delta["agent_statuses"] = completed_status
            delta["final_report"] = state_after.get("final_report")
            delta["executive_summary"] = state_after.get("executive_summary")
            delta["technical_documentation"] = state_after.get("technical_documentation")
            delta["notebook_path"] = state_after.get("notebook_path")
            delta["recommendations"] = state_after.get("recommendations")
            delta["limitations"] = state_after.get("limitations")
            delta["future_improvements"] = state_after.get("future_improvements")
            
            logger.info("Technical reporting phase completed")
            await realtime.emit(state.get("session_id"), "agent_update", {
                "agent": "technical_reporting",
                "status": str(completed_status.get("technical_reporting")),
                "progress": 98.0
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in technical reporting: {str(e)}")
            delta: Dict[str, Any] = {}
            delta["agent_statuses"] = {**state["agent_statuses"], "technical_reporting": AgentStatus.FAILED}
            delta["last_error"] = str(e)
            delta["error_count"] = state.get("error_count", 0) + 1
            return delta
    
    async def _project_management_node(self, state: ClassificationState) -> ClassificationState:
        """
        Execute project management agent for real-time updates.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after project management
        """
        try:
            logger.info("Executing Project Management node")
            
            # Execute the Project Manager Agent
            updated_state = await self.project_manager_agent.execute(state)
            
            # Return only delta updates
            delta: Dict[str, Any] = {}
            if "project_management" in updated_state:
                delta["project_management"] = updated_state["project_management"]
            
            # Update progress based on completed agents
            completed_agents = updated_state.get("completed_agents", [])
            total_agents = 8
            progress = (len(completed_agents) / total_agents) * 100
            delta["progress"] = progress
            
            # Update workflow status if all agents are completed
            if len(completed_agents) >= total_agents:
                delta["workflow_status"] = WorkflowStatus.COMPLETED
            elif updated_state.get("workflow_status") == WorkflowStatus.PAUSED:
                delta["workflow_status"] = WorkflowStatus.PAUSED
            
            return delta
            
        except Exception as e:
            logger.error(f"Error in project management: {str(e)}")
            delta: Dict[str, Any] = {}
            delta["last_error"] = str(e)
            delta["error_count"] = state.get("error_count", 0) + 1
            return delta
    
    async def _error_recovery_node(self, state: ClassificationState) -> ClassificationState:
        """
        Handle error recovery and retry logic.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after error recovery
        """
        try:
            logger.info("Starting error recovery")
            previous_agent = state.get("current_agent")
            error_count = state.get("error_count", 0)
            last_error = state.get("last_error", "Unknown error")
            
            delta: Dict[str, Any] = {}
            delta["current_agent"] = "error_recovery"
            running_status = state["agent_statuses"].copy()
            running_status["error_recovery"] = AgentStatus.RUNNING
            delta["agent_statuses"] = running_status
            
            # Log error details
            logger.error(f"Error recovery triggered. Agent: {previous_agent}, Count: {error_count}, Error: {last_error}")
            
            # Implement comprehensive error recovery logic
            if error_count < 3:
                # Retry the failed agent
                failed_agent = previous_agent
                logger.info(f"Retrying failed agent: {failed_agent} (attempt {error_count + 1}/3)")
                
                # Reset agent status
                updated_status = delta["agent_statuses"].copy()
                if failed_agent:
                    updated_status[failed_agent] = AgentStatus.PENDING
                    # Remove from failed agents list
                    failed_agents = state.get("failed_agents", [])
                    if failed_agent in failed_agents:
                        failed_agents.remove(failed_agent)
                        delta["failed_agents"] = failed_agents
                
                delta["agent_statuses"] = updated_status
                delta["error_count"] = 0
                delta["last_error"] = None
                
                # Add retry information to state
                delta["retry_attempts"] = state.get("retry_attempts", 0) + 1
                delta["recovery_timestamp"] = datetime.now().isoformat()
                
                logger.info(f"Agent {failed_agent} reset for retry")
                
            else:
                # Max retries exceeded, skip to next agent or fail workflow
                logger.error(f"Max retries exceeded for agent: {previous_agent}")
                
                # Mark agent as permanently failed
                updated_status = delta["agent_statuses"].copy()
                if previous_agent:
                    updated_status[previous_agent] = AgentStatus.FAILED
                    failed_agents = state.get("failed_agents", [])
                    if previous_agent not in failed_agents:
                        failed_agents.append(previous_agent)
                        delta["failed_agents"] = failed_agents
                
                delta["agent_statuses"] = updated_status
                delta["workflow_status"] = WorkflowStatus.FAILED
                delta["final_error"] = f"Agent {previous_agent} failed after {error_count} retries: {last_error}"
                
                logger.error(f"Workflow marked as failed due to persistent agent failure")
            
            return delta
            
        except Exception as e:
            logger.error(f"Error in error recovery: {str(e)}")
            state["workflow_status"] = WorkflowStatus.FAILED
            return state
    
    async def _workflow_completion_node(self, state: ClassificationState) -> ClassificationState:
        """
        Handle workflow completion and finalization.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final state
        """
        try:
            logger.info("Completing workflow")
            delta: Dict[str, Any] = {}
            delta["workflow_status"] = WorkflowStatus.COMPLETED
            delta["workflow_progress"] = 100.0
            delta["progress"] = 100.0
            delta["completion_time"] = datetime.now().isoformat()
            
            # Generate final summary
            summary = {
                "total_agents": len(state["agent_statuses"]),
                "completed_agents": sum(1 for s in state["agent_statuses"].values() if s == AgentStatus.COMPLETED),
                "failed_agents": sum(1 for s in state["agent_statuses"].values() if s == AgentStatus.FAILED),
                "total_errors": state.get("error_count", 0),
                "completion_time": delta["completion_time"]
            }
            # Ensure summary uses plain Python types for msgpack
            safe_summary = {
                "total_agents": int(summary["total_agents"]),
                "completed_agents": int(summary["completed_agents"]),
                "failed_agents": int(summary["failed_agents"]),
                "total_errors": int(summary["total_errors"]),
                "completion_time": str(summary["completion_time"])
            }
            delta["workflow_summary"] = safe_summary
            
            logger.info("Workflow completed successfully")
            await realtime.emit(state.get("session_id"), "workflow_complete", {
                "status": str(delta["workflow_status"]),
                "summary": delta["workflow_summary"]
            })
            return delta
            
        except Exception as e:
            logger.error(f"Error in workflow completion: {str(e)}")
            state["workflow_status"] = WorkflowStatus.FAILED
            return state
    
    def _should_continue_after_cleaning(self, state: ClassificationState) -> str:
        """Determine if workflow should continue after data cleaning."""
        if state["agent_statuses"]["data_cleaning"] == AgentStatus.COMPLETED:
            return "continue"
        elif state["agent_statuses"]["data_cleaning"] == AgentStatus.FAILED:
            return "error"
        else:
            return "end"
    
    def _should_continue_workflow(self, state: ClassificationState) -> str:
        """Determine if workflow should continue after current agent."""
        current_agent = state["current_agent"]
        if state["agent_statuses"][current_agent] == AgentStatus.COMPLETED:
            return "continue"
        elif state["agent_statuses"][current_agent] == AgentStatus.FAILED:
            return "error"
        else:
            return "end"
    
    def _should_continue_after_project_management(self, state: ClassificationState) -> str:
        """Determine if workflow should continue after project management."""
        if state["agent_statuses"]["project_manager"] == AgentStatus.COMPLETED:
            return "continue"
        elif state["agent_statuses"]["project_manager"] == AgentStatus.FAILED:
            return "error"
        else:
            return "end"
    
    def _get_next_node_from_project_management(self, state: ClassificationState) -> str:
        """Get the next node after project management based on current progress."""
        completed_agents = state.get("completed_agents", [])
        
        # Determine next agent based on what's been completed
        if "data_cleaning" not in completed_agents:
            return "data_discovery"
        elif "data_discovery" not in completed_agents:
            return "data_discovery"
        elif "eda_analysis" not in completed_agents:
            return "eda_analysis"
        elif "feature_engineering" not in completed_agents:
            return "feature_engineering"
        elif "ml_building" not in completed_agents:
            return "ml_building"
        elif "model_evaluation" not in completed_agents:
            return "model_evaluation"
        elif "technical_reporter" not in completed_agents:
            return "technical_reporting"
        else:
            return "workflow_completion"
    
    def _get_next_node(self, current_node: str) -> str:
        """Get the next node in the workflow."""
        node_sequence = [
            "data_cleaning", "data_discovery", "eda_analysis", 
            "feature_engineering", "ml_building", "model_evaluation", 
            "technical_reporting"
        ]
        
        try:
            current_index = node_sequence.index(current_node)
            return node_sequence[current_index + 1]
        except (ValueError, IndexError):
            return "workflow_completion"
    
    async def _check_approval_gates(self, state: ClassificationState, current_agent: str) -> ClassificationState:
        """
        Check if any approval gates should be triggered and pause workflow if needed.
        
        Args:
            state: Current workflow state
            current_agent: Current agent being executed
            
        Returns:
            Updated state with approval gate information
        """
        try:
            # Check each approval gate type
            for gate_type in ApprovalGateType:
                if should_trigger_approval_gate(current_agent, state, gate_type):
                    # Create approval proposal
                    proposal = create_approval_proposal(gate_type, state)
                    educational_explanation = generate_educational_explanation(gate_type, proposal, state)
                    
                    # Get gate definition
                    from ..workflows.approval_gates import get_approval_gate_definition
                    gate_definition = get_approval_gate_definition(gate_type)
                    
                    # Create the approval gate
                    gate = self.approval_manager.create_gate(
                        gate_type=gate_type,
                        title=gate_definition["title"],
                        description=gate_definition["description"],
                        proposal=proposal,
                        educational_explanation=educational_explanation
                    )
                    
                    # Pause workflow
                    state["workflow_status"] = WorkflowStatus.PAUSED
                    state["pending_approval_gate"] = gate["gate_id"]
                    state["approval_gate_type"] = gate_type.value
                    
                    logger.info(f"Workflow paused for approval gate: {gate['gate_id']}")
                    
                    # Emit realtime update
                    await realtime.emit(state.get("session_id"), "approval_gate_created", {
                        "gate_id": gate["gate_id"],
                        "gate_type": gate_type.value,
                        "title": gate["title"],
                        "workflow_paused": True
                    })
                    
                    return state
            
            return state
            
        except Exception as e:
            logger.error(f"Error checking approval gates: {str(e)}")
            return state
    
    async def _resume_after_approval(self, state: ClassificationState) -> ClassificationState:
        """
        Resume workflow after approval gate is resolved.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        try:
            gate_id = state.get("pending_approval_gate")
            if not gate_id:
                return state
            
            # Check if gate is resolved
            gate = self.approval_manager.get_gate(gate_id)
            if not gate or gate["status"].value == "pending":
                # Still pending, keep workflow paused
                return state
            
            # Gate resolved, resume workflow
            state["workflow_status"] = WorkflowStatus.RUNNING
            state["pending_approval_gate"] = None
            state["approval_gate_type"] = None
            
            logger.info(f"Workflow resumed after approval gate: {gate_id}")
            
            # Emit realtime update
            await realtime.emit(state.get("session_id"), "workflow_resumed", {
                "gate_id": gate_id,
                "workflow_paused": False
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error resuming after approval: {str(e)}")
            return state
    
    async def execute_workflow(
        self, 
        dataset: Any, 
        target_column: str, 
        description: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete classification workflow.
        
        Args:
            dataset: The input dataset
            target_column: Name of the target column to predict
            description: Description of the classification task
            user_id: Optional user identifier
            
        Returns:
            Dictionary containing workflow results and status
        """
        try:
            # Initialize state using StateManager
            session_id = self.workflow_id
            dataset_id = f"dataset_{self.workflow_id}"
            api_key = "dummy_key"  # Will be replaced with actual API key
            
            initial_state = state_manager.initialize_state(
                session_id=session_id,
                dataset_id=dataset_id,
                target_column=target_column,
                user_description=description,
                api_key=api_key,
                original_dataset=dataset
            )
            
            # Update initial state for workflow execution
            initial_state["workflow_status"] = WorkflowStatus.RUNNING
            initial_state["current_agent"] = "data_cleaning"
            initial_state["progress"] = 0.0
            initial_state["workflow_progress"] = 0.0
            
            # Execute workflow
            config = {"configurable": {"thread_id": self.workflow_id}}
            
            final_state = initial_state
            # Keys that should not be merged repeatedly within a step
            immutable_keys = {"session_id", "dataset_id", "current_agent"}
            async for state_update in self.app.astream(initial_state, config=config):
                if isinstance(state_update, dict):
                    # Update final state with new values
                    for key, value in state_update.items():
                        if key in immutable_keys:
                            continue
                        if key in final_state:
                            final_state[key] = value
                else:
                    # Handle tuple format (node_name, state_update)
                    if isinstance(state_update, tuple) and len(state_update) > 1:
                        node_name, node_state = state_update
                        if isinstance(node_state, dict):
                            for key, value in node_state.items():
                                if key in immutable_keys:
                                    continue
                                if key in final_state:
                                    final_state[key] = value
            
            return {
                "workflow_id": self.workflow_id,
                "status": final_state.get("workflow_status", WorkflowStatus.FAILED),
                "results": final_state.get("results", {}),
                "summary": final_state.get("workflow_summary", {}),
                "error": final_state.get("last_error")
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return {
                "workflow_id": self.workflow_id,
                "status": WorkflowStatus.FAILED,
                "error": str(e),
                "results": {},
                "summary": {}
            }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow.
        
        Args:
            workflow_id: The workflow identifier
            
        Returns:
            Dictionary containing workflow status
        """
        try:
            config = {"configurable": {"thread_id": workflow_id}}
            state = await self.app.aget_state(config)
            values = getattr(state, "values", {}) or {}
            
            return {
                "workflow_id": workflow_id,
                "status": values.get("workflow_status", WorkflowStatus.UNKNOWN),
                "progress": values.get("workflow_progress", 0.0),
                "agent_status": values.get("agent_statuses", {}),
                "current_agent": values.get("current_agent", ""),
                "error": values.get("last_error")
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {str(e)}")
            return {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.UNKNOWN,
                "error": str(e)
            }
