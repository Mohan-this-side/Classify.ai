"""
Project Manager Agent

This agent is responsible for:
- Coordinating workflow and providing user updates
- Tracking progress across all agents
- Generating educational insights for users
- Handling error communication
- Providing status summaries

As specified in the PRD, this is Agent #8 in the multi-agent system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..base_agent import BaseAgent, AgentResult
from ...workflows.state_management import ClassificationState, AgentStatus, WorkflowStatus
from ...services import realtime


class ProjectManagerAgent(BaseAgent):
    """
    Project Manager Agent for workflow coordination and user communication.
    
    This agent acts as the central coordinator, providing updates to users
    and managing the overall workflow progress. It generates educational
    insights and handles error communication.
    """
    
    def __init__(self):
        super().__init__("project_manager", "1.0.0")
        self.logger = logging.getLogger("agent.project_manager")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Project Manager Agent for workflow coordination and user communication",
            "capabilities": [
                "Workflow progress tracking",
                "User communication and updates",
                "Educational insights generation",
                "Error communication and handling",
                "Status summaries and reporting",
                "Real-time progress updates"
            ],
            "dependencies": ["data_cleaning", "data_discovery", "eda_analysis", "feature_engineering", "ml_building", "model_evaluation", "technical_reporter"]
        }
    
    def get_dependencies(self) -> list:
        """Get list of agent dependencies"""
        return ["data_cleaning", "data_discovery", "eda_analysis", "feature_engineering", "ml_building", "model_evaluation", "technical_reporter"]
    
    async def execute(self, state: ClassificationState) -> ClassificationState:
        """
        Execute the Project Manager Agent's main logic.
        
        This agent coordinates workflow progress and provides user updates.
        It can be called at any point in the workflow to provide status updates.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with project management information
        """
        try:
            self.logger.info("Starting Project Manager Agent execution")
            start_time = datetime.now()
            
            # Update current agent in state
            state["current_agent"] = self.agent_name
            state["agent_statuses"][self.agent_name] = AgentStatus.RUNNING
            
            # Generate progress summary
            progress_summary = await self._generate_progress_summary(state)
            
            # Generate educational insights
            educational_insights = await self._generate_educational_insights(state)
            
            # Generate status report
            status_report = await self._generate_status_report(state)
            
            # Check for any errors that need user attention
            error_summary = await self._check_for_errors(state)
            
            # Emit real-time update
            await self._emit_progress_update(state, progress_summary, educational_insights)
            
            # Update state with project management results
            state["project_management"] = {
                "progress_summary": progress_summary,
                "educational_insights": educational_insights,
                "status_report": status_report,
                "error_summary": error_summary,
                "last_updated": datetime.now().isoformat(),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Mark agent as completed
            state["agent_statuses"][self.agent_name] = AgentStatus.COMPLETED
            state["completed_agents"].append(self.agent_name)
            
            self.logger.info("Project Manager Agent execution completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in Project Manager Agent execution: {str(e)}")
            state["agent_statuses"][self.agent_name] = AgentStatus.FAILED
            state["failed_agents"].append(self.agent_name)
            state["errors"].append(f"Project Manager Agent failed: {str(e)}")
            return state
    
    async def _generate_progress_summary(self, state: ClassificationState) -> Dict[str, Any]:
        """Generate a comprehensive progress summary"""
        try:
            completed_agents = state.get("completed_agents", [])
            failed_agents = state.get("failed_agents", [])
            total_agents = 8  # Total number of agents in the system
            
            progress_percentage = (len(completed_agents) / total_agents) * 100
            
            # Calculate estimated time remaining
            estimated_remaining = self._estimate_remaining_time(state)
            
            # Get current phase description
            current_phase = self._get_current_phase_description(state)
            
            return {
                "overall_progress": progress_percentage,
                "completed_agents": completed_agents,
                "failed_agents": failed_agents,
                "current_phase": current_phase,
                "estimated_remaining_time": estimated_remaining,
                "total_agents": total_agents,
                "workflow_status": state.get("workflow_status", WorkflowStatus.UNKNOWN).value
            }
            
        except Exception as e:
            self.logger.error(f"Error generating progress summary: {str(e)}")
            return {"error": f"Failed to generate progress summary: {str(e)}"}
    
    async def _generate_educational_insights(self, state: ClassificationState) -> List[Dict[str, Any]]:
        """Generate educational insights based on current workflow state"""
        try:
            insights = []
            current_agent = state.get("current_agent", "")
            completed_agents = state.get("completed_agents", [])
            
            # Data cleaning insights
            if "data_cleaning" in completed_agents:
                cleaning_issues = state.get("cleaning_issues_found", [])
                if cleaning_issues:
                    insights.append({
                        "category": "Data Quality",
                        "title": "Data Cleaning Completed",
                        "message": f"Found and addressed {len(cleaning_issues)} data quality issues. This step is crucial for model accuracy.",
                        "importance": "high"
                    })
            
            # EDA insights
            if "eda_analysis" in completed_agents:
                eda_plots = state.get("eda_plots", [])
                insights.append({
                    "category": "Data Understanding",
                    "title": "Exploratory Analysis Complete",
                    "message": f"Generated {len(eda_plots)} visualizations to understand your data patterns and relationships.",
                    "importance": "medium"
                })
            
            # Feature engineering insights
            if "feature_engineering" in completed_agents:
                engineered_features = state.get("engineered_features", [])
                if engineered_features:
                    insights.append({
                        "category": "Feature Engineering",
                        "title": "New Features Created",
                        "message": f"Created {len(engineered_features)} new features to improve model performance.",
                        "importance": "high"
                    })
            
            # Model building insights
            if "ml_building" in completed_agents:
                model_info = state.get("model_selection_results", {})
                if model_info:
                    best_model = model_info.get("best_model_name", "Unknown")
                    insights.append({
                        "category": "Model Training",
                        "title": "Model Training Complete",
                        "message": f"Successfully trained {best_model} model. The system automatically selected the best performing algorithm.",
                        "importance": "high"
                    })
            
            # Model evaluation insights
            if "model_evaluation" in completed_agents:
                performance = state.get("model_performance_analysis", {})
                if isinstance(performance, dict):
                    accuracy = performance.get("accuracy", 0)
                    insights.append({
                        "category": "Model Performance",
                        "title": "Model Evaluation Complete",
                        "message": f"Your model achieved {accuracy:.2%} accuracy. This indicates how well the model predicts the target variable.",
                        "importance": "high"
                    })
            
            # General workflow insights
            if current_agent == "data_cleaning":
                insights.append({
                    "category": "Workflow",
                    "title": "Starting Data Analysis",
                    "message": "The system is now analyzing your dataset to understand its structure and quality. This is the foundation of good machine learning.",
                    "importance": "medium"
                })
            elif current_agent == "ml_building":
                insights.append({
                    "category": "Workflow",
                    "title": "Training Machine Learning Model",
                    "message": "The system is now training multiple machine learning algorithms to find the best one for your data. This may take a few minutes.",
                    "importance": "medium"
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating educational insights: {str(e)}")
            return [{"error": f"Failed to generate insights: {str(e)}"}]
    
    async def _generate_status_report(self, state: ClassificationState) -> Dict[str, Any]:
        """Generate a detailed status report"""
        try:
            return {
                "workflow_id": state.get("session_id", "unknown"),
                "dataset_info": {
                    "shape": state.get("dataset_shape", "unknown"),
                    "target_column": state.get("target_column", "unknown"),
                    "data_quality_score": state.get("data_quality_score", "not_available")
                },
                "agent_statuses": state.get("agent_statuses", {}),
                "workflow_status": state.get("workflow_status", WorkflowStatus.UNKNOWN).value,
                "errors": state.get("errors", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating status report: {str(e)}")
            return {"error": f"Failed to generate status report: {str(e)}"}
    
    async def _check_for_errors(self, state: ClassificationState) -> Dict[str, Any]:
        """Check for errors that need user attention"""
        try:
            errors = state.get("errors", [])
            failed_agents = state.get("failed_agents", [])
            
            if not errors and not failed_agents:
                return {"has_errors": False, "message": "No errors detected"}
            
            return {
                "has_errors": True,
                "error_count": len(errors),
                "failed_agents": failed_agents,
                "errors": errors,
                "message": f"Found {len(errors)} errors and {len(failed_agents)} failed agents"
            }
            
        except Exception as e:
            self.logger.error(f"Error checking for errors: {str(e)}")
            return {"error": f"Failed to check for errors: {str(e)}"}
    
    def _estimate_remaining_time(self, state: ClassificationState) -> str:
        """Estimate remaining time for workflow completion"""
        try:
            completed_agents = state.get("completed_agents", [])
            total_agents = 8
            
            if len(completed_agents) == 0:
                return "5-10 minutes"
            elif len(completed_agents) < 3:
                return "4-8 minutes"
            elif len(completed_agents) < 6:
                return "2-5 minutes"
            elif len(completed_agents) < 8:
                return "1-2 minutes"
            else:
                return "Almost complete"
                
        except Exception:
            return "Unknown"
    
    def _get_current_phase_description(self, state: ClassificationState) -> str:
        """Get description of current workflow phase"""
        try:
            current_agent = state.get("current_agent", "")
            completed_agents = state.get("completed_agents", [])
            
            if current_agent == "data_cleaning":
                return "Cleaning and preparing your data"
            elif current_agent == "data_discovery":
                return "Analyzing data characteristics and patterns"
            elif current_agent == "eda_analysis":
                return "Exploring data through visualizations and statistics"
            elif current_agent == "feature_engineering":
                return "Creating and selecting optimal features"
            elif current_agent == "ml_building":
                return "Training machine learning models"
            elif current_agent == "model_evaluation":
                return "Evaluating model performance"
            elif current_agent == "technical_reporter":
                return "Generating final reports and documentation"
            elif current_agent == "project_manager":
                return "Coordinating workflow progress"
            elif len(completed_agents) == 8:
                return "Workflow completed successfully"
            else:
                return "Initializing workflow"
                
        except Exception:
            return "Processing"
    
    async def _emit_progress_update(self, state: ClassificationState, progress_summary: Dict[str, Any], insights: List[Dict[str, Any]]):
        """Emit real-time progress update via WebSocket"""
        try:
            session_id = state.get("session_id")
            if not session_id:
                return
            
            update_data = {
                "agent": self.agent_name,
                "progress_summary": progress_summary,
                "educational_insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
            await realtime.emit(session_id, "project_manager_update", update_data)
            
        except Exception as e:
            self.logger.error(f"Error emitting progress update: {str(e)}")
    
    async def get_workflow_summary(self, state: ClassificationState) -> Dict[str, Any]:
        """
        Get a comprehensive workflow summary for external use.
        
        This method can be called by other components to get the current
        status of the workflow without executing the full agent.
        """
        try:
            return {
                "workflow_id": state.get("session_id", "unknown"),
                "status": state.get("workflow_status", WorkflowStatus.UNKNOWN).value,
                "progress": await self._generate_progress_summary(state),
                "insights": await self._generate_educational_insights(state),
                "errors": await self._check_for_errors(state),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting workflow summary: {str(e)}")
            return {"error": f"Failed to get workflow summary: {str(e)}"}
