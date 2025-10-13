"""
🤖 Base Agent Class for DS Capstone Project

This module provides the base class that all agents inherit from.
It includes common functionality like logging, error handling, and state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging
import time
from datetime import datetime
import traceback
import asyncio
from dataclasses import dataclass

from ..workflows.state_management import ClassificationState, AgentStatus, state_manager
from ..config import settings

@dataclass
class AgentResult:
    """Result object for agent execution"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

class BaseAgent(ABC):
    """
    🤖 Base Agent Class
    
    All agents in the multi-agent system inherit from this base class.
    It provides common functionality for logging, error handling, and state management.
    """
    
    def __init__(self, agent_name: str, agent_version: str = "1.0.0"):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Configure logging
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    @abstractmethod
    async def execute(self, state: ClassificationState) -> ClassificationState:
        """
        Execute the agent's main logic
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about this agent
        
        Returns:
            Dictionary containing agent information
        """
        pass
    
    async def run(self, state: ClassificationState) -> ClassificationState:
        """
        Run the agent with error handling and logging
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.agent_name} execution")
        
        try:
            # Update agent status to running
            state = state_manager.update_agent_status(
                state, self.agent_name, AgentStatus.RUNNING
            )
            
            # Execute agent logic
            state = await self.execute(state)
            
            # Update agent status to completed
            execution_time = (datetime.now() - self.start_time).total_seconds()
            state = state_manager.update_agent_status(
                state, self.agent_name, AgentStatus.COMPLETED, execution_time
            )
            
            self.logger.info(f"Completed {self.agent_name} execution in {execution_time:.2f}s")
            
        except Exception as e:
            # Handle errors
            execution_time = (datetime.now() - self.start_time).total_seconds()
            error_message = f"Agent {self.agent_name} failed: {str(e)}"
            
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            
            # Update agent status to failed
            state = state_manager.update_agent_status(
                state, self.agent_name, AgentStatus.FAILED, execution_time
            )
            
            # Add error to state
            state = state_manager.add_error(
                state, self.agent_name, str(e), "execution_error"
            )
            
            # Check if we should retry
            if state["retry_count"] < state["max_retries"]:
                state["retry_count"] += 1
                self.logger.info(f"Retrying {self.agent_name} (attempt {state['retry_count']})")
                
                # Wait before retry
                await asyncio.sleep(2 ** state["retry_count"])  # Exponential backoff
                
                # Retry execution
                return await self.run(state)
            else:
                self.logger.error(f"Max retries exceeded for {self.agent_name}")
                state["workflow_status"] = "failed"
        
        finally:
            self.end_time = datetime.now()
        
        return state
    
    def validate_input(self, state: ClassificationState) -> Tuple[bool, Optional[str]]:
        """
        Validate input state for this agent
        
        Args:
            state: Current workflow state
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Base validation - can be overridden by subclasses
        if not state.get("session_id"):
            return False, "Missing session_id"
        
        if not state.get("dataset_id"):
            return False, "Missing dataset_id"
        
        return True, None
    
    def get_dependencies(self) -> list:
        """
        Get list of agent dependencies
        
        Returns:
            List of agent names that must complete before this agent
        """
        return []
    
    def can_run(self, state: ClassificationState) -> bool:
        """
        Check if this agent can run given the current state
        
        Args:
            state: Current workflow state
            
        Returns:
            True if agent can run, False otherwise
        """
        # Check if agent is already completed
        if state["agent_statuses"].get(self.agent_name) == AgentStatus.COMPLETED:
            return False
        
        # Check if agent is currently running
        if state["agent_statuses"].get(self.agent_name) == AgentStatus.RUNNING:
            return False
        
        # Check dependencies
        dependencies = self.get_dependencies()
        for dep in dependencies:
            if state["agent_statuses"].get(dep) != AgentStatus.COMPLETED:
                return False
        
        # Check if workflow is in a valid state
        if state["workflow_status"] in ["failed", "cancelled"]:
            return False
        
        return True
    
    def get_progress(self, state: ClassificationState) -> float:
        """
        Get progress percentage for this agent
        
        Args:
            state: Current workflow state
            
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if state["agent_statuses"].get(self.agent_name) == AgentStatus.COMPLETED:
            return 100.0
        elif state["agent_statuses"].get(self.agent_name) == AgentStatus.RUNNING:
            # Estimate progress based on execution time
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                # Assume average execution time of 60 seconds
                return min(90.0, (elapsed / 60.0) * 100.0)
            return 50.0
        else:
            return 0.0
    
    def log_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """
        Log a metric for monitoring
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
        """
        self.logger.info(f"METRIC: {metric_name}={value}{unit}")
    
    def log_performance(self, operation: str, duration: float) -> None:
        """
        Log performance metrics
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        self.logger.info(f"PERFORMANCE: {operation} took {duration:.2f}s")
    
    def create_artifact(
        self,
        artifact_type: str,
        name: str,
        data: Any,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an artifact for storage
        
        Args:
            artifact_type: Type of artifact (dataset, model, report, etc.)
            name: Name of the artifact
            data: Artifact data
            description: Optional description
            
        Returns:
            Artifact metadata
        """
        return {
            "type": artifact_type,
            "name": name,
            "data": data,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "agent": self.agent_name
        }
    
    def update_state_progress(self, state: ClassificationState, progress: float) -> ClassificationState:
        """
        Update the overall workflow progress
        
        Args:
            state: Current workflow state
            progress: Progress percentage (0.0 to 100.0)
            
        Returns:
            Updated state
        """
        state["workflow_progress"] = progress
        return state
    
    def add_quality_check(
        self,
        state: ClassificationState,
        check_name: str,
        passed: bool,
        details: Optional[str] = None
    ) -> ClassificationState:
        """
        Add a quality check result
        
        Args:
            state: Current workflow state
            check_name: Name of the quality check
            passed: Whether the check passed
            details: Optional details about the check
            
        Returns:
            Updated state
        """
        if passed:
            state["quality_checks_passed"].append(check_name)
        else:
            state["quality_checks_failed"].append(check_name)
            if details:
                state = state_manager.add_warning(
                    state, self.agent_name, f"Quality check failed: {check_name} - {details}"
                )
        
        return state
    
    def get_agent_status(self, state: ClassificationState) -> Dict[str, Any]:
        """
        Get current status of this agent
        
        Args:
            state: Current workflow state
            
        Returns:
            Agent status information
        """
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "status": state["agent_statuses"].get(self.agent_name, AgentStatus.PENDING),
            "progress": self.get_progress(state),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": state["agent_execution_times"].get(self.agent_name),
            "can_run": self.can_run(state),
            "dependencies": self.get_dependencies()
        }
