"""
Error Handling Utilities

This module provides comprehensive error handling utilities for the
multi-agent classification system, including retry logic, error recovery,
and user-friendly error messages.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, Type, Union
from datetime import datetime, timedelta
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    VALIDATION = "validation"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    DATA = "data"
    MODEL = "model"
    AGENT = "agent"
    WORKFLOW = "workflow"
    UNKNOWN = "unknown"


class ErrorHandler:
    """
    Comprehensive error handler for the multi-agent system.
    
    Provides retry logic, error recovery, and user-friendly error messages.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_history: Dict[str, list] = {}
    
    def classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error by category and severity.
        
        Args:
            error: The exception to classify
            
        Returns:
            Tuple of (category, severity)
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Determine category
        if any(keyword in error_message for keyword in ['validation', 'invalid', 'missing', 'required']):
            category = ErrorCategory.VALIDATION
        elif any(keyword in error_message for keyword in ['network', 'connection', 'timeout', 'unreachable']):
            category = ErrorCategory.NETWORK
        elif 'timeout' in error_message:
            category = ErrorCategory.TIMEOUT
        elif any(keyword in error_message for keyword in ['memory', 'disk', 'resource', 'quota']):
            category = ErrorCategory.RESOURCE
        elif any(keyword in error_message for keyword in ['data', 'dataset', 'file', 'format']):
            category = ErrorCategory.DATA
        elif any(keyword in error_message for keyword in ['model', 'prediction', 'training']):
            category = ErrorCategory.MODEL
        elif any(keyword in error_message for keyword in ['agent', 'workflow', 'execution']):
            category = ErrorCategory.AGENT
        else:
            category = ErrorCategory.UNKNOWN
        
        # Determine severity
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            severity = ErrorSeverity.MEDIUM
        elif error_type in ['ConnectionError', 'TimeoutError']:
            severity = ErrorSeverity.HIGH
        elif error_type in ['MemoryError', 'OSError']:
            severity = ErrorSeverity.CRITICAL
        elif 'critical' in error_message or 'fatal' in error_message:
            severity = ErrorSeverity.CRITICAL
        elif 'warning' in error_message or 'minor' in error_message:
            severity = ErrorSeverity.LOW
        else:
            severity = ErrorSeverity.MEDIUM
        
        return category, severity
    
    def create_user_friendly_message(self, error: Exception, context: str = "") -> str:
        """
        Create a user-friendly error message.
        
        Args:
            error: The exception
            context: Additional context about where the error occurred
            
        Returns:
            User-friendly error message
        """
        category, severity = self.classify_error(error)
        error_type = type(error).__name__
        error_message = str(error)
        
        # Base message
        if category == ErrorCategory.VALIDATION:
            base_msg = "There was a problem with the data validation"
        elif category == ErrorCategory.NETWORK:
            base_msg = "There was a network connectivity issue"
        elif category == ErrorCategory.TIMEOUT:
            base_msg = "The operation took too long to complete"
        elif category == ErrorCategory.RESOURCE:
            base_msg = "There was a resource limitation issue"
        elif category == ErrorCategory.DATA:
            base_msg = "There was a problem processing the data"
        elif category == ErrorCategory.MODEL:
            base_msg = "There was a problem with the machine learning model"
        elif category == ErrorCategory.AGENT:
            base_msg = "There was a problem with an AI agent"
        elif category == ErrorCategory.WORKFLOW:
            base_msg = "There was a problem with the workflow execution"
        else:
            base_msg = "An unexpected error occurred"
        
        # Add context
        if context:
            base_msg += f" during {context}"
        
        # Add severity-based guidance
        if severity == ErrorSeverity.LOW:
            base_msg += ". This is a minor issue that should resolve itself."
        elif severity == ErrorSeverity.MEDIUM:
            base_msg += ". Please try again or check your input data."
        elif severity == ErrorSeverity.HIGH:
            base_msg += ". This may require system administrator attention."
        elif severity == ErrorSeverity.CRITICAL:
            base_msg += ". The system may need to be restarted."
        
        return base_msg
    
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        retries: Optional[int] = None,
        delay: Optional[float] = None,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        **kwargs
    ) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            retries: Number of retries (defaults to self.max_retries)
            delay: Initial delay in seconds (defaults to self.base_delay)
            backoff_factor: Factor to multiply delay by after each retry
            max_delay: Maximum delay between retries
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        if retries is None:
            retries = self.max_retries
        if delay is None:
            delay = self.base_delay
        
        last_exception = None
        current_delay = delay
        
        for attempt in range(retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                category, severity = self.classify_error(e)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{retries + 1} failed: {type(e).__name__}: {str(e)}"
                )
                
                # Don't retry for certain error types
                if category in [ErrorCategory.VALIDATION, ErrorCategory.DATA]:
                    logger.error(f"Non-retryable error: {e}")
                    raise e
                
                # Don't retry on last attempt
                if attempt == retries:
                    break
                
                # Wait before retry
                logger.info(f"Retrying in {current_delay:.2f} seconds...")
                await asyncio.sleep(current_delay)
                
                # Increase delay for next retry
                current_delay = min(current_delay * backoff_factor, max_delay)
        
        # All retries failed
        logger.error(f"All {retries + 1} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def log_error(
        self,
        error: Exception,
        context: str = "",
        agent_name: str = "",
        workflow_id: str = ""
    ) -> Dict[str, Any]:
        """
        Log error with full context and metadata.
        
        Args:
            error: The exception
            context: Additional context
            agent_name: Name of the agent where error occurred
            workflow_id: ID of the workflow
            
        Returns:
            Error metadata dictionary
        """
        category, severity = self.classify_error(error)
        error_id = f"{workflow_id}_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        error_metadata = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "workflow_id": workflow_id,
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.value,
            "severity": severity.value,
            "traceback": traceback.format_exc(),
            "user_friendly_message": self.create_user_friendly_message(error, context)
        }
        
        # Store in error history
        if workflow_id not in self.error_history:
            self.error_history[workflow_id] = []
        self.error_history[workflow_id].append(error_metadata)
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR [{error_id}]: {error_metadata['user_friendly_message']}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR [{error_id}]: {error_metadata['user_friendly_message']}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR [{error_id}]: {error_metadata['user_friendly_message']}")
        else:
            logger.info(f"LOW SEVERITY ERROR [{error_id}]: {error_metadata['user_friendly_message']}")
        
        return error_metadata
    
    def get_error_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get error summary for a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Error summary dictionary
        """
        errors = self.error_history.get(workflow_id, [])
        
        if not errors:
            return {"total_errors": 0, "errors_by_severity": {}, "errors_by_category": {}}
        
        # Count by severity
        errors_by_severity = {}
        for error in errors:
            severity = error["severity"]
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        # Count by category
        errors_by_category = {}
        for error in errors:
            category = error["category"]
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
        
        return {
            "total_errors": len(errors),
            "errors_by_severity": errors_by_severity,
            "errors_by_category": errors_by_category,
            "recent_errors": errors[-5:] if len(errors) > 5 else errors
        }
    
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """
        Determine if an error should be retried.
        
        Args:
            error: The exception
            retry_count: Current retry count
            
        Returns:
            True if should retry, False otherwise
        """
        if retry_count >= self.max_retries:
            return False
        
        category, severity = self.classify_error(error)
        
        # Don't retry validation or data errors
        if category in [ErrorCategory.VALIDATION, ErrorCategory.DATA]:
            return False
        
        # Don't retry critical errors
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        return True


# Global error handler instance
error_handler = ErrorHandler()


def handle_agent_error(
    error: Exception,
    agent_name: str,
    workflow_id: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    Handle an error that occurred in an agent.
    
    Args:
        error: The exception
        agent_name: Name of the agent
        workflow_id: ID of the workflow
        context: Additional context
        
    Returns:
        Error metadata dictionary
    """
    return error_handler.log_error(error, context, agent_name, workflow_id)


def create_error_response(
    error: Exception,
    agent_name: str = "",
    workflow_id: str = "",
    context: str = ""
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: The exception
        agent_name: Name of the agent
        workflow_id: ID of the workflow
        context: Additional context
        
    Returns:
        Standardized error response
    """
    error_metadata = error_handler.log_error(error, context, agent_name, workflow_id)
    
    return {
        "success": False,
        "error": error_metadata["user_friendly_message"],
        "error_id": error_metadata["error_id"],
        "error_type": error_metadata["error_type"],
        "severity": error_metadata["severity"],
        "category": error_metadata["category"],
        "timestamp": error_metadata["timestamp"],
        "retryable": error_handler.should_retry(error, 0)
    }
