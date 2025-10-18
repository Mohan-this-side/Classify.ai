"""
Utility modules for the DS Capstone Multi-Agent System.

This package contains various utility functions and classes that support
the main application functionality.
"""

from .error_handling import (
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    error_handler,
    handle_agent_error,
    create_error_response
)

__all__ = [
    'ErrorHandler',
    'ErrorSeverity', 
    'ErrorCategory',
    'error_handler',
    'handle_agent_error',
    'create_error_response'
]
