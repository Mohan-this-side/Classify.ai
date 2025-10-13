"""
🗄️ Database Models for DS Capstone Project

This module defines SQLAlchemy models for the multi-agent classification system.
It includes models for projects, agent executions, and artifacts.
"""

from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, JSON, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class Project(Base):
    """Project model representing a classification project"""
    
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=True)  # Optional user identification
    session_id = Column(String(255), unique=True, nullable=False)
    
    # Project details
    dataset_name = Column(String(255), nullable=False)
    target_column = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    api_key_hash = Column(String(255), nullable=True)  # Hashed API key for security
    
    # Status and timing
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Dataset information
    dataset_shape = Column(JSON, nullable=True)  # {"rows": 1000, "columns": 10}
    dataset_info = Column(JSON, nullable=True)  # Additional dataset metadata
    
    # Results
    final_metrics = Column(JSON, nullable=True)  # Final evaluation metrics
    best_model_info = Column(JSON, nullable=True)  # Best model details
    
    # Relationships
    agent_executions = relationship("AgentExecution", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, session_id={self.session_id}, status={self.status})>"

class AgentExecution(Base):
    """Agent execution model tracking individual agent runs"""
    
    __tablename__ = "agent_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    
    # Agent details
    agent_name = Column(String(100), nullable=False)
    agent_version = Column(String(50), nullable=True)
    
    # Execution details
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    execution_time = Column(Float, nullable=True)  # Execution time in seconds
    
    # Input and output
    input_data = Column(JSON, nullable=True)  # Input parameters and data
    output_data = Column(JSON, nullable=True)  # Output results
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Performance metrics
    memory_usage = Column(Float, nullable=True)  # Memory usage in MB
    cpu_usage = Column(Float, nullable=True)  # CPU usage percentage
    
    # Relationships
    project = relationship("Project", back_populates="agent_executions")
    
    def __repr__(self):
        return f"<AgentExecution(id={self.id}, agent={self.agent_name}, status={self.status})>"

class Artifact(Base):
    """Artifact model for storing generated files and results"""
    
    __tablename__ = "artifacts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    
    # Artifact details
    artifact_type = Column(String(50), nullable=False)  # dataset, model, report, notebook, plot
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # File information
    file_path = Column(String(500), nullable=True)  # Path to stored file
    file_size = Column(Integer, nullable=True)  # File size in bytes
    mime_type = Column(String(100), nullable=True)  # MIME type
    file_hash = Column(String(255), nullable=True)  # File hash for integrity
    
    # Metadata
    metadata = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="artifacts")
    
    def __repr__(self):
        return f"<Artifact(id={self.id}, type={self.artifact_type}, name={self.name})>"

class WorkflowState(Base):
    """Workflow state model for tracking LangGraph state"""
    
    __tablename__ = "workflow_states"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    
    # State information
    current_agent = Column(String(100), nullable=True)
    workflow_status = Column(String(50), default="initialized")
    state_data = Column(JSON, nullable=True)  # Serialized state data
    
    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(100), nullable=True)
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<WorkflowState(id={self.id}, status={self.workflow_status}, progress={self.progress_percentage}%)>"

class UserSession(Base):
    """User session model for tracking active sessions"""
    
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, nullable=False)
    
    # Session details
    user_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(Text, nullable=True)
    
    # Session state
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Session data
    session_data = Column(JSON, nullable=True)  # Additional session data
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id={self.session_id}, active={self.is_active})>"

# Pydantic models for API responses
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProjectResponse(BaseModel):
    """Pydantic model for project API responses"""
    id: str
    session_id: str
    dataset_name: str
    target_column: str
    description: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    dataset_shape: Optional[Dict[str, Any]]
    final_metrics: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True

class AgentExecutionResponse(BaseModel):
    """Pydantic model for agent execution API responses"""
    id: str
    agent_name: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time: Optional[float]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class ArtifactResponse(BaseModel):
    """Pydantic model for artifact API responses"""
    id: str
    artifact_type: str
    name: str
    description: Optional[str]
    file_path: Optional[str]
    file_size: Optional[int]
    mime_type: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class WorkflowStateResponse(BaseModel):
    """Pydantic model for workflow state API responses"""
    id: str
    current_agent: Optional[str]
    workflow_status: str
    progress_percentage: float
    current_step: Optional[str]
    total_steps: int
    completed_steps: int
    updated_at: datetime
    
    class Config:
        from_attributes = True
