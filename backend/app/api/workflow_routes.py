"""
API routes for workflow management and execution.

This module provides REST API endpoints for starting, monitoring, and managing
the multi-agent classification workflow.
"""

import logging
import io
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import google.generativeai as genai

from ..workflows.classification_workflow import ClassificationWorkflow
from ..workflows.state_management import WorkflowStatus
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# Global workflow instance (in production, this should be managed by a dependency injection system)
workflow_instance = None

# In-memory storage for workflow states (replace with database later)
workflow_states = {}

def get_workflow() -> ClassificationWorkflow:
    """Get or create workflow instance with lazy initialization."""
    global workflow_instance
    if workflow_instance is None:
        logger.info("Initializing ClassificationWorkflow...")
        try:
            workflow_instance = ClassificationWorkflow()
            logger.info("ClassificationWorkflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ClassificationWorkflow: {e}")
            raise HTTPException(status_code=500, detail=f"Workflow initialization failed: {str(e)}")
    return workflow_instance


@router.post("/start")
async def start_workflow(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    description: str = Form(...),
    api_key: str = Form("dummy_key"),  # Make API key optional for now
    user_id: Optional[str] = Form("web_user")
) -> Dict[str, Any]:
    """
    Start a new classification workflow.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Dataset file (CSV, Excel, etc.)
        target_column: Name of the target column to predict
        description: Description of the classification task
        user_id: Optional user identifier
        
    Returns:
        Dictionary containing workflow ID and initial status
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Read and validate dataset
        content = await file.read()
        
        if file.filename.lower().endswith('.csv'):
            # Try different encodings for CSV
            try:
                df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(content), encoding='cp1252')
        else:
            # Excel file
            df = pd.read_excel(io.BytesIO(content))
        
        # Validate target column exists
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}"
            )
        
        # Validate dataset size
        if len(df) < 10:
            raise HTTPException(
                status_code=400,
                detail="Dataset must have at least 10 rows"
            )
        
        if len(df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="Dataset must have at least 2 columns"
            )
        
        # Get workflow instance
        workflow = get_workflow()
        
        # Start workflow in background
        workflow_id = workflow.workflow_id
        
        # Store workflow info (in production, this should be stored in a database)
        workflow_info = {
            "workflow_id": workflow_id,
            "user_id": user_id,
            "filename": file.filename,
            "target_column": target_column,
            "description": description,
            "dataset_shape": df.shape,
            "start_time": datetime.now().isoformat(),
            "status": WorkflowStatus.RUNNING,
            "progress": 0.0,
            "current_agent": "data_cleaning",
            "agent_statuses": {
                "data_cleaning": "pending",
                "data_discovery": "pending", 
                "eda_analysis": "pending",
                "feature_engineering": "pending",
                "ml_building": "pending",
                "model_evaluation": "pending",
                "technical_reporter": "pending"
            },
            "completed_agents": [],
            "errors": []
        }
        
        # Store in memory
        workflow_states[workflow_id] = workflow_info
        
        # Execute workflow asynchronously
        background_tasks.add_task(
            execute_workflow_background,
            workflow,
            df,
            target_column,
            description,
            user_id,
            workflow_id,
            api_key
        )
        
        logger.info(f"Started workflow {workflow_id} for user {user_id}")
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow started successfully",
            "dataset_info": {
                "filename": file.filename,
                "shape": df.shape,
                "target_column": target_column,
                "columns": list(df.columns)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get the current status of a workflow.
    
    Args:
        workflow_id: The workflow identifier
        
    Returns:
        Dictionary containing workflow status and progress
    """
    try:
        # Get workflow state from memory
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        state = workflow_states[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": state.get("status", WorkflowStatus.UNKNOWN),
            "progress": state.get("progress", 0.0),
            "current_phase": state.get("current_agent", "Unknown"),
            "agent_status": state.get("agent_statuses", {}),
            "completed_agents": state.get("completed_agents", []),
            "errors": state.get("errors", []),
            "message": f"Workflow is {state.get('status', 'unknown')}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.get("/results/{workflow_id}")
async def get_workflow_results(workflow_id: str) -> Dict[str, Any]:
    """
    Get the results of a completed workflow.
    
    Args:
        workflow_id: The workflow identifier
        
    Returns:
        Dictionary containing workflow results
    """
    try:
        # Check if workflow exists in our state
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_state = workflow_states[workflow_id]
        
        # Check if workflow is completed
        if workflow_state.get("workflow_status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Workflow {workflow_id} is not completed yet. Current status: {workflow_state.get('workflow_status')}"
            )
        
        state = workflow_states[workflow_id]
        
        # Build comprehensive results from state
        results = {
            "workflow_id": workflow_id,
            "status": state.get("status"),
            "results": {
                "dataset_info": {
                    "shape": state.get("dataset_shape"),
                    "data_types": state.get("data_types"),
                    "missing_values": state.get("missing_values"),
                    "duplicate_count": state.get("duplicate_count")
                },
                "data_cleaning": {
                    "summary": state.get("cleaning_summary", "Data cleaning completed"),
                    "quality_score": state.get("data_quality_score", 0.0),
                    "actions_taken": state.get("cleaning_actions_taken", []),
                    "issues_found": state.get("cleaning_issues_found", [])
                },
                "data_discovery": {
                    "summary": "Data discovery completed",
                    "similar_datasets": state.get("discovery_results", {}).get("similar_datasets", []),
                    "research_papers": state.get("discovery_results", {}).get("research_papers", []),
                    "recommendations": state.get("discovery_results", {}).get("recommendations", [])
                },
                "eda_analysis": {
                    "summary": "EDA analysis completed",
                    "plots": state.get("eda_plots", []),
                    "statistical_summary": state.get("statistical_summary", {}),
                    "target_analysis": state.get("target_analysis", {}),
                    "distribution_analysis": state.get("distribution_analysis", {}),
                    "correlation_analysis": state.get("correlation_analysis", {}),
                    "missing_value_analysis": state.get("missing_value_analysis", {}),
                    "outlier_analysis": state.get("outlier_analysis", {}),
                    "feature_relationship_analysis": state.get("feature_relationship_analysis", {}),
                    "data_quality_assessment": state.get("data_quality_assessment", {}),
                    "eda_report": state.get("eda_report", "")
                },
                "feature_engineering": {
                    "summary": "Feature engineering completed",
                    "engineered_features": state.get("engineered_features", []),
                    "feature_selection_results": state.get("feature_selection_results", {}),
                    "feature_transformations": state.get("feature_transformations", {})
                },
                "ml_building": {
                    "summary": "ML model building completed",
                    "best_model": state.get("best_model", "Unknown"),
                    "model_hyperparameters": state.get("model_hyperparameters", {}),
                    "training_metrics": state.get("training_metrics", {}),
                    "cross_validation_scores": state.get("cross_validation_scores", {}),
                    "model_explanation": state.get("model_explanation", "")
                },
                "model_evaluation": {
                    "summary": "Model evaluation completed",
                    "evaluation_metrics": state.get("evaluation_metrics", {}),
                    "confusion_matrix": state.get("confusion_matrix"),
                    "roc_curve_data": state.get("roc_curve_data", {}),
                    "precision_recall_curve": state.get("precision_recall_curve", {}),
                    "feature_importance": state.get("feature_importance_model", {}),
                    "performance_analysis": state.get("model_performance_analysis", "")
                },
                "technical_reporting": {
                    "summary": "Technical reporting completed",
                    "final_report": state.get("final_report", ""),
                    "executive_summary": state.get("executive_summary", ""),
                    "technical_documentation": state.get("technical_documentation", ""),
                    "recommendations": state.get("recommendations", []),
                    "limitations": state.get("limitations", []),
                    "future_improvements": state.get("future_improvements", [])
                }
            },
            "downloads": {
                "notebook_path": state.get("notebook_path"),
                "model_path": state.get("model_path"),
                "report_path": state.get("report_path"),
                "downloadable_files": state.get("downloadable_files", [])
            },
            "execution_info": {
                "start_time": state.get("start_time"),
                "end_time": state.get("end_time"),
                "total_execution_time": state.get("total_execution_time"),
                "agent_execution_times": state.get("agent_execution_times", {}),
                "completed_agents": state.get("completed_agents", []),
                "failed_agents": state.get("failed_agents", []),
                "errors": state.get("errors", []),
                "warnings": state.get("warnings", [])
            }
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow results: {str(e)}")


@router.get("/plot/{plot_path:path}")
async def get_plot_image(plot_path: str) -> FileResponse:
    """
    Get a plot image by path.
    
    Args:
        plot_path: Path to the plot image
        
    Returns:
        Plot image file
    """
    try:
        # Construct full path
        full_path = Path(plot_path)
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Plot not found")
        
        # Determine media type
        if full_path.suffix.lower() == '.png':
            media_type = "image/png"
        elif full_path.suffix.lower() == '.jpg' or full_path.suffix.lower() == '.jpeg':
            media_type = "image/jpeg"
        elif full_path.suffix.lower() == '.svg':
            media_type = "image/svg+xml"
        else:
            media_type = "image/png"
        
        return FileResponse(
            path=str(full_path),
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving plot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve plot: {str(e)}")


@router.get("/download/{workflow_id}/{file_type}")
async def download_workflow_file(
    workflow_id: str, 
    file_type: str,
    response: Response
) -> FileResponse:
    """
    Download a specific file from a completed workflow.
    
    Args:
        workflow_id: The workflow identifier
        file_type: Type of file to download (model, notebook, report, plots)
        
    Returns:
        File response for download
    """
    try:
        # Get workflow state
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        state = workflow_states[workflow_id]
        
        # Check if workflow is completed
        if state.get("status") != WorkflowStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Workflow not completed yet")
        
        file_path = None
        filename = None
        media_type = "application/octet-stream"
        
        if file_type == "model":
            # Find model file
            model_files = list(Path("models").glob(f"*{workflow_id}*.joblib"))
            if model_files:
                file_path = model_files[0]
                filename = f"model_{workflow_id}.joblib"
                media_type = "application/octet-stream"
            else:
                raise HTTPException(status_code=404, detail="Model file not found")
        
        elif file_type == "notebook":
            # Find notebook file
            notebook_files = list(Path("notebooks").glob(f"*{workflow_id}*.ipynb"))
            if notebook_files:
                file_path = notebook_files[0]
                filename = f"notebook_{workflow_id}.ipynb"
                media_type = "application/x-ipynb+json"
            else:
                raise HTTPException(status_code=404, detail="Notebook file not found")
        
        elif file_type == "report":
            # Create report file
            report_content = state.get("final_report", "No report available")
            report_path = Path("reports") / f"report_{workflow_id}.txt"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            file_path = report_path
            filename = f"report_{workflow_id}.txt"
            media_type = "text/plain"
        
        elif file_type == "plots":
            # Create plots zip file
            import zipfile
            plots_dir = Path("plots") / workflow_id
            if plots_dir.exists():
                zip_path = Path("reports") / f"plots_{workflow_id}.zip"
                zip_path.parent.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for plot_file in plots_dir.glob("*.png"):
                        zipf.write(plot_file, plot_file.name)
                
                file_path = zip_path
                filename = f"plots_{workflow_id}.zip"
                media_type = "application/zip"
            else:
                raise HTTPException(status_code=404, detail="No plots found")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Use: model, notebook, report, plots")
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@router.get("/list")
async def list_workflows(
    user_id: Optional[str] = None,
    status: Optional[WorkflowStatus] = None,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List workflows with optional filtering.
    
    Args:
        user_id: Filter by user ID
        status: Filter by workflow status
        limit: Maximum number of workflows to return
        offset: Number of workflows to skip
        
    Returns:
        Dictionary containing list of workflows
    """
    try:
        # TODO: Implement actual database query
        # For now, return placeholder data
        workflows = [
            {
                "workflow_id": "example-workflow-1",
                "user_id": user_id or "anonymous",
                "filename": "sample_dataset.csv",
                "target_column": "target",
                "description": "Sample classification task",
                "status": WorkflowStatus.COMPLETED,
                "start_time": datetime.now().isoformat(),
                "completion_time": datetime.now().isoformat()
            }
        ]
        
        return {
            "workflows": workflows,
            "total": len(workflows),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")


@router.delete("/{workflow_id}")
async def cancel_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Cancel a running workflow.
    
    Args:
        workflow_id: The workflow identifier
        
    Returns:
        Dictionary containing cancellation status
    """
    try:
        # TODO: Implement actual workflow cancellation
        # For now, return success message
        return {
            "workflow_id": workflow_id,
            "status": "cancelled",
            "message": "Workflow cancellation requested"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")


async def execute_workflow_with_progress(
    workflow: ClassificationWorkflow,
    dataset: pd.DataFrame,
    target_column: str,
    description: str,
    user_id: Optional[str],
    workflow_id: str
) -> Dict[str, Any]:
    """
    Execute workflow with progress updates.
    
    Args:
        workflow: Workflow instance
        dataset: Input dataset
        target_column: Target column name
        description: Task description
        user_id: User identifier
        workflow_id: Workflow identifier
        
    Returns:
        Workflow execution result
    """
    try:
        # Actually execute workflow with real agents
        from ..agents.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
        from ..agents.data_discovery_agent import DataDiscoveryAgent
        from ..agents.eda_agent import EDAAgent
        from ..agents.feature_engineering_agent import FeatureEngineeringAgent
        from ..agents.ml_builder_agent import MLBuilderAgent
        from ..agents.model_evaluation_agent import ModelEvaluationAgent
        from ..agents.technical_reporter_agent import TechnicalReporterAgent
        
        # Initialize agents
        agents = {
            "data_cleaning": EnhancedDataCleaningAgent(),
            "data_discovery": DataDiscoveryAgent(),
            "eda_analysis": EDAAgent(),
            "feature_engineering": FeatureEngineeringAgent(),
            "ml_building": MLBuilderAgent(),
            "model_evaluation": ModelEvaluationAgent(),
            "technical_reporter": TechnicalReporterAgent()
        }
        
        # Create initial state
        from ..workflows.state_management import state_manager
        initial_state = state_manager.initialize_state(
            session_id=workflow_id,
            dataset_id=f"dataset_{workflow_id}",
            target_column=target_column,
            user_description=description,
            api_key="dummy_key",
            original_dataset=dataset
        )
        
        # Execute agents in sequence
        current_state = initial_state
        agent_names = list(agents.keys())
        
        for i, (agent_name, agent) in enumerate(agents.items()):
            logger.info(f"Starting agent: {agent_name} for workflow {workflow_id}")
            
            # Update current agent
            if workflow_id in workflow_states:
                workflow_states[workflow_id]["current_agent"] = agent_name
                workflow_states[workflow_id]["agent_statuses"][agent_name] = "running"
                workflow_states[workflow_id]["progress"] = (i + 1) * 14.0  # ~14% per agent
                logger.info(f"Updated workflow {workflow_id} status: {agent_name} running, progress: {(i + 1) * 14.0}%")
            
            try:
                # Emit WebSocket event for agent start
                try:
                    from ..services.realtime import emit
                    await emit(workflow_id, "agent_started", {
                        "agent": agent_name,
                        "workflow_id": workflow_id,
                        "message": f"Starting {agent_name} agent..."
                    })
                except Exception as e:
                    logger.warning(f"Failed to emit agent start event: {e}")
                
                # Execute the actual agent
                current_state = await agent.execute(current_state)
                logger.info(f"Completed agent: {agent_name} for workflow {workflow_id}")
                
                # Update workflow state with agent results
                if workflow_id in workflow_states:
                    workflow_states[workflow_id]["agent_statuses"][agent_name] = "completed"
                    workflow_states[workflow_id]["completed_agents"].append(agent_name)
                
                # Emit WebSocket event for agent completion
                try:
                    from ..services.realtime import emit
                    event_data = {
                        "agent": agent_name,
                        "workflow_id": workflow_id,
                        "message": f"Completed {agent_name} agent",
                        "progress": ((i + 1) / len(agents)) * 100
                    }
                    
                    # Include plots for EDA agent
                    if agent_name == "eda_analysis":
                        event_data["plots"] = current_state.get("eda_plots", [])
                    
                    await emit(workflow_id, "agent_completed", event_data)
                except Exception as e:
                    logger.warning(f"Failed to emit agent completion event: {e}")
                
                # Store key results in workflow state for easy access
                if agent_name == "data_cleaning":
                    workflow_states[workflow_id]["cleaning_summary"] = current_state.get("cleaning_summary")
                    workflow_states[workflow_id]["data_quality_score"] = current_state.get("data_quality_score")
                    workflow_states[workflow_id]["cleaning_actions_taken"] = current_state.get("cleaning_actions_taken", [])
                elif agent_name == "eda_analysis":
                    workflow_states[workflow_id]["eda_plots"] = current_state.get("eda_plots", [])
                    workflow_states[workflow_id]["statistical_summary"] = current_state.get("statistical_summary")
                    workflow_states[workflow_id]["target_analysis"] = current_state.get("target_analysis")
                    workflow_states[workflow_id]["distribution_analysis"] = current_state.get("distribution_analysis")
                    workflow_states[workflow_id]["correlation_analysis"] = current_state.get("correlation_analysis")
                    workflow_states[workflow_id]["missing_value_analysis"] = current_state.get("missing_value_analysis")
                    workflow_states[workflow_id]["outlier_analysis"] = current_state.get("outlier_analysis")
                    workflow_states[workflow_id]["feature_relationship_analysis"] = current_state.get("feature_relationship_analysis")
                    workflow_states[workflow_id]["data_quality_assessment"] = current_state.get("data_quality_assessment")
                    workflow_states[workflow_id]["eda_report"] = current_state.get("eda_report")
                elif agent_name == "ml_building":
                    workflow_states[workflow_id]["best_model"] = current_state.get("best_model")
                    workflow_states[workflow_id]["model_hyperparameters"] = current_state.get("model_hyperparameters")
                    workflow_states[workflow_id]["training_metrics"] = current_state.get("training_metrics")
                    workflow_states[workflow_id]["cross_validation_scores"] = current_state.get("cross_validation_scores")
                    workflow_states[workflow_id]["model_explanation"] = current_state.get("model_explanation")
                elif agent_name == "model_evaluation":
                    workflow_states[workflow_id]["evaluation_metrics"] = current_state.get("evaluation_metrics")
                    workflow_states[workflow_id]["confusion_matrix"] = current_state.get("confusion_matrix")
                    workflow_states[workflow_id]["roc_curve_data"] = current_state.get("roc_curve_data")
                    workflow_states[workflow_id]["precision_recall_curve"] = current_state.get("precision_recall_curve")
                    workflow_states[workflow_id]["feature_importance_model"] = current_state.get("feature_importance_model")
                    workflow_states[workflow_id]["model_performance_analysis"] = current_state.get("model_performance_analysis")
                elif agent_name == "technical_reporter":
                    workflow_states[workflow_id]["final_report"] = current_state.get("final_report")
                    workflow_states[workflow_id]["executive_summary"] = current_state.get("executive_summary")
                    workflow_states[workflow_id]["technical_documentation"] = current_state.get("technical_documentation")
                    workflow_states[workflow_id]["recommendations"] = current_state.get("recommendations", [])
                    workflow_states[workflow_id]["limitations"] = current_state.get("limitations", [])
                    workflow_states[workflow_id]["future_improvements"] = current_state.get("future_improvements", [])
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {str(e)}", exc_info=True)
                if workflow_id in workflow_states:
                    workflow_states[workflow_id]["agent_statuses"][agent_name] = "failed"
                    workflow_states[workflow_id]["failed_agents"].append(agent_name)
                    workflow_states[workflow_id]["errors"].append({
                        "agent": agent_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Emit WebSocket event for agent failure
                try:
                    from ..services.realtime import emit
                    await emit(workflow_id, "agent_failed", {
                        "agent": agent_name,
                        "workflow_id": workflow_id,
                        "error": str(e),
                        "message": f"Agent {agent_name} failed: {str(e)}"
                    })
                except Exception as emit_error:
                    logger.warning(f"Failed to emit agent failure event: {emit_error}")
                
                # Continue with next agent instead of failing entire workflow
                continue
        
        # Final completion
        if workflow_id in workflow_states:
            workflow_states[workflow_id]["progress"] = 100.0
            workflow_states[workflow_id]["status"] = WorkflowStatus.COMPLETED
            workflow_states[workflow_id]["end_time"] = datetime.now().isoformat()
        
        # Emit WebSocket event for workflow completion
        try:
            from ..services.realtime import emit
            await emit(workflow_id, "workflow_completed", {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.COMPLETED,
                "progress": 100.0,
                "message": "Workflow completed successfully"
            })
        except Exception as e:
            logger.warning(f"Failed to emit WebSocket event: {e}")
        
        return {
            "status": WorkflowStatus.COMPLETED,
            "results": {
                "message": "Workflow completed successfully",
                "agents_completed": len(agents),
                "total_agents": len(agents)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
        if workflow_id in workflow_states:
            workflow_states[workflow_id]["status"] = WorkflowStatus.FAILED
            workflow_states[workflow_id]["errors"].append({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "agent": workflow_states[workflow_id].get("current_agent", "unknown")
            })
        
        return {
            "status": WorkflowStatus.FAILED,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def execute_workflow_background(
    workflow: ClassificationWorkflow,
    dataset: pd.DataFrame,
    target_column: str,
    description: str,
    user_id: Optional[str],
    workflow_id: str,
    api_key: str
) -> None:
    """
    Execute workflow in background task.
    
    Args:
        workflow: Workflow instance
        dataset: Input dataset
        target_column: Target column name
        description: Task description
        user_id: User identifier
        workflow_id: Workflow identifier
        api_key: Gemini API key for LLM functionality
    """
    try:
        logger.info(f"Starting background execution of workflow {workflow_id}")
        
        # Configure Gemini API with the provided key
        if api_key and api_key != "dummy_key":
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured with user-provided key")
        else:
            logger.warning("Using default Gemini API key or no key provided")
        
        # Update status to running
        if workflow_id in workflow_states:
            workflow_states[workflow_id]["status"] = WorkflowStatus.RUNNING
            workflow_states[workflow_id]["current_agent"] = "data_cleaning"
            workflow_states[workflow_id]["agent_statuses"]["data_cleaning"] = "running"
            workflow_states[workflow_id]["progress"] = 10.0
        
        # Execute the workflow with progress updates
        result = await execute_workflow_with_progress(
            workflow, dataset, target_column, description, user_id, workflow_id
        )
        
        # Update final status
        if workflow_id in workflow_states:
            workflow_states[workflow_id]["status"] = result.get("status", WorkflowStatus.COMPLETED)
            workflow_states[workflow_id]["progress"] = 100.0
            workflow_states[workflow_id]["results"] = result.get("results", {})
        
        logger.info(f"Workflow {workflow_id} completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in background workflow execution: {str(e)}", exc_info=True)
        # Update error status
        if workflow_id in workflow_states:
            workflow_states[workflow_id]["status"] = WorkflowStatus.FAILED
            workflow_states[workflow_id]["errors"].append({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "agent": workflow_states[workflow_id].get("current_agent", "unknown")
            })


@router.post("/test-gemini-key")
async def test_gemini_api_key(request: Dict[str, str]) -> JSONResponse:
    """
    Test if a Gemini API key is valid by making a simple request.
    
    Args:
        request: Dictionary containing 'api_key' field
        
    Returns:
        JSONResponse with validation result
    """
    try:
        api_key = request.get("api_key")
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"valid": False, "error": "API key is required"}
            )
        
        # Configure Gemini with the provided API key
        genai.configure(api_key=api_key)
        
        # Test the API key by making a simple request
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Make a simple test request
        response = model.generate_content("Hello, this is a test.")
        
        if response and response.text:
            logger.info("Gemini API key validation successful")
            return JSONResponse(
                status_code=200,
                content={
                    "valid": True,
                    "message": "API key is valid and working",
                    "model": "gemini-2.0-flash"
                }
            )
        else:
            logger.warning("Gemini API key validation failed: Empty response")
            return JSONResponse(
                status_code=200,
                content={"valid": False, "error": "API key returned empty response"}
            )
            
    except Exception as e:
        logger.error(f"Gemini API key validation failed: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "valid": False, 
                "error": f"API key validation failed: {str(e)}"
            }
        )
