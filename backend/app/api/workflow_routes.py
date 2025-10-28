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
from ..services.storage import storage_service

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# Global workflow instance (in production, this should be managed by a dependency injection system)
workflow_instance = None

# In-memory storage for workflow states (replace with database later)
workflow_states = {}

# ===== EDUCATIONAL MESSAGE GENERATOR =====

def _get_educational_message(agent_name: str, status: str, state: Optional[Dict[str, Any]] = None) -> str:
    """Generate educational messages for each agent"""
    messages = {
        "data_discovery": {
            "starting": "🔍 **Data Discovery**: Analyzing your dataset structure and understanding its characteristics...",
            "completed": "✅ **Data Discovery Complete**: Found {} rows and {} columns. The system has identified the data types and target variable."
        },
        "eda_analysis": {
            "starting": "📊 **Exploratory Analysis**: Creating visualizations to understand patterns, distributions, and relationships in your data...",
            "completed": "✅ **EDA Complete**: Generated {} plots showing correlations, distributions, and outliers. This helps identify important features."
        },
        "data_cleaning": {
            "starting": "🧹 **Data Cleaning**: Checking for missing values, duplicates, and data quality issues...",
            "completed": "✅ **Cleaning Complete**: Processed dataset is ready. Quality score: {}%. Layer {} used for adaptive cleaning."
        },
        "feature_engineering": {
            "starting": "⚙️ **Feature Engineering**: Creating new features and transforming existing ones to improve model performance...",
            "completed": "✅ **Features Ready**: Engineered {} new features. These transformations help the model learn better patterns."
        },
        "ml_building": {
            "starting": "🤖 **Model Training**: Training multiple machine learning models and selecting the best one...",
            "completed": "✅ **Model Trained**: Selected **{}** with accuracy of {:.2%}. Used cross-validation for robust evaluation."
        },
        "model_evaluation": {
            "starting": "📈 **Model Evaluation**: Testing model performance on unseen data and generating metrics...",
            "completed": "✅ **Evaluation Complete**: Accuracy: {:.2%}, F1-Score: {:.2%}. Model is ready for predictions!"
        },
        "technical_reporter": {
            "starting": "📝 **Generating Report**: Creating comprehensive documentation of the entire analysis...",
            "completed": "✅ **Report Ready**: Technical documentation, notebook, and model are available for download."
        }
    }
    
    if agent_name not in messages:
        return f"{status.title()} {agent_name}"
    
    template = messages[agent_name].get(status, f"{status} {agent_name}")
    
    # Fill in dynamic values for completed messages
    if status == "completed" and state:
        try:
            if agent_name == "data_discovery":
                shape = state.get("dataset_shape", [0, 0])
                return template.format(shape[0] if len(shape) > 0 else 0, shape[1] if len(shape) > 1 else 0)
            elif agent_name == "eda_analysis":
                plots = len(state.get("eda_plots", []))
                return template.format(plots)
            elif agent_name == "data_cleaning":
                quality_score = state.get("data_quality_score")
                quality = (quality_score * 100) if quality_score is not None else 0
                layer = state.get("layer_usage", {}).get("data_cleaning", "Layer 1")
                return template.format(int(quality), layer)
            elif agent_name == "feature_engineering":
                features = len(state.get("engineered_features") or [])
                return template.format(features)
            elif agent_name == "ml_building":
                model = state.get("best_model") or "Unknown"
                accuracy = (state.get("training_metrics") or {}).get("test_accuracy", 0)
                return template.format(model, accuracy)
            elif agent_name == "model_evaluation":
                metrics = state.get("evaluation_metrics") or {}
                accuracy = metrics.get("accuracy", 0) or 0
                f1 = metrics.get("f1_weighted", 0) or 0
                return template.format(accuracy, f1)
        except Exception as e:
            logger.warning(f"Error formatting educational message: {e}")
    
    return template


async def _generate_pm_answer(question: str, state: Dict[str, Any]) -> str:
    """Generate context-aware answer to user's question"""
    question_lower = question.lower()
    
    # Status questions
    if any(word in question_lower for word in ["status", "progress", "how far", "when done"]):
        completed = len(state.get("completed_agents", []))
        total = 7
        progress = (completed / total) * 100
        current = state.get("current_agent", "Unknown")
        return f"📊 **Current Progress**: {progress:.0f}% complete ({completed}/{total} agents finished). Currently executing: **{current}**. Estimated completion in ~{(total-completed)*2} minutes."
    
    # Agent-specific questions
    elif "eda" in question_lower or "exploratory" in question_lower or "visualization" in question_lower:
        plots = len(state.get("eda_plots", []))
        if plots > 0:
            return f"📊 **EDA Status**: Generated {plots} visualizations including correlation heatmaps, distribution plots, and outlier analysis. These help identify patterns and relationships in your data before modeling."
        return "📊 **EDA**: Exploratory Data Analysis creates visualizations to understand your data's structure, distributions, correlations, and potential issues. This step is crucial for feature engineering and model selection."
    
    # Data cleaning questions
    elif any(word in question_lower for word in ["clean", "quality", "missing", "duplicate"]):
        quality = state.get("data_quality_score", 0)
        if quality > 0:
            return f"🧹 **Data Quality**: Quality score is {quality*100:.1f}%. Addressed missing values, duplicates, and outliers. Clean data leads to better model performance."
        return "🧹 **Data Cleaning**: This step handles missing values, removes duplicates, fixes data types, and addresses outliers to ensure high-quality input for modeling."
    
    # Model questions
    elif any(word in question_lower for word in ["model", "algorithm", "accuracy", "performance"]):
        model = state.get("best_model", "")
        metrics = state.get("evaluation_metrics", {})
        if model and metrics:
            acc = metrics.get("accuracy", 0)
            return f"🤖 **Model**: Selected **{model}** with {acc*100:.1f}% accuracy. The system tested multiple algorithms and chose the best performer using cross-validation."
        return "🤖 **Model Training**: The system automatically tests multiple ML algorithms (Random Forest, XGBoost, etc.) and selects the best one based on cross-validation performance."
    
    # Feature engineering questions
    elif "feature" in question_lower:
        features = state.get("engineered_features", [])
        if features:
            return f"⚙️ **Feature Engineering**: Created {len(features)} new features through transformations, combinations, and domain-specific engineering. This improves model learning capacity."
        return "⚙️ **Feature Engineering**: Creates new features from existing ones through mathematical transformations, combinations, and encodings to help the model learn better patterns."
    
    # Layer 1/2 questions
    elif any(word in question_lower for word in ["layer", "sandbox", "llm", "ai"]):
        return "🏗️ **Double-Layer Architecture**: Layer 1 provides reliable, hardcoded analysis. Layer 2 uses LLM-generated code executed in a secure Docker sandbox for adaptive, dataset-specific insights. Both layers work together for robust results."
    
    # Download/results questions
    elif any(word in question_lower for word in ["download", "notebook", "report", "export"]):
        if state.get("workflow_status") == "completed":
            return "📥 **Downloads**: Once complete, you can download the analysis notebook (.ipynb), trained model (.joblib), technical report, and cleaned dataset from the Results page."
        return "📥 **Downloads**: After workflow completion, you'll receive a Jupyter notebook with full analysis, the trained model, technical documentation, and the cleaned dataset."
    
    # General/default
    else:
        completed_agents = state.get("completed_agents", [])
        agent_summary = ", ".join(completed_agents[-3:]) if completed_agents else "just starting"
        return f"🤔 I'm here to help! I can answer questions about workflow progress, agent status, model performance, data quality, and more. So far, we've completed: {agent_summary}. What would you like to know?"


def _should_trigger_approval_gate(agent_name: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Determine if an approval gate should be triggered after this agent"""
    approval_gates = {
        "eda_analysis": {
            "stage": "After EDA",
            "title": "Review Data Insights",
            "question": "📊 EDA revealed data patterns and potential issues. Would you like to proceed with data cleaning?",
            "context": {
                "plots_generated": len(state.get("eda_plots", [])),
                "correlations_found": bool(state.get("correlation_analysis")),
                "outliers_detected": bool(state.get("outlier_analysis"))
            },
            "options": ["approve", "reject", "modify"],
            "default_action": "approve"
        },
        "data_cleaning": {
            "stage": "After Data Cleaning",
            "title": "Confirm Data Quality",
            "question": f"🧹 Data cleaning complete with quality score: {(state.get('data_quality_score') or 0)*100:.1f}%. Proceed to feature engineering?",
            "context": {
                "quality_score": state.get("data_quality_score") or 0,
                "actions_taken": len(state.get("cleaning_actions_taken") or []),
                "issues_found": len(state.get("cleaning_issues_found") or [])
            },
            "options": ["approve", "reject"],
            "default_action": "approve"
        },
        "feature_engineering": {
            "stage": "Before Model Training",
            "title": "Ready to Train Models",
            "question": f"⚙️ Created {len(state.get('engineered_features') or [])} new features. Start model training?",
            "context": {
                "features_created": len(state.get("engineered_features") or []),
                "total_features": (state.get("dataset_shape") or [0, 0])[1]
            },
            "options": ["approve", "reject", "modify"],
            "default_action": "approve"
        }
    }
    
    return approval_gates.get(agent_name)

# ===== SANDBOX METRICS PARSING HELPERS =====

def parse_cpu_percentage(cpu_data: Dict[str, Any]) -> float:
    """Parse CPU percentage from sandbox data."""
    try:
        percentage_str = cpu_data.get("percentage", "0%")
        # Remove % and convert to float
        return float(percentage_str.rstrip('%'))
    except Exception as e:
        logger.warning(f"Failed to parse CPU percentage: {e}")
        return 0.0

def parse_memory_mb(memory_str: str) -> float:
    """Convert memory string to MB."""
    try:
        memory_str = memory_str.upper()
        # Extract numeric value
        value = float(''.join(c for c in memory_str if c.isdigit() or c == '.'))
        
        if "GIB" in memory_str or "GB" in memory_str:
            return value * 1024
        elif "MIB" in memory_str or "MB" in memory_str:
            return value
        elif "KIB" in memory_str or "KB" in memory_str:
            return value / 1024
        return value
    except Exception as e:
        logger.warning(f"Failed to parse memory string '{memory_str}': {e}")
        return 0.0

def parse_memory_percentage(memory_data: Dict[str, Any]) -> float:
    """Parse memory percentage from sandbox data."""
    try:
        # Parse "125MiB / 2GiB" format
        current_str = memory_data.get("current", "0MiB")
        limit_str = memory_data.get("limit", "2048MiB")
        
        current_mb = parse_memory_mb(current_str)
        limit_mb = parse_memory_mb(limit_str)
        
        if limit_mb > 0:
            return (current_mb / limit_mb) * 100
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to parse memory percentage: {e}")
        return 0.0

# ===== END HELPERS =====

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
            "workflow_status": "running",  # Add this field for consistency
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
            "failed_agents": [],
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
        
        # Parse sandbox metrics if available
        sandbox_metrics_raw = state.get("sandbox_metrics", {})
        sandbox_metrics = {
            "cpu": parse_cpu_percentage(sandbox_metrics_raw.get("cpu_usage", {})),
            "memory": parse_memory_percentage(sandbox_metrics_raw.get("memory_usage", {})),
            "time": sandbox_metrics_raw.get("execution_time", 0),
            "current_agent": sandbox_metrics_raw.get("current_agent", "")
        }
        
        return {
            "workflow_id": workflow_id,
            "status": state.get("status", WorkflowStatus.UNKNOWN),
            "progress": state.get("progress", 0.0),
            "current_phase": state.get("current_agent", "Unknown"),
            "agent_status": state.get("agent_statuses", {}),
            "completed_agents": state.get("completed_agents", []),
            "errors": state.get("errors", []),
            "message": f"Workflow is {state.get('status', 'unknown')}",
            "sandbox_metrics": sandbox_metrics,  # ✅ Real-time sandbox metrics
            "layer_usage": state.get("layer_usage", {}),  # ✅ Layer 1/2 info per agent
            "current_agent_details": {
                "name": state.get("current_agent", ""),
                "status": state.get("agent_statuses", {}).get(state.get("current_agent", ""), "unknown"),
                "layer": state.get("layer_usage", {}).get(state.get("current_agent", ""), "Layer 1")
            },
            "pm_messages": state.get("pm_messages", []),  # ✅ Project Manager messages
            "pending_approval": state.get("pending_approval", None),  # ✅ Approval gate info
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
                # Top-level feature importance for frontend convenience
                "feature_importance": state.get("feature_importance", {}),
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
    Expects format: workflow_id/filename.png
    
    Args:
        plot_path: Path to the plot image (e.g., "e8d7ec01.../correlation_heatmap.png")
        
    Returns:
        Plot image file
    """
    try:
        # Construct full path from backend/plots/ directory
        base_plots_dir = Path("backend/plots")
        full_path = base_plots_dir / plot_path
        
        logger.info(f"Attempting to serve plot from: {full_path}")
        
        if not full_path.exists():
            logger.error(f"Plot not found at: {full_path}")
            raise HTTPException(status_code=404, detail=f"Plot not found: {plot_path}")
        
        # Determine media type
        if full_path.suffix.lower() == '.png':
            media_type = "image/png"
        elif full_path.suffix.lower() == '.jpg' or full_path.suffix.lower() == '.jpeg':
            media_type = "image/jpeg"
        elif full_path.suffix.lower() == '.svg':
            media_type = "image/svg+xml"
        else:
            media_type = "image/png"
        
        logger.info(f"✅ Serving plot: {full_path} ({media_type})")
        
        return FileResponse(
            path=str(full_path),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
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
        file_type: Type of file to download (cleaned_dataset, model, notebook, report, plots)
        
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
        
        # Use storage service to get file path
        file_path = storage_service.get_file_path(workflow_id, file_type)
        
        if not file_path:
            raise HTTPException(status_code=404, detail=f"{file_type} file not found")
        
        # Set appropriate filename and media type
        filename = f"{file_type}_{workflow_id}"
        media_type = "application/octet-stream"
        
        if file_type == "cleaned_dataset":
            filename += ".csv"
            media_type = "text/csv"
        elif file_type == "model":
            filename += ".joblib"
            media_type = "application/octet-stream"
        elif file_type == "notebook":
            filename += ".ipynb"
            media_type = "application/x-ipynb+json"
        elif file_type == "report":
            filename += ".md"
            media_type = "text/markdown"
        elif file_type == "plots":
            # For plots, we'll return the first plot or create a zip
            plots = storage_service.get_workflow_files(workflow_id).get("plots", [])
            if not plots:
                raise HTTPException(status_code=404, detail="No plots found")
            
            # If multiple plots, create a zip file
            if len(plots) > 1:
                import zipfile
                import tempfile
                
                temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                    for plot_path in plots:
                        zipf.write(plot_path, Path(plot_path).name)
                
                file_path = temp_zip.name
                filename = f"plots_{workflow_id}.zip"
                media_type = "application/zip"
            else:
                file_path = plots[0]
                filename = f"plot_{workflow_id}.png"
                media_type = "image/png"
        
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Use: cleaned_dataset, model, notebook, report, plots")
        
        if not Path(file_path).exists():
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


@router.post("/{workflow_id}/pm/question")
async def ask_pm_question(
    workflow_id: str,
    request: Dict[str, str]
) -> Dict[str, Any]:
    """
    Ask Project Manager a question about the workflow.
    
    Args:
        workflow_id: The workflow identifier
        request: Dictionary with 'question' key
        
    Returns:
        PM's answer
    """
    try:
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        question = request.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        state = workflow_states[workflow_id]
        
        # Generate context-aware answer
        answer = await _generate_pm_answer(question, state)
        
        # Store Q&A in state
        if "pm_qa_history" not in state:
            state["pm_qa_history"] = []
        
        qa_entry = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        state["pm_qa_history"].append(qa_entry)
        
        # Add to PM messages
        if "pm_messages" not in state:
            state["pm_messages"] = []
        
        state["pm_messages"].append({
            "type": "question",
            "content": f"**You asked**: {question}",
            "timestamp": datetime.now().isoformat()
        })
        state["pm_messages"].append({
            "type": "answer",
            "content": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "workflow_id": workflow_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling PM question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@router.post("/{workflow_id}/pm/approval")
async def respond_to_approval_gate(
    workflow_id: str,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Respond to an approval gate (approve/reject/modify).
    
    Args:
        workflow_id: The workflow identifier
        request: Dictionary with 'action' ('approve'/'reject'/'modify') and optional 'feedback'
        
    Returns:
        Approval status
    """
    try:
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        state = workflow_states[workflow_id]
        action = request.get("action")
        feedback = request.get("feedback", "")
        
        if not action or action not in ["approve", "reject", "modify"]:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'approve', 'reject', or 'modify'")
        
        # Check if there's a pending approval
        if not state.get("pending_approval"):
            raise HTTPException(status_code=400, detail="No pending approval gate")
        
        approval_gate = state["pending_approval"]
        
        # Record approval response
        approval_gate["response"] = {
            "action": action,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in approval history
        if "approval_history" not in state:
            state["approval_history"] = []
        state["approval_history"].append(approval_gate)
        
        # Clear pending approval
        state["pending_approval"] = None
        
        # Add PM message about approval
        if "pm_messages" not in state:
            state["pm_messages"] = []
        
        action_emoji = {"approve": "✅", "reject": "❌", "modify": "✏️"}
        state["pm_messages"].append({
            "type": "approval_response",
            "content": f"{action_emoji.get(action, '📝')} **{action.title()}d**: {approval_gate['stage']}. {feedback if feedback else ''}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Resume workflow if approved
        if action == "approve":
            state["workflow_paused"] = False
            logger.info(f"Workflow {workflow_id} resumed after approval")
        elif action == "reject":
            state["workflow_status"] = "rejected"
            state["errors"].append(f"User rejected at {approval_gate['stage']}: {feedback}")
            logger.info(f"Workflow {workflow_id} rejected by user")
        
        return {
            "workflow_id": workflow_id,
            "action": action,
            "stage": approval_gate["stage"],
            "message": f"Approval gate {action}d successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling approval response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process approval: {str(e)}")


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
        from ..agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
        from ..agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
        from ..agents.data_analysis.eda_agent import EDAAgent
        from ..agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
        from ..agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
        from ..agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
        from ..agents.reporting.technical_reporter_agent import TechnicalReporterAgent
        
        # Initialize agents in correct order:
        # 1. Data Discovery - Understand dataset structure first
        # 2. EDA - Analyze distributions, correlations before cleaning
        # 3. Data Cleaning - Clean based on insights from EDA
        # 4. Feature Engineering - Create features from clean data
        # 5. ML Building - Train models
        # 6. Model Evaluation - Evaluate performance
        # 7. Technical Reporter - Generate final report
        agents = {
            "data_discovery": DataDiscoveryAgent(),
            "eda_analysis": EDAAgent(),
            "data_cleaning": EnhancedDataCleaningAgent(),
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
                
                # ✅ ADD: Send educational message before agent execution
                educational_msg = _get_educational_message(agent_name, "starting")
                try:
                    from ..services.realtime import emit
                    await emit(workflow_id, "pm_message", {
                        "agent": agent_name,
                        "message": educational_msg,
                        "type": "info",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to emit PM message: {e}")
                
                # Execute the actual agent
                current_state = await agent.execute(current_state)
                logger.info(f"Completed agent: {agent_name} for workflow {workflow_id}")
                
                # ✅ CRITICAL FIX: Copy ALL state keys to workflow_states
                # This ensures results from agents are accessible via the API
                logger.info(f"📦 Copying state keys to workflow_states for {agent_name}")
                copied_keys = []
                skip_keys = {"session_id", "user_id", "workflow_status", "api_key", "original_dataset", "dataset", "processed_dataset"}
                
                for key, value in current_state.items():
                    if key not in skip_keys and value is not None:
                        workflow_states[workflow_id][key] = value
                        copied_keys.append(key)
                
                logger.info(f"📦 Copied {len(copied_keys)} keys from {agent_name}: {', '.join(copied_keys[:5])}{'...' if len(copied_keys) > 5 else ''}")
                
                # ✅ ADD: Send completion message with insights
                completion_msg = _get_educational_message(agent_name, "completed", current_state)
                try:
                    from ..services.realtime import emit
                    await emit(workflow_id, "pm_message", {
                        "agent": agent_name,
                        "message": completion_msg,
                        "type": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to emit PM completion message: {e}")
                
                # Store PM message in state
                if "pm_messages" not in workflow_states[workflow_id]:
                    workflow_states[workflow_id]["pm_messages"] = []
                workflow_states[workflow_id]["pm_messages"].append({
                    "type": "completion",
                    "agent": agent_name,
                    "content": completion_msg,
                    "timestamp": datetime.now().isoformat()
                })
                
                # ✅ ADD: Check for approval gates
                approval_gate = _should_trigger_approval_gate(agent_name, current_state)
                if approval_gate:
                    logger.info(f"Triggering approval gate at {approval_gate['stage']}")
                    
                    # Set pending approval in state
                    approval_gate["timestamp"] = datetime.now().isoformat()
                    approval_gate["agent"] = agent_name
                    workflow_states[workflow_id]["pending_approval"] = approval_gate
                    workflow_states[workflow_id]["workflow_paused"] = True
                    
                    # Emit approval gate event
                    try:
                        from ..services.realtime import emit
                        await emit(workflow_id, "approval_required", approval_gate)
                    except Exception as e:
                        logger.warning(f"Failed to emit approval gate event: {e}")
                    
                    # Add to PM messages
                    workflow_states[workflow_id]["pm_messages"].append({
                        "type": "approval_gate",
                        "content": f"⏸️ **{approval_gate['title']}**: {approval_gate['question']}",
                        "approval_data": approval_gate,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Wait for approval (with timeout)
                    wait_time = 0
                    max_wait = 300  # 5 minutes
                    while workflow_states[workflow_id].get("workflow_paused") and wait_time < max_wait:
                        await asyncio.sleep(1)
                        wait_time += 1
                    
                    # Check if approved or timed out
                    if workflow_states[workflow_id].get("workflow_paused"):
                        logger.warning(f"Approval gate timed out at {approval_gate['stage']}")
                        # Auto-approve on timeout
                        workflow_states[workflow_id]["workflow_paused"] = False
                        workflow_states[workflow_id]["pending_approval"] = None
                        workflow_states[workflow_id]["pm_messages"].append({
                            "type": "timeout",
                            "content": "⏰ **Auto-approved**: No response received, continuing workflow automatically.",
                            "timestamp": datetime.now().isoformat()
                        })
                    elif workflow_states[workflow_id].get("workflow_status") == "rejected":
                        logger.info(f"Workflow rejected by user at {approval_gate['stage']}")
                        break  # Exit agent loop
                
                # ✅ FIX: Extract and store sandbox metrics from Layer 2 execution
                if "layer2_execution_metrics" in current_state:
                    metrics = current_state["layer2_execution_metrics"]
                    workflow_states[workflow_id]["sandbox_metrics"] = {
                        "cpu_usage": metrics.get("cpu_usage", {}),
                        "memory_usage": metrics.get("memory_usage", {}),
                        "execution_time": metrics.get("execution_time", 0),
                        "current_agent": agent_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"📊 Sandbox metrics stored for {agent_name}: CPU={metrics.get('cpu_usage')}, Memory={metrics.get('memory_usage')}")
                
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
            workflow_states[workflow_id]["workflow_status"] = "completed"  # Add this field for results endpoint
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
        # Use the same model as the LLM service
        model = genai.GenerativeModel('models/gemini-flash-latest')
        
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
