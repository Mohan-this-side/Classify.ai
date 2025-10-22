"""
Results Storage Service

This module provides functionality for storing and retrieving workflow results
in a structured manner. It handles file organization, metadata storage, and
cleanup for workflow artifacts.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResultsStorageService:
    """
    Service for managing workflow results storage and retrieval.
    
    This service provides a structured approach to storing all workflow
    deliverables in organized directories with metadata tracking.
    """
    
    def __init__(self, base_results_dir: str = "results"):
        """
        Initialize the results storage service.
        
        Args:
            base_results_dir: Base directory for storing results
        """
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        logger.info(f"Results storage initialized at: {self.base_results_dir}")
    
    def create_workflow_directory(self, workflow_id: str) -> Path:
        """
        Create a directory structure for a workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            Path to the created workflow directory
        """
        workflow_dir = self.base_results_dir / workflow_id
        workflow_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (workflow_dir / "plots").mkdir(exist_ok=True)
        (workflow_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Created workflow directory: {workflow_dir}")
        return workflow_dir
    
    def store_cleaned_dataset(self, workflow_id: str, dataset: Any, filename: str = "cleaned_dataset.csv") -> str:
        """
        Store cleaned dataset as CSV file.
        
        Args:
            workflow_id: Workflow identifier
            dataset: Pandas DataFrame or data to store
            filename: Name of the file to create
            
        Returns:
            Path to the stored file
        """
        try:
            workflow_dir = self.create_workflow_directory(workflow_id)
            file_path = workflow_dir / filename
            
            # Handle pandas DataFrame
            if hasattr(dataset, 'to_csv'):
                dataset.to_csv(file_path, index=False)
            else:
                # Handle other data types
                with open(file_path, 'w') as f:
                    f.write(str(dataset))
            
            logger.info(f"Stored cleaned dataset: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing cleaned dataset: {str(e)}")
            raise
    
    def store_model(self, workflow_id: str, model: Any, filename: str = "model.joblib") -> str:
        """
        Store trained model as joblib file.
        
        Args:
            workflow_id: Workflow identifier
            model: Trained model object
            filename: Name of the file to create
            
        Returns:
            Path to the stored file
        """
        try:
            import joblib
            
            workflow_dir = self.create_workflow_directory(workflow_id)
            file_path = workflow_dir / filename
            
            joblib.dump(model, file_path)
            
            logger.info(f"Stored model: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing model: {str(e)}")
            raise
    
    def store_notebook(self, workflow_id: str, notebook_content: str, filename: str = "notebook.ipynb") -> str:
        """
        Store Jupyter notebook.
        
        Args:
            workflow_id: Workflow identifier
            notebook_content: Notebook content as string
            filename: Name of the file to create
            
        Returns:
            Path to the stored file
        """
        try:
            workflow_dir = self.create_workflow_directory(workflow_id)
            file_path = workflow_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(notebook_content)
            
            logger.info(f"Stored notebook: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing notebook: {str(e)}")
            raise
    
    def store_report(self, workflow_id: str, report_content: str, filename: str = "report.md") -> str:
        """
        Store technical report as Markdown file.
        
        Args:
            workflow_id: Workflow identifier
            report_content: Report content as string
            filename: Name of the file to create
            
        Returns:
            Path to the stored file
        """
        try:
            workflow_dir = self.create_workflow_directory(workflow_id)
            file_path = workflow_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Stored report: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing report: {str(e)}")
            raise
    
    def store_plot(self, workflow_id: str, plot_data: bytes, filename: str) -> str:
        """
        Store plot image.
        
        Args:
            workflow_id: Workflow identifier
            plot_data: Plot data as bytes
            filename: Name of the file to create
            
        Returns:
            Path to the stored file
        """
        try:
            workflow_dir = self.create_workflow_directory(workflow_id)
            plots_dir = workflow_dir / "plots"
            file_path = plots_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(plot_data)
            
            logger.info(f"Stored plot: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing plot: {str(e)}")
            raise
    
    def store_metadata(self, workflow_id: str, metadata: Dict[str, Any]) -> str:
        """
        Store workflow metadata as JSON.
        
        Args:
            workflow_id: Workflow identifier
            metadata: Metadata dictionary
            
        Returns:
            Path to the stored metadata file
        """
        try:
            workflow_dir = self.create_workflow_directory(workflow_id)
            file_path = workflow_dir / "metadata.json"
            
            # Add timestamp
            metadata["stored_at"] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Stored metadata: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing metadata: {str(e)}")
            raise
    
    def get_workflow_files(self, workflow_id: str) -> Dict[str, str]:
        """
        Get all files for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary mapping file types to file paths
        """
        try:
            workflow_dir = self.base_results_dir / workflow_id
            
            if not workflow_dir.exists():
                return {}
            
            files = {}
            
            # Check for main deliverables
            if (workflow_dir / "cleaned_dataset.csv").exists():
                files["cleaned_dataset"] = str(workflow_dir / "cleaned_dataset.csv")
            
            if (workflow_dir / "model.joblib").exists():
                files["model"] = str(workflow_dir / "model.joblib")
            
            if (workflow_dir / "notebook.ipynb").exists():
                files["notebook"] = str(workflow_dir / "notebook.ipynb")
            
            if (workflow_dir / "report.md").exists():
                files["report"] = str(workflow_dir / "report.md")
            
            # Get plots
            plots_dir = workflow_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*"))
                files["plots"] = [str(f) for f in plot_files]
            
            # Get metadata
            if (workflow_dir / "metadata.json").exists():
                files["metadata"] = str(workflow_dir / "metadata.json")
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting workflow files: {str(e)}")
            return {}
    
    def get_file_path(self, workflow_id: str, file_type: str) -> Optional[str]:
        """
        Get path to a specific file type for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            file_type: Type of file to retrieve
            
        Returns:
            Path to the file or None if not found
        """
        try:
            files = self.get_workflow_files(workflow_id)
            return files.get(file_type)
            
        except Exception as e:
            logger.error(f"Error getting file path: {str(e)}")
            return None
    
    def file_exists(self, workflow_id: str, file_type: str) -> bool:
        """
        Check if a file exists for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            file_type: Type of file to check
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_file_path(workflow_id, file_type)
        return file_path is not None and Path(file_path).exists()
    
    def cleanup_workflow(self, workflow_id: str) -> bool:
        """
        Clean up all files for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            workflow_dir = self.base_results_dir / workflow_id
            
            if workflow_dir.exists():
                shutil.rmtree(workflow_dir)
                logger.info(f"Cleaned up workflow directory: {workflow_dir}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cleaning up workflow: {str(e)}")
            return False
    
    def cleanup_old_workflows(self, max_age_hours: int = 24) -> int:
        """
        Clean up old workflow directories.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of workflows cleaned up
        """
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            for workflow_dir in self.base_results_dir.iterdir():
                if workflow_dir.is_dir():
                    # Check if directory is old enough
                    dir_time = datetime.fromtimestamp(workflow_dir.stat().st_mtime)
                    age_hours = (current_time - dir_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        shutil.rmtree(workflow_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old workflow: {workflow_dir}")
            
            logger.info(f"Cleaned up {cleaned_count} old workflows")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old workflows: {str(e)}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            total_workflows = 0
            total_size = 0
            
            for workflow_dir in self.base_results_dir.iterdir():
                if workflow_dir.is_dir():
                    total_workflows += 1
                    for file_path in workflow_dir.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            return {
                "total_workflows": total_workflows,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "base_directory": str(self.base_results_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e)}


# Global instance
storage_service = ResultsStorageService()
