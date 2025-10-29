"""
Sandbox Executor Service

This module provides functionality to execute AI-generated code in a secure Docker sandbox
with strict resource limits and no network access. It is specifically designed to handle
machine learning operations safely and efficiently.

Features:
- Secure execution of AI-generated code in isolated Docker containers
- Resource limits (CPU, memory, execution time)
- No network access
- Support for ML libraries (scikit-learn, pandas, numpy, etc.)
- Handling of large datasets and computationally intensive operations
- Optional GPU support when available
"""

import os
import time
import logging
import subprocess
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)

class SandboxExecutor:
    """
    Executes code in a secure Docker sandbox with resource limits.
    Optimized for machine learning operations with support for datasets,
    resource-intensive computations, and optional GPU acceleration.
    """
    
    def __init__(
        self,
        sandbox_image: str = "ds-capstone-ml-sandbox",
        code_volume: str = "sandbox_code",
        results_volume: str = "sandbox_results",
        data_volume: str = "sandbox_data",
        timeout: int = 120,  # Overall timeout for the entire operation (increased for ML tasks)
        memory_limit: str = "2g",
        cpu_limit: float = 1.5,
        enable_gpu: bool = False,
        gpu_count: int = 1,
    ):
        self.sandbox_image = sandbox_image
        self.code_volume = code_volume
        self.results_volume = results_volume
        self.data_volume = data_volume
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.enable_gpu = enable_gpu
        self.gpu_count = gpu_count
    
    def load_dataset(self, dataset_path: str, dataset_name: str) -> bool:
        """
        Load a dataset into the sandbox data volume.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_name: Name to give the dataset in the sandbox
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._copy_to_volume(dataset_path, self.data_volume, dataset_name)
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def execute_code(self, 
                     code: str, 
                     datasets: Optional[Dict[str, str]] = None,
                     additional_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute the provided code in the sandbox and return the results.
        
        Args:
            code: The Python code to execute
            datasets: Optional dictionary mapping dataset names to local paths
            additional_env: Optional environment variables to pass to the container
            
        Returns:
            Dict containing execution results, status, and any errors
        """
        # Generate a unique container name
        container_name = f"sandbox-{int(time.time())}"
        temp_file_path = None
        
        try:
            # Write code to a temporary file with logger setup prepended
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file_path = temp_file.name
                
                # Prepend logger setup to the code
                logger_setup = """
            import logging
            import sys

            # Setup basic logger for sandbox execution
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            logger = logging.getLogger(__name__)

            # Make logger available globally
            import builtins
            builtins.logger = logger

            """
                
                # Write logger setup + user code
                temp_file.write(logger_setup)
                temp_file.write("\n# User-generated code below:\n")
                temp_file.write(code)
            
            # Copy code to the sandbox volume
            self._copy_to_volume(temp_file_path, self.code_volume, "script.py")
            
            # Fix permissions for the script file after copying
            try:
                subprocess.run([
                    'docker', 'run', '--rm', '-v', f'{self.code_volume}:/app/code',
                    'alpine:latest', 'chmod', '755', '/app/code/script.py'
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to set permissions: {e}")
            
            # Load datasets if provided
            if datasets:
                for dataset_name, dataset_path in datasets.items():
                    success = self.load_dataset(dataset_path, dataset_name)
                    if not success:
                        return {
                            "status": "ERROR",
                            "output": "",
                            "error": f"Failed to load dataset: {dataset_name}",
                            "execution_time": 0
                        }
            
            # Create environment file if needed
            if additional_env:
                env_json = json.dumps(additional_env)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as env_file:
                    env_file_path = env_file.name
                    env_file.write(env_json)
                self._copy_to_volume(env_file_path, self.code_volume, "env.json")
                os.unlink(env_file_path)
            
            # Start the sandbox container
            self._start_sandbox(container_name)
            
            # Record start time for execution metrics
            execution_start = time.time()
            
            # Wait for execution to complete (with timeout)
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self._is_execution_complete():
                    break
                time.sleep(1)
            else:
                logger.warning(f"Sandbox execution timed out after {self.timeout} seconds")
                self._stop_sandbox(container_name)
                return {
                    "status": "TIMEOUT",
                    "output": "",
                    "error": f"Execution timed out after {self.timeout} seconds",
                    "execution_time": self.timeout,
                    "memory_usage": self._get_container_memory_usage(container_name),
                    "cpu_usage": self._get_container_cpu_usage(container_name)
                }
            
            # Check if execution actually completed successfully
            if not self._is_execution_complete():
                logger.error("Sandbox execution did not complete successfully")
                self._stop_sandbox(container_name)
                return {
                    "status": "FAILED",
                    "output": "",
                    "error": "Execution did not complete successfully"
                }
            
            # Calculate execution time
            execution_time = time.time() - execution_start
            
            # Get resource usage
            memory_usage = self._get_container_memory_usage(container_name)
            cpu_usage = self._get_container_cpu_usage(container_name)
            
            # Get results
            results = self._get_results()
            
            # Add execution metrics
            results["execution_time"] = execution_time
            results["memory_usage"] = memory_usage
            results["cpu_usage"] = cpu_usage
            
            # Clean up
            self._cleanup_results()
            self._stop_sandbox(container_name)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error executing code in sandbox: {str(e)}")
            self._stop_sandbox(container_name)
            return {
                "status": "ERROR",
                "output": "",
                "error": f"Sandbox execution error: {str(e)}",
                "execution_time": 0
            }
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _copy_to_volume(self, source_path: str, volume_name: str, dest_filename: str) -> None:
        """Copy a file to a Docker volume"""
        # Create a temporary container to access the volume
        container_id = subprocess.check_output(
            ["docker", "create", "-v", f"{volume_name}:/data", "alpine"],
            text=True
        ).strip()
        
        try:
            # Copy the file to the container
            subprocess.run(
                ["docker", "cp", source_path, f"{container_id}:/data/{dest_filename}"],
                check=True
            )
        finally:
            # Remove the temporary container
            subprocess.run(["docker", "rm", container_id], check=True)
    
    def _start_sandbox(self, container_name: str) -> None:
        """Start the sandbox container with appropriate resource limits"""
        # Base command
        cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--name", container_name,
            "--network", "none",  # No network access
            "--memory", self.memory_limit,
            "--cpus", str(self.cpu_limit),
            "--security-opt=no-new-privileges",
            "--read-only",  # Read-only filesystem
            "--tmpfs", "/tmp:exec,size=256M,nodev,nosuid",  # Temporary filesystem
            "-v", f"{self.code_volume}:/app/code",
            "-v", f"{self.results_volume}:/app/results",
            "-v", f"{self.data_volume}:/app/data"
        ]
        
        # Add GPU support if enabled
        if self.enable_gpu:
            cmd.extend(["--gpus", f"device={','.join(map(str, range(self.gpu_count)))}"])
        
        # Add image name
        cmd.append(self.sandbox_image)
        
        # Run the container
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start sandbox container: {str(e)}")
            # If GPU fails, retry without GPU
            if self.enable_gpu:
                logger.warning("Retrying without GPU support")
                self.enable_gpu = False
                self._start_sandbox(container_name)
            else:
                raise
    
    def _stop_sandbox(self, container_name: str) -> None:
        """Stop and remove the sandbox container"""
        try:
            subprocess.run(["docker", "stop", container_name], check=False)
            subprocess.run(["docker", "rm", container_name], check=False)
        except Exception as e:
            logger.warning(f"Error stopping sandbox container: {str(e)}")
    
    def _is_execution_complete(self) -> bool:
        """Check if execution is complete by looking for the completion marker file"""
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{self.results_volume}:/data",
                "alpine", "ls", "/data/execution_complete"
            ],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    def _get_results(self) -> Dict[str, Any]:
        """Get execution results from the sandbox"""
        # Create a temporary directory to store results
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a temporary container to access the volume
            container_id = subprocess.check_output(
                ["docker", "create", "-v", f"{self.results_volume}:/data", "alpine"],
                text=True
            ).strip()
            
            try:
                # Copy results from the container to the temp directory
                subprocess.run(
                    ["docker", "cp", f"{container_id}:/data/.", temp_dir],
                    check=True
                )
            finally:
                # Remove the temporary container
                subprocess.run(["docker", "rm", container_id], check=True)
            
            # Read results
            output_path = os.path.join(temp_dir, "output.txt")
            error_path = os.path.join(temp_dir, "error.txt")
            status_path = os.path.join(temp_dir, "status.txt")
            status_code_path = os.path.join(temp_dir, "status_code.txt")
            
            output = ""
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    output = f.read()
            
            error = ""
            if os.path.exists(error_path):
                with open(error_path, "r") as f:
                    error = f.read()
            
            status_message = ""
            if os.path.exists(status_path):
                with open(status_path, "r") as f:
                    status_message = f.read()
            
            status = "UNKNOWN"
            if os.path.exists(status_code_path):
                with open(status_code_path, "r") as f:
                    status = f.read().strip()
            
            # Debug: Log what we found
            logger.info(f"Sandbox results: status={status}, output_len={len(output)}, error_len={len(error)}")
            if error:
                logger.warning(f"Sandbox error content: {error[:200]}...")
            
            return {
                "status": status,
                "status_message": status_message,
                "output": output,
                "error": error
            }
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_container_memory_usage(self, container_name: str) -> Dict[str, Any]:
        """Get memory usage statistics for the container"""
        try:
            result = subprocess.run(
                ["docker", "stats", container_name, "--no-stream", "--format", "{{.MemUsage}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output (e.g., "125MiB / 2GiB")
            memory_stats = result.stdout.strip()
            
            # Extract current usage and limit
            if " / " in memory_stats:
                current, limit = memory_stats.split(" / ")
                return {
                    "current": current,
                    "limit": limit,
                    "raw": memory_stats
                }
            
            return {"raw": memory_stats}
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {str(e)}")
            return {"error": str(e)}
    
    def _get_container_cpu_usage(self, container_name: str) -> Dict[str, Any]:
        """Get CPU usage statistics for the container"""
        try:
            result = subprocess.run(
                ["docker", "stats", container_name, "--no-stream", "--format", "{{.CPUPerc}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output (e.g., "5.25%")
            cpu_stats = result.stdout.strip()
            
            return {"percentage": cpu_stats}
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {str(e)}")
            return {"error": str(e)}
    
    def _cleanup_results(self) -> None:
        """Clean up result files from the volume"""
        # Create a temporary container to access the volume
        container_id = subprocess.check_output(
            ["docker", "create", "-v", f"{self.results_volume}:/data", "alpine"],
            text=True
        ).strip()
        
        try:
            # Remove result files
            subprocess.run(
                ["docker", "exec", container_id, "rm", "-f", "/data/*"],
                check=False
            )
        finally:
            # Remove the temporary container
            subprocess.run(["docker", "rm", container_id], check=True)


# Example usage
if __name__ == "__main__":
    # Simple test code
    test_code = """
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.4f}")
"""
    
    executor = SandboxExecutor()
    results = executor.execute_code(test_code)
    print(f"Execution results: {results}")
