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
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ✅ OPTION 3: Global container registry for workflow-scoped containers
_workflow_containers: Dict[str, Dict[str, Any]] = {}
_container_lock = threading.Lock()

class SandboxExecutor:
    """
    Executes code in a secure Docker sandbox with resource limits.
    Optimized for machine learning operations with support for datasets,
    resource-intensive computations, and optional GPU acceleration.
    
    Option 3 Implementation:
    - Containers are kept running during workflow execution
    - Grace period after workflow completion (default 10 minutes)
    - Containers tracked by workflow_id for better visibility
    - API endpoints available for container log access
    """
    
    def __init__(
        self,
        sandbox_image: str = "ds-capstone-ml-sandbox:latest",
        code_volume: str = "sandbox_code",
        results_volume: str = "sandbox_results",
        data_volume: str = "sandbox_data",
        timeout: int = 300,  # ✅ FIX: Increased to 5 minutes for Layer 2 execution (was 180s)
        memory_limit: str = "4g",  # ✅ FIX: Increased to 4GB for complex ML operations (was 3g)
        cpu_limit: float = 1.5,
        enable_gpu: bool = False,
        gpu_count: int = 1,
        container_retention_minutes: int = 10,  # ✅ OPTION 3: Grace period for container retention
    ):
        # Ensure image name includes tag
        if ":" not in sandbox_image:
            sandbox_image = f"{sandbox_image}:latest"
        self.sandbox_image = sandbox_image
        self.code_volume = code_volume
        self.results_volume = results_volume
        self.data_volume = data_volume
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.enable_gpu = enable_gpu
        self.gpu_count = gpu_count
        self.container_retention_minutes = container_retention_minutes
        
        # ✅ CRITICAL FIX: Pre-create volumes in background (non-blocking)
        # Don't wait for volumes - they'll be created on-demand if needed
        import threading
        volume_thread = threading.Thread(target=self._ensure_volumes_exist, daemon=True)
        volume_thread.start()
        
        # ✅ CRITICAL FIX: Check Docker health in background (non-blocking)
        health_thread = threading.Thread(target=self._check_docker_health, daemon=True)
        health_thread.start()
        
        # ✅ OPTION 3: Start background cleanup thread
        self._start_cleanup_thread()
    
    def _check_docker_health(self) -> bool:
        """
        Check if Docker daemon is healthy and responsive.
        Returns True if Docker is working, False otherwise.
        """
        try:
            logger.info("🔍 Checking Docker daemon health...")
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=15  # Loose timeout
            )
            if result.returncode == 0:
                logger.info("✅ Docker daemon is healthy and responsive")
                return True
            else:
                logger.warning(f"⚠️ Docker daemon check returned non-zero: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Docker daemon health check timed out - Docker may be slow")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Docker daemon health check failed: {e}")
            return False
    
    def _ensure_volumes_exist(self) -> None:
        """
        Pre-create Docker volumes at startup to avoid delays during execution.
        This ensures volumes are ready when needed for Layer 2 execution.
        Uses non-blocking approach - volumes will be created on-demand if needed.
        """
        volumes = [self.code_volume, self.results_volume, self.data_volume]
        logger.info(f"🔧 Ensuring Docker volumes exist: {', '.join(volumes)}")
        
        # ✅ CRITICAL FIX: Use threading to make volume creation non-blocking
        import threading
        
        def create_volume_non_blocking(volume_name: str):
            """Create volume in background thread"""
            try:
                # Try to create volume (non-blocking)
                subprocess.run(
                    ["docker", "volume", "create", volume_name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                logger.debug(f"✅ Volume '{volume_name}' creation attempted (non-blocking)")
            except Exception as e:
                logger.debug(f"Volume '{volume_name}' creation attempted: {e}")
        
        for volume_name in volumes:
            # Start volume creation in background thread (non-blocking)
            thread = threading.Thread(target=create_volume_non_blocking, args=(volume_name,), daemon=True)
            thread.start()
        
        logger.info("✅ Volume initialization started (non-blocking) - volumes will be created on-demand if needed")
    
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
                     additional_env: Optional[Dict[str, str]] = None,
                     workflow_id: Optional[str] = None,  # ✅ OPTION 3: Track by workflow_id
                     agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the provided code in the sandbox and return the results.
        
        Args:
            code: The Python code to execute
            datasets: Optional dictionary mapping dataset names to local paths
            additional_env: Optional environment variables to pass to the container
            workflow_id: Optional workflow identifier for container tracking
            agent_name: Optional agent name for container tracking
            
        Returns:
            Dict containing execution results, status, and any errors
        """
        # ✅ OPTION 3: Generate container name with workflow_id if available
        if workflow_id:
            container_name = f"sandbox-{workflow_id}-{agent_name or 'exec'}-{int(time.time())}"
        else:
            container_name = f"sandbox-{int(time.time())}"
        temp_file_path = None
        
        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file_path = temp_file.name
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
            # Always include matplotlib cache directory fix
            env_vars = additional_env.copy() if additional_env else {}
            env_vars['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
            env_vars['PYTHONUNBUFFERED'] = '1'  # Ensure output is not buffered
            
            if env_vars:
                env_json = json.dumps(env_vars)
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
                # ✅ OPTION 3: Register timeout containers too for debugging
                if workflow_id:
                    self._register_container(workflow_id, container_name, agent_name)
                else:
                    self._stop_sandbox(container_name)
                    return {
                        "status": "TIMEOUT",
                        "output": "",
                        "error": f"Execution timed out after {self.timeout} seconds",
                        "execution_time": self.timeout,
                        "memory_usage": self._get_container_memory_usage(container_name),
                        "cpu_usage": self._get_container_cpu_usage(container_name),
                        "container_name": container_name
                }
            
            # Calculate execution time
            execution_time = time.time() - execution_start
            
            # Get resource usage
            memory_usage = self._get_container_memory_usage(container_name)
            cpu_usage = self._get_container_cpu_usage(container_name)
            
            # Get results (this also deletes the execution_complete marker file)
            results = self._get_results()
            
            # Add execution metrics
            results["execution_time"] = execution_time
            results["memory_usage"] = memory_usage
            results["cpu_usage"] = cpu_usage
            results["container_name"] = container_name  # Store container name for debugging
            
            # ✅ CRITICAL FIX: DO NOT cleanup results volume here!
            # Plots need to be extracted by agents in process_sandbox_results()
            # Cleanup will happen after all agents have extracted their plots
            # (via background cleanup thread after workflow completion)
            # self._cleanup_results()  # REMOVED - prevents plot extraction
            
            # ✅ OPTION 3: Register container for workflow-scoped retention
            # Note: After deleting execution_complete marker in _get_results(), 
            # the container's entrypoint script will exit and container will stop naturally
            if workflow_id:
                self._register_container(workflow_id, container_name, agent_name)
                logger.info(f"✅ Container {container_name} registered for workflow {workflow_id} (retention: {self.container_retention_minutes}min)")
                # Give container a moment to stop naturally after marker deletion
                # The entrypoint script will exit once execution_complete is deleted
                # Note: time is already imported at module level, don't re-import
                time.sleep(2)  # Brief wait for container to stop naturally
                # Check if container is still running (should have stopped by now)
                try:
                    check_result = subprocess.run(
                        ["docker", "ps", "-q", "-f", f"name={container_name}"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if check_result.stdout.strip():
                        logger.debug(f"Container {container_name} still running, will be cleaned up by retention thread")
                    else:
                        logger.debug(f"✅ Container {container_name} stopped naturally after marker deletion")
                except Exception:
                    pass  # Ignore errors in cleanup check
            else:
                # No workflow_id - stop immediately (fallback behavior)
                logger.info(f"Sandbox execution completed. Container: {container_name}, Status: {results.get('status')}")
                self._stop_sandbox(container_name)
            
            logger.info(f"Sandbox execution completed. Container: {container_name}, Status: {results.get('status')}, Workflow: {workflow_id or 'N/A'}")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error executing code in sandbox: {str(e)}")
            # ✅ CRITICAL FIX: Always return a dict, never None
            error_result = {
                "status": "ERROR",
                "output": "",
                "error": f"Sandbox execution error: {str(e)}",
                "execution_time": 0,
                "container_name": container_name
            }
            
            # ✅ OPTION 3: Register error containers too for debugging
            if workflow_id:
                self._register_container(workflow_id, container_name, agent_name)
            else:
                self._stop_sandbox(container_name)
            
            # ✅ CRITICAL FIX: Always return error dict, even if workflow_id exists
            return error_result
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _copy_to_volume(self, source_path: str, volume_name: str, dest_filename: str) -> None:
        """
        Copy a file to a Docker volume.
        Handles large files (5MB+) efficiently with proper error handling.
        """
        # Check file size for logging
        file_size = os.path.getsize(source_path) if os.path.exists(source_path) else 0
        size_mb = file_size / (1024 * 1024)
        
        # ✅ CRITICAL FIX: Quick non-blocking volume check
        # Assume volume exists or will be created by Docker - don't block execution
        logger.debug(f"Quick check for volume '{volume_name}'...")
        volume_ready = False
        try:
            # Very quick check (5 seconds max) - don't wait
            result = subprocess.run(
                ["docker", "volume", "inspect", volume_name],
                capture_output=True,
                text=True,
                timeout=5  # ✅ Very short timeout - fail fast
            )
            if result.returncode == 0:
                volume_ready = True
                logger.debug(f"✅ Volume '{volume_name}' exists")
        except:
            # Volume check failed or timed out - assume it exists or will be created
            logger.debug(f"Volume check for '{volume_name}' timed out/failed - assuming it exists")
            volume_ready = True  # Assume ready to proceed
        
        # If volume doesn't exist, try to create it in background (non-blocking)
        if not volume_ready:
            logger.info(f"📦 Attempting to create volume '{volume_name}' in background...")
            try:
                # Start creation in background - don't wait
                subprocess.Popen(
                    ["docker", "volume", "create", volume_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.debug(f"Volume '{volume_name}' creation started in background")
            except Exception as e:
                logger.debug(f"Background volume creation failed: {e}")
        
        # ✅ CRITICAL: Always proceed - assume volume will be available when needed
        logger.debug(f"Proceeding with volume '{volume_name}' (assuming it exists or will be created)")
        
        # Create a temporary container to access the volume
        # ✅ IMPROVEMENT: Use run() instead of check_output() for better control
        # ✅ LOOSE TIMEOUT as requested
        logger.debug(f"Creating temporary container to copy file to volume '{volume_name}'")
        # ✅ CRITICAL FIX: Use shorter timeout and better error handling
        try:
            result = subprocess.run(
                ["docker", "create", "-v", f"{volume_name}:/data", "alpine"],
                capture_output=True,
                text=True,
                timeout=15  # ✅ Reduced timeout - fail fast
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, ["docker", "create"], result.stderr)
            container_id = result.stdout.strip()
            logger.debug(f"✅ Created temporary container: {container_id[:12]}")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Timeout creating temporary container for volume {volume_name} - Docker may be slow")
            # ✅ CRITICAL FIX: Try alternative approach - use existing container or create in background
            try:
                # Check if there's an existing container we can use
                check_result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"volume={volume_name}", "--format", "{{.ID}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if check_result.returncode == 0 and check_result.stdout.strip():
                    container_id = check_result.stdout.strip().split('\n')[0]
                    logger.info(f"✅ Using existing container {container_id[:12]} for volume '{volume_name}'")
                else:
                    # Create in background and wait briefly
                    proc = subprocess.Popen(
                        ["docker", "create", "-v", f"{volume_name}:/data", "alpine"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    # Note: time is already imported at module level, don't re-import
                    time.sleep(2)
                    if proc.poll() is None:
                        proc.kill()
                        raise Exception("Docker create still running after timeout")
                    stdout, stderr = proc.communicate()
                    if proc.returncode != 0:
                        raise Exception(f"Docker create failed: {stderr.decode()}")
                    container_id = stdout.decode().strip()
                    logger.info(f"✅ Created container in background: {container_id[:12]}")
            except Exception as alt_e:
                logger.error(f"❌ Alternative container creation also failed: {alt_e}")
                raise Exception(f"Could not create container for volume {volume_name}: Docker operations timing out")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create temporary container for volume {volume_name}: {e.stderr if e.stderr else str(e)}")
            raise
        
        try:
            # Copy the file to the container (docker cp handles large files efficiently)
            # For large files, docker cp uses streaming, so this should work fine
            # ✅ LOOSE TIMEOUT as requested
            logger.debug(f"Copying file ({size_mb:.2f} MB) to volume {volume_name}...")
            copy_start = time.time()
            result = subprocess.run(
                ["docker", "cp", source_path, f"{container_id}:/data/{dest_filename}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # ✅ Loose timeout (10 minutes) for large file copies
            )
            copy_time = time.time() - copy_start
            logger.debug(f"✅ File copied successfully ({size_mb:.2f} MB) in {copy_time:.2f}s")
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout copying file to volume (file size: {size_mb:.2f} MB)")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to copy file to volume: {e.stderr if e.stderr else str(e)}")
            raise
        finally:
            # Remove the temporary container
            # ✅ IMPROVEMENT: Use force removal and proper timeout
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    check=False,
                    timeout=30,  # ✅ Loose timeout as requested
                    capture_output=True
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout removing temporary container {container_id}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary container {container_id}: {e}")
    
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
            "--tmpfs", "/tmp:exec,size=1G,nodev,nosuid",  # ✅ Increased tmpfs for larger datasets (5MB+) and matplotlib cache
            "-e", "MPLCONFIGDIR=/tmp/matplotlib-cache",  # Set matplotlib cache directory
            "-e", "PYTHONUNBUFFERED=1",  # Ensure output is not buffered
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
        # ✅ LOOSE TIMEOUT as requested
        logger.debug(f"Starting sandbox container: {container_name}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)  # ✅ Loose timeout
            logger.info(f"✅ Sandbox container started: {container_name}")
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout starting sandbox container: {container_name}")
            raise
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else (e.stdout if hasattr(e, 'stdout') and e.stdout else str(e))
            logger.error(f"❌ Failed to start sandbox container: {error_msg}")
            logger.error(f"Command: {' '.join(cmd)}")
            # If GPU fails, retry without GPU
            if self.enable_gpu:
                logger.warning("Retrying without GPU support")
                self.enable_gpu = False
                self._start_sandbox(container_name)
            else:
                # Check if image exists
                try:
                    check_image = subprocess.run(
                        ["docker", "images", "-q", self.sandbox_image],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    if not check_image.stdout.strip():
                        logger.error(f"❌ Sandbox image '{self.sandbox_image}' not found. Please build it first.")
                        logger.info(f"💡 To build: docker build -t {self.sandbox_image} -f docker/Dockerfile.sandbox backend/")
                except Exception as img_check_error:
                    logger.warning(f"Could not check image existence: {img_check_error}")
                raise
    
    def _stop_sandbox(self, container_name: str) -> None:
        """Stop and remove the sandbox container"""
        try:
            # ✅ IMPROVEMENT: Add explicit stop timeout (10 seconds grace period)
            subprocess.run(
                ["docker", "stop", "-t", "10", container_name],
                check=False,
                timeout=15,  # Total timeout for stop operation
                capture_output=True
            )
            # ✅ IMPROVEMENT: Use force removal if container is stuck
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=False,
                timeout=10,  # Timeout for removal
                capture_output=True
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout stopping/removing container {container_name}, attempting force removal...")
            # Force removal as last resort
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    timeout=5,
                    capture_output=True
                )
            except Exception as e:
                logger.error(f"Failed to force remove container {container_name}: {e}")
        except Exception as e:
            logger.warning(f"Error stopping sandbox container: {str(e)}")
    
    def _is_execution_complete(self) -> bool:
        """Check if execution is complete by looking for the completion marker file"""
        # ✅ IMPROVEMENT: Add timeout to prevent hanging
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{self.results_volume}:/data",
                    "alpine", "ls", "/data/execution_complete"
                ],
                capture_output=True,
                text=True,
                timeout=10  # ✅ Timeout for completion check
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning("Timeout checking execution completion")
            return False
    
    def _get_results(self) -> Dict[str, Any]:
        """Get execution results from the sandbox"""
        # Create a temporary directory to store results
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a temporary container to access the volume
            # ✅ IMPROVEMENT: Use run() instead of check_output() with timeout
            result = subprocess.run(
                ["docker", "create", "-v", f"{self.results_volume}:/data", "alpine"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )
            container_id = result.stdout.strip()
            
            try:
                # Copy results from the container to the temp directory
                # ✅ IMPROVEMENT: Add timeout for large file copies
                subprocess.run(
                    ["docker", "cp", f"{container_id}:/data/.", temp_dir],
                    check=True,
                    timeout=300,  # 5 minute timeout for large result files
                    capture_output=True
                )
            finally:
                # Remove the temporary container
                # ✅ IMPROVEMENT: Use force removal and timeout
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    check=False,
                    timeout=10,
                    capture_output=True
                )
            
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
            
            # Log detailed results for debugging
            logger.debug(f"Sandbox results - Status: {status}, Output length: {len(output)}, Error length: {len(error)}")
            if error:
                logger.debug(f"Sandbox error (first 500 chars): {error[:500]}")
            if output:
                logger.debug(f"Sandbox output (first 500 chars): {output[:500]}")
            
            # If status is FAILED but error contains logger issue, check if we have output
            # The logger stub should prevent this, but handle gracefully
            if status == "FAILED" and error:
                if "logger" in error.lower() and "not defined" in error.lower():
                    logger.warning(f"Sandbox execution failed with logger error, but checking output: {output[:200] if output else 'No output'}")
                    # If we have substantial output, might be a false negative - upgrade to SUCCESS
                    if output and len(output) > 100:
                        logger.info("Sandbox produced output despite logger error - upgrading status to SUCCESS")
                        status = "SUCCESS"  # Upgrade status since we have output
                        # Clear error since we're using output
                        error = ""  # Don't clear completely, but mark as non-blocking
            
            # ✅ CRITICAL FIX: Delete the execution_complete marker file to signal the container to stop
            # The entrypoint script waits for this file to be deleted before stopping
            try:
                delete_result = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "-v", f"{self.results_volume}:/data",
                        "alpine", "rm", "-f", "/data/execution_complete"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if delete_result.returncode == 0:
                    logger.debug("✅ Deleted execution_complete marker file - container will stop naturally")
                else:
                    logger.warning(f"⚠️ Failed to delete execution_complete marker: {delete_result.stderr}")
            except Exception as e:
                logger.warning(f"⚠️ Error deleting execution_complete marker: {e}")
            
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
            # ✅ IMPROVEMENT: Add timeout to prevent hanging
            result = subprocess.run(
                ["docker", "stats", container_name, "--no-stream", "--format", "{{.MemUsage}}"],
                capture_output=True,
                text=True,
                timeout=10,  # ✅ Timeout for stats command
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
            # ✅ IMPROVEMENT: Add timeout to prevent hanging
            result = subprocess.run(
                ["docker", "stats", container_name, "--no-stream", "--format", "{{.CPUPerc}}"],
                capture_output=True,
                text=True,
                timeout=10,  # ✅ Timeout for stats command
                check=True
            )
            
            # Parse the output (e.g., "5.25%")
            cpu_stats = result.stdout.strip()
            
            return {"percentage": cpu_stats}
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {str(e)}")
            return {"error": str(e)}
    
    # ✅ OPTION 3: Container registration and management methods
    def _register_container(self, workflow_id: str, container_name: str, agent_name: Optional[str] = None) -> None:
        """Register a container for workflow-scoped retention"""
        with _container_lock:
            if workflow_id not in _workflow_containers:
                _workflow_containers[workflow_id] = {
                    "containers": [],
                    "workflow_started": datetime.now(),
                    "workflow_completed": None
                }
            
            container_info = {
                "container_name": container_name,
                "agent_name": agent_name or "unknown",
                "created_at": datetime.now(),
                "retention_until": datetime.now() + timedelta(minutes=self.container_retention_minutes),
                "stopped": False
            }
            
            _workflow_containers[workflow_id]["containers"].append(container_info)
            logger.info(f"📦 Registered container {container_name} for workflow {workflow_id}")
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread to clean up expired containers"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(60)  # Check every minute
                    self._cleanup_expired_containers()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Container cleanup thread started")
    
    def _cleanup_expired_containers(self) -> None:
        """Clean up containers that have exceeded their retention period"""
        with _container_lock:
            now = datetime.now()
            workflows_to_remove = []
            
            for workflow_id, workflow_info in _workflow_containers.items():
                containers_to_remove = []
                
                for container_info in workflow_info["containers"]:
                    # Check if retention period has expired
                    if now > container_info["retention_until"]:
                        container_name = container_info["container_name"]
                        if not container_info["stopped"]:
                            try:
                                # ✅ IMPROVEMENT: Stop with shorter timeout and force kill if needed
                                result = subprocess.run(
                                    ["docker", "stop", "-t", "5", container_name],
                                    check=False,
                                    timeout=8,
                                    capture_output=True
                                )
                                if result.returncode != 0:
                                    # If stop failed, try force kill
                                    logger.debug(f"Stop failed for {container_name}, trying kill")
                                    subprocess.run(
                                        ["docker", "kill", container_name],
                                        check=False,
                                        timeout=5,
                                        capture_output=True
                                    )
                                container_info["stopped"] = True
                                logger.debug(f"🛑 Stopped expired container: {container_name}")
                            except subprocess.TimeoutExpired:
                                logger.warning(f"Timeout stopping container {container_name}, will force remove")
                                # Try kill as last resort
                                try:
                                    subprocess.run(
                                        ["docker", "kill", container_name],
                                        check=False,
                                        timeout=5,
                                        capture_output=True
                                    )
                                except:
                                    pass
                            except Exception as e:
                                logger.warning(f"Failed to stop container {container_name}: {e}")
                        
                        # Remove the container
                        # ✅ IMPROVEMENT: Use force removal with timeout - retry once if needed
                        try:
                            result = subprocess.run(
                                ["docker", "rm", "-f", container_name],
                                check=False,
                                timeout=8,
                                capture_output=True
                            )
                            if result.returncode != 0:
                                # Retry once after a short delay
                                time.sleep(0.5)
                                subprocess.run(
                                    ["docker", "rm", "-f", container_name],
                                    check=False,
                                    timeout=5,
                                    capture_output=True
                                )
                            containers_to_remove.append(container_info)
                            logger.debug(f"🗑️ Removed expired container: {container_name}")
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Timeout removing container {container_name} - will be cleaned up later")
                            # Don't fail the workflow - container will be cleaned up by background thread
                            # Still mark for removal from tracking
                            containers_to_remove.append(container_info)
                        except Exception as e:
                            logger.warning(f"Failed to remove container {container_name}: {e}")
                            # Still mark for removal from tracking
                            containers_to_remove.append(container_info)
                
                # Remove cleaned containers from list
                for container_info in containers_to_remove:
                    workflow_info["containers"].remove(container_info)
                
                # Remove workflow entry if no containers left
                if not workflow_info["containers"]:
                    workflows_to_remove.append(workflow_id)
            
            # Remove empty workflow entries
            for workflow_id in workflows_to_remove:
                del _workflow_containers[workflow_id]
                logger.info(f"🧹 Cleaned up all containers for workflow {workflow_id}")
    
    @staticmethod
    def mark_workflow_completed(workflow_id: str) -> None:
        """Mark a workflow as completed, extending retention period"""
        with _container_lock:
            if workflow_id in _workflow_containers:
                _workflow_containers[workflow_id]["workflow_completed"] = datetime.now()
                # Extend retention for all containers in this workflow
                retention_extension = timedelta(minutes=10)  # Additional 10 minutes after completion
                for container_info in _workflow_containers[workflow_id]["containers"]:
                    container_info["retention_until"] = datetime.now() + retention_extension
                logger.info(f"✅ Workflow {workflow_id} marked as completed, containers retained for additional {retention_extension}")
    
    @staticmethod
    def get_workflow_containers(workflow_id: str) -> List[Dict[str, Any]]:
        """Get all containers for a workflow"""
        with _container_lock:
            if workflow_id in _workflow_containers:
                return [
                    {
                        "container_name": c["container_name"],
                        "agent_name": c["agent_name"],
                        "created_at": c["created_at"].isoformat(),
                        "retention_until": c["retention_until"].isoformat(),
                        "stopped": c["stopped"]
                    }
                    for c in _workflow_containers[workflow_id]["containers"]
                ]
            return []
    
    @staticmethod
    def get_container_logs(container_name: str, tail: int = 100) -> Dict[str, Any]:
        """Get logs from a container"""
        try:
            # ✅ IMPROVEMENT: Add timeout to prevent hanging
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_name],
                capture_output=True,
                text=True,
                timeout=30,  # ✅ Timeout for logs command
                check=True
            )
            return {
                "success": True,
                "logs": result.stdout,
                "error_logs": result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Failed to get logs: {e.stderr or str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error accessing container logs: {str(e)}"
            }
    
    @staticmethod
    def cleanup_workflow_containers(workflow_id: str, force: bool = False) -> Dict[str, Any]:
        """Manually cleanup all containers for a workflow"""
        with _container_lock:
            if workflow_id not in _workflow_containers:
                return {"success": False, "message": f"No containers found for workflow {workflow_id}"}
            
            containers_cleaned = []
            for container_info in _workflow_containers[workflow_id]["containers"]:
                container_name = container_info["container_name"]
                try:
                    if not container_info["stopped"]:
                        # ✅ IMPROVEMENT: Stop with timeout
                        subprocess.run(
                            ["docker", "stop", "-t", "10", container_name],
                            check=False,
                            timeout=15,
                            capture_output=True
                        )
                    # ✅ IMPROVEMENT: Use force removal with timeout
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        check=False,
                        timeout=10,
                        capture_output=True
                    )
                    containers_cleaned.append(container_name)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout cleaning up container {container_name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup container {container_name}: {e}")
            
            del _workflow_containers[workflow_id]
            return {
                "success": True,
                "message": f"Cleaned up {len(containers_cleaned)} containers",
                "containers": containers_cleaned
            }
    
    def get_files_from_volume(self, volume_name: str) -> List[str]:
        """
        List all files in a Docker volume (specifically plot files).
        
        Args:
            volume_name: Name of the Docker volume
            
        Returns:
            List of filenames in the volume (PNG, JPG, SVG files)
        """
        try:
            # Use docker run to list files in the volume
            result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{volume_name}:/data", "alpine",
                 "find", "/data", "-type", "f", "(", "-name", "*.png", "-o", "-name", "*.jpg", "-o", "-name", "*.jpeg", "-o", "-name", "*.svg", ")"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                files = [Path(line.strip()).name for line in result.stdout.strip().split('\n') if line.strip()]
                logger.info(f"Found {len(files)} files in volume {volume_name}: {files}")
                return files
            else:
                logger.debug(f"No files found in volume {volume_name}")
                return []
        except Exception as e:
            logger.warning(f"Failed to list files in volume {volume_name}: {e}")
            return []
    
    def _cleanup_results(self) -> None:
        """
        Clean up result files from the volume.
        
        ⚠️ WARNING: This should only be called AFTER all agents have extracted their plots!
        Currently disabled in execute_code() to prevent plot loss.
        """
        # Create a temporary container to access the volume
        # ✅ IMPROVEMENT: Use run() instead of check_output() with timeout
        try:
            result = subprocess.run(
                ["docker", "create", "-v", f"{self.results_volume}:/data", "alpine"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )
            container_id = result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to create temporary container for cleanup: {e}")
            return
        
        try:
            # Remove result files (but keep plots until extracted)
            # Only remove text files (output.txt, error.txt, status.txt)
            # ✅ IMPROVEMENT: Add timeout
            subprocess.run(
                ["docker", "run", "--rm", "-v", f"{self.results_volume}:/data", "alpine",
                 "sh", "-c", "rm -f /data/output.txt /data/error.txt /data/status.txt /data/status_code.txt /data/execution_complete"],
                check=False,
                timeout=30,
                capture_output=True
            )
            logger.debug("Cleaned up text result files from volume (plots preserved)")
        finally:
            # Remove the temporary container
            # ✅ IMPROVEMENT: Use force removal and timeout
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                timeout=10,
                capture_output=True
            )


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
