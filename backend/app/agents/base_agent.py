"""
🤖 Base Agent Class for DS Capstone Project

This module provides the base class that all agents inherit from.
It includes common functionality like logging, error handling, and state management.

DOUBLE-LAYER ARCHITECTURE:
- Layer 1 (Hardcoded): Fast, reliable, deterministic analysis using existing components
- Layer 2 (LLM + Sandbox): Flexible, adaptive code generation with secure execution
- Fallback Logic: Always fall back to Layer 1 if Layer 2 fails
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
from ..services.sandbox_executor import SandboxExecutor
from ..services.code_validator import CodeValidator, get_code_validator
from ..services.llm_service import LLMService, get_llm_service, LLMProvider

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
    🤖 Base Agent Class with Double-Layer Architecture

    All agents in the multi-agent system inherit from this base class.
    It provides common functionality for logging, error handling, and state management.

    DOUBLE-LAYER ARCHITECTURE:
    - Layer 1: Hardcoded reliable analysis (always runs)
    - Layer 2: LLM-generated adaptive code (optional, with sandbox execution)
    - Automatic fallback to Layer 1 if Layer 2 fails

    Child agents must implement:
    - perform_layer1_analysis(): Reliable hardcoded analysis
    - generate_layer2_code(): LLM prompt for code generation
    - process_sandbox_results(): Process and validate Layer 2 results (optional)
    """

    def __init__(
        self,
        agent_name: str,
        agent_version: str = "1.0.0",
        enable_layer2: bool = True,
        sandbox_timeout: int = 120,
        sandbox_memory_limit: str = "2g",
        sandbox_cpu_limit: float = 1.5
    ):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.enable_layer2 = enable_layer2
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

        # Initialize Layer 2 services (optional, graceful degradation)
        self._init_layer2_services(sandbox_timeout, sandbox_memory_limit, sandbox_cpu_limit)

    def _init_layer2_services(
        self,
        sandbox_timeout: int,
        sandbox_memory_limit: str,
        sandbox_cpu_limit: float
    ) -> None:
        """
        Initialize Layer 2 services with graceful degradation.

        Services are optional - if initialization fails, Layer 2 is disabled.
        """
        self.sandbox_executor: Optional[SandboxExecutor] = None
        self.code_validator: Optional[CodeValidator] = None
        self.llm_service: Optional[LLMService] = None

        if not self.enable_layer2:
            self.logger.info("Layer 2 disabled by configuration")
            return

        try:
            # Initialize code validator
            self.code_validator = get_code_validator()
            self.logger.info("CodeValidator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize CodeValidator: {e}")
            self.enable_layer2 = False

        try:
            # Initialize sandbox executor
            self.sandbox_executor = SandboxExecutor(
                timeout=sandbox_timeout,
                memory_limit=sandbox_memory_limit,
                cpu_limit=sandbox_cpu_limit
            )
            self.logger.info(f"SandboxExecutor initialized (timeout={sandbox_timeout}s, memory={sandbox_memory_limit})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SandboxExecutor: {e}")
            self.enable_layer2 = False

        try:
            # Initialize LLM service (Gemini by default)
            self.llm_service = get_llm_service(LLMProvider.GEMINI)
            self.logger.info("LLMService initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLMService: {e}")
            self.enable_layer2 = False

        if not self.enable_layer2:
            self.logger.warning("⚠️ LAYER 2 DISABLED due to service initialization failures")
            self.logger.warning(f"  - CodeValidator: {hasattr(self, 'code_validator') and self.code_validator is not None}")
            self.logger.warning(f"  - SandboxExecutor: {hasattr(self, 'sandbox_executor') and self.sandbox_executor is not None}")
            self.logger.warning(f"  - LLMService: {hasattr(self, 'llm_service') and self.llm_service is not None}")
        else:
            self.logger.info("✅ LAYER 2 ENABLED - All services initialized successfully")

    # ===== DOUBLE-LAYER ABSTRACT METHODS =====
    # Child agents must implement these methods

    @abstractmethod
    async def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """
        LAYER 1: Perform reliable, hardcoded analysis.

        This is the fallback layer and must always work reliably.
        Use existing, tested components and algorithms.

        Args:
            state: Current workflow state

        Returns:
            Dictionary containing Layer 1 analysis results

        Example:
            {
                "missing_values": {...},
                "outliers": {...},
                "data_quality_score": 85.5
            }
        """
        pass

    @abstractmethod
    def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
        """
        LAYER 2: Generate prompt for LLM code generation.

        Create a detailed prompt that includes Layer 1 results and asks
        the LLM to generate adaptive Python code for improved analysis.

        Args:
            layer1_results: Results from Layer 1 analysis
            state: Current workflow state

        Returns:
            Prompt string for LLM code generation

        Example:
            "Generate Python code to clean data with these issues: {layer1_results}..."
        """
        pass

    def process_sandbox_results(
        self,
        sandbox_output: Dict[str, Any],
        _layer1_results: Dict[str, Any],
        _state: ClassificationState
    ) -> Dict[str, Any]:
        """
        LAYER 2: Process and validate sandbox execution results.

        Default implementation returns sandbox output as-is.
        Override this method for custom validation and processing.

        Args:
            sandbox_output: Raw output from sandbox execution
            layer1_results: Results from Layer 1 (for comparison)
            state: Current workflow state

        Returns:
            Processed and validated results

        Raises:
            ValueError: If results are invalid or worse than Layer 1
        """
        # Default implementation: basic validation
        if sandbox_output.get("status") != "SUCCESS":
            raise ValueError(f"Sandbox execution failed: {sandbox_output.get('error')}")

        return sandbox_output.get("output", {})

    # ===== DOUBLE-LAYER EXECUTION ORCHESTRATION =====

    async def execute(self, state: ClassificationState) -> ClassificationState:
        """
        Execute the agent with double-layer architecture.

        Flow:
        1. Always run Layer 1 (reliable fallback)
        2. If Layer 2 enabled, attempt LLM + sandbox execution
        3. If Layer 2 fails, fall back to Layer 1 results
        4. Log which layer was used

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with results
        """
        self.logger.info(f"Starting double-layer execution for {self.agent_name}")

        # Track which layer was used
        layer_used = "layer1"
        layer2_attempted = False
        layer2_error = None

        try:
            # STEP 1: Always run Layer 1 (reliable fallback)
            self.logger.info("Executing Layer 1 (hardcoded analysis)...")
            layer1_start = time.time()
            layer1_results = await self.perform_layer1_analysis(state)
            layer1_time = time.time() - layer1_start
            self.log_performance("Layer 1 execution", layer1_time)

            # Start with Layer 1 results
            final_results = layer1_results

            # STEP 2: Attempt Layer 2 if enabled
            self.logger.info(f"🔍 Layer 2 Check: enable_layer2={self.enable_layer2}, can_use={self._can_use_layer2()}")
            
            if self.enable_layer2 and self._can_use_layer2():
                layer2_attempted = True
                self.logger.info("🚀 Attempting Layer 2 (LLM + sandbox)...")

                try:
                    layer2_results = await self._execute_layer2(layer1_results, state)

                    # If Layer 2 succeeded, use those results
                    if layer2_results:
                        final_results = layer2_results
                        layer_used = "layer2"
                        self.logger.info("✅ Layer 2 execution successful, using Layer 2 results")
                    else:
                        self.logger.warning("⚠️ Layer 2 returned None, using Layer 1 results")

                except Exception as e:
                    layer2_error = str(e)
                    self.logger.warning(f"❌ Layer 2 failed: {e}, falling back to Layer 1")
                    # Keep using layer1_results
            else:
                if not self.enable_layer2:
                    self.logger.info("ℹ️ Layer 2 skipped: enable_layer2=False")
                elif not self._can_use_layer2():
                    self.logger.warning("ℹ️ Layer 2 skipped: Required services not available")

            # STEP 3: Update state with results and metadata
            state = self._update_state_with_results(
                state,
                final_results,
                layer_used,
                layer2_attempted,
                layer2_error
            )

            return state

        except Exception as e:
            # Layer 1 should never fail, but handle it anyway
            self.logger.error(f"Critical error in {self.agent_name}: {e}")
            raise

    async def _execute_layer2(
        self,
        layer1_results: Dict[str, Any],
        state: ClassificationState
    ) -> Optional[Dict[str, Any]]:
        """
        Execute Layer 2: LLM code generation + sandbox execution.

        Args:
            layer1_results: Results from Layer 1
            state: Current workflow state

        Returns:
            Layer 2 results or None if failed

        Raises:
            Exception: If any step of Layer 2 fails
        """
        layer2_start = time.time()

        try:
            # Step 1: Generate code prompt
            self.logger.info("📝 Step 1/5: Generating Layer 2 code prompt...")
            prompt = self.generate_layer2_code(layer1_results, state)
            self.logger.info(f"  ✅ Generated prompt: {len(prompt)} characters")

            # Step 2: Use LLM to generate code
            self.logger.info("🤖 Step 2/5: Calling LLM to generate code...")
            self.logger.info(f"  - LLM Service: {type(self.llm_service).__name__}")
            llm_response = await self.llm_service.generate_code(
                prompt=prompt,
                context=layer1_results,
                code_type=self.agent_name
            )

            generated_code = llm_response.get("code")
            if not generated_code:
                raise ValueError("LLM did not generate any code")

            self.logger.info(f"  ✅ LLM generated {len(generated_code)} characters of code")

            # Add warning suppression header to generated code
            warning_suppression = """import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

"""
            generated_code = warning_suppression + generated_code
            self.logger.info(f"  ✅ Added warning suppression (final: {len(generated_code)} chars)")

            # Step 3: Validate code
            self.logger.info("🔍 Step 3/5: Validating generated code...")
            validation_result = self.code_validator.validate(generated_code)

            # Always log validation report for debugging
            self.logger.info(self.code_validator.get_validation_report(validation_result))

            if not validation_result.is_valid:
                # Combine all issues into error message
                all_issues = []
                if validation_result.errors:
                    all_issues.extend([f"ERROR: {e}" for e in validation_result.errors])
                if validation_result.security_issues:
                    all_issues.extend([f"SECURITY: {s}" for s in validation_result.security_issues])
                
                error_msg = f"Code validation failed: {'; '.join(all_issues) if all_issues else 'Unknown validation error'}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            if validation_result.warnings:
                self.logger.warning(f"  ⚠️ Validation warnings: {len(validation_result.warnings)} warnings")

            # Step 4: Execute in sandbox
            self.logger.info("🐳 Step 4/5: Executing code in sandbox...")
            self.logger.info(f"  - Sandbox Executor: {type(self.sandbox_executor).__name__}")
            self.logger.info(f"  - Timeout: {getattr(self.sandbox_executor, 'timeout', 'N/A')}s")
            
            sandbox_output = await self.execute_layer2_in_sandbox(
                generated_code,
                layer1_results,
                state
            )
            
            self.logger.info(f"  ✅ Sandbox execution complete: {sandbox_output.get('status', 'UNKNOWN')}")

            # Step 5: Process sandbox results
            self.logger.info("📊 Step 5/5: Processing sandbox results...")
            layer2_results = self.process_sandbox_results(
                sandbox_output,
                layer1_results,
                state
            )
            
            self.logger.info(f"  ✅ Processed {len(layer2_results)} result keys")

            layer2_time = time.time() - layer2_start
            self.log_performance("Layer 2 execution", layer2_time)
            self.logger.info(f"✅ LAYER 2 COMPLETE in {layer2_time:.2f}s")

            return layer2_results

        except Exception as e:
            self.logger.error(f"Layer 2 execution failed: {e}")
            raise

    async def execute_layer2_in_sandbox(
        self,
        generated_code: str,
        layer1_results: Dict[str, Any],
        state: ClassificationState
    ) -> Dict[str, Any]:
        """
        Execute generated code in secure sandbox environment.

        Args:
            generated_code: Validated Python code from LLM
            layer1_results: Results from Layer 1 (for context)
            state: Current workflow state

        Returns:
            Dictionary containing sandbox execution results

        Example return:
            {
                "status": "SUCCESS",
                "output": {...},
                "execution_time": 5.2,
                "memory_usage": {"current": "150MiB", "limit": "2GiB"}
            }
        """
        self.logger.info("Preparing sandbox execution...")

        # Prepare data for sandbox (if needed)
        datasets = self._prepare_sandbox_datasets(state)

        # Prepare environment variables
        env_vars = {
            "AGENT_NAME": self.agent_name,
            "LAYER1_RESULTS": str(layer1_results)
        }

        # Execute code in sandbox
        result = self.sandbox_executor.execute_code(
            code=generated_code,
            datasets=datasets,
            additional_env=env_vars
        )

        self.logger.info(f"Sandbox execution completed: {result.get('status')}")

        # ✅ FIX: Store metrics for workflow tracking
        result["layer2_execution_metrics"] = {
            "cpu_usage": result.get("cpu_usage", {}),
            "memory_usage": result.get("memory_usage", {}),
            "execution_time": result.get("execution_time", 0),
            "status": result.get("status")
        }
        logger.info(f"📊 Sandbox metrics: CPU={result.get('cpu_usage')}, Memory={result.get('memory_usage')}, Time={result.get('execution_time')}s")

        if result.get("status") == "TIMEOUT":
            raise TimeoutError(f"Sandbox execution timed out after {self.sandbox_executor.timeout}s")

        if result.get("status") == "ERROR":
            raise RuntimeError(f"Sandbox execution error: {result.get('error')}")

        return result

    def _can_use_layer2(self) -> bool:
        """Check if Layer 2 can be used."""
        return (
            self.enable_layer2 and
            self.sandbox_executor is not None and
            self.code_validator is not None and
            self.llm_service is not None
        )

    def _prepare_sandbox_datasets(self, _state: ClassificationState) -> Dict[str, str]:
        """
        Prepare datasets for sandbox execution.

        Override this method in child agents to provide specific datasets.

        Args:
            state: Current workflow state

        Returns:
            Dictionary mapping dataset names to file paths
        """
        return {}

    def _update_state_with_results(
        self,
        state: ClassificationState,
        results: Dict[str, Any],
        layer_used: str,
        layer2_attempted: bool,
        layer2_error: Optional[str]
    ) -> ClassificationState:
        """
        Update workflow state with execution results and metadata.

        Args:
            state: Current workflow state
            results: Execution results
            layer_used: Which layer was used ("layer1" or "layer2")
            layer2_attempted: Whether Layer 2 was attempted
            layer2_error: Error message if Layer 2 failed

        Returns:
            Updated workflow state
        """
        # Store results in state
        if "agent_results" not in state:
            state["agent_results"] = {}

        state["agent_results"][self.agent_name] = {
            "results": results,
            "layer_used": layer_used,
            "layer2_attempted": layer2_attempted,
            "layer2_error": layer2_error,
            "timestamp": datetime.now().isoformat()
        }

        # Add metadata
        if "layer_usage" not in state:
            state["layer_usage"] = {}

        state["layer_usage"][self.agent_name] = layer_used
        
        # ✅ FIX: Pass through Layer 2 execution metrics if present
        if "layer2_execution_metrics" in results:
            state["layer2_execution_metrics"] = results["layer2_execution_metrics"]
            self.logger.info(f"📊 Stored Layer 2 execution metrics in state")
        
        # CRITICAL: Update processed_dataset if this agent produced a cleaned/transformed dataset
        # This ensures subsequent agents (like EDA) can access the latest version
        if "cleaned_dataset" in results and results["cleaned_dataset"] is not None:
            state["processed_dataset"] = results["cleaned_dataset"]
            state["dataset"] = results["cleaned_dataset"]  # Also update main dataset reference
            self.logger.info(f"✅ Updated processed_dataset with cleaned data from {self.agent_name}")
        
        # For feature engineering, update with engineered dataset
        if "engineered_dataset" in results and results["engineered_dataset"] is not None:
            state["processed_dataset"] = results["engineered_dataset"]
            state["dataset"] = results["engineered_dataset"]
            self.logger.info(f"✅ Updated processed_dataset with engineered features from {self.agent_name}")
        
        # ✅ FIX: Copy all result keys to state for easy access (except datasets to avoid duplication)
        excluded_keys = {"cleaned_dataset", "engineered_dataset", "processed_dataset", "original_dataset", "layer2_execution_metrics"}
        copied_count = 0
        for key, value in results.items():
            if key not in excluded_keys and value is not None:
                state[key] = value
                copied_count += 1
                self.logger.info(f"✅ Copied {key} to state from {self.agent_name} (type: {type(value).__name__})")
        
        self.logger.info(f"📦 Total keys copied from {self.agent_name}: {copied_count}")

        return state

    # ===== EXISTING BASE METHODS (updated for double-layer) =====

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
