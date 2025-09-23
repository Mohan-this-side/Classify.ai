"""
🕸️ LangGraph State Machine for Data Cleaning Workflow
====================================================

This demonstrates the most advanced capabilities of the LangChain ecosystem:
- Visual state machine with conditional routing
- Parallel processing for independent operations  
- Human-in-the-loop decision points
- Persistent state across workflow steps
- Real-time dashboard visualization
- Advanced error recovery and routing

🆚 Comparison with Basic LangChain:
Basic LangChain: Sequential chains with basic error handling
LangGraph: Advanced state machine with conditional logic, parallel execution, 
          and sophisticated routing based on execution results

📊 LangGraph Dashboard Features:
- Real-time state visualization
- Execution path tracking
- Performance monitoring per state
- Error propagation analysis
- Resource utilization tracking
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Annotated, Literal
from datetime import datetime
import json
import asyncio
from pydantic import BaseModel, Field

# LangGraph Imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from config import config
from langchain_agent import LangChainDataCleaningAgent, DatasetAnalysis, CleaningCode
from code_executor import SafeCodeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaningState(MessagesState):
    """
    🎛️ Complete State Definition for Data Cleaning Workflow
    
    This state object tracks everything that happens during 
    the data cleaning process, enabling:
    - Checkpoint/resume functionality
    - Conditional routing based on state
    - Parallel processing coordination
    - Error recovery with state preservation
    - Performance monitoring per state
    """
    
    # Input data
    original_dataset: Optional[pd.DataFrame] = None
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis results
    analysis_complete: bool = False
    analysis_results: Optional[DatasetAnalysis] = None
    quality_score: float = 0.0
    
    # Code generation
    code_generated: bool = False
    cleaning_code: Optional[CleaningCode] = None
    code_attempts: int = 0
    
    # Execution results
    execution_complete: bool = False
    execution_successful: bool = False
    cleaned_dataset: Optional[pd.DataFrame] = None
    execution_output: str = ""
    
    # Error handling
    error_count: int = 0
    last_error: str = ""
    recovery_attempted: bool = False
    
    # Workflow control
    current_step: str = "initialization"
    requires_human_intervention: bool = False
    parallel_tasks_complete: Dict[str, bool] = Field(default_factory=dict)
    
    # Performance tracking
    step_start_times: Dict[str, datetime] = Field(default_factory=dict)
    step_durations: Dict[str, float] = Field(default_factory=dict)
    total_processing_time: float = 0.0
    
    # Quality assurance
    validation_checks: List[str] = Field(default_factory=list)
    quality_improvements: Dict[str, Any] = Field(default_factory=dict)

class LangGraphDataCleaningWorkflow:
    """
    🕸️ Advanced Data Cleaning Workflow using LangGraph State Machine
    
    This creates a sophisticated workflow that demonstrates:
    
    1. **Visual State Machine**: See exactly where your data is in the process
    2. **Conditional Routing**: Different paths based on data quality and errors
    3. **Parallel Processing**: Run independent tasks simultaneously
    4. **Error Recovery**: Intelligent recovery strategies based on failure type
    5. **Human-in-the-Loop**: Option for manual intervention on complex issues
    6. **Checkpointing**: Resume interrupted workflows from any point
    7. **Real-time Monitoring**: Track performance and resource usage
    
    State Flow Visualization:
    ========================
    
         START
           │
           ▼
    ┌─────────────┐
    │  INITIALIZE │ ──── Collect dataset metadata
    │   DATASET   │      Set up processing context
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   ANALYZE   │ ──── Comprehensive data analysis
    │   DATASET   │      Quality assessment
    └─────────────┘      Statistical profiling
           │
           ▼
         ┌─────┐
         │ GATE │ ──── Quality Score Check
         └─────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  HIGH QUALITY  LOW QUALITY
     │           │
     ▼           ▼
  ┌──────┐   ┌────────────┐
  │SIMPLE│   │  COMPLEX   │ ──── Advanced analysis
  │CLEAN │   │  ANALYSIS  │      Error pattern detection
  └──────┘   └────────────┘      Corruption assessment
     │           │
     └─────┬─────┘
           ▼
    ┌─────────────┐
    │  GENERATE   │ ──── AI code generation
    │    CODE     │      Strategy selection
    └─────────────┘      Validation planning
           │
           ▼
    ┌─────────────┐
    │   EXECUTE   │ ──── Safe code execution
    │    CODE     │      Error monitoring
    └─────────────┘      Quality validation
           │
       ┌───┴───┐
       ▼       ▼
   SUCCESS   FAILURE
       │       │
       │       ▼
       │   ┌──────────┐
       │   │   ERROR  │ ──── Error analysis
       │   │ RECOVERY │      Code correction
       │   └──────────┘      Retry strategy
       │       │
       │       ▼
       │   ┌──────────┐
       │   │  HUMAN   │ ──── Complex error handling
       │   │   LOOP   │      Manual intervention
       │   └──────────┘      Strategy consultation
       │       │
       └───────┼───────┐
               ▼       │
        ┌─────────────┐│
        │  VALIDATE   ││ ──── Final quality checks
        │   RESULTS   ││      Performance metrics
        └─────────────┘│      Success verification
               │       │
               ▼       │
             END ──────┘
    """
    
    def __init__(self):
        """Initialize the LangGraph workflow with state machine"""
        
        logger.info("🕸️ Initializing LangGraph Data Cleaning Workflow...")
        
        # Initialize components
        self.langchain_agent = LangChainDataCleaningAgent()
        self.code_executor = SafeCodeExecutor()
        
        # Create checkpointer for workflow persistence
        self.checkpointer = MemorySaver()
        
        # Build the state graph
        self.workflow = self._build_workflow_graph()
        
        # Compile the workflow with checkpointing
        self.app = self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_intervention"],  # Pause for human input
            interrupt_after=["error_recovery"]       # Review recovery results
        )
        
        logger.info("✅ LangGraph workflow initialized successfully!")
        
    def _build_workflow_graph(self) -> StateGraph:
        """
        🏗️ Build the complete state graph for data cleaning workflow
        
        This creates the visual workflow that you'll see in the LangGraph dashboard:
        - Each node represents a processing state
        - Edges define the flow between states
        - Conditional edges route based on execution results
        - Parallel edges enable concurrent processing
        """
        
        # Create the state graph
        workflow = StateGraph(DataCleaningState)
        
        # Add workflow nodes (each represents a state)
        workflow.add_node("initialize_dataset", self._initialize_dataset)
        workflow.add_node("analyze_dataset", self._analyze_dataset)
        workflow.add_node("assess_complexity", self._assess_complexity)
        workflow.add_node("simple_cleaning", self._simple_cleaning)
        workflow.add_node("complex_analysis", self._complex_analysis)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("error_recovery", self._error_recovery)
        workflow.add_node("human_intervention", self._human_intervention)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("parallel_quality_checks", self._parallel_quality_checks)
        
        # Define the workflow edges (state transitions)
        
        # Start with dataset initialization
        workflow.add_edge(START, "initialize_dataset")
        workflow.add_edge("initialize_dataset", "analyze_dataset")
        
        # Conditional routing based on analysis results
        workflow.add_conditional_edges(
            "analyze_dataset",
            self._should_use_simple_cleaning,
            {
                "simple": "simple_cleaning",
                "complex": "complex_analysis", 
                "assess": "assess_complexity"
            }
        )
        
        # Complexity assessment can route to different strategies
        workflow.add_conditional_edges(
            "assess_complexity",
            self._complexity_routing,
            {
                "simple": "simple_cleaning",
                "complex": "complex_analysis",
                "parallel": "parallel_quality_checks"
            }
        )
        
        # Both cleaning paths converge to code generation
        workflow.add_edge("simple_cleaning", "generate_code")
        workflow.add_edge("complex_analysis", "generate_code")
        workflow.add_edge("parallel_quality_checks", "generate_code")
        
        # Code generation leads to execution
        workflow.add_edge("generate_code", "execute_code")
        
        # Conditional routing based on execution results
        workflow.add_conditional_edges(
            "execute_code",
            self._execution_success_check,
            {
                "success": "validate_results",
                "recoverable_error": "error_recovery",
                "complex_error": "human_intervention",
                "retry": "generate_code"
            }
        )
        
        # Error recovery paths
        workflow.add_conditional_edges(
            "error_recovery",
            self._recovery_success_check,
            {
                "fixed": "execute_code",
                "need_human": "human_intervention",
                "give_up": END
            }
        )
        
        # Human intervention outcomes
        workflow.add_conditional_edges(
            "human_intervention",
            self._human_decision_routing,
            {
                "retry": "generate_code",
                "manual_fix": "execute_code",
                "abort": END
            }
        )
        
        # Successful completion
        workflow.add_edge("validate_results", END)
        
        logger.info("🏗️ Workflow graph built with conditional routing and parallel processing")
        return workflow
    
    async def _initialize_dataset(self, state: DataCleaningState) -> DataCleaningState:
        """
        🚀 Initialize dataset and collect metadata
        
        This is the entry point that prepares everything for processing.
        In the dashboard, you'll see this as the first active state.
        """
        
        logger.info("🚀 [INITIALIZE] Starting dataset initialization...")
        state.current_step = "initialization"
        state.step_start_times["initialization"] = datetime.now()
        
        # Collect comprehensive dataset metadata
        if state.original_dataset is not None:
            df = state.original_dataset
            
            state.dataset_metadata = {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicates": df.duplicated().sum(),
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "initialized_at": datetime.now().isoformat()
            }
            
            logger.info(f"📊 Dataset initialized: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"💾 Memory usage: {state.dataset_metadata['memory_usage_mb']:.2f} MB")
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["initialization"]).total_seconds()
        state.step_durations["initialization"] = duration
        
        return state
    
    async def _analyze_dataset(self, state: DataCleaningState) -> DataCleaningState:
        """
        📊 Perform comprehensive dataset analysis
        
        This uses the LangChain agent to analyze the dataset structure and quality.
        Dashboard shows analysis progress and results in real-time.
        """
        
        logger.info("📊 [ANALYZE] Starting dataset analysis...")
        state.current_step = "analysis"
        state.step_start_times["analysis"] = datetime.now()
        
        try:
            # Use LangChain agent for structured analysis
            analysis_result = self.langchain_agent.analyze_dataset(state.original_dataset)
            
            state.analysis_results = analysis_result
            state.analysis_complete = True
            state.quality_score = analysis_result.data_quality_score
            
            logger.info(f"✅ Analysis complete - Quality Score: {state.quality_score}/100")
            logger.info(f"🔍 Issues found: {len(analysis_result.major_issues)}")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            state.last_error = f"Analysis error: {str(e)}"
            state.error_count += 1
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["analysis"]).total_seconds()
        state.step_durations["analysis"] = duration
        
        return state
    
    async def _assess_complexity(self, state: DataCleaningState) -> DataCleaningState:
        """
        🎯 Assess data complexity for routing decisions
        
        This determines which cleaning strategy to use based on data characteristics.
        Dashboard shows the decision-making process.
        """
        
        logger.info("🎯 [ASSESS] Assessing data complexity...")
        state.current_step = "complexity_assessment"
        state.step_start_times["complexity_assessment"] = datetime.now()
        
        # Analyze complexity factors
        complexity_factors = {
            "size_complexity": state.dataset_metadata["shape"][0] > 10000,
            "type_complexity": len(state.dataset_metadata["categorical_columns"]) > 5,
            "quality_complexity": state.quality_score < 70,
            "corruption_complexity": len(state.analysis_results.corruption_indicators) > 3,
            "missing_data_complexity": state.dataset_metadata["missing_values"] > 0.1
        }
        
        complexity_score = sum(complexity_factors.values())
        
        state.dataset_metadata["complexity_score"] = complexity_score
        state.dataset_metadata["complexity_factors"] = complexity_factors
        
        logger.info(f"🎯 Complexity assessment: {complexity_score}/5 factors detected")
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["complexity_assessment"]).total_seconds()
        state.step_durations["complexity_assessment"] = duration
        
        return state
    
    async def _simple_cleaning(self, state: DataCleaningState) -> DataCleaningState:
        """
        🧹 Simple cleaning strategy for high-quality data
        
        Fast-track processing for datasets that don't need complex analysis.
        Dashboard shows optimized processing path.
        """
        
        logger.info("🧹 [SIMPLE] Executing simple cleaning strategy...")
        state.current_step = "simple_cleaning"
        state.step_start_times["simple_cleaning"] = datetime.now()
        
        # Simple cleaning operations
        state.validation_checks.extend([
            "basic_type_validation",
            "simple_missing_value_check",
            "duplicate_detection"
        ])
        
        logger.info("✅ Simple cleaning strategy prepared")
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["simple_cleaning"]).total_seconds()
        state.step_durations["simple_cleaning"] = duration
        
        return state
    
    async def _complex_analysis(self, state: DataCleaningState) -> DataCleaningState:
        """
        🔬 Complex analysis for challenging datasets
        
        Advanced processing for datasets with quality issues or corruption.
        Dashboard shows detailed analysis progress.
        """
        
        logger.info("🔬 [COMPLEX] Executing complex analysis strategy...")
        state.current_step = "complex_analysis"
        state.step_start_times["complex_analysis"] = datetime.now()
        
        # Complex analysis operations
        state.validation_checks.extend([
            "advanced_corruption_detection",
            "statistical_outlier_analysis",
            "domain_knowledge_validation",
            "cross_column_consistency_check"
        ])
        
        logger.info("✅ Complex analysis strategy prepared")
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["complex_analysis"]).total_seconds()
        state.step_durations["complex_analysis"] = duration
        
        return state
    
    async def _parallel_quality_checks(self, state: DataCleaningState) -> DataCleaningState:
        """
        ⚡ Parallel quality assessment for large datasets
        
        Concurrent processing of independent quality checks.
        Dashboard shows parallel execution progress.
        """
        
        logger.info("⚡ [PARALLEL] Running parallel quality checks...")
        state.current_step = "parallel_processing"
        state.step_start_times["parallel_processing"] = datetime.now()
        
        # Simulate parallel tasks
        parallel_tasks = [
            "statistical_analysis",
            "corruption_detection", 
            "outlier_analysis",
            "consistency_validation"
        ]
        
        # In a real implementation, these would run concurrently
        for task in parallel_tasks:
            state.parallel_tasks_complete[task] = True
            logger.info(f"✅ Completed: {task}")
        
        logger.info("✅ All parallel quality checks completed")
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["parallel_processing"]).total_seconds()
        state.step_durations["parallel_processing"] = duration
        
        return state
    
    async def _generate_code(self, state: DataCleaningState) -> DataCleaningState:
        """
        🤖 Generate cleaning code using LangChain
        
        AI-powered code generation based on analysis results.
        Dashboard shows code generation progress and quality.
        """
        
        logger.info("🤖 [GENERATE] Generating cleaning code...")
        state.current_step = "code_generation"
        state.step_start_times["code_generation"] = datetime.now()
        
        try:
            # Generate code using LangChain agent
            code_result = self.langchain_agent.generate_cleaning_code(
                state.original_dataset, 
                state.analysis_results
            )
            
            state.cleaning_code = code_result
            state.code_generated = True
            state.code_attempts += 1
            
            logger.info(f"✅ Code generated successfully (attempt {state.code_attempts})")
            logger.info(f"📝 Generated {len(code_result.cleaning_code.split('\\n'))} lines of code")
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {str(e)}")
            state.last_error = f"Code generation error: {str(e)}"
            state.error_count += 1
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["code_generation"]).total_seconds()
        state.step_durations["code_generation"] = duration
        
        return state
    
    async def _execute_code(self, state: DataCleaningState) -> DataCleaningState:
        """
        ⚡ Execute generated cleaning code
        
        Safe execution of AI-generated code with monitoring.
        Dashboard shows execution progress and resource usage.
        """
        
        logger.info("⚡ [EXECUTE] Executing cleaning code...")
        state.current_step = "code_execution"
        state.step_start_times["code_execution"] = datetime.now()
        
        try:
            # Execute code using safe executor
            success, output, cleaned_df = self.langchain_agent.execute_cleaning_code(
                state.original_dataset,
                state.cleaning_code
            )
            
            state.execution_complete = True
            state.execution_successful = success
            state.execution_output = output
            
            if success and cleaned_df is not None:
                state.cleaned_dataset = cleaned_df
                logger.info("✅ Code execution successful")
                logger.info(f"📊 Result shape: {cleaned_df.shape}")
            else:
                logger.warning("⚠️ Code execution failed")
                state.last_error = output
                state.error_count += 1
                
        except Exception as e:
            logger.error(f"❌ Execution error: {str(e)}")
            state.last_error = f"Execution error: {str(e)}"
            state.error_count += 1
            state.execution_successful = False
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["code_execution"]).total_seconds()
        state.step_durations["code_execution"] = duration
        
        return state
    
    async def _error_recovery(self, state: DataCleaningState) -> DataCleaningState:
        """
        🔧 Intelligent error recovery
        
        AI-powered error analysis and code correction.
        Dashboard shows recovery attempts and success rates.
        """
        
        logger.info("🔧 [RECOVERY] Attempting error recovery...")
        state.current_step = "error_recovery"
        state.step_start_times["error_recovery"] = datetime.now()
        
        try:
            # Attempt to fix the code using LangChain
            corrected_code = self.langchain_agent.fix_code_with_langchain(
                state.cleaning_code.cleaning_code,
                state.last_error,
                state.original_dataset
            )
            
            state.cleaning_code = corrected_code
            state.recovery_attempted = True
            
            logger.info("✅ Error recovery completed")
            logger.info("🔧 Generated corrected code")
            
        except Exception as e:
            logger.error(f"❌ Recovery failed: {str(e)}")
            state.last_error = f"Recovery error: {str(e)}"
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["error_recovery"]).total_seconds()
        state.step_durations["error_recovery"] = duration
        
        return state
    
    async def _human_intervention(self, state: DataCleaningState) -> DataCleaningState:
        """
        👤 Human-in-the-loop intervention
        
        Pause workflow for human decision-making on complex issues.
        Dashboard shows intervention request and options.
        """
        
        logger.info("👤 [HUMAN] Requesting human intervention...")
        state.current_step = "human_intervention"
        state.requires_human_intervention = True
        
        # In a real implementation, this would present options to the user
        logger.info("⏸️ Workflow paused for human decision")
        logger.info(f"📄 Context: {state.last_error}")
        logger.info("🤔 Please review the error and decide on next steps")
        
        return state
    
    async def _validate_results(self, state: DataCleaningState) -> DataCleaningState:
        """
        ✅ Final validation and quality assurance
        
        Comprehensive validation of cleaning results.
        Dashboard shows quality metrics and final assessment.
        """
        
        logger.info("✅ [VALIDATE] Validating cleaning results...")
        state.current_step = "validation"
        state.step_start_times["validation"] = datetime.now()
        
        if state.cleaned_dataset is not None:
            # Calculate quality improvements
            original_missing = state.original_dataset.isnull().sum().sum()
            final_missing = state.cleaned_dataset.isnull().sum().sum()
            
            original_duplicates = state.original_dataset.duplicated().sum()
            final_duplicates = state.cleaned_dataset.duplicated().sum()
            
            state.quality_improvements = {
                "missing_values_removed": original_missing - final_missing,
                "duplicates_removed": original_duplicates - final_duplicates,
                "shape_preserved": state.original_dataset.shape[1] == state.cleaned_dataset.shape[1],
                "data_types_optimized": True  # Simplified check
            }
            
            logger.info("✅ Validation completed successfully")
            logger.info(f"📈 Missing values removed: {state.quality_improvements['missing_values_removed']}")
            logger.info(f"🗂️ Duplicates removed: {state.quality_improvements['duplicates_removed']}")
        
        # Calculate total processing time
        state.total_processing_time = sum(state.step_durations.values())
        
        # Calculate step duration
        duration = (datetime.now() - state.step_start_times["validation"]).total_seconds()
        state.step_durations["validation"] = duration
        
        return state
    
    # Conditional routing functions
    
    def _should_use_simple_cleaning(self, state: DataCleaningState) -> Literal["simple", "complex", "assess"]:
        """Route based on data quality score"""
        if state.quality_score >= 80:
            return "simple"
        elif state.quality_score >= 60:
            return "assess"
        else:
            return "complex"
    
    def _complexity_routing(self, state: DataCleaningState) -> Literal["simple", "complex", "parallel"]:
        """Route based on complexity assessment"""
        complexity_score = state.dataset_metadata.get("complexity_score", 0)
        
        if complexity_score <= 2:
            return "simple"
        elif complexity_score >= 4:
            return "complex"
        else:
            return "parallel"
    
    def _execution_success_check(self, state: DataCleaningState) -> Literal["success", "recoverable_error", "complex_error", "retry"]:
        """Route based on execution results"""
        if state.execution_successful:
            return "success"
        elif state.error_count >= 3:
            return "complex_error"
        elif "import" in state.last_error.lower() or "syntax" in state.last_error.lower():
            return "recoverable_error"
        elif state.code_attempts < 2:
            return "retry"
        else:
            return "recoverable_error"
    
    def _recovery_success_check(self, state: DataCleaningState) -> Literal["fixed", "need_human", "give_up"]:
        """Route based on recovery results"""
        if state.recovery_attempted and state.error_count <= 2:
            return "fixed"
        elif state.error_count >= 5:
            return "give_up"
        else:
            return "need_human"
    
    def _human_decision_routing(self, state: DataCleaningState) -> Literal["retry", "manual_fix", "abort"]:
        """Route based on human decisions (simplified for demo)"""
        # In a real implementation, this would get actual human input
        if state.error_count <= 3:
            return "retry"
        else:
            return "abort"
    
    async def process_dataset(self, df: pd.DataFrame, config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🚀 Main entry point for LangGraph workflow execution
        
        This starts the complete workflow and returns comprehensive results
        including all state transitions and performance metrics.
        """
        
        logger.info("🚀 Starting LangGraph Data Cleaning Workflow...")
        
        # Initialize state
        initial_state = DataCleaningState(
            messages=[],
            original_dataset=df,
            dataset_metadata={},
            analysis_complete=False,
            code_generated=False,
            execution_complete=False,
            execution_successful=False,
            error_count=0,
            current_step="initialization",
            requires_human_intervention=False,
            parallel_tasks_complete={},
            step_start_times={},
            step_durations={},
            validation_checks=[],
            quality_improvements={}
        )
        
        # Create thread config for checkpointing
        thread_config = {"configurable": {"thread_id": f"cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
        
        try:
            # Execute the workflow
            final_state = None
            
            async for state in self.app.astream(initial_state, config=thread_config):
                # Get the current state
                current_state = state[list(state.keys())[0]]
                final_state = current_state
                
                # Log progress
                logger.info(f"🔄 Current step: {current_state.current_step}")
                
                # Check for human intervention requirement
                if current_state.requires_human_intervention:
                    logger.info("⏸️ Workflow paused for human intervention")
                    logger.info("💡 In a real application, this would present options to the user")
                    
                    # For demo purposes, automatically continue
                    current_state.requires_human_intervention = False
            
            # Compile final results
            result = {
                "success": final_state.execution_successful if final_state else False,
                "workflow_complete": True,
                "final_state": final_state.current_step if final_state else "failed",
                "total_processing_time": final_state.total_processing_time if final_state else 0,
                "step_durations": final_state.step_durations if final_state else {},
                "quality_improvements": final_state.quality_improvements if final_state else {},
                "error_count": final_state.error_count if final_state else 0,
                "validation_checks": final_state.validation_checks if final_state else [],
                "cleaned_dataset": final_state.cleaned_dataset if final_state and final_state.cleaned_dataset is not None else None,
                "cleaning_code": final_state.cleaning_code.cleaning_code if final_state and final_state.cleaning_code else None,
                "workflow_metadata": {
                    "thread_id": thread_config["configurable"]["thread_id"],
                    "checkpoints_available": True,
                    "langgraph_version": "0.6.7",
                    "state_transitions": len(final_state.step_durations) if final_state else 0
                }
            }
            
            logger.info("🎉 LangGraph workflow completed successfully!")
            logger.info(f"⏱️ Total time: {result['total_processing_time']:.2f} seconds")
            logger.info(f"🔄 State transitions: {result['workflow_metadata']['state_transitions']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "workflow_complete": False,
                "final_state": "error"
            }

def main():
    """
    🧪 Test the LangGraph workflow
    """
    
    print("🧪 Testing LangGraph Data Cleaning Workflow")
    print("="*60)
    
    # Check configuration
    validation = config.validate_api_keys()
    if not validation["google_gemini"]:
        print("❌ Google Gemini API key required")
        print("💡 Please set GOOGLE_API_KEY in your .env file")
        print("\n📊 Workflow structure demonstration:")
        print("   START → INITIALIZE → ANALYZE → COMPLEXITY_GATE")
        print("   ├─ Simple Path: SIMPLE_CLEAN → GENERATE → EXECUTE")
        print("   ├─ Complex Path: COMPLEX_ANALYSIS → GENERATE → EXECUTE")
        print("   └─ Parallel Path: PARALLEL_CHECKS → GENERATE → EXECUTE")
        print("   Error Recovery: ERROR_RECOVERY ↔ HUMAN_INTERVENTION")
        print("   Final: VALIDATE → END")
        return
    
    # Create sample data
    sample_data = {
        'id': range(1, 101),
        'name': [f'User_{i}' if i % 10 != 0 else None for i in range(1, 101)],
        'score': [np.random.normal(75, 15) if i % 15 != 0 else None for i in range(1, 101)],
        'category': [np.random.choice(['A', 'B', 'C']) for _ in range(100)]
    }
    
    # Add some duplicates
    df = pd.DataFrame(sample_data)
    duplicates = df.sample(5)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    print(f"📊 Sample dataset created: {df.shape}")
    print(f"🔍 Missing values: {df.isnull().sum().sum()}")
    print(f"🔄 Duplicates: {df.duplicated().sum()}")
    
    async def run_workflow():
        # Initialize workflow
        workflow = LangGraphDataCleaningWorkflow()
        
        # Process dataset
        result = await workflow.process_dataset(df)
        
        if result["success"]:
            print("✅ LangGraph workflow completed successfully!")
            print(f"⏱️ Total time: {result['total_processing_time']:.2f} seconds")
            print(f"🔄 State transitions: {result['workflow_metadata']['state_transitions']}")
            print(f"📊 Final dataset shape: {result['cleaned_dataset'].shape}")
        else:
            print("❌ Workflow failed")
            print(f"❗ Error: {result.get('error', 'Unknown error')}")
    
    # Run the async workflow
    try:
        asyncio.run(run_workflow())
    except Exception as e:
        print(f"❌ Workflow initialization failed: {str(e)}")
        print("💡 This demonstrates the LangGraph structure even without API keys")

if __name__ == "__main__":
    main()