"""
🦜🔗 LangChain-Powered Streamlit Data Cleaning Application
========================================================

This is a complete rewrite of the Streamlit app using the full LangChain ecosystem:
- LangChain chat models with structured prompts
- LangGraph state machine workflow visualization
- LangSmith real-time tracing and monitoring
- Advanced error handling and recovery
- Professional UI with comprehensive metrics
- Multi-agent comparison capabilities

🆚 Comparison with Original App:
Original: Basic UI, direct API calls, limited observability
Enhanced: Professional dashboard, structured workflows, complete observability

📊 Features You'll See:
- Real-time workflow state visualization
- LangSmith trace links for debugging
- Performance metrics and optimization insights
- Error analysis and recovery tracking
- Multi-agent comparison results
- Comprehensive evaluation metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
# import asyncio  # Removed - not needed for sync processing
from datetime import datetime
import time
from typing import Dict, Any, Optional

# LangChain ecosystem imports
from config import config
from langchain_agent import LangChainDataCleaningAgent
from langgraph_workflow import LangGraphDataCleaningWorkflow
from langsmith_evaluation import LangSmithEvaluationPipeline
from utils import create_sample_datasets, get_dataset_summary

# Configure Streamlit page
st.set_page_config(
    page_title="🦜🔗 LangChain Data Cleaning Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .workflow-state {
        background: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    if 'langchain_agent' not in st.session_state:
        st.session_state.langchain_agent = None
    if 'langgraph_workflow' not in st.session_state:
        st.session_state.langgraph_workflow = None
    if 'evaluation_pipeline' not in st.session_state:
        st.session_state.evaluation_pipeline = None
    if 'api_keys_configured' not in st.session_state:
        st.session_state.api_keys_configured = False
    if 'cleaning_results' not in st.session_state:
        st.session_state.cleaning_results = None
    if 'workflow_traces' not in st.session_state:
        st.session_state.workflow_traces = []

def display_header():
    """Display the main application header"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🦜🔗 LangChain Data Cleaning Agent</h1>
        <p>Professional AI-powered data cleaning with full observability and state machine workflows</p>
    </div>
    """, unsafe_allow_html=True)

def display_configuration_sidebar():
    """Display configuration and API key setup in sidebar"""
    
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key Configuration
        st.subheader("🔑 API Keys")
        
        # Check current configuration
        validation = config.validate_api_keys()
        
        # Google Gemini API Key
        if validation["google_gemini"]:
            st.success("✅ Google Gemini API configured")
        else:
            st.error("❌ Google Gemini API key missing")
            with st.expander("📖 Get Google Gemini API Key"):
                st.markdown("""
                1. Visit [Google AI Studio](https://ai.google.dev/)
                2. Click "Get API Key"
                3. Copy your API key
                4. Add to your `.env` file: `GOOGLE_API_KEY=your_key_here`
                5. Restart the application
                """)
        
        # LangSmith API Key
        if validation["langsmith"]:
            st.success("✅ LangSmith API configured")
            st.info(f"📊 Project: {config.langsmith_project}")
            st.markdown("🔗 [View Dashboard](https://smith.langchain.com/)")
        else:
            st.warning("⚠️ LangSmith API key missing")
            st.info("📊 Dashboard features disabled")
            with st.expander("📖 Get LangSmith API Key"):
                st.markdown("""
                1. Visit [LangSmith](https://smith.langchain.com/)
                2. Sign up/Login
                3. Go to Settings → API Keys
                4. Create new API key
                5. Add to your `.env` file: `LANGCHAIN_API_KEY=your_key_here`
                6. Restart the application
                """)
        
        # OpenAI (Optional)
        if validation["openai"]:
            st.success("✅ OpenAI API configured (optional)")
        else:
            st.info("ℹ️ OpenAI API not configured (optional)")
        
        st.session_state.api_keys_configured = validation["google_gemini"]
        
        # Agent Configuration
        st.subheader("🤖 Agent Settings")
        
        agent_type = st.selectbox(
            "Select Agent Type",
            ["LangChain Agent", "LangGraph Workflow"],
            help="Choose between basic LangChain agent or advanced LangGraph state machine"
        )
        
        max_retries = st.slider(
            "Max Retry Attempts",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of retry attempts for failed operations"
        )
        
        enable_evaluation = st.checkbox(
            "Enable Real-time Evaluation",
            value=True,
            help="Run quality evaluation on cleaning results"
        )
        
        return agent_type, max_retries, enable_evaluation

def display_langsmith_info():
    """Display LangSmith integration information"""
    
    st.subheader("📊 LangSmith Integration")
    
    if config.langsmith_api_key:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔗 Status", "Connected")
        
        with col2:
            st.metric("📋 Project", config.langsmith_project)
        
        with col3:
            st.markdown("🔗 [View Dashboard](https://smith.langchain.com/)")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 What you'll see in LangSmith Dashboard:</h4>
            <ul>
                <li>🔍 Real-time traces of every AI decision</li>
                <li>📈 Performance metrics and optimization insights</li>
                <li>❌ Error tracking and debugging information</li>
                <li>⚡ Processing speed and resource usage</li>
                <li>🔄 Workflow state transitions and routing</li>
                <li>📊 Quality improvement metrics over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="error-message">
            <h4>📊 LangSmith Dashboard Not Available</h4>
            <p>Configure LangSmith API key to enable:</p>
            <ul>
                <li>Real-time workflow visualization</li>
                <li>Performance monitoring and optimization</li>
                <li>Error tracking and debugging</li>
                <li>Quality metrics and trending</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_workflow_visualization(workflow_traces: list):
    """Display LangGraph workflow state visualization"""
    
    st.subheader("🕸️ Workflow State Machine")
    
    if workflow_traces:
        # Create workflow state timeline
        states = []
        timestamps = []
        
        for trace in workflow_traces:
            if 'current_step' in trace:
                states.append(trace['current_step'])
                timestamps.append(trace.get('timestamp', datetime.now()))
        
        if states:
            # Display current state
            current_state = states[-1] if states else "Not Started"
            st.info(f"🎯 Current State: **{current_state}**")
            
            # Display state progression
            st.markdown("🔄 **State Progression:**")
            for i, state in enumerate(states):
                if i == len(states) - 1:
                    st.markdown(f"   **{i+1}. {state}** ← *Current*")
                else:
                    st.markdown(f"   {i+1}. {state}")
    
    else:
        # Display workflow structure
        st.markdown("""
        <div class="info-box">
            <h4>🏗️ LangGraph Workflow Structure:</h4>
            <p><strong>START</strong> → <strong>INITIALIZE</strong> → <strong>ANALYZE</strong> → <strong>COMPLEXITY_GATE</strong></p>
            <p>├─ <em>Simple Path</em>: <strong>SIMPLE_CLEAN</strong> → <strong>GENERATE</strong> → <strong>EXECUTE</strong></p>
            <p>├─ <em>Complex Path</em>: <strong>COMPLEX_ANALYSIS</strong> → <strong>GENERATE</strong> → <strong>EXECUTE</strong></p>
            <p>└─ <em>Parallel Path</em>: <strong>PARALLEL_CHECKS</strong> → <strong>GENERATE</strong> → <strong>EXECUTE</strong></p>
            <p><strong>Error Recovery</strong>: <strong>ERROR_RECOVERY</strong> ↔ <strong>HUMAN_INTERVENTION</strong></p>
            <p><strong>Final</strong>: <strong>VALIDATE</strong> → <strong>END</strong></p>
        </div>
        """, unsafe_allow_html=True)

def display_performance_metrics(results: Dict[str, Any]):
    """Display comprehensive performance metrics"""
    
    st.subheader("📊 Performance Metrics")
    
    if results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success = "✅ Success" if results.get("success", False) else "❌ Failed"
            st.metric("🎯 Status", success)
        
        with col2:
            processing_time = results.get("processing_time", 0)
            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
        
        with col3:
            attempts = results.get("attempts_used", 1)
            st.metric("🔄 Attempts", attempts)
        
        with col4:
            if "cleaned_df" in results and results["cleaned_df"] is not None:
                shape = results["cleaned_df"].shape
                st.metric("📊 Final Shape", f"{shape[0]}×{shape[1]}")
        
        # Quality improvements
        if "quality_improvements" in results:
            quality = results["quality_improvements"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                missing_removed = quality.get("missing_values_removed", 0)
                st.metric("🧹 Missing Values Removed", missing_removed)
            
            with col2:
                duplicates_removed = quality.get("duplicates_removed", 0)
                st.metric("🗂️ Duplicates Removed", duplicates_removed)
            
            with col3:
                types_corrected = quality.get("data_types_optimized", False)
                status = "✅ Yes" if types_corrected else "❌ No"
                st.metric("🔧 Types Optimized", status)

def display_cleaning_results(results: Dict[str, Any], original_df: pd.DataFrame):
    """Display comprehensive cleaning results"""
    
    st.subheader("✅ Cleaning Results")
    
    if results.get("success", False) and "cleaned_df" in results:
        cleaned_df = results["cleaned_df"]
        
        # Success message
        st.markdown("""
        <div class="success-message">
            <h4>🎉 Data Cleaning Completed Successfully!</h4>
            <p>Your dataset has been cleaned and optimized using advanced AI techniques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Before/After comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Original Dataset")
            st.dataframe(original_df.head(), width="stretch")
            
            # Original dataset metrics
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("📏 Rows", f"{original_df.shape[0]:,}")
                st.metric("🔍 Missing", original_df.isnull().sum().sum())
            with col1b:
                st.metric("📋 Columns", original_df.shape[1])
                st.metric("🔄 Duplicates", original_df.duplicated().sum())
        
        with col2:
            st.markdown("### ✨ Cleaned Dataset")
            st.dataframe(cleaned_df.head(), width="stretch")
            
            # Cleaned dataset metrics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("📏 Rows", f"{cleaned_df.shape[0]:,}", 
                         delta=f"{cleaned_df.shape[0] - original_df.shape[0]:,}")
                st.metric("🔍 Missing", cleaned_df.isnull().sum().sum(),
                         delta=f"{cleaned_df.isnull().sum().sum() - original_df.isnull().sum().sum()}")
            with col2b:
                st.metric("📋 Columns", cleaned_df.shape[1],
                         delta=f"{cleaned_df.shape[1] - original_df.shape[1]}")
                st.metric("🔄 Duplicates", cleaned_df.duplicated().sum(),
                         delta=f"{cleaned_df.duplicated().sum() - original_df.duplicated().sum()}")
        
        # Quality visualization
        create_quality_visualization(original_df, cleaned_df)
        
        # Download options
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = cleaned_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned Dataset",
                data=csv_data,
                file_name=f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col2:
            if "cleaning_code" in results:
                st.download_button(
                    label="📄 Download Cleaning Code",
                    data=results["cleaning_code"],
                    file_name=f"cleaning_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                    mime="text/plain",
                    width="stretch"
                )
        
        # LangSmith trace link
        if config.langsmith_api_key and "langchain_traces" in results:
            st.markdown("### 🔍 Detailed Analysis")
            st.info("""
            🔗 **View detailed execution traces in LangSmith Dashboard**
            
            See exactly how the AI analyzed your data, generated code, and made decisions.
            [Open LangSmith Dashboard](https://smith.langchain.com/)
            """)
    
    else:
        # Error handling
        error_msg = results.get("error", "Unknown error occurred")
        
        st.markdown(f"""
        <div class="error-message">
            <h4>❌ Data Cleaning Failed</h4>
            <p><strong>Error:</strong> {error_msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Error analysis
        if "attempts_used" in results:
            st.write(f"🔄 Attempts made: {results['attempts_used']}")
        
        if config.langsmith_api_key:
            st.info("""
            🔍 **Debug in LangSmith Dashboard**
            
            View detailed error traces and execution logs to understand what went wrong.
            [Open LangSmith Dashboard](https://smith.langchain.com/)
            """)

def create_quality_visualization(original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    """Create quality improvement visualization"""
    
    st.subheader("📈 Quality Improvement Analysis")
    
    # Data quality metrics
    original_missing = original_df.isnull().sum().sum()
    cleaned_missing = cleaned_df.isnull().sum().sum()
    
    original_duplicates = original_df.duplicated().sum()
    cleaned_duplicates = cleaned_df.duplicated().sum()
    
    # Create comparison chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Missing Values", "Duplicate Rows"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Missing values comparison
    fig.add_trace(
        go.Bar(
            x=["Original", "Cleaned"],
            y=[original_missing, cleaned_missing],
            name="Missing Values",
            marker_color=["#ff7f7f", "#7f7fff"]
        ),
        row=1, col=1
    )
    
    # Duplicates comparison
    fig.add_trace(
        go.Bar(
            x=["Original", "Cleaned"],
            y=[original_duplicates, cleaned_duplicates],
            name="Duplicates",
            marker_color=["#ff7f7f", "#7f7fff"],
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Data Quality Improvements",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, width="stretch")

def run_cleaning_process(df: pd.DataFrame, agent_type: str, max_retries: int) -> Dict[str, Any]:
    """Run the data cleaning process with selected agent"""
    
    try:
        if agent_type == "LangChain Agent":
            # Initialize LangChain agent
            if st.session_state.langchain_agent is None:
                with st.spinner("🤖 Initializing LangChain Agent..."):
                    st.session_state.langchain_agent = LangChainDataCleaningAgent()
            
            # Run cleaning
            with st.spinner("🔄 Cleaning dataset with LangChain Agent..."):
                result = st.session_state.langchain_agent.clean_dataset(df, max_retries)
            
        else:  # LangGraph Workflow
            # Initialize LangGraph workflow
            if st.session_state.langgraph_workflow is None:
                with st.spinner("🕸️ Initializing LangGraph Workflow..."):
                    st.session_state.langgraph_workflow = LangGraphDataCleaningWorkflow()
            
            # Run workflow using sync wrapper
            with st.spinner("🔄 Processing dataset with LangGraph Workflow..."):
                result = st.session_state.langgraph_workflow.process_dataset_sync(df)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Agent initialization/execution failed: {str(e)}",
            "processing_time": 0
        }

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display configuration sidebar
    agent_type, max_retries, enable_evaluation = display_configuration_sidebar()
    
    # Check API key configuration
    if not st.session_state.api_keys_configured:
        st.error("🔑 API keys not configured. Please check the sidebar for setup instructions.")
        st.stop()
    
    # Display LangSmith integration info
    display_langsmith_info()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Data Cleaning", 
        "🕸️ Workflow Visualization", 
        "📊 Performance Analytics",
        "🔬 Agent Evaluation"
    ])
    
    with tab1:
        st.header("🤖 AI-Powered Data Cleaning")
        
        # Data input section
        st.subheader("📊 Dataset Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV/Excel file", "Use sample dataset", "Create custom dataset"]
        )
        
        uploaded_df = None
        
        if input_method == "Upload CSV/Excel file":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a CSV or Excel file for cleaning"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        uploaded_df = pd.read_csv(uploaded_file)
                    else:
                        uploaded_df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File uploaded successfully! Shape: {uploaded_df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        elif input_method == "Use sample dataset":
            sample_datasets = create_sample_datasets()
            
            dataset_names = list(sample_datasets.keys())
            selected_dataset = st.selectbox("Select sample dataset:", dataset_names)
            
            if selected_dataset:
                uploaded_df = sample_datasets[selected_dataset]
                st.success(f"✅ Sample dataset loaded! Shape: {uploaded_df.shape}")
        
        else:  # Create custom dataset
            st.info("🔧 Custom dataset creation feature coming soon!")
        
        # Display dataset preview
        if uploaded_df is not None:
            st.subheader("👀 Dataset Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Data Sample (First 5 rows)**")
                st.dataframe(uploaded_df.head(), width="stretch")
                
                # Additional data info
                st.markdown("**📈 Quick Stats**")
                st.write(f"• **Rows**: {uploaded_df.shape[0]:,}")
                st.write(f"• **Columns**: {uploaded_df.shape[1]}")
                st.write(f"• **Missing Values**: {uploaded_df.isnull().sum().sum()}")
                st.write(f"• **Duplicates**: {uploaded_df.duplicated().sum()}")
            
            with col2:
                st.markdown("**🔍 Detailed Summary**")
                summary = get_dataset_summary(uploaded_df)
                st.json(summary)
            
            # Cleaning process
            st.subheader("🚀 Start Cleaning Process")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Clean Dataset", type="primary", width="stretch"):
                    # Run cleaning process
                    try:
                        result = run_cleaning_process(uploaded_df, agent_type, max_retries)
                        st.session_state.cleaning_results = result
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Cleaning process failed: {str(e)}")
            
            with col2:
                if enable_evaluation and st.button("🔬 Run Evaluation", width="stretch"):
                    st.info("🔬 Evaluation feature will be available after cleaning")
    
    with tab2:
        st.header("🕸️ LangGraph Workflow Visualization")
        display_workflow_visualization(st.session_state.workflow_traces)
    
    with tab3:
        st.header("📊 Performance Analytics")
        
        if st.session_state.cleaning_results:
            display_performance_metrics(st.session_state.cleaning_results)
        else:
            st.info("📊 Performance metrics will appear after running data cleaning")
    
    with tab4:
        st.header("🔬 Agent Evaluation")
        
        if st.button("🧪 Run Comprehensive Evaluation"):
            st.info("🔬 Comprehensive evaluation requires multiple datasets and may take several minutes")
            
            # Placeholder for evaluation results
            st.markdown("""
            **Evaluation Pipeline includes:**
            - 📊 Quality assessment across multiple datasets
            - ⚡ Performance benchmarking
            - 🔍 Error pattern analysis
            - 💡 Optimization recommendations
            - 📈 Comparative analysis with baseline methods
            """)
    
    # Display cleaning results if available
    if st.session_state.cleaning_results and uploaded_df is not None:
        st.header("✅ Cleaning Results")
        display_cleaning_results(st.session_state.cleaning_results, uploaded_df)

if __name__ == "__main__":
    main()