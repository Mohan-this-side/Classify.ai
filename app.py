import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
import traceback

# Import our custom modules
from data_cleaning_agent import DataCleaningAgent
from code_executor import SafeCodeExecutor
from utils import smart_read_csv, get_dataset_summary, create_download_link, validate_api_key, format_code_for_download, create_sample_datasets

# Page configuration
st.set_page_config(
    page_title="AI Data Cleaning Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'cleaning_result' not in st.session_state:
        st.session_state.cleaning_result = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def main():
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Cleaning Agent</h1>
        <p>Powered by Google Gemini Flash 2.0 | Intelligent Dataset Cleaning & Code Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key. Get one from https://ai.google.dev/"
        )
        
        if api_key:
            if validate_api_key(api_key):
                st.success("✅ API Key validated")
                if st.session_state.agent is None:
                    st.session_state.agent = DataCleaningAgent(api_key)
            else:
                st.error("❌ Invalid API Key format")
        
        st.divider()
        
        # Sample datasets
        st.subheader("📊 Try Sample Datasets")
        if st.button("Load Customer Dataset"):
            sample_data = create_sample_datasets()
            st.session_state.original_df = sample_data['customers']
            st.rerun()
        
        if st.button("Load Sales Dataset"):
            sample_data = create_sample_datasets()
            st.session_state.original_df = sample_data['sales']
            st.rerun()
        
        st.divider()
        
        # Help section
        st.subheader("❓ How to Use")
        st.markdown("""
        1. **Enter API Key**: Add your Gemini API key
        2. **Upload Dataset**: Choose CSV/Excel file
        3. **Start Cleaning**: Click 'Clean Dataset'
        4. **Review Results**: Check analysis & code
        5. **Download**: Get cleaned data & code
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Dataset Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file for cleaning"
        )
        
        if uploaded_file is not None:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Read the dataset
                if uploaded_file.name.endswith('.csv'):
                    df = smart_read_csv(tmp_path)
                else:
                    df = pd.read_excel(tmp_path)
                
                st.session_state.original_df = df
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
        
        # Display dataset preview
        if st.session_state.original_df is not None:
            st.subheader("📋 Dataset Preview")
            st.dataframe(st.session_state.original_df.head(), width='stretch')
            
            # Dataset summary
            with st.expander("📊 Dataset Summary"):
                summary = get_dataset_summary(st.session_state.original_df)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Rows", summary['basic_info']['shape'][0])
                with col_b:
                    st.metric("Columns", summary['basic_info']['shape'][1])
                with col_c:
                    st.metric("Missing Values", summary['missing_values']['total'])
                
                st.write("**Data Types:**")
                for dtype, cols in summary['data_types'].items():
                    if cols:
                        st.write(f"- {dtype.capitalize()}: {len(cols)} columns")
                
                if summary['duplicates']['count'] > 0:
                    st.warning(f"⚠️ {summary['duplicates']['count']} duplicate rows found")
    
    with col2:
        st.header("🤖 AI Data Cleaning")
        
        # Cleaning controls
        if st.session_state.agent is not None and st.session_state.original_df is not None:
            
            if st.button("🚀 Start Data Cleaning", type="primary", width='stretch'):
                with st.spinner("🔍 Analyzing dataset and generating cleaning code..."):
                    
                    # Load dataset into agent
                    if st.session_state.agent.load_dataset(dataframe=st.session_state.original_df):
                        
                        # Perform cleaning
                        result = st.session_state.agent.clean_dataset()
                        st.session_state.cleaning_result = result
                        
                        if result.get('success'):
                            st.session_state.cleaned_df = result['cleaned_dataframe']
                            st.session_state.processing_complete = True
                            st.success("✅ Data cleaning completed successfully!")
                        else:
                            st.error(f"❌ Cleaning failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Failed to load dataset into agent")
        
        else:
            if st.session_state.agent is None:
                st.warning("⚠️ Please enter a valid API key to start cleaning")
            elif st.session_state.original_df is None:
                st.warning("⚠️ Please upload a dataset to start cleaning")
    
    # Results section
    if st.session_state.processing_complete and st.session_state.cleaning_result:
        st.divider()
        st.header("📊 Cleaning Results")
        
        result = st.session_state.cleaning_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 Analysis", "💻 Generated Code", "📥 Downloads"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("📊 Before Cleaning")
                original_summary = get_dataset_summary(st.session_state.original_df)
                st.metric("Rows", original_summary['basic_info']['shape'][0])
                st.metric("Columns", original_summary['basic_info']['shape'][1])
                st.metric("Missing Values", original_summary['missing_values']['total'])
                st.metric("Duplicates", original_summary['duplicates']['count'])
            
            with col_b:
                st.subheader("📊 After Cleaning")
                cleaned_summary = get_dataset_summary(st.session_state.cleaned_df)
                st.metric("Rows", cleaned_summary['basic_info']['shape'][0])
                st.metric("Columns", cleaned_summary['basic_info']['shape'][1])
                st.metric("Missing Values", cleaned_summary['missing_values']['total'])
                st.metric("Duplicates", cleaned_summary['duplicates']['count'])
            
            # Quality report
            if result.get('validation_report'):
                st.subheader("📈 Quality Report")
                validation = result['validation_report']
                
                col_i, col_ii, col_iii = st.columns(3)
                with col_i:
                    rows_removed = validation.get('rows_removed', 0)
                    st.metric("Rows Removed", rows_removed, delta=f"{rows_removed}")
                
                with col_ii:
                    missing_removed = validation.get('missing_values_before', 0) - validation.get('missing_values_after', 0)
                    st.metric("Missing Values Fixed", missing_removed, delta=f"{missing_removed}")
                
                with col_iii:
                    duplicates_removed = validation.get('duplicates_before', 0) - validation.get('duplicates_after', 0)
                    st.metric("Duplicates Removed", duplicates_removed, delta=f"{duplicates_removed}")
        
        with tab2:
            st.subheader("🔍 AI Analysis")
            if result.get('analysis', {}).get('analysis'):
                st.write(result['analysis']['analysis'])
            else:
                st.write("No detailed analysis available")
        
        with tab3:
            st.subheader("💻 Generated Cleaning Code")
            if result.get('cleaning_code'):
                st.code(result['cleaning_code'], language='python')
            else:
                st.write("No code generated")
        
        with tab4:
            st.subheader("📥 Download Results")
            
            col_x, col_y = st.columns(2)
            
            with col_x:
                st.write("**Cleaned Dataset**")
                
                # CSV download
                csv_data = create_download_link(st.session_state.cleaned_df, 'cleaned_dataset', 'csv')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
                # Excel download
                excel_data = create_download_link(st.session_state.cleaned_df, 'cleaned_dataset', 'excel')
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch'
                )
            
            with col_y:
                st.write("**Cleaning Code**")
                
                if result.get('cleaning_code'):
                    formatted_code = format_code_for_download(
                        result['cleaning_code'], 
                        f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    st.download_button(
                        label="💻 Download Python Code",
                        data=formatted_code.encode('utf-8'),
                        file_name=f"data_cleaning_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                        mime="text/plain",
                        width='stretch'
                    )
        
        # Cleaned dataset preview
        st.subheader("👀 Cleaned Dataset Preview")
        st.dataframe(st.session_state.cleaned_df.head(), width='stretch')

if __name__ == "__main__":
    main()