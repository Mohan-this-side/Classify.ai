# 🤖 AI Data Cleaning Agent

An intelligent data cleaning system powered by Google Gemini Flash 2.0 that automatically analyzes datasets, generates cleaning code, and provides downloadable cleaned data with full validation.

## ✨ Features

- **🔍 Smart Dataset Analysis**: AI-powered exploration and issue identification
- **💻 Automatic Code Generation**: Creates Python code for data cleaning tasks
- **🔄 Self-Correction**: Iteratively fixes errors using LLM feedback
- **✅ Quality Validation**: Comprehensive testing and validation of cleaning results
- **🖥️ User-Friendly Interface**: Streamlit web app with upload/download functionality
- **📊 Sample Datasets**: Built-in test datasets for immediate experimentation
- **🛡️ Safe Execution**: Secure code execution environment with safety checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│  File Upload | API Key Input | Progress | Downloads         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Data Cleaning Agent                          │
│  • Dataset Analysis (Gemini Flash 2.0)                     │
│  • Code Generation & Self-Correction                       │
│  • Quality Validation & Reporting                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Safe Code Executor                             │
│  • Secure Execution Environment                            │
│  • Error Handling & Testing                                │
│  • Quality Assurance Checks                                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "agents v1"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Usage

1. **Enter API Key**: Add your Google Gemini API key in the sidebar
2. **Upload Dataset**: Choose a CSV or Excel file (or try sample datasets)
3. **Start Cleaning**: Click "Start Data Cleaning" 
4. **Review Results**: Analyze the AI's findings and generated code
5. **Download**: Get your cleaned dataset and the Python cleaning code

## 📁 Project Structure

```
agents v1/
├── app.py                    # Streamlit frontend application
├── data_cleaning_agent.py    # Core AI agent for dataset analysis
├── code_executor.py          # Safe code execution and testing
├── utils.py                  # Utility functions and helpers
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## 🔧 Core Components

### DataCleaningAgent (`data_cleaning_agent.py`)
- **Dataset Analysis**: Uses Gemini to identify data quality issues
- **Code Generation**: Creates targeted Python cleaning code
- **Self-Correction**: Iteratively improves code when errors occur
- **Validation**: Ensures cleaning maintains data integrity

### SafeCodeExecutor (`code_executor.py`)
- **Security**: Validates code safety before execution
- **Testing**: Comprehensive quality tests for cleaned data
- **Error Handling**: Robust error capture and reporting
- **Validation**: Multi-dimensional data quality checks

### Streamlit Frontend (`app.py`)
- **File Upload**: Support for CSV and Excel files
- **Progress Tracking**: Real-time cleaning status
- **Results Display**: Comprehensive analysis and code views
- **Downloads**: Easy access to cleaned data and code

## 🧠 How It Works

### 1. Dataset Analysis
The AI agent analyzes your dataset to identify:
- Missing values and patterns
- Data type inconsistencies  
- Duplicate records
- Outliers and anomalies
- Structural issues

### 2. Code Generation
Based on the analysis, the agent generates Python code to:
- Handle missing values appropriately
- Fix data type issues
- Remove or consolidate duplicates
- Address outliers
- Standardize formats

### 3. Execution & Testing
The generated code is:
- Validated for safety
- Executed in a secure environment
- Tested for quality and integrity
- Self-corrected if errors occur

### 4. Results & Downloads
You receive:
- Cleaned dataset (CSV/Excel)
- Complete Python cleaning code
- Quality validation report
- Before/after comparisons

## 📊 Supported Data Issues

The agent can handle various data quality issues:

- **Missing Values**: Intelligent imputation or removal strategies
- **Data Types**: Automatic type detection and conversion
- **Duplicates**: Identification and removal of redundant records
- **Outliers**: Detection and appropriate handling
- **Format Issues**: Date parsing, string standardization
- **Structural Problems**: Column renaming, data reorganization

## 🛡️ Safety Features

- **Code Validation**: Prevents execution of potentially harmful code
- **Sandboxed Execution**: Isolated environment for code testing
- **Input Validation**: Checks for malicious inputs
- **Error Boundaries**: Graceful handling of unexpected errors
- **Resource Limits**: Prevents excessive resource consumption

## 🎯 Example Use Cases

- **Customer Data**: Clean CRM exports with missing emails and duplicates
- **Sales Data**: Fix transaction records with type errors and outliers
- **Survey Data**: Handle incomplete responses and format inconsistencies
- **IoT Data**: Process sensor readings with missing timestamps
- **Financial Data**: Clean transaction logs with data quality issues

## 🔍 Testing

The system includes comprehensive testing capabilities:

### Quality Tests
- Data integrity validation
- Column preservation checks
- Data type consistency
- Missing value handling verification
- Duplicate removal validation

### Sample Datasets
Built-in datasets for testing:
- **Customer Dataset**: 1000+ records with various data quality issues
- **Sales Dataset**: Transaction data with missing values and outliers

## 📈 Performance

- **Processing Speed**: Optimized for datasets up to 100k rows
- **Memory Efficiency**: Streaming processing for large files
- **Scalability**: Handles various dataset sizes and complexities
- **Reliability**: Robust error handling and recovery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**API Key Issues**:
- Ensure your Gemini API key is valid and has sufficient quota
- Check that the key starts with "AIza"

**File Upload Problems**:
- Supported formats: CSV, Excel (.xlsx, .xls)
- Maximum file size: 200MB
- Ensure proper encoding (UTF-8 recommended)

**Code Execution Errors**:
- The system will attempt self-correction automatically
- Complex datasets may require multiple iterations
- Check the error logs in the interface

**Memory Issues**:
- For very large datasets (>1M rows), consider sampling
- Ensure sufficient system memory
- Close other applications if needed

### Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error messages in the app
3. Try with sample datasets first
4. Open an issue on the repository

## 🔮 Future Enhancements

- **Advanced ML Models**: Integration with specialized cleaning models
- **Real-time Processing**: Streaming data cleaning capabilities
- **Custom Rules**: User-defined cleaning rules and policies
- **Visualization**: Advanced data quality visualization
- **API Access**: REST API for programmatic access
- **Batch Processing**: Handle multiple datasets simultaneously

---

**Built with ❤️ using Google Gemini Flash 2.0, Streamlit, and Python**