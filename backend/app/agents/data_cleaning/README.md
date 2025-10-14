# Data Cleaning Agent Package

This package contains all components related to the Data Cleaning Agent functionality for the Classify AI system.

## Package Structure

```
Data_Cleaning_Agent/
├── __init__.py                           # Package initialization and exports
├── enhanced_data_cleaning_agent.py      # Main orchestrator for data cleaning operations
├── missing_value_analyzer.py            # Comprehensive missing value analysis
├── data_type_validator.py               # Data type validation and conversion
└── README.md                            # This documentation file
```

## Components

### 1. Enhanced Data Cleaning Agent (`enhanced_data_cleaning_agent.py`)

The main orchestrator that coordinates all data cleaning operations:

- **Advanced missing value imputation strategies**
- **Intelligent data type detection and conversion**
- **Sophisticated duplicate detection and removal**
- **Statistical outlier detection and treatment**
- **Format standardization and normalization**
- **Categorical data encoding and standardization**
- **Data quality assessment and validation**
- **Structural error detection and correction**

### 2. Missing Value Analyzer (`missing_value_analyzer.py`)

Comprehensive missing value analysis and visualization:

- **Statistical Analysis**: Missing value counts, percentages, and patterns
- **Pattern Detection**: MCAR, MAR, MNAR pattern identification
- **Correlation Analysis**: Missing value pattern correlations
- **Visualizations**: Heatmaps, bar charts, correlation matrices
- **Educational Explanations**: User-friendly pattern explanations
- **Smart Recommendations**: Context-aware handling suggestions

### 3. Data Type Validator (`data_type_validator.py`)

Data type validation, detection, and conversion recommendations:

- **Type Detection**: Numeric, categorical, datetime, boolean identification
- **Quality Assessment**: Type appropriateness and consistency scoring
- **Conversion Recommendations**: Immediate, investigation, and priority suggestions
- **Issue Identification**: Mixed types, inconsistencies, and problems
- **Educational Content**: Clear explanations and actionable recommendations

## Usage

### Basic Import

```python
from app.agents.Data_Cleaning_Agent import EnhancedDataCleaningAgent, MissingValueAnalyzer, DataTypeValidator
```

### Individual Component Usage

```python
# Missing value analysis
from app.agents.Data_Cleaning_Agent.missing_value_analyzer import MissingValueAnalyzer
analyzer = MissingValueAnalyzer()
results = analyzer.analyze_missing_values(df, target_column='target')

# Data type validation
from app.agents.Data_Cleaning_Agent.data_type_validator import DataTypeValidator
validator = DataTypeValidator()
validation = validator.validate_data_types(df, target_column='target')

# Full data cleaning
from app.agents.Data_Cleaning_Agent.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
agent = EnhancedDataCleaningAgent()
cleaned_state = await agent.execute(workflow_state)
```

## Features

### Missing Value Analysis
- ✅ Complete missing value statistics
- ✅ Pattern detection (MCAR, MAR, MNAR)
- ✅ Correlation analysis between missing patterns
- ✅ Target-specific analysis
- ✅ Visualization generation
- ✅ Educational explanations
- ✅ Smart recommendations

### Data Type Validation
- ✅ Automatic type detection with confidence scoring
- ✅ Quality assessment and consistency analysis
- ✅ Conversion recommendations (immediate, investigation, priority)
- ✅ Issue identification and reporting
- ✅ Educational content and explanations
- ✅ Context-aware suggestions

### Enhanced Data Cleaning
- ✅ Comprehensive workflow orchestration
- ✅ Integration of all analysis components
- ✅ State management and result storage
- ✅ Error handling and graceful fallbacks
- ✅ Detailed logging and progress tracking
- ✅ Educational reporting

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Visualization
- `seaborn`: Statistical visualization
- `scikit-learn`: Machine learning utilities
- `datetime`: Date/time handling
- `logging`: Logging functionality

## Version

- **Current Version**: 2.0.0
- **Author**: Classify AI Team
- **Description**: Comprehensive data cleaning agent with advanced analytics and validation

## Integration

This package is fully integrated with the Classify AI workflow system and can be used as part of the multi-agent data science pipeline. All components are designed to work together seamlessly and provide comprehensive data cleaning capabilities for machine learning workflows.
