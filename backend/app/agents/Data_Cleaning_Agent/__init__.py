"""
Data Cleaning Agent Package

This package contains all components related to the Data Cleaning Agent functionality:
- Enhanced Data Cleaning Agent: Main orchestrator for data cleaning operations
- Missing Value Analyzer: Comprehensive missing value analysis and visualization
- Data Type Validator: Data type validation, detection, and conversion recommendations

The Data Cleaning Agent provides:
- Advanced missing value imputation strategies
- Intelligent data type detection and conversion
- Sophisticated duplicate detection and removal
- Statistical outlier detection and treatment
- Format standardization and normalization
- Categorical data encoding and standardization
- Data quality assessment and validation
- Structural error detection and correction
"""

from .enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from .missing_value_analyzer import MissingValueAnalyzer
from .data_type_validator import DataTypeValidator
from .outlier_detector import OutlierDetector
from .missing_value_imputer import MissingValueImputer
from .educational_explainer import EducationalExplainer

__all__ = [
    'EnhancedDataCleaningAgent',
    'MissingValueAnalyzer', 
    'DataTypeValidator',
    'OutlierDetector',
    'MissingValueImputer',
    'EducationalExplainer'
]

__version__ = "2.0.0"
__author__ = "Classify AI Team"
__description__ = "Comprehensive data cleaning agent with advanced analytics and validation"
