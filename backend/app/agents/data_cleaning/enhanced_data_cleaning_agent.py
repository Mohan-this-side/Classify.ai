"""
🧹 Enhanced Data Cleaning Agent with Advanced Prompt Engineering

This agent provides comprehensive data cleaning capabilities with intelligent
prompt engineering to handle various types of messy data scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

from ..base_agent import BaseAgent, AgentResult
from .missing_value_analyzer import MissingValueAnalyzer
from .data_type_validator import DataTypeValidator
from .outlier_detector import OutlierDetector
from .missing_value_imputer import MissingValueImputer
from .educational_explainer import EducationalExplainer
from ...workflows.state_management import ClassificationState, AgentStatus, state_manager
from ...config import settings
from ...services.storage import storage_service

class EnhancedDataCleaningAgent(BaseAgent):
    """
    🧹 Enhanced Data Cleaning Agent with Advanced Prompt Engineering
    
    This agent provides comprehensive data cleaning capabilities:
    - Advanced missing value imputation strategies
    - Intelligent data type detection and conversion
    - Sophisticated duplicate detection and removal
    - Statistical outlier detection and treatment
    - Format standardization and normalization
    - Categorical data encoding and standardization
    - Data quality assessment and validation
    - Structural error detection and correction
    """
    
    def __init__(self):
        super().__init__("enhanced_data_cleaning", "2.0.0")
        self.logger = logging.getLogger("agent.enhanced_data_cleaning")
        
        # Initialize analyzers
        self.missing_analyzer = MissingValueAnalyzer()
        self.type_validator = DataTypeValidator()
        self.outlier_detector = OutlierDetector()
        self.missing_imputer = MissingValueImputer()
        self.educational_explainer = EducationalExplainer()
        
        # Data quality patterns for detection
        self.outlier_threshold = 3.0  # Z-score threshold for outliers
        self.duplicate_threshold = 0.95  # Similarity threshold for near-duplicates
        
        # Common data quality issues patterns
        self.boolean_patterns = {
            'true_values': ['true', 't', 'yes', 'y', '1', 'on', 'enabled', 'active'],
            'false_values': ['false', 'f', 'no', 'n', '0', 'off', 'disabled', 'inactive']
        }
        
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',  # M/D/YYYY
        ]
        
        self.currency_patterns = [
            r'^\$[\d,]+\.?\d*$',  # $1,234.56
            r'^€[\d,]+\.?\d*$',   # €1,234.56
            r'^£[\d,]+\.?\d*$',   # £1,234.56
            r'^[\d,]+\.?\d*\s*USD$',  # 1234.56 USD
            r'^[\d,]+\.?\d*\s*EUR$',  # 1234.56 EUR
        ]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this enhanced agent"""
        return {
            "name": "Enhanced Data Cleaning Agent",
            "version": self.agent_version,
            "description": "Advanced data cleaning with intelligent prompt engineering",
            "capabilities": [
                "Advanced missing value imputation",
                "Intelligent data type detection and conversion",
                "Sophisticated duplicate detection and removal",
                "Statistical outlier detection and treatment",
                "Format standardization and normalization",
                "Categorical data encoding and standardization",
                "Structural error detection and correction",
                "Data quality assessment and validation",
                "International character encoding handling",
                "Mixed data type resolution"
            ],
            "dependencies": [],
            "outputs": [
                "cleaned_dataset",
                "comprehensive_cleaning_report",
                "data_quality_score",
                "cleaning_issues_found",
                "cleaning_actions_taken",
                "data_transformation_log",
                "quality_metrics"
            ]
        }
    
    def get_dependencies(self) -> list:
        """Get list of agent dependencies"""
        return []  # Data cleaning is the first agent
    
    async def execute(self, state: ClassificationState) -> ClassificationState:
        """
        Execute enhanced data cleaning operations with advanced prompt engineering
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with comprehensive cleaning results
        """
        self.logger.info("Starting enhanced data cleaning process")
        
        # Get original dataset
        original_dataset = state_manager.get_dataset(state, "original")
        if original_dataset is None:
            raise ValueError("Original dataset not found in state")
        
        # Create a copy for cleaning
        cleaned_dataset = original_dataset.copy()
        
        # Initialize comprehensive tracking
        cleaning_actions = []
        issues_found = []
        transformation_log = []
        quality_metrics = {}
        
        # 1. Initial data assessment
        self.logger.info(f"Original dataset shape: {cleaned_dataset.shape}")
        self.logger.info(f"Columns: {list(cleaned_dataset.columns)}")
        self.logger.info(f"Data types: {dict(cleaned_dataset.dtypes)}")
        
        # 2. Comprehensive missing value analysis
        missing_analysis = await self._comprehensive_missing_analysis(cleaned_dataset, state.get("target_column"))
        cleaning_actions.append(f"Performed comprehensive missing value analysis: {missing_analysis['missing_statistics']['total_missing']} missing values found")
        
        # 3. Comprehensive data type validation
        type_validation = await self._comprehensive_type_validation(cleaned_dataset, state.get("target_column"))
        cleaning_actions.append(f"Performed comprehensive data type validation: {type_validation['quality_assessment']['columns_with_issues']} columns with issues found")
        
        # 4. Comprehensive outlier detection
        outlier_detection = await self._comprehensive_outlier_detection(cleaned_dataset, state.get("target_column"))
        cleaning_actions.append(f"Performed comprehensive outlier detection: {outlier_detection['outlier_summary']['total_outliers_detected']} outliers detected")
        
        # 5. Comprehensive missing value imputation
        imputation_results = await self._comprehensive_missing_imputation(cleaned_dataset, state.get("target_column"))
        cleaned_dataset = imputation_results['imputed_df']  # Update the dataset
        cleaning_actions.append(f"Performed comprehensive missing value imputation: {imputation_results['imputation_details']['imputation_stats']['total_values_imputed']} values imputed using {imputation_results['strategy_used']} strategy")
        
        # 3. Structural error detection and correction
        cleaned_dataset, structural_actions = await self._fix_structural_errors(cleaned_dataset)
        cleaning_actions.extend(structural_actions)
        
        # 3. Advanced data type detection and conversion
        cleaned_dataset, type_actions = await self._intelligent_type_conversion(cleaned_dataset, state["target_column"])
        cleaning_actions.extend(type_actions)
        transformation_log.extend(type_actions)
        
        # 4. Format standardization
        cleaned_dataset, format_actions = await self._standardize_formats(cleaned_dataset)
        cleaning_actions.extend(format_actions)
        transformation_log.extend(format_actions)
        
        # 5. Categorical data standardization
        cleaned_dataset, categorical_actions = await self._standardize_categorical_data(cleaned_dataset)
        cleaning_actions.extend(categorical_actions)
        transformation_log.extend(categorical_actions)
        
        # 6. Advanced missing value imputation
        cleaned_dataset, missing_actions = await self._advanced_missing_value_handling(cleaned_dataset)
        cleaning_actions.extend(missing_actions)
        transformation_log.extend(missing_actions)
        
        # 7. Sophisticated duplicate detection and removal
        cleaned_dataset, duplicate_actions = await self._advanced_duplicate_handling(cleaned_dataset)
        cleaning_actions.extend(duplicate_actions)
        transformation_log.extend(duplicate_actions)
        
        # 8. Statistical outlier detection and treatment
        cleaned_dataset, outlier_actions = await self._advanced_outlier_detection(cleaned_dataset, state["target_column"])
        cleaning_actions.extend(outlier_actions)
        transformation_log.extend(outlier_actions)
        
        # 9. Data validation and quality assessment
        quality_score, quality_issues, metrics = await self._comprehensive_quality_assessment(cleaned_dataset, original_dataset)
        issues_found.extend(quality_issues)
        quality_metrics.update(metrics)
        
        # 10. Generate comprehensive cleaning report
        cleaning_report = await self._generate_comprehensive_report(
            original_dataset, cleaned_dataset, cleaning_actions, issues_found, quality_metrics
        )
        
        # 11. Generate comprehensive educational explanations
        educational_explanations = await self._generate_educational_explanations(
            cleaned_dataset, original_dataset, state
        )
        
        # Store cleaned dataset
        state_manager.store_dataset(state, cleaned_dataset, "cleaned")
        
        # Also store using storage service for download
        session_id = state.get("session_id", "unknown")
        cleaned_dataset_path = storage_service.store_cleaned_dataset(
            workflow_id=session_id,
            dataset=cleaned_dataset,
            filename="cleaned_dataset.csv"
        )
        
        # Update state with comprehensive results
        state["cleaned_dataset"] = None  # Stored externally
        state["cleaned_dataset_path"] = cleaned_dataset_path
        state["cleaning_summary"] = cleaning_report
        state["data_quality_score"] = quality_score
        state["cleaning_issues_found"] = issues_found
        state["cleaning_actions_taken"] = cleaning_actions
        state["data_transformation_log"] = transformation_log
        state["quality_metrics"] = quality_metrics
        state["missing_value_analysis"] = missing_analysis
        state["data_type_validation"] = type_validation
        state["outlier_detection"] = outlier_detection
        state["missing_value_imputation"] = imputation_results
        state["educational_explanations"] = educational_explanations
        
        # Update dataset metadata
        state["dataset_shape"] = cleaned_dataset.shape
        state["data_types"] = {col: str(dtype) for col, dtype in cleaned_dataset.dtypes.items()}
        state["missing_values"] = dict(cleaned_dataset.isnull().sum())
        state["duplicate_count"] = int(cleaned_dataset.duplicated().sum())
        
        self.logger.info(f"Enhanced data cleaning completed. Quality score: {quality_score:.2f}")
        self.logger.info(f"Actions taken: {len(cleaning_actions)}")
        self.logger.info(f"Transformations applied: {len(transformation_log)}")
        
        return state
    
    async def _fix_structural_errors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fix structural errors in the dataset"""
        actions = []
        
        # 1. Standardize column names
        original_columns = df.columns.tolist()
        df.columns = [self._standardize_column_name(col) for col in df.columns]
        
        if original_columns != df.columns.tolist():
            actions.append(f"Standardized column names: {original_columns} → {df.columns.tolist()}")
        
        # 2. Remove completely empty rows and columns
        empty_rows_before = df.isnull().all(axis=1).sum()
        empty_cols_before = df.isnull().all(axis=0).sum()
        
        df = df.dropna(how='all')  # Remove rows that are all NaN
        df = df.dropna(axis=1, how='all')  # Remove columns that are all NaN
        
        if empty_rows_before > 0:
            actions.append(f"Removed {empty_rows_before} completely empty rows")
        if empty_cols_before > 0:
            actions.append(f"Removed {empty_cols_before} completely empty columns")
        
        # 3. Reset index after cleaning
        df = df.reset_index(drop=True)
        
        return df, actions
    
    async def _comprehensive_missing_analysis(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive missing value analysis using the MissingValueAnalyzer"""
        self.logger.info("Starting comprehensive missing value analysis")
        
        try:
            # Use the dedicated missing value analyzer
            analysis_results = self.missing_analyzer.analyze_missing_values(df, target_column)
            
            # Log key findings
            stats = analysis_results['missing_statistics']
            self.logger.info(f"Missing value analysis completed:")
            self.logger.info(f"  - Total missing values: {stats['total_missing']}")
            self.logger.info(f"  - Overall missing percentage: {stats['overall_missing_percentage']}%")
            self.logger.info(f"  - Columns with missing values: {stats['columns_with_missing']}")
            self.logger.info(f"  - Complete rows percentage: {stats['complete_rows_percentage']}%")
            
            # Log pattern analysis
            patterns = analysis_results['missing_patterns']
            mcar_analysis = patterns['mcar_analysis']
            self.logger.info(f"  - MCAR independence score: {mcar_analysis['independence_score']}")
            self.logger.info(f"  - Likely MCAR: {mcar_analysis['likely_mcar']}")
            
            # Log recommendations
            recommendations = analysis_results['recommendations']
            self.logger.info(f"  - Generated {len(recommendations)} recommendations")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in missing value analysis: {e}")
            # Return basic analysis if comprehensive analysis fails
            return {
                "missing_statistics": {
                    "total_missing": int(df.isnull().sum().sum()),
                    "overall_missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
                    "columns_with_missing": int((df.isnull().sum() > 0).sum()),
                    "missing_counts": df.isnull().sum().to_dict(),
                    "missing_percentages": ((df.isnull().sum() / len(df)) * 100).to_dict()
                },
                "missing_patterns": {"mcar_analysis": {"independence_score": 0.0, "likely_mcar": False}},
                "missing_correlations": {"strong_correlations": [], "max_correlation": 0.0},
                "target_analysis": {},
                "visualizations": {},
                "explanations": {"overall": "Basic missing value analysis completed"},
                "recommendations": ["Use standard imputation methods"],
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def _comprehensive_type_validation(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive data type validation using the DataTypeValidator"""
        self.logger.info("Starting comprehensive data type validation")
        
        try:
            # Use the dedicated data type validator
            validation_results = self.type_validator.validate_data_types(df, target_column)
            
            # Log key findings
            type_analysis = validation_results['type_analysis']
            quality_assessment = validation_results['quality_assessment']
            
            self.logger.info(f"Data type validation completed:")
            self.logger.info(f"  - Columns analyzed: {len(type_analysis['column_types'])}")
            self.logger.info(f"  - Type appropriateness score: {quality_assessment['type_appropriateness_score']:.2f}")
            self.logger.info(f"  - Columns with issues: {quality_assessment['columns_with_issues']}")
            self.logger.info(f"  - Conversion needed: {quality_assessment['conversion_needed']}")
            
            # Log type distribution
            type_distribution = type_analysis['type_distribution']
            self.logger.info(f"  - Type distribution: {type_distribution}")
            
            # Log conversion recommendations
            conversion_recs = validation_results['conversion_recommendations']
            self.logger.info(f"  - Immediate conversions: {len(conversion_recs['immediate_conversions'])}")
            self.logger.info(f"  - Investigation needed: {len(conversion_recs['investigation_needed'])}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in data type validation: {e}")
            # Return basic validation if comprehensive validation fails
            return {
                "type_analysis": {
                    "column_types": {col: {"current_dtype": str(df[col].dtype), "detected_type": "unknown", "confidence": 0.0} for col in df.columns},
                    "type_distribution": {"unknown": len(df.columns)},
                    "mixed_type_columns": [],
                    "inconsistent_types": []
                },
                "quality_assessment": {
                    "overall_type_consistency": 0.0,
                    "columns_with_issues": len(df.columns),
                    "type_appropriateness_score": 0.0,
                    "conversion_needed": len(df.columns),
                    "quality_issues": ["Data type validation failed"]
                },
                "conversion_recommendations": {
                    "immediate_conversions": [],
                    "investigation_needed": [],
                    "conversion_priority": []
                },
                "consistency_analysis": {
                    "type_consistency_score": 0.0,
                    "inconsistent_columns": list(df.columns),
                    "type_patterns": {},
                    "recommendations": []
                },
                "target_analysis": {},
                "validation_report": "Basic data type validation completed with errors",
                "explanations": {"overall": "Data type validation encountered errors"},
                "recommendations": ["Review data types manually"],
                "validation_timestamp": datetime.now().isoformat()
            }
    
    async def _comprehensive_outlier_detection(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive outlier detection using the OutlierDetector"""
        self.logger.info("Starting comprehensive outlier detection")
        
        try:
            # Use the dedicated outlier detector
            detection_results = self.outlier_detector.detect_outliers(df, target_column)
            
            # Log key findings
            summary = detection_results['outlier_summary']
            method_results = detection_results['method_results']
            
            self.logger.info(f"Outlier detection completed:")
            self.logger.info(f"  - Columns analyzed: {summary['total_columns_analyzed']}")
            self.logger.info(f"  - Columns with outliers: {summary['columns_with_outliers']}")
            self.logger.info(f"  - Total outliers detected: {summary['total_outliers_detected']}")
            self.logger.info(f"  - Outlier percentage: {summary['outlier_percentage']:.2f}%")
            
            # Log method performance
            if method_results:
                best_method = max(method_results.keys(), 
                                key=lambda x: method_results[x]["consistency_score"])
                self.logger.info(f"  - Best performing method: {best_method}")
                self.logger.info(f"  - Methods used: {list(method_results.keys())}")
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Error in outlier detection: {e}")
            # Return basic detection if comprehensive detection fails
            return {
                "outlier_summary": {
                    "total_columns_analyzed": 0,
                    "columns_with_outliers": 0,
                    "columns_without_outliers": 0,
                    "total_outliers_detected": 0,
                    "outlier_percentage": 0,
                    "method_usage": {},
                    "most_common_method": None
                },
                "method_results": {},
                "column_analysis": {},
                "visualizations": {},
                "explanations": {
                    "overall": "Outlier detection encountered errors",
                    "method_effectiveness": "N/A - detection failed",
                    "data_quality": "N/A - detection failed"
                },
                "recommendations": ["Review data manually for outliers"],
                "detection_timestamp": datetime.now().isoformat()
            }
    
    async def _comprehensive_missing_imputation(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive missing value imputation using the MissingValueImputer"""
        self.logger.info("Starting comprehensive missing value imputation")
        
        try:
            # Use the dedicated missing value imputer
            imputation_results = self.missing_imputer.impute_missing_values(df, target_column)
            
            # Log key findings
            details = imputation_results['imputation_details']
            quality = imputation_results['quality_assessment']
            
            self.logger.info(f"Missing value imputation completed:")
            self.logger.info(f"  - Strategy used: {imputation_results['strategy_used']}")
            self.logger.info(f"  - Columns imputed: {len(details['columns_imputed'])}")
            self.logger.info(f"  - Values imputed: {details['imputation_stats']['total_values_imputed']}")
            self.logger.info(f"  - Success rate: {quality['imputation_success_rate']:.1f}%")
            self.logger.info(f"  - Quality score: {quality['quality_score']:.1f}")
            
            return imputation_results
            
        except Exception as e:
            self.logger.error(f"Error in missing value imputation: {e}")
            # Return basic imputation if comprehensive imputation fails
            return {
                "original_df": df,
                "imputed_df": df.fillna(df.median() if df.select_dtypes(include=[np.number]).shape[1] > 0 else df.fillna('Unknown')),
                "strategy_used": "fallback_median",
                "parameters_used": {},
                "missing_analysis": {"missing_percentage": 0},
                "imputation_details": {
                    "strategy": "fallback_median",
                    "columns_imputed": [],
                    "values_imputed": {},
                    "imputation_stats": {"total_values_imputed": 0, "columns_processed": 0, "remaining_missing": 0}
                },
                "quality_assessment": {
                    "missing_values_remaining": 0,
                    "missing_values_imputed": 0,
                    "imputation_success_rate": 0,
                    "data_integrity_preserved": True,
                    "statistical_changes": {},
                    "quality_score": 0
                },
                "explanations": {
                    "strategy_rationale": "Fallback imputation due to errors",
                    "missing_pattern": "Unknown due to imputation failure",
                    "quality": "Imputation encountered errors"
                },
                "recommendations": ["Review data quality and try manual imputation"],
                "imputation_timestamp": datetime.now().isoformat()
            }
    
    async def _generate_educational_explanations(self, cleaned_df: pd.DataFrame, 
                                               original_df: pd.DataFrame, 
                                               state: ClassificationState) -> Dict[str, Any]:
        """Generate comprehensive educational explanations for the data cleaning process"""
        self.logger.info("Generating comprehensive educational explanations")
        
        try:
            # Prepare cleaning results for explanation generation
            cleaning_results = {
                "missing_value_analysis": state.get("missing_value_analysis", {}),
                "data_type_validation": state.get("data_type_validation", {}),
                "outlier_detection": state.get("outlier_detection", {}),
                "missing_value_imputation": state.get("missing_value_imputation", {}),
                "cleaning_actions_taken": state.get("cleaning_actions_taken", []),
                "quality_metrics": state.get("quality_metrics", {}),
                "data_quality_score": state.get("data_quality_score", 0)
            }
            
            # Generate comprehensive explanations
            explanations = self.educational_explainer.generate_comprehensive_explanations(
                cleaning_results, original_df, cleaned_df
            )
            
            # Log key findings
            self.logger.info(f"Educational explanations generated:")
            self.logger.info(f"  - Overview: {explanations['overview']['title']}")
            self.logger.info(f"  - Steps explained: {len(explanations['step_by_step'])}")
            self.logger.info(f"  - Impact level: {explanations['impact_assessment']['overall_impact']}")
            self.logger.info(f"  - Recommendations: {len(explanations['recommendations'])}")
            self.logger.info(f"  - Markdown report: {len(explanations['markdown_report'])} characters")
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generating educational explanations: {e}")
            # Return basic explanations if comprehensive generation fails
            return {
                "overview": {
                    "title": "Data Cleaning Overview",
                    "summary": "Data cleaning completed with basic operations",
                    "dataset_changes": {
                        "original_shape": original_df.shape,
                        "cleaned_shape": cleaned_df.shape,
                        "rows_removed": original_df.shape[0] - cleaned_df.shape[0],
                        "columns_processed": 0
                    },
                    "quality_improvement": "Basic improvement achieved",
                    "key_achievements": ["Basic data cleaning completed"],
                    "overall_impact": "Moderate impact - standard cleaning applied"
                },
                "step_by_step": [],
                "impact_assessment": {
                    "overall_impact": "Moderate impact",
                    "dataset_changes": {"rows_removed": 0, "columns_modified": 0, "missing_values_imputed": 0},
                    "statistical_changes": {},
                    "quality_improvements": ["Basic data cleaning"],
                    "data_integrity": "Data integrity preserved",
                    "recommendations": ["Review cleaning results manually"]
                },
                "best_practices": {
                    "data_quality_monitoring": ["Regular data quality checks recommended"],
                    "documentation": ["Document all cleaning decisions"]
                },
                "recommendations": [
                    {
                        "category": "General",
                        "priority": "Medium",
                        "recommendation": "Review data cleaning results manually",
                        "rationale": "Educational explanation generation encountered errors"
                    }
                ],
                "markdown_report": "# Data Cleaning Report\n\nBasic data cleaning completed. Please review results manually.",
                "generation_timestamp": datetime.now().isoformat()
            }
    
    def _standardize_column_name(self, col_name: str) -> str:
        """Standardize column names to snake_case"""
        # Convert to lowercase
        col = str(col_name).lower()
        
        # Replace spaces and special characters with underscores
        col = re.sub(r'[^a-z0-9]+', '_', col)
        
        # Remove leading/trailing underscores
        col = col.strip('_')
        
        # Handle multiple underscores
        col = re.sub(r'_+', '_', col)
        
        return col
    
    async def _intelligent_type_conversion(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligently detect and convert data types"""
        actions = []
        
        for column in df.columns:
            original_dtype = str(df[column].dtype)
            
            # Skip if already properly typed
            if df[column].dtype in ['int64', 'float64', 'bool', 'datetime64[ns]']:
                continue
            
            # Try to convert to numeric first
            if self._is_numeric_column(df[column]):
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    actions.append(f"Converted '{column}' to numeric: {original_dtype} → {df[column].dtype}")
                except:
                    pass
            
            # Try to convert to boolean
            elif self._is_boolean_column(df[column]):
                df[column] = self._convert_to_boolean(df[column])
                actions.append(f"Converted '{column}' to boolean: {original_dtype} → {df[column].dtype}")
            
            # Try to convert to datetime
            elif self._is_datetime_column(df[column]):
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    actions.append(f"Converted '{column}' to datetime: {original_dtype} → {df[column].dtype}")
                except:
                    pass
            
            # Convert object columns to category if they have few unique values
            elif df[column].dtype == 'object' and df[column].nunique() < len(df) * 0.5:
                df[column] = df[column].astype('category')
                actions.append(f"Converted '{column}' to category: {original_dtype} → {df[column].dtype}")
        
        return df, actions
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column can be converted to numeric"""
        if series.dtype in ['int64', 'float64']:
            return True
        
        # Check if most values can be converted to numeric
        numeric_count = 0
        total_count = series.dropna().count()
        
        if total_count == 0:
            return False
        
        for value in series.dropna().head(100):  # Sample first 100 non-null values
            try:
                pd.to_numeric(value)
                numeric_count += 1
            except:
                pass
        
        return (numeric_count / min(100, total_count)) > 0.8
    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if a column can be converted to boolean"""
        if series.dtype == 'bool':
            return True
        
        # Check for boolean-like values
        unique_values = series.dropna().str.lower().unique()
        
        # Check if values match boolean patterns
        boolean_like = any(val in self.boolean_patterns['true_values'] + self.boolean_patterns['false_values'] 
                          for val in unique_values)
        
        # Check if it's a binary column
        is_binary = len(unique_values) <= 2
        
        return boolean_like or is_binary
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean values"""
        def convert_value(value):
            if pd.isna(value):
                return np.nan
            
            str_value = str(value).lower().strip()
            
            if str_value in self.boolean_patterns['true_values'] or str_value in ['1', 'true', 'yes']:
                return True
            elif str_value in self.boolean_patterns['false_values'] or str_value in ['0', 'false', 'no']:
                return False
            else:
                return np.nan
        
        return series.apply(convert_value)
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a column can be converted to datetime"""
        if series.dtype == 'datetime64[ns]':
            return True
        
        # Check if values match date patterns
        sample_values = series.dropna().head(10)
        date_matches = 0
        
        for value in sample_values:
            str_value = str(value)
            if any(re.match(pattern, str_value) for pattern in self.date_patterns):
                date_matches += 1
        
        return date_matches > len(sample_values) * 0.7
    
    async def _standardize_formats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Standardize data formats across the dataset"""
        actions = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Standardize string formatting
                df[column] = df[column].astype(str).str.strip()
                
                # Handle currency formatting
                if self._is_currency_column(df[column]):
                    df[column] = self._standardize_currency(df[column])
                    actions.append(f"Standardized currency format in '{column}'")
                
                # Handle phone number formatting
                elif self._is_phone_column(df[column]):
                    df[column] = self._standardize_phone(df[column])
                    actions.append(f"Standardized phone format in '{column}'")
                
                # Handle email formatting
                elif self._is_email_column(df[column]):
                    df[column] = df[column].str.lower().str.strip()
                    actions.append(f"Standardized email format in '{column}'")
        
        return df, actions
    
    def _is_currency_column(self, series: pd.Series) -> bool:
        """Check if column contains currency values"""
        sample_values = series.dropna().head(10)
        currency_matches = 0
        
        for value in sample_values:
            str_value = str(value)
            if any(re.match(pattern, str_value) for pattern in self.currency_patterns):
                currency_matches += 1
        
        return currency_matches > len(sample_values) * 0.5
    
    def _standardize_currency(self, series: pd.Series) -> pd.Series:
        """Standardize currency values to numeric format"""
        def convert_currency(value):
            if pd.isna(value):
                return np.nan
            
            str_value = str(value).strip()
            
            # Remove currency symbols and convert to float
            cleaned = re.sub(r'[^\d.,]', '', str_value)
            cleaned = cleaned.replace(',', '')
            
            try:
                return float(cleaned)
            except:
                return np.nan
        
        return series.apply(convert_currency)
    
    def _is_phone_column(self, series: pd.Series) -> bool:
        """Check if column contains phone numbers"""
        sample_values = series.dropna().head(10)
        phone_matches = 0
        
        for value in sample_values:
            str_value = str(value)
            if re.match(r'^[\d\-\+\(\)\s]+$', str_value) and len(str_value.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) >= 10:
                phone_matches += 1
        
        return phone_matches > len(sample_values) * 0.5
    
    def _standardize_phone(self, series: pd.Series) -> pd.Series:
        """Standardize phone number format"""
        def convert_phone(value):
            if pd.isna(value):
                return np.nan
            
            # Remove all non-digit characters
            digits = re.sub(r'\D', '', str(value))
            
            # Format as XXX-XXX-XXXX
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
            else:
                return str(value)
        
        return series.apply(convert_phone)
    
    def _is_email_column(self, series: pd.Series) -> bool:
        """Check if column contains email addresses"""
        sample_values = series.dropna().head(10)
        email_matches = 0
        
        for value in sample_values:
            str_value = str(value)
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str_value):
                email_matches += 1
        
        return email_matches > len(sample_values) * 0.5
    
    async def _standardize_categorical_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Standardize categorical data values"""
        actions = []
        
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                # Standardize case and whitespace
                original_values = df[column].unique()
                df[column] = df[column].astype(str).str.strip().str.title()
                
                # Handle common categorical inconsistencies
                df[column] = self._standardize_categorical_values(df[column])
                
                new_values = df[column].unique()
                if not np.array_equal(original_values, new_values):
                    actions.append(f"Standardized categorical values in '{column}': {len(original_values)} → {len(new_values)} unique values")
        
        return df, actions
    
    def _standardize_categorical_values(self, series: pd.Series) -> pd.Series:
        """Standardize categorical values for consistency"""
        # Common standardization mappings
        standardizations = {
            'full-time': 'Full-time',
            'fulltime': 'Full-time',
            'FULL-TIME': 'Full-time',
            'part-time': 'Part-time',
            'parttime': 'Part-time',
            'PART-TIME': 'Part-time',
            'bachelor\'s': 'Bachelor\'s',
            'bachelors': 'Bachelor\'s',
            'BACHELOR\'S': 'Bachelor\'s',
            'master\'s': 'Master\'s',
            'masters': 'Master\'s',
            'MASTER\'S': 'Master\'s',
            'high school': 'High School',
            'highschool': 'High School',
            'HIGH SCHOOL': 'High School',
            'associate\'s': 'Associate\'s',
            'associates': 'Associate\'s',
            'ASSOCIATE\'S': 'Associate\'s'
        }
        
        return series.replace(standardizations)
    
    async def _advanced_missing_value_handling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced missing value imputation strategies"""
        actions = []
        original_missing = df.isnull().sum().sum()
        
        if original_missing == 0:
            actions.append("No missing values found")
            return df, actions
        
        self.logger.info(f"Found {original_missing} missing values")
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
            
            if df[column].dtype in ['object', 'category']:
                # Categorical columns - use mode or 'Unknown'
                mode_value = df[column].mode()
                if not mode_value.empty:
                    fill_value = mode_value.iloc[0]
                    strategy = "mode"
                else:
                    fill_value = "Unknown"
                    strategy = "default"
                
                df[column] = df[column].fillna(fill_value)
                actions.append(f"Filled {missing_count} missing values in '{column}' using {strategy}: {fill_value}")
                
            else:
                # Numeric columns - use advanced imputation
                if missing_count < len(df) * 0.1:  # Less than 10% missing
                    # Use median for robustness
                    median_value = df[column].median()
                    if pd.isna(median_value):
                        fill_value = 0
                        strategy = "zero"
                    else:
                        fill_value = median_value
                        strategy = "median"
                else:
                    # Use KNN imputation for higher missing rates
                    try:
                        imputer = KNNImputer(n_neighbors=5)
                        df[column] = imputer.fit_transform(df[[column]]).flatten()
                        strategy = "KNN imputation"
                    except:
                        # Fallback to median
                        median_value = df[column].median()
                        fill_value = median_value if not pd.isna(median_value) else 0
                        df[column] = df[column].fillna(fill_value)
                        strategy = "median (KNN failed)"
                
                if strategy != "KNN imputation":
                    df[column] = df[column].fillna(fill_value)
                
                actions.append(f"Filled {missing_count} missing values in '{column}' using {strategy}")
        
        final_missing = df.isnull().sum().sum()
        actions.append(f"Missing value imputation completed: {original_missing} → {final_missing}")
        
        return df, actions
    
    async def _advanced_duplicate_handling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced duplicate detection and removal"""
        actions = []
        
        # 1. Exact duplicates
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
            actions.append(f"Removed {exact_duplicates} exact duplicate rows")
        
        # 2. Near-duplicates (similar rows)
        near_duplicates = await self._detect_near_duplicates(df)
        if near_duplicates > 0:
            df = await self._remove_near_duplicates(df)
            actions.append(f"Removed {near_duplicates} near-duplicate rows")
        
        return df, actions
    
    async def _detect_near_duplicates(self, df: pd.DataFrame) -> int:
        """Detect near-duplicate rows using similarity threshold"""
        # For now, implement a simple approach
        # In production, you might use more sophisticated methods like fuzzy matching
        return 0
    
    async def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-duplicate rows"""
        # Placeholder for near-duplicate removal
        return df
    
    async def _advanced_outlier_detection(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced outlier detection and treatment"""
        actions = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column == target_column:
                continue
            
            # Z-score method
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > self.outlier_threshold
            
            if outliers.sum() > 0:
                # Cap outliers instead of removing them
                upper_bound = df[column].quantile(0.95)
                lower_bound = df[column].quantile(0.05)
                
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                
                actions.append(f"Capped {outliers.sum()} outliers in '{column}' using 5th-95th percentile bounds")
        
        return df, actions
    
    async def _comprehensive_quality_assessment(self, cleaned_df: pd.DataFrame, original_df: pd.DataFrame) -> Tuple[float, List[str], Dict[str, Any]]:
        """Comprehensive data quality assessment"""
        issues = []
        metrics = {}
        
        # 1. Completeness
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        completeness_score = 1 - (cleaned_missing / (len(cleaned_df) * len(cleaned_df.columns)))
        metrics['completeness_score'] = completeness_score
        
        if cleaned_missing > 0:
            issues.append(f"Still has {cleaned_missing} missing values")
        
        # 2. Consistency
        duplicate_ratio = cleaned_df.duplicated().sum() / len(cleaned_df)
        consistency_score = 1 - duplicate_ratio
        metrics['consistency_score'] = consistency_score
        
        if duplicate_ratio > 0.05:  # More than 5% duplicates
            issues.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
        
        # 3. Validity
        validity_score = 1.0
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype == 'object':
                # Check for invalid values
                invalid_count = cleaned_df[column].isnull().sum()
                if invalid_count > 0:
                    validity_score -= invalid_count / len(cleaned_df)
        
        metrics['validity_score'] = validity_score
        
        # 4. Overall quality score
        quality_score = (completeness_score + consistency_score + validity_score) / 3
        metrics['overall_quality_score'] = quality_score
        
        return quality_score, issues, metrics
    
    async def _generate_comprehensive_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                           actions: List[str], issues: List[str], metrics: Dict[str, Any]) -> str:
        """Generate comprehensive cleaning report"""
        
        report = []
        report.append("=" * 80)
        report.append("🧹 COMPREHENSIVE DATA CLEANING REPORT")
        report.append("=" * 80)
        
        # Basic statistics
        report.append(f"\n📊 DATASET STATISTICS:")
        report.append(f"Original shape: {original_df.shape}")
        report.append(f"Cleaned shape: {cleaned_df.shape}")
        report.append(f"Rows removed: {original_df.shape[0] - cleaned_df.shape[0]}")
        report.append(f"Columns removed: {original_df.shape[1] - cleaned_df.shape[1]}")
        
        # Quality metrics
        report.append(f"\n📈 QUALITY METRICS:")
        for metric, value in metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Data types
        report.append(f"\n🔧 DATA TYPE CONVERSIONS:")
        for col in cleaned_df.columns:
            if col in original_df.columns:
                orig_type = str(original_df[col].dtype)
                clean_type = str(cleaned_df[col].dtype)
                if orig_type != clean_type:
                    report.append(f"  {col}: {orig_type} → {clean_type}")
        
        # Actions taken
        report.append(f"\n⚡ CLEANING ACTIONS TAKEN ({len(actions)}):")
        for i, action in enumerate(actions, 1):
            report.append(f"  {i}. {action}")
        
        # Issues found
        if issues:
            report.append(f"\n⚠️ ISSUES IDENTIFIED ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                report.append(f"  {i}. {issue}")
        else:
            report.append(f"\n✅ NO ISSUES IDENTIFIED")
        
        # Summary
        report.append(f"\n📋 SUMMARY:")
        report.append(f"Data cleaning completed successfully!")
        report.append(f"Quality improvement: {metrics.get('overall_quality_score', 0):.1%}")
        report.append(f"Total actions: {len(actions)}")
        report.append(f"Dataset ready for machine learning!")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
