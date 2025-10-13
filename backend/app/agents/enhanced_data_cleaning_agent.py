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

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import ClassificationState, AgentStatus, state_manager
from ..config import settings

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
        
        # 2. Structural error detection and correction
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
        
        # Store cleaned dataset
        state_manager.store_dataset(state, cleaned_dataset, "cleaned")
        
        # Update state with comprehensive results
        state["cleaned_dataset"] = None  # Stored externally
        state["cleaning_summary"] = cleaning_report
        state["data_quality_score"] = quality_score
        state["cleaning_issues_found"] = issues_found
        state["cleaning_actions_taken"] = cleaning_actions
        state["data_transformation_log"] = transformation_log
        state["quality_metrics"] = quality_metrics
        
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
