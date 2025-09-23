import google.generativeai as genai
import pandas as pd
import numpy as np
import io
import traceback
from typing import Dict, List, Tuple, Any
import json
from code_executor import SafeCodeExecutor

class DataCleaningAgent:
    """
    AI-powered data cleaning agent using Google Gemini Flash 2.0
    """
    
    def __init__(self, api_key: str):
        """Initialize the agent with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.df = None
        self.original_df = None
        self.cleaning_history = []
        self.code_executor = SafeCodeExecutor()  # Use enhanced code executor
        
    def load_dataset(self, file_path: str = None, dataframe: pd.DataFrame = None) -> bool:
        """Load dataset from file path or dataframe"""
        try:
            if dataframe is not None:
                self.df = dataframe.copy()
            elif file_path:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    self.df = pd.read_json(file_path)
                else:
                    raise ValueError("Unsupported file format")
            else:
                raise ValueError("Either file_path or dataframe must be provided")
                
            self.original_df = self.df.copy()
            return True
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Comprehensive dataset analysis using Gemini"""
        if self.df is None:
            raise ValueError("No dataset loaded")
        
        # Generate detailed structural information
        numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=['object']).columns)
        datetime_cols = list(self.df.select_dtypes(include=['datetime64']).columns)
        
        # Get detailed shape information
        null_counts = self.df.isnull().sum()
        duplicate_info = {
            'total_duplicates': self.df.duplicated().sum(),
            'duplicate_subset_analysis': {}
        }
        
        # Analyze each column's unique values and patterns
        column_analysis = {}
        for col in self.df.columns:
            column_analysis[col] = {
                'dtype': str(self.df[col].dtype),
                'null_count': int(null_counts[col]),
                'null_percentage': float(null_counts[col] / len(self.df) * 100),
                'unique_count': int(self.df[col].nunique()),
                'unique_percentage': float(self.df[col].nunique() / len(self.df) * 100),
                'sample_values': list(self.df[col].dropna().head(3).astype(str)),
                'memory_usage': int(self.df[col].memory_usage(deep=True))
            }
            
            if col in numeric_cols and not self.df[col].empty:
                column_analysis[col].update({
                    'min_value': float(self.df[col].min()) if pd.notna(self.df[col].min()) else None,
                    'max_value': float(self.df[col].max()) if pd.notna(self.df[col].max()) else None,
                    'mean_value': float(self.df[col].mean()) if pd.notna(self.df[col].mean()) else None,
                    'std_value': float(self.df[col].std()) if pd.notna(self.df[col].std()) else None
                })
        
        # Generate enhanced analysis prompt with complete structural information
        analysis_prompt = f"""
        As an expert data scientist, analyze this dataset with COMPLETE STRUCTURAL KNOWLEDGE:
        
        CRITICAL DATASET STRUCTURE:
        - Total Rows: {self.df.shape[0]}
        - Total Columns: {self.df.shape[1]}
        - DataFrame Index: RangeIndex(start=0, stop={self.df.shape[0]}, step=1)
        - Memory Usage: {self.df.memory_usage(deep=True).sum()} bytes
        
        COLUMN CATEGORIES:
        - Numeric columns ({len(numeric_cols)}): {numeric_cols}
        - Categorical columns ({len(categorical_cols)}): {categorical_cols}
        - Datetime columns ({len(datetime_cols)}): {datetime_cols}
        
        DETAILED COLUMN ANALYSIS:
        {chr(10).join([f"Column '{col}': {info}" for col, info in column_analysis.items()])}
        
        SAMPLE DATA STRUCTURE:
        {self.df.head(3).to_string()}
        
        DATA QUALITY ISSUES:
        - Total missing values: {self.df.isnull().sum().sum()}
        - Missing value distribution: {null_counts.to_dict()}
        - Total duplicates: {duplicate_info['total_duplicates']}
        - Columns with >50% missing: {[col for col, count in null_counts.items() if count/len(self.df) > 0.5]}
        
        STATISTICAL SUMMARY:
        {self.df.describe(include='all').to_string() if not self.df.empty else "No data for statistics"}
        
        PERFORM EXPERT DATA SCIENTIST ANALYSIS:
        
        1. **STATISTICAL PROFILE ANALYSIS**:
           - For each numeric column: distribution shape, skewness, outlier patterns
           - Determine optimal imputation strategy (mean vs median vs mode vs advanced)
           - Identify correlation patterns between columns for intelligent imputation
        
        2. **DATA QUALITY ASSESSMENT**:
           - Detect corrupted values (e.g., "1.4cvc", "0.2sdfx" in numeric columns)
           - Identify malformed categorical values (e.g., "setosa45" instead of "setosa")
           - Assess data consistency and logical constraints
        
        3. **DOMAIN-AWARE CLEANING STRATEGY**:
           - Understand what each column represents (measurements, categories, IDs)
           - Apply domain-specific validation rules and constraints
           - Recommend intelligent type conversions and value corrections
        
        4. **ADVANCED IMPUTATION RECOMMENDATIONS**:
           - For numeric data: Choose mean (normal distribution) vs median (skewed) vs KNN (correlated features)
           - For categorical data: Mode vs domain knowledge vs pattern matching
           - Consider multivariate imputation for related columns
        
        5. **OUTLIER HANDLING STRATEGY**:
           - Statistical methods (IQR, Z-score) vs domain constraints
           - Decide: remove, cap, or keep outliers based on context
           - Document reasoning for outlier decisions
        
        6. **DATA TYPE OPTIMIZATION**:
           - Recommend optimal pandas dtypes for memory efficiency
           - Categorical encoding strategies for categorical variables
           - Date/time parsing and standardization
        
        CRITICAL: Think like a senior data scientist. Every cleaning decision should be:
        - **Statistically sound**: Based on data distribution analysis
        - **Domain appropriate**: Considering what the data represents
        - **Logically consistent**: Maintaining data relationships
        - **Reproducible**: With clear reasoning for each choice
        
        Format as detailed data science report with specific, actionable recommendations.
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            return {
                'analysis': response.text,
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict(),
                'missing_values': self.df.isnull().sum().to_dict(),
                'duplicates': self.df.duplicated().sum()
            }
        except Exception as e:
            print(f"Error in dataset analysis: {str(e)}")
            return {'error': str(e)}
    
    def generate_cleaning_code(self, analysis_result: Dict[str, Any]) -> str:
        """Generate Python code for data cleaning based on analysis"""
        
        # Get detailed dataset structure for robust code generation
        numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=['object']).columns)
        datetime_cols = list(self.df.select_dtypes(include=['datetime64']).columns)
        
        code_generation_prompt = f"""
        GENERATE EXPERT-LEVEL DATA CLEANING CODE AS A SENIOR DATA SCIENTIST:
        
        DATA SCIENCE ANALYSIS RESULTS:
        {analysis_result.get('analysis', '')}
        
        DATASET TECHNICAL PROFILE:
        - Shape: {self.df.shape} | Index: RangeIndex(0, {self.df.shape[0]})
        - Numeric Columns: {numeric_cols} 
        - Categorical Columns: {categorical_cols}
        - Current Data Types: {dict(self.df.dtypes)}
        - Missing Values: {dict(self.df.isnull().sum())}
        - Statistical Summary: {self.df.describe().to_dict() if len(numeric_cols) > 0 else 'No numeric data'}
        
        ACTUAL DATA FOR INSPECTION:
        {self.df.head(5).to_string()}
        
        IMPLEMENT EXPERT DATA SCIENTIST APPROACH:
        
        1. **INTELLIGENT DATA TYPE DETECTION & CORRECTION**:
           - Detect corrupted numeric values (e.g., "1.4cvc", "0.2sdfx") and extract valid numbers
           - Clean categorical values (e.g., "setosa45" → "setosa") using pattern matching
           - Convert string numbers to proper numeric types with error handling
        
        2. **STATISTICAL IMPUTATION STRATEGY**:
           - Analyze distribution: use mean for normal, median for skewed distributions
           - For categorical: use mode or domain knowledge
           - Consider KNN imputation for correlated features
           - NEVER leave missing values as None - always impute intelligently
        
        3. **ADVANCED OUTLIER HANDLING**:
           - Calculate statistical boundaries (IQR method: Q1-1.5*IQR, Q3+1.5*IQR)
           - Apply domain constraints (e.g., measurements can't be negative)
           - Document outlier treatment decisions
        
        4. **DOMAIN-AWARE DATA VALIDATION**:
           - Apply logical constraints based on data meaning
           - Ensure categorical values match expected patterns
           - Validate numeric ranges make sense for the domain
        
        MANDATORY CODING RULES TO PREVENT ERRORS:
        
        1. DUPLICATE REMOVAL (CRITICAL - DO NOT CREATE MORE DUPLICATES):
           ```python
           # CORRECT: Remove duplicates using drop_duplicates()
           initial_dups = cleaned_df.duplicated().sum()
           cleaned_df = cleaned_df.drop_duplicates(keep='first')
           final_dups = cleaned_df.duplicated().sum()
           print(f"Removed {{initial_dups - final_dups}} duplicate rows")
           
           # NEVER USE: pd.concat() for duplicate handling - it CREATES duplicates!
           # WRONG: cleaned_df = pd.concat([cleaned_df, some_subset])
           ```
        
        2. ALWAYS validate lengths before assignment:
           ```python
           # GOOD: Always check lengths
           if len(new_values) == len(df):
               df['column'] = new_values
           ```
        
        3. Use .loc for safe assignments:
           ```python
           # GOOD: Safe indexing
           df.loc[mask, 'column'] = df.loc[mask, 'column'].fillna(value)
           ```
        
        4. For imputation, ensure shape consistency:
           ```python
           # GOOD: Shape-aware imputation
           subset = df[numeric_cols].copy()
           imputed = SimpleImputer().fit_transform(subset)
           df[numeric_cols] = pd.DataFrame(imputed, columns=numeric_cols, index=df.index)
           ```
        
        5. Handle filtering operations carefully:
           ```python
           # GOOD: Maintain index alignment
           mask = some_condition
           df.loc[mask, 'column'] = df.loc[mask, 'column'].transform(some_function)
           ```
        
        6. Always copy DataFrames for safety:
           ```python
           # GOOD: Safe copying
           cleaned_df = df.copy()
           ```
        
        CRITICAL: NO IMPORT STATEMENTS ALLOWED!
        All libraries are PRE-IMPORTED and available directly:
        
        AVAILABLE VARIABLES (NO IMPORTS NEEDED):
        - pd (pandas library - already imported)
        - np (numpy library - already imported)  
        - df (your input dataframe - already available)
        - StandardScaler, LabelEncoder, MinMaxScaler (sklearn classes - directly available)
        - SimpleImputer, KNNImputer, IsolationForest, KMeans (sklearn classes - directly available)
        
        FORBIDDEN: Do NOT write any import statements like:
        - import pandas as pd  # WRONG - already available as 'pd'
        - import numpy as np   # WRONG - already available as 'np' 
        - from sklearn import anything  # WRONG - classes already available directly
        
        ERROR PREVENTION CHECKLIST:
        ✓ DUPLICATE COUNT DECREASES (never increases)
        ✓ Use drop_duplicates() not pd.concat() for duplicate removal
        ✓ All array assignments check length compatibility
        ✓ Index alignment maintained throughout
        ✓ Shape preservation after transformations
        ✓ Proper use of .loc and .iloc for indexing
        ✓ DataFrame copying to avoid reference issues
        ✓ Try-except blocks with pandas fallbacks
        
        EXPERT DATA SCIENTIST CODE TEMPLATE (NO IMPORTS!):
        ```python
        # NO IMPORT STATEMENTS - Everything pre-imported: pd, np, sklearn classes, etc.
        
        # Step 1: Initialize with data science approach
        cleaned_df = df.copy()
        print(f"Starting cleaning process - Shape: {{cleaned_df.shape}}")
        print(f"Initial missing values: {{cleaned_df.isnull().sum().sum()}}")
        
        # Step 2: INTELLIGENT DATA TYPE CORRECTION & CORRUPTION CLEANUP
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if col in {numeric_cols}:  # Should be numeric but stored as object
                # Smart corruption handling - extract numbers from mixed text
                original_values = cleaned_df[col].copy()
                
                # Try to extract numeric parts from corrupted values
                cleaned_df[col] = cleaned_df[col].astype(str).str.extract(r'([0-9]*\\.?[0-9]+)')[0]
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # If too many values became NaN, use more lenient approach
                missing_after = cleaned_df[col].isnull().sum()
                missing_rate = missing_after / len(cleaned_df)
                
                if missing_rate > 0.8:  # If >80% became missing, be more lenient
                    print(f"High corruption in {{col}}, using fallback approach")
                    # Restore and try simpler conversion
                    cleaned_df[col] = pd.to_numeric(original_values, errors='coerce')
                    
                    # If still too many missing, use domain knowledge
                    if cleaned_df[col].isnull().sum() > len(cleaned_df) * 0.5:
                        # For measurements like petal/sepal, use reasonable defaults
                        median_val = cleaned_df[col].median()
                        if pd.isna(median_val):
                            # Use domain knowledge - typical flower measurements
                            default_val = 1.0 if 'petal' in col.lower() else 3.0
                            cleaned_df[col] = cleaned_df[col].fillna(default_val)
                            print(f"Used domain default {{default_val}} for highly corrupted {{col}}")
                
                print(f"Converted {{col}} to numeric, {{cleaned_df[col].isnull().sum()}} missing values remain")
            else:  # Categorical columns  
                # Clean corrupted categories like "setosa45" → "setosa"
                cleaned_df[col] = cleaned_df[col].str.replace(r'[0-9]+', '', regex=True)
                cleaned_df[col] = cleaned_df[col].str.strip()
                print(f"Cleaned categorical values in {{col}}")
        
        # Step 3: ROBUST STATISTICAL IMPUTATION (NEVER LEAVE NONE VALUES!)
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[col].isnull().any():
                original_missing = cleaned_df[col].isnull().sum()
                
                # Calculate valid data percentage
                valid_data_pct = (len(cleaned_df) - original_missing) / len(cleaned_df)
                
                if valid_data_pct > 0.1:  # If we have >10% valid data
                    # Analyze distribution to choose best imputation
                    try:
                        skewness = cleaned_df[col].skew()
                        if pd.isna(skewness) or abs(skewness) < 0.5:  # Normal distribution
                            fill_value = cleaned_df[col].mean()
                            strategy = "mean"
                        else:  # Skewed distribution
                            fill_value = cleaned_df[col].median()
                            strategy = "median"
                        
                        # Handle case where mean/median calculation fails
                        if pd.isna(fill_value):
                            raise ValueError("Statistical calculation failed")
                            
                    except (ValueError, RuntimeWarning):
                        # Fallback to simple valid value or domain default
                        valid_values = cleaned_df[col].dropna()
                        if len(valid_values) > 0:
                            fill_value = valid_values.iloc[0]  # Use first valid value
                            strategy = "first_valid"
                        else:
                            # Domain knowledge fallback
                            fill_value = 1.0 if 'petal' in col.lower() else 3.0
                            strategy = "domain_default"
                else:
                    # Very little valid data - use domain knowledge
                    fill_value = 1.0 if 'petal' in col.lower() else 3.0
                    strategy = "domain_default"
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Imputed {{original_missing}} values in {{col}} using {{strategy}}: {{fill_value:.3f}}")
        
        # For categorical columns - ROBUST IMPUTATION
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if cleaned_df[col].isnull().any():
                original_missing = cleaned_df[col].isnull().sum()
                
                # Try mode first
                mode_series = cleaned_df[col].mode()
                if not mode_series.empty:
                    mode_value = mode_series.iloc[0]
                    strategy = "mode"
                else:
                    # No mode available - use domain knowledge or first valid value
                    valid_values = cleaned_df[col].dropna()
                    if len(valid_values) > 0:
                        mode_value = valid_values.iloc[0]
                        strategy = "first_valid"
                    else:
                        # Complete fallback - use column name for intelligent default
                        if 'species' in col.lower():
                            mode_value = 'unknown_species'
                        elif 'category' in col.lower() or 'class' in col.lower():
                            mode_value = 'unknown_category'
                        else:
                            mode_value = 'unknown'
                        strategy = "intelligent_default"
                
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                print(f"Imputed {{original_missing}} values in {{col}} using {{strategy}}: {{mode_value}}")
        
        # Step 4: STATISTICAL OUTLIER DETECTION & HANDLING
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Cap outliers to boundaries (preserve data while reducing extreme values)
                cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
                print(f"Capped {{outlier_count}} outliers in {{col}} to range [{{lower_bound:.3f}}, {{upper_bound:.3f}}]")
        
        # Step 5: DUPLICATE REMOVAL (CRITICAL: reduce count, never increase)
        initial_duplicates = cleaned_df.duplicated().sum()
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
        final_duplicates = cleaned_df.duplicated().sum()
        print(f"Duplicates removed: {{initial_duplicates}} → {{final_duplicates}}")
        
        # Step 6: DATA TYPE OPTIMIZATION & VALIDATION
        # Convert to optimal dtypes for memory efficiency
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if cleaned_df[col].nunique() < 50:  # Low cardinality categorical
                cleaned_df[col] = cleaned_df[col].astype('category')
                print(f"Optimized {{col}} to category type")
        
        # Step 7: FINAL VALIDATION & QUALITY REPORT
        print(f"\\n=== CLEANING COMPLETED ===")
        print(f"Final shape: {{cleaned_df.shape}}")
        print(f"Missing values remaining: {{cleaned_df.isnull().sum().sum()}}")
        print(f"Data types: {{dict(cleaned_df.dtypes)}}")
        
        # Critical assertions
        assert cleaned_df.isnull().sum().sum() == 0, "ERROR: Missing values still present!"
        assert cleaned_df.shape[1] >= {len(self.df.columns)}, "ERROR: Lost columns during cleaning!"
        assert cleaned_df.duplicated().sum() <= {self.df.duplicated().sum()}, "ERROR: Duplicates increased!"
        
        print("✅ All data quality checks passed!")
        ```
        
        CRITICAL: Return ONLY executable Python code that follows these safety rules.
        Every operation must preserve DataFrame integrity and prevent length mismatches.
        """
        
        try:
            response = self.model.generate_content(code_generation_prompt)
            code = response.text.strip()
            
            # Clean up code forma
            # tting
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
                
            return code.strip()
        except Exception as e:
            print(f"Error generating cleaning code: {str(e)}")
            return ""
    
    def execute_cleaning_code(self, code: str, max_retries: int = 3) -> Tuple[bool, str, pd.DataFrame]:
        """Execute the cleaning code with error handling and self-correction using SafeCodeExecutor"""
        
        for attempt in range(max_retries):
            # Use the enhanced SafeCodeExecutor with sklearn support and validation
            success, output, cleaned_df = self.code_executor.execute_with_timeout(code, self.df, timeout=30)
            
            if success and cleaned_df is not None:
                return True, output, cleaned_df
            else:
                error_msg = output  # Contains the error message
                print(f"Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < max_retries - 1:
                    # Try to fix the code using Gemini
                    code = self.fix_code_with_ai(code, error_msg)
                else:
                    return False, f"Code execution failed after {max_retries} attempts: {error_msg}", self.df
        
        return False, "Maximum retries exceeded", self.df
    
    def fix_code_with_ai(self, faulty_code: str, error_message: str) -> str:
        """Advanced AI-powered code fixing with comprehensive error analysis"""
        
        # Analyze the error type and provide targeted solutions
        error_type = self._classify_error(error_message)
        
        # Get current dataset state for precise fixing
        numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=['object']).columns)
        datetime_cols = list(self.df.select_dtypes(include=['datetime64']).columns)
        
        fix_prompt = f"""
        CRITICAL ERROR FIXING TASK - EXPERT PANDAS DEBUGGING
        
        ERROR CLASSIFICATION: {error_type}
        
        FAILED CODE:
        {faulty_code}
        
        EXACT ERROR MESSAGE:
        {error_message}
        
        COMPLETE DATASET CONTEXT FOR DEBUGGING:
        - DataFrame Shape: {self.df.shape} (EXACTLY {self.df.shape[0]} rows, {self.df.shape[1]} columns)
        - Column Names: {list(self.df.columns)}
        - Numeric Columns: {numeric_cols}
        - Categorical Columns: {categorical_cols}
        - Datetime Columns: {datetime_cols}
        - Data Types: {dict(self.df.dtypes)}
        - Index Type: RangeIndex(0, {self.df.shape[0]})
        - Missing Values: {dict(self.df.isnull().sum())}
        - Memory Usage: {self.df.memory_usage(deep=True).sum()} bytes
        
        ACTUAL DATA STRUCTURE:
        {self.df.head(2).to_string()}
        
        DETAILED ERROR ANALYSIS AND SOLUTION STRATEGY:
        
        DUPLICATE HANDLING ERROR SOLUTIONS:
        CRITICAL: If duplicates increased instead of decreased:
        1. NEVER use pd.concat() for removing duplicates - it CREATES them!
        2. Use ONLY: cleaned_df = cleaned_df.drop_duplicates(keep='first')
        3. Validate: final_dups = cleaned_df.duplicated().sum() must be <= original
        4. Remove any pd.concat(), pd.append(), or similar concatenation operations
        
        {"LENGTH MISMATCH ERROR SOLUTIONS:" if "length" in error_message.lower() else ""}
        {"If 'All arrays must be of the same length':" if "length" in error_message.lower() else ""}
        {"1. Check every array assignment for length compatibility" if "length" in error_message.lower() else ""}
        {"2. Use .loc for filtered assignments: df.loc[mask, col] = values" if "length" in error_message.lower() else ""}
        {"3. Ensure imputation returns same shape: pd.DataFrame(imputed, index=df.index)" if "length" in error_message.lower() else ""}
        {"4. Validate shapes before any assignment operation" if "length" in error_message.lower() else ""}
        
        {"KEY ERROR SOLUTIONS:" if "KeyError" in error_message else ""}
        {"1. Verify all column names exist before referencing" if "KeyError" in error_message else ""}
        {"2. Use .get() method for safe column access" if "KeyError" in error_message else ""}
        {"3. Check column names after any operations that might change them" if "KeyError" in error_message else ""}
        
        DATA SCIENCE ERROR SOLUTIONS:
        
        MISSING VALUES STILL PRESENT:
        If final assertion fails on "Missing values still present!":
        1. Handle "Mean of empty slice" errors with robust statistical fallbacks
        2. Use domain knowledge when statistical calculation fails (e.g., petal=1.0, sepal=3.0)
        3. Check corruption rate - if >80% missing after conversion, use fallback approach
        4. For categorical: use first valid value if mode calculation fails
        5. ALWAYS provide fallback strategies: first_valid_value or domain_defaults
        6. Add try-except blocks around statistical calculations (mean/median/mode)
        
        IMPORT ERROR SOLUTIONS:
        CRITICAL: If you see "__import__ not found" or any ImportError:
        1. REMOVE ALL import statements from your code - they are FORBIDDEN!
        2. Use pre-available variables: pd, np, df, StandardScaler, SimpleImputer, etc.
        3. Start your code directly with: cleaned_df = df.copy()
        4. NO lines like: import pandas, from sklearn import, etc.
        
        DATA CORRUPTION HANDLING:
        If data has corrupted values like "1.4cvc", "setosa45":
        1. Use pd.to_numeric(errors='coerce') to extract valid numbers
        2. Use regex to clean categorical values: str.replace(r'[0-9]+', '', regex=True)
        3. Always follow with intelligent imputation for created missing values
        
        {"VALUE ERROR SOLUTIONS:" if "ValueError" in error_message else ""}
        {"1. Validate data types before conversions" if "ValueError" in error_message else ""}
        {"2. Handle empty or null values before operations" if "ValueError" in error_message else ""}
        {"3. Use try-except with fallback strategies" if "ValueError" in error_message else ""}
        
        MANDATORY FIX REQUIREMENTS:
        
        1. DIAGNOSIS: Identify the EXACT line causing the error
        2. ROOT CAUSE: Understand WHY the arrays/shapes don't match  
        3. SOLUTION: Implement length-safe alternative
        4. VALIDATION: Add shape checks to prevent recurrence
        
        ROBUST CODING PATTERNS TO USE:
        
        ```python
        # Pattern 1: Safe column assignment
        if 'column_name' in cleaned_df.columns:
            cleaned_df['column_name'] = cleaned_df['column_name'].fillna(default_value)
        
        # Pattern 2: Length-validated assignment
        new_values = some_transformation()
        if len(new_values) == len(cleaned_df):
            cleaned_df['new_column'] = new_values
        else:
            print(f"Length mismatch: expected {{len(cleaned_df)}}, got {{len(new_values)}}")
        
        # Pattern 3: Shape-preserving imputation
        if numeric_cols:
            subset = cleaned_df[numeric_cols].copy()
            if not subset.empty:
                imputed = SimpleImputer().fit_transform(subset)
                cleaned_df[numeric_cols] = pd.DataFrame(imputed, columns=numeric_cols, index=cleaned_df.index)
        
        # Pattern 4: Safe filtering operations
        mask = (cleaned_df['column'] > threshold)
        cleaned_df.loc[mask, 'column'] = cleaned_df.loc[mask, 'column'] * correction_factor
        
        # Pattern 5: Defensive programming
        try:
            # risky operation
            result = risky_transformation()
        except Exception as e:
            print(f"Operation failed: {{e}}, using fallback")
            result = safe_fallback()
        ```
        
        CRITICAL SUCCESS CRITERIA:
        - Code must execute without ANY length/shape errors
        - All operations must preserve DataFrame integrity
        - Final shape must be logical and consistent
        - Include detailed error handling and validation
        
        GENERATE COMPLETELY FIXED CODE that:
        1. Eliminates the specific error completely
        2. Uses defensive programming patterns
        3. Includes comprehensive validation
        4. Handles edge cases that caused the original failure
        5. Returns 'cleaned_df' with guaranteed shape consistency
        
        Return ONLY the corrected Python code with NO explanations.
        The code must be bulletproof and execute successfully.
        """
        
        try:
            response = self.model.generate_content(fix_prompt)
            fixed_code = response.text.strip()
            
            # Clean up code formatting
            if fixed_code.startswith('```python'):
                fixed_code = fixed_code[9:]
            if fixed_code.startswith('```'):
                fixed_code = fixed_code[3:]
            if fixed_code.endswith('```'):
                fixed_code = fixed_code[:-3]
                
            return fixed_code.strip()
        except Exception as e:
            print(f"Error fixing code with AI: {str(e)}")
            return faulty_code  # Return original if fixing fails
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the error type for targeted fixing"""
        error_msg_lower = error_message.lower()
        
        if "mean of empty slice" in error_msg_lower or "statistical calculation failed" in error_msg_lower:
            return "STATISTICAL_CALCULATION_ERROR"
        elif "missing values still present" in error_msg_lower:
            return "MISSING_VALUES_ERROR"
        elif "__import__ not found" in error_msg_lower or "importerror" in error_msg_lower:
            return "FORBIDDEN_IMPORT_ERROR"
        elif "created duplicates instead of removing" in error_msg_lower:
            return "DUPLICATE_CREATION_ERROR"
        elif "all arrays must be of the same length" in error_msg_lower:
            return "LENGTH_MISMATCH_ERROR"
        elif "keyerror" in error_msg_lower:
            return "COLUMN_ACCESS_ERROR"
        elif "valueerror" in error_msg_lower:
            return "DATA_TYPE_ERROR"
        elif "indexerror" in error_msg_lower:
            return "INDEX_OUT_OF_BOUNDS"
        elif "attributeerror" in error_msg_lower:
            return "METHOD_ATTRIBUTE_ERROR"
        elif "typeerror" in error_msg_lower:
            return "TYPE_MISMATCH_ERROR"
        elif "no module named" in error_msg_lower:
            return "IMPORT_ERROR"
        elif "cannot convert" in error_msg_lower:
            return "CONVERSION_ERROR"
        elif "invalid literal" in error_msg_lower:
            return "PARSING_ERROR"
        else:
            return "UNKNOWN_ERROR"
    
    def validate_cleaning(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the cleaning process and generate report"""
        
        validation_report = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_removed': original_df.shape[1] - cleaned_df.shape[1],
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum(),
            'duplicates_before': original_df.duplicated().sum(),
            'duplicates_after': cleaned_df.duplicated().sum(),
            'data_types_changed': {}
        }
        
        # Check data type changes
        for col in original_df.columns:
            if col in cleaned_df.columns:
                if str(original_df[col].dtype) != str(cleaned_df[col].dtype):
                    validation_report['data_types_changed'][col] = {
                        'before': str(original_df[col].dtype),
                        'after': str(cleaned_df[col].dtype)
                    }
        
        return validation_report
    
    def clean_dataset(self) -> Dict[str, Any]:
        """Main method to clean the dataset"""
        if self.df is None:
            return {'error': 'No dataset loaded'}
        
        try:
            # Step 1: Analyze dataset
            print("🔍 Analyzing dataset...")
            analysis = self.analyze_dataset()
            
            if 'error' in analysis:
                return analysis
            
            # Step 2: Generate cleaning code
            print("🤖 Generating cleaning code...")
            cleaning_code = self.generate_cleaning_code(analysis)
            
            if not cleaning_code:
                return {'error': 'Failed to generate cleaning code'}
            
            # Step 3: Execute and test cleaning code
            print("⚡ Executing cleaning code...")
            success, message, cleaned_df = self.execute_cleaning_code(cleaning_code)
            
            if not success:
                return {'error': message, 'code': cleaning_code}
            
            # Step 4: Validate cleaning
            print("✅ Validating cleaning results...")
            validation = self.validate_cleaning(self.original_df, cleaned_df)
            
            return {
                'success': True,
                'analysis': analysis,
                'cleaning_code': cleaning_code,
                'cleaned_dataframe': cleaned_df,
                'validation_report': validation,
                'message': 'Dataset cleaning completed successfully!'
            }
            
        except Exception as e:
            return {'error': f'Unexpected error during cleaning: {str(e)}'}