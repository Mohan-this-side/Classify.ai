# 🧪 Test Data Collection for Data Cleaning Agent

This directory contains various types of messy datasets to test and improve our data cleaning agent's robustness.

## 📊 Test Datasets

### 1. **iris_clean.csv** - Clean Baseline Dataset
- **Purpose**: Clean reference dataset for comparison
- **Issues**: None (clean data)
- **Rows**: 150, **Columns**: 5
- **Features**: sepal_length, sepal_width, petal_length, petal_width, species

### 2. **messy_duplicates.csv** - Duplicate Records
- **Purpose**: Test duplicate detection and removal
- **Issues**: 
  - Complete duplicate rows (IDs 1, 2, 3 appear multiple times)
  - Need to identify and remove redundant records
- **Rows**: 12, **Columns**: 6
- **Features**: id, name, age, salary, department, email

### 3. **messy_missing_values.csv** - Missing Data Scenarios
- **Purpose**: Test missing value handling strategies
- **Issues**:
  - Missing last names, emails, phone numbers
  - Missing ages, income, credit scores
  - Various patterns of missingness
- **Rows**: 10, **Columns**: 10
- **Features**: customer_id, first_name, last_name, email, phone, age, income, credit_score, loan_amount, status

### 4. **messy_data_types.csv** - Data Type Inconsistencies
- **Purpose**: Test data type conversion and standardization
- **Issues**:
  - Boolean values in different formats (True/true/TRUE/False/false/FALSE)
  - Mixed string representations of boolean data
- **Rows**: 10, **Columns**: 8
- **Features**: product_id, product_name, price, quantity, in_stock, rating, launch_date, category

### 5. **messy_outliers.csv** - Outlier Detection
- **Purpose**: Test outlier detection and treatment
- **Issues**:
  - Extreme values (999) in score columns
  - Impossible attendance rates (999%)
  - Need statistical outlier detection
- **Rows**: 28, **Columns**: 7
- **Features**: student_id, name, age, math_score, english_score, science_score, attendance_rate

### 6. **messy_inconsistent_format.csv** - Format Inconsistencies
- **Purpose**: Test format standardization
- **Issues**:
  - Mixed currency symbols ($, €) and positions
  - Different date formats (YYYY-MM-DD vs DD/MM/YYYY)
  - Inconsistent decimal separators
- **Rows**: 10, **Columns**: 7
- **Features**: transaction_id, customer_name, amount, currency, date, payment_method, status

### 7. **messy_categorical_inconsistency.csv** - Categorical Data Issues
- **Purpose**: Test categorical data standardization
- **Issues**:
  - Inconsistent case (Full-time/full-time/FULL-TIME)
  - Mixed education level formats
  - Need to standardize categorical values
- **Rows**: 15, **Columns**: 7
- **Features**: employee_id, name, department, position, employment_status, education_level, experience_years

### 8. **messy_structural_errors.csv** - Structural Issues
- **Purpose**: Test structural error detection
- **Issues**:
  - Inconsistent column naming (ID vs Id vs id)
  - Mixed case in column headers
  - Need to standardize column names
- **Rows**: 10, **Columns**: 8
- **Features**: ID, Full Name, Email Address, Phone Number, Age, Salary, Department, Start Date

### 9. **messy_mixed_data_types.csv** - Mixed Data Types
- **Purpose**: Test mixed data type handling
- **Issues**:
  - Boolean values as 1/0, true/false, TRUE/FALSE
  - Mixed representations of the same logical values
  - Need to standardize boolean encoding
- **Rows**: 10, **Columns**: 8
- **Features**: record_id, product_name, price, quantity, is_available, rating, launch_date, category_id

### 10. **messy_encoding_issues.csv** - Character Encoding
- **Purpose**: Test international character handling
- **Issues**:
  - Non-ASCII characters in names and addresses
  - International names and locations
  - UTF-8 encoding challenges
- **Rows**: 10, **Columns**: 7
- **Features**: customer_id, name, email, phone, address, city, country

## 🎯 Testing Scenarios

### **Data Quality Issues to Test:**
1. **Duplicate Detection**: Identify and remove exact and near-duplicates
2. **Missing Value Handling**: Impute, delete, or flag missing values
3. **Data Type Conversion**: Convert strings to appropriate numeric/boolean types
4. **Outlier Detection**: Identify and treat statistical outliers
5. **Format Standardization**: Standardize dates, currencies, and formats
6. **Categorical Standardization**: Normalize categorical values
7. **Structural Cleanup**: Fix column names and data structure
8. **Encoding Issues**: Handle international characters properly
9. **Validation**: Ensure data integrity and consistency
10. **Transformation**: Apply appropriate data transformations

### **Expected Cleaning Actions:**
- Remove duplicate records while preserving unique data
- Impute missing values using appropriate strategies
- Convert data types to optimal formats
- Detect and treat outliers appropriately
- Standardize formats and naming conventions
- Normalize categorical variables
- Validate data accuracy and consistency
- Handle encoding issues gracefully

## 🚀 Usage

Use these datasets to test the data cleaning agent's ability to handle various real-world data quality issues. Each dataset represents common problems found in messy data that need to be addressed before machine learning model training.

## 📈 Success Metrics

A robust data cleaning agent should be able to:
- ✅ Identify and handle all types of data quality issues
- ✅ Preserve data integrity while cleaning
- ✅ Apply appropriate cleaning strategies for each issue type
- ✅ Generate comprehensive cleaning reports
- ✅ Handle edge cases and unusual data patterns
- ✅ Maintain data relationships and dependencies
