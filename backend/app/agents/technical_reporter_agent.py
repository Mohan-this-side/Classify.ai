"""
Technical Reporter Agent

This agent is responsible for:
- Generating comprehensive technical reports
- Creating executive summaries
- Producing detailed documentation
- Generating Jupyter notebooks with all code
- Creating downloadable artifacts
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import ClassificationState, AgentStatus


class TechnicalReporterAgent(BaseAgent):
    """
    Technical Reporter Agent for generating comprehensive reports and documentation
    """
    
    def __init__(self):
        super().__init__("technical_reporter", "1.0.0")
        self.logger = logging.getLogger("agent.technical_reporter")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Technical Reporter Agent for generating comprehensive reports and documentation",
            "capabilities": [
                "Executive summary generation",
                "Technical documentation creation",
                "Jupyter notebook generation",
                "Performance analysis reports",
                "Model documentation",
                "Downloadable artifact creation"
            ],
            "dependencies": ["data_cleaning", "eda_analysis", "feature_engineering", "ml_building", "model_evaluation"]
        }
    
    def get_dependencies(self) -> list:
        """Get list of agent dependencies"""
        return ["data_cleaning", "eda_analysis", "feature_engineering", "ml_building", "model_evaluation"]
    
    async def execute(self, state: ClassificationState) -> ClassificationState:
        """
        Execute technical reporting process
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with report results
        """
        try:
            self.logger.info("Starting technical reporting process")
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(state)
            
            # Generate detailed technical documentation
            technical_documentation = self._generate_technical_documentation(state)
            
            # Generate comprehensive educational content
            educational_content = self._generate_educational_content(state)
            
            # Generate detailed markdown explanations
            markdown_explanations = self._generate_markdown_explanations(state)
            
            # Generate comprehensive usage instructions
            usage_instructions = self._generate_usage_instructions(state)
            
            # Generate Jupyter notebook
            notebook_path = self._generate_jupyter_notebook(state)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(state)
            
            # Generate limitations analysis
            limitations = self._generate_limitations(state)
            
            # Generate future improvements
            future_improvements = self._generate_future_improvements(state)
            
            # Create final comprehensive report
            final_report = self._create_final_report(
                executive_summary, technical_documentation, educational_content,
                recommendations, limitations, future_improvements
            )
            
            # Update state with results
            state["final_report"] = final_report
            state["executive_summary"] = executive_summary
            state["technical_documentation"] = technical_documentation
            state["educational_content"] = educational_content
            state["markdown_explanations"] = markdown_explanations
            state["usage_instructions"] = usage_instructions
            state["notebook_path"] = notebook_path
            state["recommendations"] = recommendations
            state["limitations"] = limitations
            state["future_improvements"] = future_improvements
            
            # Update agent status
            state["agent_statuses"]["technical_reporting"] = AgentStatus.COMPLETED
            state["completed_agents"].append("technical_reporting")
            
            self.logger.info("Technical reporting completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in technical reporting: {str(e)}")
            state["agent_statuses"]["technical_reporting"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state
    
    def _generate_executive_summary(self, state: ClassificationState) -> str:
        """Generate executive summary"""
        try:
            # Extract key information from state
            dataset_shape = state.get("dataset_shape", (0, 0))
            target_column = state.get("target_column", "unknown")
            model_name = state.get("best_model", "unknown")
            accuracy = state.get("evaluation_metrics", {}).get("accuracy", 0.0)
            f1_score = state.get("evaluation_metrics", {}).get("f1_weighted", 0.0)
            
            summary = f"""
            # Executive Summary
            
            ## Project Overview
            This machine learning classification project successfully processed a dataset with {dataset_shape[0]} rows and {dataset_shape[1]} columns to predict the '{target_column}' variable.
            
            ## Key Results
            - **Model Performance**: Achieved {accuracy:.1%} accuracy with {f1_score:.3f} F1-score
            - **Selected Model**: {model_name}
            - **Data Quality**: Comprehensive data cleaning and preprocessing completed
            - **Feature Engineering**: Advanced feature selection and engineering applied
            
            ## Business Impact
            The developed model demonstrates {'excellent' if accuracy > 0.9 else 'good' if accuracy > 0.8 else 'moderate'} performance and is ready for deployment in production environments.
            
            ## Technical Highlights
            - Automated data cleaning and preprocessing pipeline
            - Advanced feature engineering and selection
            - Comprehensive model evaluation and validation
            - Complete documentation and reproducible code
            
            ## Next Steps
            1. Deploy model to production environment
            2. Monitor model performance in real-world conditions
            3. Implement continuous learning and model updates
            4. Expand to additional classification tasks
            """
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return "Executive summary generation failed due to insufficient data."
    
    def _generate_technical_documentation(self, state: ClassificationState) -> str:
        """Generate detailed technical documentation"""
        try:
            # Extract technical details
            dataset_info = self._extract_dataset_info(state)
            cleaning_info = self._extract_cleaning_info(state)
            model_info = self._extract_model_info(state)
            evaluation_info = self._extract_evaluation_info(state)
            
            documentation = f"""
            # Technical Documentation
            
            ## Dataset Information
            {dataset_info}
            
            ## Data Cleaning Process
            {cleaning_info}
            
            ## Model Development
            {model_info}
            
            ## Model Evaluation
            {evaluation_info}
            
            ## Implementation Details
            - **Framework**: Python with scikit-learn, XGBoost, and LightGBM
            - **Validation**: 5-fold cross-validation
            - **Hyperparameter Tuning**: GridSearchCV optimization
            - **Feature Engineering**: Automated feature selection and transformation
            
            ## Code Reproducibility
            All code has been generated and documented in the accompanying Jupyter notebook, ensuring complete reproducibility of results.
            """
            
            return documentation.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating technical documentation: {str(e)}")
            return "Technical documentation generation failed."
    
    def _generate_educational_content(self, state: ClassificationState) -> str:
        """Generate comprehensive educational content explaining ML concepts"""
        try:
            # Generate educational sections
            data_science_overview = self._generate_data_science_overview()
            data_cleaning_education = self._generate_data_cleaning_education(state)
            feature_engineering_education = self._generate_feature_engineering_education()
            model_selection_education = self._generate_model_selection_education(state)
            evaluation_education = self._generate_evaluation_education()
            best_practices = self._generate_best_practices()
            common_pitfalls = self._generate_common_pitfalls()
            
            educational_content = f"""
            # Educational Content: Understanding Machine Learning Classification
            
            {data_science_overview}
            
            {data_cleaning_education}
            
            {feature_engineering_education}
            
            {model_selection_education}
            
            {evaluation_education}
            
            {best_practices}
            
            {common_pitfalls}
            
            ## Conclusion
            
            This educational content provides a comprehensive understanding of the machine learning classification process. By following these concepts and best practices, you can build robust, reliable, and interpretable machine learning models for your classification tasks.
            
            Remember that machine learning is an iterative process. Don't be afraid to experiment with different approaches, but always validate your results and understand the limitations of your models.
            """
            
            return educational_content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating educational content: {str(e)}")
            return "Educational content generation failed."
    
    def _generate_markdown_explanations(self, state: ClassificationState) -> str:
        """Generate detailed markdown explanations for each step in the ML pipeline"""
        try:
            # Generate explanations for each major step
            data_loading_explanations = self._generate_data_loading_explanations(state)
            data_cleaning_explanations = self._generate_data_cleaning_explanations(state)
            feature_engineering_explanations = self._generate_feature_engineering_explanations(state)
            model_training_explanations = self._generate_model_training_explanations(state)
            model_evaluation_explanations = self._generate_model_evaluation_explanations(state)
            model_persistence_explanations = self._generate_model_persistence_explanations(state)
            
            markdown_explanations = f"""
            # Detailed Markdown Explanations for ML Pipeline Steps
            
            {data_loading_explanations}
            
            {data_cleaning_explanations}
            
            {feature_engineering_explanations}
            
            {model_training_explanations}
            
            {model_evaluation_explanations}
            
            {model_persistence_explanations}
            
            ## Summary
            
            These markdown explanations provide step-by-step guidance for understanding and implementing each part of the machine learning classification pipeline. Each explanation includes:
            
            - **Purpose**: Why this step is important
            - **Method**: How the technique works
            - **Implementation**: Code details and parameters
            - **Output**: What to expect from each step
            - **Best Practices**: Tips for optimal results
            - **Common Issues**: Potential problems and solutions
            
            Use these explanations as a learning resource and reference guide for your machine learning projects.
            """
            
            return markdown_explanations.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating markdown explanations: {str(e)}")
            return "Markdown explanations generation failed."
    
    def _generate_usage_instructions(self, state: ClassificationState) -> str:
        """Generate comprehensive usage instructions for the generated model and Jupyter notebook"""
        try:
            # Generate different types of usage instructions
            model_loading_instructions = self._generate_model_loading_instructions(state)
            prediction_instructions = self._generate_prediction_instructions(state)
            notebook_usage_instructions = self._generate_notebook_usage_instructions(state)
            deployment_instructions = self._generate_deployment_instructions(state)
            troubleshooting_guide = self._generate_troubleshooting_guide(state)
            example_workflows = self._generate_example_workflows(state)
            
            usage_instructions = f"""
            # Usage Instructions: How to Use Your Trained Model and Jupyter Notebook
            
            {model_loading_instructions}
            
            {prediction_instructions}
            
            {notebook_usage_instructions}
            
            {deployment_instructions}
            
            {troubleshooting_guide}
            
            {example_workflows}
            
            ## Quick Start Summary
            
            To get started with your trained model:
            
            1. **Load the model**: Use the provided loading function with your model file
            2. **Prepare new data**: Ensure it matches the training data format
            3. **Make predictions**: Use the prediction function with your prepared data
            4. **Run the notebook**: Execute cells step-by-step to understand the process
            5. **Deploy if needed**: Follow deployment instructions for production use
            
            For detailed explanations and troubleshooting, refer to the sections above.
            """
            
            return usage_instructions.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating usage instructions: {str(e)}")
            return "Usage instructions generation failed."
    
    def _generate_model_loading_instructions(self, state: ClassificationState) -> str:
        """Generate instructions for loading the trained model"""
        best_model = state.get('best_model', 'RandomForestClassifier')
        model_filename = state.get('model_filename', 'trained_model.joblib')
        
        return f"""
        ## 1. Loading Your Trained Model
        
        ### Prerequisites
        Before loading your model, ensure you have the required dependencies installed:
        
        ```bash
        pip install pandas numpy scikit-learn joblib matplotlib seaborn
        ```
        
        ### Basic Model Loading
        ```python
        import joblib
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        def load_trained_model(model_path):
            \"\"\"
            Load the trained model and preprocessing pipeline
            
            Args:
                model_path (str): Path to the saved model file (.joblib)
            
            Returns:
                dict: Model package containing model, scaler, and metadata
            \"\"\"
            try:
                model_package = joblib.load(model_path)
                print(f"Model loaded successfully from {{model_path}}")
                print(f"Model type: {{model_package.get('model_name', 'Unknown')}}")
                print(f"Training date: {{model_package.get('training_date', 'Unknown')}}")
                return model_package
            except FileNotFoundError:
                print(f"Error: Model file not found at {{model_path}}")
                return None
            except Exception as e:
                print(f"Error loading model: {{str(e)}}")
                return None
        
        # Load your model
        model_package = load_trained_model('{model_filename}')
        ```
        
        ### Model Package Contents
        Your saved model package contains:
        
        - **`model`**: The trained {best_model} classifier
        - **`scaler`**: Preprocessing scaler used during training
        - **`feature_names`**: List of feature names in correct order
        - **`target_column`**: Name of the target variable
        - **`model_name`**: Type of model used
        - **`training_date`**: When the model was trained
        - **`performance_metrics`**: Model performance on test data
        
        ### Verifying Model Load
        ```python
        if model_package is not None:
            print("Model package contents:")
            for key, value in model_package.items():
                if key != 'model':  # Don't print the actual model object
                    print(f"  {{key}}: {{value}}")
            
            # Check if all required components are present
            required_keys = ['model', 'scaler', 'feature_names']
            missing_keys = [key for key in required_keys if key not in model_package]
            
            if missing_keys:
                print(f"Warning: Missing required components: {{missing_keys}}")
            else:
                print("✓ All required components present")
        ```
        
        ### Common Loading Issues
        
        #### File Not Found
        **Problem**: `FileNotFoundError` when loading model
        **Solution**: Check file path and ensure model file exists
        ```python
        import os
        if os.path.exists('{model_filename}'):
            model_package = joblib.load('{model_filename}')
        else:
            print("Model file not found. Please check the file path.")
        ```
        
        #### Version Compatibility
        **Problem**: Model saved with different library versions
        **Solution**: Check and update library versions
        ```python
        import sklearn
        print(f"Current scikit-learn version: {{sklearn.__version__}}")
        
        # If you get version warnings, update scikit-learn:
        # pip install --upgrade scikit-learn
        ```
        
        #### Memory Issues
        **Problem**: Out of memory when loading large models
        **Solution**: Load model components separately if needed
        ```python
        # For very large models, you might need to load components separately
        # This is rarely needed for most classification models
        ```
        """
    
    def _generate_prediction_instructions(self, state: ClassificationState) -> str:
        """Generate instructions for making predictions with the trained model"""
        target_column = state.get('target_column', 'target')
        
        return f"""
        ## 2. Making Predictions with Your Model
        
        ### Preparing New Data
        Before making predictions, ensure your new data matches the training data format:
        
        ```python
        def prepare_new_data(new_data, model_package):
            \"\"\"
            Prepare new data for prediction using the same preprocessing as training
            
            Args:
                new_data (pd.DataFrame): New data to make predictions on
                model_package (dict): Loaded model package
            
            Returns:
                pd.DataFrame: Preprocessed data ready for prediction
            \"\"\"
            # Get feature names from training
            feature_names = model_package['feature_names']
            
            # Ensure all required features are present
            missing_features = set(feature_names) - set(new_data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {{missing_features}}")
            
            # Select only the features used in training
            X_new = new_data[feature_names].copy()
            
            # Apply the same preprocessing (scaling)
            scaler = model_package['scaler']
            X_new_scaled = scaler.transform(X_new)
            
            return pd.DataFrame(X_new_scaled, columns=feature_names)
        
        # Example: Prepare new data
        new_data = pd.DataFrame({{
            'feature1': [1.2, 3.4, 5.6],
            'feature2': [2.1, 4.3, 6.5],
            # ... other features
        }})
        
        X_new_prepared = prepare_new_data(new_data, model_package)
        ```
        
        ### Making Predictions
        ```python
        def make_predictions(model_package, new_data):
            \"\"\"
            Make predictions on new data using the trained model
            
            Args:
                model_package (dict): Loaded model package
                new_data (pd.DataFrame): New data to predict on
            
            Returns:
                dict: Prediction results including predictions and probabilities
            \"\"\"
            # Prepare the data
            X_new = prepare_new_data(new_data, model_package)
            
            # Get the trained model
            model = model_package['model']
            
            # Make predictions
            predictions = model.predict(X_new)
            
            # Get prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)
                class_names = model.classes_
            else:
                probabilities = None
                class_names = None
            
            return {{
                'predictions': predictions,
                'probabilities': probabilities,
                'class_names': class_names,
                'feature_names': X_new.columns.tolist()
            }}
        
        # Make predictions
        results = make_predictions(model_package, new_data)
        print("Predictions:", results['predictions'])
        ```
        
        ### Understanding Predictions
        ```python
        def interpret_predictions(results, new_data):
            \"\"\"
            Interpret and display prediction results in a user-friendly format
            
            Args:
                results (dict): Results from make_predictions function
                new_data (pd.DataFrame): Original new data
            \"\"\"
            predictions = results['predictions']
            probabilities = results['probabilities']
            class_names = results['class_names']
            
            # Create results DataFrame
            results_df = new_data.copy()
            results_df['prediction'] = predictions
            
            if probabilities is not None and class_names is not None:
                # Add probability columns
                for i, class_name in enumerate(class_names):
                    results_df[f'prob_{{class_name}}'] = probabilities[:, i]
            
            return results_df
        
        # Interpret results
        prediction_results = interpret_predictions(results, new_data)
        print(prediction_results)
        ```
        
        ### Batch Prediction Example
        ```python
        def batch_predictions(model_package, data_file_path):
            \"\"\"
            Make predictions on a large dataset from a file
            
            Args:
                model_package (dict): Loaded model package
                data_file_path (str): Path to CSV file with new data
            
            Returns:
                pd.DataFrame: Results with predictions
            \"\"\"
            # Load data in chunks for large files
            chunk_size = 1000
            results_list = []
            
            for chunk in pd.read_csv(data_file_path, chunksize=chunk_size):
                chunk_results = make_predictions(model_package, chunk)
                chunk_df = interpret_predictions(chunk_results, chunk)
                results_list.append(chunk_df)
            
            # Combine all results
            final_results = pd.concat(results_list, ignore_index=True)
            return final_results
        
        # Example: Process large dataset
        # large_results = batch_predictions(model_package, 'large_dataset.csv')
        ```
        
        ### Common Prediction Issues
        
        #### Feature Mismatch
        **Problem**: New data has different features than training data
        **Solution**: Ensure feature names and types match exactly
        ```python
        # Check feature compatibility
        training_features = set(model_package['feature_names'])
        new_data_features = set(new_data.columns)
        
        if training_features != new_data_features:
            print("Feature mismatch detected!")
            print(f"Missing in new data: {{training_features - new_data_features}}")
            print(f"Extra in new data: {{new_data_features - training_features}}")
        ```
        
        #### Data Type Issues
        **Problem**: Data types don't match training data
        **Solution**: Convert data types to match training format
        ```python
        # Convert data types to match training
        for col in new_data.columns:
            if new_data[col].dtype == 'object':
                # Try to convert to numeric
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
        ```
        
        #### Missing Values
        **Problem**: New data has missing values
        **Solution**: Handle missing values before prediction
        ```python
        # Check for missing values
        missing_values = new_data.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values detected:")
            print(missing_values[missing_values > 0])
            
            # Handle missing values (use same strategy as training)
            new_data = new_data.fillna(new_data.median())  # For numerical data
        ```
        """
    
    def _generate_notebook_usage_instructions(self, state: ClassificationState) -> str:
        """Generate instructions for using the Jupyter notebook"""
        notebook_path = state.get('notebook_path', 'classification_analysis.ipynb')
        
        return f"""
        ## 3. Using the Jupyter Notebook
        
        ### Opening the Notebook
        The generated Jupyter notebook (`{notebook_path}`) contains a complete, reproducible analysis of your machine learning project.
        
        #### Option 1: Jupyter Notebook
        ```bash
        # Install Jupyter if not already installed
        pip install jupyter
        
        # Start Jupyter Notebook
        jupyter notebook
        
        # Navigate to and open the notebook file
        ```
        
        #### Option 2: JupyterLab
        ```bash
        # Install JupyterLab if not already installed
        pip install jupyterlab
        
        # Start JupyterLab
        jupyterlab
        
        # Open the notebook file
        ```
        
        #### Option 3: VS Code
        ```bash
        # Open VS Code in the project directory
        code .
        
        # Open the .ipynb file
        # Make sure you have the Python extension installed
        ```
        
        ### Notebook Structure
        The notebook is organized into the following sections:
        
        1. **Project Information**: Overview and metadata
        2. **Required Libraries**: All necessary imports
        3. **Dataset Loading**: Data loading and initial exploration
        4. **Data Cleaning**: Comprehensive data preprocessing
        5. **Feature Engineering**: Feature creation and selection
        6. **Model Training**: Algorithm comparison and selection
        7. **Model Evaluation**: Performance analysis and visualization
        8. **Feature Importance**: Understanding model decisions
        9. **Model Persistence**: Saving and loading models
        10. **Usage Instructions**: How to use the trained model
        
        ### Running the Notebook
        
        #### Step-by-Step Execution
        1. **Start from the top**: Run cells in order from beginning to end
        2. **Check outputs**: Verify each cell produces expected results
        3. **Read explanations**: Each code block has detailed markdown explanations
        4. **Modify if needed**: Adjust parameters or add your own analysis
        
        #### Quick Run (All Cells)
        ```python
        # To run all cells at once (use with caution)
        # In Jupyter: Kernel -> Restart & Run All
        # Or use this Python script:
        
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        
        # Load the notebook
        with open('{notebook_path}', 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute all cells
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {{'metadata': {{'path': '.'}}}})
        
        # Save executed notebook
        with open('{notebook_path}', 'w') as f:
            nbformat.write(nb, f)
        ```
        
        ### Customizing the Notebook
        
        #### Modifying Parameters
        ```python
        # Example: Change test size
        test_size = 0.3  # Instead of default 0.2
        
        # Example: Try different models
        models = {{
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }}
        ```
        
        #### Adding Your Own Analysis
        ```python
        # Add new cells for additional analysis
        # Example: Custom visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Your custom analysis code here
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='target', y='feature1')
        plt.title('Custom Analysis')
        plt.show()
        ```
        
        ### Common Notebook Issues
        
        #### Kernel Not Starting
        **Problem**: Jupyter kernel won't start
        **Solution**: Check Python installation and dependencies
        ```bash
        # Check Python version
        python --version
        
        # Install required packages
        pip install -r requirements.txt
        
        # Restart Jupyter
        jupyter notebook --generate-config
        ```
        
        #### Import Errors
        **Problem**: Module not found errors
        **Solution**: Install missing packages
        ```python
        # Install missing packages
        !pip install missing_package_name
        
        # Or use conda
        !conda install missing_package_name
        ```
        
        #### Memory Issues
        **Problem**: Out of memory errors
        **Solution**: Use smaller datasets or optimize code
        ```python
        # For large datasets, process in chunks
        chunk_size = 1000
        for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
            # Process chunk
            process_chunk(chunk)
        ```
        
        ### Exporting Results
        
        #### Save as HTML
        ```bash
        # Convert notebook to HTML
        jupyter nbconvert --to html {notebook_path}
        ```
        
        #### Save as PDF
        ```bash
        # Convert notebook to PDF (requires LaTeX)
        jupyter nbconvert --to pdf {notebook_path}
        ```
        
        #### Save as Python Script
        ```bash
        # Convert notebook to Python script
        jupyter nbconvert --to script {notebook_path}
        ```
        """
    
    def _generate_deployment_instructions(self, state: ClassificationState) -> str:
        """Generate instructions for deploying the model"""
        return """
        ## 4. Deploying Your Model
        
        ### Local Deployment
        For simple local deployment, you can create a Python script or web application.
        
        #### Simple Python Script
        ```python
        # deploy_model.py
        import joblib
        import pandas as pd
        import sys
        from pathlib import Path
        
        def load_model(model_path):
            return joblib.load(model_path)
        
        def predict_single(model_package, input_data):
            # Prepare data
            X = pd.DataFrame([input_data])
            X = X[model_package['feature_names']]
            
            # Scale data
            X_scaled = model_package['scaler'].transform(X)
            
            # Make prediction
            prediction = model_package['model'].predict(X_scaled)[0]
            probability = model_package['model'].predict_proba(X_scaled)[0]
            
            return {
                'prediction': prediction,
                'probability': probability.tolist(),
                'confidence': max(probability)
            }
        
        if __name__ == "__main__":
            # Load model
            model_package = load_model('trained_model.joblib')
            
            # Example prediction
            input_data = {
                'feature1': 1.5,
                'feature2': 2.3,
                # ... other features
            }
            
            result = predict_single(model_package, input_data)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}")
        ```
        
        #### Flask Web API
        ```python
        # app.py
        from flask import Flask, request, jsonify
        import joblib
        import pandas as pd
        import numpy as np
        
        app = Flask(__name__)
        
        # Load model at startup
        model_package = joblib.load('trained_model.joblib')
        
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Get input data
                data = request.get_json()
                
                # Prepare data
                X = pd.DataFrame([data])
                X = X[model_package['feature_names']]
                X_scaled = model_package['scaler'].transform(X)
                
                # Make prediction
                prediction = model_package['model'].predict(X_scaled)[0]
                probability = model_package['model'].predict_proba(X_scaled)[0]
                
                return jsonify({
                    'prediction': int(prediction),
                    'probability': probability.tolist(),
                    'confidence': float(max(probability))
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        if __name__ == '__main__':
            app.run(debug=True, host='0.0.0.0', port=5000)
        ```
        
        ### Cloud Deployment Options
        
        #### AWS SageMaker
        ```python
        # sagemaker_deploy.py
        import boto3
        import joblib
        from sagemaker.sklearn import SKLearnModel
        from sagemaker import get_execution_role
        
        # Prepare model for SageMaker
        model_package = joblib.load('trained_model.joblib')
        
        # Create inference script
        inference_code = '''
        import joblib
        import pandas as pd
        import json
        
        def model_fn(model_dir):
            return joblib.load(f"{model_dir}/model.joblib")
        
        def input_fn(request_body, request_content_type):
            if request_content_type == 'application/json':
                data = json.loads(request_body)
                return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported content type: {request_content_type}")
        
        def predict_fn(input_data, model):
            X = input_data[model['feature_names']]
            X_scaled = model['scaler'].transform(X)
            prediction = model['model'].predict(X_scaled)[0]
            return prediction
        
        def output_fn(prediction, content_type):
            return json.dumps({"prediction": int(prediction)})
        '''
        
        # Save inference script
        with open('inference.py', 'w') as f:
            f.write(inference_code)
        
        # Deploy to SageMaker
        role = get_execution_role()
        sklearn_model = SKLearnModel(
            model_data='s3://your-bucket/model.tar.gz',
            role=role,
            entry_point='inference.py',
            framework_version='0.23-1'
        )
        
        predictor = sklearn_model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
        ```
        
        #### Google Cloud AI Platform
        ```python
        # gcp_deploy.py
        from google.cloud import aiplatform
        from google.cloud.aiplatform import gapic as aip
        import joblib
        
        # Initialize AI Platform
        aiplatform.init(project='your-project-id', location='us-central1')
        
        # Prepare model
        model_package = joblib.load('trained_model.joblib')
        
        # Create custom prediction routine
        class CustomPredictionRoutine:
            def __init__(self):
                self.model_package = model_package
            
            def predict(self, instances):
                import pandas as pd
                import numpy as np
                
                X = pd.DataFrame(instances)
                X = X[self.model_package['feature_names']]
                X_scaled = self.model_package['scaler'].transform(X)
                
                predictions = self.model_package['model'].predict(X_scaled)
                return predictions.tolist()
        
        # Deploy model
        endpoint = aiplatform.Endpoint.create(
            display_name='classification-model-endpoint'
        )
        
        model = aiplatform.Model.upload(
            display_name='classification-model',
            artifact_uri='gs://your-bucket/model/',
            serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-23:latest'
        )
        
        endpoint.deploy(
            model=model,
            deployed_model_display_name='classification-deployment',
            machine_type='n1-standard-2'
        )
        ```
        
        ### Docker Deployment
        ```dockerfile
        # Dockerfile
        FROM python:3.9-slim
        
        WORKDIR /app
        
        # Install dependencies
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        
        # Copy model and code
        COPY trained_model.joblib .
        COPY app.py .
        
        # Expose port
        EXPOSE 5000
        
        # Run application
        CMD ["python", "app.py"]
        ```
        
        ```bash
        # Build and run Docker container
        docker build -t classification-model .
        docker run -p 5000:5000 classification-model
        ```
        
        ### Monitoring and Maintenance
        
        #### Model Performance Monitoring
        ```python
        # monitor_model.py
        import logging
        import time
        from datetime import datetime
        
        class ModelMonitor:
            def __init__(self, model_package):
                self.model_package = model_package
                self.predictions_log = []
                self.performance_log = []
            
            def log_prediction(self, input_data, prediction, actual=None):
                log_entry = {
                    'timestamp': datetime.now(),
                    'input': input_data,
                    'prediction': prediction,
                    'actual': actual
                }
                self.predictions_log.append(log_entry)
            
            def check_model_drift(self, new_data, threshold=0.1):
                # Compare new data distribution with training data
                # This is a simplified example
                training_stats = self.model_package.get('training_stats', {})
                new_stats = new_data.describe()
                
                # Calculate drift (simplified)
                drift_score = 0
                for col in new_data.columns:
                    if col in training_stats:
                        old_mean = training_stats[col]['mean']
                        new_mean = new_stats[col]['mean']
                        drift_score += abs(new_mean - old_mean) / old_mean
                
                return drift_score > threshold
            
            def generate_report(self):
                return {
                    'total_predictions': len(self.predictions_log),
                    'last_prediction': self.predictions_log[-1] if self.predictions_log else None,
                    'model_health': 'healthy'  # Add actual health checks
                }
        ```
        
        ### Best Practices for Deployment
        
        1. **Version Control**: Keep track of model versions and their performance
        2. **Testing**: Test your deployment with sample data before going live
        3. **Monitoring**: Set up monitoring for model performance and data drift
        4. **Rollback Plan**: Have a plan to rollback to previous model versions
        5. **Documentation**: Document your deployment process and configuration
        6. **Security**: Implement proper authentication and data protection
        7. **Scalability**: Design for the expected load and scale accordingly
        """
    
    def _generate_troubleshooting_guide(self, state: ClassificationState) -> str:
        """Generate troubleshooting guide for common issues"""
        return """
        ## 5. Troubleshooting Guide
        
        ### Common Issues and Solutions
        
        #### Model Loading Issues
        
        **Problem**: `FileNotFoundError` when loading model
        ```
        FileNotFoundError: [Errno 2] No such file or directory: 'trained_model.joblib'
        ```
        **Solutions**:
        1. Check if the model file exists in the current directory
        2. Verify the file path is correct
        3. Ensure you have read permissions for the file
        ```python
        import os
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        print("Model file exists:", os.path.exists('trained_model.joblib'))
        ```
        
        **Problem**: `ModuleNotFoundError` when loading model
        ```
        ModuleNotFoundError: No module named 'sklearn'
        ```
        **Solutions**:
        1. Install missing dependencies
        2. Check Python environment
        3. Verify package versions
        ```bash
        pip install scikit-learn pandas numpy joblib
        # Or
        conda install scikit-learn pandas numpy joblib
        ```
        
        **Problem**: Version compatibility issues
        ```
        UserWarning: Trying to unpickle estimator from version 0.23.2 when using version 1.0.2
        ```
        **Solutions**:
        1. Update scikit-learn to match model version
        2. Retrain model with current version
        3. Use virtual environment with specific versions
        ```bash
        pip install scikit-learn==0.23.2
        # Or retrain with current version
        ```
        
        #### Prediction Issues
        
        **Problem**: `KeyError` when making predictions
        ```
        KeyError: 'feature_name'
        ```
        **Solutions**:
        1. Check feature names in new data
        2. Ensure all required features are present
        3. Verify feature names match exactly (case-sensitive)
        ```python
        # Check feature compatibility
        required_features = set(model_package['feature_names'])
        new_data_features = set(new_data.columns)
        print("Missing features:", required_features - new_data_features)
        print("Extra features:", new_data_features - required_features)
        ```
        
        **Problem**: `ValueError` during prediction
        ```
        ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
        ```
        **Solutions**:
        1. Check for missing values in input data
        2. Handle infinite values
        3. Ensure data types are correct
        ```python
        # Check for problematic values
        print("Missing values:", new_data.isnull().sum().sum())
        print("Infinite values:", np.isinf(new_data.select_dtypes(include=[np.number])).sum().sum())
        
        # Handle missing values
        new_data = new_data.fillna(new_data.median())
        
        # Handle infinite values
        new_data = new_data.replace([np.inf, -np.inf], np.nan)
        new_data = new_data.fillna(new_data.median())
        ```
        
        **Problem**: Shape mismatch errors
        ```
        ValueError: X has 5 features, but RandomForestClassifier is expecting 10 features
        ```
        **Solutions**:
        1. Ensure correct number of features
        2. Check feature order
        3. Verify preprocessing steps
        ```python
        # Check feature count and order
        print("Expected features:", len(model_package['feature_names']))
        print("Actual features:", len(new_data.columns))
        print("Feature order:", model_package['feature_names'])
        print("New data columns:", new_data.columns.tolist())
        ```
        
        #### Data Preprocessing Issues
        
        **Problem**: Scaling errors
        ```
        ValueError: Input contains infinity or a value too large for dtype('float64')
        ```
        **Solutions**:
        1. Check for extreme values before scaling
        2. Use robust scaling methods
        3. Handle outliers appropriately
        ```python
        # Check for extreme values
        print("Data range:", new_data.describe())
        print("Max values:", new_data.max())
        print("Min values:", new_data.min())
        
        # Use robust scaling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(new_data)
        ```
        
        **Problem**: Categorical encoding issues
        ```
        ValueError: Found unknown categories in feature
        ```
        **Solutions**:
        1. Handle unknown categories
        2. Ensure consistent encoding
        3. Use appropriate encoding strategy
        ```python
        # Handle unknown categories
        from sklearn.preprocessing import LabelEncoder
        
        # For each categorical column
        for col in categorical_columns:
            le = LabelEncoder()
            le.fit(training_data[col])
            
            # Handle unknown categories
            new_data[col] = new_data[col].apply(
                lambda x: x if x in le.classes_ else 'unknown'
            )
        ```
        
        #### Performance Issues
        
        **Problem**: Slow prediction times
        **Solutions**:
        1. Optimize data preprocessing
        2. Use batch predictions
        3. Consider model simplification
        ```python
        # Batch processing for large datasets
        def batch_predict(model_package, data, batch_size=1000):
            predictions = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                batch_pred = model_package['model'].predict(batch)
                predictions.extend(batch_pred)
            return predictions
        ```
        
        **Problem**: High memory usage
        **Solutions**:
        1. Process data in chunks
        2. Use memory-efficient data types
        3. Clear unused variables
        ```python
        # Use memory-efficient data types
        new_data = new_data.astype({
            'int_column': 'int32',
            'float_column': 'float32'
        })
        
        # Clear unused variables
        del large_dataframe
        import gc
        gc.collect()
        ```
        
        ### Debugging Tips
        
        #### Enable Verbose Logging
        ```python
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Add debug prints
        print("Model type:", type(model_package['model']))
        print("Scaler type:", type(model_package['scaler']))
        print("Feature names:", model_package['feature_names'])
        ```
        
        #### Validate Data at Each Step
        ```python
        def validate_data(data, step_name):
            print(f"\\n{step_name} validation:")
            print(f"  Shape: {data.shape}")
            print(f"  Data types: {data.dtypes.value_counts()}")
            print(f"  Missing values: {data.isnull().sum().sum()}")
            print(f"  Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Use at each step
        validate_data(new_data, "Raw data")
        validate_data(X_prepared, "Prepared data")
        validate_data(X_scaled, "Scaled data")
        ```
        
        #### Test with Sample Data
        ```python
        # Create minimal test case
        test_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            # ... other features with simple values
        })
        
        # Test prediction
        try:
            result = make_predictions(model_package, test_data)
            print("Test prediction successful:", result)
        except Exception as e:
            print("Test prediction failed:", str(e))
            import traceback
            traceback.print_exc()
        ```
        
        ### Getting Help
        
        If you encounter issues not covered in this guide:
        
        1. **Check the logs**: Look for error messages and stack traces
        2. **Verify data format**: Ensure input data matches training format
        3. **Test with simple data**: Use minimal test cases to isolate issues
        4. **Check dependencies**: Ensure all required packages are installed
        5. **Review documentation**: Check scikit-learn and pandas documentation
        6. **Search online**: Look for similar issues on Stack Overflow or GitHub
        
        ### Contact Information
        
        For additional support:
        - Check the project documentation
        - Review the generated Jupyter notebook
        - Consult the model training logs
        - Contact the development team
        """
    
    def _generate_example_workflows(self, state: ClassificationState) -> str:
        """Generate example workflows for using the model"""
        return """
        ## 6. Example Workflows
        
        ### Workflow 1: Single Prediction
        
        This workflow shows how to make a single prediction on new data.
        
        ```python
        # Complete single prediction workflow
        import joblib
        import pandas as pd
        import numpy as np
        
        def single_prediction_workflow():
            # Step 1: Load the model
            print("Loading model...")
            model_package = joblib.load('trained_model.joblib')
            print("✓ Model loaded successfully")
            
            # Step 2: Prepare new data
            print("\\nPreparing new data...")
            new_data = pd.DataFrame({
                'feature1': [1.5],
                'feature2': [2.3],
                'feature3': [0.8],
                # Add all required features
            })
            print(f"✓ New data prepared: {new_data.shape}")
            
            # Step 3: Make prediction
            print("\\nMaking prediction...")
            X = new_data[model_package['feature_names']]
            X_scaled = model_package['scaler'].transform(X)
            prediction = model_package['model'].predict(X_scaled)[0]
            probability = model_package['model'].predict_proba(X_scaled)[0]
            
            print(f"✓ Prediction: {prediction}")
            print(f"✓ Confidence: {max(probability):.2f}")
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': max(probability)
            }
        
        # Run the workflow
        result = single_prediction_workflow()
        ```
        
        ### Workflow 2: Batch Processing
        
        This workflow shows how to process multiple predictions efficiently.
        
        ```python
        def batch_prediction_workflow(input_file, output_file):
            # Step 1: Load model
            print("Loading model...")
            model_package = joblib.load('trained_model.joblib')
            
            # Step 2: Load input data
            print(f"Loading data from {input_file}...")
            data = pd.read_csv(input_file)
            print(f"✓ Loaded {len(data)} records")
            
            # Step 3: Prepare data
            print("Preparing data...")
            X = data[model_package['feature_names']]
            X_scaled = model_package['scaler'].transform(X)
            
            # Step 4: Make predictions
            print("Making predictions...")
            predictions = model_package['model'].predict(X_scaled)
            probabilities = model_package['model'].predict_proba(X_scaled)
            
            # Step 5: Create results
            print("Creating results...")
            results = data.copy()
            results['prediction'] = predictions
            results['confidence'] = np.max(probabilities, axis=1)
            
            # Add probability columns for each class
            class_names = model_package['model'].classes_
            for i, class_name in enumerate(class_names):
                results[f'prob_{class_name}'] = probabilities[:, i]
            
            # Step 6: Save results
            print(f"Saving results to {output_file}...")
            results.to_csv(output_file, index=False)
            print(f"✓ Results saved: {len(results)} records")
            
            return results
        
        # Example usage
        # results = batch_prediction_workflow('input_data.csv', 'predictions.csv')
        ```
        
        ### Workflow 3: Real-time API
        
        This workflow shows how to create a simple API for real-time predictions.
        
        ```python
        from flask import Flask, request, jsonify
        import joblib
        import pandas as pd
        import numpy as np
        
        app = Flask(__name__)
        
        # Global model variable
        model_package = None
        
        def load_model():
            global model_package
            model_package = joblib.load('trained_model.joblib')
            print("Model loaded for API")
        
        @app.route('/predict', methods=['POST'])
        def predict_api():
            try:
                # Get JSON data
                data = request.get_json()
                
                # Convert to DataFrame
                input_data = pd.DataFrame([data])
                
                # Prepare data
                X = input_data[model_package['feature_names']]
                X_scaled = model_package['scaler'].transform(X)
                
                # Make prediction
                prediction = model_package['model'].predict(X_scaled)[0]
                probability = model_package['model'].predict_proba(X_scaled)[0]
                
                # Return results
                return jsonify({
                    'success': True,
                    'prediction': int(prediction),
                    'probability': probability.tolist(),
                    'confidence': float(max(probability))
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
        
        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'model_loaded': model_package is not None})
        
        if __name__ == '__main__':
            load_model()
            app.run(host='0.0.0.0', port=5000, debug=True)
        ```
        
        ### Workflow 4: Model Validation
        
        This workflow shows how to validate your model on new data with known outcomes.
        
        ```python
        def model_validation_workflow(validation_data, true_labels):
            # Step 1: Load model
            model_package = joblib.load('trained_model.joblib')
            
            # Step 2: Prepare validation data
            X_val = validation_data[model_package['feature_names']]
            X_val_scaled = model_package['scaler'].transform(X_val)
            
            # Step 3: Make predictions
            predictions = model_package['model'].predict(X_val_scaled)
            probabilities = model_package['model'].predict_proba(X_val_scaled)
            
            # Step 4: Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(true_labels, predictions)
            report = classification_report(true_labels, predictions)
            cm = confusion_matrix(true_labels, predictions)
            
            # Step 5: Display results
            print(f"Validation Accuracy: {accuracy:.3f}")
            print("\\nClassification Report:")
            print(report)
            print("\\nConfusion Matrix:")
            print(cm)
            
            # Step 6: Identify misclassifications
            misclassified = validation_data[predictions != true_labels]
            print(f"\\nMisclassified samples: {len(misclassified)}")
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'probabilities': probabilities,
                'misclassified': misclassified
            }
        
        # Example usage
        # validation_results = model_validation_workflow(val_data, val_labels)
        ```
        
        ### Workflow 5: Model Monitoring
        
        This workflow shows how to monitor model performance over time.
        
        ```python
        import json
        from datetime import datetime, timedelta
        
        class ModelMonitor:
            def __init__(self, model_package):
                self.model_package = model_package
                self.predictions_log = []
                self.performance_log = []
            
            def log_prediction(self, input_data, prediction, actual=None):
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'input': input_data,
                    'prediction': int(prediction),
                    'actual': int(actual) if actual is not None else None
                }
                self.predictions_log.append(log_entry)
            
            def calculate_daily_accuracy(self, days=7):
                cutoff_date = datetime.now() - timedelta(days=days)
                recent_predictions = [
                    p for p in self.predictions_log 
                    if datetime.fromisoformat(p['timestamp']) > cutoff_date
                    and p['actual'] is not None
                ]
                
                if not recent_predictions:
                    return None
                
                correct = sum(1 for p in recent_predictions if p['prediction'] == p['actual'])
                total = len(recent_predictions)
                accuracy = correct / total
                
                return {
                    'accuracy': accuracy,
                    'total_predictions': total,
                    'correct_predictions': correct,
                    'period_days': days
                }
            
            def save_logs(self, filename='model_logs.json'):
                with open(filename, 'w') as f:
                    json.dump({
                        'predictions': self.predictions_log,
                        'performance': self.performance_log
                    }, f, indent=2)
            
            def load_logs(self, filename='model_logs.json'):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        self.predictions_log = data.get('predictions', [])
                        self.performance_log = data.get('performance', [])
                except FileNotFoundError:
                    print(f"Log file {filename} not found")
        
        # Example usage
        monitor = ModelMonitor(model_package)
        
        # Log a prediction
        monitor.log_prediction({'feature1': 1.5, 'feature2': 2.3}, 1, 1)
        
        # Check daily accuracy
        daily_accuracy = monitor.calculate_daily_accuracy()
        if daily_accuracy:
            print(f"Daily accuracy: {daily_accuracy['accuracy']:.3f}")
        
        # Save logs
        monitor.save_logs()
        ```
        
        ### Workflow 6: Model Retraining
        
        This workflow shows how to retrain your model with new data.
        
        ```python
        def retrain_model_workflow(new_training_data, new_target, model_package):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            import joblib
            
            # Step 1: Prepare new data
            X_new = new_training_data[model_package['feature_names']]
            y_new = new_target
            
            # Step 2: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
            )
            
            # Step 3: Preprocess data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Step 4: Train new model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Step 5: Evaluate new model
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            print(f"New model - Train accuracy: {train_accuracy:.3f}")
            print(f"New model - Test accuracy: {test_accuracy:.3f}")
            
            # Step 6: Compare with old model
            old_model = model_package['model']
            old_scaler = model_package['scaler']
            
            X_test_old_scaled = old_scaler.transform(X_test)
            old_pred = old_model.predict(X_test_old_scaled)
            old_accuracy = accuracy_score(y_test, old_pred)
            
            print(f"Old model - Test accuracy: {old_accuracy:.3f}")
            
            # Step 7: Save new model if better
            if test_accuracy > old_accuracy:
                new_model_package = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': model_package['feature_names'],
                    'target_column': model_package['target_column'],
                    'model_name': 'RandomForestClassifier',
                    'training_date': datetime.now().isoformat(),
                    'performance_metrics': {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy
                    }
                }
                
                joblib.dump(new_model_package, 'retrained_model.joblib')
                print("✓ New model saved as 'retrained_model.joblib'")
                
                return new_model_package
            else:
                print("Old model performs better, keeping original model")
                return model_package
        
        # Example usage
        # retrained_model = retrain_model_workflow(new_data, new_labels, model_package)
        ```
        
        ### Best Practices for Workflows
        
        1. **Error Handling**: Always include try-catch blocks for robust error handling
        2. **Logging**: Log important events and errors for debugging
        3. **Validation**: Validate input data before processing
        4. **Testing**: Test workflows with sample data before production use
        5. **Documentation**: Document your workflows for future reference
        6. **Monitoring**: Monitor workflow performance and results
        7. **Backup**: Keep backups of important models and data
        """
    
    def _generate_data_loading_explanations(self, state: ClassificationState) -> str:
        """Generate markdown explanations for data loading steps"""
        return """
        ## 1. Data Loading and Initial Exploration
        
        ### Purpose
        The first step in any machine learning project is to load and understand your data. This step establishes the foundation for all subsequent analysis and modeling.
        
        ### Key Concepts
        
        #### Dataset Loading
        ```python
        df = pd.read_csv('dataset.csv')
        ```
        **What it does**: Loads data from a CSV file into a pandas DataFrame
        **Why it matters**: Provides the raw data for analysis and modeling
        **Parameters**: 
        - `filepath`: Path to the CSV file
        - `sep`: Delimiter (default: comma)
        - `header`: Row to use as column names (default: 0)
        
        #### Basic Dataset Information
        ```python
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        ```
        **What it shows**: 
        - **Shape**: Number of rows and columns
        - **Columns**: List of all column names
        **Why it's important**: Gives you a quick overview of data size and structure
        
        #### Data Preview
        ```python
        print(df.head())
        print(df.info())
        print(df.describe())
        ```
        **What each shows**:
        - **head()**: First 5 rows of data
        - **info()**: Data types and memory usage
        - **describe()**: Statistical summary of numerical columns
        **Why it's useful**: Helps identify data types, missing values, and basic patterns
        
        ### Expected Outputs
        - Dataset dimensions (rows × columns)
        - Column names and data types
        - Sample data rows
        - Basic statistics for numerical columns
        - Memory usage information
        
        ### Common Issues and Solutions
        
        #### File Not Found Error
        **Problem**: `FileNotFoundError` when loading data
        **Solution**: Check file path and ensure file exists
        ```python
        import os
        if os.path.exists('dataset.csv'):
            df = pd.read_csv('dataset.csv')
        else:
            print("File not found!")
        ```
        
        #### Encoding Issues
        **Problem**: Special characters not displaying correctly
        **Solution**: Specify encoding parameter
        ```python
        df = pd.read_csv('dataset.csv', encoding='utf-8')
        # or try 'latin-1' or 'cp1252'
        ```
        
        #### Large File Loading
        **Problem**: Memory issues with large datasets
        **Solution**: Load in chunks or use data types
        ```python
        # Load in chunks
        chunk_list = []
        for chunk in pd.read_csv('large_dataset.csv', chunksize=1000):
            chunk_list.append(chunk)
        df = pd.concat(chunk_list, ignore_index=True)
        
        # Or specify data types
        df = pd.read_csv('dataset.csv', dtype={'column1': 'category', 'column2': 'int32'})
        ```
        
        ### Best Practices
        1. **Always check data shape first** - Know what you're working with
        2. **Examine data types** - Ensure they match expectations
        3. **Look for missing values** - Identify data quality issues early
        4. **Check for duplicates** - Understand data uniqueness
        5. **Validate target variable** - Ensure it exists and is properly formatted
        """
    
    def _generate_data_cleaning_explanations(self, state: ClassificationState) -> str:
        """Generate markdown explanations for data cleaning steps"""
        return """
        ## 2. Data Cleaning: Foundation of Quality Models
        
        ### Purpose
        Data cleaning transforms raw data into a clean, consistent format suitable for machine learning. This step directly impacts model performance and reliability.
        
        ### Key Concepts
        
        #### Missing Value Analysis
        ```python
        def analyze_missing_values(df, target_column=None):
            missing_stats = df.isnull().sum()
            missing_percent = (missing_stats / len(df)) * 100
            # ... analysis code ...
        ```
        **What it does**: Identifies and quantifies missing data patterns
        **Why it matters**: Missing data can bias models and reduce performance
        **Key metrics**:
        - **Missing count**: Number of missing values per column
        - **Missing percentage**: Proportion of missing data
        - **Pattern analysis**: Understanding why data is missing
        
        #### Data Type Validation
        ```python
        def validate_and_convert_types(df):
            for col in df.columns:
                if df[col].dtype == 'object':
                    numeric_converted = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_converted.isna().all():
                        df[col] = numeric_converted
        ```
        **What it does**: Ensures data types are appropriate for analysis
        **Why it matters**: Correct data types enable proper statistical operations
        **Common conversions**:
        - Object to numeric (when possible)
        - Object to datetime (for date columns)
        - Categorical encoding for text data
        
        #### Outlier Detection
        ```python
        def detect_outliers_iqr(df, columns=None):
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # ... detection logic ...
        ```
        **What it does**: Identifies data points that are significantly different from others
        **Why it matters**: Outliers can skew model training and predictions
        **Methods**:
        - **IQR Method**: Uses interquartile range (robust to outliers)
        - **Z-Score Method**: Uses standard deviations (sensitive to outliers)
        - **Isolation Forest**: Machine learning-based detection
        
        #### Missing Value Imputation
        ```python
        def impute_missing_values(df, target_column=None):
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        mode_value = df[col].mode()[0]
                        df[col].fillna(mode_value, inplace=True)
        ```
        **What it does**: Fills missing values with appropriate estimates
        **Why it matters**: Most ML algorithms cannot handle missing data
        **Strategies**:
        - **Numerical data**: Mean, median, or mode
        - **Categorical data**: Mode or "Unknown" category
        - **Advanced methods**: KNN imputation, iterative imputation
        
        ### Expected Outputs
        - Missing value statistics and visualizations
        - Data type conversion logs
        - Outlier detection results
        - Imputation summary
        - Data quality metrics
        
        ### Common Issues and Solutions
        
        #### Too Many Missing Values
        **Problem**: High percentage of missing data in columns
        **Solution**: Consider dropping columns or using advanced imputation
        ```python
        # Drop columns with >50% missing values
        threshold = 0.5
        df_cleaned = df.dropna(thresh=len(df) * threshold, axis=1)
        ```
        
        #### Outliers Affecting Analysis
        **Problem**: Extreme values skewing statistics
        **Solution**: Use robust statistics or transform data
        ```python
        # Use robust statistics
        median = df['column'].median()
        mad = np.median(np.abs(df['column'] - median))
        # Or apply log transformation
        df['column_log'] = np.log1p(df['column'])
        ```
        
        #### Categorical Data Issues
        **Problem**: Inconsistent categorical values
        **Solution**: Standardize and clean categories
        ```python
        # Standardize categorical data
        df['category'] = df['category'].str.strip().str.lower()
        # Remove duplicates
        df['category'] = df['category'].replace({'duplicate1': 'standard', 'duplicate2': 'standard'})
        ```
        
        ### Best Practices
        1. **Always analyze before imputing** - Understand why data is missing
        2. **Use appropriate imputation methods** - Match method to data type
        3. **Document all cleaning steps** - For reproducibility
        4. **Validate cleaning results** - Check data quality after cleaning
        5. **Consider domain knowledge** - Some "outliers" might be valid
        """
    
    def _generate_feature_engineering_explanations(self, state: ClassificationState) -> str:
        """Generate markdown explanations for feature engineering steps"""
        return """
        ## 3. Feature Engineering: Creating Powerful Predictors
        
        ### Purpose
        Feature engineering transforms raw data into meaningful features that improve model performance. This step often has the biggest impact on model quality.
        
        ### Key Concepts
        
        #### Categorical Variable Encoding
        ```python
        def engineer_features(df, target_column):
            categorical_columns = df.select_dtypes(include=['object']).columns
            categorical_columns = [col for col in categorical_columns if col != target_column]
            
            if len(categorical_columns) > 0:
                df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        ```
        **What it does**: Converts categorical data into numerical format
        **Why it matters**: Most ML algorithms require numerical input
        **Methods**:
        - **One-Hot Encoding**: Creates binary columns for each category
        - **Label Encoding**: Assigns numbers to categories
        - **Target Encoding**: Uses target variable statistics
        
        #### Feature Scaling
        ```python
        def scale_features(df, target_column):
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != target_column]
            
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        ```
        **What it does**: Normalizes features to similar scales
        **Why it matters**: Prevents features with larger scales from dominating
        **Methods**:
        - **StandardScaler**: Mean=0, Std=1 (sensitive to outliers)
        - **MinMaxScaler**: Range [0,1] (preserves distribution shape)
        - **RobustScaler**: Uses median and IQR (robust to outliers)
        
        #### Feature Creation
        ```python
        # Domain-specific features
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])
        
        # Statistical features
        df['income_per_person'] = df['household_income'] / df['household_size']
        
        # Interaction features
        df['income_age_interaction'] = df['income'] * df['age']
        ```
        **What it does**: Creates new features from existing ones
        **Why it matters**: New features can capture patterns not visible in raw data
        **Types**:
        - **Domain features**: Based on business knowledge
        - **Statistical features**: Mathematical transformations
        - **Interaction features**: Combinations of existing features
        
        ### Expected Outputs
        - Encoded categorical variables
        - Scaled numerical features
        - New feature creation logs
        - Feature matrix shape
        - Feature importance rankings
        
        ### Common Issues and Solutions
        
        #### High-Dimensional Data
        **Problem**: Too many features after one-hot encoding
        **Solution**: Use feature selection or dimensionality reduction
        ```python
        # Feature selection
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=10)
        X_selected = selector.fit_transform(X, y)
        ```
        
        #### Categorical Variables with Many Levels
        **Problem**: Some categories have very few observations
        **Solution**: Group rare categories or use target encoding
        ```python
        # Group rare categories
        category_counts = df['category'].value_counts()
        rare_categories = category_counts[category_counts < 10].index
        df['category'] = df['category'].replace(rare_categories, 'Other')
        ```
        
        #### Scaling Issues
        **Problem**: Some features have extreme values affecting scaling
        **Solution**: Use robust scaling or handle outliers first
        ```python
        # Use robust scaling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        ```
        
        ### Best Practices
        1. **Start with domain knowledge** - Create features based on business understanding
        2. **Handle categorical data properly** - Choose appropriate encoding method
        3. **Scale features when needed** - For distance-based algorithms
        4. **Avoid data leakage** - Don't use future information
        5. **Validate feature quality** - Check for correlation and importance
        """
    
    def _generate_model_training_explanations(self, state: ClassificationState) -> str:
        """Generate markdown explanations for model training steps"""
        return """
        ## 4. Model Training: Teaching Algorithms to Predict
        
        ### Purpose
        Model training teaches machine learning algorithms to make predictions by learning patterns from historical data. This is where the "magic" of ML happens.
        
        ### Key Concepts
        
        #### Train-Test Split
        ```python
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        ```
        **What it does**: Separates data into training and testing sets
        **Why it matters**: Prevents data leakage and provides unbiased evaluation
        **Parameters**:
        - **test_size**: Proportion of data for testing (typically 0.2-0.3)
        - **random_state**: Ensures reproducible splits
        - **stratify**: Maintains class distribution in both sets
        
        #### Multiple Algorithm Comparison
        ```python
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            # ... more models
        }
        ```
        **What it does**: Tests multiple algorithms on the same data
        **Why it matters**: Different algorithms work better for different problems
        **Common algorithms**:
        - **Linear**: Logistic Regression, SVM
        - **Tree-based**: Random Forest, XGBoost, LightGBM
        - **Ensemble**: Gradient Boosting, Voting Classifiers
        
        #### Cross-Validation
        ```python
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        ```
        **What it does**: Tests model performance on multiple data subsets
        **Why it matters**: Provides robust performance estimates
        **Benefits**:
        - More reliable performance estimates
        - Uses all data for both training and testing
        - Reduces variance in performance estimates
        
        #### Hyperparameter Tuning
        ```python
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        ```
        **What it does**: Finds optimal parameter settings for the model
        **Why it matters**: Proper tuning can significantly improve performance
        **Methods**:
        - **Grid Search**: Tests all parameter combinations
        - **Random Search**: Tests random combinations
        - **Bayesian Optimization**: Uses previous results to guide search
        
        ### Expected Outputs
        - Training and test accuracy for each model
        - Cross-validation scores
        - Best hyperparameters
        - Model comparison results
        - Feature importance rankings
        
        ### Common Issues and Solutions
        
        #### Overfitting
        **Problem**: High training accuracy, low test accuracy
        **Solution**: Use regularization or simpler models
        ```python
        # Add regularization
        model = LogisticRegression(C=0.1, penalty='l2')
        # Or use simpler model
        model = DecisionTreeClassifier(max_depth=5)
        ```
        
        #### Underfitting
        **Problem**: Low training and test accuracy
        **Solution**: Increase model complexity or add features
        ```python
        # Increase model complexity
        model = RandomForestClassifier(n_estimators=200, max_depth=20)
        # Or add more features
        X_extended = add_polynomial_features(X)
        ```
        
        #### Class Imbalance
        **Problem**: Model biased toward majority class
        **Solution**: Use class weights or resampling
        ```python
        # Use class weights
        model = RandomForestClassifier(class_weight='balanced')
        # Or resample data
        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        ```
        
        ### Best Practices
        1. **Start simple** - Begin with basic algorithms
        2. **Use cross-validation** - Get robust performance estimates
        3. **Compare fairly** - Use same data splits and preprocessing
        4. **Tune hyperparameters** - But don't over-optimize
        5. **Document everything** - Keep track of all experiments
        """
    
    def _generate_model_evaluation_explanations(self, state: ClassificationState) -> str:
        """Generate markdown explanations for model evaluation steps"""
        return """
        ## 5. Model Evaluation: Measuring Performance
        
        ### Purpose
        Model evaluation measures how well your trained model performs on unseen data. This step determines if your model is ready for deployment.
        
        ### Key Concepts
        
        #### Performance Metrics
        ```python
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        ```
        **What it measures**: Different aspects of model performance
        **Why it matters**: Single metric doesn't tell the whole story
        **Key metrics**:
        - **Accuracy**: Overall correctness (can be misleading with imbalanced data)
        - **Precision**: Of positive predictions, how many were correct?
        - **Recall**: Of actual positives, how many did we catch?
        - **F1-Score**: Harmonic mean of precision and recall
        
        #### Confusion Matrix
        ```python
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        ```
        **What it shows**: Detailed breakdown of correct and incorrect predictions
        **Why it's useful**: Identifies specific types of errors
        **Components**:
        - **True Positives (TP)**: Correctly predicted positive
        - **True Negatives (TN)**: Correctly predicted negative
        - **False Positives (FP)**: Incorrectly predicted positive
        - **False Negatives (FN)**: Incorrectly predicted negative
        
        #### ROC Curve
        ```python
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        ```
        **What it shows**: Trade-off between true positive rate and false positive rate
        **Why it's useful**: Compares models across different thresholds
        **Interpretation**:
        - **Perfect classifier**: Curve goes to top-left corner (AUC = 1.0)
        - **Random classifier**: Diagonal line (AUC = 0.5)
        - **Good classifier**: Curve above diagonal (AUC > 0.7)
        
        #### Feature Importance
        ```python
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        ```
        **What it shows**: Which features are most important for predictions
        **Why it's useful**: Understands what drives model decisions
        **Methods**:
        - **Tree-based models**: Built-in feature importance
        - **Permutation importance**: Measures performance drop when feature is shuffled
        - **SHAP values**: Explains individual predictions
        
        ### Expected Outputs
        - Performance metrics (accuracy, precision, recall, F1)
        - Confusion matrix visualization
        - ROC curve plot
        - Feature importance rankings
        - Classification report
        
        ### Common Issues and Solutions
        
        #### Misleading Accuracy
        **Problem**: High accuracy but poor performance on minority class
        **Solution**: Use additional metrics and check class distribution
        ```python
        # Check class distribution
        print(y_test.value_counts())
        # Use balanced metrics
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        ```
        
        #### Poor ROC Performance
        **Problem**: Low AUC score
        **Solution**: Check if model is learning or try different algorithms
        ```python
        # Check if model is learning
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Train: {train_score}, Test: {test_score}")
        ```
        
        #### Feature Importance Issues
        **Problem**: Unexpected feature importance rankings
        **Solution**: Check for data leakage and feature correlations
        ```python
        # Check for correlations
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=True)
        ```
        
        ### Best Practices
        1. **Use multiple metrics** - Don't rely on accuracy alone
        2. **Visualize results** - Plots reveal patterns numbers miss
        3. **Check class balance** - Imbalanced data needs special attention
        4. **Validate on holdout set** - Final test on completely unseen data
        5. **Document all results** - Keep track of performance across experiments
        """
    
    def _generate_model_persistence_explanations(self, state: ClassificationState) -> str:
        """Generate markdown explanations for model persistence steps"""
        return """
        ## 6. Model Persistence: Saving for Future Use
        
        ### Purpose
        Model persistence saves your trained model and preprocessing pipeline so you can use them later for predictions on new data.
        
        ### Key Concepts
        
        #### Model Packaging
        ```python
        model_package = {
            'model': trained_model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'target_column': 'target',
            'model_name': 'RandomForestClassifier',
            'training_date': datetime.now().isoformat(),
            'performance_metrics': evaluation_results
        }
        ```
        **What it does**: Bundles model with all necessary components
        **Why it matters**: Ensures model can be used consistently
        **Components**:
        - **Model**: The trained algorithm
        - **Preprocessing**: Scaler, encoder, imputer
        - **Metadata**: Feature names, training date, performance
        - **Configuration**: Model parameters and settings
        
        #### Saving Models
        ```python
        import joblib
        model_filename = f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model_package, model_filename)
        ```
        **What it does**: Serializes model to disk
        **Why it matters**: Allows model reuse without retraining
        **Formats**:
        - **joblib**: Fast, efficient for scikit-learn models
        - **pickle**: General Python serialization
        - **ONNX**: Cross-platform model format
        
        #### Loading Models
        ```python
        def load_and_predict(model_path, new_data):
            model_package = joblib.load(model_path)
            model = model_package['model']
            scaler = model_package['scaler']
            
            # Preprocess new data
            new_data_scaled = scaler.transform(new_data)
            
            # Make predictions
            predictions = model.predict(new_data_scaled)
            return predictions
        ```
        **What it does**: Loads saved model and makes predictions
        **Why it matters**: Enables model deployment and reuse
        **Steps**:
        1. Load model package
        2. Extract components (model, scaler, etc.)
        3. Preprocess new data
        4. Make predictions
        5. Return results
        
        ### Expected Outputs
        - Saved model files (.joblib or .pkl)
        - Preprocessing pipeline files
        - Model metadata and documentation
        - Example usage code
        - Performance benchmarks
        
        ### Common Issues and Solutions
        
        #### Version Compatibility
        **Problem**: Model saved with different library versions
        **Solution**: Document versions and use virtual environments
        ```python
        # Document versions
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
        
        # Save version info
        model_package['sklearn_version'] = sklearn.__version__
        ```
        
        #### Missing Preprocessing
        **Problem**: Model works in training but fails in production
        **Solution**: Always save and load preprocessing pipeline
        ```python
        # Save complete pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier())
        ])
        joblib.dump(pipeline, 'complete_pipeline.joblib')
        ```
        
        #### Large Model Files
        **Problem**: Model files are too large
        **Solution**: Use compression or model optimization
        ```python
        # Compress when saving
        joblib.dump(model, 'model.joblib', compress=3)
        
        # Or use model optimization
        from sklearn.tree import export_graphviz
        # Export smaller representation
        ```
        
        ### Best Practices
        1. **Save everything needed** - Model, preprocessing, metadata
        2. **Version your models** - Include timestamps and version numbers
        3. **Document dependencies** - List required library versions
        4. **Test loading** - Verify model works after saving
        5. **Keep backups** - Store multiple versions of important models
        """
    
    def _generate_data_science_overview(self) -> str:
        """Generate overview of data science and machine learning concepts"""
        return """
        ## 1. Introduction to Machine Learning Classification
        
        ### What is Machine Learning?
        Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. In classification problems, we teach the computer to categorize data into predefined classes or categories.
        
        ### Types of Machine Learning
        1. **Supervised Learning**: Learning with labeled data (what we're doing here)
        2. **Unsupervised Learning**: Finding patterns in unlabeled data
        3. **Reinforcement Learning**: Learning through interaction with an environment
        
        ### Classification vs Regression
        - **Classification**: Predicting discrete categories (e.g., spam/not spam, disease/no disease)
        - **Regression**: Predicting continuous values (e.g., house prices, temperature)
        
        ### The Machine Learning Pipeline
        1. **Data Collection**: Gathering relevant data
        2. **Data Cleaning**: Handling missing values, outliers, and inconsistencies
        3. **Exploratory Data Analysis (EDA)**: Understanding data patterns and relationships
        4. **Feature Engineering**: Creating and selecting relevant features
        5. **Model Training**: Teaching the algorithm using training data
        6. **Model Evaluation**: Testing performance on unseen data
        7. **Model Deployment**: Using the model for real-world predictions
        
        ### Why Classification is Important
        Classification algorithms are used in numerous real-world applications:
        - **Healthcare**: Disease diagnosis, drug discovery
        - **Finance**: Credit scoring, fraud detection
        - **Technology**: Email spam filtering, image recognition
        - **Business**: Customer segmentation, recommendation systems
        """
    
    def _generate_data_cleaning_education(self, state: ClassificationState) -> str:
        """Generate educational content about data cleaning"""
        return """
        ## 2. Data Cleaning: The Foundation of Good Models
        
        ### Why Data Cleaning Matters
        The quality of your data directly impacts the quality of your model. As the saying goes: "Garbage in, garbage out." Even the most sophisticated algorithms cannot overcome poor data quality.
        
        ### Common Data Quality Issues
        
        #### Missing Values
        **What they are**: Empty cells or null values in your dataset
        **Why they occur**: 
        - Data collection errors
        - Survey non-responses
        - System failures
        - Privacy concerns
        
        **Types of Missing Data**:
        1. **MCAR (Missing Completely At Random)**: Missingness is unrelated to any variable
        2. **MAR (Missing At Random)**: Missingness is related to observed variables
        3. **MNAR (Missing Not At Random)**: Missingness is related to the missing value itself
        
        **Handling Strategies**:
        - **Deletion**: Remove rows/columns with missing values (use carefully)
        - **Imputation**: Fill missing values with estimates
          - Mean/Median for numerical data
          - Mode for categorical data
          - Advanced methods like KNN imputation
        
        #### Outliers
        **What they are**: Data points that are significantly different from other observations
        **Why they matter**: Can skew model performance and lead to incorrect conclusions
        
        **Detection Methods**:
        1. **IQR Method**: Values outside Q1 - 1.5×IQR or Q3 + 1.5×IQR
        2. **Z-Score Method**: Values with |z-score| > 3
        3. **Isolation Forest**: Machine learning-based outlier detection
        
        **Handling Strategies**:
        - **Investigate**: Understand why outliers exist
        - **Remove**: If clearly erroneous
        - **Transform**: Apply log transformation or winsorization
        - **Keep**: If they represent valid extreme cases
        
        #### Data Type Issues
        **Common Problems**:
        - Numbers stored as text
        - Dates in wrong format
        - Inconsistent categorical values
        
        **Solutions**:
        - Convert data types appropriately
        - Standardize categorical values
        - Parse dates correctly
        
        ### Data Quality Metrics
        - **Completeness**: Percentage of non-missing values
        - **Consistency**: Uniformity across records
        - **Validity**: Data conforms to expected format/range
        - **Accuracy**: Data correctly represents real-world values
        """
    
    def _generate_feature_engineering_education(self) -> str:
        """Generate educational content about feature engineering"""
        return """
        ## 3. Feature Engineering: Creating Powerful Predictors
        
        ### What is Feature Engineering?
        Feature engineering is the process of creating, transforming, and selecting variables (features) that will be used to train your machine learning model. It's often said that feature engineering is more important than the choice of algorithm.
        
        ### Why Feature Engineering Matters
        - **Improves Model Performance**: Better features lead to better predictions
        - **Reduces Overfitting**: Relevant features help models generalize
        - **Handles Data Types**: Converts data into formats algorithms can understand
        - **Captures Domain Knowledge**: Incorporates expert knowledge into the model
        
        ### Common Feature Engineering Techniques
        
        #### Handling Categorical Variables
        **One-Hot Encoding**: Creates binary columns for each category
        ```python
        # Before: ['red', 'blue', 'green']
        # After: red_1, blue_0, green_0
        ```
        
        **Label Encoding**: Assigns numbers to categories
        ```python
        # Before: ['small', 'medium', 'large']
        # After: [0, 1, 2]
        ```
        
        **Target Encoding**: Uses target variable statistics for encoding
        
        #### Handling Numerical Variables
        **Scaling**: Normalizes features to similar ranges
        - **StandardScaler**: Mean=0, Std=1
        - **MinMaxScaler**: Range [0,1]
        - **RobustScaler**: Uses median and IQR
        
        **Transformation**: Changes data distribution
        - **Log transformation**: Handles skewed data
        - **Box-Cox transformation**: General power transformation
        - **Polynomial features**: Creates interaction terms
        
        #### Creating New Features
        **Domain-Specific Features**:
        - Age groups from birth dates
        - Time-based features (hour, day, month)
        - Ratios and proportions
        
        **Statistical Features**:
        - Rolling averages
        - Percentile rankings
        - Z-scores
        
        **Interaction Features**:
        - Product of two features
        - Difference between features
        - Custom combinations
        
        ### Feature Selection
        **Why it matters**: 
        - Reduces overfitting
        - Improves model interpretability
        - Reduces training time
        - Removes irrelevant features
        
        **Methods**:
        1. **Univariate Selection**: Statistical tests (chi-square, ANOVA)
        2. **Recursive Feature Elimination**: Iteratively removes least important features
        3. **Feature Importance**: Uses model's built-in importance scores
        4. **Correlation Analysis**: Removes highly correlated features
        """
    
    def _generate_model_selection_education(self, state: ClassificationState) -> str:
        """Generate educational content about model selection"""
        best_model = state.get('best_model', 'RandomForestClassifier')
        
        return f"""
        ## 4. Model Selection: Choosing the Right Algorithm
        
        ### The Model Selection Process
        Model selection involves comparing different algorithms to find the one that best fits your data and problem requirements. There's no "one-size-fits-all" solution in machine learning.
        
        ### Common Classification Algorithms
        
        #### Linear Models
        **Logistic Regression**:
        - **How it works**: Uses logistic function to model probability
        - **Pros**: Interpretable, fast, works well with small datasets
        - **Cons**: Assumes linear relationships, sensitive to outliers
        - **Best for**: Binary classification, when interpretability is important
        
        **Support Vector Machine (SVM)**:
        - **How it works**: Finds optimal boundary between classes
        - **Pros**: Works well with high-dimensional data, memory efficient
        - **Cons**: Slow on large datasets, sensitive to feature scaling
        - **Best for**: Text classification, image recognition
        
        #### Tree-Based Models
        **Decision Tree**:
        - **How it works**: Creates a tree of decisions based on feature values
        - **Pros**: Highly interpretable, handles non-linear relationships
        - **Cons**: Prone to overfitting, unstable
        - **Best for**: When interpretability is crucial
        
        **Random Forest**:
        - **How it works**: Combines multiple decision trees
        - **Pros**: Reduces overfitting, handles missing values, feature importance
        - **Cons**: Less interpretable than single tree, can be slow
        - **Best for**: General-purpose classification, when you need feature importance
        
        **Gradient Boosting**:
        - **How it works**: Sequentially builds models to correct previous errors
        - **Pros**: Often highest accuracy, handles various data types
        - **Cons**: Can overfit, requires careful tuning, slower training
        - **Best for**: When maximum accuracy is needed
        
        #### Advanced Models
        **XGBoost**:
        - **How it works**: Optimized gradient boosting with regularization
        - **Pros**: Very fast, high accuracy, built-in regularization
        - **Cons**: Many hyperparameters to tune, can overfit
        - **Best for**: Kaggle competitions, when speed and accuracy matter
        
        **LightGBM**:
        - **How it works**: Gradient boosting with leaf-wise tree growth
        - **Pros**: Very fast, memory efficient, good accuracy
        - **Cons**: Can overfit on small datasets
        - **Best for**: Large datasets, when speed is important
        
        ### How to Choose the Right Model
        
        #### Consider Your Data
        - **Dataset size**: Some algorithms work better with large datasets
        - **Feature count**: High-dimensional data may need different approaches
        - **Data quality**: Some algorithms are more robust to noise
        
        #### Consider Your Requirements
        - **Interpretability**: Do you need to understand how decisions are made?
        - **Speed**: How fast do you need predictions?
        - **Accuracy**: What level of accuracy do you need?
        
        #### Consider Your Constraints
        - **Computational resources**: Some models require more memory/CPU
        - **Deployment environment**: Some models are easier to deploy
        - **Maintenance**: How often will you retrain the model?
        
        ### Model Comparison Strategy
        1. **Start Simple**: Begin with basic algorithms
        2. **Compare Fairly**: Use same train/test split and preprocessing
        3. **Use Cross-Validation**: Get robust performance estimates
        4. **Consider Multiple Metrics**: Don't just look at accuracy
        5. **Test on Unseen Data**: Final validation on holdout set
        
        ### Hyperparameter Tuning
        **What are hyperparameters?**: Settings that control how the algorithm learns
        
        **Common Tuning Methods**:
        - **Grid Search**: Tests all combinations of parameter values
        - **Random Search**: Tests random combinations
        - **Bayesian Optimization**: Uses previous results to guide search
        
        **Important Hyperparameters**:
        - **Learning Rate**: How fast the model learns
        - **Regularization**: Controls overfitting
        - **Tree Depth**: Complexity of decision trees
        - **Number of Estimators**: How many models to combine
        """
    
    def _generate_evaluation_education(self) -> str:
        """Generate educational content about model evaluation"""
        return """
        ## 5. Model Evaluation: Measuring Performance
        
        ### Why Evaluation Matters
        Proper evaluation is crucial to understand how well your model will perform on new, unseen data. Without proper evaluation, you might deploy a model that performs poorly in real-world scenarios.
        
        ### The Train-Validation-Test Split
        **Training Set (60-70%)**: Used to train the model
        **Validation Set (15-20%)**: Used to tune hyperparameters and select models
        **Test Set (15-20%)**: Used for final, unbiased evaluation
        
        **Why this matters**: Prevents overfitting and gives honest performance estimates
        
        ### Cross-Validation
        **What it is**: Technique to get robust performance estimates by training on multiple subsets of data
        
        **K-Fold Cross-Validation**:
        1. Split data into k equal parts (folds)
        2. Train on k-1 folds, test on remaining fold
        3. Repeat k times
        4. Average the results
        
        **Benefits**:
        - More reliable performance estimates
        - Uses all data for both training and testing
        - Reduces variance in performance estimates
        
        ### Classification Metrics
        
        #### Accuracy
        **Formula**: (Correct Predictions) / (Total Predictions)
        **When to use**: When classes are balanced
        **Limitations**: Misleading with imbalanced classes
        
        #### Precision
        **Formula**: True Positives / (True Positives + False Positives)
        **Interpretation**: Of all positive predictions, how many were correct?
        **When to use**: When false positives are costly (e.g., spam detection)
        
        #### Recall (Sensitivity)
        **Formula**: True Positives / (True Positives + False Negatives)
        **Interpretation**: Of all actual positives, how many did we catch?
        **When to use**: When false negatives are costly (e.g., disease diagnosis)
        
        #### F1-Score
        **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
        **Interpretation**: Harmonic mean of precision and recall
        **When to use**: When you need balance between precision and recall
        
        #### ROC-AUC
        **What it measures**: Area under the ROC curve
        **Range**: 0 to 1 (higher is better)
        **Interpretation**: Probability that model ranks positive higher than negative
        **When to use**: When you need to compare models across different thresholds
        
        ### Confusion Matrix
        **What it shows**: Detailed breakdown of correct and incorrect predictions
        
        ```
                    Predicted
                  0    1
        Actual 0  TN   FP
               1  FN   TP
        ```
        
        **Key Terms**:
        - **True Positive (TP)**: Correctly predicted positive
        - **True Negative (TN)**: Correctly predicted negative
        - **False Positive (FP)**: Incorrectly predicted positive (Type I error)
        - **False Negative (FN)**: Incorrectly predicted negative (Type II error)
        
        ### ROC Curve
        **What it shows**: Trade-off between True Positive Rate and False Positive Rate
        **X-axis**: False Positive Rate (1 - Specificity)
        **Y-axis**: True Positive Rate (Sensitivity)
        **Perfect classifier**: Curve goes to top-left corner
        **Random classifier**: Diagonal line
        
        ### Precision-Recall Curve
        **When to use**: Better than ROC for imbalanced datasets
        **X-axis**: Recall
        **Y-axis**: Precision
        **Shows**: Trade-off between precision and recall at different thresholds
        
        ### Feature Importance
        **What it tells us**: Which features are most important for predictions
        **Methods**:
        - **Tree-based models**: Built-in feature importance
        - **Permutation importance**: Measures performance drop when feature is shuffled
        - **SHAP values**: Explains individual predictions
        
        ### Model Interpretability
        **Why it matters**: Understanding how models make decisions builds trust
        
        **Methods**:
        - **Feature importance**: Which features matter most
        - **Partial dependence plots**: How features affect predictions
        - **LIME**: Local explanations for individual predictions
        - **SHAP**: Unified framework for model explanations
        """
    
    def _generate_best_practices(self) -> str:
        """Generate educational content about ML best practices"""
        return """
        ## 6. Machine Learning Best Practices
        
        ### Data Preparation Best Practices
        
        #### Always Split Data First
        **Rule**: Split your data into train/validation/test sets BEFORE any preprocessing
        **Why**: Prevents data leakage and gives honest performance estimates
        **Implementation**: Use `train_test_split` at the very beginning
        
        #### Handle Missing Values Thoughtfully
        **Don't**: Always use mean imputation
        **Do**: 
        - Understand why data is missing
        - Choose appropriate imputation strategy
        - Consider if missingness is informative
        
        #### Scale Features Appropriately
        **When to scale**: Distance-based algorithms (SVM, KNN) and neural networks
        **When not to scale**: Tree-based algorithms (Random Forest, XGBoost)
        **Method**: Use the same scaler on training and test data
        
        ### Model Training Best Practices
        
        #### Start Simple
        **Approach**: Begin with simple models (logistic regression, decision tree)
        **Why**: 
        - Establishes baseline performance
        - Helps understand data patterns
        - Identifies obvious issues early
        
        #### Use Cross-Validation
        **Implementation**: Always use k-fold cross-validation for model selection
        **Benefits**: More robust performance estimates, better hyperparameter tuning
        
        #### Regularize to Prevent Overfitting
        **Techniques**:
        - L1/L2 regularization for linear models
        - Early stopping for gradient boosting
        - Dropout for neural networks
        - Pruning for decision trees
        
        ### Evaluation Best Practices
        
        #### Choose Appropriate Metrics
        **For balanced datasets**: Accuracy, F1-score
        **For imbalanced datasets**: Precision, Recall, ROC-AUC, PR-AUC
        **For business context**: Custom metrics that align with business goals
        
        #### Validate on Multiple Datasets
        **Approach**: Test on different time periods, demographics, or data sources
        **Why**: Ensures model generalizes beyond training data
        
        #### Monitor for Data Drift
        **What to monitor**: Feature distributions, model performance over time
        **Why**: Data patterns change, models need updating
        
        ### Deployment Best Practices
        
        #### Version Control Everything
        **Include**: Code, data, models, hyperparameters, results
        **Tools**: Git, MLflow, DVC
        **Why**: Reproducibility and rollback capability
        
        #### Test Thoroughly
        **Unit tests**: Test individual functions
        **Integration tests**: Test entire pipeline
        **A/B tests**: Compare model versions in production
        
        #### Monitor Performance
        **Metrics to track**: Accuracy, latency, throughput, error rates
        **Alerts**: Set up alerts for performance degradation
        **Retraining**: Schedule regular model updates
        
        ### Common Mistakes to Avoid
        
        #### Data Leakage
        **What it is**: Using future information to predict the past
        **Examples**: Using target variable to create features, scaling before splitting
        **Prevention**: Always split data first, be careful with time series data
        
        #### Overfitting
        **Signs**: High training accuracy, low test accuracy
        **Prevention**: Use validation set, regularize, simplify model
        
        #### Underfitting
        **Signs**: Low training and test accuracy
        **Prevention**: Increase model complexity, add features, reduce regularization
        
        #### Ignoring Class Imbalance
        **Problem**: Model learns to predict majority class
        **Solutions**: Resampling, cost-sensitive learning, different metrics
        
        #### Not Understanding the Business Context
        **Problem**: Optimizing wrong metrics
        **Solution**: Align technical metrics with business goals
        """
    
    def _generate_common_pitfalls(self) -> str:
        """Generate educational content about common ML pitfalls"""
        return """
        ## 7. Common Pitfalls and How to Avoid Them
        
        ### Data-Related Pitfalls
        
        #### Data Leakage
        **What it is**: Information from the future leaking into training data
        **Examples**:
        - Using target variable to create features
        - Scaling before train/test split
        - Using test data for feature selection
        
        **How to avoid**:
        - Always split data first
        - Fit preprocessing on training data only
        - Be extra careful with time series data
        
        #### Survivorship Bias
        **What it is**: Only analyzing successful cases
        **Example**: Only studying companies that survived bankruptcy
        **How to avoid**: Include all relevant data, not just successful cases
        
        #### Selection Bias
        **What it is**: Data not representative of the population
        **Example**: Survey data from only one demographic
        **How to avoid**: Ensure diverse, representative data collection
        
        ### Model-Related Pitfalls
        
        #### Overfitting
        **What it is**: Model memorizes training data instead of learning patterns
        **Signs**: High training accuracy, low test accuracy
        **How to avoid**:
        - Use validation set
        - Apply regularization
        - Simplify model
        - Get more data
        
        #### Underfitting
        **What it is**: Model too simple to capture data patterns
        **Signs**: Low training and test accuracy
        **How to avoid**:
        - Increase model complexity
        - Add more features
        - Reduce regularization
        - Check for data quality issues
        
        #### Catastrophic Forgetting
        **What it is**: Model forgets old patterns when learning new ones
        **How to avoid**: Use techniques like elastic weight consolidation
        
        ### Evaluation Pitfalls
        
        #### Optimizing Wrong Metrics
        **Problem**: Focusing on accuracy when precision matters more
        **Solution**: Choose metrics that align with business goals
        
        #### Data Snooping
        **What it is**: Making decisions based on test set performance
        **How to avoid**: Use separate validation set for model selection
        
        #### Cherry-Picking Results
        **Problem**: Only reporting best results
        **Solution**: Report all experiments, including failures
        
        ### Deployment Pitfalls
        
        #### Training-Serving Skew
        **What it is**: Differences between training and production environments
        **Examples**: Different data preprocessing, feature engineering
        **How to avoid**: Use same preprocessing pipeline in production
        
        #### Model Decay
        **What it is**: Model performance degrades over time
        **Causes**: Data drift, concept drift, changing user behavior
        **How to avoid**: Monitor performance, retrain regularly
        
        #### Ignoring Model Interpretability
        **Problem**: Black box models in regulated industries
        **Solution**: Use interpretable models or explainability tools
        
        ### Business Pitfalls
        
        #### Not Understanding the Problem
        **Problem**: Building the wrong solution
        **Solution**: Spend time understanding business requirements
        
        #### Ignoring Ethical Considerations
        **Issues**: Bias, fairness, privacy
        **Solution**: Implement bias testing, fairness constraints
        
        #### Overpromising Performance
        **Problem**: Setting unrealistic expectations
        **Solution**: Be honest about limitations and uncertainties
        
        ### How to Avoid These Pitfalls
        
        #### Follow a Systematic Process
        1. **Define the problem clearly**
        2. **Collect and explore data thoroughly**
        3. **Split data properly**
        4. **Start with simple models**
        5. **Validate thoroughly**
        6. **Monitor in production**
        
        #### Use Best Practices
        - Version control everything
        - Document all decisions
        - Test extensively
        - Monitor continuously
        
        #### Learn from Failures
        - Analyze what went wrong
        - Update processes
        - Share learnings with team
        
        #### Stay Updated
        - Follow ML research
        - Attend conferences
        - Join communities
        - Practice regularly
        """
    
    def _extract_dataset_info(self, state: ClassificationState) -> str:
        """Extract dataset information"""
        try:
            shape = state.get("dataset_shape", (0, 0))
            target = state.get("target_column", "unknown")
            description = state.get("user_description", "No description provided")
            
            return f"""
            - **Dataset Shape**: {shape[0]} rows × {shape[1]} columns
            - **Target Variable**: {target}
            - **User Description**: {description}
            - **Data Types**: Mixed (numeric and categorical)
            - **Missing Values**: Handled through intelligent imputation
            """
            
        except Exception as e:
            return f"Dataset information extraction failed: {str(e)}"
    
    def _extract_cleaning_info(self, state: ClassificationState) -> str:
        """Extract data cleaning information"""
        try:
            quality_score = state.get("data_quality_score", 0.0)
            actions = state.get("cleaning_actions_taken", [])
            issues = state.get("cleaning_issues_found", [])
            
            return f"""
            - **Data Quality Score**: {quality_score:.1%}
            - **Actions Taken**: {len(actions)} cleaning operations
            - **Issues Identified**: {len(issues)} data quality issues
            - **Cleaning Summary**: {state.get('cleaning_summary', 'No summary available')}
            """
            
        except Exception as e:
            return f"Cleaning information extraction failed: {str(e)}"
    
    def _extract_model_info(self, state: ClassificationState) -> str:
        """Extract model information"""
        try:
            model_name = state.get("best_model", "unknown")
            params = state.get("model_hyperparameters", {})
            metrics = state.get("training_metrics", {})
            
            return f"""
            - **Selected Model**: {model_name}
            - **Best Parameters**: {json.dumps(params, indent=2)}
            - **Training Accuracy**: {metrics.get('train_accuracy', 0):.4f}
            - **Test Accuracy**: {metrics.get('test_accuracy', 0):.4f}
            - **Cross-Validation Score**: {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}
            """
            
        except Exception as e:
            return f"Model information extraction failed: {str(e)}"
    
    def _extract_evaluation_info(self, state: ClassificationState) -> str:
        """Extract evaluation information"""
        try:
            metrics = state.get("evaluation_metrics", {})
            confusion_mat = state.get("confusion_matrix", [])
            
            return f"""
            - **Accuracy**: {metrics.get('accuracy', 0):.4f}
            - **Precision (Weighted)**: {metrics.get('precision_weighted', 0):.4f}
            - **Recall (Weighted)**: {metrics.get('recall_weighted', 0):.4f}
            - **F1-Score (Weighted)**: {metrics.get('f1_weighted', 0):.4f}
            - **Cohen's Kappa**: {metrics.get('cohen_kappa', 0):.4f}
            - **Confusion Matrix**: {len(confusion_mat)}×{len(confusion_mat[0]) if confusion_mat else 0} matrix
            """
            
        except Exception as e:
            return f"Evaluation information extraction failed: {str(e)}"
    
    def _generate_jupyter_notebook(self, state: ClassificationState) -> str:
        """Generate comprehensive Jupyter notebook with code from all agents"""
        try:
            # Create new notebook
            nb = new_notebook()
            
            # Add title and metadata
            nb.cells.append(new_markdown_cell("# Machine Learning Classification Project\n\n*Generated by Classify AI - Multi-Agent System*"))
            
            # Add project information
            nb.cells.append(new_markdown_cell("## Project Information"))
            project_info = f"""
**Project Description**: {state.get('user_description', 'No description provided')}
**Target Variable**: {state.get('target_column', 'target')}
**Dataset Shape**: {state.get('dataset_shape', (0, 0))[0]} rows × {state.get('dataset_shape', (0, 0))[1]} columns
**Generated on**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Session ID**: {state.get('session_id', 'unknown')}
"""
            nb.cells.append(new_markdown_cell(project_info))
            
            # Add imports section
            nb.cells.append(new_markdown_cell("## Required Libraries and Imports"))
            imports_code = """
# Core data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Model persistence
import joblib
import pickle

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency

print("All required libraries imported successfully!")
"""
            nb.cells.append(new_code_cell(imports_code))
            
            # Add dataset loading section
            nb.cells.append(new_markdown_cell("## 1. Dataset Loading and Initial Exploration"))
            dataset_code = f"""
# Load the dataset
df = pd.read_csv('dataset.csv')

# Basic dataset information
print("Dataset Information:")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"Target column: '{state.get('target_column', 'target')}'")
print("\\nFirst few rows:")
print(df.head())

print("\\nDataset Info:")
print(df.info())

print("\\nBasic Statistics:")
print(df.describe())
"""
            nb.cells.append(new_code_cell(dataset_code))
            
            # Add comprehensive data cleaning section
            nb.cells.append(new_markdown_cell("## 2. Comprehensive Data Cleaning"))
            
            # Missing value analysis
            nb.cells.append(new_markdown_cell("### 2.1 Missing Value Analysis"))
            missing_analysis_code = """
# Comprehensive missing value analysis
def analyze_missing_values(df, target_column=None):
    \"\"\"Analyze missing values in the dataset\"\"\"
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_stats.index,
        'Missing_Count': missing_stats.values,
        'Missing_Percentage': missing_percent.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print("Missing Value Analysis:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Visualize missing values
    if missing_stats.sum() > 0:
        plt.figure(figsize=(12, 6))
        missing_df[missing_df['Missing_Count'] > 0].plot(x='Column', y='Missing_Percentage', kind='bar')
        plt.title('Missing Values by Column')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return missing_df

missing_analysis = analyze_missing_values(df, '{state.get('target_column', 'target')}')
"""
            nb.cells.append(new_code_cell(missing_analysis_code))
            
            # Data type validation
            nb.cells.append(new_markdown_cell("### 2.2 Data Type Validation and Conversion"))
            type_validation_code = """
# Data type validation and conversion
def validate_and_convert_types(df):
    \"\"\"Validate and convert data types appropriately\"\"\"
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        # Try to convert to numeric
        if df_cleaned[col].dtype == 'object':
            # Check if it's actually numeric
            numeric_converted = pd.to_numeric(df_cleaned[col], errors='coerce')
            if not numeric_converted.isna().all():
                df_cleaned[col] = numeric_converted
                print(f"Converted {{col}} to numeric")
        
        # Check for datetime columns
        if df_cleaned[col].dtype == 'object':
            try:
                pd.to_datetime(df_cleaned[col], errors='raise')
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                print(f"Converted {{col}} to datetime")
            except:
                pass
    
    print("\\nData types after conversion:")
    print(df_cleaned.dtypes)
    return df_cleaned

df_typed = validate_and_convert_types(df)
"""
            nb.cells.append(new_code_cell(type_validation_code))
            
            # Outlier detection
            nb.cells.append(new_markdown_cell("### 2.3 Outlier Detection"))
            outlier_detection_code = """
# Comprehensive outlier detection
def detect_outliers_iqr(df, columns=None):
    \"\"\"Detect outliers using IQR method\"\"\"
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outlier_info

def detect_outliers_zscore(df, columns=None, threshold=3):
    \"\"\"Detect outliers using Z-score method\"\"\"
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[z_scores > threshold]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100
        }
    
    return outlier_info

# Detect outliers
numeric_columns = df_typed.select_dtypes(include=[np.number]).columns
iqr_outliers = detect_outliers_iqr(df_typed, numeric_columns)
zscore_outliers = detect_outliers_zscore(df_typed, numeric_columns)

print("Outlier Detection Results:")
for col in numeric_columns:
    if iqr_outliers[col]['count'] > 0:
        print(f"{{col}}: {{iqr_outliers[col]['count']}} outliers ({{iqr_outliers[col]['percentage']:.2f}}%)")
"""
            nb.cells.append(new_code_cell(outlier_detection_code))
            
            # Missing value imputation
            nb.cells.append(new_markdown_cell("### 2.4 Missing Value Imputation"))
            imputation_code = """
# Advanced missing value imputation
def impute_missing_values(df, target_column=None):
    \"\"\"Impute missing values using appropriate strategies\"\"\"
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            if df_imputed[col].dtype in ['int64', 'float64']:
                # For numeric columns, use median for robustness
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                print(f"Imputed {{col}} with median: {{df_imputed[col].median()}}")
            else:
                # For categorical columns, use mode
                mode_value = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 'Unknown'
                df_imputed[col].fillna(mode_value, inplace=True)
                print(f"Imputed {{col}} with mode: {{mode_value}}")
    
    return df_imputed

df_imputed = impute_missing_values(df_typed, '{state.get('target_column', 'target')}')
print(f"\\nMissing values after imputation: {{df_imputed.isnull().sum().sum()}}")
"""
            nb.cells.append(new_code_cell(imputation_code))
            
            # Data quality assessment
            nb.cells.append(new_markdown_cell("### 2.5 Data Quality Assessment"))
            quality_assessment_code = """
# Data quality assessment
def assess_data_quality(df, target_column):
    \"\"\"Assess overall data quality\"\"\"
    quality_metrics = {}
    
    # Completeness
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_metrics['completeness'] = completeness
    
    # Consistency (check for duplicates)
    duplicates = df.duplicated().sum()
    consistency = (1 - duplicates / len(df)) * 100
    quality_metrics['consistency'] = consistency
    
    # Validity (check for reasonable values)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    validity_score = 100
    for col in numeric_cols:
        if col != target_column:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validity_score -= (inf_count / len(df)) * 100
    
    quality_metrics['validity'] = max(0, validity_score)
    
    # Overall quality score
    overall_quality = np.mean(list(quality_metrics.values()))
    quality_metrics['overall'] = overall_quality
    
    return quality_metrics

quality_metrics = assess_data_quality(df_imputed, '{state.get('target_column', 'target')}')
print("Data Quality Metrics:")
for metric, score in quality_metrics.items():
    print(f"{{metric.title()}}: {{score:.2f}}%")
"""
            nb.cells.append(new_code_cell(quality_assessment_code))
            
            # Feature engineering section
            nb.cells.append(new_markdown_cell("## 3. Feature Engineering and Selection"))
            feature_engineering_code = """
# Feature engineering and selection
def engineer_features(df, target_column):
    \"\"\"Engineer features for machine learning\"\"\"
    df_features = df.copy()

# Handle categorical variables
    categorical_columns = df_features.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != target_column]
    
    if len(categorical_columns) > 0:
        # One-hot encoding for categorical variables
        df_features = pd.get_dummies(df_features, columns=categorical_columns, drop_first=True)
        print(f"One-hot encoded {{len(categorical_columns)}} categorical columns")
    
    # Feature scaling
    numeric_columns = df_features.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != target_column]
    
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        df_features[numeric_columns] = scaler.fit_transform(df_features[numeric_columns])
        print(f"Scaled {{len(numeric_columns)}} numeric columns")
    
    return df_features

df_features = engineer_features(df_imputed, '{state.get('target_column', 'target')}')
print(f"\\nFeature matrix shape: {{df_features.shape}}")
"""
            nb.cells.append(new_code_cell(feature_engineering_code))
            
            # Model training section
            nb.cells.append(new_markdown_cell("## 4. Model Training and Selection"))
            model_training_code = f"""
# Prepare data for training
X = df_features.drop(columns=['{state.get('target_column', 'target')}'])
y = df_features['{state.get('target_column', 'target')}']

# Split data (CRITICAL: Split before any preprocessing to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {{X_train.shape}}")
print(f"Test set shape: {{X_test.shape}}")
print(f"Target distribution in training set: {{y_train.value_counts().to_dict()}}")

# Define multiple models for comparison
models = {{
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
    'SVM': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}}

# Train and evaluate models
model_results = {{}}
for name, model in models.items():
    print(f"\\nTraining {{name}}...")

# Train model
model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    model_results[name] = {{
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred_test': y_pred_test
    }}
    
    print(f"Train Accuracy: {{train_accuracy:.4f}}")
    print(f"Test Accuracy: {{test_accuracy:.4f}}")
    print(f"CV Score: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

# Find best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_accuracy'])
best_model = model_results[best_model_name]['model']
print(f"\\nBest Model: {{best_model_name}}")
print(f"Best Test Accuracy: {{model_results[best_model_name]['test_accuracy']:.4f}}")
"""
            nb.cells.append(new_code_cell(model_training_code))
            
            # Hyperparameter tuning
            nb.cells.append(new_markdown_cell("### 4.1 Hyperparameter Tuning"))
            hyperparameter_tuning_code = f"""
# Hyperparameter tuning for the best model
def tune_hyperparameters(model, X_train, y_train, model_name):
    \"\"\"Tune hyperparameters using GridSearchCV\"\"\"
    
    if 'Random Forest' in model_name:
        param_grid = {{
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }}
    elif 'Logistic Regression' in model_name:
        param_grid = {{
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }}
    elif 'XGBoost' in model_name:
        param_grid = {{
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }}
    else:
        # Default parameter grid
        param_grid = {{}}
    
    if param_grid:
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {{model_name}}:")
        print(grid_search.best_params_)
        print(f"Best CV score: {{grid_search.best_score_:.4f}}")
        
        return grid_search.best_estimator_
    else:
        return model

# Tune hyperparameters for the best model
tuned_model = tune_hyperparameters(best_model, X_train, y_train, best_model_name)
"""
            nb.cells.append(new_code_cell(hyperparameter_tuning_code))
            
            # Model evaluation section
            nb.cells.append(new_markdown_cell("## 5. Comprehensive Model Evaluation"))
            model_evaluation_code = """
# Comprehensive model evaluation
def evaluate_model(model, X_test, y_test, model_name):
    \"\"\"Evaluate model performance comprehensively\"\"\"

# Make predictions
y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Classification report
    print(f"\\n=== {{model_name}} Evaluation Results ===")
    print(f"Accuracy: {{accuracy:.4f}}")
    print(f"Precision: {{precision:.4f}}")
    print(f"Recall: {{recall:.4f}}")
    print(f"F1-Score: {{f1:.4f}}")
    
    print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {{model_name}}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve (if probabilities available)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {{roc_auc:.2f}})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {{model_name}}')
        plt.legend(loc="lower right")
        plt.show()
    
    return {{
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }}

# Evaluate the tuned model
evaluation_results = evaluate_model(tuned_model, X_test, y_test, f"Tuned {{best_model_name}}")
"""
            nb.cells.append(new_code_cell(model_evaluation_code))

# Feature importance
            nb.cells.append(new_markdown_cell("### 5.1 Feature Importance Analysis"))
            feature_importance_code = """
# Feature importance analysis
def analyze_feature_importance(model, X_train, feature_names):
    \"\"\"Analyze and visualize feature importance\"\"\"
    
if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance_df = pd.DataFrame({{
            'feature': feature_names,
        'importance': model.feature_importances_
        }}).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    elif hasattr(model, 'coef_'):
        # Linear models
        coef_df = pd.DataFrame({{
            'feature': feature_names,
            'coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        }}).sort_values('coefficient', key=abs, ascending=False)
        
        print("Top 10 Most Important Features (by coefficient magnitude):")
        print(coef_df.head(10))
        
        # Plot coefficients
        plt.figure(figsize=(10, 8))
        top_features = coef_df.head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Top 15 Most Important Features (Coefficients)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return coef_df
    
    else:
        print("Feature importance not available for this model type")
        return None

# Analyze feature importance
feature_importance = analyze_feature_importance(tuned_model, X_train, X.columns)
"""
            nb.cells.append(new_code_cell(feature_importance_code))
            
            # Model persistence
            nb.cells.append(new_markdown_cell("## 6. Model Persistence and Deployment"))
            model_persistence_code = """
# Save the trained model and preprocessing objects
import joblib
from datetime import datetime

# Create a model package
model_package = {{
    'model': tuned_model,
    'scaler': scaler if 'scaler' in locals() else None,
    'feature_names': list(X.columns),
    'target_column': '{state.get('target_column', 'target')}',
    'model_name': best_model_name,
    'training_date': datetime.now().isoformat(),
    'performance_metrics': evaluation_results
}}

# Save model
model_filename = f"trained_model_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.joblib"
joblib.dump(model_package, model_filename)
print(f"Model saved as: {{model_filename}}")

# Save preprocessing pipeline
pipeline_filename = f"preprocessing_pipeline_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.joblib"
preprocessing_pipeline = {{
    'imputer': None,  # Add imputer if used
    'scaler': scaler if 'scaler' in locals() else None,
    'encoder': None,  # Add encoder if used
    'feature_names': list(X.columns)
}
joblib.dump(preprocessing_pipeline, pipeline_filename)
print(f"Preprocessing pipeline saved as: {{pipeline_filename}}")

# Example of how to load and use the model
def load_and_predict(model_path, new_data):
    \"\"\"Load model and make predictions on new data\"\"\"
    model_package = joblib.load(model_path)
    model = model_package['model']
    
    # Preprocess new data (same steps as training)
    # ... preprocessing code ...
    
    predictions = model.predict(new_data)
    return predictions

print("\\nModel persistence completed successfully!")
"""
            nb.cells.append(new_code_cell(model_persistence_code))
            
            # Usage instructions
            nb.cells.append(new_markdown_cell("## 7. Usage Instructions and Next Steps"))
            usage_instructions = """
# Usage Instructions

## How to Use This Model

### 1. Loading the Model
```python
import joblib
model_package = joblib.load('trained_model_YYYYMMDD_HHMMSS.joblib')
model = model_package['model']
```

### 2. Making Predictions
```python
# Load new data
new_data = pd.read_csv('new_data.csv')

# Apply same preprocessing steps
# ... (use the preprocessing pipeline from this notebook)

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)  # if available
```

### 3. Model Monitoring
- Monitor model performance over time
- Retrain when performance degrades
- Track prediction distributions

## Next Steps

1. **Deploy to Production**: Set up model serving infrastructure
2. **Monitor Performance**: Implement monitoring and alerting
3. **Continuous Learning**: Set up retraining pipeline
4. **A/B Testing**: Compare with baseline models
5. **Feature Engineering**: Explore additional features

## Model Performance Summary

- **Best Model**: {best_model_name}
- **Test Accuracy**: {evaluation_results['accuracy']:.4f}
- **F1-Score**: {evaluation_results['f1']:.4f}
- **Precision**: {evaluation_results['precision']:.4f}
- **Recall**: {evaluation_results['recall']:.4f}

## Important Notes

- Always apply the same preprocessing steps to new data
- Monitor for data drift and concept drift
- Retrain the model periodically with new data
- Document any changes to the preprocessing pipeline
"""
            nb.cells.append(new_markdown_cell(usage_instructions))
            
            # Save notebook
            session_id = state.get("session_id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            notebook_filename = f"classification_project_{session_id}_{timestamp}.ipynb"
            
            # Create notebooks directory
            notebooks_dir = "notebooks"
            os.makedirs(notebooks_dir, exist_ok=True)
            notebook_path = os.path.join(notebooks_dir, notebook_filename)
            
            # Write notebook
            with open(notebook_path, 'w') as f:
                nbf.write(nb, f)
            
            self.logger.info(f"Comprehensive Jupyter notebook saved to: {notebook_path}")
            return notebook_path
            
        except Exception as e:
            self.logger.error(f"Error generating Jupyter notebook: {str(e)}")
            return ""
    
    def _generate_recommendations(self, state: ClassificationState) -> List[str]:
        """Generate recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            accuracy = state.get("evaluation_metrics", {}).get("accuracy", 0.0)
            if accuracy > 0.9:
                recommendations.append("Model shows excellent performance and is ready for production deployment")
            elif accuracy > 0.8:
                recommendations.append("Model shows good performance with potential for further optimization")
            else:
                recommendations.append("Model performance could be improved with additional feature engineering or data collection")
            
            # Data quality recommendations
            quality_score = state.get("data_quality_score", 0.0)
            if quality_score < 0.8:
                recommendations.append("Consider improving data quality through better data collection processes")
            
            # Feature engineering recommendations
            feature_importance = state.get("feature_importance_model", {})
            if feature_importance:
                recommendations.append("Focus on the top-performing features for future model iterations")
            
            # General recommendations
            recommendations.extend([
                "Implement model monitoring in production to track performance drift",
                "Consider ensemble methods for improved robustness",
                "Regular retraining with new data to maintain model performance",
                "Document all preprocessing steps for reproducibility"
            ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Recommendations generation failed due to insufficient data"]
    
    def _generate_limitations(self, state: ClassificationState) -> List[str]:
        """Generate limitations analysis"""
        try:
            limitations = []
            
            # Dataset limitations
            shape = state.get("dataset_shape", (0, 0))
            if shape[0] < 1000:
                limitations.append("Small dataset size may limit model generalization")
            
            # Model limitations
            model_name = state.get("best_model", "unknown")
            if "tree" in model_name.lower():
                limitations.append("Tree-based models may overfit to training data")
            elif "linear" in model_name.lower():
                limitations.append("Linear models assume linear relationships between features")
            
            # General limitations
            limitations.extend([
                "Model performance is limited by the quality and quantity of training data",
                "Feature engineering decisions may introduce bias",
                "Model assumes data distribution remains stable over time",
                "Cross-validation may not fully represent real-world performance"
            ])
            
            return limitations
            
        except Exception as e:
            self.logger.error(f"Error generating limitations: {str(e)}")
            return ["Limitations analysis failed due to insufficient data"]
    
    def _generate_future_improvements(self, state: ClassificationState) -> List[str]:
        """Generate future improvements"""
        try:
            improvements = []
            
            # Data improvements
            improvements.append("Collect more diverse and representative training data")
            improvements.append("Implement automated data quality monitoring")
            
            # Model improvements
            improvements.append("Experiment with deep learning models for complex patterns")
            improvements.append("Implement ensemble methods combining multiple models")
            improvements.append("Add online learning capabilities for continuous improvement")
            
            # System improvements
            improvements.append("Implement A/B testing framework for model comparison")
            improvements.append("Add real-time model performance monitoring")
            improvements.append("Develop automated retraining pipeline")
            improvements.append("Create model explainability dashboard")
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"Error generating future improvements: {str(e)}")
            return ["Future improvements analysis failed due to insufficient data"]
    
    def _create_final_report(self, executive_summary: str, technical_documentation: str, 
                           educational_content: str, recommendations: List[str], 
                           limitations: List[str], future_improvements: List[str]) -> str:
        """Create final comprehensive report"""
        try:
            report = f"""
            {executive_summary}
            
            {technical_documentation}
            
            {educational_content}
            
            ## Recommendations
            {chr(10).join([f"- {rec}" for rec in recommendations])}
            
            ## Limitations
            {chr(10).join([f"- {lim}" for lim in limitations])}
            
            ## Future Improvements
            {chr(10).join([f"- {imp}" for imp in future_improvements])}
            
            ## Conclusion
            This machine learning classification project successfully demonstrates the application of automated data science techniques to solve real-world classification problems. The developed model and accompanying documentation provide a solid foundation for production deployment and future enhancements.
            
            Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            return report.strip()
            
        except Exception as e:
            self.logger.error(f"Error creating final report: {str(e)}")
            return "Final report generation failed due to insufficient data."
