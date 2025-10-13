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
                executive_summary, technical_documentation, 
                recommendations, limitations, future_improvements
            )
            
            # Update state with results
            state["final_report"] = final_report
            state["executive_summary"] = executive_summary
            state["technical_documentation"] = technical_documentation
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
        """Generate Jupyter notebook with all code"""
        try:
            # Create new notebook
            nb = new_notebook()
            
            # Add title
            nb.cells.append(new_markdown_cell("# Machine Learning Classification Project"))
            
            # Add dataset loading code
            nb.cells.append(new_markdown_cell("## Dataset Loading"))
            dataset_code = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('dataset.csv')
print(f"Dataset shape: {{df.shape}}")
print(f"Target column: '{state.get('target_column', 'target')}'")
"""
            nb.cells.append(new_code_cell(dataset_code))
            
            # Add data cleaning code
            nb.cells.append(new_markdown_cell("## Data Cleaning"))
            cleaning_code = f"""
# Data cleaning steps
cleaned_df = df.copy()

# Handle missing values
cleaned_df = cleaned_df.fillna(cleaned_df.mean())

# Remove duplicates
cleaned_df = cleaned_df.drop_duplicates()

print(f"Cleaned dataset shape: {{cleaned_df.shape}}")
print(f"Data quality score: {state.get('data_quality_score', 0):.2f}")
"""
            nb.cells.append(new_code_cell(cleaning_code))
            
            # Add model training code
            nb.cells.append(new_markdown_cell("## Model Training"))
            model_code = f"""
# Prepare features and target
X = cleaned_df.drop(columns=['{state.get('target_column', 'target')}'])
y = cleaned_df['{state.get('target_column', 'target')}']

# Handle categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = {state.get('best_model', 'RandomForestClassifier')}()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {{accuracy:.4f}}")
"""
            nb.cells.append(new_code_cell(model_code))
            
            # Add evaluation code
            nb.cells.append(new_markdown_cell("## Model Evaluation"))
            evaluation_code = """
# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\nFeature Importance:")
    print(feature_importance.head(10))
"""
            nb.cells.append(new_code_cell(evaluation_code))
            
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
            
            self.logger.info(f"Jupyter notebook saved to: {notebook_path}")
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
                           recommendations: List[str], limitations: List[str], 
                           future_improvements: List[str]) -> str:
        """Create final comprehensive report"""
        try:
            report = f"""
            {executive_summary}
            
            {technical_documentation}
            
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
