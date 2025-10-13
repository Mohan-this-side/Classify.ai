"""
Model Evaluation Agent

This agent is responsible for:
- Comprehensive model performance evaluation
- Confusion matrix analysis
- ROC curve and AUC calculation
- Feature importance analysis
- Model comparison and validation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_predict
import joblib
import os

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import ClassificationState, AgentStatus, state_manager


class ModelEvaluationAgent(BaseAgent):
    """
    Model Evaluation Agent for comprehensive performance assessment
    """
    
    def __init__(self):
        super().__init__("model_evaluation", "1.0.0")
        self.logger = logging.getLogger("agent.model_evaluation")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Model Evaluation Agent for comprehensive performance assessment",
            "capabilities": [
                "Confusion matrix analysis",
                "ROC curve and AUC calculation",
                "Precision-Recall curve analysis",
                "Feature importance analysis",
                "Cross-validation evaluation",
                "Model performance visualization"
            ],
            "dependencies": ["ml_building"]
        }
    
    def get_dependencies(self) -> list:
        """Get list of agent dependencies"""
        return ["ml_building"]
    
    async def execute(self, state: ClassificationState) -> ClassificationState:
        """
        Execute model evaluation process
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with evaluation results
        """
        try:
            self.logger.info("Starting model evaluation process")
            
            # Get model and data
            model_path = state.get("model_selection_results", {}).get("model_path")
            if not model_path or not os.path.exists(model_path):
                raise ValueError("No trained model available for evaluation")
            
            # Load model
            model = joblib.load(model_path)
            
            # Get cleaned dataset
            cleaned_df = state_manager.get_dataset(state, "cleaned")
            if cleaned_df is None:
                cleaned_df = state_manager.get_dataset(state, "original")
            if cleaned_df is None:
                raise ValueError("No cleaned dataset available")
            
            target_column = state.get("target_column")
            if not target_column:
                raise ValueError("No target column specified")
            
            # Prepare data
            X, y = self._prepare_data(cleaned_df, target_column)
            
            # Split data (same as training)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Generate confusion matrix
            confusion_mat = self._generate_confusion_matrix(y_test, y_pred)
            
            # Generate ROC curve data
            roc_data = self._generate_roc_curve_data(y_test, y_pred_proba)
            
            # Generate precision-recall curve data
            pr_data = self._generate_precision_recall_curve_data(y_test, y_pred_proba)
            
            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(model, X.columns)
            
            # Generate performance analysis
            performance_analysis = self._generate_performance_analysis(
                metrics, confusion_mat, roc_data, pr_data
            )
            
            # Update state with results
            state["evaluation_metrics"] = metrics
            state["confusion_matrix"] = confusion_mat.tolist()
            state["roc_curve_data"] = roc_data
            state["precision_recall_curve"] = pr_data
            state["feature_importance_model"] = feature_importance
            state["model_performance_analysis"] = performance_analysis
            
            # Update agent status
            state["agent_statuses"]["model_evaluation"] = AgentStatus.COMPLETED
            state["completed_agents"].append("model_evaluation")
            
            self.logger.info("Model evaluation completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            state["agent_statuses"]["model_evaluation"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state
    
    def _prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for evaluation"""
        try:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables (same as training)
            X = pd.get_dummies(X, drop_first=True)
            
            # Ensure no missing values
            if X.isnull().any().any():
                X = X.fillna(X.mean())
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                classification_report, cohen_kappa_score
            )
            
            # Basic metrics
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
                'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
                'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
                'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
                'cohen_kappa': float(cohen_kappa_score(y_true, y_pred))
            }
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = class_report
            
            # Additional metrics if probabilities available
            if y_pred_proba is not None:
                try:
                    from sklearn.metrics import log_loss
                    metrics['log_loss'] = float(log_loss(y_true, y_pred_proba))
                except:
                    metrics['log_loss'] = None
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'cohen_kappa': 0.0
            }
    
    def _generate_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            return cm
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix: {str(e)}")
            return np.array([[0, 0], [0, 0]])
    
    def _generate_roc_curve_data(self, y_true: pd.Series, y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate ROC curve data"""
        try:
            if y_pred_proba is None:
                return {"error": "No probability predictions available"}
            
            # Handle multi-class case
            if len(np.unique(y_true)) > 2:
                # Multi-class ROC
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc
                
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                
                roc_data = {
                    "fpr": {},
                    "tpr": {},
                    "auc": {}
                }
                
                for i, class_name in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    roc_data["fpr"][str(class_name)] = fpr.tolist()
                    roc_data["tpr"][str(class_name)] = tpr.tolist()
                    roc_data["auc"][str(class_name)] = float(roc_auc)
                
                # Micro-average
                fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
                roc_data["fpr"]["micro"] = fpr_micro.tolist()
                roc_data["tpr"]["micro"] = tpr_micro.tolist()
                roc_data["auc"]["micro"] = float(auc(fpr_micro, tpr_micro))
                
            else:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                roc_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(roc_auc)
                }
            
            return roc_data
            
        except Exception as e:
            self.logger.error(f"Error generating ROC curve data: {str(e)}")
            return {"error": str(e)}
    
    def _generate_precision_recall_curve_data(self, y_true: pd.Series, y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate precision-recall curve data"""
        try:
            if y_pred_proba is None:
                return {"error": "No probability predictions available"}
            
            # Handle multi-class case
            if len(np.unique(y_true)) > 2:
                from sklearn.preprocessing import label_binarize
                
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                
                pr_data = {
                    "precision": {},
                    "recall": {},
                    "average_precision": {}
                }
                
                for i, class_name in enumerate(classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                    
                    pr_data["precision"][str(class_name)] = precision.tolist()
                    pr_data["recall"][str(class_name)] = recall.tolist()
                    pr_data["average_precision"][str(class_name)] = float(avg_precision)
                
                # Micro-average
                precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba.ravel())
                avg_precision_micro = average_precision_score(y_true_bin.ravel(), y_pred_proba.ravel())
                
                pr_data["precision"]["micro"] = precision_micro.tolist()
                pr_data["recall"]["micro"] = recall_micro.tolist()
                pr_data["average_precision"]["micro"] = float(avg_precision_micro)
                
            else:
                # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
                
                pr_data = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "average_precision": float(avg_precision)
                }
            
            return pr_data
            
        except Exception as e:
            self.logger.error(f"Error generating precision-recall curve data: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Analyze feature importance"""
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                # Sort by importance
                importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                return importance_dict
            
            # Check if model has coef_ attribute (linear models)
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    # Binary classification
                    importance_dict = dict(zip(feature_names, abs(model.coef_[0])))
                else:
                    # Multi-class classification - use mean absolute coefficients
                    importance_dict = dict(zip(feature_names, abs(model.coef_).mean(axis=0)))
                
                importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                return importance_dict
            
            else:
                self.logger.warning("Model does not support feature importance analysis")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")
            return {}
    
    def _generate_performance_analysis(self, metrics: Dict, confusion_mat: np.ndarray, roc_data: Dict, pr_data: Dict) -> str:
        """Generate comprehensive performance analysis"""
        try:
            analysis = f"""
            Model Performance Analysis:
            
            Overall Performance:
            - Accuracy: {metrics.get('accuracy', 0):.4f}
            - Precision (Weighted): {metrics.get('precision_weighted', 0):.4f}
            - Recall (Weighted): {metrics.get('recall_weighted', 0):.4f}
            - F1-Score (Weighted): {metrics.get('f1_weighted', 0):.4f}
            - Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}
            
            Confusion Matrix Analysis:
            - True Positives: {confusion_mat.diagonal().sum()}
            - False Positives: {confusion_mat.sum(axis=0) - confusion_mat.diagonal()}
            - False Negatives: {confusion_mat.sum(axis=1) - confusion_mat.diagonal()}
            
            ROC Curve Analysis:
            """
            
            if isinstance(roc_data.get('auc'), dict):
                # Multi-class
                for class_name, auc_score in roc_data['auc'].items():
                    analysis += f"- {class_name} AUC: {auc_score:.4f}\n"
            elif isinstance(roc_data.get('auc'), (int, float)):
                # Binary
                analysis += f"- AUC: {roc_data['auc']:.4f}\n"
            
            analysis += "\nPrecision-Recall Analysis:\n"
            
            if isinstance(pr_data.get('average_precision'), dict):
                # Multi-class
                for class_name, ap_score in pr_data['average_precision'].items():
                    analysis += f"- {class_name} Average Precision: {ap_score:.4f}\n"
            elif isinstance(pr_data.get('average_precision'), (int, float)):
                # Binary
                analysis += f"- Average Precision: {pr_data['average_precision']:.4f}\n"
            
            analysis += f"""
            
            Model Quality Assessment:
            - {'Excellent' if metrics.get('accuracy', 0) > 0.9 else 'Good' if metrics.get('accuracy', 0) > 0.8 else 'Fair' if metrics.get('accuracy', 0) > 0.7 else 'Poor'} overall performance
            - {'Well-calibrated' if metrics.get('cohen_kappa', 0) > 0.8 else 'Moderately calibrated' if metrics.get('cohen_kappa', 0) > 0.6 else 'Poorly calibrated'} model
            """
            
            return analysis.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating performance analysis: {str(e)}")
            return f"Performance analysis completed with accuracy: {metrics.get('accuracy', 0):.4f}"
