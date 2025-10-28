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

from ..base_agent import BaseAgent, AgentResult
from ...workflows.state_management import ClassificationState, AgentStatus, state_manager


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
            
            # Generate confusion matrix (basic)
            confusion_mat = self._generate_confusion_matrix(y_test, y_pred)
            
            # Generate comprehensive confusion matrix with visualization
            confusion_matrix_data = self._generate_confusion_matrix_visualization(
                y_test, y_pred, class_names=None
            )
            
            # Generate ROC curve data (basic)
            roc_data = self._generate_roc_curve_data(y_test, y_pred_proba)
            
            # Generate comprehensive ROC curve with visualization
            roc_curve_data = self._generate_roc_curve_visualization(y_test, y_pred_proba)
            
            # Generate precision-recall curve data
            pr_data = self._generate_precision_recall_curve_data(y_test, y_pred_proba)
            
            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(model, X.columns)
            
            # Generate performance analysis (basic)
            performance_analysis = self._generate_performance_analysis(
                metrics, confusion_mat, roc_data, pr_data
            )
            
            # Generate comprehensive performance interpretation
            comprehensive_interpretation = self._generate_comprehensive_performance_interpretation(
                metrics, confusion_matrix_data, roc_curve_data, pr_data
            )
            
            # Update state with results
            state["evaluation_metrics"] = metrics
            state["confusion_matrix"] = confusion_mat.tolist()
            state["confusion_matrix_data"] = confusion_matrix_data
            state["roc_curve_data"] = roc_data
            state["roc_curve_visualization"] = roc_curve_data
            state["precision_recall_curve"] = pr_data
            state["feature_importance_model"] = feature_importance
            state["model_performance_analysis"] = performance_analysis
            state["comprehensive_performance_interpretation"] = comprehensive_interpretation
            
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
                classification_report, cohen_kappa_score, roc_auc_score
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
            
            # ROC-AUC calculation if probabilities available
            if y_pred_proba is not None:
                try:
                    # Handle multi-class case
                    if len(np.unique(y_true)) > 2:
                        # Multi-class ROC-AUC (one-vs-rest)
                        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                        metrics['roc_auc_weighted'] = float(roc_auc)
                        
                        # Also calculate macro average
                        roc_auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                        metrics['roc_auc_macro'] = float(roc_auc_macro)
                    else:
                        # Binary classification
                        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                        metrics['roc_auc'] = float(roc_auc)
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
                    metrics['roc_auc'] = None
            else:
                metrics['roc_auc'] = None
            
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
                'cohen_kappa': 0.0,
                'roc_auc': None
            }
    
    def _generate_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            return cm
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix: {str(e)}")
            return np.array([[0, 0], [0, 0]])
    
    def _generate_confusion_matrix_visualization(self, y_true: pd.Series, y_pred: np.ndarray, 
                                                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive confusion matrix with visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names for labeling
            
        Returns:
            Dictionary containing confusion matrix data and visualization info
        """
        try:
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Get unique classes
            classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            
            # Generate class names if not provided
            if class_names is None:
                class_names = [f"Class {i}" for i in classes]
            elif len(class_names) != len(classes):
                self.logger.warning(f"Class names length ({len(class_names)}) doesn't match classes ({len(classes)})")
                class_names = [f"Class {i}" for i in classes]
            
            # Calculate normalized confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
            
            # Calculate metrics from confusion matrix
            total_samples = cm.sum()
            correct_predictions = np.trace(cm)
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            # Calculate per-class metrics
            class_metrics = {}
            for i, class_name in enumerate(class_names):
                if i < len(classes):
                    true_positives = cm[i, i]
                    false_positives = cm[:, i].sum() - true_positives
                    false_negatives = cm[i, :].sum() - true_positives
                    true_negatives = total_samples - true_positives - false_positives - false_negatives
                    
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_metrics[class_name] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1_score),
                        'support': int(cm[i, :].sum()),
                        'true_positives': int(true_positives),
                        'false_positives': int(false_positives),
                        'false_negatives': int(false_negatives),
                        'true_negatives': int(true_negatives)
                    }
            
            # Generate visualization data
            visualization_data = {
                'confusion_matrix': cm.tolist(),
                'confusion_matrix_normalized': cm_normalized.tolist(),
                'class_names': class_names,
                'classes': classes.tolist() if hasattr(classes, 'tolist') else list(classes),
                'total_samples': int(total_samples),
                'correct_predictions': int(correct_predictions),
                'accuracy': float(accuracy),
                'class_metrics': class_metrics,
                'matrix_summary': {
                    'shape': cm.shape,
                    'total_elements': int(cm.size),
                    'non_zero_elements': int(np.count_nonzero(cm)),
                    'max_value': int(cm.max()),
                    'min_value': int(cm.min())
                }
            }
            
            # Generate matplotlib visualization code
            visualization_code = self._generate_confusion_matrix_plot_code(
                cm, cm_normalized, class_names, accuracy
            )
            
            visualization_data['plot_code'] = visualization_code
            
            self.logger.info(f"Generated confusion matrix visualization for {len(classes)} classes")
            return visualization_data
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix visualization: {str(e)}")
            return {
                'error': str(e),
                'confusion_matrix': [[0, 0], [0, 0]],
                'class_names': ['Class 0', 'Class 1'],
                'classes': [0, 1],
                'total_samples': 0,
                'accuracy': 0.0,
                'class_metrics': {},
                'plot_code': "# Error generating confusion matrix plot"
            }
    
    def _generate_confusion_matrix_plot_code(self, cm: np.ndarray, cm_normalized: np.ndarray, 
                                           class_names: List[str], accuracy: float) -> str:
        """Generate matplotlib code for confusion matrix visualization"""
        
        # Convert arrays to lists for string formatting
        cm_list = cm.tolist() if hasattr(cm, 'tolist') else cm
        cm_normalized_list = cm_normalized.tolist() if hasattr(cm_normalized, 'tolist') else cm_normalized
        
        plot_code = f'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion Matrix Data
cm = {cm_list}
cm_normalized = {cm_normalized_list}
class_names = {class_names}
accuracy = {accuracy:.4f}

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Raw Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=ax1, cbar_kws={{'label': 'Count'}})
ax1.set_title(f'Confusion Matrix\\nAccuracy: {{accuracy:.4f}}')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# Normalized Confusion Matrix
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax2, cbar_kws={{'label': 'Proportion'}})
ax2.set_title('Normalized Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# Add text annotations for better interpretation
for i in range(len(class_names)):
    for j in range(len(class_names)):
        # Add percentage text for raw matrix
        text = ax1.text(j + 0.5, i + 0.5, f'{{cm[i, j]}}\\n({{cm_normalized[i, j]:.1%}})', 
                       ha="center", va="center", color="red" if i != j else "black", fontweight='bold')
        
        # Add percentage text for normalized matrix
        text = ax2.text(j + 0.5, i + 0.5, f'{{cm_normalized[i, j]:.1%}}', 
                       ha="center", va="center", color="red" if i != j else "black", fontweight='bold')

plt.tight_layout()
plt.savefig('/app/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Confusion Matrix Summary:")
print(f"Total samples: {{cm.sum()}}")
print(f"Correct predictions: {{np.trace(cm)}}")
print(f"Accuracy: {{accuracy:.4f}}")
print("\\nPer-class metrics:")
for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{{class_name}}: Precision={{precision:.3f}}, Recall={{recall:.3f}}, F1={{f1:.3f}}")
'''
        
        return plot_code
    
    def _generate_roc_curve_data(self, y_true: pd.Series, y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate ROC curve data"""
        try:
            if y_pred_proba is None:
                return {"error": "No probability predictions available"}
            
            # Import required metrics
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Handle multi-class case
            if len(np.unique(y_true)) > 2:
                # Multi-class ROC
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
    
    def _generate_roc_curve_visualization(self, y_true: pd.Series, y_pred_proba: Optional[np.ndarray], 
                                         class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive ROC curve with visualization
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Optional list of class names for labeling
            
        Returns:
            Dictionary containing ROC curve data and visualization info
        """
        try:
            if y_pred_proba is None:
                return {"error": "No probability predictions available"}
            
            # Import required metrics
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Get unique classes
            classes = sorted(np.unique(y_true))
            
            # Generate class names if not provided
            if class_names is None:
                class_names = [f"Class {i}" for i in classes]
            elif len(class_names) != len(classes):
                self.logger.warning(f"Class names length ({len(class_names)}) doesn't match classes ({len(classes)})")
                class_names = [f"Class {i}" for i in classes]
            
            # Handle multi-class case
            if len(classes) > 2:
                # Multi-class ROC
                y_true_bin = label_binarize(y_true, classes=classes)
                
                roc_data = {
                    "fpr": {},
                    "tpr": {},
                    "auc": {},
                    "class_names": class_names,
                    "classes": classes.tolist() if hasattr(classes, 'tolist') else list(classes),
                    "is_multiclass": True
                }
                
                # Calculate ROC for each class
                for i, class_name in enumerate(class_names):
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
                
                # Macro-average
                fpr_macro = np.unique(np.concatenate([roc_data["fpr"][str(class_names[i])] for i in range(len(classes))]))
                tpr_macro = np.zeros_like(fpr_macro)
                for i in range(len(classes)):
                    tpr_macro += np.interp(fpr_macro, roc_data["fpr"][str(class_names[i])], roc_data["tpr"][str(class_names[i])])
                tpr_macro /= len(classes)
                roc_auc_macro = auc(fpr_macro, tpr_macro)
                
                roc_data["fpr"]["macro"] = fpr_macro.tolist()
                roc_data["tpr"]["macro"] = tpr_macro.tolist()
                roc_data["auc"]["macro"] = float(roc_auc_macro)
                
            else:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                roc_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(roc_auc),
                    "class_names": class_names,
                    "classes": classes.tolist() if hasattr(classes, 'tolist') else list(classes),
                    "is_multiclass": False
                }
            
            # Generate visualization code
            visualization_code = self._generate_roc_curve_plot_code(roc_data)
            roc_data['plot_code'] = visualization_code
            
            self.logger.info(f"Generated ROC curve visualization for {len(classes)} classes")
            return roc_data
            
        except Exception as e:
            self.logger.error(f"Error generating ROC curve visualization: {str(e)}")
            return {
                'error': str(e),
                'fpr': [],
                'tpr': [],
                'auc': 0.0,
                'class_names': ['Class 0', 'Class 1'],
                'classes': [0, 1],
                'is_multiclass': False,
                'plot_code': "# Error generating ROC curve plot"
            }
    
    def _generate_roc_curve_plot_code(self, roc_data: Dict[str, Any]) -> str:
        """Generate matplotlib code for ROC curve visualization"""
        
        plot_code = f'''
import matplotlib.pyplot as plt
import numpy as np

# ROC Curve Data
roc_data = {roc_data}

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

if roc_data.get('is_multiclass', False):
    # Multi-class ROC curves
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot individual class ROC curves
    for i, (class_name, fpr) in enumerate(roc_data['fpr'].items()):
        if class_name in ['micro', 'macro']:
            continue
        tpr = roc_data['tpr'][class_name]
        auc_score = roc_data['auc'][class_name]
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{{class_name}} (AUC = {{auc_score:.3f}})')
    
    # Plot micro-average
    if 'micro' in roc_data['fpr']:
        fpr_micro = roc_data['fpr']['micro']
        tpr_micro = roc_data['tpr']['micro']
        auc_micro = roc_data['auc']['micro']
        ax.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
                label=f'Micro-average (AUC = {{auc_micro:.3f}})')
    
    # Plot macro-average
    if 'macro' in roc_data['fpr']:
        fpr_macro = roc_data['fpr']['macro']
        tpr_macro = roc_data['tpr']['macro']
        auc_macro = roc_data['auc']['macro']
        ax.plot(fpr_macro, tpr_macro, color='navy', linestyle=':', linewidth=4,
                label=f'Macro-average (AUC = {{auc_macro:.3f}})')
    
    ax.set_title('Multi-class ROC Curves', fontsize=16, fontweight='bold')
    
else:
    # Binary classification ROC curve
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc_score = roc_data['auc']
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {{auc_score:.3f}})')
    ax.set_title('ROC Curve', fontsize=16, fontweight='bold')

# Plot diagonal line (random classifier)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
        label='Random Classifier (AUC = 0.5)')

# Customize the plot
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

# Add AUC interpretation text
if roc_data.get('is_multiclass', False):
    avg_auc = np.mean([auc for name, auc in roc_data['auc'].items() if name not in ['micro', 'macro']])
    interpretation = f"Average AUC: {{avg_auc:.3f}}"
else:
    auc_score = roc_data['auc']
    if auc_score >= 0.9:
        interpretation = "Excellent (AUC ≥ 0.9)"
    elif auc_score >= 0.8:
        interpretation = "Good (0.8 ≤ AUC < 0.9)"
    elif auc_score >= 0.7:
        interpretation = "Fair (0.7 ≤ AUC < 0.8)"
    else:
        interpretation = "Poor (AUC < 0.7)"
    
    ax.text(0.6, 0.2, f"Model Performance: {{interpretation}}", 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/app/results/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("ROC Curve Summary:")
if roc_data.get('is_multiclass', False):
    print(f"Number of classes: {{len(roc_data['classes'])}}")
    for class_name, auc_score in roc_data['auc'].items():
        if class_name not in ['micro', 'macro']:
            print(f"{{class_name}}: AUC = {{auc_score:.3f}}")
    if 'micro' in roc_data['auc']:
        print(f"Micro-average: AUC = {{roc_data['auc']['micro']:.3f}}")
    if 'macro' in roc_data['auc']:
        print(f"Macro-average: AUC = {{roc_data['auc']['macro']:.3f}}")
else:
    print(f"Binary Classification AUC: {{roc_data['auc']:.3f}}")
    print(f"Performance: {{interpretation}}")
'''
        
        return plot_code
    
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
    
    def _generate_comprehensive_performance_interpretation(self, metrics: Dict, confusion_matrix_data: Dict, 
                                                         roc_curve_data: Dict, pr_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive performance interpretation with detailed insights
        
        Args:
            metrics: Performance metrics dictionary
            confusion_matrix_data: Confusion matrix visualization data
            roc_curve_data: ROC curve visualization data
            pr_data: Precision-recall curve data
            
        Returns:
            Dictionary containing comprehensive performance interpretation
        """
        try:
            interpretation = {
                "executive_summary": "",
                "detailed_analysis": {},
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "model_quality_score": 0.0,
                "business_impact": {},
                "technical_insights": {},
                "visualization_insights": {}
            }
            
            # Calculate model quality score (0-100)
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision_weighted', 0)
            recall = metrics.get('recall_weighted', 0)
            f1 = metrics.get('f1_weighted', 0)
            kappa = metrics.get('cohen_kappa', 0)
            
            # Weighted quality score
            quality_score = (accuracy * 0.3 + precision * 0.2 + recall * 0.2 + f1 * 0.2 + kappa * 0.1) * 100
            interpretation["model_quality_score"] = round(quality_score, 1)
            
            # Generate executive summary
            interpretation["executive_summary"] = self._generate_executive_summary(metrics, quality_score)
            
            # Generate detailed analysis
            interpretation["detailed_analysis"] = self._generate_detailed_analysis(metrics, confusion_matrix_data, roc_curve_data)
            
            # Identify strengths and weaknesses
            interpretation["strengths"] = self._identify_strengths(metrics, confusion_matrix_data, roc_curve_data)
            interpretation["weaknesses"] = self._identify_weaknesses(metrics, confusion_matrix_data, roc_curve_data)
            
            # Generate recommendations
            interpretation["recommendations"] = self._generate_recommendations(metrics, confusion_matrix_data, roc_curve_data, quality_score)
            
            # Business impact analysis
            interpretation["business_impact"] = self._analyze_business_impact(metrics, confusion_matrix_data)
            
            # Technical insights
            interpretation["technical_insights"] = self._generate_technical_insights(metrics, confusion_matrix_data, roc_curve_data)
            
            # Visualization insights
            interpretation["visualization_insights"] = self._generate_visualization_insights(confusion_matrix_data, roc_curve_data)
            
            self.logger.info(f"Generated comprehensive performance interpretation with quality score: {quality_score:.1f}")
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive performance interpretation: {str(e)}")
            return {
                "executive_summary": "Error generating performance interpretation",
                "detailed_analysis": {},
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "model_quality_score": 0.0,
                "business_impact": {},
                "technical_insights": {},
                "visualization_insights": {}
            }
    
    def _generate_executive_summary(self, metrics: Dict, quality_score: float) -> str:
        """Generate executive summary of model performance"""
        accuracy = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_weighted', 0)
        kappa = metrics.get('cohen_kappa', 0)
        
        if quality_score >= 90:
            performance_level = "Excellent"
            recommendation = "The model is ready for production deployment with high confidence."
        elif quality_score >= 80:
            performance_level = "Good"
            recommendation = "The model shows strong performance and is suitable for production with minor monitoring."
        elif quality_score >= 70:
            performance_level = "Fair"
            recommendation = "The model has acceptable performance but may benefit from further optimization before deployment."
        else:
            performance_level = "Poor"
            recommendation = "The model requires significant improvement before it can be considered for production use."
        
        return f"""
        The machine learning model demonstrates {performance_level.lower()} performance with an overall quality score of {quality_score:.1f}/100.
        
        Key Performance Indicators:
        • Accuracy: {accuracy:.1%} - The model correctly predicts {accuracy:.1%} of all instances
        • F1-Score: {f1:.3f} - Balanced precision and recall performance
        • Cohen's Kappa: {kappa:.3f} - Model agreement beyond chance
        
        {recommendation}
        """
    
    def _generate_detailed_analysis(self, metrics: Dict, confusion_matrix_data: Dict, roc_curve_data: Dict) -> Dict[str, Any]:
        """Generate detailed performance analysis"""
        analysis = {
            "accuracy_analysis": {},
            "precision_recall_analysis": {},
            "confusion_matrix_analysis": {},
            "roc_curve_analysis": {},
            "class_performance": {}
        }
        
        # Accuracy analysis
        accuracy = metrics.get('accuracy', 0)
        analysis["accuracy_analysis"] = {
            "value": accuracy,
            "interpretation": self._interpret_accuracy(accuracy),
            "confidence_level": "High" if accuracy > 0.9 else "Medium" if accuracy > 0.8 else "Low"
        }
        
        # Precision-Recall analysis
        precision = metrics.get('precision_weighted', 0)
        recall = metrics.get('recall_weighted', 0)
        f1 = metrics.get('f1_weighted', 0)
        
        analysis["precision_recall_analysis"] = {
            "precision": {"value": precision, "interpretation": self._interpret_precision(precision)},
            "recall": {"value": recall, "interpretation": self._interpret_recall(recall)},
            "f1_score": {"value": f1, "interpretation": self._interpret_f1(f1)},
            "balance": self._assess_precision_recall_balance(precision, recall)
        }
        
        # Confusion matrix analysis
        if confusion_matrix_data and 'confusion_matrix' in confusion_matrix_data:
            cm = np.array(confusion_matrix_data['confusion_matrix'])
            analysis["confusion_matrix_analysis"] = self._analyze_confusion_matrix(cm)
        
        # ROC curve analysis
        if roc_curve_data and 'auc' in roc_curve_data:
            analysis["roc_curve_analysis"] = self._analyze_roc_curve(roc_curve_data)
        
        # Class performance analysis
        if confusion_matrix_data and 'class_metrics' in confusion_matrix_data:
            analysis["class_performance"] = self._analyze_class_performance(confusion_matrix_data['class_metrics'])
        
        return analysis
    
    def _interpret_accuracy(self, accuracy: float) -> str:
        """Interpret accuracy score"""
        if accuracy >= 0.95:
            return "Exceptional accuracy - model performs exceptionally well"
        elif accuracy >= 0.90:
            return "High accuracy - model performs very well"
        elif accuracy >= 0.80:
            return "Good accuracy - model performs well"
        elif accuracy >= 0.70:
            return "Moderate accuracy - model performs adequately"
        else:
            return "Low accuracy - model needs improvement"
    
    def _interpret_precision(self, precision: float) -> str:
        """Interpret precision score"""
        if precision >= 0.90:
            return "High precision - very few false positives"
        elif precision >= 0.80:
            return "Good precision - low false positive rate"
        elif precision >= 0.70:
            return "Moderate precision - some false positives"
        else:
            return "Low precision - many false positives"
    
    def _interpret_recall(self, recall: float) -> str:
        """Interpret recall score"""
        if recall >= 0.90:
            return "High recall - captures most positive cases"
        elif recall >= 0.80:
            return "Good recall - captures most positive cases"
        elif recall >= 0.70:
            return "Moderate recall - misses some positive cases"
        else:
            return "Low recall - misses many positive cases"
    
    def _interpret_f1(self, f1: float) -> str:
        """Interpret F1 score"""
        if f1 >= 0.90:
            return "Excellent balance between precision and recall"
        elif f1 >= 0.80:
            return "Good balance between precision and recall"
        elif f1 >= 0.70:
            return "Moderate balance between precision and recall"
        else:
            return "Poor balance between precision and recall"
    
    def _assess_precision_recall_balance(self, precision: float, recall: float) -> Dict[str, Any]:
        """Assess the balance between precision and recall"""
        diff = abs(precision - recall)
        
        if diff <= 0.05:
            balance = "Excellent"
            interpretation = "Precision and recall are well balanced"
        elif diff <= 0.10:
            balance = "Good"
            interpretation = "Precision and recall are reasonably balanced"
        elif diff <= 0.20:
            balance = "Moderate"
            interpretation = "Some imbalance between precision and recall"
        else:
            balance = "Poor"
            interpretation = "Significant imbalance between precision and recall"
        
        return {
            "balance": balance,
            "difference": diff,
            "interpretation": interpretation,
            "bias_toward": "Precision" if precision > recall else "Recall" if recall > precision else "Neither"
        }
    
    def _analyze_confusion_matrix(self, cm: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix for insights"""
        total_samples = cm.sum()
        correct_predictions = np.trace(cm)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Calculate per-class metrics
        class_metrics = {}
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = total_samples - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[f"Class_{i}"] = {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        
        # Identify most confused classes
        confusion_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        "true_class": i,
                        "predicted_class": j,
                        "count": int(cm[i, j]),
                        "percentage": float(cm[i, j] / cm[i, :].sum()) if cm[i, :].sum() > 0 else 0
                    })
        
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            "total_samples": int(total_samples),
            "correct_predictions": int(correct_predictions),
            "accuracy": float(accuracy),
            "class_metrics": class_metrics,
            "most_confused_pairs": confusion_pairs[:5],  # Top 5 most confused pairs
            "matrix_insights": self._generate_confusion_matrix_insights(cm)
        }
    
    def _generate_confusion_matrix_insights(self, cm: np.ndarray) -> List[str]:
        """Generate insights from confusion matrix"""
        insights = []
        
        # Check for perfect diagonal (perfect classification)
        if np.all(cm == np.diag(np.diag(cm))):
            insights.append("Perfect classification - no misclassifications detected")
        
        # Check for class imbalance
        class_totals = cm.sum(axis=1)
        max_class = class_totals.max()
        min_class = class_totals.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio > 3:
            insights.append(f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        elif imbalance_ratio > 2:
            insights.append(f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        
        # Check for systematic errors
        off_diagonal_sum = cm.sum() - np.trace(cm)
        if off_diagonal_sum > 0:
            insights.append(f"Total misclassifications: {int(off_diagonal_sum)}")
            
            # Find most common error pattern
            cm_copy = cm.copy()
            np.fill_diagonal(cm_copy, 0)
            max_error_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
            max_error_count = cm_copy[max_error_idx]
            if max_error_count > 0:
                insights.append(f"Most common error: Class {max_error_idx[0]} → Class {max_error_idx[1]} ({int(max_error_count)} times)")
        
        return insights
    
    def _analyze_roc_curve(self, roc_data: Dict) -> Dict[str, Any]:
        """Analyze ROC curve data for insights"""
        analysis = {
            "auc_interpretation": {},
            "curve_characteristics": {},
            "performance_level": ""
        }
        
        if isinstance(roc_data.get('auc'), dict):
            # Multi-class analysis
            auc_scores = [score for name, score in roc_data['auc'].items() if name not in ['micro', 'macro']]
            avg_auc = np.mean(auc_scores) if auc_scores else 0
            
            analysis["auc_interpretation"] = {
                "average_auc": float(avg_auc),
                "individual_scores": roc_data['auc'],
                "interpretation": self._interpret_auc(avg_auc)
            }
            
            if 'micro' in roc_data['auc']:
                analysis["auc_interpretation"]["micro_average"] = roc_data['auc']['micro']
            if 'macro' in roc_data['auc']:
                analysis["auc_interpretation"]["macro_average"] = roc_data['auc']['macro']
                
        else:
            # Binary analysis
            auc_score = roc_data.get('auc', 0)
            analysis["auc_interpretation"] = {
                "auc_score": float(auc_score),
                "interpretation": self._interpret_auc(auc_score)
            }
        
        # Determine performance level
        main_auc = analysis["auc_interpretation"].get("auc_score", analysis["auc_interpretation"].get("average_auc", 0))
        if main_auc >= 0.9:
            analysis["performance_level"] = "Excellent"
        elif main_auc >= 0.8:
            analysis["performance_level"] = "Good"
        elif main_auc >= 0.7:
            analysis["performance_level"] = "Fair"
        else:
            analysis["performance_level"] = "Poor"
        
        return analysis
    
    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC score"""
        if auc >= 0.9:
            return "Excellent discriminative ability"
        elif auc >= 0.8:
            return "Good discriminative ability"
        elif auc >= 0.7:
            return "Fair discriminative ability"
        elif auc >= 0.6:
            return "Poor discriminative ability"
        else:
            return "No discriminative ability (worse than random)"
    
    def _analyze_class_performance(self, class_metrics: Dict) -> Dict[str, Any]:
        """Analyze individual class performance"""
        analysis = {
            "best_performing_class": "",
            "worst_performing_class": "",
            "class_rankings": [],
            "performance_gaps": {}
        }
        
        if not class_metrics:
            return analysis
        
        # Rank classes by F1 score
        class_scores = [(name, metrics.get('f1_score', 0)) for name, metrics in class_metrics.items()]
        class_scores.sort(key=lambda x: x[1], reverse=True)
        
        analysis["class_rankings"] = [{"class": name, "f1_score": score} for name, score in class_scores]
        
        if class_scores:
            analysis["best_performing_class"] = class_scores[0][0]
            analysis["worst_performing_class"] = class_scores[-1][0]
            
            # Calculate performance gaps
            best_score = class_scores[0][1]
            worst_score = class_scores[-1][1]
            analysis["performance_gaps"] = {
                "max_f1_gap": best_score - worst_score,
                "relative_gap": (best_score - worst_score) / best_score if best_score > 0 else 0
            }
        
        return analysis
    
    def _identify_strengths(self, metrics: Dict, confusion_matrix_data: Dict, roc_curve_data: Dict) -> List[str]:
        """Identify model strengths"""
        strengths = []
        
        accuracy = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_weighted', 0)
        kappa = metrics.get('cohen_kappa', 0)
        
        if accuracy >= 0.9:
            strengths.append(f"High accuracy ({accuracy:.1%}) - model correctly predicts most instances")
        
        if f1 >= 0.9:
            strengths.append(f"Excellent F1-score ({f1:.3f}) - well-balanced precision and recall")
        
        if kappa >= 0.8:
            strengths.append(f"Strong agreement beyond chance (κ = {kappa:.3f})")
        
        # ROC curve strengths
        if isinstance(roc_curve_data.get('auc'), dict):
            auc_scores = [score for name, score in roc_curve_data['auc'].items() if name not in ['micro', 'macro']]
            if auc_scores and np.mean(auc_scores) >= 0.8:
                strengths.append(f"Good discriminative ability (avg AUC = {np.mean(auc_scores):.3f})")
        elif isinstance(roc_curve_data.get('auc'), (int, float)):
            if roc_curve_data['auc'] >= 0.8:
                strengths.append(f"Good discriminative ability (AUC = {roc_curve_data['auc']:.3f})")
        
        # Confusion matrix strengths
        if confusion_matrix_data and 'confusion_matrix' in confusion_matrix_data:
            cm = np.array(confusion_matrix_data['confusion_matrix'])
            diagonal_ratio = np.trace(cm) / cm.sum()
            if diagonal_ratio >= 0.9:
                strengths.append(f"Low misclassification rate ({1-diagonal_ratio:.1%})")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict, confusion_matrix_data: Dict, roc_curve_data: Dict) -> List[str]:
        """Identify model weaknesses"""
        weaknesses = []
        
        accuracy = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_weighted', 0)
        kappa = metrics.get('cohen_kappa', 0)
        precision = metrics.get('precision_weighted', 0)
        recall = metrics.get('recall_weighted', 0)
        
        if accuracy < 0.7:
            weaknesses.append(f"Low accuracy ({accuracy:.1%}) - model struggles with correct predictions")
        
        if f1 < 0.7:
            weaknesses.append(f"Poor F1-score ({f1:.3f}) - imbalanced precision and recall")
        
        if kappa < 0.6:
            weaknesses.append(f"Weak agreement beyond chance (κ = {kappa:.3f})")
        
        # Precision-Recall imbalance
        if abs(precision - recall) > 0.2:
            if precision > recall:
                weaknesses.append(f"Precision-Recall imbalance - high precision ({precision:.3f}) but low recall ({recall:.3f})")
            else:
                weaknesses.append(f"Precision-Recall imbalance - high recall ({recall:.3f}) but low precision ({precision:.3f})")
        
        # ROC curve weaknesses
        if isinstance(roc_curve_data.get('auc'), dict):
            auc_scores = [score for name, score in roc_curve_data['auc'].items() if name not in ['micro', 'macro']]
            if auc_scores and np.mean(auc_scores) < 0.7:
                weaknesses.append(f"Poor discriminative ability (avg AUC = {np.mean(auc_scores):.3f})")
        elif isinstance(roc_curve_data.get('auc'), (int, float)):
            if roc_curve_data['auc'] < 0.7:
                weaknesses.append(f"Poor discriminative ability (AUC = {roc_curve_data['auc']:.3f})")
        
        return weaknesses
    
    def _generate_recommendations(self, metrics: Dict, confusion_matrix_data: Dict, roc_curve_data: Dict, quality_score: float) -> List[str]:
        """Generate actionable recommendations for model improvement"""
        recommendations = []
        
        accuracy = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_weighted', 0)
        precision = metrics.get('precision_weighted', 0)
        recall = metrics.get('recall_weighted', 0)
        
        # General recommendations based on quality score
        if quality_score < 70:
            recommendations.append("Consider collecting more training data or using data augmentation techniques")
            recommendations.append("Try different algorithms or ensemble methods")
            recommendations.append("Perform feature engineering to improve model performance")
        
        # Precision-Recall specific recommendations
        if precision < 0.7:
            recommendations.append("Improve precision by reducing false positives - consider threshold tuning or feature selection")
        
        if recall < 0.7:
            recommendations.append("Improve recall by reducing false negatives - consider class balancing or different algorithms")
        
        if abs(precision - recall) > 0.2:
            recommendations.append("Address precision-recall imbalance through threshold optimization or cost-sensitive learning")
        
        # Confusion matrix specific recommendations
        if confusion_matrix_data and 'class_metrics' in confusion_matrix_data:
            class_metrics = confusion_matrix_data['class_metrics']
            f1_scores = [metrics.get('f1_score', 0) for metrics in class_metrics.values()]
            if f1_scores and max(f1_scores) - min(f1_scores) > 0.3:
                recommendations.append("Address class imbalance - some classes perform significantly better than others")
        
        # ROC curve specific recommendations
        if isinstance(roc_curve_data.get('auc'), dict):
            auc_scores = [score for name, score in roc_curve_data['auc'].items() if name not in ['micro', 'macro']]
            if auc_scores and np.mean(auc_scores) < 0.8:
                recommendations.append("Improve discriminative ability - consider feature selection or different algorithms")
        elif isinstance(roc_curve_data.get('auc'), (int, float)):
            if roc_curve_data['auc'] < 0.8:
                recommendations.append("Improve discriminative ability - consider feature selection or different algorithms")
        
        # Deployment recommendations
        if quality_score >= 80:
            recommendations.append("Model is ready for production deployment with monitoring")
        elif quality_score >= 70:
            recommendations.append("Consider A/B testing before full production deployment")
        else:
            recommendations.append("Model needs significant improvement before production consideration")
        
        return recommendations
    
    def _analyze_business_impact(self, metrics: Dict, confusion_matrix_data: Dict) -> Dict[str, Any]:
        """Analyze business impact of model performance"""
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision_weighted', 0)
        recall = metrics.get('recall_weighted', 0)
        
        impact = {
            "confidence_level": "",
            "risk_assessment": "",
            "deployment_readiness": "",
            "monitoring_requirements": []
        }
        
        # Confidence level
        if accuracy >= 0.9 and precision >= 0.9 and recall >= 0.9:
            impact["confidence_level"] = "High - Model can be trusted for critical decisions"
        elif accuracy >= 0.8 and precision >= 0.8 and recall >= 0.8:
            impact["confidence_level"] = "Medium - Model suitable for most business decisions"
        else:
            impact["confidence_level"] = "Low - Model requires human oversight for important decisions"
        
        # Risk assessment
        if accuracy < 0.7:
            impact["risk_assessment"] = "High risk - Model may cause significant business impact"
        elif accuracy < 0.8:
            impact["risk_assessment"] = "Medium risk - Model may cause moderate business impact"
        else:
            impact["risk_assessment"] = "Low risk - Model unlikely to cause significant business impact"
        
        # Deployment readiness
        if accuracy >= 0.85 and precision >= 0.8 and recall >= 0.8:
            impact["deployment_readiness"] = "Ready for production deployment"
        elif accuracy >= 0.75:
            impact["deployment_readiness"] = "Ready for limited deployment with monitoring"
        else:
            impact["deployment_readiness"] = "Not ready for production deployment"
        
        # Monitoring requirements
        if accuracy < 0.9:
            impact["monitoring_requirements"].append("Continuous accuracy monitoring")
        if precision < 0.8:
            impact["monitoring_requirements"].append("Precision monitoring for false positive control")
        if recall < 0.8:
            impact["monitoring_requirements"].append("Recall monitoring for false negative control")
        
        return impact
    
    def _generate_technical_insights(self, metrics: Dict, confusion_matrix_data: Dict, roc_curve_data: Dict) -> Dict[str, Any]:
        """Generate technical insights for model optimization"""
        insights = {
            "data_quality_indicators": {},
            "model_complexity_insights": {},
            "hyperparameter_tuning_suggestions": [],
            "feature_importance_insights": {}
        }
        
        # Data quality indicators
        accuracy = metrics.get('accuracy', 0)
        kappa = metrics.get('cohen_kappa', 0)
        
        insights["data_quality_indicators"] = {
            "accuracy_vs_random": accuracy - 0.5,  # Assuming binary classification baseline
            "kappa_interpretation": "Strong agreement" if kappa > 0.8 else "Moderate agreement" if kappa > 0.6 else "Weak agreement",
            "data_sufficiency": "Sufficient" if accuracy > 0.8 else "May need more data"
        }
        
        # Model complexity insights
        if accuracy > 0.95:
            insights["model_complexity_insights"]["overfitting_risk"] = "High - Consider regularization"
        elif accuracy > 0.9:
            insights["model_complexity_insights"]["overfitting_risk"] = "Medium - Monitor validation performance"
        else:
            insights["model_complexity_insights"]["overfitting_risk"] = "Low - Model may be underfitting"
        
        # Hyperparameter tuning suggestions
        if accuracy < 0.8:
            insights["hyperparameter_tuning_suggestions"].append("Consider increasing model complexity")
        if kappa < 0.7:
            insights["hyperparameter_tuning_suggestions"].append("Try different algorithms or ensemble methods")
        
        return insights
    
    def _generate_visualization_insights(self, confusion_matrix_data: Dict, roc_curve_data: Dict) -> Dict[str, Any]:
        """Generate insights from visualizations"""
        insights = {
            "confusion_matrix_insights": [],
            "roc_curve_insights": [],
            "visualization_recommendations": []
        }
        
        # Confusion matrix insights
        if confusion_matrix_data and 'confusion_matrix' in confusion_matrix_data:
            cm = np.array(confusion_matrix_data['confusion_matrix'])
            diagonal_ratio = np.trace(cm) / cm.sum()
            
            if diagonal_ratio > 0.95:
                insights["confusion_matrix_insights"].append("Excellent classification with minimal errors")
            elif diagonal_ratio > 0.85:
                insights["confusion_matrix_insights"].append("Good classification with few errors")
            else:
                insights["confusion_matrix_insights"].append("Classification needs improvement")
            
            # Check for specific error patterns
            off_diagonal = cm - np.diag(np.diag(cm))
            if np.any(off_diagonal > cm.sum() * 0.1):  # More than 10% errors in any cell
                insights["confusion_matrix_insights"].append("Significant misclassification patterns detected")
        
        # ROC curve insights
        if roc_curve_data and 'auc' in roc_curve_data:
            if isinstance(roc_curve_data['auc'], dict):
                auc_scores = [score for name, score in roc_curve_data['auc'].items() if name not in ['micro', 'macro']]
                if auc_scores:
                    avg_auc = np.mean(auc_scores)
                    if avg_auc > 0.9:
                        insights["roc_curve_insights"].append("Excellent discriminative ability across all classes")
                    elif avg_auc > 0.8:
                        insights["roc_curve_insights"].append("Good discriminative ability")
                    else:
                        insights["roc_curve_insights"].append("Poor discriminative ability - consider feature engineering")
            else:
                auc_score = roc_curve_data['auc']
                if auc_score > 0.9:
                    insights["roc_curve_insights"].append("Excellent discriminative ability")
                elif auc_score > 0.8:
                    insights["roc_curve_insights"].append("Good discriminative ability")
                else:
                    insights["roc_curve_insights"].append("Poor discriminative ability")
        
        # Visualization recommendations
        insights["visualization_recommendations"] = [
            "Use confusion matrix to identify specific misclassification patterns",
            "ROC curve helps assess discriminative ability and threshold selection",
            "Consider precision-recall curves for imbalanced datasets",
            "Feature importance plots can guide feature selection"
        ]
        
        return insights

    async def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """
        LAYER 1: Analyze model performance with hardcoded metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary containing Layer 1 evaluation results
        """
        self.logger.info("🔍 LAYER 1: Analyzing model performance")
        
        # Get model path
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
            raise ValueError("No dataset available")
        
        target_column = state.get("target_column")
        if not target_column:
            raise ValueError("No target column specified")
        
        # Prepare data
        X, y = self._prepare_data(cleaned_df, target_column)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        analysis_results = {
            "metrics": metrics,
            "model_type": type(model).__name__,
            "test_size": len(X_test),
            "has_probability": y_pred_proba is not None,
        }
        
        self.logger.info("✅ LAYER 1: Model evaluation analysis complete")
        return analysis_results
    
    def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
        """
        LAYER 2: Generate prompt for LLM to create advanced evaluation code.
        
        Args:
            layer1_results: Results from Layer 1 analysis
            state: Current workflow state
            
        Returns:
            Prompt string for LLM code generation
        """
        self.logger.info("🔧 LAYER 2: Generating LLM code generation prompt for model evaluation")
        
        metrics = layer1_results.get("metrics", {})
        model_type = layer1_results.get("model_type", "unknown")
        
        prompt = f"""Generate advanced Python code for comprehensive model evaluation based on the following analysis:

## Current Model:
- Type: {model_type}
- Test Size: {layer1_results.get('test_size', 0)}
- Has Probability: {layer1_results.get('has_probability', False)}

## Baseline Metrics:
{metrics}

## Requirements for Generated Code:
1. Generate advanced visualizations (confusion matrix, ROC curve, feature importance)
2. Perform statistical significance testing
3. Create comprehensive performance reports
4. Analyze model calibration and probability distributions
5. Detect potential overfitting or underfitting
6. Generate actionable insights and recommendations
7. Use only: sklearn, matplotlib, seaborn, numpy, pandas
8. Add clear comments explaining each evaluation
9. Return structured evaluation results dictionary

Generate comprehensive, production-ready Python code:"""
        
        return prompt
    
    def process_sandbox_results(
        self,
        sandbox_output: Dict[str, Any],
        layer1_results: Dict[str, Any],
        state: ClassificationState
    ) -> Dict[str, Any]:
        """
        LAYER 2: Process and validate sandbox execution results for model evaluation.
        
        Args:
            sandbox_output: Raw output from sandbox execution
            layer1_results: Results from Layer 1 (for comparison)
            state: Current workflow state
            
        Returns:
            Processed and validated evaluation results
        """
        self.logger.info("🔍 LAYER 2: Processing sandbox results for model evaluation")
        
        # Validate sandbox execution was successful
        if sandbox_output.get("status") != "SUCCESS":
            raise ValueError(f"Sandbox execution failed: {sandbox_output.get('error', 'Unknown error')}")
        
        # Extract evaluation results from sandbox output
        evaluation_data = sandbox_output.get("output", {})
        
        # Validate the output structure
        if not isinstance(evaluation_data, dict):
            raise ValueError("Sandbox output should contain evaluation results")
        
        result = {
            "advanced_evaluation": evaluation_data,
            "layer2_success": True,
            "sandbox_execution_time": sandbox_output.get("execution_time", 0)
        }
        
        self.logger.info("✅ LAYER 2: Sandbox results processed and validated")
        return result
