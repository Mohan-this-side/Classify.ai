"""
Human-in-the-Loop Approval Gates for Classification Workflow

This module defines the key decision points in the workflow where human approval
is required, along with the approval gate implementation and management.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval gate"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    CANCELLED = "cancelled"


class ApprovalGateType(Enum):
    """Types of approval gates in the workflow"""
    DATA_CLEANING_STRATEGY = "data_cleaning_strategy"
    FEATURE_ENGINEERING_PLAN = "feature_engineering_plan"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FINAL_MODEL_APPROVAL = "final_model_approval"


class ApprovalGate(TypedDict):
    """Structure for an approval gate"""
    gate_id: str
    gate_type: ApprovalGateType
    title: str
    description: str
    proposal: Dict[str, Any]
    educational_explanation: str
    status: ApprovalStatus
    created_at: datetime
    approved_at: Optional[datetime]
    user_comments: Optional[str]
    modifications: Optional[Dict[str, Any]]


class ApprovalGateManager:
    """
    Manages approval gates throughout the workflow execution.
    
    This class handles the creation, management, and resolution of approval gates
    at key decision points in the classification workflow.
    """
    
    def __init__(self):
        """Initialize the approval gate manager."""
        self.active_gates: Dict[str, ApprovalGate] = {}
        self.completed_gates: List[ApprovalGate] = []
        self.gate_counter = 0
    
    def create_gate(
        self,
        gate_type: ApprovalGateType,
        title: str,
        description: str,
        proposal: Dict[str, Any],
        educational_explanation: str
    ) -> ApprovalGate:
        """
        Create a new approval gate.
        
        Args:
            gate_type: Type of approval gate
            title: Human-readable title for the gate
            description: Detailed description of what needs approval
            proposal: The proposal that needs approval
            educational_explanation: Educational content explaining the proposal
            
        Returns:
            Created approval gate
        """
        self.gate_counter += 1
        gate_id = f"gate_{self.gate_counter}_{gate_type.value}"
        
        gate: ApprovalGate = {
            "gate_id": gate_id,
            "gate_type": gate_type,
            "title": title,
            "description": description,
            "proposal": proposal,
            "educational_explanation": educational_explanation,
            "status": ApprovalStatus.PENDING,
            "created_at": datetime.now(),
            "approved_at": None,
            "user_comments": None,
            "modifications": None
        }
        
        self.active_gates[gate_id] = gate
        logger.info(f"Created approval gate: {gate_id} - {title}")
        
        return gate
    
    def approve_gate(
        self,
        gate_id: str,
        user_comments: Optional[str] = None
    ) -> bool:
        """
        Approve an active gate.
        
        Args:
            gate_id: ID of the gate to approve
            user_comments: Optional user comments
            
        Returns:
            True if approved successfully, False otherwise
        """
        if gate_id not in self.active_gates:
            logger.error(f"Gate {gate_id} not found in active gates")
            return False
        
        gate = self.active_gates[gate_id]
        gate["status"] = ApprovalStatus.APPROVED
        gate["approved_at"] = datetime.now()
        gate["user_comments"] = user_comments
        
        # Move to completed gates
        self.completed_gates.append(gate)
        del self.active_gates[gate_id]
        
        logger.info(f"Approved gate: {gate_id}")
        return True
    
    def reject_gate(
        self,
        gate_id: str,
        user_comments: str
    ) -> bool:
        """
        Reject an active gate.
        
        Args:
            gate_id: ID of the gate to reject
            user_comments: User comments explaining the rejection
            
        Returns:
            True if rejected successfully, False otherwise
        """
        if gate_id not in self.active_gates:
            logger.error(f"Gate {gate_id} not found in active gates")
            return False
        
        gate = self.active_gates[gate_id]
        gate["status"] = ApprovalStatus.REJECTED
        gate["approved_at"] = datetime.now()
        gate["user_comments"] = user_comments
        
        # Move to completed gates
        self.completed_gates.append(gate)
        del self.active_gates[gate_id]
        
        logger.info(f"Rejected gate: {gate_id}")
        return True
    
    def modify_gate(
        self,
        gate_id: str,
        modifications: Dict[str, Any],
        user_comments: str
    ) -> bool:
        """
        Modify an active gate with user-specified changes.
        
        Args:
            gate_id: ID of the gate to modify
            modifications: User-specified modifications
            user_comments: User comments explaining the modifications
            
        Returns:
            True if modified successfully, False otherwise
        """
        if gate_id not in self.active_gates:
            logger.error(f"Gate {gate_id} not found in active gates")
            return False
        
        gate = self.active_gates[gate_id]
        gate["status"] = ApprovalStatus.MODIFIED
        gate["approved_at"] = datetime.now()
        gate["user_comments"] = user_comments
        gate["modifications"] = modifications
        
        # Apply modifications to proposal
        gate["proposal"].update(modifications)
        
        # Move to completed gates
        self.completed_gates.append(gate)
        del self.active_gates[gate_id]
        
        logger.info(f"Modified gate: {gate_id}")
        return True
    
    def get_active_gates(self) -> List[ApprovalGate]:
        """Get all active approval gates."""
        return list(self.active_gates.values())
    
    def get_gate(self, gate_id: str) -> Optional[ApprovalGate]:
        """Get a specific gate by ID."""
        return self.active_gates.get(gate_id)
    
    def has_pending_gates(self) -> bool:
        """Check if there are any pending approval gates."""
        return len(self.active_gates) > 0


# Define the key decision points in the workflow
APPROVAL_GATE_DEFINITIONS = {
    ApprovalGateType.DATA_CLEANING_STRATEGY: {
        "title": "Data Cleaning Strategy Approval",
        "description": "Review and approve the proposed data cleaning strategy based on data quality analysis",
        "trigger_point": "After data discovery and quality analysis",
        "proposal_fields": [
            "missing_value_strategy",
            "outlier_handling_strategy", 
            "data_type_conversions",
            "duplicate_handling",
            "quality_thresholds"
        ],
        "educational_content": "Explains different data cleaning approaches and their impact on model performance"
    },
    
    ApprovalGateType.FEATURE_ENGINEERING_PLAN: {
        "title": "Feature Engineering Plan Approval",
        "description": "Review and approve the proposed feature engineering strategy",
        "trigger_point": "After EDA analysis and before feature engineering",
        "proposal_fields": [
            "feature_transformations",
            "new_feature_creation",
            "categorical_encoding_strategy",
            "scaling_normalization_approach",
            "feature_selection_criteria"
        ],
        "educational_content": "Explains feature engineering techniques and their impact on model performance"
    },
    
    ApprovalGateType.MODEL_SELECTION: {
        "title": "Model Selection Approval",
        "description": "Review and approve the proposed model selection based on data characteristics",
        "trigger_point": "After feature engineering and before model training",
        "proposal_fields": [
            "recommended_models",
            "model_selection_criteria",
            "cross_validation_strategy",
            "performance_metrics",
            "computational_requirements"
        ],
        "educational_content": "Explains different ML algorithms and their suitability for the specific problem"
    },
    
    ApprovalGateType.HYPERPARAMETER_TUNING: {
        "title": "Hyperparameter Tuning Strategy Approval",
        "description": "Review and approve the hyperparameter tuning approach",
        "trigger_point": "After model selection and before hyperparameter optimization",
        "proposal_fields": [
            "tuning_method",
            "parameter_ranges",
            "optimization_algorithm",
            "evaluation_metrics",
            "computational_budget"
        ],
        "educational_content": "Explains hyperparameter tuning techniques and their impact on model performance"
    },
    
    ApprovalGateType.FINAL_MODEL_APPROVAL: {
        "title": "Final Model Approval",
        "description": "Review and approve the final trained model before deployment",
        "trigger_point": "After model evaluation and before technical reporting",
        "proposal_fields": [
            "final_model_performance",
            "evaluation_metrics",
            "confusion_matrix_analysis",
            "feature_importance",
            "model_limitations",
            "deployment_recommendations"
        ],
        "educational_content": "Explains model evaluation metrics and what they mean for business impact"
    }
}


def get_approval_gate_definition(gate_type: ApprovalGateType) -> Dict[str, Any]:
    """
    Get the definition for a specific approval gate type.
    
    Args:
        gate_type: Type of approval gate
        
    Returns:
        Gate definition dictionary
    """
    return APPROVAL_GATE_DEFINITIONS.get(gate_type, {})


def should_trigger_approval_gate(
    current_agent: str,
    state: Dict[str, Any],
    gate_type: ApprovalGateType
) -> bool:
    """
    Determine if an approval gate should be triggered based on current state.
    
    Args:
        current_agent: Current agent being executed
        state: Current workflow state
        gate_type: Type of approval gate to check
        
    Returns:
        True if approval gate should be triggered
    """
    # Define trigger conditions for each gate type
    trigger_conditions = {
        ApprovalGateType.DATA_CLEANING_STRATEGY: (
            current_agent == "data_discovery" and
            state.get("agent_statuses", {}).get("data_discovery") == "COMPLETED" and
            state.get("data_quality_analysis") is not None
        ),
        
        ApprovalGateType.FEATURE_ENGINEERING_PLAN: (
            current_agent == "eda_analysis" and
            state.get("agent_statuses", {}).get("eda_analysis") == "COMPLETED" and
            state.get("statistical_summary") is not None
        ),
        
        ApprovalGateType.MODEL_SELECTION: (
            current_agent == "feature_engineering" and
            state.get("agent_statuses", {}).get("feature_engineering") == "COMPLETED" and
            state.get("engineered_features") is not None
        ),
        
        ApprovalGateType.HYPERPARAMETER_TUNING: (
            current_agent == "ml_building" and
            state.get("model_selection_results") is not None and
            state.get("best_model") is not None
        ),
        
        ApprovalGateType.FINAL_MODEL_APPROVAL: (
            current_agent == "model_evaluation" and
            state.get("agent_statuses", {}).get("model_evaluation") == "COMPLETED" and
            state.get("evaluation_metrics") is not None
        )
    }
    
    return trigger_conditions.get(gate_type, False)


def create_approval_proposal(
    gate_type: ApprovalGateType,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a proposal for an approval gate based on current state.
    
    Args:
        gate_type: Type of approval gate
        state: Current workflow state
        
    Returns:
        Proposal dictionary
    """
    proposals = {
        ApprovalGateType.DATA_CLEANING_STRATEGY: {
            "missing_value_strategy": state.get("missing_value_analysis", {}).get("recommended_strategy", "mean"),
            "outlier_handling_strategy": state.get("outlier_detection", {}).get("recommended_action", "remove"),
            "data_type_conversions": state.get("data_type_validation", {}).get("conversion_plan", {}),
            "duplicate_handling": "remove",
            "quality_thresholds": {
                "min_completeness": 0.8,
                "max_outlier_ratio": 0.05
            }
        },
        
        ApprovalGateType.FEATURE_ENGINEERING_PLAN: {
            "feature_transformations": state.get("eda_analysis", {}).get("transformation_recommendations", []),
            "new_feature_creation": state.get("eda_analysis", {}).get("feature_creation_ideas", []),
            "categorical_encoding_strategy": "one_hot",
            "scaling_normalization_approach": "standard",
            "feature_selection_criteria": "mutual_information"
        },
        
        ApprovalGateType.MODEL_SELECTION: {
            "recommended_models": state.get("model_selection_results", {}).get("candidate_models", []),
            "model_selection_criteria": "cross_validation_accuracy",
            "cross_validation_strategy": "stratified_kfold",
            "performance_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "computational_requirements": "moderate"
        },
        
        ApprovalGateType.HYPERPARAMETER_TUNING: {
            "tuning_method": "grid_search",
            "parameter_ranges": state.get("model_hyperparameters", {}),
            "optimization_algorithm": "exhaustive",
            "evaluation_metrics": ["accuracy"],
            "computational_budget": "moderate"
        },
        
        ApprovalGateType.FINAL_MODEL_APPROVAL: {
            "final_model_performance": state.get("evaluation_metrics", {}),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "confusion_matrix_analysis": state.get("confusion_matrix_data", {}),
            "feature_importance": state.get("feature_importance_model", {}),
            "model_limitations": state.get("model_performance_analysis", {}).get("limitations", []),
            "deployment_recommendations": ["monitor_performance", "retrain_quarterly"]
        }
    }
    
    return proposals.get(gate_type, {})


def generate_educational_explanation(
    gate_type: ApprovalGateType,
    proposal: Dict[str, Any],
    state: Dict[str, Any]
) -> str:
    """
    Generate educational explanation for an approval gate.
    
    Args:
        gate_type: Type of approval gate
        proposal: The proposal being made
        state: Current workflow state
        
    Returns:
        Educational explanation string
    """
    explanations = {
        ApprovalGateType.DATA_CLEANING_STRATEGY: f"""
## Data Cleaning Strategy Explanation

**What is data cleaning?**
Data cleaning is the process of identifying and correcting errors, inconsistencies, and missing values in your dataset. Clean data is essential for building accurate machine learning models.

**Why is this important?**
- Dirty data leads to poor model performance
- Missing values can bias your results
- Outliers can skew model training
- Inconsistent data types cause errors

**Your proposed strategy:**
- Missing values: {proposal.get('missing_value_strategy', 'N/A')}
- Outlier handling: {proposal.get('outlier_handling_strategy', 'N/A')}
- Data type conversions: {len(proposal.get('data_type_conversions', {}))} columns

**Impact on your model:**
This strategy will help ensure your model trains on high-quality data, leading to better predictions and more reliable results.
        """,
        
        ApprovalGateType.FEATURE_ENGINEERING_PLAN: f"""
## Feature Engineering Plan Explanation

**What is feature engineering?**
Feature engineering is the process of creating new features or transforming existing ones to improve model performance. It's often the most impactful step in machine learning.

**Why is this important?**
- Better features lead to better models
- Raw data often needs transformation
- Domain knowledge can be encoded in features
- Feature selection reduces overfitting

**Your proposed plan:**
- Transformations: {len(proposal.get('feature_transformations', []))} planned
- New features: {len(proposal.get('new_feature_creation', []))} to create
- Encoding strategy: {proposal.get('categorical_encoding_strategy', 'N/A')}
- Scaling approach: {proposal.get('scaling_normalization_approach', 'N/A')}

**Impact on your model:**
These transformations will help your model better understand patterns in the data and make more accurate predictions.
        """,
        
        ApprovalGateType.MODEL_SELECTION: f"""
## Model Selection Explanation

**What is model selection?**
Model selection is the process of choosing the best machine learning algorithm for your specific problem and data characteristics.

**Why is this important?**
- Different algorithms work better for different problems
- Some models are better with small datasets, others with large ones
- Computational requirements vary by algorithm
- Interpretability needs differ by use case

**Your proposed models:**
{', '.join(proposal.get('recommended_models', ['N/A']))}

**Selection criteria:**
- Method: {proposal.get('model_selection_criteria', 'N/A')}
- Cross-validation: {proposal.get('cross_validation_strategy', 'N/A')}
- Metrics: {', '.join(proposal.get('performance_metrics', ['N/A']))}

**Impact on your model:**
The right algorithm choice can significantly improve your model's performance and suitability for your specific use case.
        """,
        
        ApprovalGateType.HYPERPARAMETER_TUNING: f"""
## Hyperparameter Tuning Strategy Explanation

**What is hyperparameter tuning?**
Hyperparameter tuning is the process of finding the best configuration of model parameters that weren't learned during training.

**Why is this important?**
- Default parameters are rarely optimal
- Tuning can significantly improve performance
- Different problems need different parameter settings
- Prevents overfitting and underfitting

**Your proposed strategy:**
- Method: {proposal.get('tuning_method', 'N/A')}
- Parameter ranges: {len(proposal.get('parameter_ranges', {}))} parameters
- Optimization: {proposal.get('optimization_algorithm', 'N/A')}
- Budget: {proposal.get('computational_budget', 'N/A')}

**Impact on your model:**
Proper hyperparameter tuning can improve your model's accuracy by 5-15% and make it more robust to different data.
        """,
        
        ApprovalGateType.FINAL_MODEL_APPROVAL: f"""
## Final Model Approval Explanation

**What are we evaluating?**
This is your final trained model with all optimizations applied. We're evaluating its performance and readiness for deployment.

**Key performance metrics:**
- Accuracy: {proposal.get('final_model_performance', {}).get('accuracy', 'N/A')}
- Precision: {proposal.get('final_model_performance', {}).get('precision_macro', 'N/A')}
- Recall: {proposal.get('final_model_performance', {}).get('recall_macro', 'N/A')}
- F1-Score: {proposal.get('final_model_performance', {}).get('f1_macro', 'N/A')}

**Model characteristics:**
- Feature importance: {len(proposal.get('feature_importance', {}).get('feature_importance', []))} features analyzed
- Confusion matrix: Available for detailed analysis
- Limitations: {len(proposal.get('model_limitations', []))} identified

**Deployment readiness:**
This model is ready for production use with the recommended monitoring and maintenance procedures.
        """
    }
    
    return explanations.get(gate_type, "No explanation available.")
