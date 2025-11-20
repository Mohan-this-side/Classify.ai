"""
Reasoning Extractor

Extracts problem→solution mappings from agent state to demonstrate data scientist reasoning.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class AgentReasoningExtractor:
    """Extract reasoning and problem-solution mappings from agent state"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AgentReasoningExtractor")
    
    def extract_all_problems(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all 10 critical problems detected"""
        problems = {}
        
        # Get problem detection from EDA agent
        eda_results = state.get("target_relationships", {}) or {}
        problem_detection = state.get("problem_detection", {}) or {}
        
        # 1. Class Imbalance
        if "class_imbalance_analysis" in eda_results:
            problems["class_imbalance"] = eda_results["class_imbalance_analysis"]
        elif "class_balance" in eda_results:
            cb = eda_results["class_balance"]
            problems["class_imbalance"] = {
                "imbalance_detected": not cb.get("is_balanced", True),
                "ratio": cb.get("balance_ratio", 1.0),
                "majority_class": cb.get("majority_class", "unknown"),
                "minority_class": cb.get("minority_class", "unknown"),
                "recommendation": cb.get("recommendation", "Use SMOTE or class weights")
            }
        
        # 2. Missing Data
        missing_analysis = state.get("missing_analysis", {})
        if missing_analysis:
            problems["missing_data"] = {
                "detected": True,
                "total_missing": missing_analysis.get("missing_statistics", {}).get("total_missing", 0),
                "overall_percentage": missing_analysis.get("missing_statistics", {}).get("overall_missing_percentage", 0),
                "columns_affected": missing_analysis.get("missing_statistics", {}).get("columns_with_missing", 0)
            }
        
        # 3. Multicollinearity
        if "multicollinearity" in problem_detection:
            problems["multicollinearity"] = problem_detection["multicollinearity"]
        
        # 4. Data Leakage
        if "data_leakage" in problem_detection:
            problems["data_leakage"] = problem_detection["data_leakage"]
        
        # 5. Temporal Patterns
        if "temporal_patterns" in problem_detection:
            problems["temporal_patterns"] = problem_detection["temporal_patterns"]
        
        # 6. Location Missing Patterns
        if "location_missing_patterns" in problem_detection:
            problems["location_missing_patterns"] = problem_detection["location_missing_patterns"]
        
        # 7. Weak Feature Correlations
        if "weak_feature_correlations" in problem_detection:
            problems["weak_feature_correlations"] = problem_detection["weak_feature_correlations"]
        
        # 8. Outliers
        outlier_detection = state.get("outlier_detection", {})
        if outlier_detection:
            outlier_summary = outlier_detection.get("outlier_summary", {})
            if outlier_summary.get("total_outliers_detected", 0) > 0:
                problems["outliers"] = {
                    "detected": True,
                    "total_outliers": outlier_summary.get("total_outliers_detected", 0),
                    "outlier_percentage": outlier_summary.get("outlier_percentage", 0),
                    "columns_affected": outlier_summary.get("columns_with_outliers", 0)
                }
        
        # 9. Target Variable Missing
        target_missing = state.get("target_relationships", {}).get("missing_count", 0)
        if target_missing > 0:
            problems["target_missing"] = {
                "detected": True,
                "count": target_missing,
                "percentage": state.get("target_relationships", {}).get("missing_percentage", 0)
            }
        
        # 10. High Cardinality Categorical
        categorical_cols = state.get("categorical_columns", [])
        if categorical_cols:
            high_cardinality = []
            for col in categorical_cols:
                # Check if we have cardinality info
                if col in state.get("data_types", {}):
                    # Would need actual cardinality from dataset
                    pass
            if high_cardinality:
                problems["high_cardinality"] = {
                    "detected": True,
                    "columns": high_cardinality
                }
        
        return problems
    
    def extract_cleaning_reasoning(self, state: Dict[str, Any], agent_name: str = "Data Cleaning") -> Dict[str, Any]:
        """Extract cleaning reasoning: problem → solution → result"""
        reasoning = {}
        
        # Get cleaning reasoning from state
        cleaning_reasoning = state.get("cleaning_reasoning", {})
        
        if cleaning_reasoning:
            # Target nulls removal
            if "target_nulls" in cleaning_reasoning:
                reasoning["target_nulls"] = cleaning_reasoning["target_nulls"]
            
            # Missing value handling per column
            for col, col_reasoning in cleaning_reasoning.items():
                if col != "target_nulls" and col != "_location_patterns":
                    if isinstance(col_reasoning, dict) and "problem" in col_reasoning:
                        reasoning[col] = col_reasoning
            
            # Location patterns
            if "_location_patterns" in cleaning_reasoning:
                reasoning["_location_patterns"] = cleaning_reasoning["_location_patterns"]
        
        # Also check cleaning_actions_taken for summary
        cleaning_actions = state.get("cleaning_actions_taken", [])
        if cleaning_actions:
            reasoning["_summary"] = {
                "total_actions": len(cleaning_actions),
                "actions": cleaning_actions[:20]  # First 20 actions
            }
        
        return reasoning
    
    def extract_imbalance_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract imbalance detection and handling reasoning"""
        reasoning = {}
        
        # Detection from EDA
        target_relationships = state.get("target_relationships", {})
        if "class_imbalance_analysis" in target_relationships:
            reasoning["detection"] = target_relationships["class_imbalance_analysis"]
        
        # Handling from ML Builder
        imbalance_handling = state.get("imbalance_handling")
        if imbalance_handling:
            reasoning["handling"] = imbalance_handling
        
        # Model selection reasoning
        model_selection = state.get("model_selection_reasoning", {})
        if model_selection:
            reasoning["model_selection"] = model_selection
        
        return reasoning
    
    def extract_multicollinearity_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract multicollinearity detection and removal reasoning"""
        reasoning = {}
        
        # Detection from EDA
        problem_detection = state.get("problem_detection", {})
        if "multicollinearity" in problem_detection:
            reasoning["detection"] = problem_detection["multicollinearity"]
        
        # Removal from Feature Engineering
        feature_reasoning = state.get("feature_reasoning", {})
        if feature_reasoning:
            removed_features = feature_reasoning.get("removed", [])
            if removed_features:
                reasoning["removal"] = {
                    "removed_features": removed_features,
                    "count": len(removed_features)
                }
        
        return reasoning
    
    def extract_feature_engineering_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feature creation/removal reasoning"""
        reasoning = {}
        
        feature_reasoning = state.get("feature_reasoning", {})
        if feature_reasoning:
            # Removed features
            if "removed" in feature_reasoning:
                reasoning["removed"] = feature_reasoning["removed"]
            
            # Created interactions
            if "created_interactions" in feature_reasoning:
                reasoning["interactions"] = feature_reasoning["created_interactions"]
            
            # Temporal features
            if "temporal_features" in feature_reasoning:
                reasoning["temporal"] = feature_reasoning["temporal_features"]
            
            # Cyclical encoding
            if "cyclical_encoding" in feature_reasoning:
                reasoning["cyclical"] = feature_reasoning["cyclical_encoding"]
            
            # Location encoding
            if "location_encoding" in feature_reasoning:
                reasoning["location"] = feature_reasoning["location_encoding"]
        
        return reasoning
    
    def extract_temporal_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal pattern detection and split reasoning"""
        reasoning = {}
        
        # Detection from EDA
        problem_detection = state.get("problem_detection", {})
        if "temporal_patterns" in problem_detection:
            reasoning["detection"] = problem_detection["temporal_patterns"]
        
        # Split method from ML Builder
        temporal_split = state.get("temporal_split_info", {})
        if temporal_split:
            reasoning["split_method"] = temporal_split
        
        return reasoning
    
    def extract_data_leakage_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data leakage detection reasoning"""
        reasoning = {}
        
        problem_detection = state.get("problem_detection", {})
        if "data_leakage" in problem_detection:
            reasoning = problem_detection["data_leakage"]
        
        return reasoning
    
    def extract_location_patterns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location-specific missing pattern analysis"""
        reasoning = {}
        
        problem_detection = state.get("problem_detection", {})
        if "location_missing_patterns" in problem_detection:
            reasoning = problem_detection["location_missing_patterns"]
        
        # Also check cleaning reasoning
        cleaning_reasoning = state.get("cleaning_reasoning", {})
        if "_location_patterns" in cleaning_reasoning:
            reasoning["cleaning_handling"] = cleaning_reasoning["_location_patterns"]
        
        return reasoning
    
    def extract_model_selection_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model selection reasoning"""
        reasoning = {}
        
        model_selection = state.get("model_selection_reasoning", {})
        if model_selection:
            reasoning = model_selection
        
        # Also get from model_selection_results
        model_results = state.get("model_selection_results", {})
        if model_results:
            reasoning["selected_model"] = model_results.get("selected_model")
            reasoning["best_parameters"] = model_results.get("best_parameters")
        
        # Data analysis that influenced selection
        data_analysis = state.get("data_analysis", {})
        if data_analysis:
            reasoning["data_characteristics"] = {
                "is_balanced": data_analysis.get("is_balanced", True),
                "complexity_score": data_analysis.get("complexity_score", 0.5),
                "missing_percentage": data_analysis.get("missing_percentage", 0),
                "recommended_models": data_analysis.get("recommended_models", [])
            }
        
        return reasoning
    
    def extract_comprehensive_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all reasoning in one comprehensive dictionary"""
        return {
            "all_problems": self.extract_all_problems(state),
            "cleaning": self.extract_cleaning_reasoning(state),
            "imbalance": self.extract_imbalance_reasoning(state),
            "multicollinearity": self.extract_multicollinearity_reasoning(state),
            "feature_engineering": self.extract_feature_engineering_reasoning(state),
            "temporal": self.extract_temporal_reasoning(state),
            "data_leakage": self.extract_data_leakage_reasoning(state),
            "location_patterns": self.extract_location_patterns(state),
            "model_selection": self.extract_model_selection_reasoning(state)
        }

