"""
🧱 Feature Engineering Agent

Creates/selects features and records preprocessing steps in a serializable way.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

from ..base_agent import BaseAgent
from ...workflows.state_management import ClassificationState, AgentStatus, state_manager


class FeatureEngineeringAgent(BaseAgent):
    """Feature Engineering Agent for creating/selecting features"""

    def __init__(self) -> None:
        super().__init__("feature_engineering", "1.0.0")
        self.logger = logging.getLogger("agent.feature_engineering")

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Generates new features and selects useful ones",
            "capabilities": [
                "Simple feature creation",
                "One-hot encoding",
                "Basic selection",
            ],
            "dependencies": ["eda_analysis"],
        }

    def get_dependencies(self) -> list:
        return ["eda_analysis"]

    
    async def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """
        LAYER 1: Perform basic feature engineering (hardcoded, reliable).
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary containing Layer 1 feature engineering results
        """
        self.logger.info("🔍 LAYER 1: Performing basic feature engineering")
        
        # Get cleaned dataset
        df = state_manager.get_dataset(state, "cleaned")
        if df is None:
            df = state_manager.get_dataset(state, "original")
        if df is None:
            raise ValueError("No dataset available for feature engineering")
        
        target = state.get("target_column")
        if not target:
            # Try alternative key names
            target = state.get("target_col") or state.get("target")
        
        if not target:
            raise ValueError("Target column not specified in state")
        
        if target not in df.columns:
            self.logger.warning(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")
            raise ValueError(f"Target column '{target}' not found in dataset")
        
        # Create a copy to avoid modifying original
        fe_df = df.copy()
        
        # Analyze data types
        numeric_cols = list(fe_df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(fe_df.select_dtypes(exclude=["number"]).columns)
        
        # Remove target from numeric/categorical lists
        if target in numeric_cols:
            numeric_cols.remove(target)
        if target in categorical_cols:
            categorical_cols.remove(target)
        
        # LAYER 1: Create basic interaction features
        created: List[str] = []
        if len(numeric_cols) >= 2:
            # Create interaction feature for top 2 numeric columns
            a, b = numeric_cols[0], numeric_cols[1]
            new_col = f"{a}_x_{b}"
            fe_df[new_col] = fe_df[a] * fe_df[b]
            created.append(new_col)
        
        # Store engineered dataset for downstream
        state_manager.store_dataset(state, fe_df, "cleaned")
        
        # Prepare results
        results = {
            "engineered_features": created,
            "feature_transformations": {
                "one_hot": "applied in ML builder via pd.get_dummies",
                "interactions": created,
            },
            "feature_selection_results": {
                "method": "heuristic",
                "kept_features": [c for c in fe_df.columns if c != target][:100],
            },
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": target,
            "total_features_before": len(df.columns),
            "total_features_after": len(fe_df.columns),
            "features_created": len(created),
        }
        
        self.logger.info(f"✅ LAYER 1: Feature engineering complete - Created {len(created)} features")
        return results
    
    def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
        """
        LAYER 2: Generate prompt for LLM to create advanced feature engineering code.
        
        Args:
            layer1_results: Results from Layer 1 analysis
            state: Current workflow state
            
        Returns:
            Prompt string for LLM code generation
        """
        self.logger.info("🔧 LAYER 2: Generating LLM code generation prompt for feature engineering")
        
        numeric_cols = layer1_results.get("numeric_columns", [])
        categorical_cols = layer1_results.get("categorical_columns", [])
        target = layer1_results.get("target_column", "")
        
        prompt = f"""Generate advanced Python code for feature engineering based on the following analysis:

## Current Features:
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Target column: {target}
- Total features before: {layer1_results.get('total_features_before', 0)}

## Opportunities Identified:
{layer1_results.get('feature_engineering_opportunities', [])}

## Requirements for Generated Code:
1. Create meaningful interaction features from numeric columns
2. Apply one-hot encoding to categorical columns with low cardinality
3. Create polynomial features if relationships are non-linear
4. Handle missing values in new features
5. Keep feature names descriptive and clear
6. Ensure no data leakage (don't use target in features)
7. Use only: pandas, numpy, sklearn
8. Add comments explaining each transformation
9. Return engineered DataFrame and list of new feature names

Generate comprehensive, production-ready Python code:"""
        
        return prompt
    
    def process_sandbox_results(
        self,
        sandbox_output: Dict[str, Any],
        layer1_results: Dict[str, Any],
        state: ClassificationState
    ) -> Dict[str, Any]:
        """
        LAYER 2: Process and validate sandbox execution results for feature engineering.
        
        Args:
            sandbox_output: Raw output from sandbox execution
            layer1_results: Results from Layer 1 (for comparison)
            state: Current workflow state
            
        Returns:
            Processed and validated feature engineering results
        """
        self.logger.info("🔍 LAYER 2: Processing sandbox results for feature engineering")
        
        # Validate sandbox execution was successful
        if sandbox_output.get("status") != "SUCCESS":
            raise ValueError(f"Sandbox execution failed: {sandbox_output.get('error', 'Unknown error')}")
        
        # Extract engineered features from sandbox output
        engineered_data = sandbox_output.get("output", {})
        
        # Validate the output structure
        if not isinstance(engineered_data, dict):
            raise ValueError("Sandbox output should contain engineered features")
        
        result = {
            "engineered_data": engineered_data,
            "layer2_success": True,
            "sandbox_execution_time": sandbox_output.get("execution_time", 0)
        }
        
        self.logger.info("✅ LAYER 2: Sandbox results processed and validated")
        return result

