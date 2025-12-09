"""
🧱 Feature Engineering Agent

Creates/selects features and records preprocessing steps in a serializable way.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

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
        
        # Case-insensitive matching for target column
        target_lower = target.lower()
        matching_cols = [col for col in df.columns if col.lower() == target_lower]
        if not matching_cols:
            self.logger.warning(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")
            raise ValueError(f"Target column '{target}' not found in dataset")
        # Use the actual column name (preserve original case)
        target = matching_cols[0]
        if target != state.get("target_column"):
            self.logger.info(f"Using case-matched target column: '{target}' (requested: '{state.get('target_column')}')")
        
        # Create a copy to avoid modifying original
        fe_df = df.copy()
        feature_reasoning = {}  # Track feature creation/removal reasoning
        
        # Analyze data types
        numeric_cols = list(fe_df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(fe_df.select_dtypes(exclude=["number"]).columns)
        
        # Remove target from numeric/categorical lists
        if target in numeric_cols:
            numeric_cols.remove(target)
        if target in categorical_cols:
            categorical_cols.remove(target)
        
        # CRITICAL: Detect and remove multicollinearity (correlation >0.95)
        removed_features = []
        if len(numeric_cols) > 1:
            corr_matrix = fe_df[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.95:  # High multicollinearity threshold
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_val))
            
            # Remove one feature from each high-correlation pair
            # Keep the feature with higher correlation to target (if available)
            for col1, col2, corr_val in high_corr_pairs:
                if col1 in fe_df.columns and col2 in fe_df.columns:
                    # Check correlation with target to decide which to keep
                    if target in fe_df.columns:
                        try:
                            corr1_target = abs(fe_df[col1].corr(fe_df[target]))
                            corr2_target = abs(fe_df[col2].corr(fe_df[target]))
                            keep_col = col1 if corr1_target >= corr2_target else col2
                            remove_col = col2 if keep_col == col1 else col1
                        except:
                            # If correlation calculation fails, remove the second one
                            keep_col = col1
                            remove_col = col2
                    else:
                        # No target available, remove the second one
                        keep_col = col1
                        remove_col = col2
                    
                    if remove_col in fe_df.columns:
                        fe_df = fe_df.drop(columns=[remove_col])
                        removed_features.append({
                            "feature": remove_col,
                            "reason": f"High correlation ({corr_val:.3f}) with {keep_col}",
                            "correlation": float(corr_val),
                            "kept_feature": keep_col
                        })
                        if remove_col in numeric_cols:
                            numeric_cols.remove(remove_col)
                        self.logger.info(f"✅ Removed {remove_col} (correlation {corr_val:.3f} with {keep_col})")
        
        # CRITICAL: Create temporal features from Date column if available
        temporal_features_created = []
        if "Date" in fe_df.columns:
            try:
                fe_df["Date"] = pd.to_datetime(fe_df["Date"])
                fe_df["month"] = fe_df["Date"].dt.month
                fe_df["day_of_year"] = fe_df["Date"].dt.dayofyear
                fe_df["season"] = fe_df["Date"].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
                temporal_features_created = ["month", "day_of_year", "season"]
                feature_reasoning["temporal_features"] = {
                    "created": temporal_features_created,
                    "reason": "Extracted temporal patterns from Date column (month, day_of_year, season)",
                    "rationale": "Temporal features help capture seasonal and cyclical patterns in time-series data"
                }
                self.logger.info(f"✅ Created temporal features: {temporal_features_created}")
            except Exception as e:
                self.logger.warning(f"Failed to create temporal features: {e}")
        
        # CRITICAL: Cyclical encoding for wind directions (circular nature)
        wind_direction_cols = [col for col in categorical_cols if "Wind" in col and "Dir" in col]
        cyclical_encoded = []
        if wind_direction_cols:
            # Wind directions: N=0, NNE=22.5, NE=45, etc. (16 directions = 360/16 = 22.5 degrees each)
            wind_mapping = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            
            for col in wind_direction_cols:
                if col in fe_df.columns:
                    # Convert to radians for cyclical encoding
                    wind_angles = fe_df[col].map(wind_mapping).fillna(0) / 180 * np.pi
                    fe_df[f"{col}_sin"] = np.sin(wind_angles)
                    fe_df[f"{col}_cos"] = np.cos(wind_angles)
                    cyclical_encoded.extend([f"{col}_sin", f"{col}_cos"])
                    # Optionally drop original column (or keep both)
                    # fe_df = fe_df.drop(columns=[col])
            
            if cyclical_encoded:
                feature_reasoning["cyclical_encoding"] = {
                    "columns_encoded": wind_direction_cols,
                    "created_features": cyclical_encoded,
                    "reason": "Wind directions have circular nature, converted to sin/cos encoding",
                    "rationale": "Preserves circular relationships (N is close to NNW)"
                }
                self.logger.info(f"✅ Created cyclical encoding for wind directions: {cyclical_encoded}")
        
        # CRITICAL: Target encoding for high cardinality categorical columns
        high_cardinality_cols = []
        for col in categorical_cols:
            if col in fe_df.columns and fe_df[col].nunique() > 20:  # High cardinality threshold
                high_cardinality_cols.append({
                    "column": col,
                    "cardinality": int(fe_df[col].nunique()),
                    "encoding_method": "target_encoding",
                    "reason": f"High cardinality ({fe_df[col].nunique()} values), target encoding preserves information",
                    "rationale": f"One-hot encoding would create {fe_df[col].nunique()} features, target encoding is more efficient"
                })
                self.logger.info(f"✅ Will use target encoding for {col} ({fe_df[col].nunique()} values)")
        
        if high_cardinality_cols:
            feature_reasoning["high_cardinality_encoding"] = high_cardinality_cols
        
        # Create interaction features based on correlation analysis
        created: List[str] = []
        interaction_reasoning = {}
        
        # Strategy: Create interactions between highly correlated numeric features
        # This captures multiplicative relationships that might be predictive
        if len(numeric_cols) >= 2:
            # Find pairs of numeric features with moderate correlation (0.3-0.7)
            # Very high correlation (>0.7) suggests redundancy, very low (<0.3) suggests independence
            corr_matrix = fe_df[numeric_cols].corr().abs()
            interaction_candidates = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if 0.3 <= corr_val <= 0.7:  # Moderate correlation - good for interactions
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        interaction_candidates.append((col1, col2, corr_val))
            
            # Sort by correlation strength and create top interactions
            interaction_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Create interactions for top 3 pairs (or fewer if not enough candidates)
            for col1, col2, corr_val in interaction_candidates[:3]:
                interaction_col = f"{col1}_x_{col2}"
                fe_df[interaction_col] = fe_df[col1] * fe_df[col2]
                created.append(interaction_col)
                
                # Calculate correlation with target
                corr_with_target = None
                if target in fe_df.columns:
                    try:
                        corr_with_target = float(abs(fe_df[interaction_col].corr(fe_df[target])))
                    except:
                        pass
                
                interaction_reasoning[interaction_col] = {
                    "features": [col1, col2],
                    "base_correlation": float(corr_val),
                    "correlation_with_target": corr_with_target,
                    "reason": f"Interaction of {col1} and {col2} (correlation: {corr_val:.3f}) captures multiplicative relationship"
                }
                target_corr_str = f"{corr_with_target:.3f}" if corr_with_target is not None else "N/A"
                self.logger.info(f"✅ Created interaction feature: {interaction_col} (base corr: {corr_val:.3f}, target corr: {target_corr_str})")
        
        # Fallback: Create basic interaction for top 2 numeric columns if no correlation-based interactions
        if len(created) == 0 and len(numeric_cols) >= 2:
            a, b = numeric_cols[0], numeric_cols[1]
            new_col = f"{a}_x_{b}"
            fe_df[new_col] = fe_df[a] * fe_df[b]
            created.append(new_col)
            
            corr_with_target = None
            if target in fe_df.columns:
                try:
                    corr_with_target = float(abs(fe_df[new_col].corr(fe_df[target])))
                except:
                    pass
            
            interaction_reasoning[new_col] = {
                "features": [a, b],
                "correlation_with_target": corr_with_target,
                "reason": "Interaction of top 2 numeric features"
            }
        
        # Store engineered dataset for downstream
        state_manager.store_dataset(state, fe_df, "cleaned")
        
        # Prepare results with reasoning
        results = {
            "engineered_features": created + temporal_features_created + cyclical_encoded,
            "feature_transformations": {
                "one_hot": "applied in ML builder via pd.get_dummies",
                "interactions": created,
                "temporal": temporal_features_created,
                "cyclical": cyclical_encoded,
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
            "features_created": len(created) + len(temporal_features_created) + len(cyclical_encoded),
            "removed_features": removed_features,
            "feature_reasoning": {
                "removed": removed_features,
                "created_interactions": interaction_reasoning,
                "temporal_features": feature_reasoning.get("temporal_features", {}),
                "cyclical_encoding": feature_reasoning.get("cyclical_encoding", {}),
                "high_cardinality_encoding": feature_reasoning.get("high_cardinality_encoding", [])
            }
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

