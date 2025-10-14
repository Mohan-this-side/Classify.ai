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

    async def execute(self, state: ClassificationState) -> ClassificationState:
        try:
            self.logger.info("Starting feature engineering")

            df = state_manager.get_dataset(state, "cleaned")
            if df is None:
                df = state_manager.get_dataset(state, "original")
            if df is None:
                raise ValueError("No dataset available for feature engineering")

            target = state.get("target_column")
            if not target or target not in df.columns:
                raise ValueError("Target column missing for feature engineering")

            # Simple, safe feature operations (do not mutate original stored df until done)
            fe_df = df.copy()

            # Example: create interaction features for top 2 numeric columns (if exist)
            numeric_cols = list(fe_df.select_dtypes(include=["number"]).columns)
            created: List[str] = []
            if len(numeric_cols) >= 2:
                a, b = numeric_cols[0], numeric_cols[1]
                new_col = f"{a}_x_{b}"
                fe_df[new_col] = fe_df[a] * fe_df[b]
                created.append(new_col)

            # Record preprocessing steps and selections (no DataFrame in state)
            state["engineered_features"] = created
            state["feature_transformations"] = {
                "one_hot": "applied in ML builder via pd.get_dummies",
                "interactions": created,
            }
            state["feature_selection_results"] = {
                "method": "heuristic",
                "kept_features": [c for c in fe_df.columns if c != target][:100],
            }

            # Store engineered dataset for downstream (as cleaned replacement)
            state_manager.store_dataset(state, fe_df, "cleaned")

            state["agent_statuses"]["feature_engineering"] = AgentStatus.COMPLETED
            state["completed_agents"].append("feature_engineering")
            self.logger.info("Feature engineering completed")
            return state

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            state["agent_statuses"]["feature_engineering"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state


