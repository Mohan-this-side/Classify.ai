"""
🔎 Data Discovery Agent

Finds similar public datasets, best practices, and prior work to guide downstream steps.
Returns lightweight, serializable insights only.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent
from ..workflows.state_management import ClassificationState, AgentStatus, state_manager


class DataDiscoveryAgent(BaseAgent):
    """Data Discovery Agent for researching similar datasets and approaches"""

    def __init__(self) -> None:
        super().__init__("data_discovery", "1.0.0")
        self.logger = logging.getLogger("agent.data_discovery")

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Finds similar datasets and best practices to inform pipeline",
            "capabilities": [
                "Search similar datasets",
                "Collect best practices",
                "Summarize recommendations",
            ],
            "dependencies": ["data_cleaning"],
        }

    def get_dependencies(self) -> list:
        return ["data_cleaning"]

    async def execute(self, state: ClassificationState) -> ClassificationState:
        try:
            self.logger.info("Starting data discovery")

            # Access cleaned dataset via state manager (do not store in state)
            cleaned = state_manager.get_dataset(state, "cleaned")
            if cleaned is None:
                cleaned = state_manager.get_dataset(state, "original")

            # Lightweight heuristics/placeholder results
            recommendations: List[str] = []
            if cleaned is not None:
                n_rows, n_cols = cleaned.shape
                if n_cols <= 10:
                    recommendations.append(
                        "Small feature space: consider tree-based models and strong validation"
                    )
                if n_rows < 1000:
                    recommendations.append(
                        "Dataset is relatively small: prefer simpler models and careful cross-validation"
                    )

            state["discovery_results"] = {
                "similar_datasets": [],
                "research_papers": [],
                "best_practices": [
                    "Standardize numeric features if using linear models",
                    "Balance classes if target is imbalanced",
                ],
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat(),
            }

            # Mark completed
            state["agent_statuses"]["data_discovery"] = AgentStatus.COMPLETED
            state["completed_agents"].append("data_discovery")
            self.logger.info("Data discovery completed")
            return state

        except Exception as e:
            self.logger.error(f"Data discovery failed: {e}")
            state["agent_statuses"]["data_discovery"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state


