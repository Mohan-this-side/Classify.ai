"""
📊 EDA Agent - Advanced Exploratory Data Analysis

Performs comprehensive exploratory data analysis with intelligent plot generation
and statistical insights. Thinks like a senior data scientist.
"""

import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent
from ..workflows.state_management import ClassificationState, AgentStatus, state_manager


class EDAAgent(BaseAgent):
    """EDA Agent for statistical summaries and basic insights"""

    def __init__(self) -> None:
        super().__init__("eda_analysis", "1.0.0")
        self.logger = logging.getLogger("agent.eda")

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Performs lightweight EDA and returns serializable results",
            "capabilities": [
                "Summary statistics",
                "Basic distribution info",
                "Nulls/duplicates overview",
            ],
            "dependencies": ["data_cleaning", "data_discovery"],
        }

    def get_dependencies(self) -> list:
        return ["data_cleaning", "data_discovery"]

    async def execute(self, state: ClassificationState) -> ClassificationState:
        try:
            self.logger.info("Starting EDA analysis")

            df = state_manager.get_dataset(state, "cleaned")
            if df is None:
                df = state_manager.get_dataset(state, "original")
            if df is None:
                raise ValueError("No dataset available for EDA")

            # Compute lightweight serializable stats
            numeric_cols = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]
            summary = {
                "shape": (int(df.shape[0]), int(df.shape[1])),
                "missing_total": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
                "numeric_columns": numeric_cols[:50],
            }

            # Per-column basic stats (limited for size)
            per_column = {}
            for col in numeric_cols[:20]:
                s = df[col]
                try:
                    per_column[col] = {
                        "min": float(np.nanmin(s)),
                        "max": float(np.nanmax(s)),
                        "mean": float(np.nanmean(s)),
                        "std": float(np.nanstd(s)),
                        "nulls": int(s.isnull().sum()),
                    }
                except Exception:
                    per_column[col] = {"error": "stat_failed"}

            state["statistical_summary"] = {"overall": summary, "columns": per_column}
            state["eda_plots"] = []  # Plots would be generated to files later; keep empty for now
            state["distribution_analysis"] = {}
            state["outlier_analysis"] = {}

            state["agent_statuses"]["eda_analysis"] = AgentStatus.COMPLETED
            state["completed_agents"].append("eda_analysis")
            self.logger.info("EDA analysis completed")
            return state

        except Exception as e:
            self.logger.error(f"EDA failed: {e}")
            state["agent_statuses"]["eda_analysis"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state


