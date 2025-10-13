"""
ML Model Builder Agent

This agent is responsible for:
- Model selection based on data characteristics
- Hyperparameter tuning
- Cross-validation
- Model training and optimization
- Model persistence
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .base_agent import BaseAgent
from ..workflows.state_management import AgentStatus, ClassificationState, state_manager


class MLBuilderAgent(BaseAgent):
    """ML Model Builder Agent for training and optimizing classification models"""

    def __init__(self):
        super().__init__("ml_builder", "1.0.0")
        self.logger = logging.getLogger("agent.ml_builder")

        # Candidate models
        self.models: Dict[str, Any] = {
            "random_forest": RandomForestClassifier(random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "svm": SVC(random_state=42, probability=True),
            "knn": KNeighborsClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "xgboost": XGBClassifier(random_state=42, eval_metric="logloss"),
            "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
        }

        # Hyperparameter grids
        self.param_grids: Dict[str, Dict[str, List[Any]]] = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
            "logistic_regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            },
            "svm": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
            },
        }

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Trains and optimizes ML models",
            "capabilities": [
                "Model selection",
                "Hyperparameter tuning",
                "Cross-validation",
                "Model persistence",
            ],
            "supported_models": list(self.models.keys()),
            "dependencies": ["data_cleaning", "feature_engineering"],
        }

    def get_dependencies(self) -> list:
        return ["data_cleaning", "feature_engineering"]

    async def execute(self, state: ClassificationState) -> ClassificationState:
        try:
            self.logger.info("Starting ML model building process")

            cleaned_df = state_manager.get_dataset(state, "cleaned")
            if cleaned_df is None:
                cleaned_df = state_manager.get_dataset(state, "original")
            if cleaned_df is None:
                raise ValueError("No cleaned dataset available")

            target_column = state.get("target_column")
            if not target_column:
                raise ValueError("No target column specified")

            X, y = self._prepare_data(cleaned_df, target_column)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            best_model_name = self._select_best_model(X_train, y_train)
            self.logger.info(f"Selected best model: {best_model_name}")

            best_model, best_params = self._tune_hyperparameters(
                best_model_name, X_train, y_train
            )

            best_model.fit(X_train, y_train)

            train_score = best_model.score(X_train, y_train)
            test_score = best_model.score(X_test, y_test)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

            y_pred = best_model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)

            model_path = self._save_model(best_model, state.get("session_id"))

            state["model_selection_results"] = {
                "selected_model": best_model_name,
                "best_parameters": best_params,
                "model_path": model_path,
            }
            state["best_model"] = best_model_name
            state["model_hyperparameters"] = best_params
            state["training_metrics"] = {
                "train_accuracy": float(train_score),
                "test_accuracy": float(test_score),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
            }
            state["cross_validation_scores"] = {
                "scores": cv_scores.tolist(),
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
            }
            state["evaluation_metrics"] = metrics
            state["model_explanation"] = self._generate_model_explanation(
                best_model_name, best_params, metrics
            )

            state["agent_statuses"]["ml_building"] = AgentStatus.COMPLETED
            state["completed_agents"].append("ml_building")

            self.logger.info("ML model building completed successfully")
            return state

        except Exception as e:
            self.logger.error(f"Error in ML model building: {str(e)}")
            state["agent_statuses"]["ml_building"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state

    def _prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X = pd.get_dummies(X, drop_first=True)
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        return X, y

    def _select_best_model(self, X: pd.DataFrame, y: pd.Series) -> str:
        scores: Dict[str, float] = {}
        for name, model in self.models.items():
            try:
                Xs = X.sample(n=min(1000, len(X)), random_state=42)
                ys = y.loc[Xs.index]
                cv = cross_val_score(model, Xs, ys, cv=3, scoring="accuracy")
                scores[name] = cv.mean()
            except Exception as e:
                self.logger.warning(f"Quick eval failed for {name}: {e}")
                scores[name] = 0.0
        best = max(scores, key=scores.get)
        self.logger.info(f"Model quick scores: {scores}")
        return best

    def _tune_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        if model_name not in self.param_grids:
            return self.models[model_name], {}
        model = self.models[model_name]
        grid = self.param_grids[model_name]
        gs = GridSearchCV(model, grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        self.logger.info(f"Best params for {model_name}: {gs.best_params_} (score={gs.best_score_:.4f})")
        return gs.best_estimator_, gs.best_params_

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted")),
            "recall": float(recall_score(y_true, y_pred, average="weighted")),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        }

    def _save_model(self, model: Any, session_id: Optional[str]) -> str:
        try:
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = session_id or "session"
            path = os.path.join(models_dir, f"model_{sid}_{ts}.joblib")
            joblib.dump(model, path)
            self.logger.info(f"Model saved to: {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return ""

    def _generate_model_explanation(self, model_name: str, params: Dict, metrics: Dict) -> str:
        return (
            f"Selected Model: {model_name}\n"
            f"Best Parameters: {params}\n"
            f"Accuracy: {metrics.get('accuracy', 0):.4f}, "
            f"Precision: {metrics.get('precision', 0):.4f}, "
            f"Recall: {metrics.get('recall', 0):.4f}, "
            f"F1: {metrics.get('f1_score', 0):.4f}"
        )


