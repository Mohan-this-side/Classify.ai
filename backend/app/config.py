"""
🔧 Configuration Management for DS Capstone Project

This module handles all configuration settings, environment variables,
and application constants for the multi-agent classification system.
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create global settings instance
settings = None

def get_settings() -> 'Settings':
    """Get application settings instance"""
    global settings
    if settings is None:
        settings = Settings()
    return settings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application settings
    app_name: str = Field(default="DS Capstone Multi-Agent System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Keys
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Database settings
    database_url: str = Field(default="postgresql://user:password@localhost/ds_capstone", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Celery settings
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # File storage
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    
    # ML settings
    max_models_to_try: int = Field(default=5, env="MAX_MODELS_TO_TRY")
    cross_validation_folds: int = Field(default=5, env="CV_FOLDS")
    test_size: float = Field(default=0.2, env="TEST_SIZE")
    random_state: int = Field(default=42, env="RANDOM_STATE")
    
    # Agent settings
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    timeout_seconds: int = Field(default=300, env="TIMEOUT_SECONDS")
    enable_parallel_processing: bool = Field(default=True, env="ENABLE_PARALLEL_PROCESSING")
    
    # LangSmith settings
    langsmith_project: str = Field(default="ds-capstone-classification", env="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, env="LANGSMITH_TRACING")
    
    # Model settings
    default_model: str = Field(default="gemini-2.0-flash-exp", env="DEFAULT_MODEL")
    model_temperature: float = Field(default=0.1, env="MODEL_TEMPERATURE")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    
    # WebSocket settings
    websocket_heartbeat_interval: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    max_websocket_connections: int = Field(default=100, env="MAX_WS_CONNECTIONS")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment

# Global settings instance
settings = Settings()

# Agent configuration
AGENT_CONFIG = {
    "data_cleaning": {
        "name": "Data Cleaning Agent",
        "description": "Cleans and preprocesses datasets for ML",
        "timeout": 300,
        "retries": 3,
        "dependencies": []
    },
    "data_discovery": {
        "name": "Data Discovery Agent", 
        "description": "Researches similar datasets and approaches",
        "timeout": 180,
        "retries": 2,
        "dependencies": ["data_cleaning"]
    },
    "eda_analysis": {
        "name": "EDA Agent",
        "description": "Performs exploratory data analysis",
        "timeout": 240,
        "retries": 2,
        "dependencies": ["data_cleaning", "data_discovery"]
    },
    "feature_engineering": {
        "name": "Feature Engineering Agent",
        "description": "Creates and selects features for ML",
        "timeout": 300,
        "retries": 3,
        "dependencies": ["eda_analysis"]
    },
    "ml_building": {
        "name": "ML Model Builder Agent",
        "description": "Trains and optimizes ML models",
        "timeout": 600,
        "retries": 2,
        "dependencies": ["feature_engineering"]
    },
    "model_evaluation": {
        "name": "Model Evaluation Agent",
        "description": "Evaluates model performance",
        "timeout": 180,
        "retries": 2,
        "dependencies": ["ml_building"]
    },
    "technical_reporting": {
        "name": "Technical Reporter Agent",
        "description": "Generates comprehensive reports",
        "timeout": 120,
        "retries": 2,
        "dependencies": ["model_evaluation"]
    },
    "project_manager": {
        "name": "Project Manager Agent",
        "description": "Orchestrates the entire workflow",
        "timeout": 0,  # No timeout for manager
        "retries": 0,
        "dependencies": []
    }
}

# ML Model configurations
ML_MODELS = {
    "classification": {
        "sklearn": [
            "LogisticRegression",
            "RandomForestClassifier", 
            "GradientBoostingClassifier",
            "SVC",
            "KNeighborsClassifier"
        ],
        "xgboost": ["XGBClassifier"],
        "lightgbm": ["LGBMClassifier"]
    },
    "regression": {
        "sklearn": [
            "LinearRegression",
            "RandomForestRegressor",
            "GradientBoostingRegressor", 
            "SVR",
            "KNeighborsRegressor"
        ],
        "xgboost": ["XGBRegressor"],
        "lightgbm": ["LGBMRegressor"]
    }
}

# Feature engineering strategies
FEATURE_ENGINEERING_STRATEGIES = {
    "categorical": [
        "one_hot_encoding",
        "label_encoding", 
        "target_encoding",
        "frequency_encoding"
    ],
    "numerical": [
        "standardization",
        "normalization",
        "log_transformation",
        "polynomial_features"
    ],
    "text": [
        "tfidf",
        "count_vectorizer",
        "word_embeddings"
    ]
}

# Evaluation metrics
EVALUATION_METRICS = {
    "classification": [
        "accuracy",
        "precision",
        "recall", 
        "f1_score",
        "roc_auc",
        "confusion_matrix"
    ],
    "regression": [
        "mse",
        "rmse",
        "mae",
        "r2_score"
    ]
}

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent"""
    return AGENT_CONFIG.get(agent_name, {})

def get_ml_models(task_type: str) -> Dict[str, list]:
    """Get available ML models for a task type"""
    return ML_MODELS.get(task_type, {})

def get_feature_strategies() -> Dict[str, list]:
    """Get available feature engineering strategies"""
    return FEATURE_ENGINEERING_STRATEGIES

def get_evaluation_metrics(task_type: str) -> list:
    """Get evaluation metrics for a task type"""
    return EVALUATION_METRICS.get(task_type, [])

# Validation functions
def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are present"""
    s = get_settings()
    return {
        "google_api_key": bool(s.google_api_key),
        "langsmith_api_key": bool(s.langsmith_api_key),
        "openai_api_key": bool(s.openai_api_key)
    }

def get_database_config() -> Dict[str, str]:
    """Get database configuration"""
    s = get_settings()
    return {
        "database_url": s.database_url,
        "redis_url": s.redis_url
    }

def get_celery_config() -> Dict[str, str]:
    """Get Celery configuration"""
    s = get_settings()
    return {
        "broker_url": s.celery_broker_url,
        "result_backend": s.celery_result_backend
    }
