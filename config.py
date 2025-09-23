"""
🔧 LangChain Ecosystem Configuration
=================================

This file manages all API keys, environment variables, and configuration
for LangChain, LangGraph, and LangSmith integration.

🔑 Required API Keys:
- GOOGLE_API_KEY: For Gemini models via LangChain
- LANGCHAIN_API_KEY: For LangSmith tracing and monitoring
- OPENAI_API_KEY: (Optional) For OpenAI models comparison

📊 LangSmith Features Enabled:
- Real-time tracing of agent decisions
- Performance monitoring and optimization
- Error tracking and debugging
- Custom evaluation pipelines
- Dashboard visualization

🎯 Where to Get Your API Keys:
1. Google Gemini: https://ai.google.dev/
2. LangSmith: https://smith.langchain.com/ 
3. OpenAI: https://platform.openai.com/api-keys (optional)
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LangChainConfig:
    """
    🎛️ Central Configuration Manager for LangChain Ecosystem
    
    This class manages all configuration settings for:
    - LangSmith tracing and monitoring
    - LangChain model providers
    - LangGraph workflow settings
    - Custom evaluation pipelines
    """
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults"""
        
        # 🔑 API Keys Configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")  # LangSmith uses LANGCHAIN_API_KEY
        self.openai_api_key = os.getenv("OPENAI_API_KEY")  # Optional for comparison
        
        # 📊 LangSmith Configuration
        self.langsmith_project = os.getenv("LANGCHAIN_PROJECT", "data-cleaning-agent")
        self.langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "true")
        self.langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
        # 🤖 Model Configuration
        self.default_model = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        self.model_provider = os.getenv("MODEL_PROVIDER", "google_genai")
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        
        # 🔧 Agent Configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
        
        # 📈 Evaluation Configuration
        self.enable_evaluation = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
        self.evaluation_dataset = os.getenv("EVALUATION_DATASET", "data-cleaning-test-cases")
        
        # Initialize LangSmith if API key is provided
        if self.langsmith_api_key:
            self.setup_langsmith()
        else:
            logging.warning("🚨 LangSmith API key not found. Tracing disabled.")
            logging.info("💡 To enable LangSmith: Set LANGCHAIN_API_KEY environment variable")
    
    def setup_langsmith(self) -> None:
        """
        🚀 Configure LangSmith for comprehensive observability
        
        This enables:
        - Real-time tracing of all agent operations
        - Performance monitoring and optimization insights
        - Error tracking and debugging capabilities
        - Custom evaluation pipelines
        """
        
        # Set LangSmith environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = self.langsmith_tracing
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
        
        # Configure LangSmith client
        try:
            from langsmith import Client
            self.langsmith_client = Client(
                api_key=self.langsmith_api_key,
                api_url=self.langsmith_endpoint
            )
            
            # Test connection
            projects = list(self.langsmith_client.list_projects(limit=1))
            logging.info(f"✅ LangSmith connected successfully!")
            logging.info(f"📊 Active project: {self.langsmith_project}")
            logging.info(f"🔗 Dashboard: https://smith.langchain.com/")
            
        except Exception as e:
            logging.error(f"❌ LangSmith setup failed: {str(e)}")
            logging.info("💡 Check your LANGCHAIN_API_KEY and internet connection")
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        🤖 Get model configuration for LangChain chat models
        
        Returns optimized settings for data cleaning tasks:
        - Low temperature for consistent, reliable outputs
        - High max_tokens for complex cleaning code
        - Streaming enabled for real-time feedback
        """
        
        return {
            "model": self.default_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.enable_streaming,
            "api_key": self.google_api_key,
        }
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        🔍 Validate all API keys and return status
        
        Returns:
            Dict with validation status for each service
        """
        
        validation_status = {
            "google_gemini": bool(self.google_api_key),
            "langsmith": bool(self.langsmith_api_key),
            "openai": bool(self.openai_api_key),
        }
        
        # Log validation results
        for service, is_valid in validation_status.items():
            status = "✅ Configured" if is_valid else "❌ Missing"
            logging.info(f"{service.replace('_', ' ').title()}: {status}")
        
        return validation_status
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        📋 Get complete environment configuration info
        
        Useful for debugging and setup verification
        """
        
        return {
            "api_keys": self.validate_api_keys(),
            "langsmith": {
                "project": self.langsmith_project,
                "tracing_enabled": self.langsmith_tracing == "true",
                "endpoint": self.langsmith_endpoint,
            },
            "model": {
                "provider": self.model_provider,
                "model": self.default_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "agent": {
                "max_retries": self.max_retries,
                "timeout": self.timeout_seconds,
                "streaming": self.enable_streaming,
                "evaluation": self.enable_evaluation,
            }
        }

def print_setup_instructions():
    """
    📖 Print detailed setup instructions for beginners
    """
    
    print("\n" + "="*80)
    print("🚀 LangChain Ecosystem Setup Instructions")
    print("="*80)
    
    print("\n📋 STEP 1: Create Environment File")
    print("Create a file called '.env' in your project root with:")
    print("""
# 🔑 Required API Keys
GOOGLE_API_KEY=your_google_gemini_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# 🔧 Optional Configuration
LANGCHAIN_PROJECT=data-cleaning-agent
LANGCHAIN_TRACING_V2=true
DEFAULT_MODEL=gemini-2.5-flash
MODEL_TEMPERATURE=0.1
MAX_TOKENS=4000
    """)
    
    print("\n🔑 STEP 2: Get Your API Keys")
    print("1. Google Gemini API Key:")
    print("   → Visit: https://ai.google.dev/")
    print("   → Click 'Get API Key'")
    print("   → Copy your key")
    
    print("\n2. LangSmith API Key:")
    print("   → Visit: https://smith.langchain.com/")
    print("   → Sign up/Login with your email")
    print("   → Go to Settings → API Keys")
    print("   → Create new API key")
    print("   → Copy your key")
    
    print("\n📊 STEP 3: Access LangSmith Dashboard")
    print("Once configured, visit https://smith.langchain.com/ to see:")
    print("   ✅ Real-time traces of your agent's decisions")
    print("   ✅ Performance metrics and optimization insights")
    print("   ✅ Error tracking and debugging information")
    print("   ✅ Custom evaluation results")
    
    print("\n🎯 STEP 4: Test Configuration")
    print("Run: python config.py")
    print("This will validate your setup and show you what's working!")
    
    print("\n" + "="*80)

# Global configuration instance
config = LangChainConfig()

def main():
    """
    🧪 Test configuration and display setup status
    """
    
    print("🔧 LangChain Ecosystem Configuration Test")
    print("="*50)
    
    # Validate API keys
    validation = config.validate_api_keys()
    
    if not validation["google_gemini"]:
        print("\n❌ Google Gemini API key missing!")
        print("💡 Set GOOGLE_API_KEY in your .env file")
    
    if not validation["langsmith"]:
        print("\n❌ LangSmith API key missing!")
        print("💡 Set LANGCHAIN_API_KEY in your .env file")
        print("📊 Without LangSmith, you won't see dashboard visualizations")
    
    if all([validation["google_gemini"], validation["langsmith"]]):
        print("\n✅ All required API keys configured!")
        print("🚀 Your agent is ready for LangChain ecosystem features!")
        
        # Display environment info
        env_info = config.get_environment_info()
        print(f"\n📊 LangSmith Project: {env_info['langsmith']['project']}")
        print(f"🤖 Default Model: {env_info['model']['model']}")
        print(f"🎯 Temperature: {env_info['model']['temperature']}")
        print(f"📈 Evaluation Enabled: {env_info['agent']['evaluation']}")
    else:
        print_setup_instructions()

if __name__ == "__main__":
    main()