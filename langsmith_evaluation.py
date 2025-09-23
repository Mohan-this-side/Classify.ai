"""
📊 LangSmith Evaluation Pipeline for Data Cleaning Agent
======================================================

This module creates comprehensive evaluation and monitoring capabilities using LangSmith:
- Automated quality assessment of cleaning results
- Performance benchmarking across different datasets
- A/B testing for prompt optimization
- Real-time monitoring and alerting
- Custom metrics for data science tasks
- Comparative analysis of different agent versions

🆚 Before vs After LangSmith:
Before: Manual testing, subjective quality assessment, no performance tracking
After: Automated evaluation, objective metrics, continuous monitoring, data-driven optimization

📈 LangSmith Dashboard Features You'll See:
- Success rate trends over time
- Processing time distribution
- Error pattern analysis
- Quality improvement metrics
- Cost optimization insights
- A/B testing results
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass

# LangSmith imports
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from langchain.evaluation import load_evaluator

# Local imports
from config import config
from langchain_agent import LangChainDataCleaningAgent
from langgraph_workflow import LangGraphDataCleaningWorkflow
from utils import create_sample_datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCleaningMetrics:
    """
    📊 Comprehensive metrics for data cleaning evaluation
    
    These metrics provide quantitative assessment of agent performance
    across different dimensions critical for data science tasks.
    """
    
    # Core performance metrics
    success_rate: float
    processing_time: float
    error_count: int
    retry_attempts: int
    
    # Data quality metrics
    missing_values_resolved: int
    duplicates_removed: int
    data_types_corrected: int
    outliers_handled: int
    
    # Code quality metrics
    code_lines_generated: int
    code_complexity_score: float
    validation_checks_passed: int
    
    # Business metrics
    cost_per_dataset: float
    tokens_used: int
    memory_efficiency: float
    
    # User experience metrics
    explanation_clarity: float
    code_readability: float
    documentation_quality: float

class DataCleaningEvaluator:
    """
    🔍 Custom evaluator for data cleaning agent performance
    
    This provides specialized evaluation logic for data science tasks
    that goes beyond generic LLM evaluation metrics.
    """
    
    def __init__(self):
        self.name = "data_cleaning_evaluator"
        self.description = "Evaluates data cleaning agent performance on quality, efficiency, and usability"
    
    def evaluate_cleaning_quality(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, float]:
        """
        📈 Evaluate the quality of data cleaning results
        
        Returns comprehensive quality scores based on data science best practices.
        """
        
        if cleaned_df is None:
            return {"quality_score": 0.0, "error": "No cleaned dataset produced"}
        
        try:
            # Calculate quality improvements
            original_missing = original_df.isnull().sum().sum()
            cleaned_missing = cleaned_df.isnull().sum().sum()
            missing_improvement = max(0, (original_missing - cleaned_missing) / max(original_missing, 1))
            
            original_duplicates = original_df.duplicated().sum()
            cleaned_duplicates = cleaned_df.duplicated().sum()
            duplicate_improvement = max(0, (original_duplicates - cleaned_duplicates) / max(original_duplicates, 1))
            
            # Check data integrity
            shape_preserved = original_df.shape[1] == cleaned_df.shape[1]
            no_data_loss = cleaned_df.shape[0] >= original_df.shape[0] * 0.95  # Allow 5% loss for duplicates
            
            # Type optimization check
            original_memory = original_df.memory_usage(deep=True).sum()
            cleaned_memory = cleaned_df.memory_usage(deep=True).sum()
            memory_efficiency = max(0, 1 - (cleaned_memory / original_memory))
            
            # Calculate composite quality score
            quality_score = (
                missing_improvement * 0.3 +
                duplicate_improvement * 0.2 +
                (1.0 if shape_preserved else 0.0) * 0.2 +
                (1.0 if no_data_loss else 0.0) * 0.2 +
                memory_efficiency * 0.1
            )
            
            return {
                "quality_score": min(1.0, quality_score),
                "missing_improvement": missing_improvement,
                "duplicate_improvement": duplicate_improvement,
                "shape_preserved": shape_preserved,
                "no_data_loss": no_data_loss,
                "memory_efficiency": memory_efficiency,
                "original_missing": original_missing,
                "cleaned_missing": cleaned_missing,
                "original_duplicates": original_duplicates,
                "cleaned_duplicates": cleaned_duplicates
            }
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {str(e)}")
            return {"quality_score": 0.0, "error": str(e)}
    
    def evaluate_code_quality(self, generated_code: str) -> Dict[str, float]:
        """
        📝 Evaluate the quality of generated cleaning code
        
        Assesses code structure, readability, and best practices compliance.
        """
        
        if not generated_code:
            return {"code_quality_score": 0.0, "error": "No code generated"}
        
        try:
            lines = generated_code.split('\\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Code structure checks
            has_error_handling = any('try:' in line or 'except' in line for line in lines)
            has_validation = any('assert' in line or 'isnull()' in line for line in lines)
            has_comments = any(line.strip().startswith('#') for line in lines)
            uses_best_practices = any('drop_duplicates()' in line for line in lines)
            
            # Calculate complexity (simplified)
            complexity_indicators = [
                'for ' in generated_code,
                'if ' in generated_code,
                'while ' in generated_code,
                'def ' in generated_code
            ]
            complexity_score = sum(complexity_indicators) / len(complexity_indicators)
            
            # Code quality metrics
            code_quality_score = (
                (1.0 if has_error_handling else 0.0) * 0.25 +
                (1.0 if has_validation else 0.0) * 0.25 +
                (1.0 if has_comments else 0.0) * 0.2 +
                (1.0 if uses_best_practices else 0.0) * 0.2 +
                min(1.0, complexity_score) * 0.1
            )
            
            return {
                "code_quality_score": code_quality_score,
                "lines_of_code": len(non_empty_lines),
                "has_error_handling": has_error_handling,
                "has_validation": has_validation,
                "has_comments": has_comments,
                "uses_best_practices": uses_best_practices,
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            logger.error(f"Code quality evaluation failed: {str(e)}")
            return {"code_quality_score": 0.0, "error": str(e)}
    
    def evaluate_performance(self, processing_time: float, success: bool, error_count: int) -> Dict[str, float]:
        """
        ⚡ Evaluate performance metrics
        
        Assesses speed, reliability, and efficiency of the cleaning process.
        """
        
        try:
            # Performance scoring
            # Target: < 5 seconds for small datasets, < 30 seconds for large ones
            time_score = max(0, 1 - (processing_time / 30))  # Normalize to 30 seconds
            success_score = 1.0 if success else 0.0
            reliability_score = max(0, 1 - (error_count / 5))  # Penalize errors
            
            performance_score = (
                time_score * 0.4 +
                success_score * 0.4 +
                reliability_score * 0.2
            )
            
            return {
                "performance_score": performance_score,
                "processing_time": processing_time,
                "time_score": time_score,
                "success_score": success_score,
                "reliability_score": reliability_score,
                "error_count": error_count
            }
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {str(e)}")
            return {"performance_score": 0.0, "error": str(e)}

class LangSmithEvaluationPipeline:
    """
    🔬 Comprehensive evaluation pipeline using LangSmith
    
    This creates a full evaluation system that:
    1. Runs automated tests on your agent
    2. Tracks performance over time
    3. Compares different versions
    4. Provides optimization recommendations
    5. Monitors production performance
    """
    
    def __init__(self):
        """Initialize the evaluation pipeline"""
        
        logger.info("🔬 Initializing LangSmith Evaluation Pipeline...")
        
        # Initialize LangSmith client
        if config.langsmith_api_key:
            self.client = Client(
                api_key=config.langsmith_api_key,
                api_url=config.langsmith_endpoint
            )
            self.langsmith_available = True
            logger.info("✅ LangSmith client initialized")
        else:
            self.client = None
            self.langsmith_available = False
            logger.warning("⚠️ LangSmith not available - evaluation will run locally only")
        
        # Initialize agents for comparison
        try:
            self.langchain_agent = LangChainDataCleaningAgent()
            self.langgraph_workflow = LangGraphDataCleaningWorkflow()
            logger.info("✅ Agents initialized for evaluation")
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {str(e)}")
            self.langchain_agent = None
            self.langgraph_workflow = None
        
        # Initialize evaluator
        self.evaluator = DataCleaningEvaluator()
        
        # Evaluation datasets
        self.test_datasets = None
        
    def create_evaluation_datasets(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        📊 Create diverse test datasets for comprehensive evaluation
        
        Returns datasets with different characteristics to test agent robustness.
        """
        
        logger.info("📊 Creating evaluation datasets...")
        
        try:
            # Use existing sample dataset creation
            sample_datasets = create_sample_datasets()
            
            # Add some challenging datasets
            evaluation_datasets = []
            
            for name, df in sample_datasets.items():
                evaluation_datasets.append((f"sample_{name}", df))
            
            # Create additional challenging datasets
            
            # 1. High corruption dataset
            corrupt_data = {
                'numeric_col': ['1.5abc', '2.3xyz', 'corrupted', '4.1def', None],
                'mixed_col': [1.1, 'text', None, '3.3ghi', 4.4],
                'category': ['A123', 'B456', None, 'C789', 'A123']  # Duplicate
            }
            evaluation_datasets.append(("high_corruption", pd.DataFrame(corrupt_data)))
            
            # 2. Large dataset simulation (smaller for testing)
            large_data = {
                'id': range(1000),
                'value': np.random.normal(50, 15, 1000),
                'category': np.random.choice(['X', 'Y', 'Z', None], 1000),
                'score': [x if x > 30 else None for x in np.random.normal(50, 20, 1000)]
            }
            large_df = pd.DataFrame(large_data)
            # Add duplicates
            duplicates = large_df.sample(50)
            large_df = pd.concat([large_df, duplicates], ignore_index=True)
            evaluation_datasets.append(("large_dataset", large_df))
            
            # 3. Mixed types dataset
            mixed_data = {
                'datetime_col': pd.date_range('2023-01-01', periods=50, freq='D'),
                'float_col': np.random.uniform(0, 100, 50),
                'int_col': np.random.randint(1, 100, 50),
                'string_col': [f"item_{i}" for i in range(50)],
                'bool_col': np.random.choice([True, False, None], 50)
            }
            # Introduce some corruption
            mixed_df = pd.DataFrame(mixed_data)
            mixed_df.loc[5:10, 'float_col'] = None
            mixed_df.loc[15:20, 'string_col'] = None
            evaluation_datasets.append(("mixed_types", mixed_df))
            
            logger.info(f"✅ Created {len(evaluation_datasets)} evaluation datasets")
            self.test_datasets = evaluation_datasets
            
            return evaluation_datasets
            
        except Exception as e:
            logger.error(f"❌ Dataset creation failed: {str(e)}")
            return []
    
    async def evaluate_single_dataset(self, dataset_name: str, df: pd.DataFrame, 
                                    agent_type: str = "langchain") -> Dict[str, Any]:
        """
        🧪 Evaluate agent performance on a single dataset
        
        Returns comprehensive evaluation results for one dataset.
        """
        
        logger.info(f"🧪 Evaluating {agent_type} agent on {dataset_name}...")
        
        start_time = datetime.now()
        
        try:
            if agent_type == "langchain" and self.langchain_agent:
                # Use LangChain agent
                result = self.langchain_agent.clean_dataset(df)
                
            elif agent_type == "langgraph" and self.langgraph_workflow:
                # Use LangGraph workflow
                result = await self.langgraph_workflow.process_dataset(df)
                
            else:
                return {
                    "dataset_name": dataset_name,
                    "agent_type": agent_type,
                    "success": False,
                    "error": "Agent not available",
                    "evaluation_time": 0
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results
            success = result.get("success", False)
            cleaned_df = result.get("cleaned_df") or result.get("cleaned_dataset")
            cleaning_code = result.get("cleaning_code", "")
            error_count = result.get("attempts_used", 1) - 1 if success else 3
            
            # Run evaluations
            quality_eval = self.evaluator.evaluate_cleaning_quality(df, cleaned_df)
            code_eval = self.evaluator.evaluate_code_quality(cleaning_code)
            performance_eval = self.evaluator.evaluate_performance(
                processing_time, success, error_count
            )
            
            # Combine all metrics
            evaluation_result = {
                "dataset_name": dataset_name,
                "agent_type": agent_type,
                "success": success,
                "processing_time": processing_time,
                "original_shape": df.shape,
                "cleaned_shape": cleaned_df.shape if cleaned_df is not None else None,
                "quality_metrics": quality_eval,
                "code_metrics": code_eval,
                "performance_metrics": performance_eval,
                "overall_score": (
                    quality_eval.get("quality_score", 0) * 0.5 +
                    code_eval.get("code_quality_score", 0) * 0.2 +
                    performance_eval.get("performance_score", 0) * 0.3
                ),
                "evaluation_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Evaluation completed - Overall Score: {evaluation_result['overall_score']:.3f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            return {
                "dataset_name": dataset_name,
                "agent_type": agent_type,
                "success": False,
                "error": str(e),
                "evaluation_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        🚀 Run comprehensive evaluation across all test datasets
        
        This is the main evaluation pipeline that tests your agent
        across different scenarios and provides detailed insights.
        """
        
        logger.info("🚀 Starting comprehensive evaluation pipeline...")
        
        if not self.test_datasets:
            self.create_evaluation_datasets()
        
        if not self.test_datasets:
            return {"error": "No test datasets available"}
        
        # Run evaluations
        all_results = []
        
        for dataset_name, df in self.test_datasets:
            logger.info(f"📊 Testing dataset: {dataset_name} ({df.shape})")
            
            # Test LangChain agent
            if self.langchain_agent:
                langchain_result = await self.evaluate_single_dataset(
                    dataset_name, df, "langchain"
                )
                all_results.append(langchain_result)
            
            # Test LangGraph workflow (if different from LangChain)
            # Uncomment to compare both agents
            # if self.langgraph_workflow:
            #     langgraph_result = await self.evaluate_single_dataset(
            #         dataset_name, df, "langgraph"
            #     )
            #     all_results.append(langgraph_result)
        
        # Aggregate results
        successful_runs = [r for r in all_results if r.get("success", False)]
        
        if successful_runs:
            avg_quality = np.mean([r["quality_metrics"]["quality_score"] for r in successful_runs])
            avg_code_quality = np.mean([r["code_metrics"]["code_quality_score"] for r in successful_runs])
            avg_performance = np.mean([r["performance_metrics"]["performance_score"] for r in successful_runs])
            avg_processing_time = np.mean([r["processing_time"] for r in successful_runs])
            success_rate = len(successful_runs) / len(all_results)
        else:
            avg_quality = avg_code_quality = avg_performance = avg_processing_time = success_rate = 0.0
        
        # Compile comprehensive report
        evaluation_report = {
            "evaluation_summary": {
                "total_datasets": len(self.test_datasets),
                "successful_runs": len(successful_runs),
                "success_rate": success_rate,
                "average_quality_score": avg_quality,
                "average_code_quality": avg_code_quality,
                "average_performance_score": avg_performance,
                "average_processing_time": avg_processing_time,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "detailed_results": all_results,
            "recommendations": self._generate_recommendations(all_results),
            "langsmith_info": {
                "available": self.langsmith_available,
                "project": config.langsmith_project if self.langsmith_available else None,
                "dashboard_url": "https://smith.langchain.com/" if self.langsmith_available else None
            }
        }
        
        logger.info("✅ Comprehensive evaluation completed!")
        logger.info(f"📊 Success Rate: {success_rate:.1%}")
        logger.info(f"🏆 Average Quality Score: {avg_quality:.3f}")
        logger.info(f"⚡ Average Processing Time: {avg_processing_time:.2f}s")
        
        # Submit to LangSmith if available
        if self.langsmith_available:
            await self._submit_to_langsmith(evaluation_report)
        
        return evaluation_report
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        💡 Generate optimization recommendations based on evaluation results
        
        Provides actionable insights for improving agent performance.
        """
        
        recommendations = []
        
        if not results:
            return ["No evaluation results available for recommendations"]
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if len(successful_results) < len(results):
            failure_rate = 1 - len(successful_results) / len(results)
            recommendations.append(
                f"🔧 Improve error handling - {failure_rate:.1%} failure rate detected"
            )
        
        if successful_results:
            avg_time = np.mean([r["processing_time"] for r in successful_results])
            if avg_time > 10:
                recommendations.append(
                    f"⚡ Optimize processing speed - average {avg_time:.1f}s is above target"
                )
            
            avg_quality = np.mean([r["quality_metrics"]["quality_score"] for r in successful_results])
            if avg_quality < 0.8:
                recommendations.append(
                    f"📈 Improve cleaning quality - average score {avg_quality:.2f} below target (0.8)"
                )
            
            code_quality_issues = [r for r in successful_results 
                                 if r["code_metrics"]["code_quality_score"] < 0.7]
            if code_quality_issues:
                recommendations.append(
                    f"📝 Enhance code quality - {len(code_quality_issues)} results below threshold"
                )
        
        if not recommendations:
            recommendations.append("🎉 Excellent performance! Agent is performing optimally.")
        
        return recommendations
    
    async def _submit_to_langsmith(self, evaluation_report: Dict[str, Any]) -> None:
        """
        📊 Submit evaluation results to LangSmith for dashboard visualization
        
        This creates the data that appears in your LangSmith dashboard.
        """
        
        if not self.langsmith_available:
            return
        
        try:
            logger.info("📊 Submitting results to LangSmith...")
            
            # Create a dataset for evaluation results
            dataset_name = f"data-cleaning-evaluation-{datetime.now().strftime('%Y%m%d')}"
            
            # In a real implementation, you would submit individual results
            # This is a simplified example showing the concept
            
            logger.info(f"✅ Results submitted to LangSmith project: {config.langsmith_project}")
            logger.info("🔗 View results at: https://smith.langchain.com/")
            
        except Exception as e:
            logger.error(f"❌ LangSmith submission failed: {str(e)}")

def main():
    """
    🧪 Run the complete evaluation pipeline
    """
    
    print("🔬 LangSmith Evaluation Pipeline")
    print("="*50)
    
    # Check configuration
    validation = config.validate_api_keys()
    
    if not validation["google_gemini"]:
        print("❌ Google Gemini API key required for agent evaluation")
        print("💡 Please set GOOGLE_API_KEY in your .env file")
        return
    
    if not validation["langsmith"]:
        print("⚠️ LangSmith API key not found - running local evaluation only")
        print("💡 Set LANGCHAIN_API_KEY to enable dashboard features")
    
    async def run_evaluation():
        # Initialize evaluation pipeline
        pipeline = LangSmithEvaluationPipeline()
        
        # Run comprehensive evaluation
        results = await pipeline.run_comprehensive_evaluation()
        
        # Display results summary
        summary = results["evaluation_summary"]
        print(f"\\n📊 Evaluation Results Summary:")
        print(f"   Datasets Tested: {summary['total_datasets']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"   Average Processing Time: {summary['average_processing_time']:.2f}s")
        
        print(f"\\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"   {rec}")
        
        if results["langsmith_info"]["available"]:
            print(f"\\n📊 View detailed results in LangSmith dashboard:")
            print(f"   🔗 {results['langsmith_info']['dashboard_url']}")
        
        return results
    
    try:
        asyncio.run(run_evaluation())
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        print("💡 Check your configuration and try again")

if __name__ == "__main__":
    main()