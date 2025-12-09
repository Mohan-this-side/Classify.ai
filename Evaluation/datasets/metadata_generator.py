"""
Dataset Metadata Generator
Creates JSON metadata files for each dataset with expected characteristics,
ground truth, and pass/fail criteria.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import yaml

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """Generates metadata files for datasets with expected characteristics."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the metadata generator."""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def analyze_dataset(self, file_path: str, dataset_config: Dict) -> Dict:
        """
        Analyze a dataset and extract ground truth characteristics.
        
        Args:
            file_path: Path to the dataset CSV
            dataset_config: Configuration for this dataset
            
        Returns:
            Dictionary with analyzed characteristics
        """
        try:
            df = pd.read_csv(file_path)
            target_col = dataset_config.get('target_column')
            
            metadata = {
                'dataset_name': dataset_config['name'],
                'file_path': str(file_path),
                'shape': df.shape,
                'columns': list(df.columns),
                'target_column': target_col,
                'has_target': target_col in df.columns if target_col else False,
            }
            
            # Analyze missing values
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df) * 100).to_dict()
            metadata['missing_values'] = {
                'counts': missing_counts.to_dict(),
                'percentages': missing_pct,
                'total_missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            }
            
            # Analyze class distribution if target exists
            if target_col and target_col in df.columns:
                class_counts = df[target_col].value_counts().to_dict()
                class_distribution = (df[target_col].value_counts(normalize=True) * 100).to_dict()
                
                # Calculate imbalance ratio
                if len(class_distribution) == 2:
                    values = list(class_distribution.values())
                    max_val = max(values)
                    min_val = min(values)
                    imbalance_ratio = max_val / min_val if min_val > 0 else float('inf')
                    metadata['class_imbalance'] = {
                        'counts': class_counts,
                        'distribution_pct': class_distribution,
                        'imbalance_ratio': imbalance_ratio,
                        'is_imbalanced': imbalance_ratio > 1.5,  # >60:40 ratio
                        'is_severely_imbalanced': imbalance_ratio > 3.0  # >75:25 ratio
                    }
                else:
                    metadata['class_distribution'] = {
                        'counts': class_counts,
                        'distribution_pct': class_distribution,
                        'n_classes': len(class_distribution)
                    }
            
            # Analyze data types
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            numeric_cols = [col for col, dtype in dtypes.items() if 'int' in dtype or 'float' in dtype]
            categorical_cols = [col for col, dtype in dtypes.items() if 'object' in dtype or 'category' in dtype]
            
            metadata['data_types'] = {
                'all': dtypes,
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'mixed_types': len(numeric_cols) > 0 and len(categorical_cols) > 0
            }
            
            # Analyze correlations (for numeric columns)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # Find high correlations (>0.8)
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': float(corr_val)
                            })
                metadata['correlations'] = {
                    'high_correlation_pairs': high_corr_pairs,
                    'has_multicollinearity': len(high_corr_pairs) > 0
                }
            
            # Check for perfect leakage (if target is numeric)
            if target_col and target_col in df.columns:
                for col in df.columns:
                    if col != target_col:
                        if df[col].equals(df[target_col]):
                            metadata['data_leakage'] = {
                                'has_leakage': True,
                                'leaked_feature': col,
                                'type': 'perfect_duplicate'
                            }
                            break
                        # Check for perfect correlation
                        if col in numeric_cols:
                            corr = df[col].corr(df[target_col])
                            if abs(corr) > 0.99:
                                metadata['data_leakage'] = {
                                    'has_leakage': True,
                                    'leaked_feature': col,
                                    'type': 'perfect_correlation',
                                    'correlation': float(corr)
                                }
                                break
            
            # Check dimensionality
            metadata['dimensionality'] = {
                'n_samples': len(df),
                'n_features': len(df.columns) - (1 if target_col else 0),
                'features_per_sample': (len(df.columns) - (1 if target_col else 0)) / len(df),
                'curse_of_dimensionality': len(df.columns) > len(df)  # More features than samples
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing dataset {file_path}: {str(e)}")
            return {'error': str(e)}
    
    def generate_metadata(self, dataset_name: str, file_path: str, 
                         dataset_config: Dict) -> Dict:
        """
        Generate complete metadata for a dataset including expected behavior.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to the dataset CSV
            dataset_config: Configuration from YAML
            
        Returns:
            Complete metadata dictionary
        """
        # Analyze the dataset
        analysis = self.analyze_dataset(file_path, dataset_config)
        
        # Get expected issues from config
        expected_issues = dataset_config.get('expected_issues', [])
        expected_behavior = dataset_config.get('expected_behavior', {})
        
        # Build complete metadata
        metadata = {
            'dataset_name': dataset_name,
            'file_path': str(file_path),
            'source': 'kaggle' if 'kaggle_id' in dataset_config else 'synthetic',
            'kaggle_id': dataset_config.get('kaggle_id'),
            'description': dataset_config.get('description', ''),
            'analysis': analysis,
            'expected_issues': expected_issues,
            'expected_behavior': expected_behavior,
            'pass_fail_criteria': self._generate_pass_fail_criteria(
                dataset_config, analysis
            )
        }
        
        return metadata
    
    def _generate_pass_fail_criteria(self, dataset_config: Dict, analysis: Dict) -> Dict:
        """
        Generate pass/fail criteria for each agent based on dataset characteristics.
        
        Args:
            dataset_config: Dataset configuration
            analysis: Analyzed dataset characteristics
            
        Returns:
            Dictionary with pass/fail criteria per agent
        """
        criteria = {}
        
        # Data Discovery Agent criteria
        criteria['data_discovery'] = {
            'should_detect_types': True,
            'should_identify_target': analysis.get('has_target', False),
            'should_report_shape': True,
            'type_detection_accuracy_min': 0.90
        }
        
        # EDA Agent criteria
        imbalance_info = analysis.get('class_imbalance', {})
        criteria['eda_analysis'] = {
            'should_detect_imbalance': imbalance_info.get('is_imbalanced', False),
            'should_flag_severe_imbalance': imbalance_info.get('is_severely_imbalanced', False),
            'should_detect_missing_values': analysis.get('missing_values', {}).get('total_missing_pct', 0) > 0,
            'should_detect_correlations': analysis.get('correlations', {}).get('has_multicollinearity', False),
            'should_not_celebrate_high_accuracy_if_imbalanced': imbalance_info.get('is_severely_imbalanced', False)
        }
        
        # Data Cleaning Agent criteria
        missing_pct = analysis.get('missing_values', {}).get('total_missing_pct', 0)
        criteria['data_cleaning'] = {
            'should_handle_missing_values': missing_pct > 0,
            'should_remove_duplicates': True,
            'should_handle_outliers': True,
            'imputation_appropriateness_min': 3.0 if missing_pct > 0 else None
        }
        
        # Feature Engineering Agent criteria
        has_multicollinearity = analysis.get('correlations', {}).get('has_multicollinearity', False)
        criteria['feature_engineering'] = {
            'should_detect_multicollinearity': has_multicollinearity,
            'should_remove_redundant_features': has_multicollinearity,
            'should_encode_categoricals': analysis.get('data_types', {}).get('mixed_types', False),
            'should_scale_features': len(analysis.get('data_types', {}).get('numeric', [])) > 0
        }
        
        # ML Builder Agent criteria
        is_severely_imbalanced = imbalance_info.get('is_severely_imbalanced', False)
        has_leakage = 'data_leakage' in analysis and analysis['data_leakage'].get('has_leakage', False)
        criteria['ml_builder'] = {
            'should_apply_class_balancing': is_severely_imbalanced,
            'should_not_celebrate_majority_class_accuracy': is_severely_imbalanced,
            'should_flag_suspicious_accuracy': has_leakage,
            'should_check_minority_recall': is_severely_imbalanced,
            'anti_cheating_score_min': 0.70
        }
        
        # Model Evaluation Agent criteria
        criteria['model_evaluation'] = {
            'should_compute_comprehensive_metrics': True,
            'should_flag_imbalance_issues': is_severely_imbalanced,
            'should_check_minority_class_performance': is_severely_imbalanced,
            'should_flag_suspicious_accuracy': has_leakage,
            'metric_completeness_min': 0.90
        }
        
        # Technical Reporter Agent criteria
        criteria['technical_reporter'] = {
            'should_document_all_steps': True,
            'should_include_visualizations': True,
            'report_completeness_min': 0.85
        }
        
        # Project Manager Agent criteria
        criteria['project_manager'] = {
            'should_explain_imbalance': is_severely_imbalanced,
            'should_explain_data_leakage': has_leakage,
            'should_provide_educational_explanations': True,
            'explanation_accuracy_min': 4.0
        }
        
        return criteria
    
    def generate_all_metadata(self) -> Dict[str, Dict]:
        """
        Generate metadata for all datasets (real-world and synthetic).
        
        Returns:
            Dictionary mapping dataset names to metadata
        """
        all_metadata = {}
        
        # Process real-world datasets
        real_world_dir = Path("Evaluation/datasets/real_world")
        real_world_configs = self.config.get('real_world_datasets', [])
        
        for dataset_config in real_world_configs:
            name = dataset_config['name']
            file_path = real_world_dir / f"{name}.csv"
            
            if file_path.exists():
                logger.info(f"Generating metadata for {name}...")
                metadata = self.generate_metadata(name, str(file_path), dataset_config)
                all_metadata[name] = metadata
                
                # Save individual metadata file
                metadata_file = Path("Evaluation/datasets/metadata") / f"{name}_metadata.json"
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            else:
                logger.warning(f"Dataset file not found: {file_path}")
        
        # Process synthetic datasets
        synthetic_dir = Path("Evaluation/datasets/synthetic")
        synthetic_configs = self.config.get('synthetic_datasets', [])
        
        for dataset_config in synthetic_configs:
            name = dataset_config['name']
            file_path = synthetic_dir / f"{name}.csv"
            
            if file_path.exists():
                logger.info(f"Generating metadata for {name}...")
                metadata = self.generate_metadata(name, str(file_path), dataset_config)
                all_metadata[name] = metadata
                
                # Save individual metadata file
                metadata_file = Path("Evaluation/datasets/metadata") / f"{name}_metadata.json"
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            else:
                logger.warning(f"Dataset file not found: {file_path}")
        
        # Save combined metadata
        combined_file = Path("Evaluation/datasets/metadata/all_datasets_metadata.json")
        with open(combined_file, 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
        
        logger.info(f"Generated metadata for {len(all_metadata)} datasets")
        
        return all_metadata


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate all metadata
    generator = MetadataGenerator()
    all_metadata = generator.generate_all_metadata()
    
    print(f"\nGenerated metadata for {len(all_metadata)} datasets")
    print("Metadata files saved to Evaluation/datasets/metadata/")

