"""
Synthetic Dataset Generator
Generates synthetic datasets with known issues for edge case testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import yaml

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generates synthetic datasets with specific characteristics for testing."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the generator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.random_seed = self.config.get('reproducibility', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_perfect_leakage_dataset(
        self, 
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        output_path: str = "Evaluation/datasets/synthetic/perfect_leakage.csv"
    ) -> str:
        """
        Generate a dataset with perfect data leakage (target duplicated as feature).
        This tests anti-cheating capabilities.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features (excluding target)
            n_classes: Number of classes
            output_path: Where to save the dataset
            
        Returns:
            Path to saved CSV file
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        if output_path_obj.exists():
            logger.info(f"Perfect leakage dataset already exists at {output_path}, skipping generation")
            return str(output_path_obj)
        
        logger.info("Generating perfect leakage dataset...")
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target variable
        y = np.random.randint(0, n_classes, n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add target column
        df['target'] = y
        
        # CRITICAL: Add perfect leakage - duplicate target as a feature
        df['leaked_feature'] = y  # Perfect correlation with target
        
        # Save to CSV
        df.to_csv(output_path_obj, index=False)
        
        logger.info(f"Perfect leakage dataset saved to {output_path}")
        logger.warning("Expected behavior: Agents should flag suspiciously high accuracy!")
        
        return str(output_path_obj)
    
    def generate_severe_imbalance_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 15,
        imbalance_ratio: float = 0.99,
        output_path: str = "Evaluation/datasets/synthetic/severe_imbalance.csv"
    ) -> str:
        """
        Generate a severely imbalanced dataset (99:1 ratio).
        Tests if agents detect imbalance and don't just celebrate high accuracy.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            imbalance_ratio: Ratio of majority class (0.99 = 99:1)
            output_path: Where to save the dataset
            
        Returns:
            Path to saved CSV file
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        if output_path_obj.exists():
            logger.info(f"Severe imbalance dataset already exists at {output_path}, skipping generation")
            return str(output_path_obj)
        
        logger.info(f"Generating severe imbalance dataset ({imbalance_ratio:.1%} majority)...")
        
        # Calculate class distribution
        n_majority = int(n_samples * imbalance_ratio)
        n_minority = n_samples - n_majority
        
        # Generate features for majority class
        X_majority = np.random.randn(n_majority, n_features)
        y_majority = np.zeros(n_majority, dtype=int)
        
        # Generate features for minority class (slightly different distribution)
        X_minority = np.random.randn(n_minority, n_features) + 2.0  # Shifted mean
        y_minority = np.ones(n_minority, dtype=int)
        
        # Combine
        X = np.vstack([X_majority, X_minority])
        y = np.hstack([y_majority, y_minority])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to CSV
        df.to_csv(output_path_obj, index=False)
        
        logger.info(f"Severe imbalance dataset saved to {output_path}")
        logger.info(f"Class distribution: {np.sum(y==0)} (class 0), {np.sum(y==1)} (class 1)")
        logger.warning("Expected behavior: Agents should detect imbalance and apply balancing!")
        
        return str(output_path_obj)
    
    def generate_multicollinearity_dataset(
        self,
        n_samples: int = 500,
        n_features: int = 20,
        n_classes: int = 3,
        correlation_threshold: float = 0.95,
        output_path: str = "Evaluation/datasets/synthetic/multicollinearity.csv"
    ) -> str:
        """
        Generate a dataset with high multicollinearity (correlation >0.95).
        Tests feature engineering agent's ability to detect and handle redundancy.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            correlation_threshold: Minimum correlation between redundant features
            output_path: Where to save the dataset
            
        Returns:
            Path to saved CSV file
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        if output_path_obj.exists():
            logger.info(f"Multicollinearity dataset already exists at {output_path}, skipping generation")
            return str(output_path_obj)
        
        logger.info("Generating multicollinearity dataset...")
        
        # Generate base features
        n_base_features = n_features // 2
        X_base = np.random.randn(n_samples, n_base_features)
        
        # Create highly correlated features by adding noise to base features
        X_correlated = []
        for i in range(n_base_features):
            # Create a feature highly correlated with base feature i
            noise_level = np.sqrt(1 - correlation_threshold**2)
            correlated_feature = X_base[:, i] + np.random.randn(n_samples) * noise_level
            X_correlated.append(correlated_feature)
        
        X_correlated = np.array(X_correlated).T
        
        # Combine base and correlated features
        X = np.hstack([X_base, X_correlated])
        
        # Generate target based on first few features
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        if n_classes > 2:
            # Convert to multiclass
            y = (y + (X[:, 2] > 0).astype(int)) % n_classes
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to CSV
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path_obj, index=False)
        
        # Verify correlation
        corr_matrix = df[feature_names].corr()
        high_corr_pairs = []
        for i in range(n_base_features):
            corr_val = corr_matrix.iloc[i, i + n_base_features]
            high_corr_pairs.append((i, i + n_base_features, corr_val))
        
        logger.info(f"Multicollinearity dataset saved to {output_path}")
        logger.info(f"Created {len(high_corr_pairs)} highly correlated feature pairs")
        logger.warning("Expected behavior: Feature Engineering agent should detect and remove redundant features!")
        
        return str(output_path_obj)
    
    def generate_high_dimensionality_dataset(
        self,
        n_samples: int = 100,
        n_features: int = 200,
        n_classes: int = 2,
        output_path: str = "Evaluation/datasets/synthetic/high_dimensionality.csv"
    ) -> str:
        """
        Generate a high-dimensional dataset (more features than samples).
        Tests curse of dimensionality detection.
        
        Args:
            n_samples: Number of samples (should be < n_features)
            n_features: Number of features
            n_classes: Number of classes
            output_path: Where to save the dataset
            
        Returns:
            Path to saved CSV file
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        if output_path_obj.exists():
            logger.info(f"High dimensionality dataset already exists at {output_path}, skipping generation")
            return str(output_path_obj)
        
        logger.info(f"Generating high dimensionality dataset ({n_samples} samples, {n_features} features)...")
        
        if n_samples >= n_features:
            logger.warning("n_samples >= n_features. This doesn't test curse of dimensionality!")
        
        # Generate features (more features than samples)
        X = np.random.randn(n_samples, n_features)
        
        # Generate target based on first few features (only first 5 are informative)
        informative_features = X[:, :5]
        y = (informative_features.sum(axis=1) + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to CSV
        df.to_csv(output_path_obj, index=False)
        
        logger.info(f"High dimensionality dataset saved to {output_path}")
        logger.info(f"Ratio: {n_features/n_samples:.1f} features per sample")
        logger.warning("Expected behavior: Agents should detect curse of dimensionality and recommend reduction!")
        
        return str(output_path_obj)
    
    def generate_all_synthetic_datasets(self) -> Dict[str, str]:
        """
        Generate all synthetic datasets specified in configuration.
        
        Returns:
            Dictionary mapping dataset names to file paths
        """
        results = {}
        datasets = self.config.get('synthetic_datasets', [])
        
        for dataset_config in datasets:
            name = dataset_config['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating synthetic dataset: {name}")
            logger.info(f"{'='*60}")
            
            if name == "perfect_leakage":
                path = self.generate_perfect_leakage_dataset(
                    n_samples=dataset_config.get('n_samples', 1000),
                    n_features=dataset_config.get('n_features', 10),
                    n_classes=dataset_config.get('n_classes', 2)
                )
                
            elif name == "severe_imbalance":
                imbalance_ratio = 0.99
                if 'imbalance_ratio' in dataset_config:
                    ratio_str = dataset_config['imbalance_ratio']
                    if isinstance(ratio_str, str) and ':' in ratio_str:
                        parts = ratio_str.split(':')
                        imbalance_ratio = float(parts[0]) / (float(parts[0]) + float(parts[1]))
                
                path = self.generate_severe_imbalance_dataset(
                    n_samples=dataset_config.get('n_samples', 1000),
                    n_features=dataset_config.get('n_features', 15),
                    imbalance_ratio=imbalance_ratio
                )
                
            elif name == "multicollinearity":
                path = self.generate_multicollinearity_dataset(
                    n_samples=dataset_config.get('n_samples', 500),
                    n_features=dataset_config.get('n_features', 20),
                    n_classes=dataset_config.get('n_classes', 3),
                    correlation_threshold=dataset_config.get('correlation_threshold', 0.95)
                )
                
            elif name == "high_dimensionality":
                path = self.generate_high_dimensionality_dataset(
                    n_samples=dataset_config.get('n_samples', 100),
                    n_features=dataset_config.get('n_features', 200),
                    n_classes=dataset_config.get('n_classes', 2)
                )
            else:
                logger.warning(f"Unknown synthetic dataset type: {name}")
                continue
            
            results[name] = path
        
        return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate all synthetic datasets
    generator = SyntheticDatasetGenerator()
    results = generator.generate_all_synthetic_datasets()
    
    # Print results
    print("\n" + "="*60)
    print("Synthetic Dataset Generation Summary")
    print("="*60)
    for name, path in results.items():
        print(f"✓ {name}: {path}")
    print("="*60)

