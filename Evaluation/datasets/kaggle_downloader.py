"""
Kaggle Dataset Downloader
Downloads real-world classification datasets from Kaggle for evaluation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging
import pandas as pd

try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    kaggle = None
    KaggleApi = None

logger = logging.getLogger(__name__)


class KaggleDownloader:
    """Downloads datasets from Kaggle using the Kaggle API."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the downloader with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.api = None
        self._setup_kaggle_api()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_kaggle_api(self):
        """Set up Kaggle API with credentials."""
        if KaggleApi is None:
            raise ImportError(
                "Kaggle API not installed. Install with: pip install kaggle"
            )
        
        # Set environment variables for Kaggle API
        os.environ['KAGGLE_USERNAME'] = self.config['kaggle']['username']
        os.environ['KAGGLE_KEY'] = self.config['kaggle']['key']
        
        # Create .kaggle directory if it doesn't exist
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write kaggle.json
        kaggle_json = {
            'username': self.config['kaggle']['username'],
            'key': self.config['kaggle']['key']
        }
        with open(kaggle_dir / 'kaggle.json', 'w') as f:
            json.dump(kaggle_json, f)
        
        # Set permissions (required on Linux/Mac)
        os.chmod(kaggle_dir / 'kaggle.json', 0o600)
        
        # Initialize API
        self.api = KaggleApi()
        self.api.authenticate()
        logger.info("Kaggle API authenticated successfully")
    
    def download_dataset(self, dataset_name: str, kaggle_id: str, 
                        output_dir: str = "Evaluation/datasets/real_world") -> Optional[str]:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_name: Name for the dataset
            kaggle_id: Kaggle dataset identifier (e.g., "c/titanic")
            output_dir: Directory to save the dataset
            
        Returns:
            Path to downloaded CSV file, or None if failed
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if dataset already exists
            standardized_name = f"{dataset_name}.csv"
            standardized_path = output_path / standardized_name
            
            if standardized_path.exists():
                logger.info(f"Dataset {dataset_name} already exists at {standardized_path}, skipping download")
                return str(standardized_path)
            
            logger.info(f"Downloading dataset: {dataset_name} ({kaggle_id})")
            
            # Download dataset
            self.api.dataset_download_files(
                kaggle_id,
                path=str(output_path),
                unzip=True
            )
            
            # Find the CSV file(s)
            csv_files = list(output_path.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found for {dataset_name}")
                return None
            
            # If multiple CSVs, try to find the main one
            if len(csv_files) > 1:
                # Prefer files without "test" in the name
                main_csv = None
                for csv_file in csv_files:
                    if "test" not in csv_file.name.lower():
                        main_csv = csv_file
                        break
                if main_csv is None:
                    main_csv = csv_files[0]
            else:
                main_csv = csv_files[0]
            
            # Rename to standardized name
            standardized_name = f"{dataset_name}.csv"
            standardized_path = output_path / standardized_name
            if main_csv != standardized_path:
                main_csv.rename(standardized_path)
            
            logger.info(f"Successfully downloaded {dataset_name} to {standardized_path}")
            return str(standardized_path)
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {str(e)}")
            return None
    
    def download_all_datasets(self) -> Dict[str, Optional[str]]:
        """
        Download all datasets specified in the configuration.
        
        Returns:
            Dictionary mapping dataset names to file paths
        """
        results = {}
        datasets = self.config.get('real_world_datasets', [])
        
        for dataset_config in datasets:
            name = dataset_config['name']
            kaggle_id = dataset_config['kaggle_id']
            
            file_path = self.download_dataset(
                name,
                kaggle_id,
                self.config['kaggle']['datasets_dir']
            )
            results[name] = file_path
        
        return results
    
    def verify_dataset(self, file_path: str, expected_target: str) -> Dict:
        """
        Verify that a downloaded dataset has the expected structure.
        
        Args:
            file_path: Path to the CSV file
            expected_target: Expected name of target column
            
        Returns:
            Dictionary with verification results
        """
        try:
            df = pd.read_csv(file_path)
            
            result = {
                'file_exists': True,
                'shape': df.shape,
                'columns': list(df.columns),
                'has_target': expected_target in df.columns,
                'target_column': expected_target if expected_target in df.columns else None,
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            return result
            
        except Exception as e:
            return {
                'file_exists': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download all datasets
    downloader = KaggleDownloader()
    results = downloader.download_all_datasets()
    
    # Print results
    print("\n" + "="*60)
    print("Dataset Download Summary")
    print("="*60)
    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"{status} {name}: {path}")
    print("="*60)

