import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Unified dataset loader for job fraud detection project.
    Handles downloading and processing of multiple datasets.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader with the root data directory.
        
        Args:
            data_dir: Root directory for storing all data
        """
        self.root_dir = Path(data_dir)
        self.raw_dir = self.root_dir / "raw"
        self.processed_dir = self.root_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            # Core fraud datasets
            "real_or_fake": {
                "source": "kaggle",
                "id": "shivamb/real-or-fake-fake-jobposting-prediction",
                "description": "Dataset containing both real and fake job postings"
            },
            "fake_jobs": {
                "source": "kaggle",
                "id": "srisaisuhassanisetty/fake-job-postings",
                "description": "Collection of fake job postings"
            },
            "indeed_jobs": {
                "source": "kaggle",
                "id": "promptcloud/indeed-job-posting-dataset",
                "description": "Genuine job postings from Indeed"
            },
            "phishing_emails": {
                "source": "huggingface",
                "id": "drorrabin/phishing_emails-data",
                "description": "Collection of phishing emails"
            }
        }
    
    def _download_kaggle_dataset(self, dataset_id: str, save_path: Path) -> bool:
        """Download a dataset from Kaggle."""
        try:
            import kagglehub
            logger.info(f"Downloading Kaggle dataset: {dataset_id}")
            download_path = kagglehub.dataset_download(dataset_id)
            shutil.copytree(download_path, save_path, dirs_exist_ok=True)
            return True
        except ImportError:
            logger.error("kagglehub package not installed. Install with: pip install kagglehub")
            return False
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {str(e)}")
            return False
    
    def _load_hf_dataset(self, dataset_id: str, save_path: Path) -> bool:
        """Load a dataset from HuggingFace Hub."""
        try:
            from datasets import load_dataset
            logger.info(f"Loading HuggingFace dataset: {dataset_id}")
            dataset = load_dataset(dataset_id)
            dataset.save_to_disk(str(save_path))
            return True
        except ImportError:
            logger.error("datasets package not installed. Install with: pip install datasets")
            return False
        except Exception as e:
            logger.error(f"Failed to load {dataset_id}: {str(e)}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a specific dataset by name."""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        dataset_info = self.datasets[dataset_name]
        save_path = self.raw_dir / dataset_name
        
        if dataset_info["source"] == "kaggle":
            return self._download_kaggle_dataset(dataset_info["id"], save_path)
        elif dataset_info["source"] == "huggingface":
            return self._load_hf_dataset(dataset_info["id"], save_path)
        else:
            logger.error(f"Unsupported data source: {dataset_info['source']}")
            return False
    
    def download_all(self) -> Dict[str, bool]:
        """Download all available datasets."""
        results = {}
        for name in self.datasets:
            results[name] = self.download_dataset(name)
        return results
    
    def load_processed_data(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a processed dataset as a pandas DataFrame."""
        file_path = self.processed_dir / f"{dataset_name}.parquet"
        if file_path.exists():
            try:
                return pd.read_parquet(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                return None
        else:
            logger.warning(f"Processed dataset not found: {file_path}")
            return None
    
    def save_processed_data(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Save a processed dataset as a parquet file."""
        try:
            file_path = self.processed_dir / f"{dataset_name}.parquet"
            df.to_parquet(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save {dataset_name}: {str(e)}")
            return False


def main():
    """Example usage of the DataLoader."""
    loader = DataLoader()
    
    # Download all datasets
    print("Downloading all datasets...")
    results = loader.download_all()
    
    # Print summary
    print("\nDownload Summary:")
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
