"""Quick test to verify all imports work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
    print("✓ yaml")
except ImportError as e:
    print(f"✗ yaml: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError as e:
    print(f"✗ matplotlib: {e}")

try:
    import seaborn as sns
    print("✓ seaborn")
except ImportError as e:
    print(f"✗ seaborn: {e}")

try:
    import kaggle
    print("✓ kaggle")
except ImportError as e:
    print(f"✗ kaggle: {e}")

try:
    from datasets.kaggle_downloader import KaggleDownloader
    print("✓ KaggleDownloader")
except ImportError as e:
    print(f"✗ KaggleDownloader: {e}")

try:
    from datasets.synthetic_generator import SyntheticDatasetGenerator
    print("✓ SyntheticDatasetGenerator")
except ImportError as e:
    print(f"✗ SyntheticDatasetGenerator: {e}")

print("\nAll imports successful!")

