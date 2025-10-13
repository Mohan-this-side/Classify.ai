#!/usr/bin/env python3
"""
🔐 Kaggle API Setup Helper

This script helps you set up Kaggle API authentication and download the Spotify dataset.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def create_kaggle_config():
    """Create Kaggle configuration directory and guide user through setup"""
    
    print("🔐 Kaggle API Setup Helper")
    print("=" * 40)
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json_path = kaggle_dir / "kaggle.json"
    
    if kaggle_json_path.exists():
        print("✅ Kaggle API token already exists!")
        return True
    
    print("📋 To set up Kaggle API authentication:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download a file called 'kaggle.json'")
    print("5. Place that file in:", str(kaggle_json_path))
    print()
    
    # Check if file exists in current directory
    current_dir_kaggle = Path("kaggle.json")
    if current_dir_kaggle.exists():
        print("📁 Found kaggle.json in current directory, moving it...")
        import shutil
        shutil.move(str(current_dir_kaggle), str(kaggle_json_path))
        print("✅ Moved kaggle.json to correct location!")
    else:
        print("❌ kaggle.json not found in current directory")
        print("Please download it from Kaggle and place it in the current directory")
        return False
    
    # Set permissions
    try:
        os.chmod(kaggle_json_path, 0o600)
        print("✅ Set proper permissions on kaggle.json")
    except Exception as e:
        print(f"⚠️ Could not set permissions: {e}")
    
    return True

def test_kaggle_api():
    """Test if Kaggle API is working"""
    try:
        result = subprocess.run(['kaggle', 'datasets', 'list'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Kaggle API is working!")
            return True
        else:
            print(f"❌ Kaggle API test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing Kaggle API: {e}")
        return False

def download_spotify_dataset():
    """Download the Spotify churn dataset"""
    try:
        print("📥 Downloading Spotify churn dataset...")
        
        # Change to test_data directory
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Download dataset
        result = subprocess.run([
            'kaggle', 'datasets', 'download', 
            'nabihazahid/spotify-dataset-for-churn-analysis',
            '-p', str(test_data_dir)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Dataset downloaded successfully!")
            
            # Extract zip file
            zip_file = test_data_dir / "spotify-dataset-for-churn-analysis.zip"
            if zip_file.exists():
                print("📦 Extracting dataset...")
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(test_data_dir)
                print("✅ Dataset extracted!")
                
                # List extracted files
                extracted_files = list(test_data_dir.glob("*.csv"))
                if extracted_files:
                    print("📄 Extracted files:")
                    for file in extracted_files:
                        print(f"  - {file.name}")
                
                return True
            else:
                print("❌ Zip file not found after download")
                return False
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def main():
    """Main setup function"""
    print("🎵 Spotify Dataset Setup for Multi-Agent Workflow Testing")
    print("=" * 60)
    
    # Step 1: Setup Kaggle API
    print("\n🔐 Step 1: Setting up Kaggle API authentication...")
    if not create_kaggle_config():
        print("❌ Please complete Kaggle API setup manually")
        return False
    
    # Step 2: Test API
    print("\n🧪 Step 2: Testing Kaggle API...")
    if not test_kaggle_api():
        print("❌ Kaggle API not working. Please check your credentials.")
        return False
    
    # Step 3: Download dataset
    print("\n📥 Step 3: Downloading Spotify dataset...")
    if not download_spotify_dataset():
        print("❌ Failed to download dataset")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("You can now run: python test_workflow_spotify.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
