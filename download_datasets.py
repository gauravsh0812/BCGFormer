#!/usr/bin/env python3
"""
Script to download Houston and Salinas hyperspectral datasets
"""

import os
import shutil
import kagglehub
import requests
from pathlib import Path

def download_houston_dataset():
    """Download Houston 2013 dataset from Kaggle"""
    print("Downloading Houston 2013 dataset from Kaggle...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("potnuruganesh/houston-2013-dataset")
        print("Path to dataset files:", path)
        
        # Copy files to our dataset folder
        dataset_dir = Path("./dataset")
        dataset_dir.mkdir(exist_ok=True)
        
        # Find and copy Houston files
        source_path = Path(path)
        for file in source_path.glob("*.mat"):
            if "houston" in file.name.lower() or "Houston" in file.name:
                dest_file = dataset_dir / file.name
                shutil.copy2(file, dest_file)
                print(f"Copied {file.name} to dataset folder")
        
        print("✓ Houston dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading Houston dataset: {e}")
        return False

def download_salinas_dataset():
    """Download Salinas dataset"""
    print("Downloading Salinas dataset...")
    
    # Alternative URLs for Salinas dataset
    salinas_urls = [
        "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas.mat",
        "http://www.ehu.es/ccwintco/uploads/a/a3/Salinas.mat",
        "https://github.com/danfenghong/IEEE_TGRS_SpectralFormer/raw/main/datasets/Salinas.mat"
    ]
    
    salinas_gt_urls = [
        "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        "http://www.ehu.es/ccwintco/uploads/f/fa/Salinas_gt.mat",
        "https://github.com/danfenghong/IEEE_TGRS_SpectralFormer/raw/main/datasets/Salinas_gt.mat"
    ]
    
    dataset_dir = Path("./dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Download Salinas image
    for url in salinas_urls:
        try:
            print(f"Trying to download from: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(dataset_dir / "Salinas.mat", "wb") as f:
                    f.write(response.content)
                print("✓ Salinas.mat downloaded successfully!")
                break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    else:
        print("❌ Failed to download Salinas.mat from all sources")
        return False
    
    # Download Salinas ground truth
    for url in salinas_gt_urls:
        try:
            print(f"Trying to download GT from: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(dataset_dir / "Salinas_gt.mat", "wb") as f:
                    f.write(response.content)
                print("✓ Salinas_gt.mat downloaded successfully!")
                break
        except Exception as e:
            print(f"Failed to download GT from {url}: {e}")
            continue
    else:
        print("❌ Failed to download Salinas_gt.mat from all sources")
        return False
    
    return True

def main():
    print("Starting dataset downloads...")
    
    # Download Houston dataset
    houston_success = download_houston_dataset()
    
    # Download Salinas dataset
    salinas_success = download_salinas_dataset()
    
    # Summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Houston dataset: {'✓ Success' if houston_success else '❌ Failed'}")
    print(f"Salinas dataset: {'✓ Success' if salinas_success else '❌ Failed'}")
    
    # List downloaded files
    dataset_dir = Path("./dataset")
    if dataset_dir.exists():
        print(f"\nFiles in dataset folder:")
        for file in sorted(dataset_dir.glob("*.mat")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()