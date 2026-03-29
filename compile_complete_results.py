#!/usr/bin/env python3
"""
Script to compile ALL results from results folder into a single comprehensive CSV file
"""

import os
import re
import csv
import glob
from pathlib import Path

def parse_result_file(file_path):
    """Parse a result text file and extract metrics"""
    results = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract basic info
        model_match = re.search(r'Model:\s*(.+)', content)
        dataset_match = re.search(r'Dataset:\s*(.+)', content)
        params_match = re.search(r'Parameters:\s*([\d.]+)\s*M', content)
        gflops_match = re.search(r'GFLOPs:\s*([\d.]+)', content)
        time_match = re.search(r'Training Time:\s*([\d.]+)\s*seconds', content)
        timestamp_match = re.search(r'Timestamp:\s*(.+)', content)
        
        # Extract metrics
        oa_match = re.search(r'Overall Accuracy:\s*([\d.]+)', content)
        aa_match = re.search(r'Average Accuracy:\s*([\d.]+)', content)
        kappa_match = re.search(r'Kappa Coefficient:\s*([\d.]+)', content)
        f1_match = re.search(r'F1 Score:\s*([\d.]+)', content)
        precision_match = re.search(r'Precision:\s*([\d.]+)', content)
        recall_match = re.search(r'Recall:\s*([\d.]+)', content)
        latency_match = re.search(r'Latency:\s*([\d.]+)\s*ms', content)
        throughput_match = re.search(r'Throughput:\s*([\d.]+)\s*samples/sec', content)
        
        # Try to extract dataset from filename if not found in content
        filename = os.path.basename(file_path)
        dataset_from_filename = 'Unknown'
        if 'pavia' in filename.lower():
            dataset_from_filename = 'pavia'
        elif 'houston' in filename.lower():
            dataset_from_filename = 'houston'
        elif 'salinas' in filename.lower():
            dataset_from_filename = 'salinas'
        elif 'indiana' in filename.lower():
            dataset_from_filename = 'indiana'
        
        # Try to extract model from filename if not found in content
        model_from_filename = 'Unknown'
        if 'sslt' in filename.lower():
            model_from_filename = 'sslt'
        elif 'spectralformer' in filename.lower():
            model_from_filename = 'spectralformer'
        elif 'ssftt' in filename.lower():
            model_from_filename = 'ssftt'
        elif 'hybridsn' in filename.lower():
            model_from_filename = 'hybridsn'
        elif '3dcnn' in filename.lower():
            model_from_filename = '3dcnn'
        elif 'swinhsi' in filename.lower():
            model_from_filename = 'swinhsi'
        elif 'spectralmamba' in filename.lower():
            model_from_filename = 'spectralmamba'
        elif 'hit' in filename.lower():
            model_from_filename = 'hit'
        
        results = {
            'Model': model_match.group(1).strip() if model_match else model_from_filename,
            'Dataset': dataset_match.group(1).strip() if dataset_match else dataset_from_filename,
            'Parameters (M)': float(params_match.group(1)) if params_match else 0.0,
            'GFLOPs': float(gflops_match.group(1)) if gflops_match else 0.0,
            'Training Time (s)': float(time_match.group(1)) if time_match else 0.0,
            'Overall Accuracy': float(oa_match.group(1)) if oa_match else 0.0,
            'Average Accuracy': float(aa_match.group(1)) if aa_match else 0.0,
            'Kappa Coefficient': float(kappa_match.group(1)) if kappa_match else 0.0,
            'F1 Score': float(f1_match.group(1)) if f1_match else 0.0,
            'Precision': float(precision_match.group(1)) if precision_match else 0.0,
            'Recall': float(recall_match.group(1)) if recall_match else 0.0,
            'Latency (ms)': float(latency_match.group(1)) if latency_match else 0.0,
            'Throughput (samples/sec)': float(throughput_match.group(1)) if throughput_match else 0.0,
            'Timestamp': timestamp_match.group(1).strip() if timestamp_match else 'Unknown',
            'File': filename
        }
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        filename = os.path.basename(file_path)
        results = {
            'Model': 'Error',
            'Dataset': 'Error',
            'Parameters (M)': 0.0,
            'GFLOPs': 0.0,
            'Training Time (s)': 0.0,
            'Overall Accuracy': 0.0,
            'Average Accuracy': 0.0,
            'Kappa Coefficient': 0.0,
            'F1 Score': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'Latency (ms)': 0.0,
            'Throughput (samples/sec)': 0.0,
            'Timestamp': 'Error',
            'File': filename
        }
    
    return results

def main():
    # Find ALL result files in results folder and current directory
    result_patterns = [
        'results/*.txt',
        'results/*/*.txt',
        'ablation_results/*.txt',
        'ablation_results/*/*.txt',
        './*.txt'  # Files in current directory that look like results
    ]
    
    all_files = []
    for pattern in result_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # Filter out non-result files (keep only files that start with 'results_' or contain result-like patterns)
    filtered_files = []
    for file_path in all_files:
        filename = os.path.basename(file_path).lower()
        if (filename.startswith('results_') or 
            'results' in filename or
            any(model in filename for model in ['sslt', 'spectralformer', 'ssftt', 'hybridsn', '3dcnn', 'swinhsi', 'spectralmamba', 'hit'])):
            filtered_files.append(file_path)
    
    # Remove duplicates
    filtered_files = list(set(filtered_files))
    filtered_files.sort()
    
    print(f"Found {len(filtered_files)} result files:")
    for f in filtered_files:
        print(f"  {f}")
    
    if not filtered_files:
        print("No result files found!")
        return
    
    # Parse all files
    all_results = []
    for file_path in filtered_files:
        print(f"Parsing {file_path}...")
        result = parse_result_file(file_path)
        all_results.append(result)
    
    # Sort results by Dataset, then by Model
    all_results.sort(key=lambda x: (x['Dataset'], x['Model']))
    
    # Write to CSV
    output_file = 'complete_results.csv'
    
    if all_results:
        fieldnames = [
            'Model', 'Dataset', 'Parameters (M)', 'GFLOPs', 'Training Time (s)',
            'Overall Accuracy', 'Average Accuracy', 'Kappa Coefficient',
            'F1 Score', 'Precision', 'Recall', 'Latency (ms)', 
            'Throughput (samples/sec)', 'Timestamp', 'File'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Complete results compiled successfully!")
        print(f"Output file: {output_file}")
        print(f"Total entries: {len(all_results)}")
        
        # Print detailed summary
        print(f"\n📊 DETAILED SUMMARY:")
        print("="*50)
        
        # Summary by dataset
        dataset_counts = {}
        for result in all_results:
            dataset = result['Dataset']
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print(f"Results by Dataset:")
        for dataset, count in sorted(dataset_counts.items()):
            print(f"  {dataset}: {count} results")
        
        # Summary by model
        model_counts = {}
        for result in all_results:
            model = result['Model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        print(f"\nResults by Model:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count} results")
        
        # Performance summary (top performers by dataset)
        print(f"\n🏆 TOP PERFORMERS BY DATASET (Overall Accuracy):")
        print("-"*50)
        
        datasets = set(r['Dataset'] for r in all_results if r['Dataset'] != 'Error')
        for dataset in sorted(datasets):
            dataset_results = [r for r in all_results if r['Dataset'] == dataset and r['Overall Accuracy'] > 0]
            if dataset_results:
                best = max(dataset_results, key=lambda x: x['Overall Accuracy'])
                print(f"{dataset.upper()}: {best['Model']} - {best['Overall Accuracy']:.4f}")
        
        print(f"\n📁 Complete results saved to: {output_file}")
        print(f"🚀 Ready to transfer to your system!")
        
    else:
        print("No valid results found to compile!")

if __name__ == "__main__":
    main()