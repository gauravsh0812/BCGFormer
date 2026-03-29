import numpy as np
import torch
import time
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, precision_score, recall_score, f1_score
from fvcore.nn import FlopCountAnalysis


def calculate_latency_per_image(model, data_loader, device):
    model.eval()
    total_time, total_images = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.shape[0]
            total_images += batch_size
            start_time = time.time()
            _ = model(inputs)
            total_time += (time.time() - start_time)
    return (total_time / total_images) * 1000


def calculate_throughput(model, data_loader, device):
    model.eval()
    total_samples, total_time = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.size(0)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            total_samples += batch_size
    return total_samples / total_time


def overall_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return np.nanmean(class_accuracies)


def kappa_coefficient(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)


def calculate_f1_precision_recall(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return f1, precision, recall


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def calculate_gflops(model, dataset, device):
    sample = dataset[0]['x'].unsqueeze(0).to(device)
    flops = FlopCountAnalysis(model, sample)
    flops.unsupported_ops_settings(raise_on_error=False)
    flops.uncalled_modules_settings(warn_on_error=False)
    total_flops = flops.total()
    if total_flops == 0:
        return 0.0
    gflops = total_flops / 1e9
    if gflops < 0.001:
        return round(total_flops / 1e6, 4)  # return MFLOPs if GFLOPs too small
    return gflops


def evaluate_model(model, test_dataset, test_indices, y, device):
    """Complete evaluation of the model"""
    from transformers import Trainer
    
    # Generate predictions
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = y[test_indices]
    
    # Calculate metrics
    oa = overall_accuracy(y_true, y_pred)
    aa = average_accuracy(y_true, y_pred)
    kappa = kappa_coefficient(y_true, y_pred)
    f1, precision, recall = calculate_f1_precision_recall(y_true, y_pred)
    
    # Performance metrics
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    latency = calculate_latency_per_image(model, test_loader, device)
    throughput = calculate_throughput(model, test_loader, device)
    params = count_model_parameters(model)
    
    try:
        gflops = calculate_gflops(model, test_dataset, device)
    except:
        gflops = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'oa': oa,
        'aa': aa,
        'kappa': kappa,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'latency': latency,
        'throughput': throughput,
        'params': params,
        'gflops': gflops,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    return results


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*50)
    print("Classification Metrics")
    print("="*50)
    print(f"Overall Accuracy (OA):     {results['oa']:.4f}")
    print(f"Average Accuracy (AA):      {results['aa']:.4f}")
    print(f"Kappa Coefficient:          {results['kappa']:.4f}")
    print(f"F1 Score (weighted):         {results['f1']:.4f}")
    print(f"Precision (weighted):        {results['precision']:.4f}")
    print(f"Recall (weighted):           {results['recall']:.4f}")
    print("="*50)
    
    print("\n" + "="*50)
    print("Performance Metrics")
    print("="*50)
    print(f"Latency per image:          {results['latency']:.4f} ms")
    print(f"Throughput:                  {results['throughput']:.2f} samples/sec")
    print(f"Model Parameters:            {results['params']:.2f} M")
    if results['gflops'] is not None:
        if results['gflops'] < 0.001:
            print(f"MFLOPs:                      {results['gflops']:.4f} MFLOPs")
        else:
            print(f"GFLOPs:                      {results['gflops']:.4f}")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(results['y_true'], results['y_pred'], digits=4))
