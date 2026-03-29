import numpy as np
import torch
import scipy.io
import h5py
import os
import argparse
import warnings
import logging
import datetime
import time
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from models.model import SpectralSpatialLinearTransformer
from models.model_v2 import SpectralSpatialLinearTransformerV2
# from models.model_v2_d1 import SpectralSpatialLinearTransformerV3 as SpectralSpatialLinearTransformerD1
from models.hybridsn import HybridSN
from models.swinhsi import SwinHSI
from models.cnn3d import CNN3D
from models.hit import HiT
from models.ssftt import SSFTT
from models.spectralformer import SpectralFormer
from models.spectralmamba import SpectralMamba
from evaluation import evaluate_model, print_results, count_model_parameters, calculate_gflops


def load_houston(image_file, gt_file):
    def load_mat_file(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size < 100:
            raise ValueError(f"File too small ({file_size} bytes), likely corrupted: {file_path}")
        
        with open(file_path, 'rb') as f:
            header = f.read(4)
        
        if header == b'MATL' or header[:2] == b'\x00\x00':
            try:
                f = h5py.File(file_path, 'r')
                keys = list(f.keys())
                return f, keys, 'h5py'
            except Exception:
                pass
        
        try:
            mat = scipy.io.loadmat(file_path)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            return mat, keys, 'scipy'
        except (ValueError, NotImplementedError) as e:
            try:
                f = h5py.File(file_path, 'r')
                keys = list(f.keys())
                return f, keys, 'h5py'
            except Exception as e2:
                raise ValueError(f"Could not load {file_path}: scipy={e}, h5py={e2}")
    
    def get_data(mat_obj, key, format_type):
        if format_type == 'scipy':
            return mat_obj[key]
        else:
            data_ref = mat_obj[key]
            if isinstance(data_ref, h5py.Dataset):
                return np.array(data_ref[:])
            elif isinstance(data_ref, h5py.Reference):
                ref_obj = mat_obj[data_ref]
                if isinstance(ref_obj, h5py.Dataset):
                    return np.array(ref_obj[:])
                else:
                    return np.array(ref_obj)
            elif hasattr(data_ref, '__array__'):
                return np.array(data_ref)
            else:
                try:
                    return np.array(data_ref[:])
                except:
                    return np.array(data_ref)
    
    image_mat, image_keys, image_format = load_mat_file(image_file)
    gt_mat, gt_keys, gt_format = load_mat_file(gt_file)

    if len(image_keys) == 0:
        raise ValueError("No data keys found in image file.")
    elif len(image_keys) == 1:
        image_data = get_data(image_mat, image_keys[0], image_format)
    else:
        possible_image_keys = ['ori_data', 'houston', 'Houston', 'Houston13', 'data', 'image', 'HSI', 'paviaU', 'PaviaU', 'pavia', 'Pavia', 'salinas', 'Salinas', 'salinas_corrected', 'Salinas_corrected', 'indian_pines', 'Indian_pines', 'indiana_pines', 'Indiana_pines']
        image_data = next((get_data(image_mat, k, image_format) for k in possible_image_keys if k in image_keys),
                          get_data(image_mat, image_keys[0], image_format))

    if len(gt_keys) == 0:
        raise ValueError("No data keys found in gt file.")
    elif len(gt_keys) == 1:
        ground_truth = get_data(gt_mat, gt_keys[0], gt_format)
    else:
        possible_gt_keys = ['map', 'houston_gt', 'Houston_gt', 'Houston13_7gt', 'gt',
                            'ground_truth', 'label', 'paviaU_gt', 'PaviaU_gt', 'pavia_gt',
                            'Pavia_gt', 'salinas_gt', 'Salinas_gt', 'indian_pines_gt', 'Indian_pines_gt', 'indiana_pines_gt', 'Indiana_pines_gt']
        ground_truth = next((get_data(gt_mat, k, gt_format) for k in possible_gt_keys if k in gt_keys),
                            get_data(gt_mat, gt_keys[0], gt_format))

    image_data = np.array(image_data)
    ground_truth = np.array(ground_truth)

    if image_format == 'h5py' and len(image_data.shape) == 3:
        if image_data.shape[0] < image_data.shape[2]:
            image_data = np.transpose(image_data, (1, 2, 0))

    if image_format == 'h5py':
        image_mat.close()
    if gt_format == 'h5py':
        gt_mat.close()
    
    return image_data, ground_truth


def preprocess_data(image_data, ground_truth, window_size=5):
    """Preprocess hyperspectral data to extract spatial-spectral patches."""
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    padded_image = np.pad(image_data, ((window_size//2, window_size//2),
                                       (window_size//2, window_size//2),
                                       (0, 0)), mode='reflect')
    spatial_spectral_data = np.zeros((image_data.shape[0], image_data.shape[1],
                                      window_size, window_size, image_data.shape[2]))
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            spatial_spectral_data[i, j] = padded_image[i:i+window_size, j:j+window_size, :]

    spatial_spectral_data = spatial_spectral_data.reshape(-1, window_size, window_size, image_data.shape[2])
    y = ground_truth.flatten()
    mask = y != 0
    spatial_spectral_data = spatial_spectral_data[mask]
    y = y[mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return spatial_spectral_data, y, label_encoder


class HyperspectralDataset(Dataset):
    def __init__(self, spatial_spectral_data, labels):
        self.spatial_spectral_data = spatial_spectral_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.spatial_spectral_data[idx].transpose(2, 0, 1)
        label = self.labels[idx]
        return {
            'x': torch.tensor(feature, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def data_collator(data):
    return {
        'x': torch.stack([d['x'] for d in data]),
        'labels': torch.stack([d['labels'] for d in data])
    }


def compute_metrics(p):
    predictions = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def save_results_to_files(model_name, data_name, results, std_results, model_params, gflops, training_time, save_path='./results'):
    """Save results to both text and CSV files."""
    os.makedirs(save_path, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    txt_filename = os.path.join(save_path, f"results_{model_name}_{data_name}.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {data_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Parameters: {model_params:.2f} M\n")
        f.write(f"GFLOPs: {gflops:.2f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write("\n=== RESULTS (mean ± std) ===\n")
        for key, label in [('oa', 'Overall Accuracy'), ('aa', 'Average Accuracy'),
                            ('kappa', 'Kappa Coefficient'), ('f1', 'F1 Score'),
                            ('precision', 'Precision'), ('recall', 'Recall')]:
            f.write(f"{label}: {results[key]:.4f} ± {std_results[key]:.4f}\n")
        f.write(f"Latency: {results['latency']:.4f} ms\n")
        f.write(f"Throughput: {results['throughput']:.2f} samples/sec\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sslt',
                        choices=['sslt', 'sslt_v2', 'sslt_d1',
                                 'hit', 'spectralmamba', 'hybridsn', '3dcnn',
                                 'spectralformer', 'ssftt', 'swinhsi'],
                        help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='pavia', choices=['pavia', 'houston', 'salinas', 'indiana'])
    parser.add_argument('--save_path', type=str, default='./results', help='Directory to save results')
    args = parser.parse_args()
    
    # Configuration
    if args.dataset == 'pavia':
        image_file = "./dataset/PaviaU.mat"
        gt_file = "./dataset/PaviaU_gt.mat"
    elif args.dataset == 'houston':
        image_file = "./dataset/Houston13.mat"
        gt_file = "./dataset/Houston13_7gt.mat"
    elif args.dataset == 'salinas':
        image_file = "./dataset/Salinas.mat"
        gt_file = "./dataset/Salinas_gt.mat"
    elif args.dataset == 'indiana':
        image_file = "./dataset/Indian_pines.mat"
        gt_file = "./dataset/Indian_pines_gt.mat"
    
    data_name = args.dataset
    window_size = 5
    patch_size = 4
    embed_dim = 192
    num_heads = 4
    depth = 4
    
    # Load and preprocess data
    image_data, ground_truth = load_houston(image_file, gt_file)
    spatial_spectral_data, y, label_encoder = preprocess_data(image_data, ground_truth, window_size=window_size)

    num_classes = len(np.unique(y))
    num_channels = spatial_spectral_data.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Temporary split for GFLOPs calculation
    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=0
    )
    train_dataset = HyperspectralDataset(spatial_spectral_data[train_indices], y[train_indices])
    test_dataset = HyperspectralDataset(spatial_spectral_data[test_indices], y[test_indices])

    if args.model == 'sslt':
        _tmp_model = SpectralSpatialLinearTransformer(
            image_size=window_size, patch_size=patch_size, num_channels=num_channels,
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=4.0
        )
    elif args.model == 'sslt_v2':
        _tmp_model = SpectralSpatialLinearTransformerV2(
            image_size=window_size, num_channels=num_channels,
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=4.0
        )
    elif args.model == 'sslt_d1':
        _tmp_model = SpectralSpatialLinearTransformerD1(
            image_size=window_size, num_channels=num_channels,
            num_classes=num_classes, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0
        )
    elif args.model == 'hit':
        _tmp_model = HiT(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
    elif args.model == 'spectralmamba':
        _tmp_model = SpectralMamba(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim,
                                   d_state=16, depth=depth, piece_size=2, dropout=0.1)
    elif args.model == 'hybridsn':
        _tmp_model = HybridSN(num_bands=num_channels, num_classes=num_classes, window_size=window_size)
    elif args.model == '3dcnn':
        _tmp_model = CNN3D(num_bands=num_channels, num_classes=num_classes, window_size=window_size)
    elif args.model == 'spectralformer':
        _tmp_model = SpectralFormer(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
    elif args.model == 'ssftt':
        _tmp_model = SSFTT(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
    elif args.model == 'swinhsi':
        _tmp_model = SwinHSI(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth, swin_window=patch_size)
    _tmp_model.to(device)

    model_params = count_model_parameters(_tmp_model)
    gflops = 0.0
    try:
        gflops = calculate_gflops(_tmp_model, train_dataset, device)
    except Exception:
        gflops = 0.0

    NUM_RUNS = 1
    all_results = []
    total_training_time = 0.0

    for run in range(NUM_RUNS):
        # Re-initialize model for each run
        if args.model == 'sslt':
            model = SpectralSpatialLinearTransformer(
                image_size=window_size, patch_size=patch_size, num_channels=num_channels,
                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=4.0
            )
        elif args.model == 'sslt_v2':
            model = SpectralSpatialLinearTransformerV2(
                image_size=window_size, num_channels=num_channels,
                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=4.0
            )
        elif args.model == 'sslt_d1':
            model = SpectralSpatialLinearTransformerD1(
                image_size=window_size, num_channels=num_channels,
                num_classes=num_classes, embed_dim=64, depth=2,
                num_heads=4, mlp_ratio=2.0
            )
        elif args.model == 'hit':
            model = HiT(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
        elif args.model == 'spectralmamba':
            model = SpectralMamba(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim,
                                  d_state=16, depth=depth, piece_size=2, dropout=0.1)
        elif args.model == 'hybridsn':
            model = HybridSN(num_bands=num_channels, num_classes=num_classes, window_size=window_size)
        elif args.model == '3dcnn':
            model = CNN3D(num_bands=num_channels, num_classes=num_classes, window_size=window_size)
        elif args.model == 'spectralformer':
            model = SpectralFormer(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
        elif args.model == 'ssftt':
            model = SSFTT(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
        elif args.model == 'swinhsi':
            model = SwinHSI(num_bands=num_channels, num_classes=num_classes, window_size=window_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth, swin_window=patch_size)
        model.to(device)

        # Re-split with different random seed per run
        train_indices, test_indices = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=run
        )
        train_dataset = HyperspectralDataset(spatial_spectral_data[train_indices], y[train_indices])
        test_dataset = HyperspectralDataset(spatial_spectral_data[test_indices], y[test_indices])

        training_args = TrainingArguments(
            output_dir=f"./results_run_{run}",
            num_train_epochs=20,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=0,
            logging_strategy="no",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        start_time = time.time()
        trainer.train()
        total_training_time += time.time() - start_time

        results = evaluate_model(model, test_dataset, test_indices, y, device)
        all_results.append(results)

    metrics = ['oa', 'aa', 'kappa', 'f1', 'precision', 'recall']
    agg = {}
    for m in metrics:
        vals = np.array([r[m] for r in all_results])
        agg[m] = (vals.mean(), vals.std())

    print(f"\nModel: {args.model} | Dataset: {args.dataset} | Params: {model_params:.2f}M | GFLOPs: {gflops:.4f}")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8}")
    print("-" * 30)
    for m in metrics:
        print(f"{m.upper():<12} {agg[m][0]:>8.4f} {agg[m][1]:>8.4f}")

    avg_results = {m: agg[m][0] for m in metrics}
    avg_results.update({k: all_results[-1][k] for k in ['latency', 'throughput', 'params', 'gflops', 'confusion_matrix', 'y_true', 'y_pred']})
    std_results = {m: agg[m][1] for m in metrics}
    save_results_to_files(args.model, data_name, avg_results, std_results, model_params, gflops, total_training_time / NUM_RUNS, args.save_path)


if __name__ == "__main__":
    main()
