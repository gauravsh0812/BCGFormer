import numpy as np
import torch
import scipy.io
import h5py
import os
import argparse
import warnings
import csv
import datetime
import time
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress the autograd warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.function")

from evaluation import evaluate_model, print_results, count_model_parameters, calculate_gflops
import torch.nn as nn


class BandWeightedPooling(nn.Module):
    """Learnable spectral-band weighting for global token aggregation."""
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        return (x * w).sum(dim=1)


class SpectralSpatialLinearAttention(nn.Module):
    """Linear attention with explicit spectral gating. O(N) complexity."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        self.spectral_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        k = k.softmax(dim=1)
        context = torch.einsum('bnhd,bnhv->bhdv', k, v)
        out = torch.einsum('bnhd,bhdv->bnhv', q, context)
        out = out.reshape(B, N, C)
        
        gate = self.spectral_gate(x)
        out = out * gate
        
        return self.proj(out)


# REMOVED: GlobalAttentionBlock class


class SpectralSpatialViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpectralSpatialLinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralSpatialLinearTransformer(nn.Module):
    """SSLT w/o Global Attention - Ablation Study"""
    def __init__(
        self,
        image_size=5,
        patch_size=1,
        num_channels=103,
        num_classes=9,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        self.patch_embed = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches_h = (image_size - patch_size) // patch_size + 1
        num_patches_w = (image_size - patch_size) // patch_size + 1
        num_patches = num_patches_h * num_patches_w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            SpectralSpatialViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # REMOVED: self.global_block = GlobalAttentionBlock(embed_dim, num_heads)
        self.spectral_pool = BandWeightedPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        # REMOVED: x = self.global_block(x)
        x = self.spectral_pool(x)
        x = self.norm(x)
        logits = self.head(x)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def load_houston(image_file, gt_file):
    """Load Houston hyperspectral dataset from .mat files."""
    print("Loading Houston dataset...")
    
    def load_mat_file(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size < 100:
            raise ValueError(f"File too small ({file_size} bytes), likely corrupted: {file_path}")
        
        with open(file_path, 'rb') as f:
            header = f.read(4)
        
        if header == b'MATL' or header[:2] == b'\\x00\\x00':
            try:
                print(f"  Detected HDF5 format, using h5py...")
                f = h5py.File(file_path, 'r')
                keys = list(f.keys())
                print(f"  Found keys: {keys}")
                return f, keys, 'h5py'
            except Exception as e:
                print(f"  h5py failed: {e}, trying scipy...")
        
        try:
            mat = scipy.io.loadmat(file_path)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            print(f"  Found keys: {keys}")
            return mat, keys, 'scipy'
        except (ValueError, NotImplementedError) as e:
            print(f"  scipy.io.loadmat failed ({e}), trying h5py...")
            try:
                f = h5py.File(file_path, 'r')
                keys = list(f.keys())
                print(f"  Found keys: {keys}")
                return f, keys, 'h5py'
            except Exception as e2:
                error_msg = f"Could not load {file_path}:\n"
                error_msg += f"  - scipy error: {e}\n"
                error_msg += f"  - h5py error: {e2}\n"
                error_msg += f"  - File size: {file_size} bytes\n"
                error_msg += f"  - File header: {header.hex()}\n"
                error_msg += "\nThe file might be corrupted or in an unsupported format.\n"
                error_msg += "Please verify the file was downloaded correctly."
                raise ValueError(error_msg)
    
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
    
    print(f"  Image file format: {image_format}")
    print(f"  GT file format: {gt_format}")
    
    # Load image data
    if len(image_keys) == 0:
        raise ValueError("No data keys found in image file.")
    elif len(image_keys) == 1:
        image_data = get_data(image_mat, image_keys[0], image_format)
        print(f"Using image data key: '{image_keys[0]}'")
    else:
        possible_image_keys = ['ori_data', 'houston', 'Houston', 'Houston13', 'data', 'image', 'HSI', 'paviaU', 'PaviaU', 'pavia', 'Pavia', 'salinas', 'Salinas', 'salinas_corrected', 'Salinas_corrected', 'indian_pines', 'Indian_pines', 'indiana_pines', 'Indiana_pines']
        image_data = None
        for key in possible_image_keys:
            if key in image_keys:
                image_data = get_data(image_mat, key, image_format)
                print(f"Found image data with key: '{key}'")
                break
        if image_data is None:
            image_data = get_data(image_mat, image_keys[0], image_format)
            print(f"Warning: Using first available key '{image_keys[0]}' from: {image_keys}")
    
    # Load ground truth data
    if len(gt_keys) == 0:
        raise ValueError("No data keys found in gt file.")
    elif len(gt_keys) == 1:
        ground_truth = get_data(gt_mat, gt_keys[0], gt_format)
        print(f"Using ground truth key: '{gt_keys[0]}'")
    else:
        possible_gt_keys = ['map', 'houston_gt', 'Houston_gt', 'Houston13_7gt', 'gt', 
                            'ground_truth', 'label', 'paviaU_gt', 'PaviaU_gt', 'pavia_gt',
                            'Pavia_gt', 'salinas_gt', 'Salinas_gt', 'indian_pines_gt', 'Indian_pines_gt', 'indiana_pines_gt', 'Indiana_pines_gt']
        ground_truth = None
        for key in possible_gt_keys:
            if key in gt_keys:
                ground_truth = get_data(gt_mat, key, gt_format)
                print(f"Found ground truth with key: '{key}'")
                break
        if ground_truth is None:
            ground_truth = get_data(gt_mat, gt_keys[0], gt_format)
            print(f"Warning: Using first available key '{gt_keys[0]}' from: {gt_keys}")
    
    image_data = np.array(image_data)
    ground_truth = np.array(ground_truth)
    
    if image_format == 'h5py' and len(image_data.shape) == 3:
        if image_data.shape[0] < image_data.shape[2]:
            image_data = np.transpose(image_data, (1, 2, 0))
            print("  Transposed image data from (C, H, W) to (H, W, C)")
    
    print(f"Image data shape: {image_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    
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


def save_results_to_files(model_name, data_name, results, model_params, gflops, training_time, save_path='./results'):
    """Save results to both text and CSV files."""
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to text file
    txt_filename = os.path.join(save_path, f"results_{model_name}_{data_name}.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"Model: {model_name} (w/o Global Attention)\n")
        f.write(f"Dataset: {data_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Parameters: {model_params:.2f} M\n")
        f.write(f"GFLOPs: {gflops:.2f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write("\n=== RESULTS ===\n")
        
        # Write overall accuracy
        f.write(f"Overall Accuracy: {results['oa']:.4f}\n")
        f.write(f"Average Accuracy: {results['aa']:.4f}\n")
        f.write(f"Kappa Coefficient: {results['kappa']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"Latency: {results['latency']:.4f} ms\n")
        f.write(f"Throughput: {results['throughput']:.2f} samples/sec\n")
    
    print(f"\n✓ Results saved to:")
    print(f"  Text file: {txt_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pavia', choices=['pavia', 'houston', 'salinas', 'indiana'])
    parser.add_argument('--save_path', type=str, default='./ablation_results', help='Directory to save results')
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
    
    print(f"\n✓ Data preprocessed successfully!")
    print(f"Spatial-spectral data shape: {spatial_spectral_data.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of spectral bands: {num_channels}")
    
    # Split dataset
    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    train_dataset = HyperspectralDataset(spatial_spectral_data[train_indices], y[train_indices])
    test_dataset = HyperspectralDataset(spatial_spectral_data[test_indices], y[test_indices])
    
    print(f"\n✓ Dataset split successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SpectralSpatialLinearTransformer(
        image_size=window_size,
        patch_size=patch_size,
        num_channels=num_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0
    )
    
    model.to(device)
    
    print(f"\n✓ Model initialized successfully!")
    print(f"Model: SSLT w/o Global Attention")
    print(f"Device: {device}")
    print(f"Number of parameters: {count_model_parameters(model):.2f} M")
    
    model_params = count_model_parameters(model)
    gflops = 0.0
    
    try:
        gflops = calculate_gflops(model, train_dataset, device)
        print(f"GFLOPs: {gflops:.2f}")
    except Exception as e:
        print(f"Warning: Could not calculate GFLOPs: {e}")
        gflops = 0.0
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_ablation_no_global_attn",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    print("\n✓ Trainer setup complete!")
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds!")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_dataset, test_indices, y, device)
    print_results(results)
    
    # Save results to files
    save_results_to_files("sslt_no_global_attn", data_name, results, model_params, gflops, training_time, args.save_path)


if __name__ == "__main__":
    main()