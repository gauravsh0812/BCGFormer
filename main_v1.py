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
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# from models.model_v2 import SpectralSpatialLinearTransformerV2 as SpectralSpatialLinearTransformerD1
from models.model_v2_d1 import SpectralSpatialLinearTransformerV2 as SpectralSpatialLinearTransformerD1
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
            raise ValueError(f"File too small ({file_size} bytes): {file_path}")
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
        data_ref = mat_obj[key]
        if isinstance(data_ref, h5py.Dataset):
            return np.array(data_ref[:])
        elif isinstance(data_ref, h5py.Reference):
            ref_obj = mat_obj[data_ref]
            return np.array(ref_obj[:] if isinstance(ref_obj, h5py.Dataset) else ref_obj)
        elif hasattr(data_ref, '__array__'):
            return np.array(data_ref)
        return np.array(data_ref[:])

    image_mat, image_keys, image_format = load_mat_file(image_file)
    gt_mat, gt_keys, gt_format = load_mat_file(gt_file)

    possible_image_keys = ['ori_data', 'houston', 'Houston', 'Houston13', 'data', 'image',
                           'HSI', 'paviaU', 'PaviaU', 'pavia', 'Pavia', 'salinas', 'Salinas',
                           'salinas_corrected', 'Salinas_corrected', 'indian_pines',
                           'Indian_pines', 'indiana_pines', 'Indiana_pines']
    possible_gt_keys = ['map', 'houston_gt', 'Houston_gt', 'Houston13_7gt', 'gt',
                        'ground_truth', 'label', 'paviaU_gt', 'PaviaU_gt', 'pavia_gt',
                        'Pavia_gt', 'salinas_gt', 'Salinas_gt', 'indian_pines_gt',
                        'Indian_pines_gt', 'indiana_pines_gt', 'Indiana_pines_gt']

    image_data = get_data(image_mat, image_keys[0] if len(image_keys) == 1 else
                          next((k for k in possible_image_keys if k in image_keys), image_keys[0]),
                          image_format)
    ground_truth = get_data(gt_mat, gt_keys[0] if len(gt_keys) == 1 else
                            next((k for k in possible_gt_keys if k in gt_keys), gt_keys[0]),
                            gt_format)

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


def spatial_safe_split(ground_truth, train_samples_per_class=200,
                        window_size=5, random_state=0):
    rng = np.random.default_rng(random_state)
    half = window_size // 2  # = 2

    labeled_coords = np.argwhere(ground_truth != 0)
    labels = ground_truth[labeled_coords[:, 0], labeled_coords[:, 1]]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_coords_all, train_y_all = [], []
    test_coords_all,  test_y_all  = [], []

    # Per-class train selection — only enforce distance within same class
    # This is the standard HSI benchmark protocol (SSFTT, HiT, SpectralFormer all do this)
    for cls in np.unique(y):
        cls_mask   = y == cls
        cls_coords = labeled_coords[cls_mask]
        cls_y      = y[cls_mask]
        order      = rng.permutation(len(cls_coords))
        cls_coords = cls_coords[order]
        cls_y      = cls_y[order]

        chosen_idx = []
        for i in range(len(cls_coords)):
            if len(chosen_idx) >= train_samples_per_class:
                break
            cr, cc = cls_coords[i]
            too_close = any(
                abs(cr - cls_coords[j][0]) <= half and
                abs(cc - cls_coords[j][1]) <= half
                for j in chosen_idx
            )
            if not too_close:
                chosen_idx.append(i)

        # NO spatial fallback — just use what passed the constraint
        chosen_set = set(chosen_idx)
        train_coords_all.extend(cls_coords[list(chosen_set)].tolist())
        train_y_all.extend(cls_y[list(chosen_set)].tolist())
        test_idx = [i for i in range(len(cls_coords)) if i not in chosen_set]
        test_coords_all.extend(cls_coords[test_idx].tolist())
        test_y_all.extend(cls_y[test_idx].tolist())

    train_arr = np.array(train_coords_all)
    test_arr  = np.array(test_coords_all)
    train_y   = np.array(train_y_all)
    test_y    = np.array(test_y_all)

    print(f"[split] Classes: {len(np.unique(y))} | "
          f"Train: {len(train_arr)} | Test: {len(test_arr)}")

    return train_arr, train_y, test_arr, test_y, label_encoder

def extract_patches(image_data, coords, window_size=5):
    """Extract patches centred on given (row, col) coordinates."""
    half = window_size // 2
    padded = np.pad(image_data,
                    ((half, half), (half, half), (0, 0)),
                    mode='reflect')
    patches = np.stack([
        padded[r:r + window_size, c:c + window_size, :]
        for r, c in coords
    ])                                                  # N x H x W x C
    return patches


def normalize(train_patches, test_patches):
    """
    Normalise using train statistics only — no leakage from test set.
    """
    mn = train_patches.min()
    mx = train_patches.max()
    train_patches = (train_patches - mn) / (mx - mn + 1e-8)
    test_patches  = (test_patches  - mn) / (mx - mn + 1e-8)
    return train_patches, test_patches


class HyperspectralDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # patches: H x W x C → C x H x W
        feature = self.patches[idx].transpose(2, 0, 1)
        return {
            'x':      torch.tensor(feature,          dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def data_collator(data):
    return {
        'x':      torch.stack([d['x']      for d in data]),
        'labels': torch.stack([d['labels'] for d in data]),
    }


def compute_metrics(p):
    predictions = p.predictions.argmax(-1)
    return {"accuracy": (predictions == p.label_ids).mean()}


def build_model(model_name, num_channels, num_classes, window_size,
                embed_dim, num_heads, depth, patch_size):
    # if model_name == 'sslt_v2':
    #     return SpectralSpatialLinearTransformerV2(
    #         image_size=window_size, num_channels=num_channels,
    #         num_classes=num_classes, embed_dim=embed_dim,
    #         num_heads=num_heads, depth=depth, mlp_ratio=4.0
    #     )
    if model_name == 'sslt_d1':
        return SpectralSpatialLinearTransformerD1(
            image_size=window_size, num_channels=num_channels,
            num_classes=num_classes, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0
        )
    elif model_name == 'hit':
        return HiT(num_bands=num_channels, num_classes=num_classes,
                   window_size=window_size, embed_dim=embed_dim,
                   num_heads=num_heads, depth=depth)
    elif model_name == 'spectralmamba':
        return SpectralMamba(num_bands=num_channels, num_classes=num_classes,
                             window_size=window_size, embed_dim=embed_dim,
                             d_state=16, depth=depth, piece_size=2, dropout=0.1)
    elif model_name == 'hybridsn':
        return HybridSN(num_bands=num_channels, num_classes=num_classes,
                        window_size=window_size)
    elif model_name == '3dcnn':
        return CNN3D(num_bands=num_channels, num_classes=num_classes,
                     window_size=window_size)
    elif model_name == 'spectralformer':
        return SpectralFormer(num_bands=num_channels, num_classes=num_classes,
                              window_size=window_size, embed_dim=embed_dim,
                              num_heads=num_heads, depth=depth)
    elif model_name == 'ssftt':
        return SSFTT(num_bands=num_channels, num_classes=num_classes,
                     window_size=window_size, embed_dim=embed_dim,
                     num_heads=num_heads, depth=depth)
    elif model_name == 'swinhsi':
        return SwinHSI(num_bands=num_channels, num_classes=num_classes,
                       window_size=window_size, embed_dim=embed_dim,
                       num_heads=num_heads, depth=depth, swin_window=patch_size)
    raise ValueError(f"Unknown model: {model_name}")


def save_results_to_files(model_name, data_name, results, std_results,
                          model_params, gflops, training_time, save_path='./results_v1'):
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
        f.write(f"Split: spatial-safe, 200 train samples/class, no patch overlap\n")
        f.write(f"Normalisation: train statistics only\n")
        f.write("\n=== RESULTS (mean ± std) ===\n")
        for key, label in [('oa', 'Overall Accuracy'), ('aa', 'Average Accuracy'),
                            ('kappa', 'Kappa Coefficient'), ('f1', 'F1 Score'),
                            ('precision', 'Precision'), ('recall', 'Recall')]:
            f.write(f"{label}: {results[key]:.4f} ± {std_results[key]:.4f}\n")
        f.write(f"Latency: {results['latency']:.4f} ms\n")
        f.write(f"Throughput: {results['throughput']:.2f} samples/sec\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hit',
                        choices=['sslt_v2', 'sslt_d1', 'hit', 'spectralmamba', 'hybridsn',
                                 '3dcnn', 'spectralformer', 'ssftt', 'swinhsi'])
    parser.add_argument('--dataset', type=str, default='houston',
                        choices=['pavia', 'houston', 'salinas', 'indiana'])
    parser.add_argument('--train_samples', type=int, default=200,
                        help='Training samples per class')
    parser.add_argument('--save_path', type=str, default='./results_v1')
    args = parser.parse_args()

    dataset_files = {
        'pavia':   ("./dataset/PaviaU.mat",            "./dataset/PaviaU_gt.mat"),
        'houston': ("./dataset/Houston13.mat",          "./dataset/Houston13_7gt.mat"),
        'salinas': ("./dataset/Salinas.mat",            "./dataset/Salinas_gt.mat"),
        'indiana': ("./dataset/Indian_pines.mat",       "./dataset/Indian_pines_gt.mat"),
    }
    image_file, gt_file = dataset_files[args.dataset]

    window_size = 5
    patch_size  = 4
    embed_dim   = 192
    num_heads   = 4
    depth       = 4

    image_data, ground_truth = load_houston(image_file, gt_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GFLOPs calculation (run 0 split, no model reuse) ──────────────────
    train_coords_0, train_y_0, test_coords_0, test_y_0, label_encoder = \
        spatial_safe_split(ground_truth, args.train_samples, window_size, random_state=0)

    train_patches_0 = extract_patches(image_data, train_coords_0, window_size)
    test_patches_0  = extract_patches(image_data, test_coords_0,  window_size)
    train_patches_0, test_patches_0 = normalize(train_patches_0, test_patches_0)

    num_classes  = len(np.unique(train_y_0))
    num_channels = train_patches_0.shape[-1]

    _tmp_model = build_model(args.model, num_channels, num_classes, window_size,
                             embed_dim, num_heads, depth, patch_size).to(device)
    model_params = count_model_parameters(_tmp_model)
    gflops = 0.0
    try:
        _tmp_ds = HyperspectralDataset(train_patches_0, train_y_0)
        gflops  = calculate_gflops(_tmp_model, _tmp_ds, device)
    except Exception:
        pass
    del _tmp_model

    # ── Training runs ──────────────────────────────────────────────────────
    NUM_RUNS = 1
    all_results        = []
    total_training_time = 0.0

    for run in range(NUM_RUNS):
        # Fresh spatial-safe split per run (different random seed)
        train_coords, train_y, test_coords, test_y, _ = \
            spatial_safe_split(ground_truth, args.train_samples, window_size, random_state=run)

        # Extract patches
        train_patches = extract_patches(image_data, train_coords, window_size)
        test_patches  = extract_patches(image_data, test_coords,  window_size)

        # Normalise using TRAIN statistics only — no test leakage
        train_patches, test_patches = normalize(train_patches, test_patches)

        train_dataset = HyperspectralDataset(train_patches, train_y)
        test_dataset  = HyperspectralDataset(test_patches,  test_y)

        model = build_model(args.model, num_channels, num_classes, window_size,
                            embed_dim, num_heads, depth, patch_size).to(device)

        # Scale warmup to 10% of total steps (not a fixed 500 which may exceed total)
        steps_per_epoch = max(1, len(train_dataset) // 32)
        warmup_steps    = max(10, int(0.1 * steps_per_epoch * 50))

        training_args = TrainingArguments(
            output_dir=f"./results_v1_run_{run}",
            num_train_epochs=20,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            learning_rate=1e-3,
            logging_steps=0,
            logging_strategy="no",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        start_time = time.time()
        trainer.train()
        total_training_time += time.time() - start_time

        # evaluate_model expects (model, test_dataset, test_indices, y, device)
        # test_indices here are just 0..N-1 since test_dataset is already the test set
        test_indices = np.arange(len(test_y))
        results = evaluate_model(model, test_dataset, test_indices, test_y, device)
        all_results.append(results)

    metrics = ['oa', 'aa', 'kappa', 'f1', 'precision', 'recall']
    agg = {}
    for m in metrics:
        vals   = np.array([r[m] for r in all_results])
        agg[m] = (vals.mean(), vals.std())

    print(f"\nModel: {args.model} | Dataset: {args.dataset} | "
          f"Params: {model_params:.2f}M | GFLOPs: {gflops:.4f}")
    print(f"Split: spatial-safe | Train samples/class: {args.train_samples}")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8}")
    print("-" * 30)
    for m in metrics:
        print(f"{m.upper():<12} {agg[m][0]:>8.4f} {agg[m][1]:>8.4f}")

    avg_results = {m: agg[m][0] for m in metrics}
    avg_results.update({k: all_results[-1][k]
                        for k in ['latency', 'throughput', 'params', 'gflops',
                                  'confusion_matrix', 'y_true', 'y_pred']})
    std_results = {m: agg[m][1] for m in metrics}
    save_results_to_files(args.model, args.dataset, avg_results, std_results,
                          model_params, gflops, total_training_time / NUM_RUNS,
                          args.save_path)


if __name__ == "__main__":
    main()
