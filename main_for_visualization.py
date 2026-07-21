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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              f1_score, precision_score, recall_score)

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# ── Model imports ──────────────────────────────────────────────────────────────
from models.ablation_no_bcg import SpectralSpatialLinearTransformerV2_NoBCG as SpectralSpatialLinearTransformerV2
from models.hybridsn       import HybridSN
from models.swinhsi        import SwinHSI
from models.hit            import HiT
from models.ssftt          import SSFTT
from models.spectralformer import SpectralFormer
from models.spectralmamba  import SpectralMamba
from evaluation import count_model_parameters, calculate_gflops


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_mat(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        mat  = scipy.io.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        return mat, keys, 'scipy'
    except Exception:
        f = h5py.File(file_path, 'r')
        return f, list(f.keys()), 'h5py'


def _get_array(obj, key, fmt):
    if fmt == 'scipy':
        return np.array(obj[key])
    d = obj[key]
    return np.array(d[:] if isinstance(d, h5py.Dataset) else d)


def load_dataset(image_file, gt_file):
    _POSSIBLE_IMG = ['ori_data', 'houston', 'Houston', 'Houston13', 'data',
                     'image', 'HSI', 'paviaU', 'PaviaU', 'pavia', 'Pavia',
                     'salinas', 'Salinas', 'salinas_corrected',
                     'Salinas_corrected', 'indian_pines', 'Indian_pines',
                     'indiana_pines', 'Indiana_pines', 'WHU_Hi_HongHu',
                     'WHU_Hi_HanChuan', 'WHU_Hi_LongKou']
    _POSSIBLE_GT  = ['map', 'houston_gt', 'Houston_gt', 'Houston13_7gt',
                     'gt', 'ground_truth', 'label', 'paviaU_gt', 'PaviaU_gt',
                     'pavia_gt', 'Pavia_gt', 'salinas_gt', 'Salinas_gt',
                     'indian_pines_gt', 'Indian_pines_gt', 'indiana_pines_gt',
                     'Indiana_pines_gt', 'WHU_Hi_HongHu_gt', 'WHU_Hi_HanChuan_gt',
                     'WHU_Hi_LongKou_gt']

    img_mat, img_keys, img_fmt = _load_mat(image_file)
    gt_mat,  gt_keys,  gt_fmt  = _load_mat(gt_file)

    img_key = img_keys[0] if len(img_keys) == 1 else next(
        (k for k in _POSSIBLE_IMG if k in img_keys), img_keys[0])
    gt_key  = gt_keys[0]  if len(gt_keys)  == 1 else next(
        (k for k in _POSSIBLE_GT  if k in gt_keys),  gt_keys[0])

    image = _get_array(img_mat, img_key, img_fmt)
    gt    = _get_array(gt_mat,  gt_key,  gt_fmt)

    if img_fmt == 'h5py' and image.ndim == 3 and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if img_fmt == 'h5py': img_mat.close()
    if gt_fmt  == 'h5py': gt_mat.close()

    return image.astype(np.float32), gt.astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# HOUSTON 2013 — OFFICIAL FIXED SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def load_houston_official_split(gt_train_file, gt_test_file):
    def load_gt(path):
        mat, keys, fmt = _load_mat(path)
        _GT = ['map', 'houston_gt', 'Houston_gt', 'Houston13_7gt',
               'Houston13_7gt_test', 'gt', 'ground_truth', 'label']
        key = keys[0] if len(keys) == 1 else next(
            (k for k in _GT if k in keys), keys[0])
        arr = _get_array(mat, key, fmt)
        if fmt == 'h5py': mat.close()
        return arr.astype(np.int32)

    gt_tr = load_gt(gt_train_file)
    gt_te = load_gt(gt_test_file)

    train_coords = np.argwhere(gt_tr != 0)
    test_coords  = np.argwhere(gt_te != 0)

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(
        gt_tr[train_coords[:, 0], train_coords[:, 1]])
    test_y  = label_encoder.transform(
        gt_te[test_coords[:, 0],  test_coords[:, 1]])

    print(f"[Houston official split] "
          f"Train: {len(train_coords)} | Test: {len(test_coords)} | "
          f"Classes: {len(label_encoder.classes_)}")
    return train_coords, train_y, test_coords, test_y, label_encoder


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC SPATIAL-SAFE SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def spatial_safe_split(ground_truth, train_samples_per_class=200,
                       window_size=5, random_state=0):
    rng  = np.random.default_rng(random_state)
    half = window_size // 2

    labeled = np.argwhere(ground_truth != 0)
    labels  = ground_truth[labeled[:, 0], labeled[:, 1]]
    le      = LabelEncoder()
    y       = le.fit_transform(labels)

    train_coords, train_y_list = [], []
    test_coords,  test_y_list  = [], []

    for cls in np.unique(y):
        mask   = y == cls
        coords = labeled[mask]
        ys_cls = y[mask]
        order  = rng.permutation(len(coords))
        coords = coords[order]
        ys_cls = ys_cls[order]

        chosen = []
        for i in range(len(coords)):
            if len(chosen) >= train_samples_per_class:
                break
            cr, cc = coords[i]
            too_close = any(
                abs(cr - coords[j][0]) <= half and
                abs(cc - coords[j][1]) <= half
                for j in chosen
            )
            if not too_close:
                chosen.append(i)

        chosen_set = set(chosen)
        train_coords.extend(coords[list(chosen_set)].tolist())
        train_y_list.extend(ys_cls[list(chosen_set)].tolist())
        rest = [i for i in range(len(coords)) if i not in chosen_set]
        test_coords.extend(coords[rest].tolist())
        test_y_list.extend(ys_cls[rest].tolist())

    train_arr = np.array(train_coords)
    test_arr  = np.array(test_coords)
    train_y   = np.array(train_y_list)
    test_y    = np.array(test_y_list)

    print(f"[split seed={random_state}] "
          f"Classes: {len(np.unique(y))} | "
          f"Train: {len(train_arr)} (~{len(train_arr)//len(np.unique(y))}/cls) | "
          f"Test: {len(test_arr)}")

    if len(test_arr) == 0:
        raise ValueError(
            "Zero test samples. Reduce train_samples_per_class or window_size.")

    return train_arr, train_y, test_arr, test_y, le


# ══════════════════════════════════════════════════════════════════════════════
# PATCH EXTRACTION & NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def extract_patches(image_data, coords, window_size=5):
    half   = window_size // 2
    padded = np.pad(image_data,
                    ((half, half), (half, half), (0, 0)),
                    mode='reflect')
    patches = np.stack([
        padded[r:r + window_size, c:c + window_size, :]
        for r, c in coords
    ])
    return patches


def normalize(train_patches, test_patches):
    mn = train_patches.min()
    mx = train_patches.max()
    train_patches = (train_patches - mn) / (mx - mn + 1e-8)
    test_patches  = (test_patches  - mn) / (mx - mn + 1e-8)
    return train_patches, test_patches


# ══════════════════════════════════════════════════════════════════════════════
# DATASET & COLLATOR
# ══════════════════════════════════════════════════════════════════════════════

class HyperspectralDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = torch.tensor(
            patches.transpose(0, 3, 1, 2), dtype=torch.float32)
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'x': self.patches[idx], 'labels': self.labels[idx]}


def data_collator(batch):
    return {
        'x':      torch.stack([b['x']      for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_dataset, val_dataset, device,
                epochs=50, batch_size=32, lr=3e-4, weight_decay=1e-2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64,
                              shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)

    total_steps  = epochs * len(train_loader)
    warmup_steps = max(10, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc   = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x      = batch['x'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            out  = model(x, labels)
            loss = (out['loss'] if 'loss' in out
                    else criterion(out['logits'], labels))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                x      = batch['x'].to(device)
                labels = batch['labels'].to(device)
                out    = model(x)
                preds  = out['logits'].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        acc = accuracy_score(all_true, all_preds)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_dataset, device, batch_size=64):
    loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    all_preds, all_true = [], []
    latency_ms = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x      = batch['x'].to(device)
            labels = batch['labels']
            if i == 0:
                for _ in range(10):
                    _ = model(x[:1])
                t0 = time.perf_counter()
                for _ in range(100):
                    _ = model(x[:1])
                latency_ms = (time.perf_counter() - t0) / 100 * 1000

            out   = model(x)
            preds = out['logits'].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)

    oa  = accuracy_score(all_true, all_preds)
    aa  = np.mean([
        accuracy_score(all_true[all_true == c], all_preds[all_true == c])
        for c in np.unique(all_true)
    ])
    k   = cohen_kappa_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average='weighted', zero_division=0)
    pr  = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    rc  = recall_score(all_true, all_preds, average='weighted', zero_division=0)
    thr = 1000.0 / latency_ms if latency_ms else 0.0

    return dict(oa=oa, aa=aa, kappa=k, f1=f1, precision=pr, recall=rc,
                latency=latency_ms, throughput=thr)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION MAP
# ══════════════════════════════════════════════════════════════════════════════

def generate_classification_map(model, image_data, ground_truth, device,
                                 window_size=5, batch_size=256,
                                 save_path='./results_avg',
                                 model_name='model',
                                 dataset_name='dataset',
                                 run=0):
    """
    Predicts every labeled pixel in the full image and saves a
    side-by-side ground truth / prediction false-colour map.
    Unlabeled pixels (gt == 0) are shown in black.
    """
    H, W, C = image_data.shape
    half    = window_size // 2
    padded  = np.pad(image_data,
                     ((half, half), (half, half), (0, 0)),
                     mode='reflect')

    all_coords = np.argwhere(ground_truth != 0)

    mn = image_data.min()
    mx = image_data.max()

    patches = np.stack([
        (padded[r:r + window_size, c:c + window_size, :] - mn) / (mx - mn + 1e-8)
        for r, c in all_coords
    ]).transpose(0, 3, 1, 2).astype(np.float32)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            x    = torch.tensor(patches[i:i + batch_size]).to(device)
            out  = model(x)
            pred = out['logits'].argmax(dim=-1).cpu().numpy()
            all_preds.extend(pred)
    all_preds = np.array(all_preds)

    # build maps — background = 0, classes = 1..K
    pred_map = np.zeros((H, W), dtype=np.int32)
    gt_map   = np.zeros((H, W), dtype=np.int32)
    for idx, (r, c) in enumerate(all_coords):
        pred_map[r, c] = all_preds[idx] + 1
        gt_map[r, c]   = ground_truth[r, c]

    num_classes = int(max(pred_map.max(), gt_map.max()))

    # distinct colour palette — index 0 always black (background)
    base_colors = [
        '#000000',                          # background
        '#e6194b', '#3cb44b', '#ffe119',    # classes 1-3
        '#4363d8', '#f58231', '#911eb4',    # classes 4-6
        '#42d4f4', '#f032e6', '#bfef45',    # classes 7-9
        '#fabed4', '#469990', '#dcbeff',    # classes 10-12
        '#9A6324', '#fffac8', '#800000',    # classes 13-15
        '#aaffc3', '#808000', '#ffd8b1',    # classes 16-18
        '#000075', '#a9a9a9',               # classes 19-20
    ]
    while len(base_colors) <= num_classes:
        np.random.seed(len(base_colors))
        base_colors.append('#%06x' % np.random.randint(0, 0xFFFFFF))

    cmap   = mcolors.ListedColormap(base_colors[:num_classes + 1])
    bounds = np.arange(-0.5, num_classes + 1.5, 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    im0 = axes[0].imshow(gt_map,   cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title('Ground Truth', fontsize=13, fontweight='bold', pad=8)
    axes[0].axis('off')

    im1 = axes[1].imshow(pred_map, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title('BCG-Former Prediction', fontsize=13,
                      fontweight='bold', pad=8)
    axes[1].axis('off')

    # shared colourbar
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02,
                        ticks=np.arange(0, num_classes + 1))
    cbar.ax.set_yticklabels(
        ['Background'] + [f'Class {i}' for i in range(1, num_classes + 1)],
        fontsize=8)

    plt.suptitle(
        f'{dataset_name.upper()} — Classification Map (Run {run + 1})',
        fontsize=14, fontweight='bold', y=1.01)

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(
        save_path,
        f'classmap_{model_name}_{dataset_name}_run{run + 1}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[classmap] Saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name, num_channels, num_classes, window_size,
                embed_dim, num_heads, depth, patch_size):
    if model_name == 'sslt_d1':
        return SpectralSpatialLinearTransformerV2(
            image_size=window_size, num_channels=num_channels,
            num_classes=num_classes, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0)
    elif model_name == 'hit':
        return HiT(num_bands=num_channels, num_classes=num_classes,
                   window_size=window_size, embed_dim=embed_dim,
                   num_heads=num_heads, depth=depth)
    elif model_name == 'spectralmamba':
        return SpectralMamba(num_bands=num_channels, num_classes=num_classes,
                             window_size=window_size, embed_dim=embed_dim,
                             d_state=16, depth=depth, piece_size=2,
                             dropout=0.1)
    elif model_name == 'hybridsn':
        return HybridSN(num_bands=num_channels, num_classes=num_classes,
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
                       num_heads=num_heads, depth=depth,
                       swin_window=patch_size)
    raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# RESULT SAVING
# ══════════════════════════════════════════════════════════════════════════════

def save_results(model_name, data_name, agg, model_params, gflops,
                 avg_train_time, save_path):
    os.makedirs(save_path, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_path,
                        f"results_{model_name}_{data_name}.txt")
    with open(path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {data_name}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Parameters: {model_params:.4f} M\n")
        f.write(f"GFLOPs: {gflops:.4f}\n")
        f.write(f"Avg Training Time: {avg_train_time:.2f} seconds\n")
        f.write(f"Protocol: 5 runs, spatial-safe split, "
                f"train-stats normalisation\n\n")
        f.write("=== RESULTS (mean ± std over 5 runs) ===\n")
        for key, label in [
            ('oa',        'Overall Accuracy'),
            ('aa',        'Average Accuracy'),
            ('kappa',     'Kappa'),
            ('f1',        'F1'),
            ('precision', 'Precision'),
            ('recall',    'Recall'),
        ]:
            m, s = agg[key]
            f.write(f"{label}: {m:.4f} ± {s:.4f}\n")
        f.write(f"Latency: {agg['latency'][0]:.4f} ms\n")
        f.write(f"Throughput: {agg['throughput'][0]:.2f} samples/sec\n")
    print(f"[saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sslt_d1',
                        choices=['sslt_d1', 'hit', 'spectralmamba', 'hybridsn',
                                 '3dcnn', 'spectralformer', 'ssftt', 'swinhsi'])
    parser.add_argument('--dataset', type=str, default='houston',
                        choices=['pavia', 'houston', 'houston18', 'salinas',
                                 'indiana', 'honghu', 'hanchuan', 'longkou'])
    parser.add_argument('--train_samples', type=int,  default=200)
    parser.add_argument('--epochs',        type=int,  default=20)
    parser.add_argument('--batch_size',    type=int,  default=32)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--num_runs',      type=int,  default=1)
    parser.add_argument('--save_path',     type=str,  default='./results_avg')
    parser.add_argument('--houston_train_gt', type=str,
                        default='./dataset/Houston13_7gt.mat')
    parser.add_argument('--houston_test_gt',  type=str,
                        default='./dataset/Houston13_7gt_test.mat')
    parser.add_argument('--save_maps', action='store_true',
                        help='Save classification maps (last run only)')
    args = parser.parse_args()

    dataset_files = {
        'pavia':     ("./dataset/PaviaU.mat",      "./dataset/PaviaU_gt.mat"),
        'houston':   ("./dataset/Houston13.mat",    args.houston_train_gt),
        'houston18': ("./dataset/Houston18.mat",    "./dataset/Houston18_gt.mat"),
        'salinas':   ("./dataset/Salinas.mat",      "./dataset/Salinas_gt.mat"),
        'indiana':   ("./dataset/Indian_pines.mat", "./dataset/Indian_pines_gt.mat"),
        'honghu':    ("./dataset/WHU/WHU-Hi-HongHu/WHU_Hi_HongHu.mat", "./dataset/WHU/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat"),
        'hanchuan':  ("./dataset/WHU/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat", "./dataset/WHU/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat"),
        'longkou':   ("./dataset/WHU/WHU-Hi-LongKou/WHU_Hi_LongKou.mat", "./dataset/WHU/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat"),
    }

    window_size = 5
    patch_size  = 4
    embed_dim   = 64
    num_heads   = 4
    depth       = 3

    image_file, gt_file = dataset_files[args.dataset]
    image_data, ground_truth = load_dataset(image_file, gt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Image: {image_data.shape} | "
          f"GT unique: {np.unique(ground_truth[ground_truth != 0])}")

    use_official_houston = (
        args.dataset == 'houston' and
        os.path.exists(args.houston_test_gt)
    )
    if use_official_houston:
        print("[Houston] Using official GRSS 2013 fixed split.")
    else:
        if args.dataset == 'houston':
            print(f"[Houston] Official test GT not found at "
                  f"{args.houston_test_gt}. "
                  f"Falling back to spatial-safe random split.")
        print(f"[Split] spatial-safe, "
              f"{args.train_samples} train/class, {args.num_runs} runs")

    # initial split for param / FLOP counting
    if use_official_houston:
        tr_c, tr_y, te_c, te_y, le = load_houston_official_split(
            gt_file, args.houston_test_gt)
    else:
        tr_c, tr_y, te_c, te_y, le = spatial_safe_split(
            ground_truth, args.train_samples, window_size, random_state=0)

    num_classes  = len(np.unique(tr_y))
    num_channels = image_data.shape[-1]

    tmp_model = build_model(
        args.model, num_channels, num_classes,
        window_size, embed_dim, num_heads, depth, patch_size
    ).to(device)
    model_params = count_model_parameters(tmp_model)
    gflops = 0.0
    try:
        tr_p      = extract_patches(image_data, tr_c[:64], window_size)
        tr_p, _   = normalize(tr_p, tr_p)
        tmp_ds    = HyperspectralDataset(tr_p, tr_y[:64])
        gflops    = calculate_gflops(tmp_model, tmp_ds, device)
    except Exception as e:
        print(f"[GFLOPs] Failed: {e}")
    del tmp_model

    print(f"\nModel : {args.model} | "
          f"Params: {model_params:.4f} M | "
          f"GFLOPs: {gflops:.4f}")
    print(f"Channels: {num_channels} | Classes: {num_classes}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING RUNS
    # ══════════════════════════════════════════════════════════════════════════
    all_results   = []
    total_tr_time = 0.0
    NUM_RUNS      = args.num_runs

    for run in range(NUM_RUNS):
        print(f"── Run {run + 1}/{NUM_RUNS} "
              f"{'─' * 40}")

        if use_official_houston:
            train_coords, train_y = tr_c, tr_y
            test_coords,  test_y  = te_c, te_y
        else:
            train_coords, train_y, test_coords, test_y, _ = spatial_safe_split(
                ground_truth, args.train_samples,
                window_size, random_state=run)

        train_patches = extract_patches(image_data, train_coords, window_size)
        test_patches  = extract_patches(image_data, test_coords,  window_size)
        train_patches, test_patches = normalize(train_patches, test_patches)

        train_ds = HyperspectralDataset(train_patches, train_y)
        test_ds  = HyperspectralDataset(test_patches,  test_y)

        torch.manual_seed(run)
        model = build_model(
            args.model, num_channels, num_classes,
            window_size, embed_dim, num_heads, depth, patch_size
        ).to(device)

        t0 = time.time()
        model, best_val_acc = train_model(
            model, train_ds, test_ds, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=1e-2)
        tr_time        = time.time() - t0
        total_tr_time += tr_time

        results = evaluate(model, test_ds, device)
        all_results.append(results)

        print(f"   Best val acc : {best_val_acc:.4f} | "
              f"Test OA: {results['oa']:.4f} | "
              f"Kappa: {results['kappa']:.4f} | "
              f"Latency: {results['latency']:.2f} ms | "
              f"Time: {tr_time:.1f}s")

        # classification map — last run only, opt-in via --save_maps
        if args.save_maps and run == NUM_RUNS - 1:
            generate_classification_map(
                model=model,
                image_data=image_data,
                ground_truth=ground_truth,
                device=device,
                window_size=window_size,
                batch_size=256,
                save_path=args.save_path,
                model_name=args.model,
                dataset_name=args.dataset,
                run=run,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATION & REPORTING
    # ══════════════════════════════════════════════════════════════════════════
    metrics = ['oa', 'aa', 'kappa', 'f1', 'precision', 'recall',
               'latency', 'throughput']
    agg = {}
    for m in metrics:
        vals   = np.array([r[m] for r in all_results])
        agg[m] = (float(vals.mean()), float(vals.std()))

    print(f"\n{'═' * 50}")
    print(f"Model: {args.model} | "
          f"Dataset: {args.dataset} | "
          f"Runs: {NUM_RUNS}")
    print(f"{'Metric':<14} {'Mean':>8} {'Std':>8}")
    print("─" * 32)
    for m in ['oa', 'aa', 'kappa', 'f1', 'precision', 'recall']:
        print(f"{m.upper():<14} {agg[m][0]:>8.4f} {agg[m][1]:>8.4f}")
    print(f"{'LATENCY(ms)':<14} {agg['latency'][0]:>8.4f}")
    print(f"{'THROUGHPUT':<14} {agg['throughput'][0]:>8.1f} samples/sec")
    print(f"{'═' * 50}\n")

    save_results(args.model, args.dataset, agg, model_params, gflops,
                 total_tr_time / NUM_RUNS, args.save_path)


if __name__ == "__main__":
    main()