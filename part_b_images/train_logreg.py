"""Part B (Images): Dataset inspection, HSV feature extraction, stratified splitting, and scaling."""

from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .image_features import FEATURE_DIM, extract_flower_features
except ImportError:
    _PART_B_DIR = Path(__file__).resolve().parent
    if str(_PART_B_DIR) not in sys.path:
        sys.path.insert(0, str(_PART_B_DIR))
    from image_features import FEATURE_DIM, extract_flower_features

RANDOM_STATE = 42
IMG_SIZE = (128, 128)
BINS = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLOWERS_DIR = PROJECT_ROOT / "flowers"
SPLITS_PATH = Path(__file__).resolve().parent / "flowers_splits.npz"


def load_and_inspect_dataset(
    flowers_dir: Path = FLOWERS_DIR,
    img_size: tuple[int, int] = IMG_SIZE,
    bins: int = BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load raw flower images, inspect dimensions, and extract HSV features."""
    if not flowers_dir.is_dir():
        raise FileNotFoundError(
            f"Raw flowers dataset directory not found: {flowers_dir}"
        )

    class_names = sorted(d.name for d in flowers_dir.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"No class subdirectories found in {flowers_dir}")

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    dimensions: list[tuple[int, int]] = []
    class_counts: dict[str, int] = {}
    skipped_count = 0

    for class_idx, class_name in enumerate(class_names):
        class_dir = flowers_dir / class_name
        valid_in_class = 0

        for img_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_count += 1
                continue

            height, width = img.shape[:2]
            dimensions.append((height, width))

            feat = extract_flower_features(
                img, img_size=img_size, bins=bins, is_rgb=False
            )
            if feat.shape != (bins * 3,):
                raise ValueError(
                    f"Unexpected feature shape {feat.shape} for {img_path}"
                )

            features_list.append(feat)
            labels_list.append(class_idx)
            valid_in_class += 1

        class_counts[class_name] = valid_in_class

    X = np.asarray(features_list, dtype=np.float32)
    y = np.asarray(labels_list, dtype=np.int64)
    classes = np.asarray(class_names)

    dims_arr = np.asarray(dimensions, dtype=np.int64)
    heights = dims_arr[:, 0]
    widths = dims_arr[:, 1]
    min_hw = (int(heights.min()), int(widths.min()))
    max_hw = (int(heights.max()), int(widths.max()))
    median_hw = (int(np.median(heights)), int(np.median(widths)))
    top_dims = Counter(dimensions).most_common(5)

    stats = {
        "class_counts": class_counts,
        "total_valid": int(len(y)),
        "skipped_count": int(skipped_count),
        "min_hw": min_hw,
        "max_hw": max_hw,
        "median_hw": median_hw,
        "top_dims": top_dims,
        "resized_size": img_size,
        "feature_shape": X.shape,
    }

    print("=== Part B: Flowers Dataset Inspection ===")
    print(f"Dataset directory: {flowers_dir}")
    print(f"Sorted classes ({len(class_names)}): {class_names}")
    for cname, count in class_counts.items():
        print(f"  - {cname:<10s}: {count} valid images")
    print(f"Total valid images : {stats['total_valid']}")
    print(f"Skipped/corrupted  : {stats['skipped_count']}")
    print("Original dimensions (H, W):")
    print(f"  - Min (H, W)     : {min_hw}")
    print(f"  - Max (H, W)     : {max_hw}")
    print(f"  - Median (H, W)  : {median_hw}")
    print("  - Top 5 common   :")
    for hw, freq in top_dims:
        print(f"      {hw}: {freq} images")
    print(f"Resized size       : {img_size}")
    print(
        f"Feature matrix     : shape={X.shape}, dtype={X.dtype} "
        f"(expected dim={FEATURE_DIM})"
    )

    return X, y, classes, stats


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, np.ndarray], StandardScaler]:
    """Perform stratified 70/15/15 split and fit StandardScaler on train only."""
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    # Fit scaler strictly on training features to prevent validation/test leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    total = len(y)
    print("\n=== Stratified Split (70% / 15% / 15%) & Scaling ===")
    print(
        f"Train : {len(y_train):4d} ({len(y_train) / total * 100:.2f}%)\n"
        f"Val   : {len(y_val):4d} ({len(y_val) / total * 100:.2f}%)\n"
        f"Test  : {len(y_test):4d} ({len(y_test) / total * 100:.2f}%)"
    )
    print("Per-class split counts:")
    for idx, cname in enumerate(classes):
        tr_c = int(np.sum(y_train == idx))
        va_c = int(np.sum(y_val == idx))
        te_c = int(np.sum(y_test == idx))
        print(
            f"  - {str(cname):<10s}: train={tr_c:3d}, val={va_c:3d}, test={te_c:3d}"
        )

    non_const_mask = scaler.var_ > 0
    max_abs_mean = float(np.max(np.abs(X_train_scaled[:, non_const_mask].mean(axis=0))))
    mean_std = float(X_train_scaled[:, non_const_mask].std(axis=0).mean())
    print(
        f"Zero-leakage check : scaler.n_samples_seen_={int(scaler.n_samples_seen_)} "
        f"(train={len(X_train_raw)}), active_features={int(non_const_mask.sum())}/{X_train_raw.shape[1]}, "
        f"max|train_mean|={max_abs_mean:.2e}, avg_train_std={mean_std:.6f}"
    )

    splits = {
        "X_train_scaled": X_train_scaled.astype(np.float32),
        "X_val_scaled": X_val_scaled.astype(np.float32),
        "X_test_scaled": X_test_scaled.astype(np.float32),
        "X_train_raw": X_train_raw.astype(np.float32),
        "X_val_raw": X_val_raw.astype(np.float32),
        "X_test_raw": X_test_raw.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "y_val": y_val.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "classes": classes,
    }
    return splits, scaler


def save_splits(
    splits: dict[str, np.ndarray],
    output_path: Path = SPLITS_PATH,
) -> Path:
    """Export exact train/val/test arrays for teammate handoff."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **splits)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved splits archive: {output_path} ({size_mb:.3f} MB)")
    return output_path


def main() -> None:
    """Run Milestone 1: dataset inspection, HSV feature extraction, and splits export."""
    X, y, classes, _ = load_and_inspect_dataset()
    splits, _ = split_and_scale(X, y, classes)
    save_splits(splits)


if __name__ == "__main__":
    main()
