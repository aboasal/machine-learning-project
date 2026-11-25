from collections import Counter
from pathlib import Path
import sys

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__:
    from .image_features import FEATURE_DIM, extract_flower_features
else:
    _PART_B_DIR = Path(__file__).resolve().parent
    if str(_PART_B_DIR) not in sys.path:
        sys.path.insert(0, str(_PART_B_DIR))
    from image_features import FEATURE_DIM, extract_flower_features

RANDOM_STATE = 42
IMG_SIZE = (128, 128)
BINS = 32
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLOWERS_DIR = PROJECT_ROOT / "flowers"
SPLITS_PATH = Path(__file__).resolve().parent / "flowers_splits.npz"
MODELS_DIR = PROJECT_ROOT / "models"
CONFUSION_MATRIX_PATH = Path(__file__).resolve().parent / "logreg_confusion_matrix.png"


def load_and_inspect_dataset(
    flowers_dir: Path = FLOWERS_DIR,
    img_size: tuple[int, int] = IMG_SIZE,
    bins: int = BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **splits)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved splits archive: {output_path} ({size_mb:.3f} MB)")
    return output_path


def tune_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float] = C_VALUES,
    random_state: int = RANDOM_STATE,
) -> tuple[LogisticRegression, float, float, list[dict]]:
    results: list[dict] = []
    best_model = None
    best_c = None
    best_val_acc = -1.0

    for c in c_values:
        model = LogisticRegression(C=c, max_iter=1000, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train_scaled)))
        val_acc = float(accuracy_score(y_val, model.predict(X_val_scaled)))
        results.append({"C": c, "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c = c
            best_model = model

    print("\n=== Logistic Regression Validation Tuning ===")
    print(f"{'C':>8s} | {'Train Acc':>10s} | {'Val Acc':>10s} | Selected")
    print("-" * 46)
    for row in results:
        marker = "  <-- best" if row["C"] == best_c else ""
        print(
            f"{row['C']:8.2f} | {row['train_acc']:10.4f} | "
            f"{row['val_acc']:10.4f} |{marker}"
        )
    print(f"Best C: {best_c} | Best Validation Accuracy: {best_val_acc:.4f}")

    return best_model, best_c, best_val_acc, results


def evaluate_on_test(
    best_model: LogisticRegression,
    best_c: float,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    cm_path: Path = CONFUSION_MATRIX_PATH,
) -> dict:
    y_pred = best_model.predict(X_test_scaled)
    test_acc = float(accuracy_score(y_test, y_pred))
    class_labels = [str(c) for c in classes]
    report_text = classification_report(
        y_test, y_pred, target_names=class_labels, digits=4
    )
    cm = confusion_matrix(y_test, y_pred)

    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_title(
        f"Logistic Regression Confusion Matrix (C={best_c}, Test Acc={test_acc:.4f})"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    top_confusions: list[tuple[int, str, str, float]] = []
    for i, true_name in enumerate(class_labels):
        row_total = int(cm[i].sum())
        for j, pred_name in enumerate(class_labels):
            if i != j and cm[i, j] > 0:
                pct = float(cm[i, j]) / float(row_total) * 100.0 if row_total > 0 else 0.0
                top_confusions.append((int(cm[i, j]), true_name, pred_name, pct))
    top_confusions.sort(key=lambda item: item[0], reverse=True)

    print("\n=== Final Test Set Evaluation (Single Evaluation) ===")
    print(f"Selected Model : LogisticRegression(C={best_c}, max_iter=1000, random_state={RANDOM_STATE})")
    print(f"Test Accuracy  : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("\nClassification Report (Test Set):")
    print(report_text)
    print(f"Saved confusion matrix heatmap: {cm_path}")
    print("Top Confused Class Pairs (True -> Predicted):")
    for count, true_name, pred_name, pct in top_confusions[:8]:
        print(f"  - {true_name:<10s} -> {pred_name:<10s}: {count:2d} ({pct:5.1f}% of true {true_name})")

    return {
        "test_acc": test_acc,
        "report_text": report_text,
        "cm": cm,
        "top_confusions": top_confusions,
        "cm_path": cm_path,
    }


def export_artifacts(
    best_model: LogisticRegression,
    scaler: StandardScaler,
    classes: np.ndarray,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "image_logreg.joblib"
    scaler_path = models_dir / "image_scaler.joblib"
    classes_path = models_dir / "image_classes.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump([str(c) for c in classes], classes_path)

    exported = {
        "model": model_path,
        "scaler": scaler_path,
        "classes": classes_path,
    }
    print("\n=== Exported Model Artifacts (joblib) ===")
    for label, path in exported.items():
        size_kb = path.stat().st_size / 1024.0
        print(f"  - {label:<7s}: {path} ({size_kb:.2f} KB)")
    return exported


def print_report_summary(
    stats: dict,
    splits: dict[str, np.ndarray],
    best_c: float,
    best_val_acc: float,
    tuning_results: list[dict],
    eval_results: dict,
    exported_paths: dict[str, Path],
) -> None:
    print("\n=== Part B Report-Ready Summary ===")
    print(
        f"Dataset      : Flowers Recognition ({stats['total_valid']} valid images, "
        f"{stats['skipped_count']} skipped)"
    )
    print(
        f"Image sizes  : original min={stats['min_hw']}, max={stats['max_hw']}, "
        f"median={stats['median_hw']} -> resized={stats['resized_size']}"
    )
    print(
        f"Splits       : train={len(splits['y_train'])}, val={len(splits['y_val'])}, "
        f"test={len(splits['y_test'])} (stratified, random_state={RANDOM_STATE})"
    )
    print(
        f"Features     : {BINS}-bin L1-normalized HSV histograms -> {FEATURE_DIM}-dim vector "
        f"+ StandardScaler (fit on train only)"
    )
    tuning_str = ", ".join(f"C={r['C']}:{r['val_acc']:.4f}" for r in tuning_results)
    print(f"Val tuning   : [{tuning_str}] -> best C={best_c} (val_acc={best_val_acc:.4f})")
    print(f"Test result  : accuracy={eval_results['test_acc']:.4f} ({eval_results['test_acc'] * 100:.2f}%)")
    print("Top 5 errors : " + ", ".join(
        f"{t}->{p} ({n})" for n, t, p, _ in eval_results["top_confusions"][:5]
    ))
    print(f"Plot saved   : {eval_results['cm_path']}")
    print("Joblib files : " + ", ".join(str(p) for p in exported_paths.values()))


def main() -> None:
    X, y, classes, stats = load_and_inspect_dataset()
    splits, scaler = split_and_scale(X, y, classes)
    save_splits(splits)
    best_model, best_c, best_val_acc, tuning_results = tune_logistic_regression(
        splits["X_train_scaled"],
        splits["y_train"],
        splits["X_val_scaled"],
        splits["y_val"],
    )
    eval_results = evaluate_on_test(
        best_model,
        best_c,
        splits["X_test_scaled"],
        splits["y_test"],
        classes,
    )
    exported_paths = export_artifacts(best_model, scaler, classes)
    print_report_summary(
        stats,
        splits,
        best_c,
        best_val_acc,
        tuning_results,
        eval_results,
        exported_paths,
    )


if __name__ == "__main__":
    main()
