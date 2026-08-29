"""
Evaluate the trained tumor classifier and find a threshold that actually
respects "no" predictions instead of defaulting everything to "yes".

Usage:
    python evaluate_model.py --data_dir data --model_path models/tumor_model.h5
"""
import argparse

import numpy as np
import tensorflow as tf


def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def print_report(y_true, y_pred, class_names, label):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    # recall for the "no" class (index 0) is what matters here:
    no_recall = tn / (tn + fp) if (tn + fp) else float("nan")
    yes_recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"\n=== {label} ===")
    print(f"Confusion matrix (rows=actual, cols=predicted) [{class_names[0]}, {class_names[1]}]")
    print(f"  actual {class_names[0]:>5}: [{tn:4d}, {fp:4d}]")
    print(f"  actual {class_names[1]:>5}: [{fn:4d}, {tp:4d}]")
    print(f"Accuracy: {acc:.3f}")
    print(f"Recall on '{class_names[0]}' (how often true 'no' is correctly kept as 'no'): {no_recall:.3f}")
    print(f"Recall on '{class_names[1]}': {yes_recall:.3f}")
    if no_recall < 0.7:
        print(f"  -> Model is biased toward predicting '{class_names[1]}'. This is why real "
              f"'{class_names[0]}' scans keep showing up as '{class_names[1]}'.")


def main(data_dir, model_path, img_size=(128, 128), batch_size=32):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    class_names = val_ds.class_names
    print(f"Class names (index order): {class_names}")

    model = tf.keras.models.load_model(model_path)

    y_true, y_prob = [], []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(y.numpy().flatten().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Baseline: current hardcoded threshold
    y_pred_default = (y_prob > 0.5).astype(int)
    print_report(y_true, y_pred_default, class_names, "Current threshold: 0.5")

    # Sweep thresholds, pick the one that best balances both classes (Youden's J)
    best_threshold, best_j = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob > t).astype(int)
        tp, tn, fp, fn = confusion_counts(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        j = tpr - fpr
        if j > best_j:
            best_j, best_threshold = j, t

    print(f"\nSuggested threshold (best separates the two classes): {best_threshold:.2f}")
    y_pred_tuned = (y_prob > best_threshold).astype(int)
    print_report(y_true, y_pred_tuned, class_names, f"Tuned threshold: {best_threshold:.2f}")

    print(
        "\nNext step: open app.py, set THRESHOLD = "
        f"{best_threshold:.2f} instead of the hardcoded 0.5, and re-test."
    )
    print(
        "If 'no' recall is still low even at the tuned threshold, the model itself needs "
        "more/better 'no' training examples or a longer/rebalanced training run — "
        "a threshold shift can only trade off false positives vs false negatives, "
        "it can't fix a model that never really learned what a normal scan looks like."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path", default="models/tumor_model.h5")
    args = parser.parse_args()
    main(args.data_dir, args.model_path)