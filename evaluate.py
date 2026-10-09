"""Evaluate the classifier on a labelled folder of chest X-rays.

Usage:
    python evaluate.py path/to/chest_xray/test

The folder must contain NORMAL/ and PNEUMONIA/ subfolders (the layout of the Kaggle dataset).
"""
import argparse
import os

import numpy as np

import analysis

CLASSES = ('NORMAL', 'PNEUMONIA')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def roc_auc(labels, scores):
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get average ranks)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_metrics(labels, scores, threshold=analysis.THRESHOLD):
    labels = np.asarray(labels)
    predicted = np.asarray(scores) > threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))

    def ratio(a, b):
        return a / b if b else float('nan')

    precision, sensitivity = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        'images': len(labels),
        'accuracy': ratio(tp + tn, len(labels)),
        'sensitivity': sensitivity,
        'specificity': ratio(tn, tn + fp),
        'precision': precision,
        'f1': ratio(2 * precision * sensitivity, precision + sensitivity),
        'roc_auc': roc_auc(labels, scores),
        'confusion': {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn},
    }


def load_dataset(data_dir):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            raise SystemExit(f"Expected a {name}/ folder inside {data_dir}")
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(folder, filename))
                labels.append(label)
    return paths, labels


def predict(model, paths, mode, batch_size=32):
    scores = []
    for start in range(0, len(paths), batch_size):
        batch = np.concatenate([analysis.preprocess(analysis.load_image(p), mode)
                                for p in paths[start:start + batch_size]])
        scores.extend(analysis.pneumonia_probability(model(batch, training=False).numpy()).tolist())
        print(f"\r{min(start + batch_size, len(paths))}/{len(paths)} images", end='', flush=True)
    print()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', help='folder with NORMAL/ and PNEUMONIA/ subfolders')
    parser.add_argument('--threshold', type=float, default=analysis.THRESHOLD)
    parser.add_argument('--preprocessing', choices=['auto', 'raw', 'rescale', 'mobilenet_v2'],
                        help='override PREPROCESSING to compare settings')
    args = parser.parse_args()

    paths, labels = load_dataset(args.data_dir)
    model = analysis.get_model()
    mode = analysis.resolve_preprocessing(model, args.preprocessing)
    m = compute_metrics(labels, predict(model, paths, mode), args.threshold)
    c = m['confusion']
    print(f"Images:       {m['images']}  (preprocessing: {mode}, threshold: {args.threshold})")
    for key in ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc'):
        print(f"{key.capitalize():<13} {m[key]:.4f}")
    print(f"Confusion:    TP={c['tp']} FN={c['fn']} FP={c['fp']} TN={c['tn']}")


if __name__ == '__main__':
    main()
