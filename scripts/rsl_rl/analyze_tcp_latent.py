"""Analyze TCP implicit latent CSV files against learned terrain prior means."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _latent_matrix(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, list[str]]:
    columns = [column for column in df.columns if column.startswith(f"{prefix}_")]
    if not columns:
        raise RuntimeError(f"No latent columns found with prefix '{prefix}_'.")
    columns = sorted(columns, key=lambda name: int(name.rsplit("_", 1)[1]))
    return df[columns].to_numpy(dtype=np.float32), columns


def _load_prior_means(checkpoint_path: str) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder_state = checkpoint.get("encoder_state_dict")
    if encoder_state is None:
        raise RuntimeError(f"Checkpoint does not contain encoder_state_dict: {checkpoint_path}")
    prior = encoder_state.get("terrain_prior_means")
    if prior is None:
        raise RuntimeError("Checkpoint encoder_state_dict does not contain terrain_prior_means.")
    return prior.detach().cpu().numpy().astype(np.float32)


def _valid_rows(df: pd.DataFrame, x: np.ndarray, num_classes: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = pd.to_numeric(df["terrain_type_id"], errors="coerce").to_numpy()
    valid = np.isfinite(labels) & (labels >= 0) & (labels < num_classes)
    labels = labels[valid].astype(np.int64)
    return df.loc[valid].reset_index(drop=True), x[valid], labels


def _nearest_prior_metrics(x: np.ndarray, labels: np.ndarray, prior_means: np.ndarray) -> dict[str, float]:
    diff = x[:, None, :] - prior_means[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    pred = distances.argmin(axis=1)
    true_distance = distances[np.arange(len(labels)), labels]
    nearest_distance = distances[np.arange(len(labels)), pred]

    metrics: dict[str, float] = {
        "nearest_prior_accuracy": float(accuracy_score(labels, pred)),
        "mean_true_prior_distance": float(true_distance.mean()),
        "mean_nearest_prior_distance": float(nearest_distance.mean()),
    }
    for class_id in range(prior_means.shape[0]):
        mask = labels == class_id
        if not mask.any():
            continue
        metrics[f"class_{class_id}_nearest_prior_accuracy"] = float(accuracy_score(labels[mask], pred[mask]))
        metrics[f"class_{class_id}_mean_true_prior_distance"] = float(true_distance[mask].mean())
    return metrics


def _distance_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    centroids = np.stack([x[labels == label].mean(axis=0) for label in unique_labels], axis=0)
    intra = []
    for centroid, label in zip(centroids, unique_labels):
        class_x = x[labels == label]
        intra.append(np.linalg.norm(class_x - centroid[None, :], axis=-1).mean())
    if len(centroids) > 1:
        centroid_diff = centroids[:, None, :] - centroids[None, :, :]
        centroid_dist = np.linalg.norm(centroid_diff, axis=-1)
        inter = centroid_dist[np.triu_indices(len(centroids), k=1)]
        mean_inter = float(inter.mean())
    else:
        mean_inter = float("nan")
    mean_intra = float(np.mean(intra)) if intra else float("nan")
    return {
        "mean_intra_class_distance": mean_intra,
        "mean_inter_centroid_distance": mean_inter,
        "inter_over_intra": float(mean_inter / mean_intra) if mean_intra > 0.0 else float("nan"),
    }


def _linear_probe_accuracy(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return float("nan")
    stratify = labels if counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="auto", random_state=seed),
    )
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TCP latent terrain alignment metrics.")
    parser.add_argument("--csv", required=True, help="Path to latent.csv produced by play.py.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing terrain_prior_means.")
    parser.add_argument("--latent", choices=("mu", "z"), default="mu", help="Latent source to analyze.")
    parser.add_argument("--label_column", default="terrain_type_id", help="Label column for silhouette/probe metrics.")
    parser.add_argument("--max_samples_per_terrain", type=int, default=10000, help="Subsample per terrain label; <=0 disables.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "terrain_type_id" not in df.columns:
        raise RuntimeError("CSV must contain terrain_type_id.")

    x, latent_columns = _latent_matrix(df, args.latent)
    prior_means = _load_prior_means(args.checkpoint)
    if x.shape[1] < prior_means.shape[1]:
        raise RuntimeError(f"Latent dim mismatch: csv={x.shape[1]} prior={prior_means.shape[1]}")
    if x.shape[1] != prior_means.shape[1]:
        x = x[:, : prior_means.shape[1]]
        latent_columns = latent_columns[: prior_means.shape[1]]

    df, x, terrain_labels = _valid_rows(df, x, prior_means.shape[0])
    if args.max_samples_per_terrain > 0:
        sampled = []
        for _label, group in df.groupby("terrain_type_id", group_keys=False):
            sampled.append(group.sample(min(len(group), args.max_samples_per_terrain), random_state=args.seed))
        df = pd.concat(sampled).reset_index(drop=True)
        x, _latent_columns = _latent_matrix(df, args.latent)
        if x.shape[1] != prior_means.shape[1]:
            x = x[:, : prior_means.shape[1]]
        _df, x, terrain_labels = _valid_rows(df, x, prior_means.shape[0])

    metrics: dict[str, float | int | str] = {
        "csv": args.csv,
        "checkpoint": args.checkpoint,
        "latent": args.latent,
        "num_samples": int(len(df)),
        "latent_dim": int(len(latent_columns)),
        "prior_dim": int(prior_means.shape[1]),
        "num_prior_classes": int(prior_means.shape[0]),
    }
    metrics.update(_nearest_prior_metrics(x, terrain_labels, prior_means))
    metrics.update(_distance_metrics(x, terrain_labels))

    label_values = pd.to_numeric(df[args.label_column], errors="coerce").to_numpy() if args.label_column in df else terrain_labels
    valid_label = np.isfinite(label_values)
    label_values = label_values[valid_label].astype(np.int64)
    x_for_label = x[valid_label]
    if len(np.unique(label_values)) > 1:
        x_scaled = StandardScaler().fit_transform(x_for_label)
        metrics[f"silhouette_{args.label_column}"] = float(silhouette_score(x_scaled, label_values))
    else:
        metrics[f"silhouette_{args.label_column}"] = float("nan")
    metrics[f"linear_probe_accuracy_{args.label_column}"] = _linear_probe_accuracy(x_for_label, label_values, args.seed)

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
