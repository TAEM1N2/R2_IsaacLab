"""Plot PCA or t-SNE projections from latent CSV files recorded by play.py."""

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _sample_per_label(df: pd.DataFrame, label_column: str, max_samples_per_label: int) -> pd.DataFrame:
    if max_samples_per_label <= 0:
        return df
    samples = []
    for _label, group in df.groupby(label_column, group_keys=False):
        samples.append(group.sample(min(len(group), max_samples_per_label), random_state=0))
    return pd.concat(samples).reset_index(drop=True)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items] if items else None


def _apply_position_labels(
    df: pd.DataFrame,
    axis: str,
    segment_names: list[str],
    segment_bounds: list[float],
    output_column: str,
) -> pd.DataFrame:
    if len(segment_bounds) != len(segment_names) - 1:
        raise RuntimeError(
            "--position_label_bounds length must be one less than --position_label_segments length: "
            f"got bounds={len(segment_bounds)} segments={len(segment_names)}"
        )

    position_column = f"root_{axis}"
    if position_column not in df.columns:
        raise RuntimeError(f"CSV must contain position column '{position_column}' for position-derived labels.")

    df = df.copy()
    values = pd.to_numeric(df[position_column], errors="coerce")
    labels = []
    for value in values:
        if pd.isna(value):
            labels.append("unknown_position")
            continue
        segment_idx = 0
        while segment_idx < len(segment_bounds) and value >= segment_bounds[segment_idx]:
            segment_idx += 1
        labels.append(segment_names[segment_idx] if segment_idx < len(segment_names) else f"unknown_segment_{segment_idx}")
    df[output_column] = labels
    return df


def _numeric_color_columns(df: pd.DataFrame, latent_prefix: str) -> list[str]:
    excluded_exact = {
        "step",
        "env_id",
        "terrain_type_id",
        "terrain_level",
        "terrain_segment_value",
    }
    excluded_prefixes = (f"{latent_prefix}_", "mu_", "z_", "logvar_")
    columns = []
    for column in df.columns:
        if column in excluded_exact or column.startswith(excluded_prefixes):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any() and numeric.nunique(dropna=True) > 1:
            columns.append(column)
    return columns


def _fit_projection(
    df: pd.DataFrame,
    latent_prefix: str,
    method: str,
    tsne_perplexity: float,
    tsne_learning_rate: str | float,
    tsne_random_state: int,
    latent_start: int = 0,
    latent_end: int | None = None,
):
    latent_cols = [column for column in df.columns if column.startswith(f"{latent_prefix}_")]
    if not latent_cols:
        raise RuntimeError(f"No latent columns found with prefix '{latent_prefix}_'")
    latent_cols = sorted(latent_cols, key=lambda name: int(name.rsplit("_", 1)[1]))
    latent_cols = latent_cols[latent_start:latent_end]
    if not latent_cols:
        raise RuntimeError(f"No latent columns selected with range [{latent_start}:{latent_end}].")

    x = df[latent_cols].to_numpy()
    x_scaled = StandardScaler().fit_transform(x)
    n_components = min(3, len(latent_cols))
    if method == "pca":
        projection = PCA(n_components=n_components)
        x_projected = projection.fit_transform(x_scaled)
    elif method == "tsne":
        if len(df) <= 3:
            raise RuntimeError("t-SNE requires more than 3 samples.")
        effective_perplexity = min(tsne_perplexity, max(1.0, (len(df) - 1) / 3))
        projection = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            learning_rate=tsne_learning_rate,
            init="pca",
            random_state=tsne_random_state,
        )
        x_projected = projection.fit_transform(x_scaled)
    else:
        raise RuntimeError(f"Unsupported projection method: {method}")
    return latent_cols, x_scaled, projection, x_projected


def _plot_label_projection(
    df: pd.DataFrame,
    x_scaled,
    x_projected,
    projection,
    method: str,
    latent_prefix: str,
    label_column: str,
    output_path: str,
    axis_pair: tuple[int, int] = (0, 1),
):
    labels = df[label_column].to_numpy()
    unique_labels = sorted(df[label_column].unique())
    silhouette = silhouette_score(x_scaled, labels) if len(unique_labels) > 1 else float("nan")

    plt.figure(figsize=(9, 7))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(x_projected[mask, axis_pair[0]], x_projected[mask, axis_pair[1]], s=5, alpha=0.35, label=label)
    plt.legend(markerscale=3)
    _finish_plot(projection, method, latent_prefix, label_column, silhouette, output_path, axis_pair)
    return silhouette


def _plot_continuous_projection(
    df: pd.DataFrame,
    x_projected,
    projection,
    method: str,
    latent_prefix: str,
    color_column: str,
    output_path: str,
    axis_pair: tuple[int, int] = (0, 1),
) -> None:
    colors = pd.to_numeric(df[color_column], errors="coerce").to_numpy()

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        x_projected[:, axis_pair[0]], x_projected[:, axis_pair[1]], c=colors, s=5, alpha=0.55, cmap="viridis"
    )
    plt.colorbar(scatter, label=color_column)
    _finish_plot(projection, method, latent_prefix, color_column, float("nan"), output_path, axis_pair)


def _axis_label(projection, method: str, axis_idx: int) -> str:
    if method == "pca":
        return f"PC{axis_idx + 1} ({projection.explained_variance_ratio_[axis_idx] * 100:.1f}%)"
    return f"t-SNE {axis_idx + 1}"


def _finish_plot(
    projection,
    method: str,
    latent_prefix: str,
    title_suffix: str,
    silhouette: float,
    output_path: str,
    axis_pair: tuple[int, int],
) -> None:
    method_label = method.upper() if method == "pca" else "t-SNE"
    plt.xlabel(_axis_label(projection, method, axis_pair[0]))
    plt.ylabel(_axis_label(projection, method, axis_pair[1]))
    plt.title(f"Implicit latent {method_label} ({latent_prefix}, {title_suffix}), silhouette={silhouette:.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {method_label} plot to: {output_path}")


def _plot_label_projection_3d(
    df: pd.DataFrame,
    x_projected,
    projection,
    method: str,
    latent_prefix: str,
    label_column: str,
    output_path: str,
) -> None:
    if x_projected.shape[1] < 3:
        return

    labels = df[label_column].to_numpy()
    unique_labels = sorted(df[label_column].unique())
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label in unique_labels:
        mask = labels == label
        ax.scatter(x_projected[mask, 0], x_projected[mask, 1], x_projected[mask, 2], s=5, alpha=0.35, label=label)
    method_label = method.upper() if method == "pca" else "t-SNE"
    ax.set_xlabel(_axis_label(projection, method, 0))
    ax.set_ylabel(_axis_label(projection, method, 1))
    ax.set_zlabel(_axis_label(projection, method, 2))
    ax.set_title(f"Implicit latent {method_label} 3D ({latent_prefix}, {label_column})")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {method_label} plot to: {output_path}")


def _plot_continuous_projection_3d(
    df: pd.DataFrame,
    x_projected,
    projection,
    method: str,
    latent_prefix: str,
    color_column: str,
    output_path: str,
) -> None:
    if x_projected.shape[1] < 3:
        return

    colors = pd.to_numeric(df[color_column], errors="coerce").to_numpy()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x_projected[:, 0], x_projected[:, 1], x_projected[:, 2], c=colors, s=5, alpha=0.55, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label=color_column)
    method_label = method.upper() if method == "pca" else "t-SNE"
    ax.set_xlabel(_axis_label(projection, method, 0))
    ax.set_ylabel(_axis_label(projection, method, 1))
    ax.set_zlabel(_axis_label(projection, method, 2))
    ax.set_title(f"Implicit latent {method_label} 3D ({latent_prefix}, {color_column})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {method_label} plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot terrain-colored PCA or t-SNE from recorded implicit latent vectors.")
    parser.add_argument("--csv", required=True, help="Path to latent.csv generated by play.py.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults next to the CSV file.")
    parser.add_argument("--latent", choices=["mu", "z"], default="mu", help="Latent source to project.")
    parser.add_argument("--latent_start", type=int, default=0, help="First latent index to include in the projection.")
    parser.add_argument("--latent_end", type=int, default=None, help="Exclusive latent index to include in the projection.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Projection method.")
    parser.add_argument(
        "--label_column",
        default=None,
        help="CSV column used for point colors. Defaults to terrain_segment if present, else terrain_type.",
    )
    parser.add_argument(
        "--color_column",
        default=None,
        help="Numeric CSV column used as a continuous color map. If omitted, all numeric diagnostic columns are plotted.",
    )
    parser.add_argument(
        "--max_samples_per_terrain",
        type=int,
        default=5000,
        help="Subsample each terrain to at most this many rows. Set <=0 to disable.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity before sample-size clamping.")
    parser.add_argument("--tsne_learning_rate", default="auto", help="t-SNE learning rate, or 'auto'.")
    parser.add_argument("--tsne_random_state", type=int, default=0, help="t-SNE random seed.")
    parser.add_argument(
        "--position_label_axis",
        choices=["x", "y", "z"],
        default=None,
        help="Override the label column by binning root position on this axis.",
    )
    parser.add_argument(
        "--position_label_segments",
        default=None,
        help="Comma-separated labels for position bins, for example 'stair,wave,rough,slope'.",
    )
    parser.add_argument(
        "--position_label_bounds",
        default=None,
        help="Comma-separated boundaries between position bins. Length must be segments-1.",
    )
    args = parser.parse_args()
    if args.tsne_learning_rate != "auto":
        args.tsne_learning_rate = float(args.tsne_learning_rate)

    df = pd.read_csv(args.csv)
    if args.position_label_axis is not None:
        position_segments = _parse_csv_strings(args.position_label_segments)
        position_bounds = _parse_csv_floats(args.position_label_bounds)
        if position_segments is None or position_bounds is None:
            raise RuntimeError(
                "--position_label_segments and --position_label_bounds are required with --position_label_axis."
            )
        label_column_name = args.label_column or "terrain_segment"
        df = _apply_position_labels(df, args.position_label_axis, position_segments, position_bounds, label_column_name)
        args.label_column = label_column_name
    label_column = args.label_column

    output_dir = args.output if args.output is not None else os.path.dirname(args.csv)
    if output_dir.endswith(".png"):
        output_dir = os.path.dirname(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if label_column is None:
        label_column = "terrain_segment" if "terrain_segment" in df.columns else "terrain_type"
    if label_column not in df.columns:
        raise RuntimeError(f"CSV must contain label column '{label_column}'.")

    label_df = _sample_per_label(df, label_column, args.max_samples_per_terrain)
    _latent_cols, x_scaled, projection, x_projected = _fit_projection(
        label_df,
        args.latent,
        args.method,
        args.tsne_perplexity,
        args.tsne_learning_rate,
        args.tsne_random_state,
        args.latent_start,
        args.latent_end,
    )
    label_silhouette = float("nan")
    axis_pairs = [(0, 1)]
    if x_projected.shape[1] >= 3:
        axis_pairs += [(0, 2), (1, 2)]
    axis_prefix = "pc" if args.method == "pca" else "tsne"
    for axis_pair in axis_pairs:
        pair_name = f"{axis_prefix}{axis_pair[0] + 1}_{axis_prefix}{axis_pair[1] + 1}"
        label_output = os.path.join(
            output_dir,
            f"latent_{args.method}_{args.latent}_{_safe_filename(label_column)}_{pair_name}.png",
        )
        label_silhouette = _plot_label_projection(
            label_df, x_scaled, x_projected, projection, args.method, args.latent, label_column, label_output, axis_pair
        )
    label_output_3d = os.path.join(
        output_dir, f"latent_{args.method}_{args.latent}_{_safe_filename(label_column)}_{axis_prefix}123.png"
    )
    _plot_label_projection_3d(label_df, x_projected, projection, args.method, args.latent, label_column, label_output_3d)

    color_source_df = label_df if args.method == "tsne" else df
    color_columns = (
        [args.color_column] if args.color_column is not None else _numeric_color_columns(color_source_df, args.latent)
    )
    for color_column in color_columns:
        if color_column not in color_source_df.columns:
            raise RuntimeError(f"CSV must contain color column '{color_column}'.")
        color_df = color_source_df.copy()
        color_df[color_column] = pd.to_numeric(color_df[color_column], errors="coerce")
        valid_mask = color_df[color_column].notna().to_numpy()
        color_df = color_df.loc[valid_mask].reset_index(drop=True)
        if color_df.empty or color_df[color_column].nunique(dropna=True) <= 1:
            print(f"Skipping {color_column}: no varying numeric values.")
            continue
        if args.method == "pca":
            _latent_cols, _x_scaled, color_projection, color_x_projected = _fit_projection(
                color_df,
                args.latent,
                args.method,
                args.tsne_perplexity,
                args.tsne_learning_rate,
                args.tsne_random_state,
                args.latent_start,
                args.latent_end,
            )
        else:
            color_projection = projection
            color_x_projected = x_projected[valid_mask]
        color_axis_pairs = [(0, 1)]
        if color_x_projected.shape[1] >= 3:
            color_axis_pairs += [(0, 2), (1, 2)]
        for axis_pair in color_axis_pairs:
            pair_name = f"{axis_prefix}{axis_pair[0] + 1}_{axis_prefix}{axis_pair[1] + 1}"
            color_output = os.path.join(
                output_dir,
                f"latent_{args.method}_{args.latent}_{_safe_filename(color_column)}_{pair_name}.png",
            )
            _plot_continuous_projection(
                color_df,
                color_x_projected,
                color_projection,
                args.method,
                args.latent,
                color_column,
                color_output,
                axis_pair,
            )
        color_output_3d = os.path.join(
            output_dir, f"latent_{args.method}_{args.latent}_{_safe_filename(color_column)}_{axis_prefix}123.png"
        )
        _plot_continuous_projection_3d(
            color_df, color_x_projected, color_projection, args.method, args.latent, color_column, color_output_3d
        )

    if args.method == "pca":
        print(
            "Explained variance:",
            f"PC1={projection.explained_variance_ratio_[0]:.4f}",
            f"PC2={projection.explained_variance_ratio_[1]:.4f}",
            f"PC3={projection.explained_variance_ratio_[2]:.4f}"
            if len(projection.explained_variance_ratio_) >= 3
            else "PC3=n/a",
        )
    print(f"Silhouette score for {label_column}: {label_silhouette:.4f}")


if __name__ == "__main__":
    main()
