"""Plot and probe Phase2 LoadAdaptive z_load CSV files recorded by play.py."""

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _sample_per_label(df: pd.DataFrame, label_column: str, max_samples_per_label: int) -> pd.DataFrame:
    if max_samples_per_label <= 0:
        return df
    samples = []
    for _label, group in df.groupby(label_column, group_keys=False):
        samples.append(group.sample(min(len(group), max_samples_per_label), random_state=0))
    return pd.concat(samples).reset_index(drop=True)


def _z_load_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("z_load_")]
    columns = [column for column in columns if column != "z_load_norm"]
    columns.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if not columns:
        raise RuntimeError("No z_load columns found. Expected columns named z_load_0, z_load_1, ...")
    return columns


def _fit_projection(
    df: pd.DataFrame,
    z_cols: list[str],
    method: str,
    tsne_perplexity: float,
    tsne_learning_rate: str | float,
    tsne_random_state: int,
):
    x = df[z_cols].to_numpy()
    x_scaled = StandardScaler().fit_transform(x)
    n_components = min(3, len(z_cols))
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
    return x_scaled, projection, x_projected


def _axis_label(projection, method: str, axis_idx: int) -> str:
    if method == "pca":
        return f"PC{axis_idx + 1} ({projection.explained_variance_ratio_[axis_idx] * 100:.1f}%)"
    return f"t-SNE {axis_idx + 1}"


def _plot_label_projection(
    df: pd.DataFrame,
    x_scaled,
    x_projected,
    projection,
    method: str,
    label_column: str,
    output_path: str,
    axis_pair: tuple[int, int],
) -> None:
    labels = df[label_column].astype(str).to_numpy()
    unique_labels = sorted(df[label_column].astype(str).unique())
    silhouette = silhouette_score(x_scaled, labels) if len(unique_labels) > 1 else float("nan")

    plt.figure(figsize=(9, 7))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(x_projected[mask, axis_pair[0]], x_projected[mask, axis_pair[1]], s=5, alpha=0.35, label=label)
    plt.legend(markerscale=3)
    plt.xlabel(_axis_label(projection, method, axis_pair[0]))
    plt.ylabel(_axis_label(projection, method, axis_pair[1]))
    plt.title(f"z_load {method.upper()} by {label_column}, silhouette={silhouette:.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved label plot: {output_path}")


def _plot_continuous_projection(
    df: pd.DataFrame,
    x_projected,
    projection,
    method: str,
    color_column: str,
    output_path: str,
    axis_pair: tuple[int, int],
) -> None:
    colors = pd.to_numeric(df[color_column], errors="coerce")
    valid = colors.notna().to_numpy()
    if valid.sum() <= 1 or colors[valid].nunique() <= 1:
        print(f"Skipping {color_column}: no varying numeric values.")
        return

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        x_projected[valid, axis_pair[0]],
        x_projected[valid, axis_pair[1]],
        c=colors[valid].to_numpy(),
        s=5,
        alpha=0.55,
        cmap="viridis",
    )
    plt.colorbar(scatter, label=color_column)
    plt.xlabel(_axis_label(projection, method, axis_pair[0]))
    plt.ylabel(_axis_label(projection, method, axis_pair[1]))
    plt.title(f"z_load {method.upper()} colored by {color_column}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved color plot: {output_path}")


def _default_color_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "payload_mass",
        "z_load_norm",
        "delta_action_norm",
        "nominal_action_norm",
        "final_action_norm",
        "foot_force_sum",
        "foot_force_max",
        "foot_contact_count",
        "base_height",
        "base_speed_xy",
        "command_0",
        "command_1",
        "command_2",
    ]
    return [column for column in candidates if column in df.columns]


def _run_linear_probes(df: pd.DataFrame, z_cols: list[str], output_dir: str) -> None:
    targets = [
        "payload_mass",
        "delta_action_norm",
        "nominal_action_norm",
        "final_action_norm",
        "foot_force_sum",
        "foot_force_max",
        "base_height",
        "base_speed_xy",
    ]
    x = df[z_cols].to_numpy()
    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.notna().to_numpy()
        if valid.sum() < 20 or y[valid].nunique() <= 1:
            continue
        x_valid = x[valid]
        y_valid = y[valid].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(x_valid, y_valid, test_size=0.25, random_state=0)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append({"target": target, "r2": r2_score(y_test, pred), "n": int(valid.sum())})

    if not rows:
        print("No linear probe targets available.")
        return

    probe_df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    output_path = os.path.join(output_dir, "z_load_linear_probe_r2.csv")
    probe_df.to_csv(output_path, index=False)
    print(f"Saved linear probe results: {output_path}")
    print(probe_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and probe Phase2 LoadAdaptive z_load CSV files.")
    parser.add_argument("--csv", required=True, help="Path to load_latent.csv generated by play.py.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults next to the CSV file.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Projection method.")
    parser.add_argument("--label_column", default="payload_mass_bin", help="Categorical label column for plots.")
    parser.add_argument(
        "--color_column",
        default=None,
        help="Numeric column for a single continuous plot. If omitted, default diagnostics are plotted.",
    )
    parser.add_argument("--max_samples_per_label", type=int, default=5000, help="Subsample each label for projection.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_learning_rate", default="auto")
    parser.add_argument("--tsne_random_state", type=int, default=0)
    parser.add_argument("--skip_probes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tsne_learning_rate != "auto":
        args.tsne_learning_rate = float(args.tsne_learning_rate)

    df = pd.read_csv(args.csv)
    z_cols = _z_load_columns(df)
    output_dir = args.output if args.output is not None else os.path.dirname(args.csv)
    os.makedirs(output_dir, exist_ok=True)

    if args.label_column not in df.columns:
        raise RuntimeError(f"CSV must contain label column '{args.label_column}'.")

    plot_df = _sample_per_label(df, args.label_column, args.max_samples_per_label)
    _x_scaled, projection, x_projected = _fit_projection(
        plot_df, z_cols, args.method, args.tsne_perplexity, args.tsne_learning_rate, args.tsne_random_state
    )

    axis_pairs = [(0, 1)]
    if x_projected.shape[1] >= 3:
        axis_pairs += [(0, 2), (1, 2)]
    axis_prefix = "pc" if args.method == "pca" else "tsne"

    for axis_pair in axis_pairs:
        pair_name = f"{axis_prefix}{axis_pair[0] + 1}_{axis_prefix}{axis_pair[1] + 1}"
        output_path = os.path.join(
            output_dir, f"z_load_{args.method}_{_safe_filename(args.label_column)}_{pair_name}.png"
        )
        _plot_label_projection(plot_df, _x_scaled, x_projected, projection, args.method, args.label_column, output_path, axis_pair)

    color_columns = [args.color_column] if args.color_column is not None else _default_color_columns(plot_df)
    for color_column in color_columns:
        if color_column not in plot_df.columns:
            raise RuntimeError(f"CSV must contain color column '{color_column}'.")
        for axis_pair in axis_pairs:
            pair_name = f"{axis_prefix}{axis_pair[0] + 1}_{axis_prefix}{axis_pair[1] + 1}"
            output_path = os.path.join(output_dir, f"z_load_{args.method}_{_safe_filename(color_column)}_{pair_name}.png")
            _plot_continuous_projection(plot_df, x_projected, projection, args.method, color_column, output_path, axis_pair)

    if not args.skip_probes:
        _run_linear_probes(df, z_cols, output_dir)


if __name__ == "__main__":
    main()
