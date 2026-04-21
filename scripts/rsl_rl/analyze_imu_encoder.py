#!/usr/bin/env python3
"""Analyze policy-stream CSVs and create IMU estimator comparison plots."""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to generate plots. Install it in the current Python environment and rerun."
    ) from exc


def load_csv_columns(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
    for row in rows:
        for key, value in row.items():
            columns[key].append(float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def extract_series(columns: dict[str, np.ndarray], prefix: str, count: int) -> np.ndarray | None:
    names = [f"{prefix}_{idx}" for idx in range(count)]
    if not all(name in columns for name in names):
        return None
    return np.stack([columns[name] for name in names], axis=1)


def infer_from_imu_csv(imu_csv_path: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    columns = load_csv_columns(imu_csv_path)
    steps = columns["step"]
    imu_est = extract_series(columns, "imu_est", 9)
    imu_gt = extract_series(columns, "imu_gt", 9)
    if imu_est is None or imu_gt is None:
        raise ValueError(f"imu_encoder.csv does not contain expected imu_est_/imu_gt_ columns: {imu_csv_path}")

    groups = {
        "lin_vel": (imu_est[:, 0:3], imu_gt[:, 0:3]),
        "ang_vel": (imu_est[:, 3:6], imu_gt[:, 3:6]),
        "projected_gravity": (imu_est[:, 6:9], imu_gt[:, 6:9]),
    }
    norms = {
        "lin_vel": columns.get("lin_vel_err_l2", np.linalg.norm(imu_est[:, 0:3] - imu_gt[:, 0:3], axis=1)),
        "ang_vel": columns.get("ang_vel_err_l2", np.linalg.norm(imu_est[:, 3:6] - imu_gt[:, 3:6], axis=1)),
        "projected_gravity": columns.get(
            "proj_gravity_err_l2", np.linalg.norm(imu_est[:, 6:9] - imu_gt[:, 6:9], axis=1)
        ),
        "imu_total": columns.get("imu_err_l2", np.linalg.norm(imu_est - imu_gt, axis=1)),
    }
    return steps, groups, norms


def infer_from_obs_csv(obs_csv_path: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    columns = load_csv_columns(obs_csv_path)
    steps = columns["step"]
    encoder = np.stack([columns[f"encoder_output_{idx}"] for idx in range(9)], axis=1)
    obs = np.stack([columns[f"obs_{idx}"] for idx in range(6)], axis=1)

    groups = {
        "ang_vel": (encoder[:, 3:6], obs[:, 0:3]),
        "projected_gravity": (encoder[:, 6:9], obs[:, 3:6]),
    }
    norms = {
        "ang_vel": np.linalg.norm(encoder[:, 3:6] - obs[:, 0:3], axis=1),
        "projected_gravity": np.linalg.norm(encoder[:, 6:9] - obs[:, 3:6], axis=1),
    }
    return steps, groups, norms


def plot_groups(
    steps: np.ndarray,
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    num_groups = len(groups)
    fig, axes = plt.subplots(num_groups, 3, figsize=(18, 4 * num_groups), sharex=True)
    if num_groups == 1:
        axes = np.asarray([axes])

    component_labels = ["x", "y", "z"]
    y_units = {
        "lin_vel": "[m/s]",
        "ang_vel": "[rad/s]",
        "projected_gravity": "",
    }
    for row_idx, (group_name, (est, gt)) in enumerate(groups.items()):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.plot(steps, gt[:, col_idx], label="gt", linewidth=2)
            ax.plot(steps, est[:, col_idx], label="est", linewidth=1.5, alpha=0.85)
            ax.set_title(f"{group_name}.{component_labels[col_idx]}")
            ax.set_ylabel(y_units.get(group_name, ""))
            if group_name == "projected_gravity" and component_labels[col_idx] == "z":
                ax.set_ylim(-1.2, -0.8)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend()
    axes[-1, 0].set_xlabel("step [count]")
    axes[-1, 1].set_xlabel("step [count]")
    axes[-1, 2].set_xlabel("step [count]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_norms(steps: np.ndarray, norms: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, values in norms.items():
        ax.plot(steps, values, label=name, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("step [count]")
    ax.set_ylabel("l2 error [mixed units]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IMU estimator comparison plots from policy-stream CSV files.")
    parser.add_argument("obs_csv", type=Path, help="Path to obs_IsaacLab.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to a sibling folder next to the input CSV.",
    )
    args = parser.parse_args()

    obs_csv_path = args.obs_csv.expanduser().resolve()
    if not obs_csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {obs_csv_path}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else obs_csv_path.parent / "imu_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_csv_path = obs_csv_path.parent / "imu_encoder.csv"
    if imu_csv_path.is_file():
        steps, groups, norms = infer_from_imu_csv(imu_csv_path)
        source = imu_csv_path
        title_prefix = "IMU Encoder vs GT"
    else:
        steps, groups, norms = infer_from_obs_csv(obs_csv_path)
        source = obs_csv_path
        title_prefix = "IMU Encoder vs GT (partial from obs_IsaacLab.csv)"

    compare_path = output_dir / "imu_compare.png"
    error_path = output_dir / "imu_error_norms.png"

    plot_groups(steps, groups, compare_path, f"{title_prefix}\nsource: {source.name}")
    plot_norms(steps, norms, error_path, f"{title_prefix} error norms")

    print(f"[INFO] Input obs CSV: {obs_csv_path}")
    print(f"[INFO] Analysis source: {source}")
    print(f"[INFO] Saved plot: {compare_path}")
    print(f"[INFO] Saved plot: {error_path}")
    print("[INFO] Detected groups:")
    for name, (est, gt) in groups.items():
        mae = np.mean(np.abs(est - gt), axis=0)
        print(f"  - {name}: est shape={est.shape}, gt shape={gt.shape}, mean_abs_err={mae.round(6).tolist()}")


if __name__ == "__main__":
    main()
