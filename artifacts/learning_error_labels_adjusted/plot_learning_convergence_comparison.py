#!/usr/bin/env python3
"""Plot and quantify three-way learning-error convergence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "vel_tracking_error_mps": (
        "Reward-derived equivalent forward-velocity error",
        "Equivalent error (m/s)",
    ),
    "base_orientation_error_deg": ("Base orientation error", "Error (deg)"),
    "foot_contact_error_n": ("Undesired leg-contact error", "Force error (N)"),
}
BASE_HEIGHT_METRIC = {
    "base_height_error_m": ("Base height error", "Error (m)"),
}
TARGET_UNITS = {
    "vel_tracking_error_mps": "m/s",
    "base_orientation_error_deg": "deg",
    "foot_contact_error_n": "N",
    "base_height_error_m": "m",
}

COLORS = {
    "Hybrid": "#2F6B9A",
    "MLP-only (simple reward)": "#E07A2D",
    "MLP-only (auxiliary reward)": "#3A8F5B",
}
DISPLAY_NAMES = {
    "Hybrid": "LocoDyad",
    "MLP-only (simple reward)": "MLP-E2E",
    "MLP-only (auxiliary reward)": "MLP-E2E (Aux)",
}
PNG_METADATA = {"Software": None}
PDF_METADATA = {
    "Creator": None,
    "Producer": None,
    "CreationDate": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smooth-span", type=int, default=50)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--hold", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--velocity-target",
        type=float,
        default=None,
        help=(
            "Use a shared absolute forward-velocity-error target instead of "
            "each run's relative t90 marker."
        ),
    )
    parser.add_argument(
        "--orientation-target",
        type=float,
        default=None,
        help=(
            "Use a shared absolute base-orientation-error target instead of "
            "each run's relative t90 marker."
        ),
    )
    parser.add_argument(
        "--contact-target",
        type=float,
        default=None,
        help=(
            "Use a shared absolute undesired-contact-error target instead of "
            "each run's relative t90 marker."
        ),
    )
    parser.add_argument(
        "--base-height-target",
        type=float,
        default=None,
        help=(
            "Use a shared absolute base-height-error target instead of each "
            "run's relative t90 marker."
        ),
    )
    parser.add_argument(
        "--range-mode",
        choices=("common", "full"),
        default="common",
        help=(
            "Use the shared iteration range or retain every run through its "
            "own final iteration."
        ),
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=None,
        help=(
            "Optionally truncate every run at this iteration. Use this when the "
            "reported policy is an early-stopped checkpoint."
        ),
    )
    parser.add_argument(
        "--selected-checkpoint-iteration",
        type=int,
        default=None,
        help="Draw a marker at the iteration of the checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--selected-checkpoint-label",
        default="Selected Aux checkpoint",
        help="Legend label for --selected-checkpoint-iteration.",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.08,
        help="Opacity of the unsmoothed scalar traces.",
    )
    parser.add_argument(
        "--paper-four-panel",
        action="store_true",
        help="Use the paper's 2-by-2 layout, including base-height error.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale all text relative to the default figure typography.",
    )
    return parser.parse_args()


def persistent_crossing(
    values: pd.Series,
    threshold: float,
    hold: int,
) -> float:
    array = values.to_numpy(dtype=float)
    iterations = values.index.to_numpy(dtype=int)
    if array.size < hold:
        return float("nan")
    for index in range(array.size - hold + 1):
        window = array[index : index + hold]
        if np.all(np.isfinite(window)) and np.all(window <= threshold):
            return float(iterations[index])
    return float("nan")


def main() -> None:
    args = parse_args()
    if min(args.smooth_span, args.window, args.hold) < 1:
        raise ValueError("smooth-span, window, and hold must be positive")
    if args.font_scale <= 0.0:
        raise ValueError("font-scale must be positive")
    if args.max_iteration is not None and args.max_iteration < 1:
        raise ValueError("max-iteration must be positive")
    if (
        args.selected_checkpoint_iteration is not None
        and args.selected_checkpoint_iteration < 1
    ):
        raise ValueError("selected-checkpoint-iteration must be positive")
    if not 0.0 <= args.raw_alpha <= 1.0:
        raise ValueError("raw-alpha must be in [0, 1]")

    target_thresholds = {
        "vel_tracking_error_mps": args.velocity_target,
        "base_orientation_error_deg": args.orientation_target,
        "foot_contact_error_n": args.contact_target,
        "base_height_error_m": args.base_height_target,
    }
    invalid_targets = {
        metric: target
        for metric, target in target_thresholds.items()
        if target is not None and target <= 0.0
    }
    if invalid_targets:
        raise ValueError(f"Absolute error targets must be positive: {invalid_targets}")

    metrics = (
        {**METRICS, **BASE_HEIGHT_METRIC}
        if args.paper_four_panel
        else METRICS
    )
    display_names = (
        {
            "Hybrid": "Hybrid",
            "MLP-only (simple reward)": "MLP-only (simple reward)",
            "MLP-only (auxiliary reward)": "MLP-only (auxiliary reward)",
        }
        if args.paper_four_panel
        else DISPLAY_NAMES
    )

    frame = pd.read_csv(args.data)
    required = {"run", "iteration", *metrics}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    run_names = list(frame["run"].drop_duplicates())
    expected = list(COLORS)
    if set(run_names) != set(expected):
        raise ValueError(f"Expected runs {expected}, found {run_names}")

    source_per_run = {
        run: frame.loc[frame["run"] == run].set_index("iteration").sort_index()
        for run in expected
    }
    if args.range_mode == "common":
        common_start = max(int(data.index.min()) for data in source_per_run.values())
        common_end = min(int(data.index.max()) for data in source_per_run.values())
        if args.max_iteration is not None:
            common_end = min(common_end, int(args.max_iteration))
        per_run = {
            run: data.loc[common_start:common_end, list(metrics)]
            for run, data in source_per_run.items()
        }
        plot_start = common_start
        plot_end = common_end
    else:
        common_start = None
        common_end = None
        per_run = {}
        for run, data in source_per_run.items():
            selected = data.loc[:, list(metrics)]
            if args.max_iteration is not None:
                selected = selected.loc[: int(args.max_iteration)]
            per_run[run] = selected
        plot_start = min(int(data.index.min()) for data in per_run.values())
        plot_end = max(int(data.index.max()) for data in per_run.values())

    too_short = {
        run: len(data)
        for run, data in per_run.items()
        if len(data) < 2 * args.window
    }
    if too_short:
        raise ValueError(
            "Run ranges are too short for the requested windows: "
            f"{too_short}"
        )
    if (
        args.selected_checkpoint_iteration is not None
        and not plot_start <= args.selected_checkpoint_iteration <= plot_end
    ):
        raise ValueError(
            "selected-checkpoint-iteration is outside the plotted range: "
            f"{args.selected_checkpoint_iteration} not in [{plot_start}, {plot_end}]"
        )
    smoothed = {
        run: data.ewm(span=args.smooth_span, adjust=False).mean()
        for run, data in per_run.items()
    }

    summary_rows: list[dict[str, float | int | str]] = []
    convergence: dict[tuple[str, str], float] = {}
    for metric in metrics:
        for run in expected:
            values = smoothed[run][metric].dropna()
            initial_mean = float(values.iloc[: args.window].mean())
            final_mean = float(values.iloc[-args.window :].mean())
            improvement = initial_mean - final_mean
            if improvement > 0.0:
                threshold_50 = initial_mean - 0.50 * improvement
                threshold_90 = initial_mean - 0.90 * improvement
                iteration_50 = persistent_crossing(values, threshold_50, args.hold)
                iteration_90 = persistent_crossing(values, threshold_90, args.hold)
            else:
                threshold_50 = float("nan")
                threshold_90 = float("nan")
                iteration_50 = float("nan")
                iteration_90 = float("nan")
            absolute_target = target_thresholds[metric]
            iteration_absolute_target = (
                persistent_crossing(values, absolute_target, args.hold)
                if absolute_target is not None
                else float("nan")
            )
            absolute_target_retained_at_end = (
                bool((values.iloc[-args.hold :] <= absolute_target).all())
                if absolute_target is not None
                else None
            )
            convergence[(run, metric)] = (
                iteration_absolute_target
                if absolute_target is not None
                else iteration_90
            )
            summary_rows.append(
                {
                    "run": run,
                    "metric": metric,
                    "common_iteration_start": common_start,
                    "common_iteration_end": common_end,
                    "run_iteration_start": int(values.index.min()),
                    "run_iteration_end": int(values.index.max()),
                    "ema_span": args.smooth_span,
                    "initial_window": args.window,
                    "final_window": args.window,
                    "persistence_hold": args.hold,
                    "initial_mean": initial_mean,
                    "final_mean": final_mean,
                    "absolute_improvement": improvement,
                    "relative_improvement_fraction": (
                        improvement / initial_mean if initial_mean != 0.0 else float("nan")
                    ),
                    "threshold_50pct": threshold_50,
                    "iteration_50pct": iteration_50,
                    "threshold_90pct": threshold_90,
                    "iteration_90pct": iteration_90,
                    "absolute_target": absolute_target,
                    "iteration_absolute_target": iteration_absolute_target,
                    "absolute_target_retained_at_end": (
                        absolute_target_retained_at_end
                    ),
                }
            )

    font_scale = float(args.font_scale)
    plt.rcParams.update(
        {
            "font.size": 10 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 10 * font_scale,
            "xtick.labelsize": 10 * font_scale,
            "ytick.labelsize": 10 * font_scale,
            "legend.fontsize": 9 * font_scale,
            "figure.dpi": 120,
        }
    )
    if args.paper_four_panel:
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True)
        plot_axes = tuple(axes.flat)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12.0, 6.0), sharex=True)
        plot_axes = tuple(axes)
    for panel, (metric, (title, ylabel)) in zip(plot_axes, metrics.items()):
        absolute_target = target_thresholds[metric]
        unreached_runs: list[str] = []
        unretained_runs: list[str] = []
        for run_index, run in enumerate(expected):
            raw = per_run[run][metric]
            smooth = smoothed[run][metric]
            color = COLORS[run]
            panel.plot(
                raw.index,
                raw,
                color=color,
                alpha=args.raw_alpha,
                linewidth=0.80,
            )
            panel.plot(
                smooth.index,
                smooth,
                color=color,
                linewidth=3.2 if args.paper_four_panel else 2.9,
                label=f"{display_names[run]} (EMA {args.smooth_span})",
            )
            if args.range_mode == "full":
                final_iteration = int(smooth.index[-1])
                final_value = float(smooth.iloc[-1])
                panel.scatter(
                    [final_iteration],
                    [final_value],
                    s=42,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )
                if panel is plot_axes[0]:
                    is_right_edge = final_iteration == plot_end
                    panel.annotate(
                        f"end {final_iteration}",
                        xy=(final_iteration, final_value),
                        xytext=((-7, 8) if is_right_edge else (7, 8)),
                        textcoords="offset points",
                        ha=("right" if is_right_edge else "left"),
                        va="bottom",
                        color=color,
                        fontsize=8.2 * font_scale,
                        fontweight="semibold",
                    )
            convergence_iteration = convergence[(run, metric)]
            if np.isfinite(convergence_iteration):
                panel.axvline(
                    convergence_iteration,
                    color=color,
                    linestyle="--",
                    linewidth=2.6,
                    alpha=0.95,
                )
                x_offset = (5, -5, 5)[run_index]
                y_position = (0.97, 0.97, 0.84)[run_index]
                horizontal_alignment = ("left", "right", "left")[run_index]
                # Place nearby orientation labels outside their marker pair.
                if metric == "base_orientation_error_deg":
                    if run == "Hybrid":
                        x_offset, horizontal_alignment = 5, "left"
                    elif run == "MLP-only (simple reward)":
                        x_offset, horizontal_alignment = 5, "left"
                    else:
                        y_position = 0.78
                # Keep the late contact marker label below the retention note.
                if metric == "foot_contact_error_n" and run == "MLP-only (simple reward)":
                    y_position = 0.73
                panel.annotate(
                    f"{int(convergence_iteration)}",
                    xy=(convergence_iteration, y_position),
                    xycoords=(panel.transData, panel.transAxes),
                    xytext=(x_offset, 0),
                    textcoords="offset points",
                    ha=horizontal_alignment,
                    va="top",
                    color=color,
                    fontsize=8.4 * font_scale,
                    fontweight="semibold",
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "alpha": 0.82,
                        "edgecolor": "none",
                    },
                )
                if absolute_target is not None:
                    target_retained = bool(
                        (smooth.iloc[-args.hold :] <= absolute_target).all()
                    )
                    if not target_retained:
                        unretained_runs.append(run)
            elif absolute_target is not None:
                unreached_runs.append(run)
        if absolute_target is not None:
            panel.axhline(
                absolute_target,
                color="#555555",
                linestyle=(0, (4, 3)),
                linewidth=2.1,
                alpha=0.85,
                zorder=1,
                label=("Shared absolute-error target" if panel is plot_axes[0] else None),
            )
            panel.annotate(
                f"target ≤ {absolute_target:g} {TARGET_UNITS[metric]}",
                xy=(0.99, absolute_target),
                xycoords=(panel.transAxes, panel.transData),
                xytext=(-3, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                color="#444444",
                fontsize=8.0 * font_scale,
                fontweight="semibold",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.82,
                    "edgecolor": "none",
                },
            )
            for missing_index, run in enumerate(unreached_runs):
                panel.text(
                    0.985,
                    0.96 - 0.09 * missing_index,
                    f"{display_names[run]}: not reached",
                    transform=panel.transAxes,
                    ha="right",
                    va="top",
                    color=COLORS[run],
                    fontsize=8.0 * font_scale,
                    fontweight="semibold",
                    bbox={
                        "boxstyle": "round,pad=0.14",
                        "facecolor": "white",
                        "alpha": 0.85,
                        "edgecolor": "none",
                    },
                )
            for unretained_index, run in enumerate(unretained_runs):
                panel.text(
                    0.985,
                    0.96 - 0.09 * (len(unreached_runs) + unretained_index),
                    f"{display_names[run]}: not retained at end",
                    transform=panel.transAxes,
                    ha="right",
                    va="top",
                    color=COLORS[run],
                    fontsize=8.0 * font_scale,
                    fontweight="semibold",
                    bbox={
                        "boxstyle": "round,pad=0.14",
                        "facecolor": "white",
                        "alpha": 0.85,
                        "edgecolor": "none",
                    },
                )
        if args.selected_checkpoint_iteration is not None:
            panel.axvline(
                args.selected_checkpoint_iteration,
                color="#6B4C9A",
                linestyle=":",
                linewidth=2.8,
                alpha=0.95,
                label=(
                    f"{args.selected_checkpoint_label} "
                    f"(ep{args.selected_checkpoint_iteration})"
                    if panel is plot_axes[0]
                    else None
                ),
            )
        panel.set_title(
            title,
            fontsize=15 * font_scale,
            fontweight="bold",
            pad=8,
        )
        panel.set_ylabel(ylabel.replace("Equivalent error (m/s)", "Equivalent error\n(m/s)"))
        panel.set_xlim(plot_start, plot_end)
        panel.set_ylim(bottom=0.0)
        panel.grid(alpha=0.28, linewidth=0.85)
        panel.tick_params(width=1.1, length=4.5)
        panel.spines["left"].set_linewidth(1.15)
        panel.spines["bottom"].set_linewidth(1.15)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

    if args.paper_four_panel:
        axes[1, 0].set_xlabel("Training iteration")
        axes[1, 1].set_xlabel("Training iteration")
    else:
        axes[-1].set_xlabel("Training iteration")
    handles, labels = plot_axes[0].get_legend_handles_labels()
    if args.paper_four_panel:
        fig.suptitle(
            "Learning-error convergence: Hybrid and MLP-only controls",
            fontsize=17 * font_scale,
            y=0.995,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(expected),
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            fontsize=10.5,
            handlelength=2.0,
            columnspacing=1.2,
            handletextpad=0.6,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "learning_error_convergence_comparison_600dpi.png"
    pdf_path = out_dir / "learning_error_convergence_comparison.pdf"
    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
        metadata=PNG_METADATA,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "convergence_summary.csv", index=False, float_format="%.9g")
    metadata = {
        "source_data": os.path.relpath(
            args.data.expanduser().resolve(),
            out_dir,
        ),
        "runs": expected,
        "display_names": display_names,
        "range_mode": args.range_mode,
        "full_range_endpoint_markers": args.range_mode == "full",
        "common_iteration_range": (
            [common_start, common_end] if args.range_mode == "common" else None
        ),
        "plot_iteration_range": [plot_start, plot_end],
        "plotted_iteration_ranges": {
            run: [int(data.index.min()), int(data.index.max())]
            for run, data in per_run.items()
        },
        "requested_max_iteration": args.max_iteration,
        "selected_checkpoint_iteration": args.selected_checkpoint_iteration,
        "selected_checkpoint_label": (
            args.selected_checkpoint_label
            if args.selected_checkpoint_iteration is not None
            else None
        ),
        "ema_span": args.smooth_span,
        "raw_trace_alpha": args.raw_alpha,
        "initial_and_final_window": args.window,
        "persistence_hold": args.hold,
        "convergence_marker_mode": {
            metric: (
                "absolute_target"
                if target_thresholds[metric] is not None
                else "relative_t90"
            )
            for metric in metrics
        },
        "absolute_targets": {
            metric: target_thresholds[metric]
            for metric in metrics
            if target_thresholds[metric] is not None
        },
        "paper_four_panel": bool(args.paper_four_panel),
        "font_scale": font_scale,
        "t50_definition": "First EMA iteration at or below 50% of observed initial-to-final improvement for the full persistence hold.",
        "t90_definition": "First EMA iteration at or below 90% of observed initial-to-final improvement for the full persistence hold.",
        "absolute_target_definition": "First EMA iteration at or below the shared absolute target for the full persistence hold.",
        "outputs": [png_path.name, pdf_path.name, "convergence_summary.csv"],
    }
    (out_dir / "convergence_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"[INFO] Wrote convergence analysis to {out_dir}")


if __name__ == "__main__":
    main()
