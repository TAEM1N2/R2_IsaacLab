#!/usr/bin/env python3
"""Create a paper-style four-preset contact-map figure with gait snapshots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


FOOT_ORDER = ("FL", "FR", "RL", "RR")
FOOT_COLORS = {
    "FL": "#0072B2",
    "FR": "#E69F00",
    "RL": "#009E73",
    "RR": "#D55E00",
}
GAIT_PROFILES = {
    0: ("Slow trot", 1.20, 0.50, 0.66, 0.07),
    1: ("Nominal trot", 1.45, 0.50, 0.60, 0.08),
    2: ("Fast trot", 1.75, 0.50, 0.54, 0.08),
    3: ("High-clearance trot", 1.45, 0.50, 0.60, 0.13),
}
SNAPSHOT_TIMES = (0.5, 1.5, 2.5)
SNAPSHOT_CROP = (430, 700, 780, 1140)  # y0, y1, x0, x1


def load_rows(path: Path) -> list[dict[str, object]]:
    """Load fields needed for plotting, summary generation, and validation."""
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "elapsed_s",
            "sample",
            "foot",
            "contact",
            "touchdown",
            "gait_mode",
            "command_x_mps",
            "command_y_mps",
            "command_yaw_radps",
            "stair_collision_enabled",
            "slope_collision_enabled",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
        for source in reader:
            rows.append(
                {
                    "elapsed_s": float(source["elapsed_s"]),
                    "sample": int(source["sample"]),
                    "foot": source["foot"],
                    "contact": bool(int(source["contact"])),
                    "touchdown": bool(int(source["touchdown"])),
                    "gait_mode": int(source["gait_mode"]),
                    "command_x_mps": float(source["command_x_mps"]),
                    "command_y_mps": float(source["command_y_mps"]),
                    "command_yaw_radps": float(source["command_yaw_radps"]),
                    "stair_collision_enabled": bool(int(source["stair_collision_enabled"])),
                    "slope_collision_enabled": bool(int(source["slope_collision_enabled"])),
                }
            )
    return rows


def validate_rows(rows: list[dict[str, object]]) -> None:
    """Verify the requested flat, straight, 0.5 m/s four-gait experiment."""
    modes = sorted({int(row["gait_mode"]) for row in rows})
    if modes != [0, 1, 2, 3]:
        raise ValueError(f"Expected gait modes [0, 1, 2, 3], got {modes}")
    if any(bool(row["stair_collision_enabled"]) for row in rows):
        raise ValueError("Stair collision was enabled in the recorded data.")
    if any(bool(row["slope_collision_enabled"]) for row in rows):
        raise ValueError("Slope collision was enabled in the recorded data.")
    for row in rows:
        command = (
            float(row["command_x_mps"]),
            float(row["command_y_mps"]),
            float(row["command_yaw_radps"]),
        )
        if not np.allclose(command, (0.5, 0.0, 0.0), atol=1e-9):
            raise ValueError(f"Unexpected velocity command: {command}")


def contact_intervals(
    times: list[float],
    contacts: list[bool],
    dt: float,
) -> list[tuple[float, float]]:
    """Convert a sampled binary contact signal to start-duration intervals."""
    intervals: list[tuple[float, float]] = []
    start = None
    for time_s, contact in zip(times, contacts):
        if contact and start is None:
            start = time_s
        elif not contact and start is not None:
            intervals.append((start, max(dt, time_s - start)))
            start = None
    if start is not None and times:
        intervals.append((start, max(dt, times[-1] + dt - start)))
    return intervals


def snapshot_path(snapshot_dir: Path, mode: int, time_s: float) -> Path:
    time_slug = f"{time_s:.1f}".replace(".", "p")
    return snapshot_dir / f"mode{mode}_t{time_slug}s.png"


def load_cropped_snapshot(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot: {path}")
    image = mpimg.imread(path)
    y0, y1, x0, x1 = SNAPSHOT_CROP
    if image.shape[0] < y1 or image.shape[1] < x1:
        raise ValueError(f"Snapshot is too small for the configured crop: {path} {image.shape}")
    return image[y0:y1, x0:x1]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Write measured segment duration, contact ratio, and touchdown count."""
    summary: list[dict[str, object]] = []
    for mode in sorted(GAIT_PROFILES):
        mode_rows = [row for row in rows if int(row["gait_mode"]) == mode]
        mode_start = min(float(row["elapsed_s"]) for row in mode_rows)
        mode_end = max(float(row["elapsed_s"]) for row in mode_rows)
        for foot in FOOT_ORDER:
            selected = [row for row in mode_rows if row["foot"] == foot]
            summary.append(
                {
                    "gait_mode": mode,
                    "gait_name": GAIT_PROFILES[mode][0],
                    "foot": foot,
                    "segment_start_s": mode_start,
                    "segment_end_s": mode_end,
                    "segment_duration_s": mode_end - mode_start,
                    "sample_count": len(selected),
                    "contact_ratio": sum(bool(row["contact"]) for row in selected) / len(selected),
                    "touchdown_count": sum(bool(row["touchdown"]) for row in selected),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def draw_contact_axis(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    mode: int,
) -> None:
    mode_rows = [row for row in rows if int(row["gait_mode"]) == mode]
    mode_start = min(float(row["elapsed_s"]) for row in mode_rows)
    mode_end = max(float(row["elapsed_s"]) for row in mode_rows)
    duration = mode_end - mode_start

    for foot_index, foot in enumerate(FOOT_ORDER):
        axis.axhspan(
            foot_index - 0.34,
            foot_index + 0.34,
            color="#F0F0F0",
            zorder=0,
        )
        selected = sorted(
            (row for row in mode_rows if row["foot"] == foot),
            key=lambda row: float(row["elapsed_s"]),
        )
        times = [float(row["elapsed_s"]) - mode_start for row in selected]
        contacts = [bool(row["contact"]) for row in selected]
        intervals = contact_intervals(times, contacts, dt=0.002)
        axis.broken_barh(
            intervals,
            (foot_index - 0.31, 0.62),
            facecolors=FOOT_COLORS[foot],
            edgecolors="none",
            zorder=2,
        )

    name, frequency, offset, duty, height = GAIT_PROFILES[mode]
    axis.set_title(
        rf"$f={frequency:.2f}$ Hz  |  offset={offset:.2f}  |  "
        rf"duty={duty:.2f}  |  $h_{{swing}}={height:.2f}$ m",
        fontsize=8.5,
        pad=4,
        color="#333333",
    )
    axis.set_yticks(range(len(FOOT_ORDER)), FOOT_ORDER)
    for label, foot in zip(axis.get_yticklabels(), FOOT_ORDER):
        label.set_color(FOOT_COLORS[foot])
        label.set_fontweight("bold")
    axis.set_ylim(-0.55, len(FOOT_ORDER) - 0.45)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 3.05)
    axis.set_xticks((0.0, 1.0, 2.0, 3.0))
    axis.set_xlabel("Time within gait segment [s]", labelpad=2)
    axis.set_ylabel("Foot", labelpad=2)
    axis.grid(axis="x", color="#B8B8B8", linewidth=0.6, alpha=0.65, zorder=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.995,
        0.04,
        f"recorded {duration:.3f} s",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )


def plot_contact_map(
    png_path: Path,
    pdf_path: Path,
    rows: list[dict[str, object]],
    snapshot_dir: Path,
    figure_title: str,
) -> None:
    """Plot four paper-style gait panels, each with snapshots over contact bars."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(14.2, 10.2), facecolor="white")
    outer = figure.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.91,
        wspace=0.14,
        hspace=0.24,
    )
    panel_labels = ("a", "b", "c", "d")

    for mode, panel_label in zip(sorted(GAIT_PROFILES), panel_labels):
        row_index, column_index = divmod(mode, 2)
        block = outer[row_index, column_index].subgridspec(
            2,
            3,
            height_ratios=(1.35, 1.0),
            hspace=0.15,
            wspace=0.025,
        )
        snapshot_axes = [figure.add_subplot(block[0, index]) for index in range(3)]
        contact_axis = figure.add_subplot(block[1, :])
        name = GAIT_PROFILES[mode][0]

        for snapshot_axis, time_s in zip(snapshot_axes, SNAPSHOT_TIMES):
            image = load_cropped_snapshot(snapshot_path(snapshot_dir, mode, time_s))
            snapshot_axis.imshow(image)
            snapshot_axis.set_xticks([])
            snapshot_axis.set_yticks([])
            snapshot_axis.set_title(f"t = {time_s:.1f} s", fontsize=8.5, pad=2)
            for spine in snapshot_axis.spines.values():
                spine.set_color("#4A4A4A")
                spine.set_linewidth(0.65)

        snapshot_axes[0].text(
            -0.02,
            1.20,
            f"({panel_label})  Mode {mode}: {name}",
            transform=snapshot_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#111111",
            clip_on=False,
        )
        draw_contact_axis(contact_axis, rows, mode)

    figure.suptitle(
        figure_title,
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.943,
        "Preset applied every 3 s; snapshots every 1 s starting at 0.5 s "
        "(0.5, 1.5, 2.5 s per segment)",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.legend(
        handles=[Patch(facecolor=FOOT_COLORS[foot], label=foot) for foot in FOOT_ORDER],
        title="Stance contact",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=4,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.8,
    )
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Tip-toe policy gait presets at 0.5 m/s on flat terrain",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    validate_rows(rows)
    plot_contact_map(args.output, args.pdf, rows, args.snapshot_dir, args.title)
    write_summary(args.summary, rows)
    modes = sorted({int(row["gait_mode"]) for row in rows})
    print(f"validated_rows={len(rows)} modes={modes} flat_terrain=true command=(0.5,0,0)")
    print(f"saved_png={args.output}")
    print(f"saved_pdf={args.pdf}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
