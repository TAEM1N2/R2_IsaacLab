#!/usr/bin/env python3
"""Plot the baseline policy at its one fixed training gait."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


FOOT_ORDER = ("FL", "FR", "RL", "RR")
FOOT_COLORS = {
    "FL": "#0072B2",
    "FR": "#E69F00",
    "RL": "#009E73",
    "RR": "#D55E00",
}
GAIT = {
    "frequency_hz": 2.00,
    "offset": 0.50,
    "duty": 0.50,
    "swing_height_m": 0.05,
}
SNAPSHOT_TIMES = (0.5, 1.5, 2.5)
SNAPSHOT_CROP = (430, 700, 780, 1140)  # y0, y1, x0, x1


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "elapsed_s",
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
    modes = sorted({int(row["gait_mode"]) for row in rows})
    if modes != [0]:
        raise ValueError(f"Expected only fixed gait mode 0, got {modes}")
    if any(bool(row["stair_collision_enabled"]) for row in rows):
        raise ValueError("Stair collision was enabled.")
    if any(bool(row["slope_collision_enabled"]) for row in rows):
        raise ValueError("Slope collision was enabled.")
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
    dt: float = 0.002,
) -> list[tuple[float, float]]:
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


def snapshot_path(snapshot_dir: Path, time_s: float) -> Path:
    slug = f"{time_s:.1f}".replace(".", "p")
    return snapshot_dir / f"t{slug}s.png"


def load_snapshot(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    image = mpimg.imread(path)
    y0, y1, x0, x1 = SNAPSHOT_CROP
    return image[y0:y1, x0:x1]


def draw_contact_map(axis: plt.Axes, rows: list[dict[str, object]]) -> None:
    start_s = min(float(row["elapsed_s"]) for row in rows)
    end_s = max(float(row["elapsed_s"]) for row in rows)
    for foot_index, foot in enumerate(FOOT_ORDER):
        axis.axhspan(
            foot_index - 0.34,
            foot_index + 0.34,
            color="#F0F0F0",
            zorder=0,
        )
        selected = sorted(
            (row for row in rows if row["foot"] == foot),
            key=lambda row: float(row["elapsed_s"]),
        )
        times = [float(row["elapsed_s"]) - start_s for row in selected]
        contacts = [bool(row["contact"]) for row in selected]
        axis.broken_barh(
            contact_intervals(times, contacts),
            (foot_index - 0.31, 0.62),
            facecolors=FOOT_COLORS[foot],
            edgecolors="none",
            zorder=2,
        )

    axis.set_yticks(range(len(FOOT_ORDER)), FOOT_ORDER)
    for label, foot in zip(axis.get_yticklabels(), FOOT_ORDER):
        label.set_color(FOOT_COLORS[foot])
        label.set_fontweight("bold")
    axis.set_ylim(-0.55, len(FOOT_ORDER) - 0.45)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 3.05)
    axis.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Foot")
    axis.grid(axis="x", color="#B8B8B8", linewidth=0.6, alpha=0.65, zorder=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.995,
        0.04,
        f"recorded {end_s - start_s:.3f} s",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    start_s = min(float(row["elapsed_s"]) for row in rows)
    end_s = max(float(row["elapsed_s"]) for row in rows)
    summary: list[dict[str, object]] = []
    for foot in FOOT_ORDER:
        selected = [row for row in rows if row["foot"] == foot]
        summary.append(
            {
                "policy": "baseline",
                "frequency_hz": GAIT["frequency_hz"],
                "offset": GAIT["offset"],
                "duty": GAIT["duty"],
                "swing_height_m": GAIT["swing_height_m"],
                "foot": foot,
                "duration_s": end_s - start_s,
                "sample_count": len(selected),
                "contact_ratio": sum(bool(row["contact"]) for row in selected) / len(selected),
                "touchdown_count": sum(bool(row["touchdown"]) for row in selected),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def plot(
    png_path: Path,
    pdf_path: Path,
    rows: list[dict[str, object]],
    snapshot_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.0, 6.0), facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.15,
        top=0.78,
        height_ratios=(1.25, 1.0),
        hspace=0.22,
        wspace=0.035,
    )
    for index, time_s in enumerate(SNAPSHOT_TIMES):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(load_snapshot(snapshot_path(snapshot_dir, time_s)))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"t = {time_s:.1f} s", fontsize=10, pad=3)
        for spine in axis.spines.values():
            spine.set_color("#4A4A4A")
            spine.set_linewidth(0.7)

    contact_axis = figure.add_subplot(grid[1, :])
    draw_contact_map(contact_axis, rows)

    figure.suptitle(
        "Baseline policy at its trained fixed gait — 0.5 m/s, flat terrain",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.5,
        0.885,
        r"Training gait: $f=2.00$ Hz  |  offset=0.50  |  duty=0.50  |  "
        r"$h_{swing}=0.05$ m",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )
    figure.text(
        0.5,
        0.835,
        "Snapshots every 1 s starting at 0.5 s; raw stance contact threshold = 1 N",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
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
    args = parser.parse_args()

    rows = load_rows(args.input)
    validate_rows(rows)
    plot(args.output, args.pdf, rows, args.snapshot_dir)
    write_summary(args.summary, rows)
    print(
        f"validated_rows={len(rows)} gait=[2.0,0.5,0.5,0.05] "
        "flat_terrain=true command=(0.5,0,0)"
    )
    print(f"saved_png={args.output}")
    print(f"saved_pdf={args.pdf}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
