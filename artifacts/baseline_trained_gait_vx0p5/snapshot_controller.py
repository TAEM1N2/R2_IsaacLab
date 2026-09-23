#!/usr/bin/env python3
"""Capture three MuJoCo snapshots from the 100 Hz policy observation clock."""

from __future__ import annotations

import argparse
import csv
import queue
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class SnapshotController(Node):
    """Capture at 0.5, 1.5, and 2.5 s during one 3 s gait segment."""

    def __init__(self, snapshot_dir: Path) -> None:
        super().__init__("baseline_trained_gait_snapshot_controller")
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/obs_MuJoCo",
            self._observation_callback,
            10,
        )
        self.observation_count = 0
        self.done = False
        self.schedule = {50: 0.5, 150: 1.5, 250: 2.5}
        self.jobs: queue.Queue[tuple[int, float] | None] = queue.Queue()
        self.records: list[dict[str, object]] = []
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.get_logger().info("Snapshots scheduled at gait t=0.5/1.5/2.5 s.")

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                observation_count, time_s = job
                slug = f"{time_s:.1f}".replace(".", "p")
                output_path = self.snapshot_dir / f"t{slug}s.xwd"
                result = subprocess.run(
                    [
                        "xwd",
                        "-silent",
                        "-name",
                        "RoK-4 MuJoCo Simulation",
                        "-out",
                        str(output_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                status = "ok" if result.returncode == 0 and output_path.exists() else "failed"
                self.records.append(
                    {
                        "observation_count": observation_count,
                        "gait_time_s": time_s,
                        "wall_time_s": time.time(),
                        "status": status,
                        "xwd_path": str(output_path),
                        "stderr": result.stderr.strip(),
                    }
                )
                self.get_logger().info(
                    f"Snapshot {status}: gait_t={time_s:.1f} s, observation={observation_count}."
                )
            except (OSError, subprocess.SubprocessError) as error:
                self.records.append(
                    {
                        "observation_count": job[0] if job else -1,
                        "gait_time_s": job[1] if job else float("nan"),
                        "wall_time_s": time.time(),
                        "status": "failed",
                        "xwd_path": "",
                        "stderr": str(error),
                    }
                )
                self.get_logger().error(f"Snapshot failed: {error}")
            finally:
                self.jobs.task_done()

    def _observation_callback(self, _: Float32MultiArray) -> None:
        self.observation_count += 1
        time_s = self.schedule.get(self.observation_count)
        if time_s is not None:
            self.jobs.put((self.observation_count, time_s))
        if self.observation_count >= 300:
            self.done = True

    def finish(self) -> None:
        self.jobs.join()
        self.jobs.put(None)
        self.worker.join(timeout=5.0)
        manifest_path = self.snapshot_dir / "snapshot_manifest.csv"
        fieldnames = [
            "observation_count",
            "gait_time_s",
            "wall_time_s",
            "status",
            "xwd_path",
            "stderr",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(self.records, key=lambda row: row["observation_count"]))
        self.get_logger().info(f"Wrote snapshot manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()

    rclpy.init()
    node = SnapshotController(args.snapshot_dir)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.finish()
        node.get_logger().info(
            f"Controller finished after {node.observation_count} policy observations."
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
