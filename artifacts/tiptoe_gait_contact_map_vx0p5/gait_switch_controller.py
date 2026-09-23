#!/usr/bin/env python3
"""Switch gait presets and capture synchronized MuJoCo window snapshots."""

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
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray


class GaitSwitchController(Node):
    """Publish gait selections and schedule screenshots from the 100 Hz policy clock."""

    def __init__(self, snapshot_dir: Path, switch_counts: tuple[int, int, int]) -> None:
        super().__init__("tiptoe_gait_switch_controller")
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.publisher = self.create_publisher(Joy, "/joy", 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/obs_MuJoCo",
            self._observation_callback,
            10,
        )
        self.observation_count = 0
        self.done = False
        # Publish slightly early to compensate the measured ROS-to-simulator Joy
        # callback latency, so applied gait boundaries land at 3/6/9 simulation s.
        self.switch_schedule = {
            switch_counts[0]: 1,
            switch_counts[1]: 2,
            switch_counts[2]: 3,
        }
        self.snapshot_schedule = {
            mode * 300 + local_count: (mode, local_count / 100.0)
            for mode in range(4)
            for local_count in (50, 150, 250)
        }
        self.snapshot_queue: queue.Queue[tuple[int, int, float] | None] = queue.Queue()
        self.snapshot_records: list[dict[str, object]] = []
        self.snapshot_worker = threading.Thread(
            target=self._snapshot_worker,
            name="mujoco-snapshot-worker",
            daemon=True,
        )
        self.snapshot_worker.start()
        self.get_logger().info(
            "Waiting for policy observations; initial mode=0, latency-compensated "
            f"gait requests at {switch_counts[0]}/{switch_counts[1]}/{switch_counts[2]} "
            "observations for applied 3/6/9 s boundaries, "
            "snapshots at local t=0.5/1.5/2.5 s."
        )

    def _publish_mode(self, mode: int) -> None:
        press = Joy()
        press.axes = [0.0, 0.0, 0.0, 0.0]
        press.buttons = [0, 0, 0, 0]
        press.buttons[mode] = 1
        self.publisher.publish(press)

        release = Joy()
        release.axes = [0.0, 0.0, 0.0, 0.0]
        release.buttons = [0, 0, 0, 0]
        self.publisher.publish(release)
        self.get_logger().info(
            f"Switched to gait mode {mode} at policy observation {self.observation_count}."
        )

    def _snapshot_worker(self) -> None:
        while True:
            job = self.snapshot_queue.get()
            try:
                if job is None:
                    return
                observation_count, mode, local_time_s = job
                time_slug = f"{local_time_s:.1f}".replace(".", "p")
                output_path = self.snapshot_dir / f"mode{mode}_t{time_slug}s.xwd"
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
                record = {
                    "observation_count": observation_count,
                    "gait_mode": mode,
                    "segment_time_s": local_time_s,
                    "wall_time_s": time.time(),
                    "status": status,
                    "xwd_path": str(output_path),
                    "stderr": result.stderr.strip(),
                }
                self.snapshot_records.append(record)
                self.get_logger().info(
                    f"Snapshot {status}: mode={mode}, segment_t={local_time_s:.1f} s, "
                    f"observation={observation_count}."
                )
            except (OSError, subprocess.SubprocessError) as error:
                self.snapshot_records.append(
                    {
                        "observation_count": job[0] if job else -1,
                        "gait_mode": job[1] if job else -1,
                        "segment_time_s": job[2] if job else float("nan"),
                        "wall_time_s": time.time(),
                        "status": "failed",
                        "xwd_path": "",
                        "stderr": str(error),
                    }
                )
                self.get_logger().error(f"Snapshot failed: {error}")
            finally:
                self.snapshot_queue.task_done()

    def _observation_callback(self, _: Float32MultiArray) -> None:
        self.observation_count += 1
        snapshot = self.snapshot_schedule.get(self.observation_count)
        if snapshot is not None:
            mode, local_time_s = snapshot
            self.snapshot_queue.put((self.observation_count, mode, local_time_s))
        mode = self.switch_schedule.get(self.observation_count)
        if mode is not None:
            self._publish_mode(mode)
        if self.observation_count >= 1200:
            self.done = True

    def finish_snapshots(self) -> None:
        self.snapshot_queue.join()
        self.snapshot_queue.put(None)
        self.snapshot_worker.join(timeout=5.0)
        manifest_path = self.snapshot_dir / "snapshot_manifest.csv"
        fieldnames = [
            "observation_count",
            "gait_mode",
            "segment_time_s",
            "wall_time_s",
            "status",
            "xwd_path",
            "stderr",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(self.snapshot_records, key=lambda row: row["observation_count"]))
        self.get_logger().info(
            f"Wrote {len(self.snapshot_records)} snapshot records to {manifest_path}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--switch-counts",
        type=int,
        nargs=3,
        default=(297, 598, 898),
        metavar=("MODE1", "MODE2", "MODE3"),
    )
    args = parser.parse_args()

    rclpy.init()
    node = GaitSwitchController(args.snapshot_dir, tuple(args.switch_counts))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.finish_snapshots()
        node.get_logger().info(
            f"Controller finished after {node.observation_count} policy observations."
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
