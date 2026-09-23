#!/usr/bin/env python3
"""Stateful 100 Hz wrapper for the exported MiniPB v10 Hybrid ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class MiniPBV10HybridController:
    def __init__(self, bundle: Path, provider: str = "CPUExecutionProvider") -> None:
        self.bundle = bundle.resolve()
        self.manifest = json.loads(
            (self.bundle / "export_manifest.json").read_text(encoding="utf-8")
        )
        self.config = json.loads(
            (self.bundle / "deployment_config.json").read_text(encoding="utf-8")
        )
        self.session = ort.InferenceSession(
            str(self.bundle / self.manifest["hybrid_controller_onnx"]),
            providers=[provider],
        )
        timing = self.config["timing"]
        self.dt = np.float32(timing["policy_step_s"])
        self.period = np.float32(timing["gait_period_s"])
        self.phase_offsets = np.asarray(
            timing["leg_phase_offsets_FL_FR_RL_RR"], dtype=np.float32
        )
        self.lag_steps = int(self.config["residual_projection"]["half_cycle_lag_steps"])
        self.q_default = np.asarray(
            self.config["fusion"]["default_joint_position_rad"], dtype=np.float32
        )
        self.reset()

    def reset(self, base_phase: float = 0.0) -> None:
        self.base_phase = np.float32(base_phase % 1.0)
        self.previous_actor_action = np.zeros(12, dtype=np.float32)
        self.previous_joint_target = self.q_default.copy()
        self.mlp_history = np.zeros((self.lag_steps, 12), dtype=np.float32)
        self.history_cursor = 0
        self.history_age = 0

    @staticmethod
    def _vector(name: str, value: np.ndarray | list[float], size: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain {size} finite floats; got {array.shape}")
        return array

    def _phase_features(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.linalg.norm(command) == 0.0:
            zeros = np.zeros(4, dtype=np.float32)
            return zeros, zeros
        leg_phase = np.remainder(self.base_phase + self.phase_offsets, 1.0)
        angle = np.float32(2.0 * np.pi) * leg_phase
        return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)

    def step(
        self,
        base_angular_velocity_body_rad_s: np.ndarray | list[float],
        projected_gravity_body_unit: np.ndarray | list[float],
        command_vx_vy_yaw_rate: np.ndarray | list[float],
        absolute_joint_position_rad: np.ndarray | list[float],
        joint_velocity_rad_s: np.ndarray | list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        angular_velocity = self._vector(
            "base_angular_velocity_body_rad_s", base_angular_velocity_body_rad_s, 3
        )
        gravity = self._vector("projected_gravity_body_unit", projected_gravity_body_unit, 3)
        command = self._vector("command_vx_vy_yaw_rate", command_vx_vy_yaw_rate, 3)
        joint_position = self._vector(
            "absolute_joint_position_rad", absolute_joint_position_rad, 12
        )
        joint_velocity = self._vector("joint_velocity_rad_s", joint_velocity_rad_s, 12)
        phase_sine, phase_cosine = self._phase_features(command)
        observation = np.concatenate(
            (
                angular_velocity * np.float32(0.25),
                gravity,
                command,
                joint_position,
                joint_velocity * np.float32(0.1),
                self.previous_actor_action,
                phase_sine,
                phase_cosine,
            )
        ).astype(np.float32)[None, :]

        # Isaac Lab advances the gait clock immediately before applying action.
        next_phase = np.float32(np.remainder(self.base_phase + self.dt / self.period, 1.0))
        history_valid = np.float32(self.history_age >= self.lag_steps)
        outputs = self.session.run(
            None,
            {
                "observation": observation,
                "cpg_phase": np.asarray([[next_phase]], dtype=np.float32),
                "previous_joint_target": self.previous_joint_target[None, :],
                "delayed_mlp_action": self.mlp_history[self.history_cursor][None, :],
                "half_cycle_history_valid": np.asarray([[history_valid]], dtype=np.float32),
            },
        )
        joint_target, actor_action, projected_mlp_action, cpg_action = outputs
        self.mlp_history[self.history_cursor] = projected_mlp_action[0]
        self.history_cursor = (self.history_cursor + 1) % self.lag_steps
        self.history_age += 1
        self.base_phase = next_phase
        self.previous_actor_action = actor_action[0].copy()
        self.previous_joint_target = joint_target[0].copy()
        return (
            joint_target[0],
            actor_action[0],
            projected_mlp_action[0],
            cpg_action[0],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    controller = MiniPBV10HybridController(args.bundle)
    for tick in range(40):
        target, actor, projected, cpg = controller.step(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.0, 0.0],
            controller.q_default,
            [0.0] * 12,
        )
        if tick in (0, 34, 35, 39):
            print(
                f"tick={tick:02d} target={np.array2string(target, precision=4)} "
                f"max|actor|={np.max(np.abs(actor)):.4f} "
                f"max|projected|={np.max(np.abs(projected)):.4f} "
                f"max|cpg|={np.max(np.abs(cpg)):.4f}"
            )


if __name__ == "__main__":
    main()
