"""Play RSL-RL checkpoints with PongBot diagnostics and controls.

@version 0.0.2
@update 2026-07-13: Select the paper-barrier runner for its isolated task.
@update 2026-07-12: Apply the training-time dilated-history contract during play.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
from datetime import datetime
import importlib
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTS_ROOT = os.path.join(REPO_ROOT, "exts")


def _register_local_extensions() -> None:
    """Expose local repo extensions and import them so gym tasks are registered."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "rsl_rl"))

    if not os.path.isdir(EXTS_ROOT):
        return

    for entry in sorted(os.scandir(EXTS_ROOT), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        package_root = entry.path
        package_name = entry.name
        package_init = os.path.join(package_root, package_name, "__init__.py")
        if not os.path.isfile(package_init):
            continue

        sys.path.insert(0, package_root)
        importlib.import_module(package_name)


def _remove_default_ground_planes() -> None:
    """Remove global ground-plane prims that may remain in the stage."""
    from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

    for prim_path in ("/World/defaultGroundPlane", "/World/ground_plane"):
        if is_prim_path_valid(prim_path):
            delete_prim(prim_path)


def _remove_robot_embedded_ground_planes() -> None:
    """Remove GroundPlane prims embedded under cloned robot assets."""
    from isaacsim.core.utils.prims import delete_prim, find_matching_prim_paths

    removed = 0
    for pattern in ("/World/envs/env_.*/Robot/GroundPlane", "/World/envs/env_.*/robot/GroundPlane"):
        for prim_path in find_matching_prim_paths(pattern):
            delete_prim(prim_path)
            removed += 1
    if removed:
        print(f"[INFO] Removed {removed} embedded robot GroundPlane prim(s).")


from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--mass_play_payload_kg",
    type=int,
    default=None,
    help="For mass tasks only: fix the play-time payload mass in kg and disable interval payload changes.",
)
parser.add_argument(
    "--mass_status_print_interval",
    type=int,
    default=50,
    help="For mass tasks only: print env0 payload/estimate every N control steps. Set <= 0 to disable.",
)
parser.add_argument(
    "--print_torque",
    action="store_true",
    default=False,
    help="Print env0 12 joint torques during play.",
)
parser.add_argument(
    "--torque_print_interval",
    type=int,
    default=50,
    help="Print env0 12 joint torques every N control steps when --print_torque is set.",
)
parser.add_argument(
    "--control_mode",
    type=str,
    default="policy",
    choices=["policy", "keyboard", "remote_keyboard", "joystick"],
    help="How to generate base velocity commands during play.",
)
parser.add_argument(
    "--fixed_command",
    type=float,
    nargs=3,
    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
    default=None,
    help="Override base velocity commands for all envs as [lin_vel_x, lin_vel_y, ang_vel_z].",
)
parser.add_argument(
    "--gait_mode",
    type=int,
    choices=[0, 1, 2, 3],
    default=None,
    help="Override gait command for all envs with a deployment mode id: 0 slow, 1 nominal, 2 fast, 3 high-clearance.",
)
parser.add_argument(
    "--publish_policy_stream_ros",
    action="store_true",
    default=False,
    help="Publish policy streams to ROS2 topics for PlotJuggler.",
)
parser.add_argument(
    "--record_policy_stream_csv",
    action="store_true",
    default=False,
    help="Record obs_IsaacLab, action_IsaacLab, and imu_encoder CSV files until play.py exits.",
)
parser.add_argument(
    "--policy_stream_csv_dir",
    type=str,
    default=None,
    help="Optional output directory for policy stream CSV files. Defaults to <run>/policy_stream_csv.",
)
parser.add_argument(
    "--record_latent_csv",
    action="store_true",
    default=False,
    help="Record implicit context-estimator latent vectors for offline PCA analysis.",
)
parser.add_argument(
    "--latent_csv_dir",
    type=str,
    default=None,
    help="Optional output directory for latent CSV files. Defaults to <run>/latent_csv.",
)
parser.add_argument(
    "--latent_record_interval",
    type=int,
    default=5,
    help="Record latent vectors every N control steps when --record_latent_csv is set.",
)
parser.add_argument(
    "--latent_record_warmup",
    type=int,
    default=100,
    help="Skip the first N control steps before recording latent vectors.",
)
parser.add_argument(
    "--latent_terrain_label_source",
    type=str,
    default="tile",
    choices=["tile", "step", "position"],
    help="How to map latent samples to terrain_segment labels.",
)
parser.add_argument(
    "--latent_terrain_segments",
    type=str,
    default=None,
    help="Comma-separated terrain segment names, for example 'stair,waves,rough,slope'.",
)
parser.add_argument(
    "--latent_terrain_segment_bounds",
    type=str,
    default=None,
    help="Comma-separated step or position boundaries between terrain segments. Length must be segments-1.",
)
parser.add_argument(
    "--latent_terrain_position_axis",
    type=str,
    default="x",
    choices=["x", "y", "z"],
    help="Robot root position axis used when --latent_terrain_label_source=position.",
)
parser.add_argument(
    "--record_load_latent_csv",
    action="store_true",
    default=False,
    help="Record Phase2 LoadAdaptive z_load vectors and diagnostics to CSV during play.",
)
parser.add_argument(
    "--load_latent_csv_dir",
    type=str,
    default=None,
    help="Optional output directory for load latent CSV files. Defaults to <run>/load_latent_csv.",
)
parser.add_argument(
    "--load_latent_record_interval",
    type=int,
    default=5,
    help="Record load latent vectors every N control steps when --record_load_latent_csv is set.",
)
parser.add_argument(
    "--load_latent_record_warmup",
    type=int,
    default=100,
    help="Skip the first N control steps before recording load latent vectors.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_remove_default_ground_planes()

# Local extensions may import Isaac Sim / Omniverse modules, so register them
# only after the simulator app has been launched.
_register_local_extensions()

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

try:
    import carb.input
except ImportError:
    carb = None
try:
    import omni.appwindow
except ImportError:
    omni_appwindow = None
else:
    omni_appwindow = omni.appwindow
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
except ImportError:
    rclpy = None
    Node = None
    Float32MultiArray = None

from rsl_rl.runner import OnPolicyRunner, PaperBarrierRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
try:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
except ImportError:
    capture_viewport_to_file = None
    get_active_viewport = None
from pongbot_r2.tasks.controllers import JoystickController, LocalKeyboardController, RemoteKeyboardController
from pongbot_r2.utils.wrappers.rsl_rl import (
    RslRlPpoAlgorithmMlpCfg,
    export_encoder_as_onnx,
    export_mlp_as_onnx,
    export_policy_as_jit,
)


def _load_play_checkpoint_forgiving(ppo_runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load a checkpoint for inference while tolerating non-policy state_dict drift."""
    try:
        ppo_runner.load(checkpoint_path)
        return
    except RuntimeError as err:
        error_text = str(err)
        tolerated_errors = ("size mismatch", "Missing key(s)", "Unexpected key(s)")
        if not any(token in error_text for token in tolerated_errors):
            raise

    print("[WARN] Full checkpoint load failed due to state_dict mismatch. Falling back to actor/encoder-only load.")
    loaded_dict = torch.load(checkpoint_path, map_location=ppo_runner.device)

    actor_critic_state = ppo_runner.alg.actor_critic.state_dict()
    checkpoint_actor_critic_state = loaded_dict["model_state_dict"]
    filtered_actor_critic_state = {}
    skipped_actor_critic_keys = []
    for key, value in checkpoint_actor_critic_state.items():
        if key.startswith("critic."):
            skipped_actor_critic_keys.append(key)
            continue
        if key not in actor_critic_state or actor_critic_state[key].shape != value.shape:
            skipped_actor_critic_keys.append(key)
            continue
        filtered_actor_critic_state[key] = value

    missing_actor_critic_keys, unexpected_actor_critic_keys = ppo_runner.alg.actor_critic.load_state_dict(
        filtered_actor_critic_state, strict=False
    )

    encoder_state = ppo_runner.alg.encoder.state_dict()
    checkpoint_encoder_state = loaded_dict["encoder_state_dict"]
    filtered_encoder_state = {}
    skipped_encoder_keys = []
    for key, value in checkpoint_encoder_state.items():
        if key not in encoder_state or encoder_state[key].shape != value.shape:
            skipped_encoder_keys.append(key)
            continue
        filtered_encoder_state[key] = value

    missing_encoder_keys, unexpected_encoder_keys = ppo_runner.alg.encoder.load_state_dict(
        filtered_encoder_state, strict=False
    )

    ppo_runner.current_learning_iteration = loaded_dict.get("iter", 0)
    print(
        "[WARN] Partial checkpoint load summary:",
        f"skipped_actor_critic={len(skipped_actor_critic_keys)}",
        f"missing_actor_critic={len(missing_actor_critic_keys)}",
        f"unexpected_actor_critic={len(unexpected_actor_critic_keys)}",
        f"skipped_encoder={len(skipped_encoder_keys)}",
        f"missing_encoder={len(missing_encoder_keys)}",
        f"unexpected_encoder={len(unexpected_encoder_keys)}",
    )


def _build_manual_controller_cfg(env_cfg: ManagerBasedRLEnvCfg) -> SimpleNamespace:
    """Convert velocity-command ranges into the controller config shape expected by controllers.py."""
    ranges = env_cfg.commands.base_velocity.ranges
    return SimpleNamespace(
        command_minus_x_range=ranges.lin_vel_x[0],
        command_plus_x_range=ranges.lin_vel_x[1],
        command_minus_y_range=ranges.lin_vel_y[0],
        command_plus_y_range=ranges.lin_vel_y[1],
        command_minus_yaw_range=ranges.ang_vel_z[0],
        command_plus_yaw_range=ranges.ang_vel_z[1],
    )


def _create_manual_controller(control_mode: str, env_cfg: ManagerBasedRLEnvCfg):
    """Create a manual controller instance for the requested mode."""
    if control_mode == "policy":
        return None

    cfg = _build_manual_controller_cfg(env_cfg)
    controller_types = {
        "keyboard": LocalKeyboardController,
        "remote_keyboard": RemoteKeyboardController,
        "joystick": JoystickController,
    }
    controller = controller_types[control_mode](cfg)
    controller.start()
    return controller


GAIT_MODE_COMMANDS = (
    (1.20, 0.50, 0.66, 0.07),  # slow trot
    (1.45, 0.50, 0.60, 0.08),  # nominal trot
    (1.75, 0.50, 0.54, 0.08),  # fast trot
    (1.45, 0.50, 0.60, 0.13),  # high-clearance trot
)


def _apply_manual_command(env, command: torch.Tensor) -> torch.Tensor:
    """Overwrite the base-velocity command term with the user command for all envs."""
    base_env = env.unwrapped
    device = base_env.device
    command = command.to(device=device, dtype=torch.float32)
    if command.shape[0] == 1 and base_env.num_envs > 1:
        command = command.repeat(base_env.num_envs, 1)

    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:] = command
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "time_left"):
        command_term.time_left[:] = float("inf")

    return command_term.command.clone()


def _apply_gait_mode(env, gait_mode: int) -> torch.Tensor:
    """Overwrite the gait command term with one of the deployment gait modes."""
    base_env = env.unwrapped
    device = base_env.device
    command = torch.tensor([GAIT_MODE_COMMANDS[gait_mode]], device=device, dtype=torch.float32)
    if base_env.num_envs > 1:
        command = command.repeat(base_env.num_envs, 1)

    command_term = base_env.command_manager.get_term("gait_command")
    if not hasattr(command_term, "gait_command"):
        raise RuntimeError("The 'gait_command' term does not expose a writable gait_command buffer.")

    command_term.gait_command[:] = command
    if hasattr(command_term, "time_left"):
        command_term.time_left[:] = float("inf")

    return command_term.command.clone()


def _flatten_obs_history(obs_history: torch.Tensor) -> torch.Tensor:
    return obs_history.flatten(start_dim=1) if obs_history.dim() > 2 else obs_history


def _prepare_play_obs_history(
    obs_history: torch.Tensor,
    uses_context_estimator: bool,
    history_offsets: tuple[int, ...] = (),
    expected_history_len: int | None = None,
) -> torch.Tensor:
    if uses_context_estimator:
        if obs_history.dim() != 3:
            raise RuntimeError(f"CE Net requires obsHistory [batch, H, obs_dim], got {tuple(obs_history.shape)}")
        if history_offsets:
            raw_history_len = obs_history.shape[1]
            if min(history_offsets) < 0 or max(history_offsets) >= raw_history_len:
                raise RuntimeError(
                    f"history_offsets={history_offsets} are invalid for raw history length {raw_history_len}"
                )
            indices = torch.as_tensor(
                [raw_history_len - 1 - offset for offset in history_offsets],
                device=obs_history.device,
                dtype=torch.long,
            )
            obs_history = torch.index_select(obs_history, dim=1, index=indices)
        if expected_history_len is not None and obs_history.shape[1] != expected_history_len:
            raise RuntimeError(
                f"Prepared obsHistory length mismatch: got {obs_history.shape[1]}, expected {expected_history_len}"
            )
        return obs_history
    return _flatten_obs_history(obs_history)


def _supports_mass_play_features(env_cfg: ManagerBasedRLEnvCfg) -> bool:
    events_cfg = getattr(env_cfg, "events", None)
    observations_cfg = getattr(env_cfg, "observations", None)
    encoder_target_cfg = getattr(observations_cfg, "encoder_target", None) if observations_cfg is not None else None
    has_payload_events = events_cfg is not None and hasattr(events_cfg, "payload_reset")
    has_payload_target = encoder_target_cfg is not None and hasattr(encoder_target_cfg, "payload_mass")
    return bool(has_payload_events and has_payload_target)


def _configure_mass_play_payload(env_cfg: ManagerBasedRLEnvCfg) -> None:
    if args_cli.mass_play_payload_kg is None or not _supports_mass_play_features(env_cfg):
        return

    payload_mass = float(args_cli.mass_play_payload_kg)
    events_cfg = env_cfg.events

    if hasattr(events_cfg, "payload_reset"):
        events_cfg.payload_reset.params["payload_mass_range"] = (payload_mass, payload_mass)
        events_cfg.payload_reset.params["zero_payload_prob"] = 0.0

    if hasattr(events_cfg, "payload_interval"):
        events_cfg.payload_interval.params["payload_mass_range"] = (payload_mass, payload_mass)
        events_cfg.payload_interval.params["zero_payload_prob"] = 0.0
        events_cfg.payload_interval.params["probability"] = 0.0

    print(
        "[INFO] Mass-task play override:",
        f"fixed payload mass={args_cli.mass_play_payload_kg} kg,",
        "interval payload changes disabled.",
    )


def _get_mass_task_status(base_env, encoder_module) -> tuple[float | None, float | None]:
    actual_mass = None
    estimated_mass = None

    payload_mass = getattr(base_env, "_payload_mass", None)
    if payload_mass is not None and payload_mass.numel() > 0:
        actual_mass = float(payload_mass[0].item())

    if hasattr(encoder_module, "get_encoder_out"):
        raw_out = encoder_module.get_encoder_out()
        if raw_out is not None and raw_out.numel() > 0:
            estimated_mass = float(raw_out[0, 0].item())

    return actual_mass, estimated_mass


def _get_payload_mass_env0(base_env) -> float | None:
    payload_mass = getattr(base_env, "_payload_mass", None)
    if payload_mass is None or payload_mass.numel() == 0:
        return None
    return float(payload_mass.reshape(-1)[0].item())


def _print_payload_mass_if_changed(base_env, step_count: int, last_payload_mass: float | None) -> float | None:
    payload_mass = _get_payload_mass_env0(base_env)
    if payload_mass is None:
        return last_payload_mass
    if last_payload_mass is None or abs(payload_mass - last_payload_mass) > 1.0e-6:
        print(
            "[PAYLOAD]",
            f"step={step_count}",
            f"payload_mass_env0={payload_mass:.2f} kg",
            flush=True,
        )
    return payload_mass


def _set_payload_mass_from_play(base_env, payload_mass_kg: float, step_count: int) -> float | None:
    events_cfg = getattr(base_env.cfg, "events", None)
    payload_reset = getattr(events_cfg, "payload_reset", None) if events_cfg is not None else None
    if payload_reset is None:
        print("[WARN] Payload adjustment ignored: this task has no payload_reset event.", flush=True)
        return _get_payload_mass_env0(base_env)

    params = dict(payload_reset.params)
    asset_cfg = params.get("asset_cfg")
    if asset_cfg is not None and hasattr(asset_cfg, "resolve"):
        import copy

        asset_cfg = copy.copy(asset_cfg)
        asset_cfg.resolve(base_env.scene)
        params["asset_cfg"] = asset_cfg
    params["payload_mass_range"] = (payload_mass_kg, payload_mass_kg)
    params["payload_mass_choices"] = None
    params["zero_payload_prob"] = 0.0
    payload_reset.func(base_env, None, **params)

    actual_mass = _get_payload_mass_env0(base_env)
    actual_str = f"{actual_mass:.2f}" if actual_mass is not None else "n/a"
    print(
        "[PAYLOAD_SET]",
        f"step={step_count}",
        f"payload_mass_env0={actual_str} kg",
        flush=True,
    )
    return actual_mass


def _handle_joystick_payload_buttons(
    base_env,
    controller,
    step_count: int,
    payload_buttons_were_active: bool,
) -> tuple[bool, float | None]:
    if controller is None or not hasattr(controller, "get_button"):
        return payload_buttons_were_active, _get_payload_mass_env0(base_env)

    plus_pressed = controller.get_button(2) > 0
    minus_pressed = controller.get_button(0) > 0
    buttons_active = plus_pressed or minus_pressed
    if not buttons_active:
        return False, _get_payload_mass_env0(base_env)
    if payload_buttons_were_active:
        return True, _get_payload_mass_env0(base_env)

    current_payload = _get_payload_mass_env0(base_env)
    if current_payload is None:
        current_payload = 0.0
    delta = 30.0 if plus_pressed else -5.0
    next_payload = max(0.0, min(30.0, current_payload + delta))
    actual_payload = _set_payload_mass_from_play(base_env, next_payload, step_count)
    return True, actual_payload


def _get_joint_torques_env0(base_env) -> torch.Tensor | None:
    try:
        torque = base_env.scene["robot"].data.applied_torque
    except Exception:
        return None
    if torque is None or torque.numel() == 0:
        return None
    return torque[0].detach().cpu()


def _print_joint_torques_env0(base_env, step_count: int) -> None:
    torque = _get_joint_torques_env0(base_env)
    if torque is None:
        return
    torque_12 = torque[:12].tolist()
    torque_str = ", ".join(f"{value:.2f}" for value in torque_12)
    print(
        "[TORQUE]",
        f"step={step_count}",
        f"joint_torque_12=[{torque_str}] Nm",
        flush=True,
    )


class ViewportCaptureHotkey:
    """Capture the active Isaac Sim viewport when the user presses P."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._save_requested = False
        self._keyboard_sub = None
        self._viewport_api = None
        if carb is None or omni_appwindow is None or capture_viewport_to_file is None or get_active_viewport is None:
            print("[INFO] Viewport capture hotkey disabled: viewport input/capture APIs are unavailable.")
            self._input = None
            self._app_window = None
            self._keyboard = None
            return
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni_appwindow.get_default_app_window()
        self._keyboard = None

        if self._app_window is None:
            print("[WARN] Default app window not found. Viewport capture hotkey disabled.")
            return

        self._keyboard = self._app_window.get_keyboard()
        if self._keyboard is None:
            print("[WARN] App window keyboard not found. Viewport capture hotkey disabled.")
            return

        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        print(f"[INFO] Viewport capture hotkey enabled. Press 'P' to save images under: {self.output_dir}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input == carb.input.KeyboardInput.P
        ):
            self._save_requested = True
        return True

    def _get_viewport_api(self):
        if self._viewport_api is None:
            self._viewport_api = get_active_viewport()
        return self._viewport_api

    def update(self) -> None:
        if not self._save_requested:
            return

        viewport_api = self._get_viewport_api()
        if viewport_api is None:
            print("[WARN] Active viewport not available yet. Skipping capture request.")
            self._save_requested = False
            return

        self._save_requested = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(self.output_dir, f"viewport_{timestamp}.png")
        capture_viewport_to_file(viewport_api, file_path)
        print(f"[INFO] Saved viewport capture to: {file_path}")

    def close(self) -> None:
        if self._keyboard_sub is not None and self._keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None


class PolicyStreamRosPublisher:
    """Publish policy streams for env 0 as ROS2 Float32MultiArray topics."""

    def __init__(self):
        if rclpy is None or Float32MultiArray is None:
            raise ImportError("rclpy/std_msgs is required to publish policy streams to ROS2 topics.")

        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)

        self._node = Node("isaaclab_policy_stream_publisher")
        self._obs_publisher = self._node.create_publisher(Float32MultiArray, "obs_Isaaclab", 10)
        self._action_publisher = self._node.create_publisher(Float32MultiArray, "action_isaaclab", 10)

    def publish(
        self, encoder_output: torch.Tensor, obs: torch.Tensor, vel_command: torch.Tensor, policy_output: torch.Tensor
    ) -> None:
        obs_msg = Float32MultiArray()
        obs_msg.data = torch.cat((encoder_output[0], obs[0], vel_command[0]), dim=-1).detach().cpu().tolist()
        self._obs_publisher.publish(obs_msg)

        action_msg = Float32MultiArray()
        action_msg.data = policy_output[0].detach().cpu().tolist()
        self._action_publisher.publish(action_msg)

        rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()


class PolicyStreamCsvRecorder:
    """Record policy streams for env 0 to CSV files until stopped or an optional step limit is reached."""

    def __init__(self, output_dir: str, max_steps: int | None = None):
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.start_time = None
        self.step_count = 0
        self.obs_file = None
        self.action_file = None
        self.imu_file = None
        self.obs_writer = None
        self.action_writer = None
        self.imu_writer = None
        self._finished = False

        os.makedirs(self.output_dir, exist_ok=True)

    def _init_writers(
        self,
        encoder_output: torch.Tensor,
        obs: torch.Tensor,
        vel_command: torch.Tensor,
        policy_output: torch.Tensor,
        imu_gt: torch.Tensor | None = None,
    ):
        obs_path = os.path.join(self.output_dir, "obs_IsaacLab.csv")
        action_path = os.path.join(self.output_dir, "action_IsaacLab.csv")
        imu_path = os.path.join(self.output_dir, "imu_encoder.csv")

        self.obs_file = open(obs_path, "w", newline="", encoding="utf-8")
        self.action_file = open(action_path, "w", newline="", encoding="utf-8")
        self.imu_file = open(imu_path, "w", newline="", encoding="utf-8")
        self.obs_writer = csv.writer(self.obs_file)
        self.action_writer = csv.writer(self.action_file)
        self.imu_writer = csv.writer(self.imu_file)

        obs_header = ["step"]
        obs_header += [f"encoder_output_{i}" for i in range(encoder_output.shape[1])]
        obs_header += [f"obs_{i}" for i in range(obs.shape[1])]
        obs_header += [f"vel_command_{i}" for i in range(vel_command.shape[1])]
        action_header = ["step"]
        action_header += [f"policy_output_{i}" for i in range(policy_output.shape[1])]

        self.obs_writer.writerow(obs_header)
        self.action_writer.writerow(action_header)
        if imu_gt is not None:
            imu_header = ["step"]
            imu_header += [f"imu_est_{i}" for i in range(imu_gt.shape[1])]
            imu_header += [f"imu_gt_{i}" for i in range(imu_gt.shape[1])]
            imu_header += [f"imu_err_{i}" for i in range(imu_gt.shape[1])]
            imu_header += ["lin_vel_err_l2", "ang_vel_err_l2", "proj_gravity_err_l2", "imu_err_l2"]
            self.imu_writer.writerow(imu_header)

    def record(
        self,
        encoder_output: torch.Tensor,
        obs: torch.Tensor,
        vel_command: torch.Tensor,
        policy_output: torch.Tensor,
        imu_gt: torch.Tensor | None = None,
    ) -> None:
        if self._finished:
            return

        if self.start_time is None:
            self.start_time = 0
            self._init_writers(encoder_output, obs, vel_command, policy_output, imu_gt)

        if self.max_steps is not None and self.step_count >= self.max_steps:
            self._finished = True
            self.close()
            return

        obs_row = [self.step_count]
        obs_row += encoder_output[0].detach().cpu().tolist()
        obs_row += obs[0].detach().cpu().tolist()
        obs_row += vel_command[0].detach().cpu().tolist()
        self.obs_writer.writerow(obs_row)
        self.obs_file.flush()

        action_row = [self.step_count]
        action_row += policy_output[0].detach().cpu().tolist()
        self.action_writer.writerow(action_row)
        self.action_file.flush()

        if imu_gt is not None:
            imu_est = encoder_output[0, : imu_gt.shape[1]]
            imu_gt_env0 = imu_gt[0]
            imu_err = imu_est - imu_gt_env0
            imu_row = [self.step_count]
            imu_row += imu_est.detach().cpu().tolist()
            imu_row += imu_gt_env0.detach().cpu().tolist()
            imu_row += imu_err.detach().cpu().tolist()
            imu_row += [
                torch.norm(imu_err[0:3]).item(),
                torch.norm(imu_err[3:6]).item(),
                torch.norm(imu_err[6:9]).item(),
                torch.norm(imu_err).item(),
            ]
            self.imu_writer.writerow(imu_row)
            self.imu_file.flush()

        self.step_count += 1
        if self.max_steps is not None and self.step_count >= self.max_steps:
            self._finished = True
            self.close()

    def close(self) -> None:
        if self.obs_file is not None and not self.obs_file.closed:
            self.obs_file.close()
        if self.action_file is not None and not self.action_file.closed:
            self.action_file.close()
        if self.imu_file is not None and not self.imu_file.closed:
            self.imu_file.close()


class LatentCsvRecorder:
    """Record all-env implicit CE latent vectors with terrain labels for offline PCA."""

    def __init__(
        self,
        output_dir: str,
        terrain_names: list[str],
        segment_names: list[str] | None = None,
        segment_bounds: list[float] | None = None,
        segment_ids: torch.Tensor | None = None,
    ):
        self.output_dir = output_dir
        self.terrain_names = terrain_names
        self.segment_names = segment_names
        self.segment_bounds = segment_bounds
        self.segment_ids = segment_ids.detach().cpu().reshape(-1) if segment_ids is not None else None
        self.file = None
        self.writer = None
        self._header_written = False

        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, "latent.csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

    def _terrain_label(self, terrain_type_id: int) -> str:
        if 0 <= terrain_type_id < len(self.terrain_names):
            return self.terrain_names[terrain_type_id]
        return f"unknown_{terrain_type_id}"

    def _terrain_group_label(self, terrain_type_id: int) -> str:
        terrain_label = self._terrain_label(terrain_type_id)
        if "stair" in terrain_label:
            return "stair"
        if "wave" in terrain_label:
            return "wave"
        if "slope" in terrain_label:
            return "slope"
        if terrain_label in {"boxes", "random_rough"} or "rough" in terrain_label or "box" in terrain_label:
            return "rough"
        return terrain_label

    def _segment_label(self, terrain_type_id: int, segment_value: float | None, segment_id: int | None) -> str:
        if segment_id is not None and self.segment_names is not None:
            if 0 <= segment_id < len(self.segment_names):
                return self.segment_names[segment_id]
            return f"unknown_segment_{segment_id}"

        if self.segment_names is None:
            return self._terrain_group_label(terrain_type_id)

        if segment_value is None or self.segment_bounds is None:
            return self.segment_names[0]

        segment_idx = 0
        while segment_idx < len(self.segment_bounds) and segment_value >= self.segment_bounds[segment_idx]:
            segment_idx += 1
        if 0 <= segment_idx < len(self.segment_names):
            return self.segment_names[segment_idx]
        return f"unknown_segment_{segment_idx}"

    def _write_header(self, latent_dim: int, command_dim: int, metric_names: list[str] | None = None) -> None:
        header = [
            "step",
            "env_id",
            "terrain_type_id",
            "terrain_type",
            "terrain_segment",
            "terrain_segment_value",
            "terrain_level",
            "root_x",
            "root_y",
            "root_z",
        ]
        header += [f"command_{i}" for i in range(command_dim)]
        if metric_names is not None:
            header += metric_names
        header += [f"mu_{i}" for i in range(latent_dim)]
        header += [f"z_{i}" for i in range(latent_dim)]
        header += [f"logvar_{i}" for i in range(latent_dim)]
        self.writer.writerow(header)
        self._header_written = True

    def record(
        self,
        step: int,
        latent_mu: torch.Tensor,
        latent_z: torch.Tensor,
        latent_logvar: torch.Tensor,
        commands: torch.Tensor,
        terrain_type_ids: torch.Tensor,
        terrain_levels: torch.Tensor | None = None,
        root_pos_w: torch.Tensor | None = None,
        segment_values: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self._header_written:
            metric_names = list(metrics.keys()) if metrics is not None else None
            self._write_header(latent_mu.shape[1], commands.shape[1], metric_names)

        latent_mu = latent_mu.detach().cpu()
        latent_z = latent_z.detach().cpu()
        latent_logvar = latent_logvar.detach().cpu()
        commands = commands.detach().cpu()
        terrain_type_ids = terrain_type_ids.detach().cpu().reshape(-1)
        if terrain_levels is not None:
            terrain_levels = terrain_levels.detach().cpu().reshape(-1)
        if root_pos_w is not None:
            root_pos_w = root_pos_w.detach().cpu()
        if segment_values is not None:
            segment_values = segment_values.detach().cpu().reshape(-1)
        if segment_ids is not None:
            segment_ids = segment_ids.detach().cpu().reshape(-1)
        if metrics is not None:
            metrics = {name: value.detach().cpu().reshape(-1) for name, value in metrics.items()}

        for env_id in range(latent_mu.shape[0]):
            terrain_type_id = int(terrain_type_ids[env_id].item())
            terrain_level = int(terrain_levels[env_id].item()) if terrain_levels is not None else -1
            root_pos = root_pos_w[env_id].tolist() if root_pos_w is not None else [float("nan")] * 3
            segment_value = float(segment_values[env_id].item()) if segment_values is not None else None
            segment_id = int(segment_ids[env_id].item()) if segment_ids is not None else None
            row = [
                step,
                env_id,
                terrain_type_id,
                self._terrain_label(terrain_type_id),
                self._segment_label(terrain_type_id, segment_value, segment_id),
                segment_value if segment_value is not None else float("nan"),
                terrain_level,
                root_pos[0],
                root_pos[1],
                root_pos[2],
            ]
            row += commands[env_id].tolist()
            if metrics is not None:
                row += [float(value[env_id].item()) for value in metrics.values()]
            row += latent_mu[env_id].tolist()
            row += latent_z[env_id].tolist()
            row += latent_logvar[env_id].tolist()
            self.writer.writerow(row)

        self.file.flush()

    def close(self) -> None:
        if self.file is not None and not self.file.closed:
            self.file.close()


class LoadLatentCsvRecorder:
    """Record Phase2 LoadAdaptive z_load vectors with payload/action diagnostics."""

    def __init__(
        self,
        output_dir: str,
        terrain_names: list[str],
        segment_names: list[str] | None = None,
        segment_bounds: list[float] | None = None,
    ):
        self.output_dir = output_dir
        self.terrain_names = terrain_names
        self.segment_names = segment_names
        self.segment_bounds = segment_bounds
        self.file = None
        self.writer = None
        self._header_written = False

        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, "load_latent.csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

    def _terrain_label(self, terrain_type_id: int) -> str:
        if 0 <= terrain_type_id < len(self.terrain_names):
            return self.terrain_names[terrain_type_id]
        return f"unknown_{terrain_type_id}"

    def _terrain_group_label(self, terrain_type_id: int) -> str:
        terrain_label = self._terrain_label(terrain_type_id)
        if "stair" in terrain_label:
            return "stair"
        if "wave" in terrain_label:
            return "wave"
        if "slope" in terrain_label:
            return "slope"
        if terrain_label in {"boxes", "random_rough"} or "rough" in terrain_label or "box" in terrain_label:
            return "rough"
        return terrain_label

    def _segment_label(self, terrain_type_id: int, segment_value: float | None, segment_id: int | None) -> str:
        if self.segment_names is None:
            return self._terrain_group_label(terrain_type_id)
        if segment_value is None or self.segment_bounds is None:
            return self.segment_names[0]

        segment_idx = 0
        while segment_idx < len(self.segment_bounds) and segment_value >= self.segment_bounds[segment_idx]:
            segment_idx += 1
        if 0 <= segment_idx < len(self.segment_names):
            return self.segment_names[segment_idx]
        return f"unknown_segment_{segment_idx}"

    @staticmethod
    def _payload_mass(base_env, num_envs: int, device) -> torch.Tensor:
        payload_mass = getattr(base_env, "_payload_mass", None)
        if payload_mass is None:
            return torch.full((num_envs,), float("nan"), device=device)
        return payload_mass.to(device=device).reshape(-1)

    @staticmethod
    def _payload_pos_b(base_env, num_envs: int, device) -> torch.Tensor:
        payload_pos_b = getattr(base_env, "_payload_pos_b", None)
        if payload_pos_b is None:
            return torch.full((num_envs, 3), float("nan"), device=device)
        return payload_pos_b.to(device=device).reshape(num_envs, -1)[:, :3]

    def _write_header(
        self,
        z_load_dim: int,
        command_dim: int,
        action_dim: int,
        metric_names: list[str] | None = None,
    ) -> None:
        header = [
            "step",
            "env_id",
            "payload_mass",
            "payload_mass_bin",
            "payload_pos_x",
            "payload_pos_y",
            "payload_pos_z",
            "terrain_type_id",
            "terrain_type",
            "terrain_segment",
            "terrain_segment_value",
            "terrain_level",
            "root_x",
            "root_y",
            "root_z",
        ]
        header += [f"command_{i}" for i in range(command_dim)]
        if metric_names is not None:
            header += metric_names
        header += [f"nominal_action_{i}" for i in range(action_dim)]
        header += [f"delta_action_{i}" for i in range(action_dim)]
        header += [f"final_action_{i}" for i in range(action_dim)]
        header += ["z_load_norm", "nominal_action_norm", "delta_action_norm", "final_action_norm"]
        header += [f"z_load_{i}" for i in range(z_load_dim)]
        self.writer.writerow(header)
        self._header_written = True

    @staticmethod
    def _payload_mass_bin(payload_mass: float) -> str:
        if payload_mass != payload_mass:
            return "unknown"
        lower = int(max(0.0, min(30.0, payload_mass)) // 5 * 5)
        upper = min(lower + 5, 30)
        return f"{lower}_{upper}kg"

    def record(
        self,
        step: int,
        z_load: torch.Tensor,
        commands: torch.Tensor,
        nominal_action: torch.Tensor,
        delta_action: torch.Tensor,
        final_action: torch.Tensor,
        base_env,
        terrain_type_ids: torch.Tensor,
        terrain_levels: torch.Tensor | None = None,
        root_pos_w: torch.Tensor | None = None,
        segment_values: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self._header_written:
            metric_names = list(metrics.keys()) if metrics is not None else None
            self._write_header(z_load.shape[1], commands.shape[1], final_action.shape[1], metric_names)

        num_envs = z_load.shape[0]
        device = z_load.device
        payload_mass = self._payload_mass(base_env, num_envs, device).detach().cpu()
        payload_pos_b = self._payload_pos_b(base_env, num_envs, device).detach().cpu()
        z_load = z_load.detach().cpu()
        commands = commands.detach().cpu()
        nominal_action = nominal_action.detach().cpu()
        delta_action = delta_action.detach().cpu()
        final_action = final_action.detach().cpu()
        terrain_type_ids = terrain_type_ids.detach().cpu().reshape(-1)
        if terrain_levels is not None:
            terrain_levels = terrain_levels.detach().cpu().reshape(-1)
        if root_pos_w is not None:
            root_pos_w = root_pos_w.detach().cpu()
        if segment_values is not None:
            segment_values = segment_values.detach().cpu().reshape(-1)
        if segment_ids is not None:
            segment_ids = segment_ids.detach().cpu().reshape(-1)
        if metrics is not None:
            metrics = {name: value.detach().cpu().reshape(-1) for name, value in metrics.items()}

        for env_id in range(num_envs):
            terrain_type_id = int(terrain_type_ids[env_id].item())
            terrain_level = int(terrain_levels[env_id].item()) if terrain_levels is not None else -1
            root_pos = root_pos_w[env_id].tolist() if root_pos_w is not None else [float("nan")] * 3
            segment_value = float(segment_values[env_id].item()) if segment_values is not None else None
            segment_id = int(segment_ids[env_id].item()) if segment_ids is not None else None
            mass = float(payload_mass[env_id].item())
            row = [
                step,
                env_id,
                mass,
                self._payload_mass_bin(mass),
                float(payload_pos_b[env_id, 0].item()),
                float(payload_pos_b[env_id, 1].item()),
                float(payload_pos_b[env_id, 2].item()),
                terrain_type_id,
                self._terrain_label(terrain_type_id),
                self._segment_label(terrain_type_id, segment_value, segment_id),
                segment_value if segment_value is not None else float("nan"),
                terrain_level,
                root_pos[0],
                root_pos[1],
                root_pos[2],
            ]
            row += commands[env_id].tolist()
            if metrics is not None:
                row += [float(value[env_id].item()) for value in metrics.values()]
            row += nominal_action[env_id].tolist()
            row += delta_action[env_id].tolist()
            row += final_action[env_id].tolist()
            row += [
                torch.linalg.norm(z_load[env_id]).item(),
                torch.linalg.norm(nominal_action[env_id]).item(),
                torch.linalg.norm(delta_action[env_id]).item(),
                torch.linalg.norm(final_action[env_id]).item(),
            ]
            row += z_load[env_id].tolist()
            self.writer.writerow(row)

        self.file.flush()

    def close(self) -> None:
        if self.file is not None and not self.file.closed:
            self.file.close()


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items]


def _terrain_name_to_group(terrain_name: str) -> str:
    if "stair" in terrain_name:
        return "stair"
    if "wave" in terrain_name:
        return "wave"
    if "slope" in terrain_name:
        return "slope"
    if terrain_name in {"boxes", "random_rough"} or "rough" in terrain_name or "box" in terrain_name:
        return "rough"
    return terrain_name


def _active_terrain_groups_from_generator(terrain_generator) -> list[str] | None:
    if terrain_generator is None or not hasattr(terrain_generator, "sub_terrains"):
        return None

    active_groups = []
    for terrain_name, terrain_cfg in terrain_generator.sub_terrains.items():
        if float(getattr(terrain_cfg, "proportion", 0.0)) <= 0.0:
            continue
        group_name = _terrain_name_to_group(terrain_name)
        active_groups.append(group_name)

    return active_groups or None


def _default_rough_play_segments(terrain_generator) -> tuple[list[str] | None, list[float] | None]:
    segment_names = _active_terrain_groups_from_generator(terrain_generator)
    if segment_names is None or len(segment_names) <= 1:
        return None, None

    terrain_size_x = float(terrain_generator.size[0])
    segment_bounds = [terrain_size_x * idx for idx in range(1, len(segment_names))]
    return segment_names, segment_bounds


def _prepare_latent_segment_mapping(terrain_generator) -> tuple[list[str] | None, list[float] | None]:
    if args_cli.latent_terrain_label_source == "tile":
        return None, None

    segment_names = _parse_csv_strings(args_cli.latent_terrain_segments)
    segment_bounds = _parse_csv_floats(args_cli.latent_terrain_segment_bounds)
    if args_cli.latent_terrain_label_source == "position" and segment_names is None:
        segment_names = _active_terrain_groups_from_generator(terrain_generator)
        return segment_names, segment_bounds

    if segment_names is None and segment_bounds is None:
        segment_names, segment_bounds = _default_rough_play_segments(terrain_generator)

    if segment_names is None:
        raise RuntimeError(
            "--latent_terrain_segments is required when --latent_terrain_label_source is 'step' or 'position'."
        )
    if segment_bounds is None:
        raise RuntimeError(
            "--latent_terrain_segment_bounds is required when --latent_terrain_label_source is 'step' or 'position'."
        )
    if len(segment_bounds) != len(segment_names) - 1:
        raise RuntimeError(
            "--latent_terrain_segment_bounds length must be one less than --latent_terrain_segments length: "
            f"got bounds={len(segment_bounds)} segments={len(segment_names)}"
        )
    return segment_names, segment_bounds


def _nearest_terrain_tile_ids(base_env, root_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    terrain_origins = getattr(base_env.scene.terrain, "terrain_origins", None)
    if terrain_origins is None:
        return None

    origins = torch.as_tensor(terrain_origins, device=root_pos_w.device, dtype=root_pos_w.dtype)
    if origins.ndim != 3 or origins.shape[-1] < 2:
        return None

    origins_xy = origins[..., :2].reshape(-1, 2)
    root_xy = root_pos_w[:, :2]
    distances = torch.sum((root_xy[:, None, :] - origins_xy[None, :, :]) ** 2, dim=-1)
    nearest_flat_ids = torch.argmin(distances, dim=1)
    num_cols = origins.shape[1]
    row_ids = nearest_flat_ids // num_cols
    column_ids = nearest_flat_ids % num_cols
    return nearest_flat_ids, row_ids, column_ids


def _current_terrain_type_ids(base_env, root_pos_w: torch.Tensor, fallback_terrain_type_ids: torch.Tensor) -> torch.Tensor:
    tile_ids = _nearest_terrain_tile_ids(base_env, root_pos_w)
    if tile_ids is None:
        return fallback_terrain_type_ids

    flat_ids, _row_ids, column_ids = tile_ids
    terrain_origins = getattr(base_env.scene.terrain, "terrain_origins", None)
    num_tiles = None
    if terrain_origins is not None:
        origins = torch.as_tensor(terrain_origins, device=root_pos_w.device)
        if origins.ndim == 3:
            num_tiles = origins.shape[0] * origins.shape[1]

    terrain_types = getattr(base_env.scene.terrain, "terrain_types", None)
    if terrain_types is not None:
        terrain_types_tensor = torch.as_tensor(terrain_types, device=root_pos_w.device)
        if terrain_types_tensor.ndim >= 2 or (
            num_tiles is not None and terrain_types_tensor.numel() == num_tiles
        ):
            flat_terrain_types = terrain_types_tensor.reshape(-1)
            if flat_terrain_types.numel() > int(flat_ids.max().item()):
                return flat_terrain_types[flat_ids].to(dtype=fallback_terrain_type_ids.dtype)

    terrain_generator = getattr(base_env.cfg.scene.terrain, "terrain_generator", None)
    num_terrain_classes = len(getattr(terrain_generator, "sub_terrains", {}) or {})
    if num_terrain_classes > 0 and int(column_ids.max().item()) < num_terrain_classes:
        return column_ids.to(dtype=fallback_terrain_type_ids.dtype)
    return fallback_terrain_type_ids


def _latent_segment_values(
    base_env, step_count: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        root_pos_w = base_env.scene["robot"].data.root_pos_w
    except Exception:
        root_pos_w = None

    if args_cli.latent_terrain_label_source == "tile":
        if root_pos_w is None:
            return root_pos_w, None, None
        tile_ids = _nearest_terrain_tile_ids(base_env, root_pos_w)
        if tile_ids is None:
            return root_pos_w, None, None
        _flat_ids, _row_ids, column_ids = tile_ids
        return root_pos_w, column_ids.to(dtype=torch.float32), column_ids

    if args_cli.latent_terrain_label_source == "step":
        num_envs = base_env.num_envs
        values = torch.full((num_envs,), float(step_count), device=base_env.device)
        return root_pos_w, values, None

    if root_pos_w is None:
        raise RuntimeError("Robot root_pos_w is required for --latent_terrain_label_source=position.")

    tile_ids = _nearest_terrain_tile_ids(base_env, root_pos_w)
    if tile_ids is not None:
        _flat_ids, _row_ids, column_ids = tile_ids
        return root_pos_w, column_ids.to(dtype=torch.float32), column_ids

    axis_index = {"x": 0, "y": 1, "z": 2}[args_cli.latent_terrain_position_axis]
    return root_pos_w, root_pos_w[:, axis_index], None


def _nan_metric(base_env, value: float = float("nan")) -> torch.Tensor:
    return torch.full((base_env.num_envs,), value, device=base_env.device)


def _foot_tip_sensor_cfg(base_env) -> SceneEntityCfg:
    cache_key = id(base_env.scene)
    cached = getattr(_foot_tip_sensor_cfg, "_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*TIP")
    sensor_cfg.resolve(base_env.scene)
    _foot_tip_sensor_cfg._cache = (cache_key, sensor_cfg)
    return sensor_cfg


def _collect_latent_metrics(base_env) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    robot = base_env.scene["robot"]
    data = robot.data

    root_pos_w = getattr(data, "root_pos_w", None)
    if root_pos_w is not None:
        metrics["base_height"] = root_pos_w[:, 2]

    root_lin_vel_b = getattr(data, "root_lin_vel_b", None)
    if root_lin_vel_b is not None:
        metrics["base_lin_vel_b_x"] = root_lin_vel_b[:, 0]
        metrics["base_lin_vel_b_y"] = root_lin_vel_b[:, 1]
        metrics["base_lin_vel_b_z"] = root_lin_vel_b[:, 2]
        metrics["base_speed_xy"] = torch.linalg.norm(root_lin_vel_b[:, :2], dim=1)

    root_ang_vel_b = getattr(data, "root_ang_vel_b", None)
    if root_ang_vel_b is not None:
        metrics["base_ang_vel_b_x"] = root_ang_vel_b[:, 0]
        metrics["base_ang_vel_b_y"] = root_ang_vel_b[:, 1]
        metrics["base_ang_vel_b_z"] = root_ang_vel_b[:, 2]

    projected_gravity_b = getattr(data, "projected_gravity_b", None)
    if projected_gravity_b is not None:
        metrics["projected_gravity_b_x"] = projected_gravity_b[:, 0]
        metrics["projected_gravity_b_y"] = projected_gravity_b[:, 1]
        metrics["projected_gravity_b_z"] = projected_gravity_b[:, 2]
        metrics["gravity_xy_norm"] = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)

    scene_sensors = getattr(base_env.scene, "sensors", {})
    contact_sensor = scene_sensors.get("contact_forces", None) if hasattr(scene_sensors, "get") else None
    if contact_sensor is not None:
        try:
            sensor_cfg = _foot_tip_sensor_cfg(base_env)
            foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
            foot_force_norms = torch.linalg.norm(foot_forces, dim=-1)
            foot_contacts = foot_force_norms > 1.0
            metrics["foot_contact_count"] = torch.sum(foot_contacts.float(), dim=1)
            metrics["foot_force_sum"] = torch.sum(foot_force_norms, dim=1)
            metrics["foot_force_max"] = torch.max(foot_force_norms, dim=1).values
        except Exception:
            metrics["foot_contact_count"] = _nan_metric(base_env)
            metrics["foot_force_sum"] = _nan_metric(base_env)
            metrics["foot_force_max"] = _nan_metric(base_env)

    return metrics


def main():
    os.environ["PONGBOT_PLAY_IGNORE_REWARD_NAN"] = "1"
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if args_cli.control_mode != "policy" and args_cli.num_envs is None:
        env_cfg.scene.num_envs = 1
    if env_cfg.scene.num_envs == 1:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (25.0, 25.0, 18.0)

    mass_play_enabled = _supports_mass_play_features(env_cfg)
    if args_cli.mass_play_payload_kg is not None and not mass_play_enabled:
        print("[INFO] --mass_play_payload_kg ignored because this task does not expose mass-play payload events.")
    _configure_mass_play_payload(env_cfg)

    # specify directory for logging experiments
    if args_cli.checkpoint is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _remove_robot_embedded_ground_planes()

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap around environment for rsl-rl. PaperBarrier's action contract is
    # raw action clip [-2, 2] (the JointPositionActionCfg applies the 0.25
    # scale afterwards); legacy tasks retain the historical [-1, 1] wrapper.
    play_runner_type = getattr(agent_cfg, "runner_type", "OnPolicyRunner")
    wrapper_clip_actions = 2.0 if play_runner_type == "PaperBarrierRunner" else 1.0
    env = RslRlVecEnvWrapper(env, clip_actions=wrapper_clip_actions)
    print(f"[INFO] Play action clip: {wrapper_clip_actions:.1f} ({play_runner_type})")
    # env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner_classes = {
        "OnPolicyRunner": OnPolicyRunner,
        "PaperBarrierRunner": PaperBarrierRunner,
    }
    runner_type = play_runner_type
    if runner_type not in runner_classes:
        raise ValueError(f"Unsupported runner_type={runner_type!r}; available={tuple(runner_classes)}")
    runner_cfg = agent_cfg.to_dict()
    if runner_type == "PaperBarrierRunner":
        # A one-environment visual play run cannot satisfy the training-only
        # full-range terrain/role distribution check.
        runner_cfg["validate_training_contract"] = False
    ppo_runner = runner_classes[runner_type](env, runner_cfg, log_dir=None, device=agent_cfg.device)
    _load_play_checkpoint_forgiving(ppo_runner, resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)
    uses_context_estimator = getattr(ppo_runner, "uses_context_estimator", False)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "nominal_actor",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        if hasattr(ppo_runner.alg, "adaptive_actor_critic"):
            export_mlp_as_onnx(
                ppo_runner.alg.adaptive_actor_critic.actor,
                export_model_dir,
                "adaptive_actor",
                ppo_runner.alg.adaptive_actor_critic.num_actor_obs,
            )
        if hasattr(ppo_runner.alg, "load_encoder"):
            export_mlp_as_onnx(
                ppo_runner.alg.load_encoder.net,
                export_model_dir,
                "load_encoder",
                ppo_runner.alg.load_encoder.input_dim,
            )
        export_encoder_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.get_encoder_export_shape(),
        )
    controller = _create_manual_controller(args_cli.control_mode, env_cfg)
    if controller is not None:
        print(f"[INFO] Manual control enabled: {args_cli.control_mode}")
    fixed_command = None
    if args_cli.fixed_command is not None:
        fixed_command = torch.tensor([args_cli.fixed_command], device=env.unwrapped.device, dtype=torch.float32)
        print(f"[INFO] Fixed command enabled for all envs: {args_cli.fixed_command}")
    if args_cli.gait_mode is not None:
        print(f"[INFO] Fixed gait mode enabled for all envs: {args_cli.gait_mode} -> {GAIT_MODE_COMMANDS[args_cli.gait_mode]}")

    policy_stream_publisher = None
    if args_cli.publish_policy_stream_ros:
        policy_stream_publisher = PolicyStreamRosPublisher()
        print(
            "[INFO] Publishing ROS2 topics 'obs_Isaaclab' and 'action_isaaclab' "
            "for env 0 as Float32MultiArray."
        )

    policy_stream_recorder = None
    if args_cli.record_policy_stream_csv:
        csv_dir = args_cli.policy_stream_csv_dir
        if csv_dir is None:
            csv_dir = os.path.join(log_dir, "policy_stream_csv")
        policy_stream_recorder = PolicyStreamCsvRecorder(csv_dir)
        print(f"[INFO] Recording policy streams to CSV until exit under: {csv_dir}")

    latent_recorder = None
    if args_cli.record_latent_csv:
        if not uses_context_estimator:
            raise RuntimeError("--record_latent_csv is only supported for ContextEstimatorNet implicit policies.")
        if args_cli.latent_record_interval <= 0:
            raise RuntimeError("--latent_record_interval must be positive.")

        latent_dir = args_cli.latent_csv_dir
        if latent_dir is None:
            latent_dir = os.path.join(log_dir, "latent_csv")

        terrain_generator = env.unwrapped.cfg.scene.terrain.terrain_generator
        terrain_names = list(terrain_generator.sub_terrains.keys()) if terrain_generator is not None else []
        segment_names, segment_bounds = _prepare_latent_segment_mapping(terrain_generator)
        latent_recorder = LatentCsvRecorder(latent_dir, terrain_names, segment_names, segment_bounds)
        print(f"[INFO] Recording latent CSV to: {latent_recorder.path}")
        if segment_names is not None:
            print(
                "[INFO] Latent terrain segment mapping:",
                f"source={args_cli.latent_terrain_label_source}",
                f"axis={args_cli.latent_terrain_position_axis}",
                f"segments={segment_names}",
                f"bounds={segment_bounds}",
            )

    load_latent_recorder = None
    if args_cli.record_load_latent_csv:
        if not getattr(ppo_runner, "uses_phase2_load_adaptive", False):
            raise RuntimeError("--record_load_latent_csv is only supported for Phase2_LoadAdaptive_PPO tasks.")
        if args_cli.load_latent_record_interval <= 0:
            raise RuntimeError("--load_latent_record_interval must be positive.")

        load_latent_dir = args_cli.load_latent_csv_dir
        if load_latent_dir is None:
            load_latent_dir = os.path.join(log_dir, "load_latent_csv")

        terrain_generator = env.unwrapped.cfg.scene.terrain.terrain_generator
        terrain_names = list(terrain_generator.sub_terrains.keys()) if terrain_generator is not None else []
        segment_names, segment_bounds = _prepare_latent_segment_mapping(terrain_generator)
        load_latent_recorder = LoadLatentCsvRecorder(load_latent_dir, terrain_names, segment_names, segment_bounds)
        print(f"[INFO] Recording load latent CSV to: {load_latent_recorder.path}")
        if segment_names is not None:
            print(
                "[INFO] Load latent terrain segment mapping:",
                f"source={args_cli.latent_terrain_label_source}",
                f"axis={args_cli.latent_terrain_position_axis}",
                f"segments={segment_names}",
                f"bounds={segment_bounds}",
            )

    viewport_capture_hotkey = ViewportCaptureHotkey(os.path.join(log_dir, "viewport_captures"))
    # reset environment
    try:
        obs, obs_dict = env.get_observations()
        obs_history = obs_dict["observations"].get("obsHistory")
        if obs_history is None:
            raise RuntimeError("obsHistory not found in observations")
        history_offsets = tuple(getattr(ppo_runner, "obs_history_offsets", ()))
        if obs_history.dim() >= 3 and not history_offsets and obs_history.shape[1] != int(ppo_runner.obs_history_len):
            raise RuntimeError(
                f"obsHistory length mismatch: env={obs_history.shape[1]} cfg={int(ppo_runner.obs_history_len)}"
            )
        obs_history = _prepare_play_obs_history(
            obs_history,
            uses_context_estimator,
            history_offsets,
            int(ppo_runner.obs_history_len),
        )
        commands = obs_dict["observations"].get("commands")
        critic_obs = obs_dict["observations"].get("critic")
        adaptive_obs = obs_dict["observations"].get("adaptive") if getattr(ppo_runner, "uses_phase2_load_adaptive", False) else None
        adaptive_history = (
            obs_dict["observations"].get("adaptiveHistory")
            if getattr(ppo_runner, "uses_phase2_load_adaptive", False)
            else None
        )
        last_payload_mass = _print_payload_mass_if_changed(env.unwrapped, 0, None)
        payload_buttons_were_active = False
        if fixed_command is not None:
            commands = _apply_manual_command(env, fixed_command)
        elif controller is not None:
            commands = _apply_manual_command(env, controller.get_commands())
        if args_cli.gait_mode is not None:
            _apply_gait_mode(env, args_cli.gait_mode)
        # simulate environment
        imu_log_period = 50
        step_count = 0
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if fixed_command is not None:
                    commands = _apply_manual_command(env, fixed_command)
                elif controller is not None:
                    commands = _apply_manual_command(env, controller.get_commands())
                    payload_buttons_were_active, button_payload_mass = _handle_joystick_payload_buttons(
                        env.unwrapped, controller, step_count, payload_buttons_were_active
                    )
                    if button_payload_mass is not None:
                        last_payload_mass = button_payload_mass
                if args_cli.gait_mode is not None:
                    _apply_gait_mode(env, args_cli.gait_mode)
                # agent stepping
                est = encoder(obs_history)
                if (
                    latent_recorder is not None
                    and step_count >= args_cli.latent_record_warmup
                    and step_count % args_cli.latent_record_interval == 0
                ):
                    _v_t, _z_sample, latent_mu, latent_logvar, _o_next_recon = ppo_runner.alg.encoder.forward_train(
                        obs_history
                    )
                    latent_z = est[:, 3:]
                    terrain = env.unwrapped.scene.terrain
                    terrain_type_ids = getattr(terrain, "terrain_types", None)
                    if terrain_type_ids is None:
                        raise RuntimeError("scene.terrain.terrain_types not found; cannot label latent samples.")
                    terrain_levels = getattr(terrain, "terrain_levels", None)
                    root_pos_w, segment_values, segment_ids = _latent_segment_values(env.unwrapped, step_count)
                    if root_pos_w is not None:
                        terrain_type_ids = _current_terrain_type_ids(env.unwrapped, root_pos_w, terrain_type_ids)
                    latent_metrics = _collect_latent_metrics(env.unwrapped)
                    latent_recorder.record(
                        step_count,
                        latent_mu,
                        latent_z,
                        latent_logvar,
                        commands,
                        terrain_type_ids,
                        terrain_levels,
                        root_pos_w,
                        segment_values,
                        segment_ids,
                        latent_metrics,
                    )
                imu_gt = None
                if est.shape[1] >= 9 and critic_obs is not None and critic_obs.shape[1] >= 9:
                    imu_gt = critic_obs[:, :9]
                    if step_count % imu_log_period == 0:
                        base_height = env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item()
                        imu_err = est[0, :9] - imu_gt[0]
                        # print(
                        #     "[PLAY_DEBUG]",
                        #     f"step={step_count}",
                        #     f"base_height={base_height:.4f}",
                        #     f"lin_vel_l2={torch.norm(imu_err[0:3]).item():.4f}",
                        #     f"ang_vel_l2={torch.norm(imu_err[3:6]).item():.4f}",
                        #     f"proj_gravity_l2={torch.norm(imu_err[6:9]).item():.4f}",
                        #     f"imu_l2={torch.norm(imu_err).item():.4f}",
                        #     flush=True,
                        # )
                if mass_play_enabled and args_cli.mass_status_print_interval > 0:
                    if step_count % args_cli.mass_status_print_interval == 0:
                        actual_mass, estimated_mass = _get_mass_task_status(env.unwrapped, ppo_runner.alg.encoder)
                        if actual_mass is not None or estimated_mass is not None:
                            actual_str = f"{actual_mass:.2f}" if actual_mass is not None else "n/a"
                            estimated_str = f"{estimated_mass:.2f}" if estimated_mass is not None else "n/a"
                            print(
                                "[MASS_PLAY]",
                                f"step={step_count}",
                                f"payload_mass_env0={actual_str} kg",
                                f"estimator_mass_env0={estimated_str} kg",
                                flush=True,
                            )
                if args_cli.print_torque and args_cli.torque_print_interval > 0:
                    if step_count % args_cli.torque_print_interval == 0:
                        _print_joint_torques_env0(env.unwrapped, step_count)
                actor_obs = obs
                if (
                    load_latent_recorder is not None
                    and step_count >= args_cli.load_latent_record_warmup
                    and step_count % args_cli.load_latent_record_interval == 0
                ):
                    if adaptive_obs is None or adaptive_history is None:
                        raise RuntimeError("load-adaptive CSV recording requires adaptive and adaptiveHistory observations")
                    z_load = ppo_runner.alg.load_encoder(adaptive_history)
                    nominal_policy_input = ppo_runner.build_inference_actor_obs(actor_obs, commands, obs_history, est)
                    nominal_action = ppo_runner.alg.actor_critic.act_inference(nominal_policy_input)
                    adaptive_policy_input = torch.cat((adaptive_obs, z_load), dim=-1)
                    delta_action = ppo_runner.alg.adaptive_actor_critic.act_inference(adaptive_policy_input)
                    final_action = nominal_action + delta_action
                    terrain = env.unwrapped.scene.terrain
                    terrain_type_ids = getattr(terrain, "terrain_types", None)
                    if terrain_type_ids is None:
                        raise RuntimeError("scene.terrain.terrain_types not found; cannot label load latent samples.")
                    terrain_levels = getattr(terrain, "terrain_levels", None)
                    root_pos_w, segment_values, segment_ids = _latent_segment_values(env.unwrapped, step_count)
                    if root_pos_w is not None:
                        terrain_type_ids = _current_terrain_type_ids(env.unwrapped, root_pos_w, terrain_type_ids)
                    load_latent_metrics = _collect_latent_metrics(env.unwrapped)
                    load_latent_recorder.record(
                        step_count,
                        z_load,
                        commands,
                        nominal_action,
                        delta_action,
                        final_action,
                        env.unwrapped,
                        terrain_type_ids,
                        terrain_levels,
                        root_pos_w,
                        segment_values,
                        segment_ids,
                        load_latent_metrics,
                    )
                if getattr(ppo_runner, "uses_phase2_load_adaptive", False):
                    if adaptive_obs is None or adaptive_history is None:
                        raise RuntimeError("load-adaptive play requires adaptive and adaptiveHistory observations")
                    actions = ppo_runner.alg.act_inference(
                        actor_obs,
                        obs_history,
                        commands,
                        adaptive_obs,
                        adaptive_history,
                    )
                else:
                    policy_input = ppo_runner.build_inference_actor_obs(actor_obs, commands, obs_history, est)
                    actions = policy(policy_input.detach())
                # actions = policy(torch.cat((actor_obs, commands, obs_history), dim=-1).detach())
                # actions = policy(torch.cat((est, actor_obs, commands), dim=-1).detach())
                if policy_stream_publisher is not None:
                    policy_stream_publisher.publish(est, obs, commands, actions)
                if policy_stream_recorder is not None:
                    policy_stream_recorder.record(est, obs, commands, actions, imu_gt)
                # env stepping
                obs, _, dones, infos = env.step(actions)
                viewport_capture_hotkey.update()
                commands = infos["observations"].get("commands")
                critic_obs = infos["observations"].get("critic")
                adaptive_obs = (
                    infos["observations"].get("adaptive")
                    if getattr(ppo_runner, "uses_phase2_load_adaptive", False)
                    else None
                )
                adaptive_history = (
                    infos["observations"].get("adaptiveHistory")
                    if getattr(ppo_runner, "uses_phase2_load_adaptive", False)
                    else None
                )
                obs_history = infos["observations"].get("obsHistory")
                if obs_history is None:
                    raise RuntimeError("obsHistory not found in step observations")
                obs_history = _prepare_play_obs_history(
                    obs_history,
                    uses_context_estimator,
                    history_offsets,
                    int(ppo_runner.obs_history_len),
                )
                last_payload_mass = _print_payload_mass_if_changed(
                    env.unwrapped, step_count + 1, last_payload_mass
                )
                if fixed_command is not None:
                    commands = _apply_manual_command(env, fixed_command)
                elif controller is not None:
                    commands = _apply_manual_command(env, controller.get_commands())
                if args_cli.gait_mode is not None:
                    _apply_gait_mode(env, args_cli.gait_mode)
                step_count += 1
    finally:
        if controller is not None:
            controller.stop()
        if policy_stream_publisher is not None:
            policy_stream_publisher.close()
        if policy_stream_recorder is not None:
            policy_stream_recorder.close()
        if latent_recorder is not None:
            latent_recorder.close()
        if load_latent_recorder is not None:
            load_latent_recorder.close()
        viewport_capture_hotkey.close()
        # close the simulator
        env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
