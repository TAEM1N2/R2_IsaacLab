"""Script to train RL agents with the workspace RSL-RL runners.

Periodic image capture uses one viewport camera and can alternate between a
representative robot close-up and a robust overview of active environments.

@version 0.0.2
@update 2026-07-13: Alternate one offscreen camera between close and active-environment overview captures.
@update 2026-07-13: Select the isolated paper-barrier runner from task configuration.
"""

"""Launch Isaac Sim Simulator first."""
import importlib
import os
import sys


import argparse


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
parser.add_argument("--video_length", type=int, default=250, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=3000, help="Interval between video recordings (in steps).")
parser.add_argument("--camera", action="store_true", default=False, help="Save image captures during training.")
parser.add_argument("--camera_interval", type=int, default=3000, help="Interval between image captures (in steps).")
parser.add_argument(
    "--camera_view_mode",
    choices=("alternate", "fixed"),
    default="alternate",
    help="Alternate close/overview views, or keep the task's fixed viewport camera.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations to train.")
parser.add_argument("--save_interval", type=int, default=None, help="The number of iterations between saves")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video or image captures
if args_cli.video or args_cli.camera:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_remove_default_ground_planes()

# Local extensions may import Isaac Sim / Omniverse modules, so register them
# only after the simulator app has been launched.
_register_local_extensions()

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import struct
import torch
import zlib
from datetime import datetime

# from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runner import OnPolicyRunner, PaperBarrierRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from pongbot_r2.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class PeriodicImageCaptureWrapper(gym.Wrapper):
    """Save one frame per interval, optionally alternating close and overview views."""

    def __init__(self, env, image_folder: str, step_interval: int, view_mode: str = "alternate"):
        super().__init__(env)
        if step_interval <= 0:
            raise ValueError(f"--camera_interval must be positive, got {step_interval}")
        if view_mode not in ("alternate", "fixed"):
            raise ValueError(f"Unsupported camera view mode: {view_mode!r}")
        self.image_folder = image_folder
        self.step_interval = step_interval
        self.view_mode = view_mode
        self.step_count = 0
        self.capture_count = 0
        self.base_env = env.unwrapped
        os.makedirs(self.image_folder, exist_ok=True)

    def step(self, action):
        result = self.env.step(action)
        if self.step_count % self.step_interval == 0:
            self._capture_frame()
        self.step_count += 1
        return result

    def _capture_frame(self) -> None:
        view_label = self._set_capture_view()
        frame = self.render()
        self.capture_count += 1
        if frame is None:
            print("[WARN] Camera capture skipped because env.render() returned None.")
            return
        frame = self._normalize_frame(frame)
        if frame is None:
            return

        suffix = f"_{view_label}" if view_label else ""
        file_path = os.path.join(self.image_folder, f"camera_step_{self.step_count:09d}{suffix}.png")
        self._write_png(file_path, frame)
        print(f"[INFO] Saved camera capture to: {file_path}")

    def _set_capture_view(self) -> str:
        """Move the single viewport camera without adding another render target."""
        if self.view_mode == "fixed":
            return ""

        try:
            robot = self.base_env.scene["robot"]
            root_positions = robot.data.root_pos_w.detach()
            if root_positions.ndim != 2 or root_positions.shape[0] == 0:
                raise RuntimeError(f"Unexpected robot root position shape: {tuple(root_positions.shape)}")

            if self.capture_count % 2 == 0:
                view_label, eye, target = self._close_view(root_positions)
            else:
                view_label, eye, target = self._overview_view(root_positions)

            viewer_cfg = getattr(self.base_env.cfg, "viewer", None)
            camera_prim_path = getattr(viewer_cfg, "cam_prim_path", "/OmniverseKit_Persp")
            self.base_env.sim.set_camera_view(
                eye=eye,
                target=target,
                camera_prim_path=camera_prim_path,
            )
            return view_label
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[WARN] Dynamic camera view disabled; using the task's fixed view: {exc}")
            self.view_mode = "fixed"
            return "fixed"

    def _close_view(
        self,
        root_positions: torch.Tensor,
    ) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        """Track one representative environment and rotate the choice across close captures."""
        num_envs = root_positions.shape[0]
        num_close_views = min(8, num_envs)
        close_index = self.capture_count // 2
        env_id = min(
            num_envs - 1,
            (close_index % num_close_views) * max(1, num_envs // num_close_views),
        )
        root = root_positions[env_id]
        if not bool(torch.isfinite(root).all()):
            raise RuntimeError(f"Robot root position is not finite for env {env_id}")

        root = root.cpu()
        target = (float(root[0]), float(root[1]), float(root[2] + 0.35))
        eye = (target[0] + 3.5, target[1] + 3.5, target[2] + 2.5)
        return f"close_env_{env_id:03d}", eye, target

    @staticmethod
    def _overview_view(
        root_positions: torch.Tensor,
    ) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        """Frame the robust bounds of currently active robot positions."""
        finite = torch.isfinite(root_positions).all(dim=1)
        valid_positions = root_positions[finite]
        if valid_positions.shape[0] == 0:
            raise RuntimeError("No finite robot root positions are available for overview")

        xy = valid_positions[:, :2]
        low = torch.quantile(xy, 0.02, dim=0)
        high = torch.quantile(xy, 0.98, dim=0)
        center = 0.5 * (low + high)
        span = torch.clamp(torch.max(high - low), min=8.0)
        target_z = torch.median(valid_positions[:, 2])

        center_x, center_y, span_value, target_z_value = (
            float(center[0]),
            float(center[1]),
            float(span),
            float(target_z),
        )
        target = (center_x, center_y, target_z_value)
        eye = (center_x - 0.20 * span_value, center_y, target_z_value + 0.95 * span_value)
        return "overview", eye, target

    @staticmethod
    def _normalize_frame(frame):
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.moveaxis(frame, 0, -1)
        if frame.ndim not in (2, 3):
            print(f"[WARN] Camera capture skipped because frame shape is unsupported: {frame.shape}")
            return None
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and np.max(frame, initial=0.0) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[:, :, 0]
        if frame.ndim == 3 and frame.shape[-1] not in (3, 4):
            print(f"[WARN] Camera capture skipped because channel count is unsupported: {frame.shape[-1]}")
            return None
        return np.ascontiguousarray(frame)

    @staticmethod
    def _write_png(file_path: str, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if frame.ndim == 2:
            color_type = 0
            row_width = width
        else:
            color_type = 6 if frame.shape[-1] == 4 else 2
            row_width = width * frame.shape[-1]

        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + chunk_type
                + data
                + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
            )

        raw_rows = bytearray()
        frame_bytes = frame.tobytes()
        for row in range(height):
            raw_rows.append(0)
            start = row * row_width
            raw_rows.extend(frame_bytes[start : start + row_width])

        png_bytes = b"\x89PNG\r\n\x1a\n"
        png_bytes += png_chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        png_bytes += png_chunk("IDAT".encode(), zlib.compress(bytes(raw_rows)))
        png_bytes += png_chunk("IEND".encode(), b"")
        with open(file_path, "wb") as file:
            file.write(png_bytes)

# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if (args_cli.video or args_cli.camera) else None)
    _remove_robot_embedded_ground_planes()
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if args_cli.camera:
        camera_kwargs = {
            "image_folder": os.path.join(log_dir, "camera"),
            "step_interval": args_cli.camera_interval,
            "view_mode": args_cli.camera_view_mode,
        }
        print("[INFO] Saving image captures during training.")
        print_dict(camera_kwargs, nesting=4)
        env = PeriodicImageCaptureWrapper(env, **camera_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=10.0)
    # env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    # on_policy_runner_class = eval(agent_cfg.runner_type)
    # runner: OnPolicyRunner | OnPolicyRunnerMlp = on_policy_runner_class(
    #     env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    # )
    runner_classes = {
        "OnPolicyRunner": OnPolicyRunner,
        "PaperBarrierRunner": PaperBarrierRunner,
    }
    runner_type = getattr(agent_cfg, "runner_type", "OnPolicyRunner")
    if runner_type not in runner_classes:
        raise ValueError(f"Unsupported runner_type={runner_type!r}; available={tuple(runner_classes)}")
    runner = runner_classes[runner_type](env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        if args_cli.checkpoint_path is not None:
            resume_path = args_cli.checkpoint_path
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
