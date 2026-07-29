"""Evaluate a basic PongBot R2 Flat/Rough locomotion checkpoint.

@version 0.0.2
@update 2026-07-29: Allow AppLauncher to build complete CLI help before enforcing --task.
@update 2026-07-29: Isolate the MLP-estimator inference path for source distribution.
"""

import argparse
import importlib
import os
import sys

from isaaclab.app import AppLauncher

import cli_args


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTS_ROOT = os.path.join(REPO_ROOT, "exts")


def _register_local_extensions() -> None:
    """Add the bundled packages to Python and register their Gym tasks."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "rsl_rl"))
    for entry in sorted(os.scandir(EXTS_ROOT), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        package_init = os.path.join(entry.path, entry.name, "__init__.py")
        if not os.path.isfile(package_init):
            continue
        sys.path.insert(0, entry.path)
        importlib.import_module(entry.name)


def _remove_robot_embedded_ground_planes() -> None:
    """Remove GroundPlane prims embedded below cloned robot assets."""
    from isaacsim.core.utils.prims import delete_prim, find_matching_prim_paths

    for pattern in (
        "/World/envs/env_.*/Robot/GroundPlane",
        "/World/envs/env_.*/robot/GroundPlane",
    ):
        for prim_path in find_matching_prim_paths(pattern):
            delete_prim(prim_path)


parser = argparse.ArgumentParser(description="Evaluate a basic PongBot R2 policy.")
parser.add_argument("--task", type=str, default=None, help="Registered Flat/Rough play task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of evaluation environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment and policy seed.")
parser.add_argument("--video", action="store_true", help="Record one evaluation video.")
parser.add_argument("--video_length", type=int, default=500, help="Recorded video length in steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task is None:
    parser.error("--task is required")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_register_local_extensions()

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner


def main() -> None:
    """Load a checkpoint and execute its actor and velocity estimator."""
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    if args_cli.checkpoint is None:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint_path = get_checkpoint_path(
            log_root,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )
    else:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    try:
        _remove_robot_embedded_ground_planes()
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(os.path.dirname(checkpoint_path), "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=1.0)

        runner_type = getattr(agent_cfg, "runner_type", "OnPolicyRunner")
        if runner_type != "OnPolicyRunner":
            raise ValueError(f"Unsupported runner_type={runner_type!r}")
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(checkpoint_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        encoder = runner.get_inference_encoder(device=env.unwrapped.device)
        obs, extras = env.get_observations()
        obs_history = runner._prepare_obs_history(extras["observations"]["obsHistory"])
        commands = extras["observations"]["commands"]

        step = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                estimate = encoder(obs_history)
                actor_obs = runner.build_inference_actor_obs(
                    obs,
                    commands,
                    obs_history,
                    estimate,
                )
                actions = policy(actor_obs)
                obs, _, _, infos = env.step(actions)
                obs_history = runner._prepare_obs_history(
                    infos["observations"]["obsHistory"]
                )
                commands = infos["observations"]["commands"]
            step += 1
            if args_cli.video and step >= args_cli.video_length:
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
