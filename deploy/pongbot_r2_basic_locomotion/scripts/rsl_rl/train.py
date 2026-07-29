"""Train the basic PongBot R2 Flat/Rough locomotion tasks.

@version 0.0.2
@update 2026-07-29: Allow AppLauncher to build complete CLI help before enforcing --task.
@update 2026-07-29: Isolate the basic OnPolicyRunner training path for source distribution.
"""

import argparse
from datetime import datetime
import importlib
import math
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


parser = argparse.ArgumentParser(description="Train a basic PongBot R2 locomotion policy.")
parser.add_argument("--task", type=str, default=None, help="Registered Flat/Rough training task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iteration override.")
parser.add_argument("--save_interval", type=int, default=None, help="Checkpoint interval override.")
parser.add_argument("--seed", type=int, default=None, help="Environment and policy seed.")
parser.add_argument("--sigma", type=float, default=None, help="Initial policy action standard deviation.")
parser.add_argument("--video", action="store_true", help="Record periodic training videos.")
parser.add_argument("--video_length", type=int, default=250, help="Recorded video length in steps.")
parser.add_argument("--video_interval", type=int, default=3000, help="Video interval in steps.")
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
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner


def main() -> None:
    """Create the selected task and run PPO learning."""
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    env_cfg.seed = agent_cfg.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    try:
        _remove_robot_embedded_ground_planes()
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=10.0)

        runner_type = getattr(agent_cfg, "runner_type", "OnPolicyRunner")
        if runner_type != "OnPolicyRunner":
            raise ValueError(f"Unsupported runner_type={runner_type!r}")
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )

        if args_cli.checkpoint is not None:
            checkpoint_path = os.path.abspath(args_cli.checkpoint)
            print(f"[INFO] Loading checkpoint: {checkpoint_path}")
            runner.load(checkpoint_path)
        if args_cli.sigma is not None:
            if args_cli.sigma <= 0.0:
                raise ValueError(f"--sigma must be positive, got {args_cli.sigma}")
            with torch.no_grad():
                runner.alg.actor_critic.logstd.fill_(math.log(args_cli.sigma))

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
