"""Export trained workspace RSL-RL checkpoints to ONNX files.

@version 0.0.1
@update 2026-07-13: Select the isolated paper-barrier runner from task configuration.
"""

import argparse
import importlib
import os
import sys

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


from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export an RSL-RL checkpoint to ONNX.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_register_local_extensions()

import gymnasium as gym

from rsl_rl.runner import OnPolicyRunner, PaperBarrierRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from pongbot_r2.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_encoder_as_onnx, export_mlp_as_onnx


def main():
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # Export can run on CPU while a separate training process occupies the
    # GPU.  The task device and the runner device must agree; otherwise the
    # runner would still allocate CUDA tensors from its training config.
    agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for ONNX export.")
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    export_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner_classes = {
        "OnPolicyRunner": OnPolicyRunner,
        "PaperBarrierRunner": PaperBarrierRunner,
    }
    runner_type = getattr(agent_cfg, "runner_type", "OnPolicyRunner")
    if runner_type not in runner_classes:
        raise ValueError(f"Unsupported runner_type={runner_type!r}; available={tuple(runner_classes)}")
    runner_cfg = agent_cfg.to_dict()
    # The strict PaperBarrier training contract requires a full multi-level
    # terrain calibration.  It is unnecessary for export and would force a
    # large environment count, defeating CPU-only checkpoint conversion.
    if runner_type == "PaperBarrierRunner":
        runner_cfg["validate_training_contract"] = False
    runner = runner_classes[runner_type](env, runner_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    export_mlp_as_onnx(
        runner.alg.actor_critic.actor,
        export_dir,
        "nominal_actor",
        runner.alg.actor_critic.num_actor_obs,
    )
    if hasattr(runner.alg, "adaptive_actor_critic"):
        export_mlp_as_onnx(
            runner.alg.adaptive_actor_critic.actor,
            export_dir,
            "adaptive_actor",
            runner.alg.adaptive_actor_critic.num_actor_obs,
        )
    if hasattr(runner.alg, "load_encoder"):
        export_mlp_as_onnx(
            runner.alg.load_encoder.net,
            export_dir,
            "load_encoder",
            runner.alg.load_encoder.input_dim,
        )
    export_encoder_as_onnx(
        runner.alg.encoder,
        export_dir,
        "encoder",
        runner.get_encoder_export_shape(),
    )

    print(f"Exported ONNX models to: {export_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
