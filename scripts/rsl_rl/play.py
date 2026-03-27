"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
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
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument(
    "--control_mode",
    type=str,
    default="policy",
    choices=["policy", "keyboard", "remote_keyboard", "joystick"],
    help="How to generate base velocity commands during play.",
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
    help="Record obs_IsaacLab and action_IsaacLab to timestamped CSV files for 1 second from the first policy input.",
)
parser.add_argument(
    "--policy_stream_csv_dir",
    type=str,
    default=None,
    help="Optional output directory for policy stream CSV files. Defaults to <run>/policy_stream_csv.",
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
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
except ImportError:
    rclpy = None
    Node = None
    Float32MultiArray = None

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from pongbot_r2.tasks.controllers import JoystickController, LocalKeyboardController, RemoteKeyboardController
from pongbot_r2.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


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
    """Record policy streams for env 0 to CSV files for a fixed number of policy steps."""

    def __init__(self, output_dir: str, max_steps: int = 50):
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.start_time = None
        self.step_count = 0
        self.obs_file = None
        self.action_file = None
        self.obs_writer = None
        self.action_writer = None
        self._finished = False

        os.makedirs(self.output_dir, exist_ok=True)

    def _init_writers(self, encoder_output: torch.Tensor, obs: torch.Tensor, vel_command: torch.Tensor, policy_output: torch.Tensor):
        obs_path = os.path.join(self.output_dir, "obs_IsaacLab.csv")
        action_path = os.path.join(self.output_dir, "action_IsaacLab.csv")

        self.obs_file = open(obs_path, "w", newline="", encoding="utf-8")
        self.action_file = open(action_path, "w", newline="", encoding="utf-8")
        self.obs_writer = csv.writer(self.obs_file)
        self.action_writer = csv.writer(self.action_file)

        obs_header = ["step"]
        obs_header += [f"encoder_output_{i}" for i in range(encoder_output.shape[1])]
        obs_header += [f"obs_{i}" for i in range(obs.shape[1])]
        obs_header += [f"vel_command_{i}" for i in range(vel_command.shape[1])]
        action_header = ["step"]
        action_header += [f"policy_output_{i}" for i in range(policy_output.shape[1])]

        self.obs_writer.writerow(obs_header)
        self.action_writer.writerow(action_header)

    def record(
        self, encoder_output: torch.Tensor, obs: torch.Tensor, vel_command: torch.Tensor, policy_output: torch.Tensor
    ) -> None:
        if self._finished:
            return

        if self.start_time is None:
            self.start_time = 0
            self._init_writers(encoder_output, obs, vel_command, policy_output)

        if self.step_count >= self.max_steps:
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

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self._finished = True
            self.close()

    def close(self) -> None:
        if self.obs_file is not None and not self.obs_file.closed:
            self.obs_file.close()
        if self.action_file is not None and not self.action_file.closed:
            self.action_file.close()


def main():
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

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

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
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    controller = _create_manual_controller(args_cli.control_mode, env_cfg)
    if controller is not None:
        print(f"[INFO] Manual control enabled: {args_cli.control_mode}")

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
        policy_stream_recorder = PolicyStreamCsvRecorder(csv_dir, max_steps=50)
        print(f"[INFO] Recording 50 policy steps to CSV under: {csv_dir}")
    # reset environment
    try:
        obs, obs_dict = env.get_observations()
        obs_history = obs_dict["observations"].get("obsHistory")
        obs_history = obs_history.flatten(start_dim=1)
        commands = obs_dict["observations"].get("commands")
        if controller is not None:
            commands = _apply_manual_command(env, controller.get_commands())
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller is not None:
                    commands = _apply_manual_command(env, controller.get_commands())
                # agent stepping
                est = encoder(obs_history)
                actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
                if policy_stream_publisher is not None:
                    policy_stream_publisher.publish(est, obs, commands, actions)
                if policy_stream_recorder is not None:
                    policy_stream_recorder.record(est, obs, commands, actions)
                # env stepping
                obs, _, _, infos = env.step(actions)
                obs_history = infos["observations"].get("obsHistory")
                obs_history = obs_history.flatten(start_dim=1)
                commands = infos["observations"].get("commands")
                if controller is not None:
                    commands = _apply_manual_command(env, controller.get_commands())
    finally:
        if controller is not None:
            controller.stop()
        if policy_stream_publisher is not None:
            policy_stream_publisher.close()
        if policy_stream_recorder is not None:
            policy_stream_recorder.close()
        # close the simulator
        env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
