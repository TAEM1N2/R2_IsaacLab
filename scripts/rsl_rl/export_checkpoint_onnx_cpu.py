"""Export RSL-RL checkpoint modules to ONNX without launching Isaac Sim.

This script reconstructs the policy modules from the checkpoint tensor shapes
and the saved params/agent.yaml file. It does not create a Gym environment or
start AppLauncher, so it can run on CPU while another training job owns the GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.modules import ActorCritic, ContextEstimatorNet, IMU_Encoder, MLP_Encoder  # noqa: E402
from rsl_rl.modules.adaptive_actor_critic import AdaptiveActorCritic  # noqa: E402
from rsl_rl.modules.load_adaptive import LoadEncoder  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _default_agent_cfg_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "params" / "agent.yaml"


def _linear_weight_shape(state_dict: dict[str, torch.Tensor], key: str) -> tuple[int, int]:
    value = state_dict[key]
    if value.ndim != 2:
        raise ValueError(f"Expected 2-D weight for {key}, got shape {tuple(value.shape)}")
    return int(value.shape[0]), int(value.shape[1])


def _last_linear_out_dim(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    weight_items = [
        (name, tensor)
        for name, tensor in state_dict.items()
        if name.startswith(prefix) and name.endswith(".weight") and tensor.ndim == 2
    ]
    if not weight_items:
        raise ValueError(f"No Linear weights found with prefix '{prefix}'")
    name, tensor = weight_items[-1]
    print(f"[INFO] inferred output dim from {name}: {tuple(tensor.shape)}")
    return int(tensor.shape[0])


def _clean_module_cfg(cfg: dict[str, Any], ignored_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    clean = dict(cfg or {})
    clean.pop("class_name", None)
    for key in ignored_keys:
        clean.pop(key, None)
    return clean


def _build_actor_critic(agent_cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> ActorCritic:
    num_actions = int(state_dict["logstd"].shape[0])
    _, num_actor_obs = _linear_weight_shape(state_dict, "actor.0.weight")
    _, num_critic_obs = _linear_weight_shape(state_dict, "critic.0.weight")
    policy_cfg = _clean_module_cfg(agent_cfg.get("policy", {}), ignored_keys=("noise_std_type",))
    model = ActorCritic(num_actor_obs, num_critic_obs, num_actions, **policy_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_encoder(agent_cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    encoder_cfg = dict(agent_cfg.get("encoder", {}))
    class_name = encoder_cfg.pop("class_name", "MLP_Encoder")
    encoder_cfg.pop("target_source", None)
    encoder_cfg.pop("target_observation_group", None)

    if class_name in ("MLP_Encoder", "IMU_Encoder"):
        _, num_input_dim = _linear_weight_shape(state_dict, "encoder.0.weight")
        num_output_dim = _last_linear_out_dim(state_dict, "encoder.")
        encoder_cfg["num_input_dim"] = num_input_dim
        encoder_cfg["num_output_dim"] = num_output_dim
        encoder_class = {"MLP_Encoder": MLP_Encoder, "IMU_Encoder": IMU_Encoder}[class_name]
        model = encoder_class(**encoder_cfg)
    elif class_name == "ContextEstimatorNet":
        first_encoder = next(
            tensor
            for name, tensor in state_dict.items()
            if name.startswith("encoder.") and name.endswith(".weight") and tensor.ndim == 2
        )
        history_len = int(encoder_cfg.get("history_len", 1))
        flat_input_dim = int(first_encoder.shape[1])
        if flat_input_dim % history_len != 0:
            raise ValueError(
                f"Cannot infer ContextEstimatorNet obs_dim: input={flat_input_dim}, history_len={history_len}"
            )
        encoder_cfg["obs_dim"] = flat_input_dim // history_len
        model = ContextEstimatorNet(**encoder_cfg)
    else:
        raise ValueError(f"Unsupported encoder class for CPU export: {class_name}")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_adaptive_actor(agent_cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> AdaptiveActorCritic:
    num_actions = int(state_dict["logstd"].shape[0])
    _, num_actor_obs = _linear_weight_shape(state_dict, "actor.0.weight")
    _, num_critic_obs = _linear_weight_shape(state_dict, "critic.0.weight")
    adaptive_cfg = _clean_module_cfg(agent_cfg.get("adaptive_policy", {}))
    model = AdaptiveActorCritic(num_actor_obs, num_critic_obs, num_actions, **adaptive_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_load_encoder(agent_cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> LoadEncoder:
    _, input_dim = _linear_weight_shape(state_dict, "net.0.weight")
    latent_dim = _last_linear_out_dim(state_dict, "net.")
    alg_cfg = agent_cfg.get("algorithm", {})
    model = LoadEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=alg_cfg.get("load_encoder_hidden_dims", (128, 64)),
        activation=alg_cfg.get("load_activation", "elu"),
        orthogonal_init=alg_cfg.get("load_orthogonal_init", False),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


class _EncoderExportWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, encoder_input: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(encoder_input)


def _export_onnx(model: torch.nn.Module, output_path: Path, input_shape: tuple[int, ...], input_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(output_path),
        verbose=False,
        input_names=[input_name],
        output_names=["output"],
        export_params=True,
        opset_version=13,
    )
    print(f"[OK] exported {output_path} input_shape={input_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RSL-RL checkpoint ONNX files on CPU without Isaac Sim.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model_*.pt checkpoint.")
    parser.add_argument(
        "--agent-cfg",
        type=Path,
        default=None,
        help="Path to params/agent.yaml. Defaults to <checkpoint_dir>/params/agent.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <checkpoint_dir>/exported_cpu.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    agent_cfg_path = (args.agent_cfg or _default_agent_cfg_path(checkpoint)).expanduser().resolve()
    output_dir = (args.output_dir or (checkpoint.parent / "exported_cpu")).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not agent_cfg_path.is_file():
        raise FileNotFoundError(agent_cfg_path)

    agent_cfg = _load_yaml(agent_cfg_path)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] agent_cfg={agent_cfg_path}")
    print(f"[INFO] output_dir={output_dir}")

    actor_critic = _build_actor_critic(agent_cfg, checkpoint_data["model_state_dict"])
    _export_onnx(
        actor_critic.actor,
        output_dir / "nominal_actor.onnx",
        (actor_critic.num_actor_obs,),
        "actor_input",
    )

    encoder_state = checkpoint_data.get("encoder_state_dict")
    if encoder_state:
        encoder = _build_encoder(agent_cfg, encoder_state)
        if isinstance(encoder, ContextEstimatorNet):
            input_shape = (1, encoder.history_len, encoder.obs_dim)
        else:
            input_shape = (encoder.num_input_dim,)
        _export_onnx(_EncoderExportWrapper(encoder), output_dir / "encoder.onnx", input_shape, "encoder_input")

    adaptive_state = checkpoint_data.get("adaptive_model_state_dict")
    if adaptive_state:
        adaptive_actor = _build_adaptive_actor(agent_cfg, adaptive_state)
        _export_onnx(
            adaptive_actor.actor,
            output_dir / "adaptive_actor_raw.onnx",
            (adaptive_actor.num_actor_obs,),
            "adaptive_actor_input",
        )

    load_encoder_state = checkpoint_data.get("load_encoder_state_dict")
    if load_encoder_state:
        load_encoder = _build_load_encoder(agent_cfg, load_encoder_state)
        _export_onnx(
            load_encoder.net,
            output_dir / "load_encoder.onnx",
            (load_encoder.input_dim,),
            "load_encoder_input",
        )


if __name__ == "__main__":
    main()
