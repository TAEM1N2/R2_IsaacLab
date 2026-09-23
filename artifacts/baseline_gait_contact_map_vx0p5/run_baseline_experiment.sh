#!/usr/bin/env bash
set -eo pipefail

export GAIT_ARTIFACT_DIR="/home/rclab/r2_isaaclab_trone/artifacts/baseline_gait_contact_map_vx0p5"
export GAIT_MODEL_DIR="/home/rclab/pbr2_ws/src/rclab_mujoco_sim/src/nn/implicit/real"
export GAIT_EXPERIMENT_LABEL="baseline — four gait presets"
export GAIT_OUTPUT_STEM="baseline"
export GAIT_SWITCH_3_COUNT="893"

exec /home/rclab/r2_isaaclab_trone/artifacts/tiptoe_gait_contact_map_vx0p5/run_experiment.sh
