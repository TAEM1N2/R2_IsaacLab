#!/usr/bin/env bash
set -eo pipefail

ARTIFACT_DIR="/home/rclab/r2_isaaclab_trone/artifacts/baseline_trained_gait_vx0p5"
MODEL_DIR="/home/rclab/pbr2_ws/src/rclab_mujoco_sim/src/nn/implicit/real"
SIM_BINARY="/tmp/baseline_trained_gait_ws/install/rclab_mujoco_sim/lib/rclab_mujoco_sim/rclab_mujoco_sim"
SNAPSHOT_DIR="${ARTIFACT_DIR}/snapshots"
CSV_PATH="${ARTIFACT_DIR}/baseline_trained_gait_raw.csv"

mkdir -p "${SNAPSHOT_DIR}"

source /opt/ros/humble/setup.bash
source /home/rclab/pbr2_ws/install/setup.bash
set -u

/usr/bin/python3 "${ARTIFACT_DIR}/snapshot_controller.py" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    >"${ARTIFACT_DIR}/snapshot_controller.log" 2>&1 &
CONTROLLER_PID=$!

cleanup_controller() {
    if kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
        kill "${CONTROLLER_PID}" 2>/dev/null || true
        wait "${CONTROLLER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_controller EXIT

"${SIM_BINARY}" --ros-args \
    -p task_mode:=implicit \
    -p encoder_model_path:="${MODEL_DIR}/encoder.onnx" \
    -p policy_model_path:="${MODEL_DIR}/policy.onnx" \
    -p foot_metrics_enabled:=true \
    -p foot_metrics_auto_run:=true \
    -p foot_metrics_flat_terrain:=true \
    -p foot_metrics_experiment_label:="baseline — trained fixed gait" \
    -p foot_metrics_output_path:="${CSV_PATH}" \
    -p foot_metrics_contact_threshold_n:=1.0 \
    -p foot_metrics_command_x_mps:=0.5 \
    -p foot_metrics_command_y_mps:=0.0 \
    -p foot_metrics_command_yaw_radps:=0.0 \
    -p foot_metrics_gait_mode:=0 \
    -p foot_metrics_duration_s:=3.0 \
    >"${ARTIFACT_DIR}/mujoco.log" 2>&1

wait "${CONTROLLER_PID}"
trap - EXIT

printf 'Raw CSV: %s\n' "${CSV_PATH}"
printf 'Snapshots: %s\n' "${SNAPSHOT_DIR}"
