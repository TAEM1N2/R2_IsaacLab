#!/usr/bin/env bash
set -eo pipefail

TOOL_DIR="/home/rclab/r2_isaaclab_trone/artifacts/tiptoe_gait_contact_map_vx0p5"
ARTIFACT_DIR="${GAIT_ARTIFACT_DIR:-${TOOL_DIR}}"
MODEL_DIR="${GAIT_MODEL_DIR:-/home/rclab/pbr2_ws/src/rclab_mujoco_sim/src/nn/implicit/사뿐사뿐_now}"
EXPERIMENT_LABEL="${GAIT_EXPERIMENT_LABEL:-tip toe — four gait presets}"
OUTPUT_STEM="${GAIT_OUTPUT_STEM:-tiptoe}"
SWITCH_1_COUNT="${GAIT_SWITCH_1_COUNT:-297}"
SWITCH_2_COUNT="${GAIT_SWITCH_2_COUNT:-598}"
SWITCH_3_COUNT="${GAIT_SWITCH_3_COUNT:-898}"
SNAPSHOT_DIR="${ARTIFACT_DIR}/snapshots"
CSV_PATH="${ARTIFACT_DIR}/${OUTPUT_STEM}_four_gaits_raw.csv"

mkdir -p "${SNAPSHOT_DIR}"

source /opt/ros/humble/setup.bash
source /home/rclab/pbr2_ws/install/setup.bash
set -u

/usr/bin/python3 "${TOOL_DIR}/gait_switch_controller.py" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --switch-counts "${SWITCH_1_COUNT}" "${SWITCH_2_COUNT}" "${SWITCH_3_COUNT}" \
    >"${ARTIFACT_DIR}/gait_switch_controller.log" 2>&1 &
CONTROLLER_PID=$!

cleanup_controller() {
    if kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
        kill "${CONTROLLER_PID}" 2>/dev/null || true
        wait "${CONTROLLER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_controller EXIT

ros2 run rclab_mujoco_sim rclab_mujoco_sim --ros-args \
    -p task_mode:=implicit \
    -p encoder_model_path:="${MODEL_DIR}/encoder.onnx" \
    -p policy_model_path:="${MODEL_DIR}/policy.onnx" \
    -p foot_metrics_enabled:=true \
    -p foot_metrics_auto_run:=true \
    -p foot_metrics_flat_terrain:=true \
    -p foot_metrics_experiment_label:="${EXPERIMENT_LABEL}" \
    -p foot_metrics_output_path:="${CSV_PATH}" \
    -p foot_metrics_contact_threshold_n:=1.0 \
    -p foot_metrics_command_x_mps:=0.5 \
    -p foot_metrics_command_y_mps:=0.0 \
    -p foot_metrics_command_yaw_radps:=0.0 \
    -p foot_metrics_gait_mode:=0 \
    -p foot_metrics_duration_s:=12.0 \
    >"${ARTIFACT_DIR}/mujoco.log" 2>&1

wait "${CONTROLLER_PID}"
trap - EXIT

printf 'Raw CSV: %s\n' "${CSV_PATH}"
printf 'Snapshots: %s\n' "${SNAPSHOT_DIR}"
