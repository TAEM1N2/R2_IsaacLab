#!/usr/bin/env python3
"""Independent MuJoCo runner for the stateful MiniPB v10 controller bundle."""
import argparse
import contextlib
import csv
import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from controller import MiniPBV10HybridController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path.home() / 'Downloads/minipb_hybrid_v10_ep200_nuc_20260908')
    parser.add_argument('--model', type=Path, default=Path.home() / 'minipb_ws/src/rclab_minipb_sim/models/minipb_ver3_mujoco.xml')
    parser.add_argument('--command', type=float, nargs=3, default=[0.5, 0, 0], metavar=('VX', 'VY', 'YAW'))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, default=0, help='Simulation seconds including settling; 0 runs until closed')
    parser.add_argument('--settle', type=float, default=3, help='Initial default-pose PD seconds')
    parser.add_argument('--csv', type=Path, help='Optional policy-rate telemetry')
    args = parser.parse_args()
    if args.duration < 0 or args.settle < 0 or not np.isfinite([args.duration, args.settle, *args.command]).all():
        parser.error('Timing and command must be finite; timing must be nonnegative')
    controller = MiniPBV10HybridController(args.bundle)
    cfg = controller.config
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    model.opt.timestep = 1 / cfg['timing']['physics_rate_hz']
    decimation = cfg['timing']['simulation_decimation']
    if not np.isclose(model.opt.timestep * decimation, controller.dt):
        raise ValueError('Inconsistent physics/policy timing in bundle')

    def lookup(kind, name):
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError(f'Model is missing {name}')
        return index

    names = [name + '_JOINT' for name in cfg['joint_order']]
    joints = np.array([lookup(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    actuators = np.array([lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names])
    if not np.array_equal(model.actuator_trnid[actuators, 0], joints) or not np.allclose(model.actuator_gear[actuators, 0], 1):
        raise ValueError('Expected direct, unit-gear joint motors')
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    root = lookup(mujoco.mjtObj.mjOBJ_BODY, 'root')
    root_joint = lookup(mujoco.mjtObj.mjOBJ_JOINT, 'root')
    root_qadr = model.jnt_qposadr[root_joint]
    data = mujoco.MjData(model)
    pd = cfg['low_level_pd']
    command = np.array(args.command, dtype=float)
    lock = threading.Lock()
    reset_requested = threading.Event()

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = controller.q_default
        data.qpos[root_qadr + 2] = 0.34
        controller.reset()
        mujoco.mj_forward(model, data)

    def key_callback(key):
        commands = {87: [0.5, 0, 0], 83: [-0.5, 0, 0], 65: [0, 0.5, 0],
                    68: [0, -0.5, 0], 81: [0, 0, 0.5], 69: [0, 0, -0.5], 32: [0, 0, 0]}
        if key == 82:
            reset_requested.set()
        elif key in commands:
            with lock:
                command[:] = commands[key]
            print('Command:', commands[key], flush=True)

    reset()
    print(f'Bundle: {controller.bundle}\nModel: {args.model}\n500 Hz physics / 100 Hz policy; PD={pd["stiffness_Nm_per_rad"]}/{pd["damping_Nm_s_per_rad"]}')
    print('W/S forward/back, A/D lateral, Q/E yaw, Space zero command, R reset. Close window or Ctrl+C to exit.')
    print('Nominal flat MuJoCo physics; no training randomization or actuator delay.')
    with contextlib.ExitStack() as stack:
        viewer = None
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = stack.enter_context(mj_viewer.launch_passive(model, data, key_callback=key_callback))
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        writer = None
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            writer = csv.writer(stack.enter_context(args.csv.open('w', newline='')))
            writer.writerow(['time', 'height', 'vx_world', 'vy_world', 'cmd_x', 'cmd_y', 'cmd_yaw'] + ['q_' + n for n in names] + ['target_' + n for n in names])
        tick = 0
        target = controller.q_default.copy()
        wall_start = time.monotonic()
        heights, velocities = [], []
        while (not args.duration or data.time < args.duration) and (viewer is None or viewer.is_running()):
            if reset_requested.is_set():
                reset_requested.clear()
                reset()
                tick = 0
                target = controller.q_default.copy()
                wall_start = time.monotonic()
                heights.clear()
                velocities.clear()
            if tick % decimation == 0 and data.time >= args.settle:
                mujoco.mj_forward(model, data)
                rotation = data.xmat[root].reshape(3, 3)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 1)
                with lock:
                    cmd = command.copy()
                target, *_ = controller.step(velocity[:3], rotation.T @ np.array([0., 0., -1.]), cmd, data.qpos[qadr], data.qvel[vadr])
                if not np.isfinite(target).all():
                    raise RuntimeError('Non-finite controller target')
                world_velocity = rotation @ velocity[3:]
                heights.append(float(data.xpos[root, 2]))
                velocities.append(float(world_velocity[0]))
                if writer:
                    writer.writerow([data.time, data.xpos[root, 2], *world_velocity[:2], *cmd, *data.qpos[qadr], *target])
            torque = pd['stiffness_Nm_per_rad'] * (target - data.qpos[qadr]) - pd['damping_Nm_s_per_rad'] * data.qvel[vadr]
            data.ctrl[actuators] = np.clip(torque, -pd['effort_limit_Nm'], pd['effort_limit_Nm'])
            mujoco.mj_step(model, data)
            tick += 1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Simulation state became non-finite')
            if viewer is not None:
                if tick % 10 == 0:
                    viewer.cam.lookat[:] = data.xpos[root]
                    viewer.sync()
                time.sleep(max(0, wall_start + data.time - time.monotonic()))
        if heights:
            print(json.dumps({'policy_ticks': len(heights), 'height_min_m': min(heights), 'height_mean_m': float(np.mean(heights)), 'vx_world_mean_m_s': float(np.mean(velocities))}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
