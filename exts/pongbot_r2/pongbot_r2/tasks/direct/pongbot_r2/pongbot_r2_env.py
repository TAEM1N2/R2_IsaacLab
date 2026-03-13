# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import random # random 모듈 추가

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from ... import controllers 


from .pongbot_r2_env_cfg import PongbotR2EnvCfg

import websocket
import json
ws = websocket.WebSocket()
ws.connect("ws://localhost:9871")

class PongbotR2Env(DirectRLEnv):
    cfg: PongbotR2EnvCfg

    def __init__(self, cfg: PongbotR2EnvCfg, render_mode: str | None = None, control_mode: str = "remote", **kwargs):
        """
        control_mode (str): 제어 모드를 선택합니다. 
                            'joy': 조이스틱 (로컬)
                            'keyboard': 키보드 (로컬, 누를때만 동작)
                            'remote': 원격 키보드 (원격, 속도 증감)
                            None: 컨트롤러 사용 안함
        """
        super().__init__(cfg, render_mode, **kwargs)  

        # +++ 로봇의 관절 및 링크 정보 출력 (초록색) +++
        print("\n" + "="*50)
        print("\033[92mInitializing MiniPb Environment...\033[0m")
        
        # 관절(Joint) 이름 및 Index 출력
        print("\033[92m\n--- Joint Information ---\033[0m")
        if hasattr(self._robot, 'joint_names'):
            for i, name in enumerate(self._robot.joint_names):
                print(f"\033[92mJoint Name : {name} (Index: {i})\033[0m")
        
        # 링크(Body) 이름 및 Index 출력
        print("\033[92m\n--- Link (Body) Information ---\033[0m")
        if hasattr(self._robot, 'body_names'):
            for i, name in enumerate(self._robot.body_names):
                print(f"\033[92mLink Name : {name} (Index: {i})\033[0m")

        print("\033[92m" + "="*50 + "\033[0m\n")
        # +++++++++++++++++++++++++++++++++++++

        if isinstance(self.cfg.action_scale, (list, tuple)):
            self._action_scale_tensor = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)
        else:
            # 기존 스칼라 값도 호환되도록 처리합니다.
            action_dim = gym.spaces.flatdim(self.single_action_space)
            self._action_scale_tensor = torch.full((action_dim,), self.cfg.action_scale, device=self.device, dtype=torch.float32)

        # +++ 명목상 ROM(Range of Motion)의 중심 위치를 정의합니다. +++
        # 이 값은 default_joint_pos를 대체하여 action 계산에 사용됩니다.
        # HR = 0, HP = 0.33, KNP = -1.0
        nominal_rom_center_list = [0.0] * 4 + [0.33] * 4 + [-1.0] * 4
        self._nominal_rom_center = torch.tensor(nominal_rom_center_list, device=self.device, dtype=torch.float32)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._plot_obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.commands_x          = self._commands.view(self.num_envs, 3)[..., 0]
        self.commands_y          = self._commands.view(self.num_envs, 3)[..., 1]
        self.commands_yaw        = self._commands.view(self.num_envs, 3)[..., 2] 

    
        self.stand_env_range = [0, self.cfg.stand_env_range - 1]
        self.only_plus_x_envs_range = [self.stand_env_range[1] + 1, self.stand_env_range[1] + self.cfg.only_plus_x_envs_range]
        self.only_minus_x_envs_range = [self.only_plus_x_envs_range[1] + 1, self.only_plus_x_envs_range[1] + self.cfg.only_minus_x_envs_range]
        self.only_plus_y_envs_range = [self.only_minus_x_envs_range[1] + 1, self.only_minus_x_envs_range[1] + self.cfg.only_plus_y_envs_range]
        self.only_minus_y_envs_range = [self.only_plus_y_envs_range[1] + 1, self.only_plus_y_envs_range[1] + self.cfg.only_minus_y_envs_range]
        self.only_plus_yaw_envs_range = [self.only_minus_y_envs_range[1] + 1, self.only_minus_y_envs_range[1] + self.cfg.only_plus_yaw_envs_range]
        self.only_minus_yaw_envs_range = [self.only_plus_yaw_envs_range[1] + 1, self.only_plus_yaw_envs_range[1] + self.cfg.only_minus_yaw_envs_range]

        self.sin_cycle    = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.cos_cycle    = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.cycle_t      = torch.zeros((self.num_envs,1), device=self.device, dtype=torch.float)
        self.cycle_period = 0.6  # 주기 (초)
        self.phi = torch.tensor([0,torch.pi,torch.pi,0], device=self.device, dtype=torch.float)

        # ROS2 노드 및 퍼블리셔 초기화
        # if not rclpy.ok():
        #     rclpy.init()
        # self.node = rclpy.create_node('minipb_plot_juggler_node')
        
        self.observe_envs = 0
        if self.num_envs != 1:
            self.observe_envs = 501

        self._reward_pubs = {}
        self.rew_scales = {}

        self._recovery_duration = getattr(self.cfg, "recovery_duration", 0.7)
        self._recovery_time_counter = torch.full((self.num_envs,), self._recovery_duration, device=self.device)

        # +++ freeze 기능 상태 변수 초기화 +++
        self._global_step_counter: int = 0
        self._freeze_flag: bool = False
        self._freeze_cnt: int = 0
        self._freeze_steps: int = 0
        # +++++++++++++++++++++++++++++++++++++

        for key, value in vars(self.cfg).items():
            if key.startswith("reward_") or key.startswith("penalty_"):
                self.rew_scales[key] = value
        
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.step_dt

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.rew_scales.keys()
        }   
        self._reward_container = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.rew_scales.keys()
        }   
         
        self._base_id, _    = self._contact_sensor.find_bodies("BODY")
        self._feet_ids, _   = self._contact_sensor.find_bodies(".*TIP")
        self._thigh_ids, _  = self._contact_sensor.find_bodies(".*THIGH")
        self._calf_ids, _   = self._contact_sensor.find_bodies(".*CALF")

        # self._thigh_ids  = [5,6,7,8]
        # self._calf_ids   = [9,10,11,12]
        # self._feet_ids   = [13,14,15,16]
        print("======================================================")
        print("feet_ids:", self._feet_ids)
        print("thigh_ids:", self._thigh_ids)
        print("calf_ids:", self._calf_ids)
        print("======================================================")

        # +++ 로봇 밀기(Push) 기능 관련 파라미터 +++
        # 로봇을 밀어주는 주기 (단위: 시뮬레이션 스텝)
        self.push_interval = 2100
        # 로봇을 밀 때의 선속도(x, y) 최대 강도
        self.push_strength_lin = 0.75
        # 로봇을 밀 때의 각속도 최대 강도
        self.push_strength_ang = 0.5 
        # 각 환경별로 마지막 push 이후 지난 스텝을 카운트
        self._steps_since_push = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # +++++++++++++++++++++++++++++++++++++++++++++++        
        
        # +++ 발 접촉 개수 이동 평균을 위한 버퍼 추가 +++
        self.contact_feet_buffer_len = 5  # 이동 평균을 계산할 버퍼 크기
        self._num_contact_feet_buffer = torch.zeros(
            (self.num_envs, self.contact_feet_buffer_len), 
            device=self.device, 
            dtype=torch.float
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++
        self._was_commanded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # +++ '짧은 스윙' 페널티를 위한 상태 변수 추가 +++
        self._last_foot_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._swing_start_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        # +++++++++++++++++++++++++++++++++++++++++++++++

        # +++ 컨트롤러 초기화 로직 +++
        self.controller = None
        if self.num_envs == 1 and control_mode is not None:
            try:
                if control_mode == "joy":
                    self.controller = controllers.JoystickController(self.cfg)
                elif control_mode == "keyboard":
                    self.controller = controllers.LocalKeyboardController(self.cfg)
                elif control_mode == "remote":
                    self.controller = controllers.RemoteKeyboardController(self.cfg)
                else:
                    print(f"\033[93m[WARN] Invalid control_mode '{control_mode}'. No controller initialized.\033[0m")
                
                if self.controller:
                    self.controller.start() # 컨트롤러의 백그라운드 스레드 시작
            
            except (ImportError, ConnectionError) as e:
                print(f"\033[91m[ERROR] Failed to initialize '{control_mode}' controller: {e}\033[0m")
                print(f"\033[93m[WARN] Continuing without a controller.\033[0m")
                self.controller = None
        # ++++++++++++++++++++++++++++++++++++++++++++

        # --- [Fix A: Reward Stats Init 포함 키 확장] ---
        reward_keys_for_stats = list(self.rew_scales.keys()) + ["total_reward"]

        self._rew_stats = {
            "count": torch.tensor(0, device=self.device, dtype=torch.long),
            "mean": {k: torch.zeros((), device=self.device, dtype=torch.float32) for k in reward_keys_for_stats},
            "m2":   {k: torch.zeros((), device=self.device, dtype=torch.float32) for k in reward_keys_for_stats},
        }
        self._rew_stats_last = None
        # ----------------------------------------------
        # __init__ 마지막 부분 근처(조명 세팅 이후, terrain 생성 직후 등)에 추가
        # self._terrain 은 이미 생성됨: self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # ==== 커리큘럼 플래그/메타 ====
        self._use_curriculum = getattr(self.cfg.terrain, "curriculum", False)

        # IsaacLab terrain이 제공하는 env_origins (일반적으로 [num_envs, 3])
        self._terrain_origins = None
        if hasattr(self._terrain, "env_origins"):
            self._terrain_origins = torch.as_tensor(
                self._terrain.env_origins, device=self.device, dtype=torch.float32
            )

        # 그리드 크기 & 래핑 정책
        rows_cfg = int(getattr(self.cfg.terrain, "curriculum_rows", 1))
        cols_cfg = int(getattr(self.cfg.terrain, "curriculum_cols", max(1, self.num_envs)))
        wrap_rows = bool(getattr(self.cfg.terrain, "curriculum_wrap_rows", True))

        self._grid_rows = max(1, rows_cfg)
        self._grid_cols = max(1, cols_cfg)
        self._wrap_rows = wrap_rows

        # num_envs 와 grid 불일치 시 보정
        # - grid가 더 크면 사용 가능한 선형 Index만 쓴다.
        # - grid가 더 작으면 cols를 키워 맞춘다(행 수는 유지).
        total_slots = self._grid_rows * self._grid_cols
        if total_slots < self.num_envs:
            # 열 확장
            needed_cols = (self.num_envs + self._grid_rows - 1) // self._grid_rows
            self._grid_cols = max(self._grid_cols, needed_cols)
            total_slots = self._grid_rows * self._grid_cols

        # (row, col) Index: [num_envs]
        lin = torch.arange(self.num_envs, device=self.device)
        self._row_idx = torch.div(lin, self._grid_cols, rounding_mode='floor').clamp(max=self._grid_rows - 1)
        self._col_idx = (lin % self._grid_cols)

        # row/col → linear Index로 바꿔 env_origins에서 가져오도록 헬퍼 준비
        def _rc_to_lin(row_t, col_t):
            lin_idx = (row_t * self._grid_cols + col_t).clamp(min=0, max=total_slots - 1)
            # grid에 빈 슬롯이 있을 수 있으므로 num_envs-1로도 한 번 더 클램프
            lin_idx = lin_idx.clamp(max=self.num_envs - 1)
            return lin_idx

        self._rc_to_lin = _rc_to_lin  # 바인딩 (파이썬 클로저)

        # 기대 이동 임계 계산용 메타
        self._terrain_env_length = float(getattr(self._terrain, "env_length", 5.0))

        # 캐시: 항상 [num_envs, 3] 유지
        if self._terrain_origins is not None and self._terrain_origins.ndim == 2:
            # 2D per-env: 초기엔 자신의 origin으로 채운다
            self._env_origins_cache = self._terrain_origins.clone()
        elif self._terrain_origins is not None and self._terrain_origins.ndim == 3:
            # 3D grid: 여긴 이전에 구현한 3D 경로 사용(그대로 두어도 됨)
            L, T, _ = self._terrain_origins.shape
            self._terrain_levels = torch.randint(low=0, high=L, size=(self.num_envs,), device=self.device)
            self._terrain_types  = torch.randint(low=0, high=T, size=(self.num_envs,), device=self.device)
            self._env_origins_cache = self._terrain_origins[self._terrain_levels, self._terrain_types]
        else:
            self._env_origins_cache = torch.zeros(self.num_envs, 3, device=self.device)


    def _update_terrain_level(self, env_ids: torch.Tensor):
        if not self._use_curriculum or self._terrain_origins is None:
            return
        if env_ids is None or env_ids.numel() == 0:
            return

        # 현재 위치 & origin (항상 캐시 사용)
        base_xy   = self._robot.data.root_state_w[env_ids, :2]
        origin_xy = self._env_origins_cache[env_ids][:, :2]
        distance  = torch.norm(base_xy - origin_xy, dim=1)

        # 기대 이동량 스케일 (명령 크기 × 에피소드 시간 × 계수)
        cmd_xy = torch.norm(self._commands[env_ids, :2], dim=1)
        expect_scale = cmd_xy * self.max_episode_length_s * 0.25

        # 업/다운 결정
        up_mask   = (distance > (self._terrain_env_length * 0.5))    # 더 멀리 갔으면 업
        down_mask = (distance < expect_scale)                        # 기대보다 못 갔으면 다운
        delta = up_mask.to(torch.int32) - down_mask.to(torch.int32)  # +1 / 0 / -1

        if self._terrain_origins.ndim == 3:
            # 기존 3D-grid 경로(이미 구현되어 있으면 그대로 사용)
            lv = self._terrain_levels[env_ids]
            tp = self._terrain_types[env_ids]
            lv = lv + delta
            L = self._terrain_origins.shape[0]
            if L > 0:
                if getattr(self, "_wrap_rows", True):
                    lv = torch.remainder(lv, L)
                else:
                    lv = lv.clamp(0, L - 1)
            self._terrain_levels[env_ids] = lv
            new_origins = self._terrain_origins[lv, tp]
            self._env_origins_cache[env_ids] = new_origins
            return

        # ===== 2D per-env: row만 이동 =====
        row = self._row_idx[env_ids]
        col = self._col_idx[env_ids]

        # 행 업데이트
        row = row + delta

        if self._wrap_rows:
            row = torch.remainder(row, self._grid_rows)
        else:
            row = row.clamp(0, self._grid_rows - 1)

        # 갱신 저장
        self._row_idx[env_ids] = row

        # (row, col) → 선형 Index
        lin_idx = self._rc_to_lin(row, col)

        # 캐시 오리진 갱신
        new_origins = self._terrain_origins[lin_idx]  # shape [|env_ids|, 3]
        self._env_origins_cache[env_ids] = new_origins


    def _get_noise_scale_vec(self) -> torch.Tensor:
        """
        설정(cfg)을 바탕으로 최종 관측 노이즈 스케일 텐서를 생성합니다.
        """
        # 1. 전역 스위치 확인: cfg.add_noise가 False이면 노이즈를 적용하지 않습니다.
        if not self.cfg.add_noise:
            return torch.zeros(self.single_observation_space.shape[0], device=self.device)

        noise_map = self.cfg.observation_noise_map
        scales_list = []

        # 2. observation_noise_map을 기반으로 스케일 리스트를 동적으로 생성합니다.

        scales_list.extend([noise_map["root_ang_vel_b"]] * 3)
        scales_list.extend([noise_map["projected_gravity_b"]] * 3)
        scales_list.extend([noise_map["commands"]] * 3)
        scales_list.extend([noise_map["joint_pos"]] * self._robot.num_joints) # 12
        scales_list.extend([noise_map["joint_vel"]] * self._robot.num_joints) # 12
        scales_list.extend([noise_map["actions"]] * self.cfg.action_space)    # 12
        scales_list.extend([noise_map["cycles"]] * 8) # sin, cos 각 4개씩      # 8  

        # 리스트를 텐서로 변환
        noise_vec = torch.tensor(scales_list, device=self.device, dtype=torch.float)



        # 3. 전역 noise_level을 곱하여 최종 스케일 텐서를 완성합니다.
        noise_vec *= self.cfg.noise_level



        return noise_vec
    
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._global_step_counter += 1
        self._recovery_time_counter += self.step_dt
        self.no_commands = torch.norm(self._commands, dim=-1) == 0.
        # --- 로직 끝 ---

        self.cycle_t += self.step_dt 
        self.cycle_t =  torch.where(self.cycle_t > self.cycle_period,
                                    self.cycle_t - self.cycle_period,
                                    self.cycle_t)        
        self.sin_cycle = torch.sin(2 * torch.pi * (self.cycle_t / self.cycle_period) + self.phi)* ~self.no_commands.unsqueeze(1)
        self.cos_cycle = torch.cos(2 * torch.pi * (self.cycle_t / self.cycle_period) + self.phi)* ~self.no_commands.unsqueeze(1)

        # +++ push 카운터 증가 및 push 메서드 호출 +++
        if self.num_envs != 1:
            self._steps_since_push += 1
            # self._push_robots()
        # ++++++++++++++++++++++++++++++++++++++++++++

        # +++ [수정] 컨트롤러로부터 커맨드를 받아오는 로직 +++
        if self.num_envs == 1 and self.controller is not None:
            self._commands[:] = self.controller.get_commands()
            
            # 터미널에 현재 목표 속도 출력 (선택 사항)
            if self._global_step_counter % 10 == 0:
                vx = self._commands[0, 0].item()
                vy = self._commands[0, 1].item()
                vyaw = self._commands[0, 2].item()
                print(f"\r\033[K>> Target Velocity | X: {vx: 6.2f}, Y: {vy: 6.2f}, Yaw: {vyaw: 6.2f}", end="")



        # --- freeze 로직 시작 ---
        is_play_mode = self.num_envs == 1
        # 1. freeze 상태가 아닐 때, freeze를 시작할지 확인
        if not self._freeze_flag and not is_play_mode and (self._global_step_counter % self.cfg.freeze_interval == 0):
            self._freeze_flag = True
            self._freeze_cnt = 0
            # Cfg에서 설정한 min/max 값으로 무작위 지속시간 설정
            self._freeze_steps = random.randint(self.cfg.freeze_duration_min, self.cfg.freeze_duration_max)
        
        # 2. 현재 freeze 상태일 경우, 로직 처리
        if self._freeze_flag:
            # 모든 환경의 목표 속도를 0으로 설정 (정지 명령)
            self._commands[:] = 0.0
            self._freeze_cnt += 1
            
            # freeze 지속 시간이 끝나면,
            if self._freeze_cnt >= self._freeze_steps:
                # 새로운 무작위 명령을 모든 환경에 할당
                all_env_ids = torch.arange(self.num_envs, device=self.device)
                self.set_cmd(all_env_ids)
                # 플래그 및 카운터 리셋
                self._freeze_flag = False
                self._freeze_cnt = 0
        # --- freeze 로직 끝 ---



        
        # print("self._robot.data.default_joint_pos : " , self._robot.data.default_joint_pos)
        self._actions = actions.clone()
        self._processed_actions = self._action_scale_tensor * self._actions + self._robot.data.default_joint_pos
        # --------------------------------------------

        # self._processed_actions = self._robot.data.default_joint_pos
   
    def close(self):
        """환경과 관련된 리소스를 정리합니다."""
        if self.controller is not None:
            self.controller.stop()
        # 만약 부모 클래스에도 close 메서드가 있다면 호출해주는 것이 좋습니다.
        # super().close() 

        # --- [Patch: Reward Stats Functions] -------------------------------
    def _update_reward_stats(self):
        """
        self._reward_container의 현재 step 값들로 리워드별 러닝 평균/분산 갱신 (Welford).
        """
        # count 증가
        self._rew_stats["count"] += 1
        n = self._rew_stats["count"].item()

        for k, x in self._reward_container.items():
            # x: scalar tensor (이미 device 상에 있음)
            mean = self._rew_stats["mean"][k]
            m2   = self._rew_stats["m2"][k]

            delta  = x - mean
            mean_n = mean + delta / n
            delta2 = x - mean_n
            m2_n   = m2 + delta * delta2

            # 갱신
            self._rew_stats["mean"][k] = mean_n
            self._rew_stats["m2"][k]   = m2_n


    def _get_reward_stats_snapshot(self):
        """
        현재까지 누적된 리워드별 mean/std/cv 딕셔너리 반환.
        cv = std / (|mean| + 1e-6)
        """
        n = self._rew_stats["count"].item()
        stats = {}
        for k in self.rew_scales.keys():
            mean = self._rew_stats["mean"][k]
            # 표본분산: n>1이면 m2/(n-1), 아니면 0
            if n > 1:
                var = self._rew_stats["m2"][k] / (n - 1)
            else:
                var = torch.zeros_like(mean)
            var = torch.clamp(var, min=0.0)
            std = torch.sqrt(var)
            cv  = std / (torch.abs(mean) + 1e-6)

            stats[k] = {
                "mean": mean.item(),
                "std":  std.item(),
                "cv":   cv.item(),
            }
        # total_reward도 참고용으로 함께 계산(선택)
        if "total_reward" in self._reward_container:
            # total_reward는 _reward_container 업데이트 시에만 최신임
            # 통계 버퍼는 개별 키만 누적하므로 여기선 단발 스냅샷만 붙인다.
            stats["total_reward_snapshot"] = float(self._reward_container["total_reward"].item())
        return stats
    # -------------------------------------------------------------------


    def plot_data(self):
        """
        self._reward_container에 있는 모든 보상/페널티 항목을
        웹소켓을 통해 JSON 형식으로 전송합니다.
        """
        # 1) 원래 payload (step당 리워드 스칼라들)
        payload = {key: value.item() for key, value in self._reward_container.items()}

        # 2) 통계 스냅샷 준비: None이면 즉석 생성
        #    (초기 구간에도 항상 전송되도록)
        if getattr(self, "_rew_stats_last", None) is None and hasattr(self, "_get_reward_stats_snapshot"):
            self._rew_stats_last = self._get_reward_stats_snapshot()

        # 3) 중첩 평탄화: mean/std/cv 각각 1-계층 딕셔너리로 변환
        rew_mean, rew_std, rew_cv = {}, {}, {}
        if isinstance(self._rew_stats_last, dict):
            for k, v in self._rew_stats_last.items():
                # total_reward_snapshot 같이 스칼라만 있는 키는 건너뜀
                if isinstance(v, dict):
                    # float()로 JSON 직렬화 보장
                    rew_mean[k] = float(v.get("mean", 0.0))
                    rew_std[k]  = float(v.get("std", 0.0))
                    rew_cv[k]   = float(v.get("cv", 0.0))

        data_to_send = {
            "rewards" : payload,
            "obs"     : self._plot_obs[self.observe_envs].cpu().numpy().tolist(),
            "actions" : self._actions[self.observe_envs].cpu().numpy().tolist(),
            "lin_vel"  : self._robot.data.root_lin_vel_b[self.observe_envs].cpu().numpy().tolist(),
            "modified_actions" : self._processed_actions[self.observe_envs].cpu().numpy().tolist(),
            "joint_torque" : self._robot.data.applied_torque[self.observe_envs].cpu().numpy().tolist(),
                    # 👇 새로 추가: PlotJuggler에서 바로 보이도록 평탄화된 통계
            "rew_mean"       : rew_mean,
            "rew_std"        : rew_std,
            "rew_cv"         : rew_cv,

        }


        ws.send(json.dumps(data_to_send))


    def _apply_action(self):
        # print("self._processed_actions", self._processed_actions[0,:])
        self._robot.set_joint_position_target(self._processed_actions)

    # def _plot_data(self):
    #       # +++ ROS2 환경에 맞게 변환된 plotting 메서드 +++
    #     """Publishes simulation data to ROS2 topics for visualization."""

    #     # --- 데이터 추출 ---
    #     obs_env_id = self.observe_envs

    #     # +++ Action 데이터 퍼블리싱 +++
    #     action_msg = JointState()
    #     action_msg.header.stamp = self.node.get_clock().now().to_msg()
    #     action_msg.name = self._action_names
    #     # .tolist()를 사용하여 torch 텐서를 파이썬 리스트로 변환
    #     action_msg.position = self._actions[obs_env_id].tolist()
    #     self._action_pub.publish(action_msg)
    #     # ++++++++++++++++++++++++++++++

    #     # +++ Observation 데이터 퍼블리싱 +++
    #     obs_msg = JointState()
    #     obs_msg.header.stamp = self.node.get_clock().now().to_msg()
    #     obs_msg.name = self._observation_names
    #     obs_msg.position = self._last_obs[obs_env_id].tolist()
    #     self._observation_pub.publish(obs_msg)
    #     # ++++++++++++++++++++++++++++++       

        
    #     # --- 메시지 생성 및 데이터 채우기 ---

    #     # 4. 보상 메시지
    #     for reward_name, reward_value in self._reward_container.items():
    #         topic_name = f"/{reward_name}"
    #         # 해당 토픽의 퍼블리셔가 없으면 생성
    #         if topic_name not in self._reward_pubs:
    #             self._reward_pubs[topic_name] = self.node.create_publisher(Float32, topic_name, 10)
            
    #         reward_msg = Float32()
    #         # dt로 나누어 스케일링 (원본 코드와 동일하게)
    #         reward_msg.data = reward_value.item() / self.step_dt
    #         self._reward_pubs[topic_name].publish(reward_msg)

    #     # 5. 총 보상 메시지
    #     # if "total_reward" in self._reward_container:
    #     #     tot_reward_msg = Float32()
    #     #     tot_reward_msg.data = self._reward_container["total_reward"].item() / self.step_dt
    #     #     self._total_reward_pub.publish(tot_reward_msg)

    #     # --- 메시지 발행 ---
    #     # ROS2 콜백 처리
    #     rclpy.spin_once(self.node, timeout_sec=0)
    # # ++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_observations(self) -> dict:
        # self._previous_actions = self._actions.clone()
        self.no_commands = torch.norm(self._commands, dim=-1) == 0.
        if not hasattr(self, "noise_scales"):
            self.noise_scales = self._get_noise_scale_vec()
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_ang_vel_b * 0.25,             # 3 : [0:3] 
                    self._robot.data.projected_gravity_b,               # 3 : [3:6]
                    self._commands,                                     # 3 : [6:9]
                    self._robot.data.joint_pos,                         # 12: [9:21]
                    self._robot.data.joint_vel * 0.1,                   # 12: [21:33]
                    self._actions,                                      # 12: [33:45]
                    self.sin_cycle * ~self.no_commands.unsqueeze(1),    # 4 : [45:49]
                    self.cos_cycle * ~self.no_commands.unsqueeze(1),    # 4 : [49:53]
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # print(f"Noise vector: {self.noise_scales.shape}")
        # print(f"Observation vector: {obs.shape}")
        self._plot_obs = obs.clone()

        if self.num_envs != 1:
            if self.noise_scales.sum() > 0.0:
                noise = torch.randn_like(obs)
                obs += noise * self.noise_scales


        observations = {"policy": obs}
        return observations

    def _check_recovering_state(self) -> torch.Tensor:
        """
        각 환경이 외란으로부터 '균형 회복' 상태에 있는지 확인합니다.

        '회복 상태'는 마지막으로 외란을 받은 후, 설정된 '회복 시간'
        (self._recovery_duration)이 지나지 않은 상태를 의미합니다.

        Returns:
            torch.Tensor: 각 환경에 대한 boolean 값의 텐서.
                          회복 상태이면 True, 아니면 False. (shape: [num_envs,])
        """
        return self._recovery_time_counter < self._recovery_duration
    
    def _get_rewards(self) -> torch.Tensor:
        is_recovering = self._check_recovering_state() # push 이후 경과 시간을 체크하는 함수
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        calf_forces     = net_contact_forces[:,:,self._calf_ids,:].mean(dim=1) 
        
        foot_forces     = net_contact_forces[:,:,self._feet_ids,:].mean(dim=1) 
        foot_contact    = torch.abs(foot_forces[:,:,2]) > 1.
        
        foot_pos        = self._robot.data.body_link_state_w[:,[13,14,15,16], 0:3]
        foot_velocities = self._robot.data.body_link_state_w[:, [13,14,15,16], 7:10]

        dof_pos = self._robot.data.joint_pos
        default_dof_pos = self._robot.data.default_joint_pos

        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.1)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.125)

        # +++ 지지 다각형(Support Polygon) 보상 계산 시작 +++
        base_pos_xy = self._robot.data.root_state_w[:, :2]
        
        # 2. 모든 발의 수평(x, y) 위치를 가져옵니다.
        foot_pos_xy = foot_pos[:, :, :2]

        # 3. 지면에 닿은 발의 개수를 계산합니다. (0으로 나누는 것을 방지하기 위해 최소 1로 클램핑)
        num_contact_feet = torch.sum(foot_contact, dim=1).clamp(min=1.0)

        # 4. 지면에 닿은 발들의 x, y 좌표의 합을 구합니다.
        contact_foot_pos_sum = torch.sum(foot_pos_xy * foot_contact.unsqueeze(-1), dim=1)

        # 5. 지지 다각형의 중심(centroid)을 계산합니다. (좌표 합 / 닿은 발 개수)
        support_polygon_center = contact_foot_pos_sum / num_contact_feet.unsqueeze(-1)
        # 6. 베이스 위치와 지지 다각형 중심 사이의 거리를 계산합니다.
        dist_to_center = torch.norm(base_pos_xy - support_polygon_center, dim=1)
        reward_support_polygon = torch.exp(-dist_to_center/0.125) * (num_contact_feet > 1)

        # +++ 지지 다각형 보상 계산 종료 +++

        # =======================================================================================================
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.norm(self._robot.data.root_ang_vel_b[:, :2], dim=-1)
        # base_height
        ref_base_height = 0.55
        base_height_err = torch.abs(self._robot.data.root_state_w[:, 2] - ref_base_height)
        # print("root_state_w",self._robot.data.root_state_w[:, 2])


        # feet air time
        # first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        # last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
        #     torch.norm(self._commands[:, :2], dim=1) > 0.1
        # )
        # undesired contacts

        #Swing_phase
        cycle_lower = 0.5
        swing_phase = self.sin_cycle > cycle_lower
        stance_phase = self.sin_cycle < -cycle_lower

        foot_velocities_norm = torch.sum(torch.square(foot_velocities[:,:,:]), dim=-1)
        foot_forces_norm     = torch.clamp(torch.sum(torch.square(foot_forces[:,:,:]), dim=-1), max=10.,)
        calf_forces_norm     = torch.clamp(torch.sum(torch.square(calf_forces[:,:,:]), dim=-1), max=10.,)

        swing_stance_penalty =  torch.sum(foot_velocities_norm * 0.5  * stance_phase , dim=-1) +\
                                torch.sum(foot_forces_norm     * 0.01 * swing_phase , dim=-1) +\
                                torch.sum(calf_forces_norm     * 0.001 , dim=-1)
        penalty_swing_stance_phase = swing_stance_penalty * ~self.no_commands

        # print("foot_velocities_norm : ", foot_velocities_norm[self.observe_envs])
        # print("foot_forces_norm : ", foot_forces_norm[self.observe_envs])

        penalty_swing_stance_phase = swing_stance_penalty * ~self.no_commands
        
        
        # +++ 기본 GRF(지면반발력) 균등 분배 페널티 +++
        # 1. 각 발의 수직 지면 반발력(GRF)을가져옵니다.
        #    안정적인 값을 위해 접촉 센서의 이력(history)을 평균내어 사용합니다.
        foot_grfs_z = net_contact_forces.mean(dim=1)[:, self._feet_ids, 2]

        # 2. 각 환경(env)별로 네 발의 평균 GRF를 계산합니다.
        #    keepdim=True로 차원을 유지하여 브로드캐스팅이 용이하게 합니다. (결과 shape: [num_envs, 1])
        mean_grf_per_env = torch.mean(foot_grfs_z, dim=1, keepdim=True)
        
        # 3. 각 발의 GRF가 평균과 얼마나 다른지(편차)의 총합을 구합니다.
        #    힘이 고르게 분산될수록 이 오차 값은 0에 가까워집니다.
        grf_err = torch.sum(torch.abs(foot_grfs_z - mean_grf_per_env), dim=1)
        
        # 4. '정지' 명령 상태(`self.no_commands`가 True)일 때만 이 페널티를 활성화합니다.
        penalty_default_grf = grf_err * self.no_commands * ~is_recovering
        # --- 로직 끝 ---




        # foot_height
        ref_foot_h = 0.075
        foot_height_err = torch.norm((foot_pos[:, :, 2] - ref_foot_h) ,dim=-1) * ~self.no_commands

        # default dof pos
        default_dof_pos_err = torch.abs(dof_pos - default_dof_pos)   
        penalty_default_pos_standing = torch.sum(default_dof_pos_err, dim=1) * self.no_commands * ~is_recovering

        y_cmd_zero = (torch.abs(self._commands[:,1]) <= 0.05)
        yaw_cmd_zero = self._commands[:,2] == 0
        # HR penalty
        HR_pos_err = torch.sum(default_dof_pos_err[:,0:4],dim=-1)   * ~self.no_commands* ~is_recovering * y_cmd_zero * yaw_cmd_zero
        # HP penalty
        HP_pos_err = torch.sum(default_dof_pos_err[:,4:8],dim=-1)   * ~self.no_commands* ~is_recovering
        # KNP penalty
        KNP_pos_err = torch.sum(default_dof_pos_err[:,8:12],dim=-1) * ~self.no_commands* ~is_recovering

        # Joint acceleration
        joint_accel = torch.norm(self._robot.data.joint_acc, dim=-1)

        joint_vel = torch.norm(self._robot.data.joint_vel, dim=-1)


        # trot
        trot_pitch_err = torch.sum(abs(dof_pos[:,[4,5,8,9]] - dof_pos[:,[7,6,11,10]]),dim=-1)

        # HR:  0 1 2 3
        # HP:  4 5 6 7
        # KNP: 8 9 10 11

        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)


        calf_contact_err = torch.sum(torch.norm(calf_forces,dim=-1),dim=-1)


        foot_xy_velocities = torch.norm(foot_velocities[:,:,:2],dim=-1)
        # foot_xy_velocities_norm = torch.norm(foot_xy_velocities, dim=-1)
        slip_err = torch.sum(foot_contact * foot_xy_velocities, dim=-1)

        # +++ 이동 평균을 이용한 발 접촉 패널티 계산  +++
        # 1. 현재 스텝의 발 접촉 개수를 계산합니다. (정수 타입을 float으로 변환)
        current_num_contact_feet = torch.sum(foot_forces[:,:,2] > 1.0, dim=-1).float()
        # print("foot_forces:", foot_forces[self.observe_envs])
        # print("contact_forces:", foot_forces[:,:,2] > 1.0)
        # print("current_num_contact_feet:", current_num_contact_feet[self.observe_envs])

        # 2. 버퍼를 왼쪽으로 한 칸씩 밀고, 가장 오른쪽에 새로운 관측값을 추가합니다.
        self._num_contact_feet_buffer = torch.roll(self._num_contact_feet_buffer, shifts=-1, dims=1)
        self._num_contact_feet_buffer[:, -1] = current_num_contact_feet
        
        # 3. 버퍼의 평균을 계산하여 이동 평균을 구합니다.
        moving_avg_contact_feet = torch.mean(self._num_contact_feet_buffer, dim=1)

        # 4. 이동 평균 값과 이상적인 접촉 상태(2 또는 4)와의 오차를 계산합니다.
        dist_to_2 = torch.abs(moving_avg_contact_feet - 2.0)
        dist_to_4 = torch.abs(moving_avg_contact_feet - 4.0)
        num_contact_feet_err = torch.where(~self.no_commands,
                                           torch.min(dist_to_2, dist_to_4)**2,
                                           (2 * dist_to_4 **2)) 

        num_contact_feet_err = torch.where(is_recovering,
                                           torch.zeros_like(num_contact_feet_err),
                                           num_contact_feet_err)

        # num_contact_feet_err *= ~self._freeze_flag 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # --- '짧은 스윙' 페널티 계산 로직 ---
        # 1. 현재 시뮬레이션 시간을 명시적으로 계산합니다.
        current_sim_time = self._sim_step_counter * self.physics_dt
        # 2. 현재 스텝의 발 접촉 상태를 가져옵니다.
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        current_foot_contact = net_contact_forces[:, -1, self._feet_ids, 2] > 1.0

        # 3. 이전 상태와 비교하여 '이륙'과 '착지' 이벤트를 감지합니다.
        liftoff_events = ~current_foot_contact & self._last_foot_contact
        touchdown_events = current_foot_contact & ~self._last_foot_contact

        # 4. '이륙'한 발의 스윙 시작 시간을 기록합니다.
        self._swing_start_time[liftoff_events] = current_sim_time

        # 5. '착지'한 발의 총 스윙 시간을 계산합니다.
        swing_duration = current_sim_time - self._swing_start_time
        
        # 6. '착지'한 발 중에서 최소 스윙 시간(min_swing_time)을 만족하지 못한 경우, 그 시간 차이를 계산합니다.
        #    (min_swing_time - swing_duration)이 양수일 경우에만 페널티가 됩니다.
        time_difference = self.cfg.min_swing_time - swing_duration

        # 7. 착지 이벤트가 발생했고, 스윙 시간이 짧았던 경우에만 페널티를 적용합니다.
        #    relu 함수를 사용하여 시간 차이가 음수인 경우(즉, 스윙이 충분히 길었던 경우)는 0으로 만듭니다.
        #    그 후, 각 환경(env)별로 모든 발의 페널티를 합산합니다.
        short_swing_penalty_per_foot = torch.nn.functional.relu(time_difference) * touchdown_events
        penalty_short_swing = torch.sum(short_swing_penalty_per_foot, dim=1)
        
        # 8. 다음 스텝을 위해 현재 접촉 상태를 저장합니다. (매우 중요)
        self._last_foot_contact = current_foot_contact.clone()
        # --- 로직 끝 ---
        

        # +++ 관절 위치 한계 근접 페널티 (Joint Limit Proximity Penalty) +++
        # 1. 현재 관절 위치 및 USD에서 설정된 한계값을 가져옵니다.
        current_joint_pos = self._robot.data.joint_pos
        joint_limits = self._robot.data.joint_pos_limits
        lower_limits = joint_limits[..., 0] # 하한 (e.g., -1.0)
        upper_limits = joint_limits[..., 1] # 상한 (e.g., 1.0)

        # print("lower_limits:", lower_limits)
        # print("upper_limits:", upper_limits)

        # 2. 페널티가 시작되는 임계값(threshold)을 계산합니다. (전체 가동 범위의 20%를 버퍼로 설정)
        #    예: 범위가 [-1, 1]이면, 총 범위는 2.0이고, 버퍼는 0.4가 됩니다.
        #    따라서 하한 임계값은 -0.6, 상한 임계값은 0.6이 됩니다.
        joint_range = upper_limits - lower_limits
        upper_threshold = upper_limits * 0.75
        lower_threshold = lower_limits * 0.75

        # 3. 각 관절이 임계값을 얼마나 초과했는지 계산합니다.
        #    relu 함수를 사용하여 임계값 내에 있는 경우는 0으로 처리합니다.
        #    - 상한 임계값 초과량: current_pos가 upper_threshold보다 클 때만 양수가 됨
        #    - 하한 임계값 초과량: current_pos가 lower_threshold보다 작을 때만 양수가 됨
        violation_upper = torch.nn.functional.relu(current_joint_pos - upper_threshold)
        violation_lower = torch.nn.functional.relu(lower_threshold - current_joint_pos)

        # 4. 상한과 하한 초과량을 합산하고 제곱하여, 한계에 가까워질수록 페널티를 기하급수적으로 증가시킵니다.
        #    그 후, 각 환경(env)에 대해 모든 관절의 페널티 값을 합산합니다.
        penalty_joint_limit_proximity = torch.sum((violation_upper + violation_lower)**2, dim=1)
        # --- 로직 끝 ---


        # +++ 착지 직전 속도 (Pre-contact Velocity) 페널티 구현 +++
        # 핵심 아이디어: 충격의 원인인 '착지 직전의 빠른 다리 속도'를 직접 제어합니다.
        # 지면에 닿기 직전 발의 수직 속도가 낮으면, 충격력이 자연스럽게 줄어듭니다.

        # 1. 스윙 중인 발을 식별합니다. (지면에 닿지 않은 발)
        # is_swinging = ~foot_contact

        # 2. '착지 임박' 상태를 정의합니다. (스윙 중 + 지면과 매우 가까운 상태)
        #    참고: 평평한 지형을 가정하여 발의 Z좌표를 직접 사용합니다.
        #    거친 지형에서는 `_height_scanner` 등을 통해 얻은 실제 지형 높이를 빼주어야 합니다.
        pre_contact_height_threshold = 0.075  # 4cm 임계값
        # 월드 좌표계 기준 발의 높이
        foot_height_in_world = foot_pos[:, :, 2]
        is_near_ground = foot_height_in_world < (pre_contact_height_threshold)
        is_pre_contact = is_near_ground
 
        # 3. '착지 임박' 상태인 발의 수직(Z) 속도를 가져옵니다.
        foot_vel_z = foot_velocities[:, :, 2]

        # 4. 페널티를 계산합니다.
        #    - 하강 속도(음수)에만 페널티를 부과하기 위해 clamp(max=0.0) 사용
        #    - 속도를 제곱하여 빠른 속도에 더 큰 페널티 부과
        downward_vel_in_pre_contact = torch.square(torch.clamp(foot_vel_z, max=0.0))
        penalty_pre_contact_velocity = torch.sum(downward_vel_in_pre_contact, dim=1)
        # --- 로직 끝 ---
                
        # =======================================================================================================
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # print("torque", self._robot.data.applied_torque)
        # action rate
        action_rate = torch.sum(torch.square(self._processed_actions - self._previous_actions), dim=1)
        self._previous_actions = self._processed_actions.clone()
        # print("action_rate : ", action_rate[self.observe_envs])

        # =======================================================================================================

        # +++ '균형 회복' 상태라고 가정 (예: push 직후 0.5초간) +++
        # 베이스 속도 추종 보상/페널티 변조

        rewards = {
            "reward_lin_vel": lin_vel_error_mapped * self.rew_scales["reward_lin_vel"],
            "reward_yaw_rate": yaw_rate_error_mapped * self.rew_scales["reward_yaw_rate"],
            "reward_support_polygon": reward_support_polygon * self.rew_scales["reward_support_polygon"],
        }
        penalties = {
            "penalty_z_vel": z_vel_error * self.rew_scales["penalty_z_vel"],
            "penalty_ang_vel": ang_vel_error * self.rew_scales["penalty_ang_vel"],
            "penalty_joint_torque": joint_torques * self.rew_scales["penalty_joint_torque"],
            "penalty_action_rate": action_rate * self.rew_scales["penalty_action_rate"],
            "penalty_swing_stance_phase": penalty_swing_stance_phase * self.rew_scales["penalty_swing_stance_phase"],
            "penalty_undesired_contact": calf_contact_err * self.rew_scales["penalty_undesired_contact"],
            "penalty_flat_orientation": flat_orientation * self.rew_scales["penalty_flat_orientation"],
            "penalty_num_contact_feet_err": num_contact_feet_err * self.rew_scales["penalty_num_contact_feet_err"],
            "penalty_base_height": base_height_err * self.rew_scales["penalty_base_height"],
            "penalty_slip":slip_err * self.rew_scales["penalty_slip"],
            "penalty_joint_accel": joint_accel * self.rew_scales["penalty_joint_accel"],
            "penalty_joint_vel": joint_vel * self.rew_scales["penalty_joint_vel"],
            "penalty_default_grf": penalty_default_grf * self.rew_scales["penalty_default_grf"],

            "penalty_short_swing": penalty_short_swing * self.rew_scales["penalty_short_swing"],
            "penalty_trot_pitch": trot_pitch_err * self.rew_scales["penalty_trot_pitch"],
            "penalty_pre_contact_velocity": penalty_pre_contact_velocity * self.rew_scales["penalty_pre_contact_velocity"],

            "penalty_default_pos_standing": penalty_default_pos_standing * self.rew_scales["penalty_default_pos_standing"],
            "penalty_HR_pos_err": HR_pos_err * self.rew_scales["penalty_HR_pos_err"],
            "penalty_HP_pos_err": HP_pos_err * self.rew_scales["penalty_HP_pos_err"],
            "penalty_KNP_pos_err": KNP_pos_err * self.rew_scales["penalty_KNP_pos_err"],
            "penalty_foot_height_err": foot_height_err * self.rew_scales["penalty_foot_height_err"],
            "penalty_joint_limit_proximity": penalty_joint_limit_proximity * self.rew_scales["penalty_joint_limit_proximity"],

        }

        tot_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        tot_penalty = torch.sum(torch.stack(list(penalties.values())), dim=0)
        reward = tot_reward + 0.5 * tot_penalty
        reward = torch.clamp(reward, min=0.0)

        # Store total reward for plotting
        all_rewards_dict = {**rewards, **penalties}
        self._reward_container = {
                key: value[self.observe_envs] / self.step_dt for key, value in all_rewards_dict.items()
            }
        
        
        self._reward_container["total_reward"] = reward[self.observe_envs] / self.step_dt 
        # --- [Patch: Reward Stats Update + Logging] ------------------------
        # 러닝 통계 업데이트
        self._update_reward_stats()

        # 매 스텝 스냅샷 생성
        stats = self._get_reward_stats_snapshot()
        self._rew_stats_last = stats  # 캐시 갱신

        # extras["log"]에 항상 동일 키를 채움
        if "log" not in self.extras:
            self.extras["log"] = dict()

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        for key, value in penalties.items():
            self._episode_sums[key] -= value

        self.plot_data()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        thigh_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._thigh_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        # +++ 추가된 종료 조건: 관절 위치 제한 초과 +++
        # 1. 현재 모든 관절의 위치를 가져옵니다.
        current_joint_pos = self._robot.data.joint_pos
        
        # 2. USD에서 로드된 관절 위치 제한 값을 가져옵니다. (shape: [num_envs, num_joints, 2])
        joint_limits = self._robot.data.joint_pos_limits
        # print("joint_pos_limits", self._robot.data.joint_pos_limits)
        
        # 3. 하한(lower)과 상한(upper)을 분리합니다.
        lower_limits = joint_limits[..., 0] * 1.25
        upper_limits = joint_limits[..., 1] * 1.25
        
        # 4. 현재 위치가 하한보다 작거나 상한보다 큰 경우를 확인합니다.
        is_out_of_limits = (current_joint_pos < lower_limits) | (current_joint_pos > upper_limits)
        # print("is_out_of_limits", is_out_of_limits)
        
        # 5. 각 환경(env)별로 단 하나의 관절이라도 제한을 초과했는지 확인합니다.
        joint_limit_violation = torch.any(is_out_of_limits, dim=1)
        # +++++++++++++++++++++++++++++++++++++++++++++++

        # 최종 died 조건: 기존 조건 또는 관절 제한 초과
        died = base_contact | thigh_contact | joint_limit_violation


        return died, time_out
    
    # +++ 로봇을 미는 로직을 담은 신규 메서드 +++
    def _push_robots(self):
        """
        일정 주기마다 로봇에 무작위 외력을 가해 밀어냅니다.
        """
        env_ids_to_push = (self._steps_since_push >= self.push_interval).nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids_to_push) > 0:
            current_lin_vel = self._robot.data.root_lin_vel_w[env_ids_to_push]
            current_ang_vel = self._robot.data.root_ang_vel_w[env_ids_to_push]

            push_lin_vel = torch.zeros_like(current_lin_vel)
            push_lin_vel = (torch.rand(len(env_ids_to_push), 3, device=self.device) * 2 - 1) * self.push_strength_lin
            push_ang_vel = (torch.rand_like(current_ang_vel) * 2 - 1) * self.push_strength_ang

            new_lin_vel = current_lin_vel + push_lin_vel
            new_ang_vel = current_ang_vel + push_ang_vel

            new_root_vel = torch.cat((new_lin_vel, new_ang_vel), dim=1)
            self._robot.write_root_velocity_to_sim(new_root_vel, env_ids_to_push)
            
            self._steps_since_push[env_ids_to_push] = 0
            self._recovery_time_counter[env_ids_to_push] = 0.0


    def set_cmd(self,env_ids):
        self.commands_x[env_ids]   = (self.cfg.command_minus_x_range + (self.cfg.command_plus_x_range - self.cfg.command_minus_x_range) * torch.rand((len(env_ids),1), device=self.device)).squeeze()
        self.commands_y[env_ids]   = (self.cfg.command_minus_y_range + (self.cfg.command_plus_y_range - self.cfg.command_minus_y_range) * torch.rand((len(env_ids),1), device=self.device)).squeeze()
        self.commands_yaw[env_ids] = (self.cfg.command_minus_yaw_range + (self.cfg.command_plus_yaw_range - self.cfg.command_minus_yaw_range) * torch.rand((len(env_ids),1), device=self.device)).squeeze()
        
        self.commands_x[env_ids] = torch.where(torch.abs(self.commands_x[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_x[env_ids])
        self.commands_y[env_ids] = torch.where(torch.abs(self.commands_y[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_y[env_ids])
        self.commands_yaw[env_ids] = torch.where(torch.abs(self.commands_yaw[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_yaw[env_ids])


        no_command_env_ids = env_ids[(env_ids >= self.stand_env_range[0]) & (env_ids <= self.stand_env_range[1])]
        self.commands_x[no_command_env_ids] = 0.
        self.commands_y[no_command_env_ids] = 0.
        self.commands_yaw[no_command_env_ids] = 0. 


    
        # only_plus_x = env_ids[(env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])]
        # self.commands_x[only_plus_x] = self.cfg.command_plus_x_range* torch.ones((len(only_plus_x),1), device=self.device).squeeze()
        # self.commands_y[only_plus_x] = 0.
        # self.commands_yaw[only_plus_x] = 0. 

        only_plus_x = env_ids[(env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])]
        self.commands_x[only_plus_x] = (self.cfg.command_plus_x_range * torch.rand((len(only_plus_x), 1), device=self.device)).squeeze()
        self.commands_y[only_plus_x] = 0.
        self.commands_yaw[only_plus_x] = 0.

        only_minus_x = env_ids[(env_ids >= self.only_minus_x_envs_range[0]) & (env_ids <= self.only_minus_x_envs_range[1])]
        self.commands_x[only_minus_x] = self.cfg.command_minus_x_range * torch.ones((len(only_minus_x), 1), device=self.device).squeeze()
        self.commands_y[only_minus_x] = 0.
        self.commands_yaw[only_minus_x] = 0. 

        only_plus_y = env_ids[(env_ids >= self.only_plus_y_envs_range[0]) & (env_ids <= self.only_plus_y_envs_range[1])]
        self.commands_x[only_plus_y] = 0.
        self.commands_y[only_plus_y] = self.cfg.command_plus_y_range 
        self.commands_yaw[only_plus_y] = 0. 
        only_minus_y = env_ids[(env_ids >= self.only_minus_y_envs_range[0]) & (env_ids <= self.only_minus_y_envs_range[1])]
        self.commands_x[only_minus_y] = 0.
        self.commands_y[only_minus_y] = self.cfg.command_minus_y_range 
        self.commands_yaw[only_minus_y] = 0. 

        only_plus_yaw = env_ids[(env_ids >= self.only_plus_yaw_envs_range[0]) & (env_ids <= self.only_plus_yaw_envs_range[1])]
        self.commands_x[only_plus_yaw] = 0.
        self.commands_y[only_plus_yaw] = 0.
        self.commands_yaw[only_plus_yaw] = self.cfg.command_plus_yaw_range
        only_minus_yaw = env_ids[(env_ids >= self.only_minus_yaw_envs_range[0]) & (env_ids <= self.only_minus_yaw_envs_range[1])]
        self.commands_x[only_minus_yaw] = 0.
        self.commands_y[only_minus_yaw] = 0.
        self.commands_yaw[only_minus_yaw] = self.cfg.command_minus_yaw_range


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._recovery_time_counter[env_ids] = self._recovery_duration

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        if len(env_ids)>0:
            self.set_cmd(env_ids)


        # if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
        #     self._update_terrain_level(env_ids)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]  # 기존 코드
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # if hasattr(self, "_env_origins_cache") and self._env_origins_cache is not None:
        #     default_root_state[:, :3] += self._env_origins_cache[env_ids]
        # else:
        #     default_root_state[:, :3] += self.scene.env_origins[env_ids]  # 기존 코드
        # # 1. 위치 랜덤화 (Position Randomization)
        # 기본 관절 위치에 0.925 ~ 1.075 사이의 랜덤 스케일 적용
        # torch_rand_float 함수를 torch.rand로 대체합니다.
        # 공식: torch.rand(shape) * (max - min) + min
        positions_offset = (
            torch.rand((len(env_ids), self._robot.num_joints), device=self.device) 
            * (1.075 - 0.925) + 0.925
        )
        randomized_joint_pos = joint_pos * positions_offset

        # 2. 속도 랜덤화 (Velocity Randomization)
        # 관절 속도를 -0.1 ~ 0.1 사이의 랜덤 값으로 설정
        randomized_joint_vel = (
            torch.rand((len(env_ids), self._robot.num_joints), device=self.device)
            * (0.1 - (-0.1)) + (-0.1) # 즉, * 0.2 - 0.1
        )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(randomized_joint_pos, randomized_joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
