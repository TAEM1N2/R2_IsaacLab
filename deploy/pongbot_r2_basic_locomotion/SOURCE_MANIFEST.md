# Source manifest

이 배포본은 원본 커밋 `65d867a`의 기본 Flat/Rough 학습 경로를 기준으로 구성했다.

## 환경 패키지

- `exts/pongbot_r2/pongbot_r2/assets/config/pongbot_r2.py`
  - R2 articulation, actuator, default pose와 torque limit
- `exts/pongbot_r2/pongbot_r2/assets/usd/pongbot_r2/ktm/`
  - R2 USD composition, configuration layer와 STL mesh
- `exts/pongbot_r2/pongbot_r2/tasks/locomotion/cfg/pongbot_r2/normal_base_env_cfg.py`
  - 공통 ManagerBasedRLEnv 구성
- `exts/pongbot_r2/pongbot_r2/tasks/locomotion/cfg/pongbot_r2/terrains_cfg.py`
  - Flat/Rough에서 사용하는 rough train/play terrain
- `exts/pongbot_r2/pongbot_r2/tasks/locomotion/robots/`
  - Flat/Rough robot-specific override와 Gym 등록
- `exts/pongbot_r2/pongbot_r2/tasks/locomotion/mdp/`
  - 위 환경에서 참조하는 공유 MDP term

공유 MDP 파일은 함수 단위로 잘라내지 않았다. reward와 event 구현 내부 helper 사이의 결합을 보존하고,
원본 학습 의미가 달라지는 것을 막기 위해 필요한 모듈 단위로 복사했다.

## 학습 패키지

- `exts/pongbot_r2/pongbot_r2/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
  - Flat/Rough runner 설정만 유지
- `exts/pongbot_r2/pongbot_r2/utils/wrappers/rsl_rl/`
  - workspace RSL-RL 설정 및 export helper
- `rsl_rl/rsl_rl/`
  - 현재 PPO 파일의 import 폐쇄 집합

`ppo.py`와 `on_policy_runner.py`는 하나의 파일 안에서 여러 알고리즘 구현을 함께 참조한다. 런타임
import 실패를 피하기 위해 해당 파일이 직접 import하는 작은 module들은 보존했지만, PaperBarrier
전용 algorithm, module, runner와 storage는 제외했다.

## 실행 진입점

- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`
- `scripts/rsl_rl/export_onnx.py`
- `scripts/rsl_rl/cli_args.py`

진입점은 Flat/Rough에서 쓰는 `OnPolicyRunner` 경로만 남긴 최소 버전이다. 원본의 Phase2,
PaperBarrier, ROS2 stream, latent 분석, 수동 조작기 코드는 포함하지 않았고, Flat/Rough의 환경
생성, wrapper action clip, checkpoint 처리와 학습·추론 루프는 유지했다.
