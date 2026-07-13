# 오류 기록

<details open>
<summary>2026-07-13 16:36 [E007] [테스트] PaperBarrier helper의 AppLauncher 외부 package import 실패</summary>

- 증상: 순수 episode/rollout metric helper mock test에서 `rsl_rl.runner`를 일반 import하자 `ModuleNotFoundError: omni.kit`으로 중단됨.
- 원인: `rsl_rl.runner.__init__`가 기존 `OnPolicyRunner`를 함께 import하며, 해당 모듈은 Isaac Sim AppLauncher가 먼저 생성되어야 하는 `omni.kit`에 의존함.
- 확인: `paper_barrier_runner.py` 파일만 직접 로드한 동일 mock test에서 timeout/early-termination, terrain별 성공률, finite rollout 평균 계산이 모두 통과함.
- 해결: 순수 helper 검증은 파일 직접 로드를 사용하고, 전체 runner 통합 검증은 AppLauncher를 사용하는 실제 train smoke로 수행하도록 구분함.
- 관련 파일: `rsl_rl/rsl_rl/runner/paper_barrier_runner.py`

</details>

<details>
<summary>2026-07-13 15:47 [E006] [IsaacLab] PaperBarrier task 초기화 및 CPU smoke device 불일치</summary>

- 증상: 첫 smoke test는 observation manager 생성 중 `episode_length_buf` 부재로 실패했고, 수정 후 CPU smoke에서는 timeout bootstrap reward와 CUDA value tensor의 device가 달라 실패함.
- 원인: observation shape discovery가 RL episode buffer 생성보다 먼저 실행되며, reward manager의 raw reward tensor는 환경 device에 남아 있었음.
- 확인: 수정 후 CPU 1환경에서 calibration, 400-step rollout, dual GAE/PPO update를 완료하고 `model_0.pt`, `model_1.pt` 및 유한 TensorBoard scalar를 생성함.
- 해결: manager 구성 중 gait phase는 0으로 처리하고, raw reward와 done tensor를 runner device로 명시적으로 이동함.
- 관련 파일: `exts/pongbot_r2/pongbot_r2/tasks/locomotion/mdp/paper_barrier_terms.py`, `rsl_rl/rsl_rl/runner/paper_barrier_runner.py`

</details>

<details>
<summary>2026-07-13 00:45 [E005] [IsaacLab] 4차 obstacle curriculum level 0 고착</summary>

- 증상: 4차 Frontier scratch 학습에서 약 300 iteration 이후 평균 terrain level이 0에 고착되고 recovery reward가 약 566 iteration까지 0으로 유지됨.
- 원인: 초기 policy 실패로 모든 competence 표본이 강등됐고, 완전 접촉 해제 후 0.3m 진행만 보상하는 recovery 조건이 지나치게 희소했음.
- 확인: episode length는 약 1,234로 증가했지만 velocity error는 약 0.29까지 증가했고 obstacle recovery reward는 발생하지 않음.
- 해결: 5차에서 competence/challenge 역할과 level을 별도 보존하고, TIP 수평 충돌을 포함한 단계형·상한형 recovery credit으로 교체함.
- 관련 파일: `exts/pongbot_r2/pongbot_r2/tasks/locomotion/mdp/curriculums.py`, `exts/pongbot_r2/pongbot_r2/tasks/locomotion/mdp/rewards.py`

</details>

<details>
<summary>2026-07-12 21:08 [E003] [IsaacLab] frontier smoke test 로깅 중 NameError</summary>

- 증상: `PongBot-R2-Implicit-Rough-Frontier-v0`의 첫 smoke run이 iteration 로그 기록 시 `NameError: name 'math' is not defined`로 종료됨.
- 원인: frontier metric의 유한값 검사에 `math.isfinite()`를 추가하면서 runner에 `math` import를 누락함.
- 확인: 환경·reward·history·curriculum·PPO 초기화와 첫 rollout은 완료됐고 traceback이 `on_policy_runner.py`의 frontier metric logging 줄을 지목함.
- 해결: `on_policy_runner.py`에 `import math`를 추가하고 동일 smoke test를 재실행함.
- 관련 파일: `rsl_rl/rsl_rl/runner/on_policy_runner.py`

</details>

<details>
<summary>2026-07-10 10:30 [E002] [IsaacLab] implicit rough 학습 중 base contact 증가</summary>

- 증상: rollout 48, gamma 0.995와 pyramid velocity/gait curriculum을 동시에 적용한 run에서 base contact가 증가하고 평균 episode 길이와 terrain level이 감소함.
- 원인: 단일 원인은 확정하지 못했으며, rollout·discount·gait command 분포를 동시에 변경하여 영향 분리가 불가능했음.
- 확인: iteration 0~129 구간에서 평균 episode 길이가 최대 약 1,398 step에서 약 126 step으로 감소하고 terrain level이 약 0.39에서 0.0006으로 감소함.
- 해결: rollout 24와 gamma 0.99를 복원하고 pyramid gait curriculum을 비활성화하며 velocity curriculum만 유지하는 ablation으로 전환함.
- 관련 파일: `exts/pongbot_r2/pongbot_r2/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`, `exts/pongbot_r2/pongbot_r2/tasks/locomotion/cfg/pongbot_r2_implicit/implicit_base_env_cfg.py`

</details>

<details>
<summary>2026-07-10 10:23 [E001] [IsaacLab] implicit rough runner import 실패</summary>

- 증상: `PongBot_R2ImplicitRoughPyramidPPORunnerCfg`를 import할 때 `AttributeError`가 발생하여 학습이 시작되지 않음.
- 원인: `@configclass` 처리 후 부모 설정의 `algorithm`을 클래스 속성으로 접근함.
- 확인: traceback이 `PongBot_R2ImplicitRoughPPORunnerCfg.algorithm.replace(gamma=0.995)`를 지목했으며, 수정 후 Isaac Sim package import와 설정 생성 결과가 rough `48/0.995`, flat·stair `24/0.99`로 확인됨.
- 해결: 인스턴스 초기화 이후 `__post_init__()`에서 `self.algorithm.gamma = 0.995`를 적용함.
- 관련 파일: `exts/pongbot_r2/pongbot_r2/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

</details>
