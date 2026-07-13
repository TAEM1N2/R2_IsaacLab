# PongBot R2 PaperBarrier Rough Task

`PongBot-R2-PaperBarrier-Rough-v0`는 기존 implicit/rough/frontier task를 변경하지 않고,
barrier-based style reward 논문의 quadruped 방법을 R2에 적용하기 위한 독립 학습 task다.

## 실행

```bash
python scripts/rsl_rl/train.py \
  --task PongBot-R2-PaperBarrier-Rough-v0 \
  --headless
```

기본 설정은 400개 환경, 환경당 400 step(총 160,000 sample), 100 Hz 제어,
4초 episode, 최대 10,000 iteration이다. 12개 관절의 action scale은 0.1 rad이다.

## 학습 카메라

`--camera`를 사용하면 이미지 한 장씩만 렌더링하면서 close와 overview 시점을 번갈아 저장한다.
close 시점은 최대 8개의 대표 환경을 순환하며 로봇 root를 따라가고, overview 시점은 현재
모든 로봇 위치의 2~98 percentile 범위를 기준으로 활성 terrain row를 자동 추적한다. 카메라와
저장 이미지 수를 늘리지 않으므로 고정 시점 촬영과 렌더 횟수가 같다.

```bash
python scripts/rsl_rl/train.py \
  --task PongBot-R2-PaperBarrier-Rough-v0 \
  --headless \
  --camera \
  --camera_interval 2000
```

파일명에는 `close_env_000` 또는 `overview`가 포함된다. 기존 고정 viewport가 필요하면
`--camera_view_mode fixed`를 추가한다. 카메라 설정은 환경 생성 시 적용되므로 실행 중인
학습에는 반영되지 않으며 다음 실행부터 사용된다.

## 학습 구조

- actor 입력: 141차원 R2 proprioception + 3차원 command + estimator의 11차원 추정값
- estimator 출력: body linear velocity 3, terrain-relative foot height 4, contact probability 4
- critic 입력: actor와 같은 proprioception/command에 실제 11차원 privileged target을 결합
- critic은 standard reward와 barrier reward용으로 각각 하나씩 사용한다.
- 두 reward는 별도 GAE와 advantage 정규화를 거친 뒤 actor에서 `0.5 / 0.5`로 결합한다.
- actor, standard critic, barrier critic, estimator는 optimizer와 gradient clipping을 공유하지 않는다.
- 두 critic은 raw value를 예측하되 각 rollout return 표준편차로 value loss와 value clip 폭을
  독립 정규화한다.
- rollout storage에는 중복 observation history를 저장하지 않는다.

## R2 자동 보정

runner는 시작 시 모든 환경을 terrain level 0의 canonical root/joint state에 놓는다.
0.2초 시점의 형상을 고정하고, zero action 100 control step 동안 stance force를 모아
다음 값을 측정한다. 초기 policy가 없는 상태의 낙상 자세가 형상 기준을 오염시키지 않도록
geometry와 force 측정 시점을 분리한다.

- body-frame nominal foot positions
- front/rear thigh-height 중심과 앞뒤 높이 차
- HOUND 기준에 대한 R2 길이 scale
- stance force 기반 contact threshold

측정 결과는 checkpoint의 `calibration` 필드에 저장되고 재개 시 복원된다. 관절 각도,
관절 속도, action bound처럼 단위가 로봇 길이와 무관한 barrier는 scale하지 않는다.

## Rough-trot terrain

이 task는 논문의 quadruped rough-terrain trot policy만 대상으로 한다. 비율은 flat 25%,
bumpy 25%, slope up/down 각 12.5%, stairs up/down 각 12.5%다. 논문이 공개한 최대값인
6 cm bump, 27도 slope, 20 cm stair를 사용한다. 요청 범위 밖인 34.5 cm discrete step과
별도 high-step/gallop task는 포함하지 않는다.

논문은 rough-trot policy에 대한 난도 curriculum을 명시하지 않았으므로 기존의 임의
iteration ramp를 제거했다. 시작부터 0~100% terrain row를 400개 병렬 환경에 분산해 사용하며,
`Curriculum/terrain_difficulty`는 고정값 1.0이다.

## 재현 가정

논문은 rough terrain 종류별 표본 비율을 공개하지 않았다. 위의 25/25/12.5% 비율은 flat과
bumpy를 충분히 유지하면서 상승·하강 slope/stairs를 균형 있게 배분한 재현 가정이다. 또한
optimizer 분리와 critic value-scale 처리는 논문에 명시된 설정이 아니라, 두 reward scale의
간섭을 막기 위해 이번 실험에서 명시적으로 요청된 PPO 안정화 확장이다. actor의 update는
논문대로 개별 정규화한 standard/barrier advantage를 0.5/0.5로 결합한다.

논문에 수치가 공개되지 않은 nominal Cartesian foot-position 항과 front/rear
height-difference 항은 보수적으로 각각 1.0을 사용한다. 이는 논문의 공개값이 아니라
명시적인 reproduction assumption이다.

논문의 raw torque 계수는 로봇 effort scale이 달라 R2에 그대로 대응하지 않는다. Standard
torque regularization은 R2 actuator limit(HR/HP 120 Nm, knee 320 Nm)으로 torque를 나눈 뒤
제곱합에 weight 1.0을 적용한다. 따라서 관절별 실제 사용 가능 torque 비율을 같은 기준으로
제한한다.

기존 coupled-optimizer checkpoint의 model weight는 읽을 수 있지만 optimizer state는 새 구조와
호환되지 않는다. 이 변경의 본 학습 비교는 checkpoint 재개가 아니라 scratch run으로 수행한다.

주요 TensorBoard 항목은 `Reward/paper_standard_per_step`,
`Reward/paper_barrier_per_step`, `Loss/*estimator_loss`,
`Diagnostics/*violation`, `Diagnostics/illegal_contact`,
`Curriculum/terrain_difficulty`다.

## 학습 성공 기준과 로그 해석

이 task의 목표는 4초 동안 단순히 넘어지지 않는 것이 아니다. 외부 지형 관측 없이 명령
속도를 추종하면서, rough terrain에서 trot·foot clearance·body/joint 동작 범위를 지키고
BODY/THIGH 접촉에 의한 조기 종료 없이 episode를 끝까지 완료하는 것이 목표다. Episode의
최대 길이는 400 step이므로 평균 길이 300은 최대의 75%이지만, 길이만으로는 정지 policy와
정상 추종 policy를 구분할 수 없다.

다음 실행부터 터미널에는 기본 요약과 핵심 진단이 지표별 한 줄로 출력된다.

```text
[PAPER_BARRIER]
  iteration                 : ...
  total_steps               : ...
  iteration_time_s          : ...
  standard_reward_per_step  : ...
  barrier_reward_per_step   : ...
  mean_episode_reward       : ...
  mean_episode_length       : ...
  terrain_difficulty        : ...
  estimator_loss            : ...
  policy_loss               : ...
[PAPER_BARRIER_DIAG]
  timeout_rate              : ...
  tracking_success_rate     : ...
  lin_vel_xy_error          : ...
  body_contact_rate         : ...
  gait_violation_rate       : ...
  clearance_violation_rate  : ...
  actual_mean_terrain_level : ...
  mean_kl                   : ...
  learning_rate             : ...
```

TensorBoard에서는 다음 순서로 판단한다.

1. `Episode/timeout_rate`: 1에 가까워져야 한다. 4초를 채운 episode 비율이다.
2. `Episode/early_termination_rate`: 0에 가까워져야 한다. BODY/THIGH 접촉 종료 비율이다.
3. `Episode/tracking_success_rate`: timeout이면서 episode 평균 XY 성분 최대 절대오차와 yaw 오차가
   각각 0.4 이하인 비율이다. 논문의 축별 barrier 범위와 같은 기준이다.
4. `Tracking/moving_lin_vel_xy_error`: 이동 명령 환경의 XY 속도 오차이며 낮아져야 한다.
5. `Tracking/stationary_xy_speed`: 정지 명령 환경의 미끄러짐·떨림 속도이며 0에 가까워야 한다.
6. `Contact/body_rate`, `Contact/thigh_rate`: rollout 전체 contact 비율이며 0에 가까워야 한다.
7. `TerrainSuccess/*_timeout_rate`, `TerrainSuccess/*_tracking_rate`: terrain 종류별 생존·추종 성공률이다.
8. `Terrain/actual_mean_level`: full-range terrain에서 실제 표본된 평균 row다.
9. `Diagnostics/*violation`, `ConstraintViolation/*`: gait, clearance, joint, body height,
   base motion 제약 위반율이며 낮아져야 한다.
10. `BarrierTerm/*`: barrier 총합이 악화될 때 어떤 제약 항이 원인인지 찾는 데 사용한다.
11. `StandardPenalty/*`, `Motion/*`: slip, torque, action rate/acceleration과 실제 RMS 움직임을
    함께 보고 sim-to-real regularization이 과도하거나 약한지 판단한다.
12. `Loss/mean_kl`, `Loss/clip_fraction`, `Loss/*explained_variance`,
    `Loss/*gradient_norm_pre_clip`, `Loss/*value_scale`, `Loss/learning_rate`: PPO 최적화 안정성을
    판단한다. KL은 설정값 0.01 주변, explained variance는 1 방향이 바람직하며 actor와 두 critic의
    gradient norm이 독립적으로 안정되는지 확인한다.

기존 `Diagnostics/*`는 마지막 simulation step만 기록했지만, 새 로그는 환경 400개 × 400 step
전체를 누적한 rollout 평균이다. `Train/mean_episode_length`는 최근 100개 episode의 이동 평균으로
유지하고, `Episode/mean_length_current`는 해당 iteration에서 끝난 모든 episode의 평균을 나타낸다.
초기 episode 길이를 무작위화하므로 iteration 0의 episode 통계는 참고만 하고 이후 추세를 본다.
