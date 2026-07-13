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

## Terrain과 curriculum

terrain 비율은 flat 20%, bumpy 20%, slope up/down 각 10%, stairs up/down 각 10%,
steps 20%다. iteration 0~500은 전체 난도의 20%까지 사용하고, 500~3,000에서
논문 최대 난도까지 선형 확장한다. reset 시 현재 상한 이하 row를 균등 표본화하며,
성공률 기반 승급·강등은 사용하지 않는다.

## 재현 가정

논문에 수치가 공개되지 않은 nominal Cartesian foot-position 항과 front/rear
height-difference 항은 보수적으로 각각 1.0을 사용한다. 이는 논문의 공개값이 아니라
명시적인 reproduction assumption이다.

주요 TensorBoard 항목은 `Reward/paper_standard_per_step`,
`Reward/paper_barrier_per_step`, `Loss/*estimator_loss`,
`Diagnostics/*violation`, `Diagnostics/illegal_contact`,
`Curriculum/terrain_difficulty`다.
