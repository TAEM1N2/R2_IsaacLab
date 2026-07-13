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
