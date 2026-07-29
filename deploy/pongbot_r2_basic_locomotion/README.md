# PongBot R2 기본 보행 학습 소스

이 폴더는 원본 워크스페이스에서 기본 보행 학습에 필요한 소스만 분리한 배포본이다.
평지와 일반 험지 학습 및 평가 task를 제공하며, 로그·체크포인트·실험 산출물은 포함하지 않는다.

## 포함 task

- `PongBot-R2-Blind-Flat-v0`
- `PongBot-R2-Blind-Flat-Play-v0`
- `PongBot-R2-Blind-Rough-v0`
- `PongBot-R2-Blind-Rough-Play-v0`

`-Play-v0`는 학습 task가 아니라 작은 환경 수를 사용하는 평가 설정이다. 관측 corruption,
push event와 base mass randomization은 비활성화되지만, 나머지 물리 randomization은 원본
평가 설정대로 유지된다.

## 중요한 모델 계약

이 배포본은 estimator가 전혀 없는 순수 MLP baseline이 아니다. 현재 검증된 기본 보행 구현의 계약을
그대로 보존하며 다음 입력 경로를 사용한다.

- policy observation: 48차원
- observation history: 10 frame × 48차원
- MLP encoder 입력: 480차원 history
- MLP encoder 출력: 3차원 body linear velocity 추정값
- velocity command: 3차원
- actor 입력: 현재 관측 48 + history 480 + 추정값 3 + command 3 = 534차원
- action 출력: 12개 관절 position residual
- action scale: 0.25 rad, default joint position 기준

학습 환경은 physics timestep 0.002초, decimation 5, policy 주기 100 Hz, episode 20초를 사용한다.
관측 순서, history 길이, action 순서와 scale을 바꾸면 기존 checkpoint와 호환되지 않는다.

## 외부 요구 사항

이 저장본은 NVIDIA Isaac Sim 자체를 포함하지 않는다. 다음 환경이 먼저 준비되어 있어야 한다.

- 원본과 호환되는 Isaac Sim 및 Isaac Lab
- `isaaclab`, `isaaclab_tasks`, `isaaclab_assets`, `isaaclab_rl`
- Python 3.10

Isaac Lab Python 환경에서 추가 Python 의존성을 설치한다.

```bash
python -m pip install -r requirements.txt
python -m pip install -e rsl_rl
python -m pip install -e exts/pongbot_r2
```

## 학습

이 폴더의 루트에서 실행한다.

```bash
python scripts/rsl_rl/train.py \
  --task PongBot-R2-Blind-Flat-v0 \
  --headless
```

험지 학습:

```bash
python scripts/rsl_rl/train.py \
  --task PongBot-R2-Blind-Rough-v0 \
  --headless
```

자원에 맞춰 병렬 환경 수와 iteration을 덮어쓸 수 있다.

```bash
python scripts/rsl_rl/train.py \
  --task PongBot-R2-Blind-Rough-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 10000 \
  --run_name rough_baseline
```

결과는 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/` 아래에 저장된다.

## 평가

```bash
python scripts/rsl_rl/play.py \
  --task PongBot-R2-Blind-Flat-Play-v0 \
  --checkpoint /absolute/path/to/model_XXXX.pt
```

험지 평가는 `PongBot-R2-Blind-Rough-Play-v0`를 사용한다.

## ONNX export

기본 보행 정책은 actor와 encoder가 분리되어 있으므로 두 ONNX 파일을 함께 배포해야 한다.

```bash
python scripts/rsl_rl/export_onnx.py \
  --task PongBot-R2-Blind-Flat-Play-v0 \
  --checkpoint /absolute/path/to/model_XXXX.pt \
  --headless
```

checkpoint 옆의 `exported/` 폴더에 다음 파일이 생성된다.

- `nominal_actor.onnx`
- `encoder.onnx`

## 배포본 범위

포함:

- R2 robot configuration과 해당 USD/mesh 자산
- Flat/Rough 환경, command, observation, reward, event, termination, curriculum
- 기본 PPO, actor-critic, 3차원 MLP encoder와 rollout runner
- train, play, ONNX export 스크립트

제외:

- IMU, Mass, Implicit/VAE, TCP, Phase2, joint-fault task
- PaperBarrier task와 전용 dual-critic runner
- stair 전용 task 및 terrain
- 학습 로그, checkpoint, TensorBoard 파일, 이미지와 분석 결과

세부 파일 목록과 원본 대응은 `SOURCE_MANIFEST.md`를 참고한다.
