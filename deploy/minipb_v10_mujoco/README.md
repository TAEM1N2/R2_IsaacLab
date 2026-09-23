# MiniPB v10 전용 MuJoCo 실행기

기존 ROS 2 실행기, XML, 정책 및 Downloads 번들을 수정하지 않는 독립 Python 실행기입니다.
`controller.py`는 v10 번들의 `minipb_v10_deployment_example.py`를 그대로 복사한
상태 관리 코드입니다. 모델은 `export_manifest.json`, 제어 설정은
`deployment_config.json`에서 읽습니다. ONNX 내부의 CPG/대칭 투영/target limit와
외부의 35-tick history, previous actor action, phase 갱신 순서를 유지합니다.

## GUI 실행

```bash
cd /home/rclab/r2_isaaclab_trone
/home/rclab/miniconda3/envs/env_isaaclab/bin/python deploy/minipb_v10_mujoco/run.py \
  --bundle /home/rclab/Downloads/minipb_hybrid_v10_ep200_nuc_20260908 \
  --command 0.5 0 0
```

3초간 기본 자세 PD로 안정화한 후 전진합니다. 기본 실행 시간은 무제한입니다.
창을 닫거나 Ctrl+C로 종료합니다. 창에 포커스를 둔 상태에서:

| 키 | 동작 |
|---|---|
| W / S | 전진 / 후진 (0.5 m/s) |
| A / D | 좌 / 우 이동 (0.5 m/s) |
| Q / E | 좌 / 우 회전 (0.5 rad/s) |
| Space | 명령을 0으로 설정 (MLP는 계속 활성) |
| R | 시뮬레이션, 위상, history 리셋 후 3초 안정화; 현재 명령 유지 |

정지 명령으로 시작하려면 `--command 0 0 0`을 지정합니다.
학습 및 기본 검증 명령은 전진 0.5 m/s이며 다른 방향의 보행 성능은 별도 확인이 필요합니다.

## 화면 없이 검증

```bash
/home/rclab/miniconda3/envs/env_isaaclab/bin/python deploy/minipb_v10_mujoco/run.py \
  --headless --duration 23 --csv /tmp/minipb_v10_forward.csv
```

`--duration`은 초기 안정화 시간을 포함합니다. GUI는 실시간, headless는 최대 속도로 실행합니다.
CSV는 정책 구간의 시간, 높이, 월드 속도, 명령, 관절 위치와 target을 기록합니다.
R 리셋 시 같은 CSV에 시간이 0부터 다시 기록됩니다.

기본 XML은 `/home/rclab/minipb_ws/src/rclab_minipb_sim/models/minipb_ver3_mujoco.xml`이며
기존 mesh를 읽기만 합니다. `--model /path/to/model.xml`로 변경할 수 있습니다.
관절/모터 이름으로 정책 순서를 매핑하므로 XML 내부 순서에 의존하지 않습니다.
현재 환경의 `env_isaaclab`에 MuJoCo 3.9.0, ONNX Runtime 1.23.2가 설치되어 있습니다.

## 기존 정책과의 공존

기존 정책은 기존 ROS 2 명령을 그대로 사용합니다. 예를 들어 기존 기본 정책 GUI:

```bash
source /opt/ros/humble/setup.bash
source /home/rclab/minipb_ws/install/setup.bash
ros2 launch rclab_minipb_sim minipb_sim.launch.py \
  headless:=false real_time:=true duration_sec:=0.0
```

이 Python 실행기의 `--bundle` 변경은 동일한 v10 입력/출력 및 설정 형식의 번들에만 사용합니다.
기존 actor 단독 정책은 기존 실행기를 사용합니다.

## 검증 및 물리 범위

- 번들 ONNX test vectors 통과 (허용 오차 1e-5).
- 초기 3초 + 전진 정책 20초: 평균 월드 vx 0.498 m/s, 평균 높이 0.306 m, 최소 높이 0.246 m.
- 초기 3초 + 정지 명령 5초: 수치 오류 없이 완료, 평균 월드 vx 0.015 m/s.
- GUI 화면 및 키 입력은 실제 창에서 별도 확인해야 합니다.

500 Hz 물리/PD, 100 Hz 추론, PD 20/1, 토크 제한 21 Nm를 적용합니다.
기존 평지 XML의 접촉·관성 모델을 사용하며 Isaac Lab의 terrain randomization,
actuator delay, 19 rad/s의 별도 actuator velocity-limit 동작은 재현하지 않습니다.
ONNX 내부 target slew limit는 그대로 적용됩니다. 따라서 Isaac Lab 물리의 완전한 복제는 아닙니다.
