# AGENTS.md

## 학습 목표

4족로봇이 목표 선속도와 yaw command를 안정적으로 추종하며, 규칙적으로 발을 충분히 들어 올리는 자연스러운 보행 policy를 확보한다.

최종 목표는 기존 설정을 유지하는 것 자체가 아니라, 아래의 변경 금지 범위와 물리적 제약을 준수하면서 실제 보행 성능이 좋은 policy를 만드는 것이다.

### 최우선 목표

* 안정적인 보행
* 정확한 command tracking
* 규칙적이고 자연스러운 gait
* 충분한 foot clearance

### 보조 목표

* 낮은 토크 사용량
* 부드러운 action 변화

보조 목표는 최우선 목표를 훼손하지 않는 범위에서 최적화한다. 토크 사용량이나 action 변화를 줄이기 위해 보행 안정성 또는 command tracking 성능을 희생하지 않는다.

### 반드시 유지해야 하는 성능

* 평지 전진
* 정지
* 회전
* nominal trot 보행

### 허용할 수 없는 현상

* 정지 명령에서 지속적으로 발을 움직이는 현상
* 목표 속도를 무시하고 제자리에서 버티는 현상
* 과도한 joint saturation
* action 또는 observation에서 NaN/Inf가 발생하는 현상
* 특정 지형 성능을 높이기 위해 평지 성능이 크게 감소하는 현상
* 기존 observation/action 인터페이스를 변경하는 행위
* 발을 충분히 들지 않고 지면에 끌면서 걷는 현상
* 비정상적으로 낮은 자세로 걷는 현상
* 과도한 base 흔들림 또는 joint 진동이 발생하는 현상
* 균형 유지에 필요한 순간적인 움직임을 제외하고, 직진 및 정지 시 hip roll 관절이 중립 자세인 0도에서 과도하게 벗어나거나 한쪽 방향의 지속적인 offset에 의존하여 보행하는 현상

평균 reward가 증가했다는 이유만으로 학습 성공으로 판단하지 않는다. 실제 rollout에서 보행 안정성, command tracking, gait 규칙성, foot clearance, base 자세 및 joint saturation을 함께 평가한다.

---

## 학습 코드 자동튜닝 변경 범위

### 변경 가능한 범위

보행 성능 개선을 위해 다음 항목을 변경할 수 있다.

1. PPO 알고리즘 관련 파라미터
2. reward scale 및 reward weight
3. reward term의 추가, 제거 및 계산식 변경
4. domain randomization 범위
5. curriculum 설정
6. command sampling 범위와 분포
7. termination 및 reset 조건
8. policy 및 critic network 구조
9. action scale
10. PD gain
11. 아래의 변경 금지 범위를 침범하지 않는 기타 학습 설정

모든 변경은 안정적이고 자연스러운 보행과 command tracking 개선을 목적으로 해야 한다.

### PD Gain 변경 규칙

PD gain은 보행 성능 개선을 위해 변경할 수 있다. 단, 시뮬레이션에서만 성능이 좋아지는 비현실적인 값을 사용하지 않는다.

PD gain 변경 시 다음 조건을 준수한다.

* 실제 로봇의 모터, 감속기, 모터 드라이버 및 제어 주기에서 구현 가능한 값이어야 한다.
* 기존 실물 로봇에서 사용한 PD gain을 기준으로 합리적인 범위 안에서 탐색한다.
* 지나치게 큰 `Kp` 또는 `Kd`로 강제로 자세를 유지하거나 진동을 억제하지 않는다.
* 작은 position error 또는 velocity error만으로 torque limit에 빈번하게 도달하는 gain을 사용하지 않는다.
* joint oscillation, 고주파 진동, 과도한 충격 또는 반복적인 torque saturation을 발생시키는 설정은 사용하지 않는다.
* 시뮬레이션에서 안정적이더라도 실물 로봇에 적용하기 어려운 gain은 최종 설정으로 채택하지 않는다.
* 관절별 모터 사양과 torque limit이 다르면 관절 그룹별로 적절한 gain을 사용한다.
* PD gain 변경 전후의 값과 변경 근거를 반드시 보고한다.

PD torque는 다음 관계를 기준으로 검토한다.

`torque_PD = Kp * position_error + Kd * velocity_error`

예상 가능한 position error와 velocity error에서 계산되는 torque가 해당 관절의 torque limit을 지속적으로 초과하지 않아야 한다.

적절한 PD gain 범위를 판단할 근거가 부족하면 임의의 큰 범위를 탐색하지 않는다. 기존 실물 로봇 gain과 actuator 사양을 기준으로 보수적인 탐색 범위를 설정한다.

### 변경하면 안 되는 범위

다음 항목은 사용자의 명시적인 승인 없이 수정하지 않는다.

#### Observation

* 구성
* 순서
* 차원
* noise
* randomization

Observation noise와 randomization 값은 실제 로봇 센서 오차를 반영한 값이므로 수정하지 않는다.

#### Action

* 구성
* 순서
* 차원

#### Simulation 및 Control Timing

* control timestep
* physics timestep
* decimation

#### 로봇 모델

* URDF
* USD
* MJCF

#### 로봇 물리 제약조건

* joint limit
* torque limit

NaN, 학습 발산, 학습 실패 또는 낮은 reward가 발생하더라도 금지된 항목을 임의로 변경하여 문제를 숨기지 않는다.

---

# Agent Workflow Defaults

## Edit Approval Policy

코드 수정은 다음 순서로 진행한다.

1. 현재 Git 상태와 미커밋 변경사항을 확인한다.
2. 변경 목적, 대상 파일 및 예상 diff 요약을 사용자에게 보고한다.
3. 사용자의 명시적인 승인을 기다린다.
4. 승인 후 실제 코드 수정 전에 현재 상태를 원본 보존용 Git commit으로 저장한다.
5. 코드를 수정하고 검증한다.
6. 변경 내용과 검증 결과를 보고한다.
7. 최종 수정 결과를 Git commit에 반영한다.

사용자 승인 전에는 코드, 설정 파일 및 Git 이력을 변경하지 않는다.

수정 전 원본 보존용 commit을 완료할 수 없다면 코드를 수정하지 않고 그 이유를 보고한다.

이미 존재하는 미커밋 변경사항을 임의로 덮어쓰거나 삭제하지 않는다.

기존 checkpoint, log 및 실험 결과를 덮어쓰지 않는다.

## Code Style

* 신규 코드는 Google 스타일 가이드를 따른다.
* 기존 코드는 수정하는 범위 안에서만 스타일을 개선한다.
* 기능과 무관한 대규모 formatting 또는 refactoring을 수행하지 않는다.
* 프로젝트 또는 사용 중인 프레임워크의 기존 스타일이 명확하면 기존 스타일을 우선한다.
* `README.md`를 포함한 사용자 대상 문서는 기본적으로 한글로 작성한다.
* 동작, 인터페이스, 설정 또는 로직이 변경된 `*.py` 파일은 최상단 헤더의 `@version`을 패치 단위로 올린다.
* 주석, 서식 또는 오탈자만 수정한 경우에는 `@version`을 올리지 않는다.
* 기존에 프로젝트 표준 헤더가 없는 파일에는 사용자 승인 없이 새 헤더 형식을 추가하지 않는다.
* 외부 라이브러리, vendored code 및 generated code에는 파일 헤더와 버전 규칙을 적용하지 않는다.
* 파일 헤더에는 해당 파일의 역할과 주요 기능 또는 구조를 이해할 수 있는 설명을 유지하거나 보강한다.
* 파일 헤더의 `@update` 항목은 최신 2개만 유지하고 오래된 항목은 정리한다.
* 수정 내역의 상세 로그를 파일 헤더 주석에 누적하지 않는다.
* 실제 수정 내용은 Git commit 메시지와 히스토리에 반영한다.
* Git commit 메시지는 한글과 영문을 함께 작성하는 것을 원칙으로 한다.
* 파일 헤더 `@version`은 기본적으로 `0.0.x` 패치 번호만 증가시킨다.
* `0.x.0` minor 버전 증가는 사용자가 명시적으로 요청한 경우에만 수행한다.
* `0.0.x` 패치 번호는 필요한 경우 `999`까지 증가할 수 있다.

## 검증 규칙

* 코드 수정 후 변경 내용과 직접 관련된 최소 검증을 수행한다.
* 수행한 검증 명령어와 결과를 보고한다.
* 실행하지 못한 검증은 실행한 것처럼 표현하지 않고, 실행하지 못한 이유를 명시한다.
* 검증 실패를 숨기기 위해 test, assertion, safety check 또는 error handling을 제거하지 않는다.
* 기존 정상 동작을 깨뜨리는 변경을 완료 상태로 처리하지 않는다.
* 짧은 smoke test 통과를 학습 성능 개선으로 간주하지 않는다.
* 학습 코드의 실행 가능 여부와 policy 성능 개선 여부를 구분하여 보고한다.

## 에러 기록 규칙

* 각 프로젝트는 `logs/error_log.md`를 유지한다.
* 학습, 실행, 빌드, 시뮬레이션, ROS 2 통신 또는 기존 장애와 관련된 코드를 수정하기 전에 해당 프로젝트의 `logs/error_log.md`를 확인한다.
* 문서 오탈자나 장애와 무관한 단순 수정에서는 `error_log.md` 확인을 생략할 수 있다.
* `error_log.md`에는 확인된 사실을 중심으로 기록한다.
* 확인된 사실과 추정 원인을 명확히 구분한다.
* 미확인 가설은 사실처럼 기록하지 않고 `보류 가설` 또는 `미확인`으로 표시한다.
* 장문의 작업 일지는 기록하지 않는다.
* 항목은 최신 내용이 문서 최상단에 오도록 역시간순으로 정렬한다.
* 항목 제목은 `YYYY-MM-DD HH:MM [EXXX] [분류] 제목` 형식을 사용한다.
* `분류`는 검색 가능한 짧은 키워드로 통일한다.
* 분류 예시는 `[FAST-LIO]`, `[빌드]`, `[IsaacSim]`, `[ROS2]`, `[Training]`, `[Runtime]`이다.
* 각 항목은 기본적으로 `<details>` 및 `<summary>` 접기 형식을 사용한다.
* 최신 항목만 `open` 상태로 유지한다.
* 각 항목은 `증상 / 원인 / 확인 / 해결 / 관련 파일` 형식을 기본으로 사용한다.
* 해결된 문제뿐 아니라 중요한 실패 실험, 보류한 가설 및 효과가 제한적이었던 우회도 짧게 기록한다.
* 실패 기록에는 최소한 `무엇을 시도했는지`, `결과가 어땠는지`, `왜 다음 우선순위에서 제외했는지`를 남긴다.
* 단순히 성능이 개선되지 않은 모든 실험을 장애로 기록하지 않는다. 재발 방지 가치가 있는 핵심 실패만 기록한다.
* 동일 원인의 중복 항목은 추가하지 않고 기존 항목을 갱신한다.
* 오래된 해결 완료 항목은 삭제하지 않고 `logs/error_log_archive_YYYY.md` 형식의 파일로 이동한다.
* `logs/error_log.md`는 최근 또는 활성 이슈 중심으로 유지한다.
* 아카이브 파일 경로를 `logs/error_log.md` 상단에 명시한다.
* 임시 장애 기록이 반복적으로 적용되는 운영 규칙으로 확정되면 `README.md` 또는 `AGENTS.md`에 반영한다.

## AGENTS 적용 규칙

* 워크스페이스 루트의 `AGENTS.md`를 공통 규칙의 기준 원본으로 사용한다.
* 각 로봇 패키지의 하위 `AGENTS.md`에는 해당 패키지에만 적용되는 추가 규칙 또는 예외만 작성한다.
* 루트 `AGENTS.md`의 내용을 하위 파일에 불필요하게 복제하지 않는다.
* 하위 `AGENTS.md`는 루트의 공통 규칙을 무효화하기 위한 용도로 사용하지 않는다.
* 여러 독립 Git 저장소의 구조로 인해 복제본 유지가 반드시 필요한 경우, 공통 영역과 패키지 고유 영역을 구분한다.
* 동기화 과정에서 패키지별 고유 규칙을 덮어쓰지 않는다.

## 문서 반영 규칙

* 구조, 설정 계약, 운영 원칙 또는 실행 절차를 수정하면 관련 Markdown 문서도 같은 변경 단위에서 갱신한다.
* 변경 성격에 따라 `README.md`와 `logs/error_log.md` 중 관련 문서에 반영한다.
* 어떤 문서에 반영해야 하는지 불분명하면 임의로 `AGENTS.md`에 내용을 추가하지 않는다.
* 문서 반영 위치가 불분명한 경우 변경 결과 보고에서 사용자에게 알린다.
* `AGENTS.md`에는 여러 작업에 반복적으로 적용되는 지속적인 운영 원칙만 기록한다.

## 워크스페이스 구성 및 적용 절차

### 워크스페이스 역할

* `r2_isaaclab_trone`

  * Isaac Lab 및 MJX 기반 policy 학습 워크스페이스
  * reward, PPO, domain randomization, curriculum 및 학습 설정을 조정한다.
  * 최종 policy를 ONNX 파일로 export한다.

* `pbr2_ws`

  * ROS 2 및 MuJoCo 기반 Sim-to-Sim 검증 워크스페이스
  * 학습된 ONNX policy의 observation, action, joint order, scaling 및 control timing이 학습 환경과 일치하는지 검증한다.

* `pongbot_ros2_ws`

  * 실제 PongBot R2 구동 워크스페이스
  * Sim-to-Sim 검증을 통과한 ONNX policy를 실제 로봇에 적용한다.

### 적용 순서

1. `r2_isaaclab_trone`에서 policy를 학습한다.
2. 학습 결과를 ONNX 파일로 export한다.
3. `pbr2_ws`에서 Sim-to-Sim 검증을 수행한다.
4. 검증을 통과한 ONNX policy를 `pongbot_ros2_ws`에 적용한다.

앞 단계의 검증을 통과하지 않은 policy를 다음 단계에 적용하지 않는다.

### ONNX 적용 규칙

`pbr2_ws`와 `pongbot_ros2_ws`에서는 학습된 ONNX policy에 맞추기 위해 필요한 코드만 수정한다.

변경 가능한 항목:

* ONNX 파일 경로와 모델 로딩
* 입력 및 출력 tensor shape
* observation buffer와 history buffer
* observation 순서, scaling 및 normalization
* action 순서, scaling 및 clipping
* policy joint order와 시뮬레이터 또는 실제 로봇 joint order 사이의 mapping
* policy inference 주기 연결
* ONNX 입출력 검증용 로그와 assertion

ONNX 적용 시 다음 항목이 학습 코드와 정확히 일치해야 한다.

* observation 구성, 순서, 단위 및 scaling
* history 길이와 시간 순서
* action 구성, 순서, scaling 및 clipping
* joint order
* default joint position
* control timestep과 policy inference timestep

입력과 출력 shape만 같다고 호환되는 것으로 판단하지 않는다. 각 index의 물리적 의미와 단위까지 일치해야 한다.

### 실제 로봇 코드 변경 제한

`pongbot_ros2_ws`에서는 ONNX 적용과 직접 관련 없는 코드를 사용자의 명시적인 승인 없이 수정하지 않는다.

특히 다음 항목은 임의로 변경하지 않는다.

* CAN/CAN FD 및 모터 통신
* ROS 2 topic과 message interface
* 모터 enable, disable 및 torque-off 절차
* 비상 정지와 safety check
* encoder 방향과 zero offset
* 실제 joint limit과 torque limit
* 저수준 모터 제어 주기
* standing 및 initialization 절차

실물 로봇에서 성능이 좋지 않더라도 임시 offset, 임의의 부호 변경 또는 근거 없는 gain 조정으로 문제를 숨기지 않는다.
