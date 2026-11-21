# CLAUDE

## 대화 규칙
* 모든 대화는 한국어로 한다.

## 구조
* 2026-11-11 `~/dev/ai-forge/comfyui/training` 에서 분리. LoRA 학습 전용.
    * `ai-toolkit/` — 학습 도구 (서브모듈, ostris/ai-toolkit)
    * `configs/` — 학습 설정 yaml (git 추적)
    * `scripts/` — 학습 실행(`train_*.sh`)·데이터셋 준비(얼굴 크롭·캡션·마스크) (git 추적)
    * `inputs/` — 학습 데이터셋. `inputs/sheets/` 는 ai-forge assets 작업의 합격 이미지 (git 무시)
    * `outputs/` — 학습 결과물 (git 무시)
    * `models/` — 학습용 베이스 모델 링크 (git 무시)
* 베이스 모델은 `~/dev/comfyui/models` 를 쓴다 (`MODELS_PATH`, config 의 절대경로).
* 실행: `bash scripts/train_<name>.sh` (리포 루트 기준 상대경로). ComfyUI(8188)가 떠 있으면 VRAM 충돌로 멈춘다.
* Python 환경: 리포의 `.venv` (uv, Python 3.11.15, 옛 conda `aitoolkit` 과 같은 버전). 고정 목록 `requirements.lock.txt`, 복구 `./scripts/venv_sync.sh`, 목록 갱신 `--freeze`, 인덱스 설정 `uv.toml`. VLM 캡션(`vlm_caption*.py`)만 아직 conda `florence-cuda`
* 일부 옛 config(`zib_*`, `klein_*`, `train_*`)는 `~/dev/ai-forge/data/vton/lora/` 를 가리킨다 (VTON 데이터, ai-forge 에 그대로 둠).

## 문서 작성 원칙
* 사용자 지시에 의해 작업을 할 때마다, `docs/histories/` 밑에 '2026-03-22.md' 형식으로 작업 내역을 요약하여 기록한다.
    * 작성 방식은 한 단락을 사용자요청과 처리내역요약으로 나눈다.
    * 시도를 하여서 실패를 하여도 기록한다.
    * 사용자 지시가 있으면 그 내역을 항상 먼저 기록한다.
* 문서를 마감해 달라고 하면, 모든 파일을 커밋&푸쉬한다.

## 작업 원칙
* 서브모듈 디렉토리(`ai-toolkit/`) 아래에는 데이터를 두지 않는다. 설정·데이터·결과물은 반드시 서브모듈 밖에 위치해야 한다.

## 환경
* 윈도우즈 WSL2로 AlmaLinux10으로 설치함
