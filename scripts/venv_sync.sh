#!/bin/bash
# .venv (uv) 만들기 / 복구 / 고정 목록 갱신
# 사용법:
#   ./scripts/venv_sync.sh            # .venv 가 없으면 만들고, requirements.lock.txt 그대로 설치 (--no-deps, 정확히 같은 버전)
#   ./scripts/venv_sync.sh --freeze   # 지금 .venv 상태를 requirements.lock.txt 에 기록
#
# requirements.lock.txt 는 2026-11-11 conda 환경에서 옮겨 온 고정 목록이다.
# uv 프로젝트(pyproject + uv.lock) 대신 venv + 고정 목록을 쓰는 이유: 커스텀 노드·학습 도구가
# 제각기 pip/uv pip 로 패키지를 더 설치하는데, uv sync 는 목록에 없는 패키지를 지워 버린다.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
PY=".venv/bin/python"
LOCK="requirements.lock.txt"

if [ "${1:-}" = "--freeze" ]; then
  uv pip freeze --python "$PY" | LC_ALL=C sort -f > "$LOCK"
  echo "기록: $LOCK ($(wc -l < "$LOCK") 개)"
  exit 0
fi

[ -x "$PY" ] || uv venv --python 3.11.15 .venv
uv pip install --python "$PY" --no-deps -r "$LOCK"
# 같은 파일을 쓰는 패키지 쌍(cu12/cu13 등)은 설치 순서에 따라 결과가 달라진다 → 이길 쪽을 한 줄씩 다시 설치
if [ -f requirements.last.txt ]; then
  grep -vE '^\s*(#|$)' requirements.last.txt | while read -r pkg; do
    uv pip install --python "$PY" --no-deps --reinstall "$pkg" --quiet
  done
fi
echo "완료: $(uv pip list --python "$PY" 2>/dev/null | tail -n +3 | wc -l) 개 설치됨"
