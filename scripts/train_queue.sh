#!/usr/bin/env bash
# H3 캐릭터 LoRA 큐: IAn v2 6000 → WonHee v2 6000 (맥락 캡션 재학습) 순차 실행.
# 사용: bash scripts/train_h3_queue.sh            (포그라운드, 화면+로그 동시)
#       nohup bash scripts/train_h3_queue.sh &    (백그라운드)
# 로그: outputs/<name>/train.log  — Claude 감시는 이 파일 기준.
set -u
cd /home/sasair/dev/training
PY=.venv/bin/python   # uv venv (2026-11-11, 옛 conda aitoolkit)
export MODELS_PATH=/home/sasair/dev/comfyui/models

if ss -ltn 2>/dev/null | grep -q ":8188 "; then  # pgrep -f 는 자기 셸까지 잡아 오탐 → 포트로 판정
  echo "[queue] ComfyUI 가 떠 있음 — VRAM 충돌. 먼저 내리고 다시 실행." >&2; exit 1
fi

run() {
  local name="$1"
  local cfg="configs/${name}.yaml"
  local out="outputs/${name}"
  mkdir -p "$out"
  echo "[queue] $(date '+%F %T') START $name" | tee -a "$out/train.log"
  "$PY" ai-toolkit/run.py "$cfg" 2>&1 | tee -a "$out/train.log"
  local rc=${PIPESTATUS[0]}
  echo "[queue] $(date '+%F %T') END $name rc=$rc" | tee -a "$out/train.log"
  return $rc
}

run krea2_ian_v2
run krea2_wonhee_v2
run krea2_ahyun_v2
run krea2_baeyi_v2
echo "[queue] $(date '+%F %T') ALL DONE"
