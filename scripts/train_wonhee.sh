#!/usr/bin/env bash
set -u
cd /home/sasair/dev/training
source "$HOME/miniconda3/etc/profile.d/conda.sh"   # 캡션(florence-cuda)만 아직 conda

NAME=krea2_wonhee_v2
DATA=inputs/wonhee
OUT=outputs/$NAME
mkdir -p "$OUT"

if ss -ltn 2>/dev/null | grep -q ":8188 "; then  # 캡션(Qwen2.5-VL 7B)·학습 모두 VRAM 을 거의 다 씀
  echo "[wonhee] ComfyUI 가 떠 있음 — VRAM 충돌. 먼저 내리고 다시 실행." >&2; exit 1
fi

# 1) 캡션: 아직 자리표시("wonhee, a woman") 캡션이면 VLM 맥락 캡션으로 바꿈(얼굴 묘사 제외). 이미 했으면 건너뜀.
if grep -lx "WonHee" "$DATA"/*.txt >/dev/null 2>&1; then
  echo "[wonhee] $(date '+%F %T') CAPTION start" | tee -a "$OUT/train.log"
  conda activate florence-cuda
  python scripts/vlm_caption_v3.py --input-dir "$DATA" --trigger WonHee --overwrite 2>&1 | tee -a "$OUT/train.log"
  rc=${PIPESTATUS[0]}
  conda deactivate
  if [ "$rc" -ne 0 ]; then echo "[wonhee] 캡션 실패 rc=$rc" | tee -a "$OUT/train.log"; exit "$rc"; fi
  echo "[wonhee] $(date '+%F %T') CAPTION done — 예시:" | tee -a "$OUT/train.log"
  head -c 400 "$DATA/wonhee_001.txt" | tee -a "$OUT/train.log"; echo | tee -a "$OUT/train.log"
fi

# 2) 학습 (ai-toolkit, 500 스텝마다 저장·샘플 → $OUT/samples)
export MODELS_PATH=/home/sasair/dev/comfyui/models
echo "[wonhee] $(date '+%F %T') TRAIN start $NAME" | tee -a "$OUT/train.log"
.venv/bin/python ai-toolkit/run.py "configs/$NAME.yaml" 2>&1 | tee -a "$OUT/train.log"
rc=${PIPESTATUS[0]}
echo "[wonhee] $(date '+%F %T') TRAIN end rc=$rc" | tee -a "$OUT/train.log"
exit "$rc"
