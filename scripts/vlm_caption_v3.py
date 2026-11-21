#!/usr/bin/env python3
"""
LoRA 학습용 VLM 캡션 생성기 (v4 - 순수 트리거 워드 전용).

- 캡션 시작을 "[shot-type] of {trigger}, [description]" 형식으로 고정 (anchor)
- shot-type 태그를 강제하여 LoRA 가 composition ↔ identity 를 분리 학습
- "a woman" 일반 명사를 제거해 기본 모델의 평균 얼굴 희석 방지

사용:
    python scripts/vlm_caption_v4.py --input-dir /home/sasair/dev/training/inputs/캐릭터폴더 --trigger 트리거명

옵션:
    --overwrite      기존 .txt 덮어쓰기 (기본: 건너뜀)
    --max-tokens N   생성 토큰 한계 (기본 180)
    --max-pixels N   이미지 리사이즈 상한 (기본 1638400 = ~1280×1280)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SHOT_TYPES = [
    "close-up portrait",
    "head-and-shoulders shot",
    "chest-up shot",
    "waist-up shot",
    "half-body shot",
    "knee-up shot",
    "full-body shot",
]

CAPTION_PROMPT_TEMPLATE = """Describe this photo for diffusion model training. Output format:

"[SHOT-TYPE] of {trigger}, [description]"

Where [SHOT-TYPE] is EXACTLY one of:
- close-up portrait  (head only or head-focused selfie)
- head-and-shoulders shot
- chest-up shot
- waist-up shot
- half-body shot  (thighs visible)
- knee-up shot
- full-body shot  (legs and feet visible)

In [description], describe:
- Hair (length, color, style)
- Clothing (top, bottom, accessories, color, material)
- Pose and body position
- Background and setting
- Lighting (indoor/outdoor, soft/hard, color temperature)

DO NOT describe:
- Facial features (eyes, nose, mouth, face shape)
- Body proportions, height, or build (e.g., slender, tall, petite)
- Skin tone or ethnicity
- Age or emotions
- Who the person is

Keep the whole output under 70 words. Start directly with the shot-type. Do not add preamble like "This image shows"."""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_caption(caption: str, trigger: str, filename: str = "") -> str:
    """VLM 출력이 규칙에서 조금 벗어나도 shot-type anchor 를 보장."""
    c = re.sub(r"\s+", " ", caption.strip())
    c = re.sub(r"^(a|an|the)\s+", "", c, flags=re.IGNORECASE)
    # 이미 규칙대로 시작하면 그대로 사용
    for st in SHOT_TYPES:
        prefix = f"{st} of {trigger}"
        if c.lower().startswith(prefix.lower()):
            return c
    # shot-type 만 잡혔지만 trigger 누락 시 보정
    for st in SHOT_TYPES:
        if c.lower().startswith(st.lower()):
            rest = c[len(st):].lstrip(", .")
            return f"{st} of {trigger}, {rest}"
    # shot-type 자체 누락 시 fallback
    print(f"    [WARN] fallback used for {filename}: {c[:60]}")
    return f"half-body shot of {trigger}, {c}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--trigger", required=True, help="트리거 토큰 (예: BoHae)")
    ap.add_argument("--model", default="Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed")
    ap.add_argument("--max-tokens", type=int, default=180)
    ap.add_argument("--max-pixels", type=int, default=1280 * 1280)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input dir not found: {args.input_dir}")

    images = sorted(
        f for f in args.input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"No images in {args.input_dir}")

    todo = []
    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            print(f"skip (exists): {img.name}")
            continue
        todo.append(img)

    if not todo:
        print("Nothing to caption.")
        return

    prompt = CAPTION_PROMPT_TEMPLATE.format(trigger=args.trigger)

    print(f"Loading {args.model} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels,
    )
    print(f"Loaded in {time.time()-t0:.1f}s. Processing {len(todo)} images...")

    for i, img_path in enumerate(todo, 1):
        inputs = None
        out = None
        try:
            image = Image.open(img_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], padding=True, return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                )

            out_trimmed = out[:, inputs.input_ids.shape[1]:]
            caption = processor.batch_decode(
                out_trimmed, skip_special_tokens=True
            )[0].strip()

            normalized = normalize_caption(caption, args.trigger, img_path.name)
            txt_path = img_path.with_suffix(".txt")
            txt_path.write_text(normalized + "\n", encoding="utf-8")
            print(f"[{i}/{len(todo)}] {img_path.name}")
            print(f"    → {normalized[:120]}")
        except Exception as e:
            print(f"[{i}/{len(todo)}] {img_path.name} ERROR: {e}", file=sys.stderr)
        finally:
            if inputs is not None:
                del inputs
            if out is not None:
                del out
            torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()