#!/usr/bin/env python3
"""
LoRA 학습 데이터용 VLM 캡션 생성기.

Qwen2.5-VL-Captioner-Relaxed 로 이미지별 캡션을 만들어 kohya_ss 표준 포맷
(이미지 동명 .txt) 으로 저장. identity 요소(얼굴/인종/나이) 는 제외하고
가변 요소(머리·의상·포즈·배경·조명·프레이밍) 만 묘사하도록 유도 → 트리거
토큰이 identity 를 흡수.

사용:
    python scripts/vlm_caption.py --input-dir data/vton/lora/inputs/bh --trigger BoHae
    python scripts/vlm_caption.py --input-dir data/vton/lora/inputs/sj --trigger SuJin

옵션:
    --overwrite      기존 .txt 덮어쓰기 (기본: 건너뜀)
    --max-tokens N   생성 토큰 한계 (기본 120)
    --max-pixels N   이미지 리사이즈 상한 (기본 1048576 = ~1024×1024)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

CAPTION_PROMPT = """Describe this photo for diffusion model training. Focus ONLY on:
- Hair (length, color, style)
- Clothing (top, bottom, accessories, color/material)
- Pose and body position
- Background and setting
- Lighting (indoor/outdoor, soft/hard, color temperature)
- Camera angle and framing

DO NOT describe:
- Facial features (eyes, nose, mouth, face shape, skin tone)
- Ethnicity, age, emotions
- The person's identity or who they are

Output one concise paragraph under 60 words. Start directly with the description, no preamble."""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--trigger", required=True, help="앞에 붙일 트리거 토큰 (예: BoHae)")
    ap.add_argument("--model", default="Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
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

    print(f"Loading {args.model} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels
    )
    print(f"Loaded in {time.time()-t0:.1f}s. Processing {len(todo)} images...")

    for i, img_path in enumerate(todo, 1):
        try:
            image = Image.open(img_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], return_tensors="pt"
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

            full = f"{args.trigger}, {caption}"
            txt_path = img_path.with_suffix(".txt")
            txt_path.write_text(full + "\n", encoding="utf-8")
            print(f"[{i}/{len(todo)}] {img_path.name} → {txt_path.name}")
        except Exception as e:
            print(f"[{i}/{len(todo)}] {img_path.name} ERROR: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
