#!/usr/bin/env python3
"""
Z-Image LoRA 테스트 이미지 생성.

베이스 모델(Tongyi-MAI/Z-Image) + 학습된 LoRA 를 로드해서 10개 다양한 테스트
프롬프트로 이미지 생성. 트리거를 치환하므로 여러 인물에 같은 프롬프트 구조 재사용 가능.

사용:
    python scripts/lora_inference.py \\
        --lora data/vton/lora/outputs/bohae_v2/bohae_v2.safetensors \\
        --trigger BoHae \\
        --output data/vton/lora/outputs/bohae_v2/tests

번호가 같은 파일(= 같은 프롬프트)은 다른 인물 간 비교용.
"""

import argparse
import torch
from pathlib import Path
from diffusers import ZImagePipeline

# 프롬프트 템플릿. {TRIGGER} 위치에 인물 토큰 치환
TEST_PROMPTS = [
    "{TRIGGER}, portrait photo, natural daylight",
    "{TRIGGER}, upper body, indoor soft lighting, casual clothes",
    "{TRIGGER}, outdoor, golden hour, candid pose",
    "{TRIGGER}, laughing loudly, teeth showing, outdoor park, spring",
    "{TRIGGER}, serious expression, no smile, studio dramatic lighting",
    "{TRIGGER}, wearing red puffer jacket, snowing, winter mountain",
    "{TRIGGER}, side profile, looking left, minimalist gray background",
    "{TRIGGER}, formal business suit, office background, professional",
    "{TRIGGER}, running, athletic wear, morning jog in park",
    "{TRIGGER}, surprised expression, wide eyes, open mouth, casual outfit",
]

NEGATIVE = "blurry, low quality, cartoon, illustration, bad anatomy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.lora.is_file():
        raise SystemExit(f"LoRA not found: {args.lora}")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    pipe = ZImagePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")

    print(f"Loading LoRA: {args.lora}")
    pipe.load_lora_weights(str(args.lora), adapter_name="test")

    for i, template in enumerate(TEST_PROMPTS):
        prompt = template.replace("{TRIGGER}", args.trigger)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=gen,
        ).images[0]
        fname = args.output / f"{i:02d}_{args.trigger}.png"
        image.save(fname)
        print(f"[{i+1:02d}/{len(TEST_PROMPTS)}] {fname.name}  ← {prompt}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
