#!/usr/bin/env python3
"""Face-crop dataset for H3 character LoRAs (No-resize lossless crop).

    python scripts/make_face_dataset.py <src_dir> <dst_dir> --name IAn
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--name", required=True)
    ap.add_argument("--scale", type=float, default=1.8)
    ap.add_argument("--min-face", type=int, default=160)
    ap.add_argument("--min-crop", type=int, default=384, help="최종 크롭 한 변의 최소 픽셀 (미만 시 저화질 제외)")
    a = ap.parse_args()

    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in EXT)
    kept, skipped = 0, {"no face": 0, "small": 0, "two faces": 0}

    for p in files:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        faces = fa.get(np.array(im)[:, :, ::-1])
        if not faces:
            skipped["no face"] += 1
            continue

        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        x0, y0, x1, y1 = faces[0].bbox
        side = max(x1 - x0, y1 - y0)
        if side < a.min_face:
            skipped["small"] += 1
            continue
        if len(faces) > 1 and max(faces[1].bbox[2] - faces[1].bbox[0], faces[1].bbox[3] - faces[1].bbox[1]) >= side / 2:
            skipped["two faces"] += 1
            continue

        half = side * a.scale / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.08 * side
        box = [cx - half, cy - half, cx + half, cy + half]

        W, H = im.size
        dx = max(0, -box[0]) - max(0, box[2] - W)
        dy = max(0, -box[1]) - max(0, box[3] - H)
        box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        s = min(box[2] - box[0], box[3] - box[1])

        if int(round(s)) < a.min_crop:
            skipped["small"] += 1
            continue

        box = [box[0], box[1], box[0] + s, box[1] + s]
        kept += 1
        out = dst / f"{a.name}_{kept:03d}.png"
        # 리사이즈 없이 원본 픽셀 그대로 저장
        im.crop([int(round(v)) for v in box]).save(out)
        (dst / f"{a.name}_{kept:03d}.txt").write_text(f"close-up portrait of {a.name}\n")

    print(f"{len(files)} photos -> kept {kept}, skipped {skipped}")


if __name__ == "__main__":
    main()