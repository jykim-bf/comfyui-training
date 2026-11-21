#!/usr/bin/env python3
"""의류 학습용 loss 마스크 생성 (rembg u2net_cloth_seg).

이미지 폴더의 각 PNG에 대해 상의/하의/전신 3분할 마스크를 합집합으로 합쳐
같은 파일명으로 출력 폴더에 저장한다. (ai-toolkit datasets[].mask_path 용)

사용법:
  .venv/bin/python scripts/make_cloth_masks.py <입력폴더> <출력폴더>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = new_session("u2net_cloth_seg")

    pngs = sorted(src_dir.glob("*.png"))
    for i, p in enumerate(pngs, 1):
        dest = out_dir / p.name
        if dest.exists():
            print(f"[{i}/{len(pngs)}] skip (있음): {p.name}")
            continue
        img = Image.open(p).convert("RGB")
        m = np.array(remove(img, session=sess, only_mask=True).convert("L"))
        h = m.shape[0] // 3
        union = np.maximum(np.maximum(m[:h], m[h:2*h]), m[2*h:])
        cover = (union > 128).mean() * 100
        Image.fromarray(union).save(dest)
        print(f"[{i}/{len(pngs)}] {p.name}  의류비율 {cover:.1f}%")


if __name__ == "__main__":
    main()
