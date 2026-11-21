#!/usr/bin/env python3
"""
특정 학습 이미지 캡션에 왜곡 태그를 삽입.

LoRA 가 얼굴 왜곡(광각 셀카 등) 을 identity 로 흡수하는 것을 방지하기 위해,
문제 이미지 캡션의 "a woman," 직후에 고유 태그를 주입한다. 이 태그는 추론 시
절대 쓰지 않음으로써 왜곡 효과가 자동 배제된다.

사용:
    python scripts/inject_distortion_tags.py
"""

import re
from pathlib import Path

# (파일명 일부, 삽입할 태그)
# 파일명은 부분 매칭으로 찾음 (확장자/접미어 무관)
INJECTIONS_BH = [
    # Screenshot, 2023-05-22 는 전신+얼굴 1.5% 로 _excluded/ 이동 — 대상 제외
    ("IMG_0599",               "wide-angle lens distortion"),
    ("20160516_122011",        "low-angle kneeling pose"),
]

INJECTIONS_SJ = [
    ("240417587_2972473016404127", "wide-angle lens distortion"),
    ("649452521_18094169539863944", "wide-angle lens distortion"),
    ("650683506_18097366465966577", "wide-angle lens distortion"),
    ("126904941_2728154594065996", "low-angle view"),
]

ROOT = Path("/home/sasair/dev/ai-forge/data/vton/lora/inputs")


def inject_tag(txt_path: Path, tag: str) -> bool:
    text = txt_path.read_text(encoding="utf-8")
    # "a woman," 직후에 태그 삽입. 이미 같은 태그 있으면 skip.
    if tag in text:
        return False
    new = re.sub(
        r'(a woman,)',
        rf'\1 {tag},',
        text,
        count=1,
    )
    if new == text:
        # "a woman," 앵커 없으면 경고만 하고 skip
        print(f"  [WARN] 'a woman,' anchor not found in {txt_path.name}")
        return False
    txt_path.write_text(new, encoding="utf-8")
    return True


def process(dir_name: str, injections: list[tuple[str, str]]) -> None:
    input_dir = ROOT / dir_name
    if not input_dir.is_dir():
        print(f"skip (not a dir): {input_dir}")
        return
    for partial_name, tag in injections:
        matches = [p for p in input_dir.glob("*.txt") if partial_name in p.name]
        if not matches:
            print(f"  [MISS] no caption matches '{partial_name}' in {dir_name}")
            continue
        for m in matches:
            changed = inject_tag(m, tag)
            status = "UPDATED" if changed else "SKIPPED"
            print(f"  [{status}] {m.name} <- '{tag}'")


def main() -> None:
    print("[bh]")
    process("bh", INJECTIONS_BH)
    print("[sj]")
    process("sj", INJECTIONS_SJ)
    print("Done.")


if __name__ == "__main__":
    main()
