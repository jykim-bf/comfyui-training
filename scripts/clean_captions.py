#!/usr/bin/env python3
"""
LoRA 학습 캡션에서 identity 단어가 포함된 절(clause)을 제거한다.

소량 데이터 LoRA 에선 identity 속성이 캡션에 있으면 "가변 요소" 로 학습돼 트리거
토큰이 identity 를 못 잡음. 반대로 캡션에 없으면 트리거가 해당 특성을 흡수.

동작:
  1) 먼저 **마침표(문장 단위)** 로 분리
  2) 각 문장을 **쉼표(절 단위)** 로 분리
  3) 절에 타겟 단어 있으면 그 절만 제거 (문장 경계 유지)
  4) 빈 문장은 전체 제거

사용:
    python scripts/clean_captions.py data/vton/lora/inputs/bh data/vton/lora/inputs/sj
"""

import re
import sys
from pathlib import Path

TARGET = re.compile(
    r'\b(faces?|eyes?|noses?|mouths?|skin|looking|Koreans?|Asians?|young)\b',
    re.IGNORECASE,
)


def clean_caption(text: str) -> str:
    text = text.strip().rstrip(".").strip()
    sentences = re.split(r'\.\s+', text)
    out_sents = []
    for s in sentences:
        s = s.strip().rstrip(",").strip()
        if not s:
            continue
        clauses = [c.strip() for c in s.split(",")]
        kept = [c for c in clauses if c and not TARGET.search(c)]
        if kept:
            out_sents.append(", ".join(kept))
    if not out_sents:
        return ""
    # 한 문장 한 줄 (편집 편의)
    out_sents = [re.sub(r"\s+", " ", s).strip() for s in out_sents]
    return "\n".join(s + "." for s in out_sents)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: clean_captions.py <dir> [<dir> ...]")
    total, changed = 0, 0
    for d in sys.argv[1:]:
        root = Path(d)
        if not root.is_dir():
            print(f"skip (not a dir): {d}", file=sys.stderr)
            continue
        for txt in sorted(root.glob("*.txt")):
            original = txt.read_text(encoding="utf-8").strip()
            cleaned = clean_caption(original)
            total += 1
            if cleaned != original:
                txt.write_text(cleaned + "\n", encoding="utf-8")
                changed += 1
    print(f"processed: {total}, changed: {changed}")


if __name__ == "__main__":
    main()
