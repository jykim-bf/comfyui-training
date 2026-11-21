#!/usr/bin/env python3
"""
LoRA 학습 데이터 폴더의 이미지/캡션 쌍을 정리한다.

폴더의 모든 이미지에 동일 캡션의 .txt 를 생성하고, 짝이 안 맞는 파일을 정리:
  - 이미지 있음 + .txt 없음   → .txt 생성
  - 이미지 있음 + .txt 있음   → 캡션 덮어쓰기 (--no-overwrite 로 끄기)
  - 이미지 없음 + .txt 있음   → orphan .txt 삭제

사용:
    python scripts/lora_captions.py <folder> "<caption>"
    python scripts/lora_captions.py inputs/IAn "IAn"
    python scripts/lora_captions.py inputs/IAn_Wonhee "IAnWonHee, from left to right: IAn, WonHee"

옵션:
    --dry-run        : 실제 변경 없이 결과만 출력
    --no-overwrite   : 기존 .txt 는 건드리지 않음 (이미지+txt 둘 다 있을 때 skip)
"""

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description="LoRA 학습 캡션 일괄 정리")
    ap.add_argument("folder", type=Path, help="대상 폴더")
    ap.add_argument("caption", type=str, help="모든 이미지에 적용할 캡션 텍스트")
    ap.add_argument("--dry-run", action="store_true", help="실제 변경 없이 결과만 출력")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="기존 .txt 는 그대로 두기 (디폴트는 덮어쓰기)")
    args = ap.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"폴더가 없거나 디렉토리가 아닙니다: {folder}", file=sys.stderr)
        sys.exit(1)

    # 한 단계 깊이만 스캔 (서브폴더 무시)
    entries = [p for p in folder.iterdir() if p.is_file()]

    # stem (확장자 제외 이름) 기준으로 이미지/텍스트 분류
    image_stems: dict[str, Path] = {}
    txt_stems: dict[str, Path] = {}

    for p in entries:
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            image_stems[p.stem] = p
        elif ext == ".txt":
            txt_stems[p.stem] = p

    created = 0
    overwritten = 0
    skipped = 0
    deleted = 0

    # 1) 이미지가 있는 stem 처리
    for stem, img_path in sorted(image_stems.items()):
        txt_path = folder / f"{stem}.txt"
        if stem in txt_stems:
            # 이미지 + txt 둘 다 있음
            if args.no_overwrite:
                skipped += 1
                continue
            existing = txt_path.read_text(encoding="utf-8").strip()
            if existing == args.caption.strip():
                skipped += 1
                continue
            action = "[DRY] 덮어쓰기" if args.dry_run else "덮어쓰기"
            print(f"  {action}: {txt_path.name}")
            if not args.dry_run:
                txt_path.write_text(args.caption, encoding="utf-8")
            overwritten += 1
        else:
            # 이미지만 있음 → txt 생성
            action = "[DRY] 생성" if args.dry_run else "생성"
            print(f"  {action}: {txt_path.name}")
            if not args.dry_run:
                txt_path.write_text(args.caption, encoding="utf-8")
            created += 1

    # 2) 이미지 없는 orphan .txt 삭제
    for stem, txt_path in sorted(txt_stems.items()):
        if stem not in image_stems:
            action = "[DRY] 삭제 (orphan)" if args.dry_run else "삭제 (orphan)"
            print(f"  {action}: {txt_path.name}")
            if not args.dry_run:
                txt_path.unlink()
            deleted += 1

    print()
    print(f"폴더: {folder}")
    print(f"  이미지 {len(image_stems)} / 기존 txt {len(txt_stems)}")
    print(f"  생성 {created} / 덮어쓰기 {overwritten} / 동일/skip {skipped} / orphan 삭제 {deleted}")
    if args.dry_run:
        print("  (dry-run 모드: 실제 변경 없음)")


if __name__ == "__main__":
    main()
