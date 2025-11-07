#!/usr/bin/env python3
"""
compose_from_dict.py
--------------------
Compose a YOLO split (images/ + labels/) from a dictionary of sources.

Key rules:
- Flat sources: labels are under <labels_root>/<stem>.txt  (label_mode="flat")
- Positive sequences: image .../seqX/images_seqX/<file> → label <labels_root>/seqX/<stem>.txt  (label_mode="seq_pos")
- Negative sequences: image .../seqY_neg/<file>         → label <labels_root>/seqY_neg/<stem>.txt (label_mode="seq_neg")
- If a label is missing, DO NOT create an empty one. Report it. Optionally fail with non-zero exit unless --allow_missing.

Usage example:
  python compose_from_dict.py \
    --sources "C:/.../sources_train.py::SOURCES" \
    --include train_single,train_seq_pos,train_seq_neg,real_colon \
    --out "C:/.../data/detection2/yolo_splits/train" \
    --copy_mode copy \
    --verbose 1
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import argparse, json, re, shutil, sys, random
import re

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -------------------------- helpers --------------------------

def list_images(base: Path, recursive: bool, glob: Optional[str]) -> List[Path]:
    if glob:
        hits = list(base.glob(glob))
        return [p for p in hits if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if recursive:
        return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


_SEQ_RE = re.compile(r"^seq_?\d+(_neg)?$", re.IGNORECASE)

def _find_seq_folder_name(image_path: Path) -> str | None:
    """Return the sequence folder name (e.g., 'seq3', 'seq_4', 'seq16_neg') from parent dirs."""
    for parent in image_path.parents:
        name = parent.name
        if _SEQ_RE.match(name):
            return name
    return None

def label_path_for(image_path: Path, labels_root: Path, label_mode: str) -> Path:
    """
    Resolve label path:
      - flat    → labels_root/<stem>.txt
      - seq_pos → labels_root/<seqX>/<stem>.txt
      - seq_neg → labels_root/<seqY_neg>/<stem>.txt
    """
    lm = (label_mode or "flat").lower()
    if lm == "flat":
        return labels_root / f"{image_path.stem}.txt"

    seq_name = _find_seq_folder_name(image_path)
    if lm == "seq_pos":
        if seq_name and "neg" not in seq_name.lower():
            return labels_root / seq_name / f"{image_path.stem}.txt"
        # fallback to flat if no proper seq match
        return labels_root / f"{image_path.stem}.txt"

    if lm == "seq_neg":
        if seq_name and "neg" in seq_name.lower():
            return labels_root / seq_name / f"{image_path.stem}.txt"
        return labels_root / f"{image_path.stem}.txt"

    # default fallback
    return labels_root / f"{image_path.stem}.txt"

def copy_or_link(src: Path, dst: Path, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            dst.unlink(missing_ok=True)
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def ensure_clean(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=False)

def load_sources(sources_arg: str) -> Dict[str, Dict]:
    """
    Load SOURCES:
      - JSON file path (*.json)
      - Python module variable: "path/to/module.py::SOURCES"
      - Inline JSON string
    """
    p = Path(sources_arg)
    if p.suffix.lower() == ".json" and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))

    if "::" in sources_arg:
        mod_path, var_name = sources_arg.split("::", 1)
        mod_file = Path(mod_path)
        if not mod_file.exists():
            raise FileNotFoundError(f"Module file not found: {mod_file}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("src_cfg", str(mod_file))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, var_name):
            raise AttributeError(f"{mod_file} has no variable named {var_name}")
        data = getattr(mod, var_name)
        # Convert any Path values to str so downstream Path() works uniformly
        def _normalize(d):
            if isinstance(d, dict):
                return {k: _normalize(v) for k, v in d.items()}
            if isinstance(d, list):
                return [_normalize(x) for x in d]
            if isinstance(d, Path):
                return str(d)
            return d
        return _normalize(data)

    # Inline JSON
    return json.loads(sources_arg)

def parse_include_list(s: str) -> List[str]:
    return [k.strip() for k in s.split(",") if k.strip()]

# -------------------------- core --------------------------

def compose_from_sources(
    sources: Dict[str, Dict],
    include_keys: List[str],
    out_root: Path,
    copy_mode: str = "copy",
    seed: int = 42,
    dry_run: bool = False,
    verbose: int = 1,
    allow_missing: bool = False,
) -> int:
    """
    Returns exit code: 0 on success, 1 if missing labels found and allow_missing=False.
    """
    rng = random.Random(seed)
    out_images = out_root / "images"
    out_labels = out_root / "labels"

    if verbose:
        print(f"[info] OUT={out_root} | copy_mode={copy_mode} | dry_run={dry_run} | allow_missing={allow_missing}")

    if not dry_run:
        ensure_clean(out_images)
        ensure_clean(out_labels)

    total_written = 0
    total_missing = 0
    missing_examples = []

    for key in include_keys:
        if key not in sources:
            print(f"[error] unknown source key: {key}")
            return 1

        spec = sources[key]
        images_dir = Path(spec["images"])
        labels_dir = Path(spec["labels"])
        recursive  = bool(spec.get("recursive", False))
        glob       = spec.get("glob")
        label_mode = str(spec.get("label_mode", "flat")).lower()

        imgs = list_images(images_dir, recursive=recursive, glob=glob)
        if verbose:
            print(f"[src] {key:15s} imgs_dir={images_dir} labels_dir={labels_dir} mode={label_mode} rec={recursive} glob={glob} | imgs={len(imgs)}")

        kept = 0
        for ip in imgs:
            lp = label_path_for(ip, labels_dir, label_mode=label_mode)

            if not lp.exists():
                total_missing += 1
                if len(missing_examples) < 10:
                    missing_examples.append((str(ip), str(lp)))
                if verbose >= 2:
                    print(f"[missing] label not found for image:\n  img={ip}\n  lbl={lp}")
                # Skip writing this pair entirely
                continue

            if not dry_run:
                # image
                dst_img = out_images / f"{ip.stem}{ip.suffix.lower()}"
                copy_or_link(ip, dst_img, mode=copy_mode)

                # label (must exist, we checked above)
                dst_lbl = out_labels / f"{ip.stem}.txt"
                copy_or_link(Path(lp), dst_lbl, mode="copy")

            kept += 1
            total_written += 1

            if verbose >= 2:
                print(f"  -> {ip} | {lp}")

        if verbose:
            print(f"[keep] {key}: {kept} files (missing_labels={total_missing})")

    # Final report on missing
    if total_missing > 0:
        print(f"\n[warn] Missing labels detected: {total_missing}")
        for i, (imgp, lblp) in enumerate(missing_examples, 1):
            print(f"  {i:02d}) img={imgp}\n      exp_label={lblp}")
        if not allow_missing:
            print("[fail] Missing labels present and --allow_missing is not set.")
            return 1

    print(f"\n[done] wrote {total_written} image+label pairs into {out_root}")
    return 0

# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="Compose a YOLO split (images/labels) from a dictionary of sources.")
    ap.add_argument("--out", type=str, required=True, help="Output directory (creates images/ and labels/ inside)")
    ap.add_argument("--sources", type=str, required=True,
                    help="Sources dict: JSON file, inline JSON, or 'path/to/file.py::SOURCES'")
    ap.add_argument("--include", type=str, required=True,
                    help="Comma-separated keys to include from the sources dict")
    ap.add_argument("--copy_mode", type=str, default="copy", choices=["copy","link"], help="Copy files or attempt hardlinks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--verbose", type=int, default=1, help="0=minimal, 1=per-source counts, 2=file-level logs")
    ap.add_argument("--allow_missing", action="store_true", help="If set, do not fail when labels are missing; still report them.")
    args = ap.parse_args()

    try:
        SOURCES: Dict[str, Dict] = load_sources(args.sources)
    except Exception as e:
        print(f"[error] failed to load sources: {e}")
        sys.exit(1)

    include_keys = parse_include_list(args.include)
    missing = [k for k in include_keys if k not in SOURCES]
    if missing:
        print(f"[error] unknown source keys: {missing}")
        sys.exit(1)

    rc = compose_from_sources(
        sources=SOURCES,
        include_keys=include_keys,
        out_root=Path(args.out),
        copy_mode=args.copy_mode,
        seed=args.seed,
        dry_run=args.dry_run,
        verbose=args.verbose,
        allow_missing=args.allow_missing,
    )
    sys.exit(rc)

if __name__ == "__main__":
    main()
