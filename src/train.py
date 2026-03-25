from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from .runtime import resolve_preferred_yolo_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass
class Sample:
    image_path: Path
    relative_key: Path


def _collect_images(images_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(images_dir)
        samples.append(Sample(image_path=path, relative_key=rel))
    return samples


def _extract_video_frames(
    videos_dir: Path,
    output_dir: Path,
    sample_every: int,
    max_frames_per_video: int,
) -> list[Sample]:
    samples: list[Sample] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = [
        p
        for p in sorted(videos_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[warn] skip unreadable video: {video_path}")
            continue

        safe_stem = video_path.stem.replace(" ", "_")
        video_out_dir = output_dir / safe_stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % max(1, sample_every) != 0:
                continue
            saved += 1
            out_name = f"{safe_stem}_f{frame_idx:06d}.jpg"
            out_path = video_out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            rel = out_path.relative_to(output_dir)
            samples.append(Sample(image_path=out_path, relative_key=rel))
            if max_frames_per_video > 0 and saved >= max_frames_per_video:
                break
        cap.release()
    return samples


def _find_label_path(image_sample: Sample, labels_dir: Path) -> Path | None:
    candidate_by_rel = (labels_dir / image_sample.relative_key).with_suffix(".txt")
    if candidate_by_rel.is_file():
        return candidate_by_rel

    candidate_by_name = labels_dir / f"{image_sample.image_path.stem}.txt"
    if candidate_by_name.is_file():
        return candidate_by_name
    return None


def _safe_token(path: Path) -> str:
    raw = str(path.with_suffix(""))
    token = raw.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return token


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src.resolve(), dst)


def _write_data_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_classes(class_names: str | None, classes_file: str | None) -> list[str]:
    if classes_file:
        file_path = Path(classes_file)
        names = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    if class_names:
        names = [item.strip() for item in class_names.split(",") if item.strip()]
        if names:
            return names
    return ["ball", "goalkeeper", "player", "referee"]


def _auto_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def build_dataset(args: argparse.Namespace) -> Path | None:
    dataset_root = Path(args.dataset_dir).resolve()
    raw_frames_dir = dataset_root / "raw_frames"
    images_train_dir = dataset_root / "images" / "train"
    images_val_dir = dataset_root / "images" / "val"
    labels_train_dir = dataset_root / "labels" / "train"
    labels_val_dir = dataset_root / "labels" / "val"
    data_yaml_path = dataset_root / "data.yaml"
    unlabeled_manifest = dataset_root / "unlabeled_manifest.txt"

    dataset_root.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    if args.images_dir:
        images_dir = Path(args.images_dir).resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {images_dir}")
        samples.extend(_collect_images(images_dir))

    if args.videos_dir:
        videos_dir = Path(args.videos_dir).resolve()
        if not videos_dir.exists():
            raise FileNotFoundError(f"videos dir not found: {videos_dir}")
        frame_samples = _extract_video_frames(
            videos_dir=videos_dir,
            output_dir=raw_frames_dir,
            sample_every=args.sample_every,
            max_frames_per_video=args.max_frames_per_video,
        )
        samples.extend(frame_samples)

    if not samples:
        raise RuntimeError("no image sample found from --images-dir/--videos-dir")

    labels_dir = Path(args.labels_dir).resolve() if args.labels_dir else None
    if labels_dir is None:
        if args.prepare_only:
            print(f"[ok] extracted/collected {len(samples)} samples at {dataset_root}")
            print(f"[hint] provide --labels-dir later to build train/val split and train model.")
            return None
        raise RuntimeError("--labels-dir is required when training")

    labeled_pairs: list[tuple[Sample, Path]] = []
    unlabeled: list[Sample] = []
    for sample in samples:
        label_path = _find_label_path(sample, labels_dir)
        if label_path is None:
            unlabeled.append(sample)
            continue
        labeled_pairs.append((sample, label_path))

    unlabeled_manifest.write_text(
        "\n".join(str(item.image_path) for item in unlabeled) + ("\n" if unlabeled else ""),
        encoding="utf-8",
    )
    print(f"[info] total samples={len(samples)}, labeled={len(labeled_pairs)}, unlabeled={len(unlabeled)}")
    print(f"[info] unlabeled manifest: {unlabeled_manifest}")

    if len(labeled_pairs) < 2:
        raise RuntimeError("not enough labeled samples to split train/val (need >=2)")

    random.seed(args.seed)
    random.shuffle(labeled_pairs)

    val_count = int(round(len(labeled_pairs) * args.val_ratio))
    val_count = min(max(1, val_count), len(labeled_pairs) - 1)

    val_pairs = labeled_pairs[:val_count]
    train_pairs = labeled_pairs[val_count:]

    _ensure_clean_dir(images_train_dir)
    _ensure_clean_dir(images_val_dir)
    _ensure_clean_dir(labels_train_dir)
    _ensure_clean_dir(labels_val_dir)

    def place_pairs(pairs: list[tuple[Sample, Path]], image_dir: Path, label_dir: Path) -> None:
        for index, (sample, label_path) in enumerate(pairs, start=1):
            base = _safe_token(sample.relative_key)
            ext = sample.image_path.suffix.lower() or ".jpg"
            dst_img = image_dir / f"{base}_{index:06d}{ext}"
            dst_lbl = label_dir / f"{base}_{index:06d}.txt"
            _link_or_copy(sample.image_path, dst_img, args.copy_mode)
            _link_or_copy(label_path, dst_lbl, args.copy_mode)

    place_pairs(train_pairs, images_train_dir, labels_train_dir)
    place_pairs(val_pairs, images_val_dir, labels_val_dir)

    class_names = _resolve_classes(args.class_names, args.classes_file)
    _write_data_yaml(path=data_yaml_path, dataset_root=dataset_root, class_names=class_names)
    print(f"[ok] dataset prepared: {dataset_root}")
    print(f"[ok] train={len(train_pairs)} val={len(val_pairs)} classes={class_names}")
    print(f"[ok] data.yaml: {data_yaml_path}")
    return data_yaml_path


def run_training(args: argparse.Namespace, data_yaml_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run `pip install -r requirements.txt`.") from exc

    device = args.device if args.device != "auto" else _auto_device()
    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": device,
        "workers": int(args.workers),
        "project": args.project,
        "name": args.name,
        "patience": int(args.patience),
        "cache": bool(args.cache),
    }
    if args.resume:
        train_kwargs["resume"] = True

    print(f"[run] training with device={device}, model={args.model}")
    model.train(**train_kwargs)
    print("[ok] training finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from images/videos and train model")

    parser.add_argument("--images-dir", help="Directory containing labeled/unlabeled images")
    parser.add_argument("--videos-dir", help="Directory containing videos to extract frames from")
    parser.add_argument("--labels-dir", help="Directory containing YOLO txt labels")
    parser.add_argument("--dataset-dir", default="./datasets/soccer_train", help="Output dataset directory")
    parser.add_argument("--sample-every", type=int, default=30, help="Extract one frame every N frames")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--class-names", default="ball,goalkeeper,player,referee")
    parser.add_argument("--classes-file", help="Optional text file, one class name per line")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset; skip training")

    parser.add_argument("--model", default=resolve_preferred_yolo_model(), help="Base model/weights for finetuning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="./runs/train")
    parser.add_argument("--name", default="soccer-finetune")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml_path = build_dataset(args)
    if args.prepare_only:
        return
    if data_yaml_path is None:
        raise RuntimeError("dataset yaml not generated")
    run_training(args, data_yaml_path)


if __name__ == "__main__":
    main()
