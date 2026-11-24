#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progressive reference propagation for object removal with ArtiFixer.

ArtiFixer's output/jobs/propagate_removal.py, unchanged in behaviour. Seed frames (already cleanly
inpainted by the single all_frames rollout) become anchors. Each round, ArtiFixer generates the next
--step frames at every frontier of the anchor set, conditioned on the nearest anchors (covisibility).
Each generated frame is composited -- real photo (or object-free render, --outside render) outside the
removal mask, generation inside it (feathered, never below the hole) -- and added to the anchors. Every
rollout is short and starts right next to clean anchors, so the autoregressive drift of one long
rollout cannot build up. The model is loaded once for all rounds.

  --scene_root   the photo scene (artifixer_gs.removal split writes it to <split>/scene)
  --removal_dir  the removal split (renders/, opacity/ with the hole at 0, hole/)
  --seed_frames  '0-6,116-122', or 'auto' = the auto_seeds of frames.json (the first 7 frames per window)

Outputs in --output_dir:
  anchors/<i>.png   composited frame (photo or render outside the mask, ArtiFixer inside)
  pred/<i>.png      raw ArtiFixer output
  rounds/<k>/       per-round mini split and generation (deleted unless --keep_rounds)
  order.json        which round produced each frame (0 = seed)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from artifixer_gs.removal import parse_frame_ranges

# generate(split_path, gen_dir, num_views): one covisibility inference over the mini split's targets,
# writing <gen_dir>/<scene_id>/frames/batch_0000/pred/<pos:05d>.png.
Generator = Callable[[Path, Path, int], None]


def sequence_structure(centers: np.ndarray, jump_factor: float = 4.0) -> tuple[list[tuple[int, int]], bool, np.ndarray]:
    """Split the frame order at camera jumps (> jump_factor x the median step); a single sequence whose
    ends are that close is cyclic. Returns the [a, b) bounds, cyclic, and the sequence of every frame."""
    n = len(centers)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median_step = np.median(steps) if len(steps) else 0.0
    cuts = [i + 1 for i, s in enumerate(steps) if s > jump_factor * median_step]
    bounds = list(zip([0] + cuts, cuts + [n]))
    cyclic = len(bounds) == 1 and n > 1 and bool(np.linalg.norm(centers[0] - centers[-1]) < jump_factor * median_step)
    seq_of = np.zeros(n, dtype=int)
    for k, (a, b) in enumerate(bounds):
        seq_of[a:b] = k
    return bounds, cyclic, seq_of


def neighbor(i: int, direction: int, n: int, cyclic: bool) -> int | None:
    j = i + direction
    if cyclic:
        return j % n
    if j < 0 or j >= n:
        return None
    return j


def frontier_chunks(done: dict[int, int], n: int, step: int, cyclic: bool, seq_of: np.ndarray) -> list[tuple[int, list[int]]]:
    """(frontier anchor, chunk) pairs: every done frame grows up to step undone, unclaimed frames in each
    direction. A chunk may start across a camera jump (a new rollout), but never spans one."""
    chunks, claimed = [], set()
    for i in sorted(done):
        for direction in (1, -1):
            chunk, j = [], neighbor(i, direction, n, cyclic)
            while j is not None and j not in done and j not in claimed and len(chunk) < step:
                if chunk and seq_of[chunk[-1]] != seq_of[j]:
                    break
                chunk.append(j)
                claimed.add(j)
                j = neighbor(j, direction, n, cyclic)
            if chunk:
                chunks.append((i, chunk))
    return chunks


def mini_split_order(chunks: list[tuple[int, list[int]]], done: dict[int, int]) -> tuple[list[int], list[int], list[int]]:
    """Frame order [frontier anchor, chunk...] per chunk, then every other anchor; selected/target positions.
    Anchors between chunks split them into separate rollouts; all anchors are reference candidates."""
    order, selected, targets = [], [], []
    for anchor, chunk in chunks:
        selected.append(len(order))
        order.append(anchor)
        for j in chunk:
            targets.append(len(order))
            order.append(j)
    frontier_anchors = {anchor for anchor, _ in chunks}
    for i in sorted(done):
        if i not in frontier_anchors:
            selected.append(len(order))
            order.append(i)
    return order, selected, targets


def resolve_seed_frames(spec: str, scene_root: Path, removal_dir: Path) -> str:
    if spec != "auto":
        return spec
    for path in (scene_root / "frames.json", removal_dir / "frames.json"):
        if path.is_file():
            return json.loads(path.read_text())["auto_seeds"]
    raise FileNotFoundError(f"--seed_frames auto needs frames.json in {scene_root} or {removal_dir}")


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).astype(np.float32)


def model_generator(args: argparse.Namespace) -> Generator:
    """Load ArtiFixer once; the generator runs one covisibility trajectory inference per round."""
    import torch

    from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
    from model_eval.run_inference import (
        get_eval_pipe,
        load_transformer_checkpoint,
        parse_args,
        process_items_with_context_parallel,
    )
    from model_training.data.utils import NeighborSelectionMode

    out = args.output_dir
    run_args = parse_args([
        "--evalset", "reconstructed_colmap", "--checkpoint_pt", args.checkpoint_pt, "--model_id", args.model_id,
        "--save_dir", str(out / "unused"), "--split_path", str(out / "unused.json"),
        "--render_trajectory", "trajectory", "--neighbor_selection_mode", "covisibility", "--save_frame_outputs_only",
        *(["--max_neighbors_per_encode", str(args.max_neighbors_per_encode)] if args.max_neighbors_per_encode else []),
    ])
    device = torch.device("cuda:0")
    with torch.inference_mode():
        pipe = get_eval_pipe(run_args, device)
        load_transformer_checkpoint(pipe.transformer, run_args)
        pipe.transformer.eval()

    def generate(split_path: Path, gen_dir: Path, num_views: int) -> None:
        dataset = ReconstructedColmapEvalDataset(
            split="test", split_path=split_path, num_views=num_views,
            neighbor_selection_mode=NeighborSelectionMode.COVISIBILITY, use_target_indices=True,
        )
        with torch.inference_mode():
            process_items_with_context_parallel(pipe, dataset, run_args, gen_dir, 0, 1, device)

    return generate


def propagate(args: argparse.Namespace, generate: Generator | None = None) -> Path:
    scene_root = args.scene_root.resolve()
    removal_dir = args.removal_dir.resolve()
    out = args.output_dir.resolve()
    args.output_dir = out
    (scene_id, entry), = json.loads((scene_root / "split.json").read_text())["test"].items()
    transforms = json.loads((scene_root / entry["transforms_path"]).read_text())
    frames = transforms["frames"]
    n = len(frames)
    photo_root = scene_root / entry["image_root"]
    for sub in ("anchors", "pred"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    centers = np.array([f["transform_matrix"] for f in frames])[:, :3, 3]
    bounds, cyclic, seq_of = sequence_structure(centers, args.jump_factor)
    print(f"{scene_id}: {n} frames, sequences={bounds}, cyclic={cyclic}", flush=True)

    def composite(i: int, generated_path: Path) -> None:
        if args.outside == "photo":
            photo = load_rgb(photo_root / frames[i]["file_path"])
        else:
            photo = load_rgb(removal_dir / "renders" / f"{i:05d}.png")
        generated = load_rgb(generated_path)
        with Image.open(removal_dir / "hole" / f"{i:05d}.png") as hole_image:
            hole = np.asarray(hole_image.convert("L")).astype(np.float32) / 255.0
        alpha = np.clip(gaussian_filter(hole, args.feather_sigma), 0, 1)[..., None]
        alpha = np.maximum(alpha, hole[..., None])  # never let the photo (with the object) leak into the hole
        mixed = alpha * generated + (1 - alpha) * photo
        Image.fromarray(mixed.round().clip(0, 255).astype(np.uint8)).save(out / "anchors" / f"{i:05d}.png")
        shutil.copy(generated_path, out / "pred" / f"{i:05d}.png")

    seed_frames = resolve_seed_frames(args.seed_frames, scene_root, removal_dir)
    print(f"seed frames: {seed_frames}", flush=True)
    done: dict[int, int] = {}  # frame -> round that produced it (0 = seed)
    for i in parse_frame_ranges(seed_frames):
        assert 0 <= i < n, f"seed frame {i} is outside [0, {n - 1}]"
        composite(i, args.seed_pred_dir / f"{i:05d}.png")
        done[i] = 0

    if len(done) < n and generate is None:
        generate = model_generator(args)

    round_idx = 0
    while len(done) < n:
        round_idx += 1
        chunks = frontier_chunks(done, n, args.step, cyclic, seq_of)
        assert chunks, f"No frontier left but {n - len(done)} frames undone (sequence without a seed?)"
        order, selected, targets = mini_split_order(chunks, done)

        rdir = out / "rounds" / f"{round_idx:03d}"
        if rdir.exists():
            shutil.rmtree(rdir)
        for sub in ("images", "renders", "opacity"):
            (rdir / sub).mkdir(parents=True)
        mini_frames = []
        for pos, i in enumerate(order):
            frame = dict(frames[i])
            if i in done:
                frame["file_path"] = f"images/{pos:05d}.png"
                (rdir / "images" / f"{pos:05d}.png").symlink_to((out / "anchors" / f"{i:05d}.png").resolve())
            for sub in ("renders", "opacity"):
                (rdir / sub / f"{pos:05d}.png").symlink_to((removal_dir / sub / f"{i:05d}.png").resolve())
            mini_frames.append(frame)
        (rdir / "transforms.json").write_text(json.dumps({**transforms, "frames": mini_frames}))
        (rdir / "selected.json").write_text(json.dumps(selected))
        (rdir / "targets.json").write_text(json.dumps(targets))
        (rdir / "split.json").write_text(json.dumps({"test": {scene_id: {
            "transforms_path": "transforms.json", "image_root": ".", "render_dir": "renders", "opacity_dir": "opacity",
            "selected_indices_path": "selected.json", "target_indices_path": "targets.json",
            "prompt_path": str((scene_root / entry["prompt_path"]).resolve()),
            "camera_scale": entry["camera_scale"], "has_gt": False}}}))

        generate(rdir / "split.json", rdir / "gen", min(args.num_views, len(selected)))

        pred_dir = rdir / "gen" / scene_id / "frames" / "batch_0000" / "pred"
        for pos in targets:
            composite(order[pos], pred_dir / f"{pos:05d}.png")
            done[order[pos]] = round_idx
        print(f"round {round_idx}: generated {[c for _, c in chunks]} -> {len(done)}/{n} done", flush=True)
        if not args.keep_rounds:
            shutil.rmtree(rdir)

    (out / "order.json").write_text(json.dumps({f"{i:05d}": r for i, r in sorted(done.items())}, indent=1))
    print(f"Done: {n} frames in {round_idx} rounds -> {out}", flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_root", type=Path, required=True, help="Photo scene (<removal split>/scene).")
    parser.add_argument("--removal_dir", type=Path, required=True, help="artifixer_gs.removal split output (renders/opacity/hole).")
    parser.add_argument("--seed_pred_dir", type=Path, required=True, help="Pass-1 ArtiFixer pred frames.")
    parser.add_argument("--seed_frames", required=True, help="e.g. '0-6' or '0-6,116-122', or 'auto' (frames.json).")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--num_views", type=int, default=12)
    parser.add_argument("--feather_sigma", type=float, default=3.0)
    parser.add_argument("--jump_factor", type=float, default=4.0, help="Camera step > factor*median splits sequences.")
    parser.add_argument("--outside", choices=("photo", "render"), default="photo",
                        help="What anchors show outside the removal mask: the real photo, or the object-free render.")
    parser.add_argument("--keep_rounds", action="store_true")
    parser.add_argument("--checkpoint_pt", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--max_neighbors_per_encode", type=int, default=None,
                        help="Passed to run_inference; 1 VAE-encodes one reference at a time (low memory).")
    return parser


def main(argv: list[str] | None = None, generate: Generator | None = None) -> None:
    propagate(build_parser().parse_args(argv), generate)


if __name__ == "__main__":
    main()
