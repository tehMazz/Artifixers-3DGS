#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object removal on a 3DGS PLY, and the hole split that ArtiFixer inpaints.

Adapted from ArtiFixer's data_processing/render_flashsplat_extraction.py (the removal rule),
scripts/bakerh/derive_scene.py (frame subset, windows, auto seeds) and output/jobs/build_removal_split.py
(the hole split), with a PLY + a per-Gaussian object mask instead of 3DGUT + FlashSplat.

  render  deleted = the object Gaussians (--object_mask: bool [P] over the PLY vertices, e.g. BEA-GS
          labels.pt == k) plus every other Gaussian whose centre lies inside the 3D convex hull of the
          object centres: trimmed at --hull_trim percent of their distance to the centroid, at most 10k
          points, every plane pushed out by --hull_expand x their mean radius (the
          --background-convex-hull --convex-hull-trim-percentile 99.5 rule; BEA-GS labels come from
          multi-view votes, so there is no FlashSplat contribution filter). Writes removed.ply (kept),
          deleted.ply, deleted_mask.npy, hull_points.npy, removal.json and, on every photo frame of the
          prepared transforms.json, background/{renders,opacity}/ and deleted/{renders,opacity}/; with
          --depth also background/depth/ (what artifixer_gs.hole_mesh compares the hole mesh against).
  split   per frame, --hole_mode
            dilate (default): hole = 2D object mask | deleted opacity > --fg_opacity_threshold, optionally its
                    2D convex hull (--hole_shape hull), binary_dilation --dilate_px times (3x3 cross);
            alpha:  the pixels of that dilated hole where the removed scene is see-through (background opacity
                    < --bg_opacity_threshold) or the deleted Gaussians show (deleted opacity >
                    --fg_opacity_threshold);
            mesh:   alpha | <removal_dir>/mesh_front (artifixer_gs.hole_mesh: the hole mesh in front of the
                    removed scene).
          Optional, alpha / mesh: --confidence_dir + --conf_threshold also keep the pixels of the dilated hole
          where the model's per-view confidence is below the threshold (BEA-GS beags_artifixer.py confidence/,
          <photo stem>.png, uint8 = 255 * min(c, 1), resized BILINEAR; a frame without a map adds nothing; in
          dilate mode the whole dilated hole is kept anyway). Writes the
          reconstructed_colmap split for one all_frames rollout: renders/ (background render),
          opacity/ (background opacity, 0 in the hole: ArtiFixer regenerates it), hole/, refs/
          (reference views, never showing the object with --refs render), transforms.json (frames
          restricted to --frames, file_path -> refs), selected_indices.json (every frame), frames.json
          (source_indices, window_starts, auto_seeds) and split.json (has_gt false); and scene/, the
          photo scene restricted to --frames that artifixer_gs.propagate and distill read.

2D masks (--mask_dir): <photo stem>.npy or .png. A 2D map gives object = > 0 (or == --mask_label); a
(C,H,W) BEA-GS mask (channel 0 background, k object k) gives object = m[--mask_label or 1] > 0.5.
They are resized NEAREST to the render size.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, QhullError

from artifixer_gs.prepare import DEFAULT_CAMERA_CONDITIONING_SCALE, DEFAULT_CAPTION_FILENAME, prepared_paths, read_metric_scale
from artifixer_gs.render import render_ply
from data_processing.camera_trajectories import write_json

EMPTY_CAPTION = (
    "An empty floor without holes: one continuous, clean floor surface with nothing standing on it, in an industrial hall."
)
MAX_HULL_POINTS = 10_000
CHUNK = 65536
CHUNK_BYTES = 1 << 27
PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


# ---------------------------------------------------------------------------- frames


def parse_frame_ranges(spec: str) -> list[int]:
    """'0-6,116-122' -> [0..6, 116..122] (inclusive ranges, single numbers allowed, '+' = ',')."""
    indices: list[int] = []
    for part in spec.replace("+", ",").split(","):
        if not part.strip():
            continue
        a, _, b = part.strip().partition("-")
        indices += range(int(a), int(b or a) + 1)
    return indices


def select_frames(spec: str, frame_count: int) -> list[int]:
    """derive_scene --frames: 0-based inclusive ranges in transforms.json order; empty keeps every frame."""
    indices = parse_frame_ranges(spec) if spec else list(range(frame_count))
    assert indices and max(indices) < frame_count and min(indices) >= 0, f"--frames out of range [0, {frame_count - 1}]"
    assert len(set(indices)) == len(indices), "--frames overlap"
    return indices


def frame_windows(indices: list[int], seed_length: int = 7) -> tuple[list[int], str]:
    """Window starts (where the subset stops being contiguous) and auto seeds: the first seed_length
    frames of each window, as a propagate --seed_frames string."""
    window_starts = [k for k in range(len(indices)) if k == 0 or indices[k] != indices[k - 1] + 1]
    window_ends = window_starts[1:] + [len(indices)]
    auto_seeds = ",".join(f"{s}-{min(s + seed_length, e) - 1}" for s, e in zip(window_starts, window_ends))
    return window_starts, auto_seeds


# ---------------------------------------------------------------------------- 3D removal rule


def object_hull(object_points: np.ndarray, trim_percentile: float = 99.5, expand: float = 0.0) -> tuple[np.ndarray, float]:
    """Hull points (trimmed at trim_percentile of the centroid distance, <= 10k, seed 0) and the plane margin
    (expand x their mean radius), as render_flashsplat_extraction / aurafusion render_views."""
    assert 0 < trim_percentile <= 100, "--hull_trim must be in (0, 100]"
    points = np.asarray(object_points)
    if trim_percentile < 100:
        dist = np.linalg.norm(points - points.mean(0), axis=1)
        points = points[dist <= np.percentile(dist, trim_percentile)]
    if points.shape[0] < 4:
        raise SystemExit(f"At least four object Gaussians are required for a 3D convex hull, got {points.shape[0]}")
    if points.shape[0] > MAX_HULL_POINTS:
        points = points[np.random.default_rng(0).choice(points.shape[0], MAX_HULL_POINTS, replace=False)]
    margin = expand * float(np.linalg.norm(points - points.mean(0), axis=1).mean())  # hull planes are unit-normal
    return points, margin


def points_in_hull(points: np.ndarray, hull_points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Which points lie inside the convex hull of hull_points, every plane pushed out by margin."""
    try:
        hull = ConvexHull(hull_points)
    except QhullError as error:
        raise SystemExit(f"Could not compute object convex hull: {error}") from error
    tol = 1e-7 * max(float(np.ptp(hull_points, axis=0).max()), 1.0)
    # The originals test 65536 points at a time; fewer when the hull has many faces (same result, less memory).
    chunk = max(1, min(CHUNK, CHUNK_BYTES // (8 * len(hull.equations))))
    inside = np.zeros(len(points), dtype=bool)
    for s in range(0, len(points), chunk):
        p = points[s : s + chunk]
        inside[s : s + chunk] = ((p @ hull.equations[:, :3].T + hull.equations[:, 3]) <= tol + margin).all(axis=1)
    return inside


def removal_mask(
    means: np.ndarray, object_mask: np.ndarray, hull_trim: float = 99.5, hull_expand: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """deleted [P] = object | (not object and centre inside the object hull); also the hull points and margin."""
    object_mask = np.asarray(object_mask, dtype=bool)
    assert object_mask.shape == (len(means),), f"object mask {object_mask.shape} is not aligned with {len(means)} Gaussians"
    hull_points, margin = object_hull(means[object_mask], hull_trim, hull_expand)
    others = np.flatnonzero(~object_mask)
    deleted = object_mask.copy()
    deleted[others[points_in_hull(means[others], hull_points, margin)]] = True
    return deleted, hull_points, margin


# ---------------------------------------------------------------------------- PLY subsets


def read_ply_vertices(path: Path) -> tuple[list[str], np.ndarray]:
    """Header lines and the raw vertex records of a binary little-endian PLY (every property kept)."""
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").rstrip("\r\n")
            header.append(line)
            if line.strip() == "end_header":
                break
        assert header[0] == "ply", f"{path} is not a PLY file"
        assert "format binary_little_endian 1.0" in header, f"{path} must be binary_little_endian"
        elements = [line.split() for line in header if line.startswith("element ")]
        assert elements and elements[0][1] == "vertex", f"{path}: the first element must be vertex"
        assert all(int(e[2]) == 0 for e in elements[1:]), f"{path}: only the vertex element may hold data"
        count, fields, in_vertex = int(elements[0][2]), [], False
        for line in header:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                in_vertex = True
            elif parts and parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                assert parts[1] != "list", f"{path}: list properties are not supported"
                fields.append((parts[2], PLY_TYPES[parts[1]]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    assert len(data) == count, f"{path}: expected {count} vertices, read {len(data)}"
    return header, data


def write_ply_vertices(path: Path, header: list[str], data: np.ndarray) -> None:
    lines = [f"element vertex {len(data)}" if line.startswith("element vertex ") else line for line in header]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


def vertex_means(data: np.ndarray) -> np.ndarray:
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------- prepared photo scene


@dataclass(frozen=True)
class PhotoScene:
    scene_id: str
    # transforms.json of the photo frames (OpenGL c2w, file_path relative to image_root).
    transforms_path: Path
    image_root: Path
    prompt_path: Path
    metric_scale: float | None
    camera_scale: float | None


def photo_scene(path: Path) -> PhotoScene:
    """The photo scene of an artifixer_gs.prepare root (no split.json needed: --phases prepare,scale,caption
    is enough), or of a single-scene split.json whose frames are photos."""
    path = path.expanduser().resolve()
    if path.is_dir():
        paths = prepared_paths(path, path.name)
        transforms_path = paths.input_dir / "nerfstudio" / "transforms.json"
        assert transforms_path.is_file(), f"Missing {transforms_path}: run artifixer_gs.prepare --phases prepare first"
        scale_info = paths.scale_dir / "scale_info.txt"
        metric_scale = read_metric_scale(scale_info) if scale_info.is_file() else None
        return PhotoScene(
            scene_id=paths.scene_id,
            transforms_path=transforms_path,
            image_root=paths.input_dir,
            prompt_path=paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME,
            metric_scale=metric_scale,
            camera_scale=None if metric_scale is None else metric_scale * DEFAULT_CAMERA_CONDITIONING_SCALE,
        )
    (scene_id, entry), = json.loads(path.read_text())["test"].items()
    root = path.parent

    def resolve(value: str) -> Path:
        return Path(value) if Path(value).is_absolute() else root / value

    transforms_path = resolve(entry["transforms_path"])
    frames = json.loads(transforms_path.read_text())["frames"]
    assert all("file_path" in frame for frame in frames), (
        f"{transforms_path} has frames without file_path (a novel-path split): pass the prepared root instead"
    )
    return PhotoScene(
        scene_id=scene_id,
        transforms_path=transforms_path,
        image_root=resolve(entry["image_root"]),
        prompt_path=resolve(entry["prompt_path"]),
        metric_scale=entry.get("metric_scale"),
        camera_scale=float(entry["camera_scale"]),
    )


# ---------------------------------------------------------------------------- render


Renderer = Callable[..., Path]


def render_removal(args: argparse.Namespace, render: Renderer = render_ply) -> Path:
    scene = photo_scene(args.split_path)
    out = args.output_dir.expanduser().resolve()
    ply = args.ply.expanduser().resolve()
    header, vertices = read_ply_vertices(ply)
    object_mask = np.load(args.object_mask)
    assert object_mask.dtype == np.bool_ and object_mask.shape == (len(vertices),), (
        f"{args.object_mask} must be a bool mask over the {len(vertices)} Gaussians of {ply}, got {object_mask.dtype} {object_mask.shape}"
    )
    deleted, hull_points, margin = removal_mask(vertex_means(vertices), object_mask, args.hull_trim, args.hull_expand)
    n_object, n_deleted = int(object_mask.sum()), int(deleted.sum())
    print(
        f"{len(vertices)} Gaussians: object={n_object}, background deleted inside the hull={n_deleted - n_object}, "
        f"kept={len(vertices) - n_deleted} (hull trim {args.hull_trim}, expand {args.hull_expand})",
        flush=True,
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "removal.json").unlink(missing_ok=True)  # written last: its presence marks a finished render
    write_ply_vertices(out / "removed.ply", header, vertices[~deleted])
    write_ply_vertices(out / "deleted.ply", header, vertices[deleted])
    np.save(out / "deleted_mask.npy", deleted)
    np.save(out / "hull_points.npy", hull_points)

    frame_count = len(json.loads(scene.transforms_path.read_text())["frames"])
    selected = list(range(frame_count))
    render(ply=out / "removed.ply", trajectory_path=scene.transforms_path, output_dir=out / "background",
           selected_indices=selected, depth=args.depth)
    render(ply=out / "deleted.ply", trajectory_path=scene.transforms_path, output_dir=out / "deleted", selected_indices=selected)
    write_json(
        out / "removal.json",
        {
            "ply": str(ply),
            "object_mask": str(Path(args.object_mask).expanduser().resolve()),
            "scene_id": scene.scene_id,
            "transforms_path": str(scene.transforms_path),
            "frames": frame_count,
            "object": n_object,
            "deleted": n_deleted,
            "hull_deleted": n_deleted - n_object,
            "kept": len(vertices) - n_deleted,
            "hull_trim": args.hull_trim,
            "hull_expand": args.hull_expand,
            "hull_margin": margin,
            "depth": bool(args.depth),
        },
    )
    print(f"Wrote {out}: removed.ply, deleted.ply, background/ and deleted/ renders of {frame_count} frames", flush=True)
    return out


# ---------------------------------------------------------------------------- hole split


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_mask_2d(mask_dir: Path, stem: str, size: tuple[int, int], label: int | None = None) -> np.ndarray:
    """Object pixels [h,w] of <mask_dir>/<stem>.npy|.png at size (w,h), see the module docstring."""
    path = next((mask_dir / f"{stem}{suffix}" for suffix in (".npy", ".png") if (mask_dir / f"{stem}{suffix}").is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No 2D object mask {stem}.npy or {stem}.png in {mask_dir}")
    if path.suffix == ".npy":
        mask = np.load(path)
    else:
        with Image.open(path) as image:
            mask = np.asarray(image if image.mode in ("1", "L", "P", "I", "I;16", "F") else image.convert("L"))
    if mask.ndim == 3 and mask.shape[0] > 1:
        k = 1 if label is None else label
        assert 1 <= k < mask.shape[0], f"{path}: label {k} not in its {mask.shape[0]} channels"
        obj = mask[k].astype(np.float32) > 0.5
    else:
        mask = mask.reshape(mask.shape[-2:])
        obj = mask == label if label is not None else mask > 0
    w, h = size
    if obj.shape != (h, w):
        obj = np.asarray(Image.fromarray(obj.astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
    return obj


def convex_hull_fill(hole: np.ndarray) -> np.ndarray:
    """The 2D convex hull of the hole, rasterised; a collinear footprint (object at the border) is kept."""
    if np.count_nonzero(hole) < 3:
        return hole
    ys, xs = np.nonzero(hole)
    points = np.stack([xs, ys], 1)
    try:
        vertices = points[ConvexHull(points).vertices]
    except QhullError:
        return hole
    h, w = hole.shape
    polygon = Image.new("L", (w, h), 0)
    ImageDraw.Draw(polygon).polygon([tuple(int(v) for v in p) for p in vertices], fill=1)
    return np.asarray(polygon, dtype=bool)


HOLE_MODES = ("dilate", "alpha", "mesh", "gavis")


def hole_mask(
    object_2d: np.ndarray,
    deleted_opacity: np.ndarray,
    threshold: float = 0.05,
    hole_shape: str = "mask",
    dilate_px: int = 12,
    hole_mode: str = "dilate",
    background_opacity: np.ndarray | None = None,
    bg_threshold: float = 0.5,
    mesh_front: np.ndarray | None = None,
    low_confidence: np.ndarray | None = None,
    unseen: np.ndarray | None = None,
) -> np.ndarray:
    """dilate: (mask > 0) | (deleted alpha > threshold), optional 2D hull, dilated dilate_px times (cross element).
    alpha: that, where background alpha < bg_threshold or deleted alpha > threshold [or low_confidence].
    mesh: alpha | mesh_front.
    gavis: that, where unseen (GAVIS: the background there comes from Gaussians the training views never saw from
    this direction, or from nothing) or deleted alpha > threshold [or low_confidence]; None = alpha's rule."""
    assert hole_mode in HOLE_MODES, f"unknown hole mode {hole_mode!r} ({', '.join(HOLE_MODES)})"
    deleted = deleted_opacity / 255.0 > threshold
    hole = object_2d.astype(bool) | deleted
    if hole_shape == "hull":
        hole = convex_hull_fill(hole)
    if dilate_px > 0:
        hole = binary_dilation(hole, iterations=dilate_px)
    if hole_mode == "dilate":
        return hole
    assert background_opacity is not None, f"--hole_mode {hole_mode} needs the background opacity"
    if hole_mode == "gavis" and unseen is not None:
        assert unseen.shape == hole.shape, f"unseen map {unseen.shape} vs hole {hole.shape}"
        keep = unseen.astype(bool) | deleted
    else:
        keep = (background_opacity / 255.0 < bg_threshold) | deleted
    if low_confidence is not None:
        assert low_confidence.shape == hole.shape, f"confidence map {low_confidence.shape} vs hole {hole.shape}"
        keep = keep | low_confidence.astype(bool)
    hole = hole & keep
    if hole_mode == "mesh":
        assert mesh_front is not None and mesh_front.shape == hole.shape, "--hole_mode mesh needs a mesh_front mask per frame"
        hole = hole | mesh_front.astype(bool)
    return hole


def load_confidence(confidence_dir: Path, stem: str, size: tuple[int, int]) -> np.ndarray | None:
    """Confidence [h,w] in [0, 1] of <confidence_dir>/<stem>.png at size (w,h), None when the frame has no map."""
    path = confidence_dir / f"{stem}.png"
    if not path.is_file():
        return None
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_unseen(unseen_dir: Path, stem: str, size: tuple[int, int]) -> np.ndarray | None:
    """Unseen score [h,w] in [0, 1] of <unseen_dir>/<stem>.png (255 * U, high = unseen) at size (w,h), None when the
    frame has no map."""
    return load_confidence(unseen_dir, stem, size)


def write_photo_scene(out: Path, scene: PhotoScene, transforms: dict, indices: list[int], prompt_path: Path, frames_info: dict) -> Path:
    """derive_scene.py: split.json + transforms.json restricted to indices + frames.json (absolute paths)."""
    out.mkdir(parents=True, exist_ok=True)
    subset = [dict(transforms["frames"][i]) for i in indices]
    (out / "transforms.json").write_text(json.dumps({**transforms, "frames": subset}, indent=2))
    (out / "frames.json").write_text(json.dumps(frames_info, indent=1))
    entry = {
        "scene_id": scene.scene_id,
        "transforms_path": str((out / "transforms.json").resolve()),
        "image_root": str(scene.image_root.resolve()),
        "prompt_path": str(prompt_path),
        "camera_scale": scene.camera_scale,
        "metric_scale": scene.metric_scale,
    }
    (out / "split.json").write_text(json.dumps({"test": {scene.scene_id: entry}}, indent=2) + "\n")
    return out / "split.json"


def build_split(args: argparse.Namespace) -> Path:
    scene = photo_scene(args.scene_root)
    assert scene.camera_scale is not None, f"No metric scale for {args.scene_root}: run artifixer_gs.prepare --phases scale"
    removal_dir = args.removal_dir.expanduser().resolve()
    removal = json.loads((removal_dir / "removal.json").read_text())
    transforms = json.loads(scene.transforms_path.read_text())
    frames = transforms["frames"]
    assert removal["frames"] == len(frames), (
        f"{removal_dir} renders {removal['frames']} frames, but {scene.transforms_path} has {len(frames)}: rerun removal render"
    )
    indices = select_frames(args.frames, len(frames))
    window_starts, auto_seeds = frame_windows(indices, args.seed_length)
    prompt_path = (args.prompt_path or scene.prompt_path).expanduser().resolve()
    assert prompt_path.is_file(), f"Missing prompt {prompt_path}"
    mask_dir = args.mask_dir.expanduser().resolve()
    mesh_dir = (args.mesh_front_dir or removal_dir / "mesh_front").expanduser().resolve()
    if args.hole_mode == "mesh":
        assert mesh_dir.is_dir(), f"--hole_mode mesh: no {mesh_dir} (run artifixer_gs.hole_mesh after removal render --depth)"
    assert (args.confidence_dir is None) == (args.conf_threshold is None), "--confidence_dir and --conf_threshold go together"
    unseen_dir = None
    if args.hole_mode == "gavis":
        assert args.unseen_dir is not None, "--hole_mode gavis needs --unseen_dir (beags/unseen_map.py)"
        assert 0 <= args.unseen_threshold < 1, "--unseen_threshold is an unseen score in [0, 1)"
        unseen_dir = args.unseen_dir.expanduser().resolve()
        assert unseen_dir.is_dir(), f"no unseen dir {unseen_dir}"
    conf_dir = None
    if args.conf_threshold is not None:
        assert 0 < args.conf_threshold <= 1, "--conf_threshold is a confidence in (0, 1]"
        conf_dir = args.confidence_dir.expanduser().resolve()
        assert conf_dir.is_dir(), f"no confidence dir {conf_dir}"
        if args.hole_mode == "dilate":
            print("--hole_mode dilate keeps the whole dilated hole: the confidence maps add nothing", flush=True)

    out = args.output_dir.expanduser().resolve()
    for sub in ("renders", "opacity", "hole", "refs", "scene"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        if sub != "scene":
            (out / sub).mkdir(parents=True)
    frames_info = {"source_indices": indices, "window_starts": window_starts, "auto_seeds": auto_seeds}
    scene_split = write_photo_scene(out / "scene", scene, transforms, indices, prompt_path, frames_info)
    (out / "frames.json").write_text(json.dumps(frames_info, indent=1))

    subset, hole_fractions, conf_frames, conf_added, unseen_frames = [], [], 0, [], 0
    for k, i in enumerate(indices):
        frame = dict(frames[i])
        name = Path(frame["file_path"]).name
        background = load_rgb(removal_dir / "background" / "renders" / f"{i:05d}.png")
        background_opacity = load_gray(removal_dir / "background" / "opacity" / f"{i:05d}.png")
        deleted_opacity = load_gray(removal_dir / "deleted" / "opacity" / f"{i:05d}.png")
        real = load_rgb(scene.image_root / frame["file_path"])
        h, w = background.shape[:2]
        assert real.shape[:2] == (h, w), f"{name}: photo {real.shape[:2]} vs render {(h, w)}"
        object_2d = load_mask_2d(mask_dir, Path(name).stem, (w, h), args.mask_label)
        front = load_gray(mesh_dir / f"{i:05d}.png") > 127 if args.hole_mode == "mesh" else None
        low = None
        if conf_dir is not None:
            confidence = load_confidence(conf_dir, Path(name).stem, (w, h))
            low = None if confidence is None else confidence < args.conf_threshold
            conf_frames += low is not None
        unseen = None
        if unseen_dir is not None:
            score = load_unseen(unseen_dir, Path(name).stem, (w, h))
            unseen = None if score is None else score > args.unseen_threshold
            unseen_frames += unseen is not None
        hole_args = (object_2d, deleted_opacity, args.fg_opacity_threshold, args.hole_shape, args.dilate_px,
                     args.hole_mode, background_opacity, args.bg_opacity_threshold, front)
        hole = hole_mask(*hole_args, low, unseen)
        if low is not None and args.hole_mode != "dilate":
            conf_added.append((hole & ~hole_mask(*hole_args, None, unseen)).mean())
        hole_fractions.append(hole.mean())

        Image.fromarray(background).save(out / "renders" / f"{k:05d}.png")
        opacity = np.where(hole & (args.opacity == "hole0"), 0, background_opacity).astype(np.uint8)
        Image.fromarray(opacity).save(out / "opacity" / f"{k:05d}.png")
        Image.fromarray((hole * 255).astype(np.uint8)).save(out / "hole" / f"{k:05d}.png")
        ref_path = (out / "refs" / frame["file_path"]).with_suffix(".png")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref = {"composite": np.where(hole[..., None], background, real), "render": background, "photo": real}[args.refs]
        Image.fromarray(ref).save(ref_path)
        frame["file_path"] = ref_path.relative_to(out / "refs").as_posix()
        subset.append(frame)

    assert conf_dir is None or conf_frames, f"no confidence map in {conf_dir} is named after a selected photo"
    assert unseen_dir is None or unseen_frames, f"no unseen map in {unseen_dir} is named after a selected photo"
    (out / "transforms.json").write_text(json.dumps({**transforms, "frames": subset}, indent=2))
    (out / "selected_indices.json").write_text(json.dumps(list(range(len(subset)))))
    entry = {
        "scene_id": scene.scene_id,
        "transforms_path": str(out / "transforms.json"),
        "image_root": str(out / "refs"),
        "render_dir": str(out / "renders"),
        "opacity_dir": str(out / "opacity"),
        "selected_indices_path": str(out / "selected_indices.json"),
        "prompt_path": str(prompt_path),
        "camera_scale": scene.camera_scale,
        "metric_scale": scene.metric_scale,
        "has_gt": False,
        # Not read by run_inference: where propagate / distill removal_handoff find the photos and the removal.
        "scene_root": str(scene_split.parent),
        "removal_dir": str(removal_dir),
        "hole_shape": args.hole_shape,
        "dilate_px": args.dilate_px,
        "hole_mode": args.hole_mode,
    }
    if conf_dir is not None:
        entry.update(confidence_dir=str(conf_dir), conf_threshold=args.conf_threshold, confidence_frames=conf_frames)
    if unseen_dir is not None:
        entry.update(unseen_dir=str(unseen_dir), unseen_threshold=args.unseen_threshold, unseen_frames=unseen_frames)
    (out / "split.json").write_text(json.dumps({"test": {scene.scene_id: entry}}, indent=2) + "\n")
    if conf_dir is not None:
        print(f"confidence < {args.conf_threshold}: maps for {conf_frames}/{len(subset)} frames, "
              f"hole fraction added mean={np.mean(conf_added) if conf_added else 0.0:.4f}", flush=True)
    if unseen_dir is not None:
        print(f"unseen > {args.unseen_threshold}: maps for {unseen_frames}/{len(subset)} frames "
              f"(the others use alpha's rule)", flush=True)
    print(
        f"wrote {out / 'split.json'}: {len(subset)}/{len(frames)} frames, windows at {window_starts}, auto seeds {auto_seeds}, "
        f"hole fraction mean={np.mean(hole_fractions):.3f} min={np.min(hole_fractions):.3f} max={np.max(hole_fractions):.3f}",
        flush=True,
    )
    return out / "split.json"


# ---------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render", help="Delete the object (+ its 3D hull) from the PLY and render background / deleted.")
    r.add_argument("--ply", type=Path, required=True, help="Full-scene 3DGS PLY.")
    r.add_argument("--object_mask", type=Path, required=True, help="bool [P] .npy aligned with the PLY vertices.")
    r.add_argument("--split_path", type=Path, required=True, help="Prepared root (artifixer_gs.prepare) or its split.json.")
    r.add_argument("--output_dir", type=Path, required=True)
    r.add_argument("--hull_trim", type=float, default=99.5, help="Hull points kept within this centroid-distance percentile.")
    r.add_argument("--hull_expand", type=float, default=0.0, help="Grow the hull by this fraction of the object's mean radius.")
    r.add_argument("--depth", action="store_true", help="Also render background/depth/ (for --hole_mode mesh).")
    s = sub.add_parser("split", help="Hole masks + the reconstructed_colmap split for removal inpainting.")
    s.add_argument("--scene_root", type=Path, required=True, help="Prepared root (artifixer_gs.prepare) or its split.json.")
    s.add_argument("--removal_dir", type=Path, required=True, help="removal render output.")
    s.add_argument("--mask_dir", type=Path, required=True, help="2D object masks <photo stem>.npy|.png.")
    s.add_argument("--mask_label", type=int, default=None, help="Object id in the masks (default: > 0, or channel 1).")
    s.add_argument("--output_dir", type=Path, required=True)
    s.add_argument("--hole_shape", choices=("mask", "hull"), default="mask",
                   help="hull: fill the 2D convex hull of the footprint, hiding the object's silhouette.")
    s.add_argument("--dilate_px", type=int, default=12)
    s.add_argument("--fg_opacity_threshold", type=float, default=0.05)
    s.add_argument("--hole_mode", choices=HOLE_MODES, default="dilate",
                   help="dilate: the dilated mask; alpha: its see-through / deleted pixels; mesh: alpha | mesh_front; "
                        "gavis: its unseen (--unseen_dir > --unseen_threshold) / deleted pixels.")
    s.add_argument("--bg_opacity_threshold", type=float, default=0.5, help="alpha / mesh: background see-through below this.")
    s.add_argument("--mesh_front_dir", type=Path, default=None, help="mesh: hole_mesh output (default <removal_dir>/mesh_front).")
    s.add_argument("--confidence_dir", type=Path, default=None,
                   help="Per-view confidence maps <photo stem>.png (255 * min(c, 1)); with --conf_threshold.")
    s.add_argument("--conf_threshold", type=float, default=None,
                   help="alpha / mesh: also keep the dilated-hole pixels with confidence below this (default: off).")
    s.add_argument("--unseen_dir", type=Path, default=None,
                   help="gavis: per-view unseen maps <photo stem>.png (255 * U, high = unseen; beags/unseen_map.py).")
    s.add_argument("--unseen_threshold", type=float, default=0.5, help="gavis: unseen score above this is in the hole.")
    s.add_argument("--frames", default="", help="e.g. '49-148,284-304'; empty keeps every frame.")
    s.add_argument("--seed_length", type=int, default=7, help="Frames per auto seed range (one per window).")
    s.add_argument("--prompt_path", type=Path, default=None, help="caption.h5 to use instead of the prepared one.")
    s.add_argument("--refs", choices=("render", "composite", "photo"), default="render",
                   help="Reference views: pure background render, photo with the hole patched, or raw photo.")
    s.add_argument("--opacity", choices=("hole0", "render"), default="hole0",
                   help="hole0: zero opacity in the hole; render: raw background opacity.")
    return parser


def main(argv: list[str] | None = None, render: Renderer = render_ply) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "render":
        render_removal(args, render)
    else:
        build_split(args)


if __name__ == "__main__":
    main()
