#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ArtiFixer3D with an external 3DGS trainer: build its training scene, then prepare ArtiFixer3D+.

Adapted from ArtiFixer's data_processing/artifixer3d.py, minus the 3DGUT training.

  handoff  COLMAP scene that any 3DGS trainer reads: real anchor photos + ArtiFixer predictions at
           their camera poses, PINHOLE cameras, the source points3D.bin for initialisation, and
           generated_images.txt (the prediction images; ArtiFixer3D supervises them with LPIPS only,
           see FastGS train.py --lpips_images).
  plus     render the PLY the trainer produced on every frame of the prepared split and write
           split_artifixer3d_plus.json for the second ArtiFixer pass (ArtiFixer3D+).
  removal_handoff
           object removal (ArtiFixer's output/jobs/build_af3d_split.py + handoff): every image is the
           object-free composite of artifixer_gs.propagate at its photo pose; anchors = every
           --anchor_every-th frame + the seeds (L1 + D-SSIM), the rest are listed in
           generated_images.txt (LPIPS only). Also writes split.json + transforms.json there, the
           ArtiFixer3D split that `plus` reads (val_frames = the non-anchors, as the 3DGUT AF3D+).
           points3D.bin is copied as is (ArtiFixer does the same), or without the points inside the
           removed object's hull with --drop_points_mask.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from artifixer_gs.colmap import CAMERA_MODEL_NAMES, Camera, read_colmap_extrinsics_binary, read_colmap_intrinsics_binary
from artifixer_gs.removal import parse_frame_ranges, points_in_hull
from artifixer_gs.render import DISTORTION_KEYS, check_pinhole, render_ply
from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
    write_json,
)

GENERATED_IMAGES_FILENAME = "generated_images.txt"
POSE_TOLERANCE_DEGREES = 1e-3
POSE_TOLERANCE_TRANSLATION = 1e-4


@dataclass(frozen=True)
class PreparedScene:
    scene_id: str
    split_path: Path
    # transforms.json defining frame order, intrinsics and OpenGL c2w poses of every frame.
    transforms_path: Path
    # Prepared COLMAP root with the real photos (split image_root).
    colmap_dir: Path
    prompt_path: Path
    camera_scale: float
    has_gt: bool
    # Frames that keep their real photo; every other frame takes the ArtiFixer prediction.
    selected_indices: list[int]
    target_indices_path: Path | None
    frame_count: int


def resolve(split_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else split_root / path


def load_prepared_scene(split_path: Path, scene_id: str | None = None) -> PreparedScene:
    split_path = split_path.resolve()
    scenes = json.loads(split_path.read_text())["test"]
    if scene_id is None:
        assert len(scenes) == 1, f"--scene_id is required when {split_path} has {len(scenes)} scenes"
        scene_id = next(iter(scenes))
    meta = scenes[scene_id]
    root = split_path.parent
    transforms_path = resolve(root, meta["transforms_path"])
    selected_indices = json.loads(resolve(root, meta["selected_indices_path"]).read_text())
    return PreparedScene(
        scene_id=scene_id,
        split_path=split_path,
        transforms_path=transforms_path,
        colmap_dir=resolve(root, meta["image_root"]),
        prompt_path=resolve(root, meta["prompt_path"]),
        camera_scale=float(meta["camera_scale"]),
        has_gt=bool(meta.get("has_gt", True)),
        selected_indices=[int(index) for index in selected_indices],
        target_indices_path=resolve(root, meta["target_indices_path"]) if "target_indices_path" in meta else None,
        frame_count=len(json.loads(transforms_path.read_text())["frames"]),
    )


def generated_frame_indices(scene: PreparedScene) -> list[int]:
    selected = set(scene.selected_indices)
    assert len(selected) == len(scene.selected_indices), f"Duplicate selected indices: {scene.selected_indices}"
    return [index for index in range(scene.frame_count) if index not in selected]


def pinhole_camera(camera_id: int, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> Camera:
    return Camera(id=camera_id, model="PINHOLE", width=int(width), height=int(height), params=np.array([fx, fy, cx, cy], dtype=np.float64))


def pinhole_from_colmap(camera_id: int, camera: Camera, image_size: tuple[int, int]) -> Camera:
    """Source COLMAP camera scaled to image_size, as PINHOLE (3DGS trainers take no distortion)."""
    sx, sy = image_size[0] / camera.width, image_size[1] / camera.height
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx, fy, distortion = f, f, []
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        distortion = []
    elif camera.model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy, *distortion = camera.params
        fx = fy = f
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, *distortion = camera.params
    else:
        assert False, f"Unsupported COLMAP camera model: {camera.model}"
    check_pinhole(distortion, f"COLMAP camera {camera.id} ({camera.model})")
    return pinhole_camera(camera_id, image_size[0], image_size[1], fx * sx, fy * sy, cx * sx, cy * sy)


def colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return (qvec if qvec[0] >= 0 else -qvec) / np.linalg.norm(qvec)


def write_cameras(path: Path, cameras: list[Camera]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            fid.write(struct.pack("<iiQQ", camera.id, CAMERA_MODEL_NAMES[camera.model].model_id, camera.width, camera.height))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def write_images(path: Path, images: list[tuple[int, np.ndarray, np.ndarray, int, str]]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, camera_id, name in images:
            fid.write(struct.pack("<idddddddi", image_id, *qvec, *tvec, camera_id))
            fid.write(name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", 0))


def shared_camera_id(cameras: list[Camera], ids: dict[tuple, int], camera: Camera) -> int:
    key = (camera.width, camera.height, tuple(np.round(camera.params, 6)))
    if key not in ids:
        ids[key] = camera.id
        cameras.append(camera)
    return ids[key]


def link_or_resize(source: Path, target: Path, size: tuple[int, int]) -> None:
    """Symlink source, or write a resized copy when it is not size (ArtiFixer outputs its own resolution)."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing frame: {source}")
    with PILImage.open(source) as image:
        if image.size == size:
            target.symlink_to(source.resolve())
        else:
            image.convert("RGB").resize(size, resample=PILImage.Resampling.LANCZOS).save(target)


def handoff(scene: PreparedScene, artifixer_frames_dir: Path, output_dir: Path) -> Path:
    """Write the trainer's COLMAP scene: anchors keep their exact source pose, predictions take the split pose."""
    generated = generated_frame_indices(scene)
    assert generated, "Every frame is an anchor: there is nothing to distill"
    missing = [index for index in generated if not (artifixer_frames_dir / f"{index:05d}.png").is_file()]
    assert not missing, f"Missing ArtiFixer predictions in {artifixer_frames_dir}: {missing[:10]} ({len(missing)} missing)"

    transforms = json.loads(scene.transforms_path.read_text())
    frames = transforms["frames"]
    applied_transform = applied_transform_matrix(transforms)
    sparse_dir = scene.colmap_dir / "sparse" / "0"
    source_images = {Path(image.name).name: image for image in read_colmap_extrinsics_binary(sparse_dir / "images.bin")}
    source_cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")

    if output_dir.exists():
        assert output_dir.is_dir() and not output_dir.is_symlink(), f"Expected a directory: {output_dir}"
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "sparse" / "0").mkdir(parents=True)

    selected = set(scene.selected_indices)
    cameras: list[Camera] = []
    camera_ids: dict[tuple, int] = {}
    images, generated_names = [], []
    for index, frame in enumerate(frames):
        w2c = opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform)
        if index in selected:
            source = scene.colmap_dir / str(frame["file_path"])
            source_image = source_images[Path(str(frame["file_path"])).name]
            with PILImage.open(source) as image:
                size = image.size
            camera = pinhole_from_colmap(len(cameras) + 1, source_cameras[source_image.camera_id], size)
            qvec, tvec = source_image.qvec, source_image.tvec
            # The split pose must round-trip to the source COLMAP pose, or the conventions disagree.
            error = (Rotation.from_quat(np.roll(colmap_qvec(w2c[:3, :3]), -1)) * Rotation.from_quat(np.roll(qvec, -1)).inv()).magnitude()
            assert np.degrees(error) < POSE_TOLERANCE_DEGREES and np.linalg.norm(w2c[:3, 3] - tvec) < POSE_TOLERANCE_TRANSLATION, (
                f"Anchor frame {index} ({source.name}) does not match its COLMAP pose; regenerate the split"
            )
            name = f"frame_{index:05d}{source.suffix.lower()}"
            (output_dir / "images" / name).symlink_to(source.resolve())
        else:
            intrinsics = camera_intrinsics_for_frame(transforms, frame)
            check_pinhole([intrinsics[key] for key in DISTORTION_KEYS], f"frame {index}")
            camera = pinhole_camera(
                len(cameras) + 1, intrinsics["w"], intrinsics["h"], intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"]
            )
            qvec, tvec = colmap_qvec(w2c[:3, :3]), w2c[:3, 3]
            name = f"frame_{index:05d}.png"
            link_or_resize(artifixer_frames_dir / f"{index:05d}.png", output_dir / "images" / name, (camera.width, camera.height))
            generated_names.append(name)
        camera_id = shared_camera_id(cameras, camera_ids, camera)
        images.append((index + 1, qvec, tvec, camera_id, name))

    write_cameras(output_dir / "sparse" / "0" / "cameras.bin", cameras)
    write_images(output_dir / "sparse" / "0" / "images.bin", images)
    shutil.copyfile((sparse_dir / "points3D.bin").resolve(), output_dir / "sparse" / "0" / "points3D.bin")
    (output_dir / GENERATED_IMAGES_FILENAME).write_text("\n".join(generated_names) + "\n")
    write_json(output_dir / "selected_indices.json", scene.selected_indices)
    print(
        f"Wrote {output_dir}: {len(selected)} photos + {len(generated_names)} ArtiFixer frames, "
        f"{len(cameras)} camera(s); LPIPS-only list {output_dir / GENERATED_IMAGES_FILENAME}",
        flush=True,
    )
    return output_dir


def filter_points3d(source: Path, target: Path, keep: Callable[[np.ndarray], np.ndarray]) -> tuple[int, int]:
    """Copy a COLMAP points3D.bin, keeping the points whose xyz [M,3] keep() marks (tracks copied raw)."""
    with source.open("rb") as fid:
        count = struct.unpack("<Q", fid.read(8))[0]
        records, xyzs = [], []
        for _ in range(count):
            head = fid.read(43)
            track_length = struct.unpack("<Q", fid.read(8))[0]
            records.append(head + struct.pack("<Q", track_length) + fid.read(8 * track_length))
            xyzs.append(struct.unpack("<ddd", head[8:32]))
    mask = keep(np.asarray(xyzs, dtype=np.float64).reshape(-1, 3)) if count else np.zeros(0, dtype=bool)
    with target.open("wb") as fid:
        fid.write(struct.pack("<Q", int(mask.sum())))
        for record, kept in zip(records, mask):
            if kept:
                fid.write(record)
    return count, int(mask.sum())


def removal_handoff(
    removal_split: Path,
    propagated_dir: Path,
    output_dir: Path,
    anchor_every: int = 3,
    seed_frames: str | None = None,
    drop_points_mask: bool = False,
) -> Path:
    """FastGS scene of the propagated composites + the ArtiFixer3D split (anchors = every k-th + seeds)."""
    (scene_id, entry), = json.loads(removal_split.read_text())["test"].items()
    scene_root = Path(entry.get("scene_root", removal_split.parent / "scene"))
    (_, scene_entry), = json.loads((scene_root / "split.json").read_text())["test"].items()
    transforms = json.loads((scene_root / scene_entry["transforms_path"]).read_text())
    frames = transforms["frames"]
    photo_root = scene_root / scene_entry["image_root"]
    applied_transform = applied_transform_matrix(transforms)
    if seed_frames is None:
        seeds = [int(i) for i, r in json.loads((propagated_dir / "order.json").read_text()).items() if r == 0]
    else:
        seeds = parse_frame_ranges(seed_frames)
    assert anchor_every >= 1, "--anchor_every must be >= 1"
    anchors = sorted(set(range(0, len(frames), anchor_every)) | set(seeds))
    assert max(anchors) < len(frames), f"seed frames {seeds} outside the {len(frames)} frames"
    missing = [i for i in range(len(frames)) if not (propagated_dir / "anchors" / f"{i:05d}.png").is_file()]
    assert not missing, f"Missing composites in {propagated_dir / 'anchors'}: {missing[:10]} ({len(missing)} missing)"

    sparse_dir = photo_root / "sparse" / "0"
    source_images = {Path(image.name).name: image for image in read_colmap_extrinsics_binary(sparse_dir / "images.bin")}
    source_cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")
    if output_dir.exists():
        assert output_dir.is_dir() and not output_dir.is_symlink(), f"Expected a directory: {output_dir}"
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "sparse" / "0").mkdir(parents=True)

    anchor_set = set(anchors)
    cameras: list[Camera] = []
    camera_ids: dict[tuple, int] = {}
    images, generated_names, af3d_frames = [], [], []
    for index, frame in enumerate(frames):
        name = Path(str(frame["file_path"])).name
        assert name in source_images, f"Frame {index} ({name}) is not in {sparse_dir / 'images.bin'}"
        source_image = source_images[name]
        intrinsics = camera_intrinsics_for_frame(transforms, frame)
        size = (int(intrinsics["w"]), int(intrinsics["h"]))
        camera = pinhole_from_colmap(len(cameras) + 1, source_cameras[source_image.camera_id], size)
        # Every frame is a photo pose: keep the source COLMAP pose, checked against the split pose.
        w2c = opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform)
        error = (Rotation.from_quat(np.roll(colmap_qvec(w2c[:3, :3]), -1)) * Rotation.from_quat(np.roll(source_image.qvec, -1)).inv()).magnitude()
        assert np.degrees(error) < POSE_TOLERANCE_DEGREES and np.linalg.norm(w2c[:3, 3] - source_image.tvec) < POSE_TOLERANCE_TRANSLATION, (
            f"Frame {index} ({name}) does not match its COLMAP pose; regenerate the removal split"
        )
        image_name = f"frame_{index:05d}.png"
        link_or_resize(propagated_dir / "anchors" / f"{index:05d}.png", output_dir / "images" / image_name, size)
        if index not in anchor_set:
            generated_names.append(image_name)
        camera_id = shared_camera_id(cameras, camera_ids, camera)
        images.append((index + 1, source_image.qvec, source_image.tvec, camera_id, image_name))
        af3d_frames.append({**frame, "file_path": f"images/{image_name}"})

    write_cameras(output_dir / "sparse" / "0" / "cameras.bin", cameras)
    write_images(output_dir / "sparse" / "0" / "images.bin", images)
    points_source = (sparse_dir / "points3D.bin").resolve()
    if drop_points_mask:
        removal_dir = Path(entry.get("removal_dir", ""))
        hull_points = np.load(removal_dir / "hull_points.npy").astype(np.float64)
        margin = float(json.loads((removal_dir / "removal.json").read_text())["hull_margin"])
        total, kept = filter_points3d(
            points_source, output_dir / "sparse" / "0" / "points3D.bin", lambda xyz: ~points_in_hull(xyz, hull_points, margin)
        )
        print(f"points3D.bin: dropped {total - kept}/{total} points inside the removed object's hull", flush=True)
    else:
        shutil.copyfile(points_source, output_dir / "sparse" / "0" / "points3D.bin")
    (output_dir / GENERATED_IMAGES_FILENAME).write_text("\n".join(generated_names) + "\n")
    write_json(output_dir / "selected_indices.json", anchors)
    (output_dir / "transforms.json").write_text(json.dumps({**transforms, "frames": af3d_frames}, indent=2))
    prompt_path = Path(scene_entry["prompt_path"])
    af3d_entry = {
        "scene_id": scene_id,
        "transforms_path": "transforms.json",
        "image_root": ".",
        "selected_indices_path": "selected_indices.json",
        "prompt_path": str((prompt_path if prompt_path.is_absolute() else scene_root / prompt_path).resolve()),
        "camera_scale": entry["camera_scale"],
        "metric_scale": entry.get("metric_scale"),
        "has_gt": False,
    }
    write_json(output_dir / "split.json", {"test": {scene_id: af3d_entry}})
    print(
        f"Wrote {output_dir}: {len(anchors)} anchor composites + {len(generated_names)} LPIPS-only composites, "
        f"{len(cameras)} camera(s); ArtiFixer3D split {output_dir / 'split.json'}",
        flush=True,
    )
    return output_dir


def path_for_split(split_path: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(split_path.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def plus(scene: PreparedScene, ply: Path, render_dir: Path, output_split: Path) -> Path:
    """Render ply on every split frame and write the ArtiFixer3D+ split (same anchors, same targets)."""
    render_ply(ply=ply, trajectory_path=scene.transforms_path, output_dir=render_dir, selected_indices=scene.selected_indices)
    entry = {
        "scene_id": scene.scene_id,
        "transforms_path": path_for_split(output_split, scene.transforms_path),
        "image_root": path_for_split(output_split, scene.colmap_dir),
        "render_dir": path_for_split(output_split, render_dir / "renders"),
        "opacity_dir": path_for_split(output_split, render_dir / "opacity"),
        "selected_indices_path": path_for_split(output_split, render_dir / "selected_indices.json"),
        "prompt_path": path_for_split(output_split, scene.prompt_path),
        "reconstruction_checkpoint": path_for_split(output_split, ply),
        "camera_scale": scene.camera_scale,
    }
    if scene.target_indices_path is not None:
        entry["target_indices_path"] = path_for_split(output_split, scene.target_indices_path)
    if not scene.has_gt:
        entry["has_gt"] = False
    write_json(output_split, {"test": {scene.scene_id: entry}})
    print(f"Wrote ArtiFixer3D+ split {output_split}", flush=True)
    return output_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("handoff", help="Build the 3DGS trainer's COLMAP scene from photos + ArtiFixer frames.")
    h.add_argument("--split_path", type=Path, required=True, help="split.json written by artifixer_gs.prepare.")
    h.add_argument("--artifixer_frames_dir", type=Path, required=True, help="<save_dir>/<scene>/frames/batch_0000/pred")
    h.add_argument("--output_dir", type=Path, required=True)
    h.add_argument("--scene_id", default=None)
    p = sub.add_parser("plus", help="Render the trained PLY and write the ArtiFixer3D+ split.")
    p.add_argument("--split_path", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True, help="PLY trained on the handoff scene.")
    p.add_argument("--render_dir", type=Path, required=True)
    p.add_argument("--output_split", type=Path, required=True)
    p.add_argument("--scene_id", default=None)
    r = sub.add_parser("removal_handoff", help="FastGS scene + ArtiFixer3D split from the propagated removal composites.")
    r.add_argument("--removal_split", type=Path, required=True, help="split.json of artifixer_gs.removal split.")
    r.add_argument("--propagated_dir", type=Path, required=True, help="artifixer_gs.propagate output (anchors/, order.json).")
    r.add_argument("--output_dir", type=Path, required=True)
    r.add_argument("--anchor_every", type=int, default=3)
    r.add_argument("--seed_frames", default=None, help="Extra anchors, e.g. '0-6'; default: the seeds in order.json.")
    r.add_argument("--drop_points_mask", action="store_true",
                   help="Drop the COLMAP points inside the removed object's hull from the initialisation.")
    args = parser.parse_args()

    if args.command == "removal_handoff":
        removal_handoff(
            args.removal_split.resolve(), args.propagated_dir.resolve(), args.output_dir.resolve(),
            args.anchor_every, args.seed_frames, args.drop_points_mask,
        )
        return
    scene = load_prepared_scene(args.split_path, args.scene_id)
    if args.command == "handoff":
        handoff(scene, args.artifixer_frames_dir.resolve(), args.output_dir.resolve())
    else:
        plus(scene, args.ply.resolve(), args.render_dir.resolve(), args.output_split.resolve())


if __name__ == "__main__":
    main()
