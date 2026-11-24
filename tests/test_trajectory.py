# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of artifixer_gs/trajectory.py (port of ArtiFixer's tests/test_make_trajectory.py).

The scene is an analytic box room (walls, floor and ceiling; optional pillar and doorway), both as
a ray-cast renderer standing in for the gsplat render checks and as tiled flat Gaussians in a 3DGS
PLY for the occupancy checks. World up is +y, cameras are OpenGL camera-to-world as in
transforms.json. No gsplat needed.

  PYTHONPATH=. python -m pytest tests/test_trajectory.py

With the ArtiFixer repo at $ARTIFIXER_REPO (default: the Leonardo checkout), the equivalence
tests run the original scripts/bakerh/make_trajectory.py on the same scene (3DGUT-style checkpoint
+ FlashSplat labels) and require the same path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from artifixer_gs import trajectory as mt  # noqa: E402

UP = np.array([0.0, 1.0, 0.0])
INTR = (48.0, 48.0, 32.0, 24.0, 64, 48)  # fx, fy, cx, cy, w, h of the check camera
AF_REPO = Path(os.environ.get("ARTIFIXER_REPO", "/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer"))
AF_SCRIPT = AF_REPO / "scripts/bakerh/make_trajectory.py"


def look_at(position, target):
    return np.r_[np.c_[mt.look_at(np.asarray(position, float), np.asarray(target, float), UP), position], [[0, 0, 0, 1]]]


class Room:
    """Axis-aligned room [x0, x1] x [0, height] x [z0, z1], an optional vertical pillar (x, z, r) and
    an optional doorway (z-range x y-range) in the wall x = x1."""

    def __init__(self, x=(-3, 3), z=(-2, 2), height=3.0, pillar=None, doorway=None):
        self.x, self.z, self.height, self.pillar, self.doorway = x, z, height, pillar, doorway

    def render(self, c2w_gl):
        fx, fy, cx, cy, w, h = INTR
        v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)  # OpenCV, z = 1
        depths, opacities = [], []
        for pose in np.asarray(c2w_gl):
            c2w = pose @ mt.FLIP
            o, d = c2w[:3, 3], rays @ c2w[:3, :3].T
            t = np.full(len(d), np.inf)
            for axis, value, (a, b), bounds in (
                (0, self.x[0], (1, 2), ((0, self.height), self.z)), (0, self.x[1], (1, 2), ((0, self.height), self.z)),
                (2, self.z[0], (0, 1), (self.x, (0, self.height))), (2, self.z[1], (0, 1), (self.x, (0, self.height))),
                (1, 0.0, (0, 2), (self.x, self.z)), (1, self.height, (0, 2), (self.x, self.z)),
            ):
                with np.errstate(divide="ignore", invalid="ignore"):
                    tt = (value - o[axis]) / d[:, axis]
                p = o + tt[:, None] * d
                hit = (tt > 1e-9) & (p[:, a] >= bounds[0][0]) & (p[:, a] <= bounds[0][1]) \
                    & (p[:, b] >= bounds[1][0]) & (p[:, b] <= bounds[1][1])
                if axis == 0 and value == self.x[1] and self.doorway is not None:
                    (z0, z1), (y0, y1) = self.doorway
                    hit &= ~((p[:, 2] > z0) & (p[:, 2] < z1) & (p[:, 1] > y0) & (p[:, 1] < y1))
                t = np.where(hit & (tt < t), tt, t)
            if self.pillar is not None:
                px, pz, r = self.pillar
                ox, oz = o[0] - px, o[2] - pz
                a2 = d[:, 0] ** 2 + d[:, 2] ** 2
                b2 = 2 * (ox * d[:, 0] + oz * d[:, 2])
                disc = b2**2 - 4 * a2 * (ox**2 + oz**2 - r**2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tt = (-b2 - np.sqrt(disc)) / (2 * a2)
                y = o[1] + tt * d[:, 1]
                hit = (disc > 0) & (tt > 1e-9) & (y > 0) & (y < self.height)
                t = np.where(hit & (tt < t), tt, t)
            found = np.isfinite(t)
            depths.append(np.where(found, t, 0.0).reshape(h, w))  # z-depth: the rays have z = 1
            opacities.append(found.astype(float).reshape(h, w))
        return torch.tensor(np.array(depths)), torch.tensor(np.array(opacities))

    def gaussians(self, step=0.05, blob=None):
        """Flat Gaussians tiling every surface (and an isotropic blob box (centre, half) if given)."""
        mu, scale = [], []
        xs, zs = np.arange(self.x[0], self.x[1] + 1e-9, step), np.arange(self.z[0], self.z[1] + 1e-9, step)
        ys = np.arange(0, self.height + 1e-9, step)
        for value in self.x:
            g = np.stack(np.meshgrid([value], ys, zs, indexing="ij"), -1).reshape(-1, 3)
            if value == self.x[1] and self.doorway is not None:
                (z0, z1), (y0, y1) = self.doorway
                g = g[~((g[:, 2] > z0) & (g[:, 2] < z1) & (g[:, 1] > y0) & (g[:, 1] < y1))]
            mu.append(g)
            scale.append(np.tile([0.002, step, step], (len(g), 1)))
        for value in self.z:
            g = np.stack(np.meshgrid(xs, ys, [value], indexing="ij"), -1).reshape(-1, 3)
            mu.append(g)
            scale.append(np.tile([step, step, 0.002], (len(g), 1)))
        for value in (0.0, self.height):
            g = np.stack(np.meshgrid(xs, [value], zs, indexing="ij"), -1).reshape(-1, 3)
            mu.append(g)
            scale.append(np.tile([step, 0.002, step], (len(g), 1)))
        if self.pillar is not None:
            px, pz, r = self.pillar
            a, y = np.meshgrid(np.arange(0, 2 * np.pi, step / r), ys)
            g = np.stack([px + r * np.cos(a), y, pz + r * np.sin(a)], -1).reshape(-1, 3)
            mu.append(g)
            scale.append(np.full((len(g), 3), step))
        n_scene = sum(len(m) for m in mu)
        if blob is not None:
            centre, half = blob
            g = np.random.default_rng(0).uniform(-1, 1, (6000, 3)) * half + centre
            mu.append(g)
            scale.append(np.full((len(g), 3), 0.04))
        mu, scale = np.concatenate(mu), np.concatenate(scale)
        return mu, scale, n_scene


def write_ply(path, mu, scale, opacity, quats=None):
    """Binary INRIA 3DGS PLY (SH degree 0) with log scales and logit opacities."""
    n = len(mu)
    quats = np.tile([1.0, 0, 0, 0], (n, 1)) if quats is None else quats
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=[(name, "<f4") for name in names])
    for k, name in enumerate("xyz"):
        data[name] = mu[:, k]
    data["opacity"] = np.log(opacity / (1 - opacity))
    for k in range(3):
        data[f"scale_{k}"] = np.log(scale[:, k])
    for k in range(4):
        data[f"rot_{k}"] = quats[:, k]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" \
        + "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def write_scene(folder, room, photos_c2w, blob=None, checkpoint=False):
    """transforms.json and scene.ply (+ object_mask.npy for the blob) in ``folder``; with checkpoint,
    also the original's ckpt.pt and flashsplat/labels.pt of the same Gaussians."""
    folder = Path(folder)
    json.dump({"camera_model": "OPENCV", "w": 64, "h": 48, "fl_x": 48, "fl_y": 48, "cx": 32, "cy": 24,
               "frames": [{"file_path": f"images/{k:04d}.jpg", "transform_matrix": m.tolist()} for k, m in enumerate(photos_c2w)]},
              open(folder / "transforms.json", "w"))
    mu, scale, n_scene = room.gaussians(blob=blob)
    n = len(mu)
    write_ply(folder / "scene.ply", mu, scale, np.full(n, 0.95))
    is_obj = np.arange(n) >= n_scene
    if blob is not None:
        np.save(folder / "object_mask.npy", is_obj)
    if checkpoint:
        from types import SimpleNamespace

        torch.save({"positions": torch.tensor(mu).float(), "rotation": torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1))).float(),
                    "scale": torch.tensor(np.log(scale)).float(), "density": torch.full((n, 1), float(np.log(0.95 / 0.05))),
                    "config": SimpleNamespace(model=SimpleNamespace(density_activation="sigmoid", scale_activation="exp"))},
                   folder / "ckpt.pt")
        if blob is not None:
            (folder / "flashsplat").mkdir(exist_ok=True)
            torch.save(torch.from_numpy(np.stack([~is_obj, is_obj])), folder / "flashsplat" / "labels.pt")
    return folder


def ring(n, radius_x, radius_z, target, height=1.5, arc=(0, 2 * np.pi)):
    angles = np.linspace(arc[0], arc[1], n, endpoint=arc[1] - arc[0] < 2 * np.pi - 1e-9)
    return np.array([look_at([radius_x * np.cos(a), height, radius_z * np.sin(a)], target) for a in angles])


def frames_of(path):
    return np.array([f["transform_matrix"] for f in json.loads(Path(path).read_text())["frames"]])


def info_of(out):
    return json.loads(Path(out).with_name(Path(out).stem + "_info.json").read_text())


def head_travel(transforms, side=2.0, up=0.5, periods=2.0, sigma=2.0, jump=4.0):
    """The travel algorithm as committed at ArtiFixer HEAD before the modes (scripts/bakerh/make_trajectory.py)."""
    frames = sorted(transforms["frames"], key=lambda f: f["file_path"])
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    n = len(c2w)
    pos = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    spacing = float(np.median(steps))
    segments = np.split(np.arange(n), np.where(steps > jump * spacing)[0] + 1)
    path = []
    for seg in segments:
        rot = Rotation.from_matrix(c2w[seg, :3, :3])
        smooth = []
        for i in seg:
            w = np.exp(-0.5 * ((seg - i) / sigma) ** 2)
            w /= w.sum()
            m = np.eye(4)
            m[:3, :3] = rot.mean(weights=w).as_matrix()
            m[:3, 3] = (w[:, None] * pos[seg]).sum(0)
            smooth.append(m)
        smooth = np.stack(smooth)
        travel = np.gradient(smooth[:, :3, 3], axis=0) if len(seg) > 1 else np.zeros((1, 3))
        for m, i, t in zip(smooth, seg, travel):
            cam_up = m[:3, 1]
            across = np.cross(t, cam_up)
            across = across / np.linalg.norm(across) if np.linalg.norm(across) > 1e-9 * spacing else m[:3, 0]
            m[:3, 3] += side * spacing * np.sin(2 * np.pi * periods * i / (n - 1)) * across + up * spacing * cam_up
            path.append(m)
    keys = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")
    out = {k: transforms[k] for k in keys if k in transforms}
    out["frames"] = [{"transform_matrix": m.tolist()} for m in np.stack(path)]
    return json.dumps(out, indent=2) + "\n"


# ---------------------------------------------------------------------------- travel mode

def test_travel_bit_identical_to_head_without_ply(tmp_path):
    photos = ring(40, 1.6, 1.2, [0, 0.7, 0])
    write_scene(tmp_path, Room(), photos)
    transforms = json.load(open(tmp_path / "transforms.json"))
    np.random.default_rng(1).shuffle(transforms["frames"])  # file order, not list order
    json.dump(transforms, open(tmp_path / "transforms.json", "w"))
    expected = head_travel(transforms)
    for extra in ([], ["--mode", "travel"]):
        out = tmp_path / f"traj{len(extra)}.json"
        assert mt.main(["--transforms", f"{tmp_path}/transforms.json", "--output", str(out)] + extra) == 0
        assert out.read_text() == expected
        info = info_of(out)
        assert info["mode"] == "travel" and info["collision_checks"] is False


def test_travel_max_frames_keeps_every_kth_frame(tmp_path):
    photos = ring(40, 1.6, 1.2, [0, 0.7, 0])
    write_scene(tmp_path, Room(), photos)
    full, capped = tmp_path / "full.json", tmp_path / "capped.json"
    assert mt.main(["--transforms", f"{tmp_path}/transforms.json", "--output", str(full)]) == 0
    assert mt.main(["--transforms", f"{tmp_path}/transforms.json", "--output", str(capped), "--max_frames", "15"]) == 0
    all_frames = json.load(open(full))["frames"]
    assert json.load(open(capped))["frames"] == all_frames[::3]
    assert info_of(capped)["frame_stride"] == 3


def test_travel_a_gap_in_the_photos_makes_two_pieces(tmp_path):
    photos = np.concatenate([ring(20, 1.6, 1.2, [0, 0.7, 0], arc=(0, 0.5 * np.pi)),
                             ring(20, 1.6, 1.2, [0, 0.7, 0], arc=(np.pi, 1.5 * np.pi))])  # a jump after photo 19
    write_scene(tmp_path, Room(), photos)
    for extra, pieces in (([], [[0, 19], [20, 39]]), (["--max_frames", "15"], [[0, 6], [7, 13]])):
        out = tmp_path / "traj.json"
        assert mt.main(["--transforms", f"{tmp_path}/transforms.json", "--output", str(out)] + extra) == 0
        info = info_of(out)
        assert info["pieces"] == pieces
        assert len(info["jumps_spacings"]) == 1
        assert info["jumps_spacings"][0] > 4


def test_travel_phase_offset_shifts_the_weave(tmp_path):
    photos = ring(40, 1.6, 1.2, [0, 0.7, 0])
    write_scene(tmp_path, Room(), photos)
    paths = {}
    for offset in ("0", "0.5", "1"):
        out = tmp_path / f"traj{offset}.json"
        assert mt.main(["--transforms", f"{tmp_path}/transforms.json", "--output", str(out), "--phase_offset", offset]) == 0
        paths[offset] = frames_of(out)
    assert np.abs(paths["0.5"] - paths["0"]).max() > 0.01
    np.testing.assert_allclose(paths["1"], paths["0"], atol=1e-9)  # a whole period later: the same path


def test_travel_with_ply_keeps_clear_of_the_walls(tmp_path):
    photos = ring(40, 2.75, 1.6, [0, 0.7, 0])  # 0.25 from the wall x = 3: the weave must pull back there
    room = Room(x=(-3, 3), z=(-2, 2))
    write_scene(tmp_path, room, photos)
    out = tmp_path / "traj.json"
    argv = ["--transforms", f"{tmp_path}/transforms.json", "--output", str(out), "--ply", f"{tmp_path}/scene.ply",
            "--clearance", "0.2"]
    assert mt.main(argv) == 0
    info = info_of(out)
    assert info["collision_checks"] is True and len(info["offset_scale"]) == len(photos)
    assert min(info["offset_scale"]) < 1  # pulled back somewhere
    path = frames_of(out)[:, :3, 3]
    wall = np.minimum.reduce([path[:, 0] + 3, 3 - path[:, 0], path[:, 2] + 2, 2 - path[:, 2], path[:, 1], 3 - path[:, 1]])
    base = photos[:, :3, 3]
    wall0 = np.minimum.reduce([base[:, 0] + 3, 3 - base[:, 0], base[:, 2] + 2, 2 - base[:, 2]])
    assert wall.min() >= 0.9 * min(0.2 / 8, wall0.min())
    assert out.with_suffix(".png").exists() or importlib.util.find_spec("matplotlib") is None


# ---------------------------------------------------------------------------- photos from COLMAP

def write_colmap(folder, photos_c2w, w=64, h=48, f=48.0, folders=(("images", 1),)):
    """A COLMAP scene of the photos: one PINHOLE camera at w x h, PNGs in every (folder, downscale)."""
    from artifixer_gs.colmap import Camera, Image as ColmapImage
    from artifixer_gs.prepare import write_colmap_cameras, write_colmap_images

    folder = Path(folder)
    (folder / "sparse" / "0").mkdir(parents=True)
    write_colmap_cameras(folder / "sparse/0/cameras.bin", [Camera(1, "PINHOLE", w, h, np.array([f, f, w / 2, h / 2]))])
    images = []
    order = np.random.default_rng(3).permutation(len(photos_c2w))  # ids not in name order
    for k, c2w in enumerate(photos_c2w):
        w2c = mt.FLIP @ np.linalg.inv(c2w)  # OpenCV world-to-camera
        x, y, z, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        images.append(ColmapImage(int(order[k]) + 1, np.array([qw, x, y, z]), w2c[:3, 3], 1, f"{k:04d}.png",
                                  np.zeros((0, 2)), np.zeros(0, dtype=np.int64)))
    write_colmap_images(folder / "sparse/0/images.bin", images)
    (folder / "sparse/0/points3D.bin").write_bytes(struct.pack("<Q", 0))
    for name, down in folders:
        (folder / name).mkdir()
        for k in range(len(photos_c2w)):
            Image.new("RGB", (w // down, h // down)).save(folder / name / f"{k:04d}.png")
    return folder


def test_colmap_photo_transforms_match_prepare(tmp_path):
    from artifixer_gs import prepare

    photos = ring(12, 1.6, 1.2, [0, 0.7, 0])
    colmap = write_colmap(tmp_path / "colmap", photos, folders=(("images", 1), ("images_2", 2)))
    ply = tmp_path / "dummy.ply"
    write_ply(ply, np.zeros((1, 3)), np.ones((1, 3)), np.full(1, 0.5))
    has_images = "--images" in prepare.build_parser().format_help()
    for images in ("images", "images_2") if has_images else ("images",):
        root = tmp_path / f"prepared_{images}"
        argv = ["--ply", str(ply), "--colmap_dir", str(colmap), "--output_root", str(root), "--phases", "prepare"]
        prepare.prepare_scene(prepare.build_parser().parse_args(argv + (["--images", images] if has_images else [])))
        expected = json.loads((root / "input" / root.name / "nerfstudio/transforms.json").read_text())
        got = mt.colmap_photo_transforms(colmap, images)
        assert {k: v for k, v in got.items() if k != "frames"} == {k: v for k, v in expected.items() if k != "frames"}
        assert [f["file_path"] for f in got["frames"]] == [f["file_path"] for f in expected["frames"]]
        np.testing.assert_allclose(frames_of_dict(got), frames_of_dict(expected), atol=1e-12)
    for factor in (2, 3):  # prepare --image_downscale_factor
        root = tmp_path / f"prepared_down{factor}"
        argv = ["--ply", str(ply), "--colmap_dir", str(colmap), "--output_root", str(root), "--phases", "prepare",
                "--image_downscale_factor", str(factor)]
        prepare.prepare_scene(prepare.build_parser().parse_args(argv))
        expected = json.loads((root / "input" / root.name / "nerfstudio/transforms.json").read_text())
        got = mt.colmap_photo_transforms(colmap, image_downscale_factor=factor)
        assert {k: v for k, v in got.items() if k != "frames"} == {k: v for k, v in expected.items() if k != "frames"}
        np.testing.assert_allclose(frames_of_dict(got), frames_of_dict(expected), atol=1e-12)
    assert (got["w"], got["h"]) == (21, 16)
    got = mt.colmap_photo_transforms(colmap, "images_2")
    assert (got["w"], got["h"], got["fl_x"], got["cx"]) == (32, 24, 24.0, 16.0)  # scaled to the image files
    np.testing.assert_allclose(mt.load_photos(got).c2w, photos, atol=1e-9)  # round trip of the poses


def frames_of_dict(transforms):
    return np.array([f["transform_matrix"] for f in transforms["frames"]])


def test_travel_from_colmap_equals_travel_from_transforms(tmp_path):
    photos = ring(30, 1.6, 1.2, [0, 0.7, 0])
    colmap = write_colmap(tmp_path / "colmap", photos)
    (tmp_path / "t.json").write_text(json.dumps(mt.colmap_photo_transforms(colmap)))
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert mt.main(["--colmap_dir", str(colmap), "--output", str(a)]) == 0
    assert mt.main(["--transforms", str(tmp_path / "t.json"), "--output", str(b)]) == 0
    assert a.read_text() == b.read_text()
    out = json.loads(a.read_text())
    assert out["camera_model"] == "OPENCV" and (out["w"], out["h"]) == (64, 48)
    assert all(set(f) == {"transform_matrix"} for f in out["frames"])  # the prepare --trajectory_path contract
    with pytest.raises(SystemExit):  # a photo source is needed
        mt.main(["--output", str(a)])
    c = tmp_path / "c.json"
    assert mt.main(["--transforms", str(tmp_path / "t.json"), "--colmap_dir", str(colmap), "--output", str(c)]) == 0
    assert c.read_text() == b.read_text()  # --colmap_dir next to --transforms only checks the world frame
    d = tmp_path / "d.json"
    assert mt.main(["--colmap_dir", str(colmap), "--image_downscale_factor", "2", "--output", str(d)]) == 0
    assert (json.loads(d.read_text())["w"], json.loads(d.read_text())["h"]) == (32, 24)


def test_transforms_in_another_world_frame_are_refused(tmp_path):
    photos = ring(30, 1.6, 1.2, [0, 0.7, 0])
    colmap = write_colmap(tmp_path / "colmap", photos)
    transforms = mt.colmap_photo_transforms(colmap)
    out = tmp_path / "t.json"
    moved = json.loads(json.dumps(transforms))
    for f in moved["frames"]:
        f["transform_matrix"][0][3] += 0.01  # a shift of ~0.03 photo spacings: another frame
    (tmp_path / "moved.json").write_text(json.dumps(moved))
    with pytest.raises(SystemExit, match="another world frame"):
        mt.main(["--transforms", str(tmp_path / "moved.json"), "--colmap_dir", str(colmap), "--output", str(out)])
    other = json.loads(json.dumps(transforms))
    other["frames"][0]["file_path"] = "images/elsewhere.png"
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="not in"):
        mt.main(["--transforms", str(tmp_path / "other.json"), "--colmap_dir", str(colmap), "--output", str(out)])
    assert not out.exists()


def test_trajectory_is_accepted_by_prepare(tmp_path):
    from data_processing.camera_trajectories import assert_target_only_trajectory, read_camera_trajectory

    photos = ring(30, 1.6, 1.2, [0, 0.7, 0])
    write_scene(tmp_path, Room(), photos)
    out = tmp_path / "traj.json"
    assert mt.main(["--transforms", f"{tmp_path}/transforms.json", "--output", str(out)]) == 0
    trajectory = read_camera_trajectory(out)
    assert_target_only_trajectory(trajectory, str(out))
    assert len(trajectory["frames"]) == len(photos)


# ---------------------------------------------------------------------------- loaders and the gsplat camera

def test_load_gaussians_from_ply(tmp_path):
    rng = np.random.default_rng(0)
    mu, scale, opacity = rng.normal(size=(50, 3)), rng.uniform(0.01, 0.2, (50, 3)), rng.uniform(0.05, 0.95, 50)
    quats = rng.normal(size=(50, 4))
    quats[0] = 0  # degenerate: identity
    write_ply(tmp_path / "g.ply", mu, scale, opacity, quats * 3)
    g = mt.load_gaussians(tmp_path / "g.ply")
    np.testing.assert_allclose(g.mu, mu, atol=1e-6)
    np.testing.assert_allclose(g.scale, scale, rtol=1e-5)
    np.testing.assert_allclose(g.opacity, opacity, rtol=1e-5)
    q = quats[1:] / np.linalg.norm(quats[1:], axis=1, keepdims=True)
    np.testing.assert_allclose(g.rot[1:], Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(), atol=1e-5)
    np.testing.assert_allclose(g.rot[0], np.eye(3))
    assert g.mu.dtype == np.float64


def test_object_mask_must_match_the_ply(tmp_path):
    photos = ring(30, 2.75, 1.8, [0, 0.5, 0])
    write_scene(tmp_path, Room(x=(-3, 3), z=(-3, 3)), photos, blob=BLOB)
    mask = np.load(tmp_path / "object_mask.npy")
    np.save(tmp_path / "short.npy", mask[:-1])
    argv = ["--transforms", f"{tmp_path}/transforms.json", "--output", str(tmp_path / "t.json"),
            "--ply", f"{tmp_path}/scene.ply", "--object_mask", f"{tmp_path}/short.npy"]
    with pytest.raises(AssertionError, match="another scene"):
        mt.main(argv)
    with pytest.raises(SystemExit):
        mt.main(argv[:4] + ["--mode", "object", "--ply", f"{tmp_path}/scene.ply"])  # no mask
    with pytest.raises(SystemExit):
        mt.main(argv[:4] + ["--mode", "vidsplat"])  # no PLY


def test_check_intrinsics_scale_to_check_width():
    intr, full = mt.check_intrinsics({"w": 1000, "h": 750, "fl_x": 800.0, "fl_y": 790.0, "cx": 501.5, "cy": 374.0}, 256)
    fx, fy, cx, cy, w, h = intr
    assert (w, h, full) == (256, 192, (1000, 750))
    np.testing.assert_allclose([fx, fy, cx, cy], [800 * 0.256, 790 * 0.256, 501.5 * 0.256, 374 * 0.256], rtol=1e-6)


# ---------------------------------------------------------------------------- orbit pose

def test_orbit_anchor_stays_centred_and_level():
    anchor = np.array([0.5, 0.3, -0.2])
    photo = look_at([2.0, 1.4, 1.0], anchor)
    for yaw, pitch in ((0.4, 0.0), (-0.3, 0.25), (0.2, -0.5)):
        m = mt.orbit_pose(photo, anchor, yaw, pitch, UP, np.radians(60))
        local = m[:3, :3].T @ (anchor - m[:3, 3])
        np.testing.assert_allclose(local[:2], 0, atol=1e-9)  # on the optical axis
        assert local[2] < 0  # in front (OpenGL looks along -z)
        assert np.linalg.norm(m[:3, 3] - anchor) == pytest.approx(np.linalg.norm(photo[:3, 3] - anchor), abs=1e-9)
        np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.linalg.det(m[:3, :3]) == pytest.approx(1.0, abs=1e-9)
        assert m[:3, 0] @ UP == pytest.approx(0.0, abs=1e-9)  # no roll: the x axis stays horizontal


def test_orbit_elevation_is_capped():
    anchor = np.zeros(3)
    m = mt.orbit_pose(look_at([2.0, 0.0, 0.0], anchor), anchor, 0.0, np.radians(80), UP, np.radians(60))
    elevation = np.degrees(np.arcsin(m[1, 3] / np.linalg.norm(m[:3, 3])))
    assert elevation == pytest.approx(60.0, abs=1e-6)


# ---------------------------------------------------------------------------- view checker

def checker_of(room, photos, tol=0.05):
    return mt.ViewChecker(room.render, photos, INTR, depth_tol=tol)


def test_photo_pose_is_seen_and_unphotographed_ceiling_is_not():
    room = Room()
    photos = ring(24, 1.6, 1.2, [0, 0.7, 0])
    checker = checker_of(room, photos)
    stats = checker.measure(photos[:3], 0.5)
    assert (stats["unseen"] < 0.01).all(), stats
    ceiling = look_at([0, 1.5, 0], [0.0, 3.0, 0.01])
    assert checker.measure(ceiling[None], 0.5)["unseen"][0] > 0.9


def test_occlusion_behind_a_pillar_is_unseen():
    room = Room(pillar=(0.0, 0.0, 0.35))
    photo = look_at([-2.5, 1.5, 0.0], [3.0, 1.5, 0.0])
    view = look_at([-2.5, 1.5, 1.2], [3.0, 1.5, 0.0])  # sees the wall behind the pillar
    with_depth = checker_of(room, photo[None]).measure(view[None], 0.5)["unseen"][0]
    without_depth = checker_of(room, photo[None], tol=1e9).measure(view[None], 0.5)["unseen"][0]
    assert with_depth > without_depth + 0.02


def test_near_wall_and_doorway():
    room = Room(doorway=((-0.5, 0.5), (0.0, 2.0)))
    photo = look_at([-1.0, 1.0, 0.0], [3.0, 1.0, 0.0])
    checker = checker_of(room, photo[None])
    near = look_at([2.8, 1.5, 1.2], [3.5, 1.5, 1.2])  # 0.2 from the wall, facing it
    assert checker.measure(near[None], 0.5, unseen=False)["near_frac"][0] > 0.95
    stats = checker.measure(photo[None], 0.5)
    assert stats["empty"][0] > 0.05  # the doorway: no geometry
    assert stats["unseen"][0] < 0.01  # ...which never counts as unseen


def test_anchor_on_central_ray():
    room = Room()
    photo = look_at([-1.0, 1.5, 0.0], [3.0, 0.5, 0.0])
    anchor = checker_of(room, photo[None]).anchor(0, photo)
    local = photo[:3, :3].T @ (anchor - photo[:3, 3])
    np.testing.assert_allclose(local[:2], 0, atol=1e-6)
    assert abs(anchor[0] - 3.0) < 0.1  # on the far wall, near where the ray meets it


# ---------------------------------------------------------------------------- vidsplat mode

BLOB = (np.array([0.0, 0.5, 0.0]), np.array([0.5, 0.5, 0.4]))  # the object: (centre, half-size) of a box of Gaussians
VIDSPLAT_ROOM = Room(x=(-4, 4), z=(-3, 3), height=3.0)


def vidsplat_photos():
    return ring(48, 1.5, 1.2, [3.5, 1.0, 0.0], arc=(0.5 * np.pi, 1.5 * np.pi))  # half ring facing +x


def vidsplat_argv(tmp, out, extra=()):
    return ["--transforms", f"{tmp}/transforms.json", "--output", str(out), "--mode", "vidsplat",
            "--ply", f"{tmp}/scene.ply", "--clearance", "0.3", "--metric_scale", "1.0"] + list(extra)


def run_vidsplat(tmp, extra=(), name="traj.json"):
    photos = vidsplat_photos()
    if not (Path(tmp) / "scene.ply").exists():
        write_scene(tmp, VIDSPLAT_ROOM, photos)
    out = Path(tmp) / name
    code = mt.main(vidsplat_argv(tmp, out, extra), renderer=(VIDSPLAT_ROOM.render, INTR))
    return code, out, info_of(out), photos


def test_vidsplat_valid_orbits_layout_and_bounds(tmp_path):
    code, out, info, photos = run_vidsplat(tmp_path, ["--budget", "97"])
    assert code == 0, info.get("failed")
    frames = frames_of(out)
    L, clips = info["clip_len"], info["clips"]
    assert len(clips) >= info["min_clips"]
    assert len(frames) == 1 + L * len(clips)
    assert [c["seed_index"] for c in clips] == sorted(c["seed_index"] for c in clips)
    np.testing.assert_allclose(frames[0], photos[clips[0]["seed_index"]])
    assert info["pieces"] == [[0, L]] + [[1 + L * c, L * (c + 1)] for c in range(1, len(clips))]
    assert len(info["jumps_spacings"]) == len(clips) - 1
    for c_i, clip in enumerate(clips):
        start = 1 + L * c_i
        assert start % 4 == 1  # never shares a 4-frame VAE group with the previous clip
        seg = frames[start:start + L]
        np.testing.assert_allclose(np.linalg.norm(seg[:, :3, 3] - clip["anchor"], axis=1), clip["anchor_distance"], rtol=1e-6)
        k = clip["keyframes"]
        assert max(k["unseen"]) <= 0.4 and max(k["near_frac"]) <= 0.02, k
        inside = (np.abs(seg[:, 0, 3]) < 4 - 0.3) & (np.abs(seg[:, 2, 3]) < 3 - 0.3) \
            & (seg[:, 1, 3] > 0.3) & (seg[:, 1, 3] < 3 - 0.3)
        assert inside.all(), f"clip {c_i} leaves the room or touches a wall"
    code, again, _, _ = run_vidsplat(tmp_path, ["--budget", "97"], name="again.json")
    assert code == 0
    assert again.read_text() == out.read_text()  # deterministic


def test_vidsplat_seen_views_count_as_observed(tmp_path):
    code, out, info, photos = run_vidsplat(tmp_path, ["--budget", "97"])
    assert code == 0
    first = {(c["seed_index"], c["direction"], c["extent_deg"]) for c in info["clips"]}
    code, _, info2, _ = run_vidsplat(tmp_path, ["--budget", "97", "--seen_transforms", str(out), "--min_clips", "0"],
                                     name="round2.json")
    assert code == 0
    second = {(c["seed_index"], c["direction"], c["extent_deg"]) for c in info2["clips"]}
    assert not first & second, "round 2 repeated an orbit whose views are already seen"
    assert info2["rejected"]["nothing_new"] > info["rejected"]["nothing_new"]


def test_vidsplat_seeds_spread_over_the_capture_and_between_rounds(tmp_path):
    rounds = []
    for k in range(3):  # a small scene's loop: one orbit per round, phase (k-1)/3 of a seed spacing
        code, out, info, photos = run_vidsplat(tmp_path, ["--budget", "17", "--phase_offset", str(k / 3)])
        assert code == 0, info.get("failed")
        assert info["clips_wanted"] == 1
        seeds = info["seed_indices"]
        assert len(seeds) == 12
        gaps = np.diff(seeds + [seeds[0] + len(photos)])
        assert gaps.max() - gaps.min() <= 1, seeds  # evenly over the whole capture
        assert info["clips"][0]["seed_index"] in info["valid_seeds"]
        rounds.append(set(seeds))
    for a in range(3):
        for b in range(a + 1, 3):
            assert not rounds[a] & rounds[b], f"rounds {a + 1} and {b + 1} try the same seed photos"


def test_vidsplat_one_orbit_per_stretch_of_the_capture(tmp_path):
    code, out, info, photos = run_vidsplat(tmp_path, ["--budget", "49", "--seeds", "24"])
    assert code == 0, info.get("failed")
    n, wanted = len(photos), info["clips_wanted"]
    assert (wanted, info["seeds_tried"]) == (3, 24)
    stretches = {min(s * wanted // n, wanted - 1) for s in info["valid_seeds"]}
    got = [min(c["seed_index"] * wanted // n, wanted - 1) for c in info["clips"]]
    assert sorted(set(got)) == sorted(stretches)[:len(info["clips"])], got


def test_vidsplat_max_frames_sets_the_budget(tmp_path):
    code, out, info, _ = run_vidsplat(tmp_path, ["--max_frames", "33"])
    assert code == 0, info.get("failed")
    assert info["clips_wanted"] == 2 and len(frames_of(out)) <= 33


def test_vidsplat_fails_loudly(tmp_path):
    code, out, info, _ = run_vidsplat(tmp_path, ["--s_high", "0.001"])
    assert code == 2
    assert "valid VidSplat orbits" in info["failed"]
    assert info["rejected"]["too_unseen"] > 0
    assert not out.exists()
    assert out.with_suffix(".png").exists() or importlib.util.find_spec("matplotlib") is None


def test_vidsplat_clip_len_must_be_multiple_of_4(tmp_path):
    with pytest.raises(SystemExit):
        run_vidsplat(tmp_path, ["--clip_len", "10"])


# ---------------------------------------------------------------------------- object mode

def run_object(tmp, room, photos, extra=(), renderer=None):
    write_scene(tmp, room, photos, blob=BLOB)
    out = Path(tmp) / "traj.json"
    argv = ["--transforms", f"{tmp}/transforms.json", "--output", str(out), "--ply", f"{tmp}/scene.ply",
            "--object_mask", f"{tmp}/object_mask.npy", "--clearance", "0.25"] + list(extra)
    code = mt.main(argv, renderer=renderer)
    return code, out, info_of(out)


def test_object_roomy_room_passes_with_a_photo_near_a_wall(tmp_path):
    photos = ring(60, 2.75, 1.8, [0, 0.5, 0])  # the photo at angle 0 is 0.25 from the wall x = 3
    code, out, info = run_object(tmp_path, Room(x=(-3, 3), z=(-3, 3)), photos)
    assert code == 0, info.get("failed")
    assert info["mode"] == "object"  # the default with --object_mask
    assert info["collapsed_share"] <= 0.5
    assert len(frames_of(out)) == len(photos)
    assert info["object"]["gaussians"] == 6000
    np.testing.assert_allclose(info["object"]["centre"], BLOB[0], atol=0.05)


def test_object_photos_close_to_walls_move_no_closer(tmp_path):
    photos = ring(60, 2.85, 1.85, [0, 0.5, 0])  # 9 photos within the clearance of the walls x = +-3
    room = Room(x=(-3, 3), z=(-2.3, 2.3))
    code, out, info = run_object(tmp_path, room, photos)
    assert code == 0, info.get("failed")
    assert info["base_cameras_touching"] > 0
    assert info["collapsed_share"] <= 0.25

    def wall_distance(p):
        return np.minimum.reduce([p[:, 0] + 3, 3 - p[:, 0], p[:, 2] + 2.3, 2.3 - p[:, 2]])

    path = frames_of(out)
    assert wall_distance(path[:, :3, 3]).min() >= 0.9 * wall_distance(photos[:, :3, 3]).min()


def test_object_tight_walls_fail_loudly_unless_fallback(tmp_path):
    photos = ring(60, 1.45, 1.3, [0, 0.5, 0])
    room = Room(x=(-1.6, 1.6), z=(-1.45, 1.45))  # the orbit has no room: walls just outside the photos
    (tmp_path / "a").mkdir()
    code, out, info = run_object(tmp_path / "a", room, photos)
    assert code == 2
    assert "collapsed" in info["failed"]
    assert not out.exists()
    (tmp_path / "b").mkdir()
    code, out, info = run_object(tmp_path / "b", room, photos, ["--allow_fallback"])
    assert code == 0
    assert out.exists()


def test_object_render_check_with_injected_renderer(tmp_path):
    photos = ring(60, 2.75, 1.8, [0, 0.5, 0])
    room = Room(x=(-3, 3), z=(-3, 3))
    code, out, info = run_object(tmp_path, room, photos, ["--render_check", "--metric_scale", "1.0"],
                                 renderer=(room.render, INTR))
    assert code == 0, info.get("failed")
    assert info["render_check"] is True
    # Every moved camera passed the rendered near-plane test at its base camera's d0.
    checker = mt.ViewChecker(room.render, photos, INTR)
    base, _, _ = mt.smooth_base(mt.load_photos(tmp_path / "transforms.json"), 2.0, 4.0)
    d0_base = np.minimum(0.5, 0.8 * checker.near_depth(base))
    moved = np.asarray(info["offset_scale"]) > 0
    assert moved.any()
    near = checker.measure(frames_of(out)[moved], d0_base[moved], unseen=False)["near_frac"]
    assert (near <= 0.02).all(), near


# ---------------------------------------------------------------------------- equivalence with the original

DRIVER = r"""
import importlib.util, json, sys, types
spec = importlib.util.spec_from_file_location("test_trajectory", sys.argv[1])
tt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tt)
# The labels-only branch of load_object never calls removal_keep_masks; keep its 3DGUT imports out.
stub = types.ModuleType("aurafusion.render_views"); stub.removal_keep_masks = None
sys.modules["aurafusion"] = types.ModuleType("aurafusion"); sys.modules["aurafusion.render_views"] = stub
spec = importlib.util.spec_from_file_location("make_trajectory", sys.argv[2])
mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
room = json.loads(sys.argv[4])
renderer = (tt.Room(**room).render, tt.INTR) if room else None
sys.exit(mt.main(json.loads(sys.argv[3]), renderer=renderer))
"""

needs_af = pytest.mark.skipif(not AF_SCRIPT.is_file(), reason=f"no original make_trajectory.py at {AF_SCRIPT}")


def run_original(tmp, argv, room=None):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])}
    room_kwargs = json.dumps({"x": room.x, "z": room.z, "height": room.height} if room else {})
    result = subprocess.run([sys.executable, "-c", DRIVER, __file__, str(AF_SCRIPT), json.dumps(argv), room_kwargs],
                            env=env, cwd=tmp, capture_output=True, text=True)
    assert result.returncode in (0, 2), result.stdout + result.stderr
    return result.returncode


def assert_same_outputs(ours, theirs, info_keys):
    np.testing.assert_allclose(frames_of(ours), frames_of(theirs), atol=1e-5)
    a, b = info_of(ours), info_of(theirs)
    for key in info_keys:
        if isinstance(a[key], float) or (isinstance(a[key], list) and a[key] and isinstance(a[key][0], float)):
            np.testing.assert_allclose(a[key], b[key], atol=1e-5, err_msg=key)
        else:
            assert a[key] == b[key], key
    ours_head = {k: v for k, v in json.loads(ours.read_text()).items() if k != "frames"}
    assert ours_head == {k: v for k, v in json.loads(theirs.read_text()).items() if k != "frames"}


@needs_af
def test_same_object_path_as_the_original(tmp_path):
    photos = ring(60, 2.85, 1.85, [0, 0.5, 0])
    room = Room(x=(-3, 3), z=(-2.3, 2.3))
    write_scene(tmp_path, room, photos, blob=BLOB, checkpoint=True)
    common = ["--transforms", f"{tmp_path}/transforms.json", "--clearance_m", "0.25", "--metric_scale", "1.0",
              "--orbit", "0.6", "--radial", "0.3", "--rise", "0.3"]
    ours, theirs = tmp_path / "ours.json", tmp_path / "theirs.json"
    assert mt.main(common + ["--output", str(ours), "--ply", f"{tmp_path}/scene.ply",
                             "--object_mask", f"{tmp_path}/object_mask.npy"]) == 0
    assert run_original(tmp_path, common + ["--output", str(theirs), "--checkpoint", f"{tmp_path}/ckpt.pt",
                                            "--flashsplat_dir", f"{tmp_path}/flashsplat"]) == 0
    assert_same_outputs(ours, theirs, ("mode", "clearance", "offset_scale", "pieces", "base_cameras_touching",
                                       "base_cameras_inside", "collapsed_share"))
    a, b = info_of(ours)["object"], info_of(theirs)["object"]
    np.testing.assert_allclose(a["centre"], b["centre"], atol=1e-6)
    np.testing.assert_allclose(a["half_extents"], b["half_extents"], atol=1e-6)
    assert a["gaussians"] == b["gaussians"]


@needs_af
def test_same_travel_path_with_walls_as_the_original(tmp_path):
    photos = ring(40, 2.75, 1.6, [0, 0.7, 0])
    write_scene(tmp_path, Room(x=(-3, 3), z=(-2, 2)), photos, checkpoint=True)
    common = ["--transforms", f"{tmp_path}/transforms.json", "--clearance", "0.2", "--max_frames", "20", "--mode", "travel"]
    ours, theirs = tmp_path / "ours.json", tmp_path / "theirs.json"
    assert mt.main(common + ["--output", str(ours), "--ply", f"{tmp_path}/scene.ply"]) == 0
    assert run_original(tmp_path, common + ["--output", str(theirs), "--checkpoint", f"{tmp_path}/ckpt.pt"]) == 0
    assert_same_outputs(ours, theirs, ("mode", "clearance", "offset_scale", "pieces", "frame_stride", "jumps_spacings"))


@needs_af
def test_same_vidsplat_path_as_the_original(tmp_path):
    photos = vidsplat_photos()
    write_scene(tmp_path, VIDSPLAT_ROOM, photos, checkpoint=True)
    common = ["--transforms", f"{tmp_path}/transforms.json", "--mode", "vidsplat", "--clearance", "0.3",
              "--metric_scale", "1.0", "--budget", "49", "--seeds", "12"]
    ours, theirs = tmp_path / "ours.json", tmp_path / "theirs.json"
    assert mt.main(common + ["--output", str(ours), "--ply", f"{tmp_path}/scene.ply"],
                   renderer=(VIDSPLAT_ROOM.render, INTR)) == 0
    assert run_original(tmp_path, common + ["--output", str(theirs), "--checkpoint", f"{tmp_path}/ckpt.pt"],
                        room=VIDSPLAT_ROOM) == 0
    assert_same_outputs(ours, theirs, ("clips_wanted", "min_clips", "seed_indices", "valid_seeds", "rejected", "pieces"))
    assert [(c["seed_index"], c["direction"], c["extent_deg"]) for c in info_of(ours)["clips"]] == \
        [(c["seed_index"], c["direction"], c["extent_deg"]) for c in info_of(theirs)["clips"]]


@needs_af
def test_same_render_checked_object_path_as_the_original(tmp_path):
    photos = ring(60, 2.75, 1.8, [0, 0.5, 0])
    room = Room(x=(-3, 3), z=(-3, 3))
    write_scene(tmp_path, room, photos, blob=BLOB, checkpoint=True)
    common = ["--transforms", f"{tmp_path}/transforms.json", "--clearance", "0.25", "--metric_scale", "1.0",
              "--render_check", "--d0_m", "0.5"]
    ours, theirs = tmp_path / "ours.json", tmp_path / "theirs.json"
    assert mt.main(common + ["--output", str(ours), "--ply", f"{tmp_path}/scene.ply",
                             "--object_mask", f"{tmp_path}/object_mask.npy"], renderer=(room.render, INTR)) == 0
    assert run_original(tmp_path, common + ["--output", str(theirs), "--checkpoint", f"{tmp_path}/ckpt.pt",
                                            "--flashsplat_dir", f"{tmp_path}/flashsplat"], room=room) == 0
    assert_same_outputs(ours, theirs, ("mode", "clearance", "offset_scale", "pieces", "render_check", "collapsed_share"))
