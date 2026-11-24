# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the object-removal flow: removal rule, hole masks, frame windows, propagation logic and the
removal handoff, end to end on a synthetic COLMAP scene with an injected renderer (no gsplat, no GPU).

  pytest tests/test_removal.py --basetemp <scratch dir>
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifixer_gs import distill, hole_mesh, propagate, removal, render  # noqa: E402
from artifixer_gs.colmap import Camera, Image as ColmapImage, read_colmap_extrinsics_binary, read_colmap_intrinsics_binary  # noqa: E402
from artifixer_gs.prepare import build_parser as prepare_parser, prepare_scene, write_colmap_cameras, write_colmap_images  # noqa: E402
from artifixer_gs.render import read_ply, trajectory_cameras  # noqa: E402

W, H, F = 64, 48, 40.0


# ---------------------------------------------------------------------------- frames, seeds


def test_select_frames_windows_and_auto_seeds():
    indices = removal.select_frames("49-148,284-304", 400)
    assert len(indices) == 121 and indices[0] == 49 and indices[100] == 284
    starts, seeds = removal.frame_windows(indices, 7)
    assert starts == [0, 100] and seeds == "0-6,100-106"
    # A window shorter than the seed length is seeded whole.
    starts, seeds = removal.frame_windows(removal.select_frames("0-2,10-20", 30), 7)
    assert starts == [0, 3] and seeds == "0-2,3-9"
    assert removal.select_frames("", 5) == [0, 1, 2, 3, 4]
    assert removal.select_frames("3", 5) == [3]
    with pytest.raises(AssertionError):
        removal.select_frames("0-3,2-4", 10)
    with pytest.raises(AssertionError):
        removal.select_frames("8-12", 10)
    assert removal.parse_frame_ranges("0-2+5,7-8") == [0, 1, 2, 5, 7, 8]


# ---------------------------------------------------------------------------- hole masks


def test_hole_mask_dilation_is_a_cross_diamond():
    mask = np.zeros((21, 21), bool)
    mask[10, 10] = True
    hole = removal.hole_mask(mask, np.zeros((21, 21)), dilate_px=3)
    ys, xs = np.nonzero(hole)
    assert hole.sum() == 2 * 3 * 3 + 2 * 3 + 1  # L1 ball of radius 3
    assert (np.abs(ys - 10) + np.abs(xs - 10)).max() == 3


def test_hole_mask_deleted_opacity_threshold_and_hull():
    opacity = np.zeros((20, 20), np.uint8)
    opacity[2, 2], opacity[2, 3] = 13, 12  # 13/255 > 0.05 > 12/255
    hole = removal.hole_mask(np.zeros((20, 20), bool), opacity, 0.05, "mask", 0)
    assert hole[2, 2] and not hole[2, 3] and hole.sum() == 1
    # An L-shaped footprint: the hull fills the triangle between its arms.
    mask = np.zeros((20, 20), bool)
    mask[2:18, 2] = True
    mask[17, 2:18] = True
    filled = removal.hole_mask(mask, np.zeros((20, 20)), hole_shape="hull", dilate_px=0)
    assert filled[12, 6] and not mask[12, 6] and not filled[3, 15]
    assert (filled | mask).sum() == filled.sum()
    # Collinear footprint: no hull, the mask is kept.
    line = np.zeros((20, 20), bool)
    line[5, 3:15] = True
    assert np.array_equal(removal.hole_mask(line, np.zeros((20, 20)), hole_shape="hull", dilate_px=0), line)


def test_hole_mask_alpha_and_mesh_modes():
    obj = np.zeros((20, 20), bool)
    obj[8:12, 8:12] = True
    deleted = np.zeros((20, 20), np.uint8)
    deleted[9, 9] = 200
    background = np.full((20, 20), 255, np.uint8)
    background[10:12, 10:16] = 100  # see-through (100/255 < 0.5), partly outside the dilated mask
    dilated = removal.hole_mask(obj, deleted, dilate_px=2)
    alpha = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="alpha", background_opacity=background)
    assert np.array_equal(alpha, dilated & ((background < 128) | (deleted > 12)))
    assert alpha[9, 9] and alpha[10, 10] and not alpha[8, 8] and alpha[10, 13] and not alpha[10, 14] and alpha.sum() == 1 + 2 * 4
    front = np.zeros((20, 20), bool)
    front[0, 0] = True  # the mesh is not restricted to the dilated mask
    mesh = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="mesh", background_opacity=background, mesh_front=front)
    assert np.array_equal(mesh, alpha | front)
    assert np.array_equal(removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="dilate", background_opacity=background), dilated)
    with pytest.raises(AssertionError):
        removal.hole_mask(obj, deleted, hole_mode="mesh", background_opacity=background)
    with pytest.raises(AssertionError):
        removal.hole_mask(obj, deleted, hole_mode="alpha")


def test_hole_mask_low_confidence_union():
    """alpha / mesh: low-confidence pixels join the hole inside the dilated mask only; dilate is unchanged."""
    obj = np.zeros((20, 20), bool)
    obj[8:12, 8:12] = True
    deleted = np.zeros((20, 20), np.uint8)
    background = np.full((20, 20), 255, np.uint8)
    background[10, 10] = 0
    low = np.zeros((20, 20), bool)
    low[7, 7:12] = True  # inside the dilated mask
    low[0, 0] = True     # outside it: not a hole
    dilated = removal.hole_mask(obj, deleted, dilate_px=2)
    alpha = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="alpha", background_opacity=background)
    with_conf = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="alpha", background_opacity=background, low_confidence=low)
    assert np.array_equal(with_conf, alpha | (dilated & low)) and with_conf.sum() == 1 + 5 and not with_conf[0, 0]
    front = np.zeros((20, 20), bool)
    front[19, 19] = True
    mesh = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="mesh", background_opacity=background, mesh_front=front,
                             low_confidence=low)
    assert np.array_equal(mesh, with_conf | front)
    assert np.array_equal(removal.hole_mask(obj, deleted, dilate_px=2, low_confidence=low), dilated)
    with pytest.raises(AssertionError, match="confidence map"):
        removal.hole_mask(obj, deleted, hole_mode="alpha", background_opacity=background, low_confidence=low[:10])


def test_hole_mask_gavis_mode():
    """gavis: the dilated hole where the unseen map is set or the deleted Gaussians show [or low confidence];
    no map = alpha's rule."""
    obj = np.zeros((20, 20), bool)
    obj[8:12, 8:12] = True
    deleted = np.zeros((20, 20), np.uint8)
    deleted[9, 9] = 200
    background = np.full((20, 20), 255, np.uint8)
    background[10, 10] = 0          # see-through, but seen-from-here (unseen not set): not in the gavis hole
    unseen = np.zeros((20, 20), bool)
    unseen[7, 7:12] = True           # opaque, never seen from this direction: in the hole
    unseen[0, 0] = True              # outside the dilated mask: not a hole
    dilated = removal.hole_mask(obj, deleted, dilate_px=2)
    gavis = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="gavis", background_opacity=background, unseen=unseen)
    assert np.array_equal(gavis, dilated & (unseen | (deleted > 12)))
    assert gavis.sum() == 1 + 5 and gavis[9, 9] and not gavis[10, 10] and not gavis[0, 0]
    low = np.zeros((20, 20), bool)
    low[12, 8] = True
    with_conf = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="gavis", background_opacity=background,
                                  low_confidence=low, unseen=unseen)
    assert np.array_equal(with_conf, gavis | low)
    alpha = removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="alpha", background_opacity=background)
    assert np.array_equal(removal.hole_mask(obj, deleted, dilate_px=2, hole_mode="gavis", background_opacity=background), alpha)
    with pytest.raises(AssertionError, match="unseen map"):
        removal.hole_mask(obj, deleted, hole_mode="gavis", background_opacity=background, unseen=unseen[:10])


def test_load_confidence(tmp_path):
    Image.fromarray(np.array([[0, 255], [128, 255]], np.uint8)).save(tmp_path / "IMG_000.png")
    assert removal.load_confidence(tmp_path, "IMG_001", (2, 2)) is None
    c = removal.load_confidence(tmp_path, "IMG_000", (2, 2))
    assert c.dtype == np.float32 and np.allclose(c, [[0, 1], [128 / 255, 1]])
    big = removal.load_confidence(tmp_path, "IMG_000", (8, 6))  # (w, h): resized BILINEAR to the render size
    assert big.shape == (6, 8) and big.min() >= 0 and big.max() <= 1 and big[0, 0] < 0.1 and big[0, -1] > 0.9


def test_mesh_front_rule_and_clip_coords():
    covered = np.array([[True, True, True, False]])
    mesh_depth = np.array([[1.0, 2.03, 2.05, 1.0]])
    background_depth = np.array([[2.0, 2.0, 2.0, 2.0]])
    background_alpha = np.array([[1.0, 1.0, 1.0, 1.0]])
    # In front, coplanar (within 2%: the contact patch), behind; uncovered pixels never count.
    assert hole_mesh.front_mask(mesh_depth, covered, background_depth, background_alpha).tolist() == [[True, True, False, False]]
    background_alpha[0, 2] = 0.2  # see-through background: the mesh is the front
    assert hole_mesh.front_mask(mesh_depth, covered, background_depth, background_alpha).tolist() == [[True, True, True, False]]
    torch = pytest.importorskip("torch")
    verts = torch.tensor([[0.3, -0.2, 2.0], [-1.0, 0.5, 4.0], [0.0, 0.0, 1.0]])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    viewmat[:3, 3] = torch.tensor([0.1, 0.2, 0.5])
    K = torch.tensor([[F, 0, 30.0], [0, F, 25.0], [0, 0, 1]])
    clip, z = hole_mesh.clip_coords(verts, viewmat, K, W, H)
    cam = verts @ viewmat[:3, :3].T + viewmat[:3, 3]
    uv = cam[:, :2] / cam[:, 2:] * F + torch.tensor([30.0, 25.0])
    ndc = clip[:, :3] / clip[:, 3:]
    assert torch.allclose(z, cam[:, 2]) and torch.allclose(clip[:, 3], cam[:, 2])
    assert torch.allclose((ndc[:, 0] + 1) / 2 * W, uv[:, 0], atol=1e-4) and torch.allclose((ndc[:, 1] + 1) / 2 * H, uv[:, 1], atol=1e-4)
    assert (ndc[:, 2].abs() < 1).all()


def test_read_mesh(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    verts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    faces = np.array([([0, 1, 2], 3), ([0, 2, 3], 3)], dtype=[("vertex_indices", "i4", (3,)), ("label", "i4")])
    plyfile.PlyData([plyfile.PlyElement.describe(verts, "vertex"), plyfile.PlyElement.describe(faces, "face")]).write(str(tmp_path / "m.ply"))
    v, f = hole_mesh.read_mesh(tmp_path / "m.ply")
    assert v.shape == (4, 3) and f.tolist() == [[0, 1, 2], [0, 2, 3]] and f.dtype == np.int32


def test_mesh_front_cuda(tmp_path):
    """hole_mesh end to end (nvdiffrast, GPU): a quad at depth 1 over a background at depth 2 (left) / 0.5 (right)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    pytest.importorskip("nvdiffrast.torch")
    plyfile = pytest.importorskip("plyfile")
    # Identity OpenGL c2w: the camera looks down -z, world +y is image up. The quad covers u 12..52, v 4..44.
    transforms = {"camera_model": "OPENCV", "fl_x": F, "fl_y": F, "cx": W / 2, "cy": H / 2, "w": W, "h": H,
                  "frames": [{"transform_matrix": np.eye(4).tolist()}]}
    (tmp_path / "transforms.json").write_text(json.dumps(transforms))
    rdir = tmp_path / "removal"
    (rdir / "background" / "depth").mkdir(parents=True)
    (rdir / "background" / "opacity").mkdir()
    (rdir / "removal.json").write_text(json.dumps({"transforms_path": str(tmp_path / "transforms.json"), "frames": 1}))
    depth = np.full((H, W), 2.0, np.float32)
    depth[:, W // 2:] = 0.5
    alpha = np.full((H, W), 255, np.uint8)
    alpha[30:, 40:] = 0  # see-through: the mesh shows even behind
    np.save(rdir / "background" / "depth" / "00000.npy", depth)
    Image.fromarray(alpha).save(rdir / "background" / "opacity" / "00000.png")
    verts = np.array([(-0.5, -0.5, -1), (0.5, -0.5, -1), (0.5, 0.5, -1), (-0.5, 0.5, -1)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    faces = np.array([([0, 1, 2],), ([0, 2, 3],)], dtype=[("vertex_indices", "i4", (3,))])
    plyfile.PlyData([plyfile.PlyElement.describe(verts, "vertex"), plyfile.PlyElement.describe(faces, "face")]).write(str(tmp_path / "hole.ply"))
    hole_mesh.main(["--mesh", str(tmp_path / "hole.ply"), "--removal_dir", str(rdir)])
    front = np.asarray(Image.open(rdir / "mesh_front" / "00000.png")) > 127
    covered = np.zeros((H, W), bool)
    covered[4:44, 12:52] = True
    expected = covered & ((depth >= 1.0) | (alpha < 128))
    inner = np.zeros((H, W), bool)
    inner[5:43, 13:51] = True  # rasterization edges aside
    assert np.array_equal(front[inner], expected[inner]) and not front[~covered].any()
    info = json.loads((rdir / "mesh_front.json").read_text())
    assert info["front_views"] == 1 and info["front_fraction_mean"] == pytest.approx(front.mean())


def test_eps2d_default(monkeypatch):
    monkeypatch.delenv("EPS2D", raising=False)
    assert render.default_eps2d() == 0.3
    monkeypatch.setenv("EPS2D", "0")
    assert render.default_eps2d() == 0.0
    monkeypatch.setenv("EPS2D", "")
    assert render.default_eps2d() == 0.3


def test_load_mask_2d_formats(tmp_path):
    ids = np.zeros((10, 20), np.uint8)
    ids[2:4, 2:4], ids[6:8, 10:12] = 1, 2
    np.save(tmp_path / "a.npy", ids)
    assert removal.load_mask_2d(tmp_path, "a", (20, 10)).sum() == 8
    assert removal.load_mask_2d(tmp_path, "a", (20, 10), label=2).sum() == 4
    beags = np.zeros((3, 10, 20), np.float16)
    beags[0] = 1
    beags[2, 0:5, 0:5] = 1.0
    beags[1, 5:10, 0:2] = 0.4  # below 0.5: not the object
    np.save(tmp_path / "b.npy", beags)
    assert removal.load_mask_2d(tmp_path, "b", (20, 10), label=2).sum() == 25
    assert removal.load_mask_2d(tmp_path, "b", (20, 10)).sum() == 0
    two = np.stack([np.ones((10, 20), np.uint8), np.zeros((10, 20), np.uint8)])
    two[1, :5, :5], two[0, :5, :5] = 1, 0
    np.save(tmp_path / "c.npy", two)
    assert removal.load_mask_2d(tmp_path, "c", (20, 10)).sum() == 25
    # PNG (RGB 0/255) at half resolution: resized NEAREST.
    png = np.zeros((5, 10, 3), np.uint8)
    png[1:3, 1:3] = 255
    Image.fromarray(png).save(tmp_path / "d.png")
    up = removal.load_mask_2d(tmp_path, "d", (20, 10))
    assert up.shape == (10, 20) and up.sum() == 16
    with pytest.raises(FileNotFoundError):
        removal.load_mask_2d(tmp_path, "missing", (20, 10))


# ---------------------------------------------------------------------------- 3D removal rule


def cube_points(rng, count, center=(0.0, 0.0, 0.0), half=1.0):
    return (rng.uniform(-half, half, (count, 3)) + np.asarray(center)).astype(np.float32)


def test_removal_mask_hull_rule():
    rng = np.random.default_rng(1)
    obj = cube_points(rng, 2000)
    obj[0] = (30.0, 0.0, 0.0)  # an outlier: trimmed away at 99.5
    background = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.2], [1.5, 0.0, 0.0], [1.05, 0.0, 0.0], [20.0, 0.0, 0.0]], np.float32)
    means = np.concatenate([obj, background])
    mask = np.zeros(len(means), bool)
    mask[: len(obj)] = True

    deleted, hull_points, margin = removal.removal_mask(means, mask, 99.5, 0.0)
    assert margin == 0.0 and len(hull_points) < len(obj) and np.abs(hull_points).max() < 1.0 + 1e-6
    assert deleted[: len(obj)].all()  # the object always goes, outlier included
    assert deleted[len(obj) :].tolist() == [True, True, False, False, False]
    # Without trimming the outlier stretches the hull out to x = 30.
    deleted, _, _ = removal.removal_mask(means, mask, 100.0, 0.0)
    assert deleted[len(obj) :].tolist() == [True, True, True, True, True]
    # expand: planes pushed out by expand x mean radius (~0.96 for a [-1,1] cube).
    deleted, _, margin = removal.removal_mask(means, mask, 99.5, 0.1)
    assert 0.08 < margin < 0.11
    assert deleted[len(obj) :].tolist() == [True, True, False, True, False]


def test_removal_mask_subsamples_large_objects_deterministically():
    rng = np.random.default_rng(2)
    means = cube_points(rng, 25_000)
    mask = np.ones(len(means), bool)
    mask[-10:] = False
    _, hull_a, _ = removal.removal_mask(means, mask, 100.0)
    _, hull_b, _ = removal.removal_mask(means, mask, 100.0)
    assert len(hull_a) == removal.MAX_HULL_POINTS and np.array_equal(hull_a, hull_b)
    with pytest.raises(SystemExit):
        removal.removal_mask(means[:3], np.ones(3, bool))


# ---------------------------------------------------------------------------- synthetic scene


def write_ply(path: Path, means: np.ndarray) -> None:
    n = len(means)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=[(name, "<f4") for name in names])
    data["x"], data["y"], data["z"] = means.T
    data["f_dc_0"] = np.arange(n) / max(n, 1)
    data["opacity"], data["rot_0"] = 2.0, 1.0
    data["scale_0"] = data["scale_1"] = data["scale_2"] = -3.0
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"] + [f"property float {name}" for name in names] + ["end_header"]
    removal.write_ply_vertices(path, header, data)


def test_ply_subset_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    means = cube_points(rng, 50)
    write_ply(tmp_path / "full.ply", means)
    header, data = removal.read_ply_vertices(tmp_path / "full.ply")
    keep = np.arange(50) % 3 == 0
    removal.write_ply_vertices(tmp_path / "sub.ply", header, data[keep])
    sub = read_ply(tmp_path / "sub.ply")
    full = read_ply(tmp_path / "full.ply")
    assert len(sub["means"]) == keep.sum()
    for key in ("means", "quats", "scales", "opacities", "sh"):
        np.testing.assert_array_equal(sub[key], full[key][keep])


def look_at_w2c(position: np.ndarray, target=np.zeros(3)) -> np.ndarray:
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = position
    return np.linalg.inv(c2w)


def write_points3d(path: Path, xyz: np.ndarray) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(xyz)))
        for k, p in enumerate(xyz):
            fid.write(struct.pack("<QdddBBBd", k + 1, *map(float, p), 128, 128, 128, 0.5))
            fid.write(struct.pack("<Q", 1) + struct.pack("<ii", 1, k))


def read_points3d_xyz(path: Path) -> np.ndarray:
    from artifixer_gs.colmap import read_colmap_points3D_binary

    return read_colmap_points3D_binary(path)[0]


def make_colmap(root: Path, n: int, rng) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    camera = Camera(id=1, model="PINHOLE", width=W, height=H, params=np.array([F, F, W / 2, H / 2]))
    images = []
    for i in range(n):
        angle = 2 * np.pi * i / n * 0.6  # an arc: not cyclic
        w2c = look_at_w2c(np.array([4 * np.cos(angle), 4 * np.sin(angle), 0.8]))
        name = f"IMG_{i:03d}.png"
        images.append(ColmapImage(id=i + 1, qvec=distill.colmap_qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1, name=name,
                                  xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, dtype=np.int64)))
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(root / "images" / name)
    write_colmap_cameras(root / "sparse" / "0" / "cameras.bin", [camera])
    write_colmap_images(root / "sparse" / "0" / "images.bin", images)
    points = np.concatenate([cube_points(rng, 30, half=0.15), cube_points(rng, 30, half=2.5) + [0, 0, -3.0]])
    write_points3d(root / "sparse" / "0" / "points3D.bin", points)
    return root


def project(means: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
    ok = cam[:, 2] > 1e-3
    uv = cam[ok, :2] / cam[ok, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return np.round(uv).astype(int), ok


def splat(means: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels hit by the projected centres (3x3 blocks): a stand-in for a rasterised footprint."""
    image = np.zeros((H, W), bool)
    uv, _ = project(means, viewmat, K)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            u, v = uv[:, 0] + du, uv[:, 1] + dv
            keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            image[v[keep], u[keep]] = True
    return image


def stub_render(*, ply, trajectory_path, output_dir, selected_indices, depth=False):
    """render_ply's contract (renders/, opacity/, [depth/,] selected_indices.json) without gsplat."""
    means = read_ply(Path(ply))["means"]
    viewmats, Ks, width, height = trajectory_cameras(Path(trajectory_path))
    assert (width, height) == (W, H)
    for sub in ("renders", "opacity", *(("depth",) if depth else ())):
        (Path(output_dir) / sub).mkdir(parents=True, exist_ok=True)
    for i, (viewmat, K) in enumerate(zip(viewmats, Ks)):
        alpha = splat(means, viewmat, K)
        rgb = np.where(alpha[..., None], np.array([50, 100, 150], np.uint8), 0).astype(np.uint8)
        Image.fromarray(rgb).save(Path(output_dir) / "renders" / f"{i:05d}.png")
        Image.fromarray(alpha.astype(np.uint8) * 255).save(Path(output_dir) / "opacity" / f"{i:05d}.png")
        if depth:
            np.save(Path(output_dir) / "depth" / f"{i:05d}.npy", np.ones((H, W), np.float32))
    (Path(output_dir) / "selected_indices.json").write_text(json.dumps(selected_indices))
    return Path(output_dir)


def write_fake_caption(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        dataset = hf.create_dataset("frame_00000", data=np.zeros((3, 4096), np.uint16))
        dataset.attrs["caption"] = "test"
        dataset.attrs["image_indices"] = np.array([0])


@pytest.fixture()
def scene(tmp_path):
    """Prepared root (prepare,scale; no render) + PLY + object mask + BEA-GS style 2D masks."""
    rng = np.random.default_rng(0)
    n = 12
    colmap = make_colmap(tmp_path / "colmap", n, rng)
    obj = cube_points(rng, 300, half=0.4)
    floor = cube_points(rng, 400, half=2.0) * [1, 1, 0.02] + [0, 0, -0.45]
    inside = cube_points(rng, 20, half=0.15)  # unlabelled Gaussians inside the object: the hull removes them
    means = np.concatenate([obj, floor, inside])
    object_mask = np.zeros(len(means), bool)
    object_mask[: len(obj)] = True
    write_ply(tmp_path / "scene.ply", means)
    np.save(tmp_path / "object_mask.npy", object_mask)

    out = tmp_path / "out" / "synth"
    prepare_scene(prepare_parser().parse_args([
        "--ply", str(tmp_path / "scene.ply"), "--colmap_dir", str(colmap), "--output_root", str(out),
        "--phases", "prepare,scale", "--metric_scale", "2.0",
    ]))
    write_fake_caption(out / "captions" / "synth" / "caption.h5")

    masks = tmp_path / "masks"
    masks.mkdir()
    viewmats, Ks, _, _ = trajectory_cameras(out / "input" / "synth" / "nerfstudio" / "transforms.json")
    frames = json.loads((out / "input" / "synth" / "nerfstudio" / "transforms.json").read_text())["frames"]
    for frame, viewmat, K in zip(frames, viewmats, Ks):
        m = splat(obj, viewmat, K).astype(np.uint8)
        np.save(masks / f"{Path(frame['file_path']).stem}.npy", np.stack([1 - m, m]))  # (2,H,W) [~obj, obj]
    return {"root": tmp_path, "out": out, "colmap": colmap, "means": means, "object_mask": object_mask, "masks": masks, "n": n}


def removal_render(scene, *flags, **kwargs):
    out = scene["out"]
    argv = ["render", "--ply", str(scene["root"] / "scene.ply"), "--object_mask", str(scene["root"] / "object_mask.npy"),
            "--split_path", str(out), "--output_dir", str(out / "removal"), *flags]
    for key, value in kwargs.items():
        argv += [f"--{key}", str(value)]
    removal.main(argv, render=stub_render)
    return out / "removal"


def removal_split(scene, *extra):
    out = scene["out"]
    removal.main(["split", "--scene_root", str(out), "--removal_dir", str(out / "removal"), "--mask_dir", str(scene["masks"]),
                  "--output_dir", str(out / "removal_split"), "--dilate_px", "2", *extra])
    return out / "removal_split"


def test_removal_render_and_split(scene):
    rdir = removal_render(scene)
    info = json.loads((rdir / "removal.json").read_text())
    assert info["object"] == 300 and info["hull_deleted"] == 20 and info["kept"] == 400 and info["frames"] == scene["n"]
    assert len(read_ply(rdir / "removed.ply")["means"]) == 400 and len(read_ply(rdir / "deleted.ply")["means"]) == 320
    deleted = np.load(rdir / "deleted_mask.npy")
    assert deleted[:300].all() and deleted[-20:].all() and not deleted[300:700].any()
    for sub in ("background/renders", "background/opacity", "deleted/opacity"):
        assert len(list((rdir / sub).glob("*.png"))) == scene["n"]

    sdir = removal_split(scene)
    split = json.loads((sdir / "split.json").read_text())["test"]["synth"]
    assert split["has_gt"] is False and split["camera_scale"] == pytest.approx(0.02) and split["metric_scale"] == 2.0
    assert json.loads((sdir / "selected_indices.json").read_text()) == list(range(scene["n"]))
    assert json.loads((sdir / "frames.json").read_text())["auto_seeds"] == "0-6"
    transforms = json.loads((sdir / "transforms.json").read_text())
    assert transforms["frames"][0]["file_path"] == "images/IMG_000.png"
    for k in range(scene["n"]):
        hole = np.asarray(Image.open(sdir / "hole" / f"{k:05d}.png")) > 0
        opacity = np.asarray(Image.open(sdir / "opacity" / f"{k:05d}.png"))
        deleted_alpha = np.asarray(Image.open(rdir / "deleted" / "opacity" / f"{k:05d}.png")) > 0
        assert hole.any() and (opacity[hole] == 0).all() and hole[deleted_alpha].all()
        ref = np.asarray(Image.open(sdir / "refs" / "images" / f"IMG_{k:03d}.png"))
        assert np.array_equal(ref, np.asarray(Image.open(sdir / "renders" / f"{k:05d}.png")))  # --refs render
    scene_split = json.loads((sdir / "scene" / "split.json").read_text())["test"]["synth"]
    assert Path(scene_split["image_root"]) == (scene["out"] / "input" / "synth").resolve()

    # The split is what run_inference --render_trajectory all_frames reads.
    torch = pytest.importorskip("torch")
    from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
    from model_training.data.utils import NeighborSelectionMode

    dataset = ReconstructedColmapEvalDataset(split="test", split_path=sdir / "split.json", num_views=6,
                                             neighbor_selection_mode=NeighborSelectionMode.EVENLY_SPACED, include_all_frames=True)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["frame_indices"].tolist() == list(range(scene["n"])) and item["target_h"] == H
    assert isinstance(item["opacity"], torch.Tensor)


def test_split_hole_modes(scene):
    rdir = removal_render(scene, "--depth")
    assert json.loads((rdir / "removal.json").read_text())["depth"] is True
    assert len(list((rdir / "background" / "depth").glob("*.npy"))) == scene["n"] and not (rdir / "deleted" / "depth").exists()
    holes = {}
    for mode in ("dilate", "alpha"):
        sdir = removal_split(scene, "--hole_mode", mode)
        assert json.loads((sdir / "split.json").read_text())["test"]["synth"]["hole_mode"] == mode
        holes[mode] = [np.asarray(Image.open(sdir / "hole" / f"{k:05d}.png")) > 0 for k in range(scene["n"])]
    for k in range(scene["n"]):
        background = np.asarray(Image.open(rdir / "background" / "opacity" / f"{k:05d}.png"))
        deleted = np.asarray(Image.open(rdir / "deleted" / "opacity" / f"{k:05d}.png"))
        assert np.array_equal(holes["alpha"][k], holes["dilate"][k] & ((background < 128) | (deleted > 12)))
        assert holes["alpha"][k].sum() < holes["dilate"][k].sum()
    with pytest.raises(AssertionError, match="mesh_front"):
        removal_split(scene, "--hole_mode", "mesh")
    (rdir / "mesh_front").mkdir()
    for k in range(scene["n"]):
        front = np.zeros((H, W), np.uint8)
        front[:3, :3] = 255
        Image.fromarray(front).save(rdir / "mesh_front" / f"{k:05d}.png")
    sdir = removal_split(scene, "--hole_mode", "mesh")
    for k in range(scene["n"]):
        hole = np.asarray(Image.open(sdir / "hole" / f"{k:05d}.png")) > 0
        assert hole[:3, :3].all() and np.array_equal(hole[3:], holes["alpha"][k][3:])


def test_split_confidence_union(scene):
    rdir = removal_render(scene)
    alpha_dir = removal_split(scene, "--hole_mode", "alpha")
    alpha = [np.asarray(Image.open(alpha_dir / "hole" / f"{k:05d}.png")) > 0 for k in range(scene["n"])]
    assert "conf_threshold" not in json.loads((alpha_dir / "split.json").read_text())["test"]["synth"]
    dilate_dir = scene["out"] / "split_dilate"
    removal_split(scene, "--hole_mode", "dilate", "--output_dir", str(dilate_dir))
    dilated = [np.asarray(Image.open(dilate_dir / "hole" / f"{k:05d}.png")) > 0 for k in range(scene["n"])]
    conf = scene["root"] / "confidence"
    conf.mkdir()
    lows = {}
    for k in range(0, scene["n"], 2):  # maps for every other frame, at half the render size (resized BILINEAR)
        c = np.full((H // 2, W // 2), 255, np.uint8)
        c[: H // 4] = 50  # top half: confidence 0.2
        Image.fromarray(c).save(conf / f"IMG_{k:03d}.png")
        lows[k] = removal.load_confidence(conf, f"IMG_{k:03d}", (W, H)) < 0.5
    sdir = removal_split(scene, "--hole_mode", "alpha", "--confidence_dir", str(conf), "--conf_threshold", "0.5")
    entry = json.loads((sdir / "split.json").read_text())["test"]["synth"]
    assert entry["conf_threshold"] == 0.5 and entry["confidence_frames"] == len(lows) and entry["confidence_dir"] == str(conf.resolve())
    added = 0
    for k in range(scene["n"]):
        hole = np.asarray(Image.open(sdir / "hole" / f"{k:05d}.png")) > 0
        opacity = np.asarray(Image.open(sdir / "opacity" / f"{k:05d}.png"))
        expected = alpha[k] | (dilated[k] & lows[k]) if k in lows else alpha[k]
        assert np.array_equal(hole, expected) and (opacity[hole] == 0).all()
        added += int(hole.sum() - alpha[k].sum())
    assert added > 0
    with pytest.raises(AssertionError, match="go together"):
        removal_split(scene, "--hole_mode", "alpha", "--conf_threshold", "0.5")
    with pytest.raises(AssertionError, match="in \\(0, 1\\]"):
        removal_split(scene, "--hole_mode", "alpha", "--confidence_dir", str(conf), "--conf_threshold", "2")
    empty = scene["root"] / "no_maps"
    empty.mkdir()
    with pytest.raises(AssertionError, match="no confidence map"):
        removal_split(scene, "--hole_mode", "alpha", "--confidence_dir", str(empty), "--conf_threshold", "0.5")
    assert rdir.is_dir()


def test_split_gavis(scene):
    rdir = removal_render(scene)
    dilate_dir = scene["out"] / "split_dilate"
    removal_split(scene, "--hole_mode", "dilate", "--output_dir", str(dilate_dir))
    dilated = [np.asarray(Image.open(dilate_dir / "hole" / f"{k:05d}.png")) > 0 for k in range(scene["n"])]
    alpha_dir = scene["out"] / "split_alpha"
    removal_split(scene, "--hole_mode", "alpha", "--output_dir", str(alpha_dir))
    alpha = [np.asarray(Image.open(alpha_dir / "hole" / f"{k:05d}.png")) > 0 for k in range(scene["n"])]
    with pytest.raises(AssertionError, match="needs --unseen_dir"):
        removal_split(scene, "--hole_mode", "gavis")
    unseen = scene["root"] / "unseen"
    unseen.mkdir()
    for k in range(0, scene["n"], 2):  # maps for every other frame, at half the render size: left half unseen (U 0.8)
        u = np.zeros((H // 2, W // 2), np.uint8)
        u[:, : W // 4] = 204
        Image.fromarray(u).save(unseen / f"IMG_{k:03d}.png")
    sdir = removal_split(scene, "--hole_mode", "gavis", "--unseen_dir", str(unseen))
    entry = json.loads((sdir / "split.json").read_text())["test"]["synth"]
    assert entry["hole_mode"] == "gavis" and entry["unseen_threshold"] == 0.5 and entry["unseen_frames"] == (scene["n"] + 1) // 2
    for k in range(scene["n"]):
        hole = np.asarray(Image.open(sdir / "hole" / f"{k:05d}.png")) > 0
        deleted = np.asarray(Image.open(rdir / "deleted" / "opacity" / f"{k:05d}.png")) > 12
        if k % 2:
            assert np.array_equal(hole, alpha[k])                              # no map: alpha's rule
        else:
            high = removal.load_unseen(unseen, f"IMG_{k:03d}", (W, H)) > 0.5
            assert np.array_equal(hole, dilated[k] & (high | deleted))
    sdir = removal_split(scene, "--hole_mode", "gavis", "--unseen_dir", str(unseen), "--unseen_threshold", "0.9")
    for k in range(0, scene["n"], 2):                                          # nothing above 0.9: deleted only
        deleted = np.asarray(Image.open(rdir / "deleted" / "opacity" / f"{k:05d}.png")) > 12
        assert np.array_equal(np.asarray(Image.open(sdir / "hole" / f"{k:05d}.png")) > 0, dilated[k] & deleted)
    empty = scene["root"] / "no_unseen"
    empty.mkdir()
    with pytest.raises(AssertionError, match="no unseen map"):
        removal_split(scene, "--hole_mode", "gavis", "--unseen_dir", str(empty))


def test_split_frames_subset_hull_and_labels(scene):
    removal_render(scene)
    sdir = removal_split(scene, "--frames", "1-4,8-10", "--hole_shape", "hull", "--mask_label", "1", "--refs", "composite")
    info = json.loads((sdir / "frames.json").read_text())
    assert info == {"source_indices": [1, 2, 3, 4, 8, 9, 10], "window_starts": [0, 4], "auto_seeds": "0-3,4-6"}
    transforms = json.loads((sdir / "transforms.json").read_text())
    assert [Path(f["file_path"]).name for f in transforms["frames"]] == [f"IMG_{i:03d}.png" for i in info["source_indices"]]
    scene_transforms = json.loads((sdir / "scene" / "transforms.json").read_text())
    assert scene_transforms["frames"][4]["file_path"] == "images/IMG_008.png"
    hole = np.asarray(Image.open(sdir / "hole" / "00004.png")) > 0
    background = np.asarray(Image.open(sdir / "renders" / "00004.png"))
    photo = np.asarray(Image.open(scene["out"] / "input" / "synth" / "images" / "IMG_008.png"))
    ref = np.asarray(Image.open(sdir / "refs" / "images" / "IMG_008.png"))
    assert np.array_equal(ref[hole], background[hole]) and np.array_equal(ref[~hole], photo[~hole])


# ---------------------------------------------------------------------------- propagation logic


def test_sequence_structure_cuts_and_cycles():
    line = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], 1)
    line[6:] += [50.0, 0, 0]
    bounds, cyclic, seq_of = propagate.sequence_structure(line, 4.0)
    assert bounds == [(0, 6), (6, 10)] and not cyclic and seq_of.tolist() == [0] * 6 + [1] * 4
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles), np.zeros(20)], 1)
    bounds, cyclic, _ = propagate.sequence_structure(circle, 4.0)
    assert bounds == [(0, 20)] and cyclic


def test_frontier_chunks_and_mini_split():
    n, seq_of = 20, np.zeros(20, int)
    done = {i: 0 for i in range(7)}
    chunks = propagate.frontier_chunks(done, n, 7, False, seq_of)
    assert chunks == [(6, [7, 8, 9, 10, 11, 12, 13])]
    order, selected, targets = propagate.mini_split_order(chunks, done)
    assert order == [6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5]
    assert selected == [0, 8, 9, 10, 11, 12, 13] and targets == [1, 2, 3, 4, 5, 6, 7]
    # Cyclic: one seed grows both ways; the far side is claimed once.
    chunks = propagate.frontier_chunks({0: 0}, 10, 3, True, np.zeros(10, int))
    assert chunks == [(0, [1, 2, 3]), (0, [9, 8, 7])]
    # Two seeds in the middle: chunks stop at already-claimed frames.
    chunks = propagate.frontier_chunks({3: 0, 9: 0}, 12, 7, False, np.zeros(12, int))
    assert chunks == [(3, [4, 5, 6, 7, 8]), (3, [2, 1, 0]), (9, [10, 11])]
    # A chunk may start across a cut but never spans one.
    seq_of = np.array([0] * 5 + [1] * 5)
    chunks = propagate.frontier_chunks({3: 0}, 10, 7, False, seq_of)
    assert chunks == [(3, [4]), (3, [2, 1, 0])]
    chunks = propagate.frontier_chunks({i: 0 for i in range(5)}, 10, 7, False, seq_of)
    assert chunks == [(4, [5, 6, 7, 8, 9])]


def make_generator(colors: dict, validate: bool = True):
    """Stub ArtiFixer: a flat colour per round at every target, after loading the mini split like run_inference."""

    def generate(split_path: Path, gen_dir: Path, num_views: int) -> None:
        split_dir = split_path.parent
        (scene_id, entry), = json.loads(split_path.read_text())["test"].items()
        targets = json.loads((split_dir / "targets.json").read_text())
        selected = json.loads((split_dir / "selected.json").read_text())
        assert num_views == min(6, len(selected)) and not set(targets) & set(selected)
        if validate:
            from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
            from model_training.data.utils import NeighborSelectionMode

            dataset = ReconstructedColmapEvalDataset(split="test", split_path=split_path, num_views=num_views,
                                                     neighbor_selection_mode=NeighborSelectionMode.COVISIBILITY,
                                                     use_target_indices=True)
            assert sorted(i for _, pair in dataset.inference_items for i in pair.test_indices) == targets
            dataset[0]
        colors["round"] = colors.get("round", 0) + 1
        pred = gen_dir / scene_id / "frames" / "batch_0000" / "pred"
        pred.mkdir(parents=True)
        for pos in targets:
            Image.fromarray(np.full((H, W, 3), 10 * colors["round"], np.uint8)).save(pred / f"{pos:05d}.png")

    return generate


def test_propagate_handoff_and_plus(scene, monkeypatch):
    pytest.importorskip("torch")
    removal_render(scene)
    sdir = removal_split(scene)
    out = scene["out"]
    seed_pred = out / "seed_pred"
    seed_pred.mkdir()
    for k in range(scene["n"]):
        Image.fromarray(np.full((H, W, 3), 200, np.uint8)).save(seed_pred / f"{k:05d}.png")
    colors: dict = {}
    propagate.main([
        "--scene_root", str(sdir / "scene"), "--removal_dir", str(sdir), "--seed_pred_dir", str(seed_pred),
        "--seed_frames", "auto", "--output_dir", str(out / "propagation"), "--num_views", "6", "--step", "3",
        "--checkpoint_pt", "unused.pt", "--model_id", "unused",
    ], generate=make_generator(colors))
    order = json.loads((out / "propagation" / "order.json").read_text())
    assert [order[f"{i:05d}"] for i in range(scene["n"])] == [0] * 7 + [1, 1, 1, 2, 2]
    assert not list((out / "propagation" / "rounds").iterdir())  # round dirs removed without --keep_rounds
    # Composite: the generation inside the hole, the photo far outside it.
    hole = np.asarray(Image.open(sdir / "hole" / "00008.png")) > 0
    anchor = np.asarray(Image.open(out / "propagation" / "anchors" / "00008.png"))
    photo = np.asarray(Image.open(out / "input" / "synth" / "images" / "IMG_008.png"))
    assert (anchor[hole] == 10).all()
    from scipy.ndimage import binary_dilation

    far = ~binary_dilation(hole, iterations=12)
    assert np.array_equal(anchor[far], photo[far])

    # Handoff: composites at the photo poses, anchors every 3rd + seeds, the rest LPIPS-only.
    af3d = out / "af3d" / "input"
    distill.removal_handoff(sdir / "split.json", out / "propagation", af3d, anchor_every=3)
    anchors = json.loads((af3d / "selected_indices.json").read_text())
    assert anchors == [0, 1, 2, 3, 4, 5, 6, 9]
    generated = (af3d / "generated_images.txt").read_text().split()
    assert generated == ["frame_00007.png", "frame_00008.png", "frame_00010.png", "frame_00011.png"]
    images = read_colmap_extrinsics_binary(af3d / "sparse" / "0" / "images.bin")
    cameras = read_colmap_intrinsics_binary(af3d / "sparse" / "0" / "cameras.bin")
    source = {im.name: im for im in read_colmap_extrinsics_binary(scene["colmap"] / "sparse" / "0" / "images.bin")}
    assert [im.name for im in images] == [f"frame_{i:05d}.png" for i in range(scene["n"])]
    assert len(cameras) == 1 and cameras[images[0].camera_id].model == "PINHOLE"
    np.testing.assert_allclose(cameras[images[0].camera_id].params, [F, F, W / 2, H / 2])
    for im in images:
        src = source[f"IMG_{im.id - 1:03d}.png"]
        np.testing.assert_allclose(im.qvec, src.qvec, atol=1e-12)
        np.testing.assert_allclose(im.tvec, src.tvec, atol=1e-12)
        assert np.array_equal(np.asarray(Image.open(af3d / "images" / im.name)),
                              np.asarray(Image.open(out / "propagation" / "anchors" / f"{im.id - 1:05d}.png")))
    assert len(read_points3d_xyz(af3d / "sparse" / "0" / "points3D.bin")) == 60

    # --drop_points_mask: the 30 points inside the object's hull are not used for initialisation.
    distill.removal_handoff(sdir / "split.json", out / "propagation", out / "af3d" / "dropped", drop_points_mask=True)
    xyz = read_points3d_xyz(out / "af3d" / "dropped" / "sparse" / "0" / "points3D.bin")
    assert len(xyz) == 30 and xyz[:, 2].max() <= -0.5

    # plus on the ArtiFixer3D split: val_frames = the non-anchors, references = anchor composites.
    monkeypatch.setattr(distill, "render_ply", stub_render)
    scene_af3d = distill.load_prepared_scene(af3d / "split.json")
    assert scene_af3d.selected_indices == anchors and not scene_af3d.has_gt and scene_af3d.colmap_dir == af3d
    write_ply(out / "af3d.ply", scene["means"])
    plus_split = distill.plus(scene_af3d, out / "af3d.ply", out / "af3d" / "renders", out / "split_artifixer3d_plus.json")
    from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
    from model_training.data.utils import NeighborSelectionMode

    dataset = ReconstructedColmapEvalDataset(split="test", split_path=plus_split, num_views=6,
                                             neighbor_selection_mode=NeighborSelectionMode.COVISIBILITY)
    tests = sorted({i for _, pair in dataset.inference_items for i, t in zip(pair.test_indices, pair.is_test_frame or [True] * 99) if t})
    assert tests == [7, 8, 10, 11]
    assert all(set(pair.neighbor_indices) <= set(anchors) for _, pair in dataset.inference_items)
    dataset[0]


def test_render_ply_uses_render_cmd(tmp_path, monkeypatch):
    """RENDER_CMD replaces gsplat: run in RENDER_CWD with this repo on PYTHONPATH, same output contract."""
    from artifixer_gs import render as render_module

    transforms = tmp_path / "transforms.json"
    transforms.write_text(json.dumps({"camera_model": "OPENCV", "w": 4, "h": 3, "fl_x": 2.0, "fl_y": 2.0, "cx": 2.0, "cy": 1.5, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "frames": [
        {"file_path": f"images/{i}.png", "transform_matrix": np.eye(4).tolist()} for i in range(2)]}))
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    stub = tmp_path / "stub_renderer.py"
    stub.write_text(
        "import argparse, json, os, sys\n"
        "from pathlib import Path\n"
        "from PIL import Image\n"
        "p = argparse.ArgumentParser(); p.add_argument('--flag'); p.add_argument('--ply'); p.add_argument('--trajectory')\n"
        "p.add_argument('--out_dir'); p.add_argument('--depth', action='store_true'); a = p.parse_args()\n"
        "n = len(json.loads(Path(a.trajectory).read_text())['frames'])\n"
        "for sub in ('renders', 'opacity'):\n"
        "    (Path(a.out_dir) / sub).mkdir(parents=True, exist_ok=True)\n"
        "    for i in range(n): Image.new('L', (4, 3)).save(Path(a.out_dir) / sub / f'{i:05d}.png')\n"
        "Path(a.out_dir, 'call.json').write_text(json.dumps({'cwd': os.getcwd(), 'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "    'flag': a.flag, 'ply': a.ply, 'depth': a.depth}))\n")
    cwd = tmp_path / "gw_repo"
    cwd.mkdir()
    monkeypatch.setenv("RENDER_CMD", f"{sys.executable} {stub} --flag 'a b'")
    monkeypatch.setenv("RENDER_CWD", str(cwd))
    monkeypatch.setattr(render_module, "render_frames", lambda *a, **k: pytest.fail("gsplat must not render"))
    out = render_module.render_ply(ply=ply, trajectory_path=transforms, output_dir=tmp_path / "out", selected_indices=[1])
    call = json.loads((out / "call.json").read_text())
    assert call["cwd"] == str(cwd) and call["pythonpath"] == str(render_module.REPO_ROOT)
    assert call["flag"] == "a b" and call["ply"] == str(ply.resolve()) and call["depth"] is False
    assert json.loads((out / "selected_indices.json").read_text()) == [1]
