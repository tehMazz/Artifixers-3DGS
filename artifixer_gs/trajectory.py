#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a novel camera path for vanilla ArtiFixer3D from a scene's photo cameras and a 3DGS PLY.

Port of ArtiFixer's scripts/bakerh/make_trajectory.py with the 3DGUT parts replaced: the Gaussians
come from a standard 3DGS PLY (--ply, any method that writes the INRIA layout: 3DGS, FastGS,
BEA-GS), the object from a per-Gaussian mask (--object_mask, bool [P] in PLY vertex order), the
render checks use gsplat. The photo cameras are a prepared transforms.json (--transforms, as
artifixer_gs.prepare writes it to <root>/input/<id>/nerfstudio/transforms.json) or are read from
the COLMAP scene directly (--colmap_dir [--images] [--image_downscale_factor]), with the helpers
prepare uses, so intrinsics, names and world frame match what prepare writes. The PLY must live in
that COLMAP world frame. With both, the photos come from --transforms and are checked against the
COLMAP poses (all names present, positions within 1e-3 photo spacings), as the original did; with
--transforms alone the world frame is not checked.

Reads the photos (OpenGL camera-to-world) and writes a camera path in one of three modes (--mode):

  travel (default without --object_mask): one camera per photo, in photo (file name) order, on
    the photo path Gaussian-smoothed in position and rotation (--sigma photos, within runs of
    photos without a jump larger than --jump x the median photo spacing), moved --side x spacing x
    sin(...) across the direction of travel (--periods weaves) and --up x spacing up, keeping the
    photo rotations.
  object (default with --object_mask): the same smoothed path, built around the object. Its
    centre of mass and principal extents come from the object Gaussians (--object_mask), each
    weighted by its opacity; its radius R is the largest robust half-extent. Around the vertical
    axis through the centre of mass every camera orbits by an arc of up to --orbit x R, moves
    toward/away from the object by up to --radial x R and rises by --rise x R. It never gets closer
    to the object's ellipsoid than --standoff x R (or than its photo, if that was closer), and it
    looks at the centre of mass (--look_at blends the photo's rotation, 0, with looking at the
    object, 1).
  vidsplat: short orbits in the spirit of VidSplat (Tang et al., SIGGRAPH 2026, Sec. 3.3,
    "visibility-based camera pose sampling"). For seed photos spread evenly over the capture (--seeds,
    at least two per orbit; --phase_offset moves them by that share of one seed spacing), the anchor
    is where the photo's central ray meets the surface; 8 directions x --orbit_deg extents of
    orbits about the anchor are tried (yaw about the vertical, pitch about the horizontal, the
    camera rigidly rotated so the anchor stays at the image centre), --clip_len frames each. A
    candidate is valid when, on its keyframes (the last frame of every 4-frame group), the view is
    not blocked by a near surface (paper: min depth > d0), shows no more than --s_high unseen area
    (area of surface no photo observes; a camera behind a wall sees the wall's unobserved back),
    and at its end shows at least --s_low more unseen area than the photo (paper: S_low < area(M)).
    Per seed the valid orbit that reveals the most is kept; the capture is cut into one stretch per
    orbit wanted and each stretch gets its seed with the most revealing orbit (a stretch without a valid
    one leaves its place to the next best seed anywhere, then to second orbits). The path is frame 0 = the first seed
    photo, then the clips in capture order: every clip starts at an index = 1 (mod 4), so no
    4-frame VAE group of ArtiFixer mixes two orbits. The paper's +25% tail is not generated.

Collision checks. With --ply the Gaussians become an occupancy field (see Occupancy).
In travel and object mode every path camera must
  (a) keep --clearance from any surface, the object included (a base camera already closer than
      that, because the photographer stood there, keeps its own distance instead: the largest of
      half, a quarter or an eighth of the clearance it has),
  (b) be reachable from its base camera in a straight line through free space, so it can never
      end up inside or behind a wall, and
  (c) in object mode, see the centre of mass with no surface in between, wherever its base camera
      does (--no_visibility: off), and with --render_check not have its view blocked by a near
      surface (rendered, as the vidsplat mode does).
A camera that fails is pulled back toward its base camera (its offset scaled down in --levels
steps), and the offset scale is smoothed over neighbouring frames without ever exceeding what
each frame allows. Scale 0 is the base camera, where a photo was taken, i.e. free space. The
vidsplat mode applies (a) to every orbit frame and (b) between consecutive frames.
Gaussians that contain a photo camera (floaters) are left out of the field, since a camera stood
there. The clearance is --clearance scene units, else --clearance_m metres with --metric_scale
(metres per scene unit, as in metric_alignment/scale_info.txt), else 0.15 R (object mode) or one
photo spacing. d0 is --d0_m metres with --metric_scale, else the clearance (object mode) or a
quarter of the anchor distance (vidsplat mode), and never more than 0.8 x the photo's own nearest
surface, so a photo taken close to a wall does not fail by construction.

Failing loudly. The object mode fails (exit 2) when more than --max_collapsed of its frames had to
be pulled back to less than a quarter of their offset, i.e. the path collapsed onto the photos
(--allow_fallback accepts that); the vidsplat mode fails when fewer than --min_clips valid orbits
exist. The info file and the plot are written either way, with the reason.

The render checks (vidsplat, --render_check) rasterize the PLY with gsplat on the GPU at
--check_width pixels wide (expected depth and alpha, the photos' pinhole intrinsics, lens
distortion ignored as the 3DGUT checks did). A point of a rendered view counts as seen when some
photo frames it, has geometry there, and its depth agrees within --depth_tol (relative) in a 3x3
neighbourhood.

The output is the transforms-style trajectory JSON that artifixer_gs.prepare --trajectory_path
takes: the photos' shared OPENCV intrinsics and frames with transform_matrix only. Next to it:
<output>_info.json (mode, clearance, per-frame offset scale or per-clip stats, rejection counts,
failure reason) and, with matplotlib, <output>.png, a top view of the walls at camera height, the
object, photos and path.

  python -m artifixer_gs.trajectory --ply point_cloud.ply --colmap_dir scene --images images_4 \\
      --output traj.json --mode object --object_mask object_mask.npy --metric_scale 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL <-> OpenCV camera axes (c2w_cv = c2w_gl @ FLIP)
REJECT_REASONS = ("no_anchor", "too_close", "wall", "near_plane", "too_unseen", "empty", "nothing_new")


class TrajectoryError(Exception):
    """No acceptable path: main() writes the info file and the plot, then exits with code 2."""

    def __init__(self, reason, path=None, info=None):
        super().__init__(reason)
        self.reason, self.path, self.info = reason, path, info or {}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transforms", type=Path, default=None,
                        help="Photo transforms.json, e.g. a prepared <root>/input/<id>/nerfstudio/transforms.json. "
                             "With --colmap_dir too, its photos are checked against the COLMAP poses (world frame).")
    parser.add_argument("--colmap_dir", type=Path, default=None,
                        help="COLMAP scene of the PLY (sparse/0 + --images). Without --transforms the photo transforms "
                             "are built from it as prepare builds them.")
    parser.add_argument("--images", default="images", help="Image folder of --colmap_dir whose size sets the intrinsics.")
    parser.add_argument("--normalize_near_shared_cameras", action="store_true",
                        help="As artifixer_gs.prepare: merge near-identical COLMAP cameras into one (--colmap_dir).")
    parser.add_argument("--image_downscale_factor", type=int, default=1,
                        help="As artifixer_gs.prepare: intrinsics of the photos downscaled by this factor (--colmap_dir).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("travel", "object", "vidsplat"), default=None,
                        help="Default: object with --object_mask, else travel.")
    parser.add_argument("--periods", type=float, default=2.0, help="Weaves over the whole path.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Smoothing width, in photos.")
    parser.add_argument("--jump", type=float, default=4.0, help="A step above this many spacings splits the path.")
    parser.add_argument("--phase_offset", type=float, default=0.0,
                        help="Shift of the weave (travel/object) or of the seed photos (vidsplat), in periods/seed "
                             "spacings: repeated rounds with different offsets pick different viewpoints.")
    parser.add_argument("--seen_transforms", type=Path, default=None,
                        help="Trajectory JSON of views already generated (earlier rounds): their surface counts as "
                             "seen in the unseen test, so new orbits look for what is still missing.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="At most this many path frames (travel/object: every k-th frame; vidsplat: the default budget).")
    travel = parser.add_argument_group("travel mode, in median photo spacings")
    travel.add_argument("--side", type=float, default=2.0, help="Weave amplitude across the direction of travel.")
    travel.add_argument("--up", type=float, default=0.5, help="Constant upward shift.")
    obj = parser.add_argument_group("object mode, in object radii R")
    obj.add_argument("--object_mask", type=Path, default=None,
                     help="object_mask.npy: bool [P], True for the object's Gaussians, in --ply vertex order.")
    obj.add_argument("--orbit", type=float, default=0.6, help="Arc of the weave around the object.")
    obj.add_argument("--radial", type=float, default=0.3, help="Toward/away amplitude.")
    obj.add_argument("--rise", type=float, default=0.3, help="Constant upward shift.")
    obj.add_argument("--standoff", type=float, default=0.5, help="Minimum distance from the object's ellipsoid.")
    obj.add_argument("--look_at", type=float, default=1.0, help="0 keeps the photo's rotation, 1 looks at the centre of mass.")
    obj.add_argument("--max_collapsed", type=float, default=0.5,
                     help="Fail when more than this share of frames keeps under a quarter of its offset.")
    obj.add_argument("--allow_fallback", action="store_true", help="Accept a collapsed path instead of failing.")
    obj.add_argument("--render_check", action="store_true", help="Also reject frames whose rendered view a near surface blocks (GPU).")
    vid = parser.add_argument_group("vidsplat mode")
    vid.add_argument("--clip_len", type=int, default=16, help="Frames per orbit, a multiple of 4.")
    vid.add_argument("--orbit_deg", type=str, default="15,30,45", help="Orbit extents tried, degrees.")
    vid.add_argument("--max_elev", type=float, default=60.0, help="Highest/lowest camera elevation about the anchor, degrees.")
    vid.add_argument("--budget", type=int, default=None, help="Total frames; default one per photo.")
    vid.add_argument("--seeds", type=int, default=12,
                     help="Seed photos tried, spread evenly over the capture (at least two per orbit wanted).")
    vid.add_argument("--min_clips", type=int, default=None, help="Fail below this many valid orbits; default half of them.")
    vid.add_argument("--s_low", type=float, default=0.03, help="Minimum extra unseen area at the end of an orbit.")
    vid.add_argument("--s_high", type=float, default=0.4, help="Maximum unseen (and extra empty) area on a keyframe.")
    checks = parser.add_argument_group("collision checks")
    checks.add_argument("--ply", type=Path, default=None, help="3DGS PLY: enables the wall checks (needed by object and vidsplat).")
    checks.add_argument("--clearance", type=float, default=None, help="Scene units.")
    checks.add_argument("--clearance_m", type=float, default=0.3, help="Metres, used with --metric_scale.")
    checks.add_argument("--metric_scale", type=float, default=None, help="Metres per scene unit.")
    checks.add_argument("--solid", type=float, default=0.5, help="A point is solid where a Gaussian this opaque, widened by the clearance, covers it.")
    checks.add_argument("--levels", type=int, default=9, help="Offset scales tried, from 1 down to 0.")
    checks.add_argument("--no_visibility", action="store_true", help="Object mode: do not require a clear view of the object.")
    checks.add_argument("--d0_m", type=float, default=0.5, help="Near-plane distance, metres (with --metric_scale).")
    checks.add_argument("--near_frac", type=float, default=0.02, help="Largest share of a view allowed closer than d0.")
    checks.add_argument("--check_width", type=int, default=256, help="Width of the render-check images.")
    checks.add_argument("--depth_tol", type=float, default=0.05, help="Relative depth agreement for 'seen'.")
    return parser


# ---------------------------------------------------------------------------- scene geometry

class Occupancy:
    """Solidity of space from 3D Gaussians, each widened isotropically by ``widen``:

        A(x) = max_i o_i exp(-0.5 |(S_i^2 + widen^2)^-1/2 R_i^T (x - mu_i)|^2)

    Widening keeps each Gaussian's peak opacity o_i, so A(x) > threshold means a Gaussian at least
    that opaque lies within about ``widen`` of x, however thin it is. The maximum, not a sum, keeps
    that distance independent of how densely a wall is splatted. Gaussians are grouped by size and
    only the ``k`` nearest centres of each group within reach are looked at."""

    QUANTILES = (0.5, 0.9, 0.99, 0.999)
    CHUNK = 16384

    def __init__(self, mu, rot, scale, opacity, k=32):
        self.mu, self.rot, self.scale2, self.opacity, self.k = mu, rot, scale**2, opacity, k
        sigma = scale.max(1)
        edges = np.quantile(sigma, self.QUANTILES) if len(sigma) else []
        group = np.searchsorted(edges, sigma)
        self.groups = []
        for g in range(len(edges) + 1):
            idx = np.flatnonzero(group == g)
            if idx.size:
                self.groups.append((idx, cKDTree(mu[idx]), float(sigma[idx].max())))

    def _pairs(self, x, reach):
        """(point, Gaussian) pairs with the Gaussian's centre within reach(group sigma) of the point."""
        for idx, tree, sigma in self.groups:
            k = min(self.k, len(idx))
            for s in range(0, len(x), self.CHUNK):
                dist, nn = tree.query(x[s:s + self.CHUNK], k=k, distance_upper_bound=reach(sigma), workers=-1)
                dist, nn = dist.reshape(len(dist), k), nn.reshape(len(nn), k)
                rows, cols = np.nonzero(np.isfinite(dist))
                yield s + rows, idx[nn[rows, cols]]

    def _mahalanobis2(self, x, g, widen):
        local = np.einsum("nji,nj->ni", self.rot[g], x - self.mu[g])
        return (local**2 / (self.scale2[g] + np.reshape(widen, (-1, 1)) ** 2)).sum(1)

    def alpha(self, x, widen, skip=None):
        """A(x) at points x [M, 3], widened by widen (one value, or one per point)."""
        out = np.zeros(len(x))
        widen = np.broadcast_to(np.asarray(widen, dtype=np.float64), (len(x),))
        reach = float(widen.max(initial=0))
        for p, g in self._pairs(x, lambda sigma: 3 * np.hypot(sigma, reach)):
            value = self.opacity[g] * np.exp(-0.5 * self._mahalanobis2(x[p], g, widen[p]))
            if skip is not None:
                value[skip[g]] = 0
            np.maximum.at(out, p, value)
        return out

    def containing(self, x, within=2.0):
        """Mask of the Gaussians whose `within`-sigma ellipsoid contains one of the points x."""
        mask = np.zeros(len(self.mu), dtype=bool)
        for p, g in self._pairs(x, lambda sigma: within * sigma):
            mask[g[self._mahalanobis2(x[p], g, 0.0) < within**2]] = True
        return mask


def occupancy_field(gaussians, solid, cameras):
    """Occupancy over the Gaussians at least `solid` opaque, minus the ones containing a camera."""
    keep = gaussians.opacity >= solid  # fainter ones can never make a point solid
    field = Occupancy(gaussians.mu[keep], gaussians.rot[keep], gaussians.scale[keep], gaussians.opacity[keep])
    floaters = field.containing(cameras)
    keep[np.flatnonzero(keep)[floaters]] = False
    field = Occupancy(gaussians.mu[keep], gaussians.rot[keep], gaussians.scale[keep], gaussians.opacity[keep])
    print(f"occupancy field: {int(keep.sum())} of {len(keep)} Gaussians (opacity >= {solid}, "
          f"{int(floaters.sum())} floaters around photo cameras dropped)")
    return field, keep


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cdf, q * cdf[-1])])


def load_gaussians(path):
    """Centres, rotation matrices, scales (exp) and opacities (sigmoid) of a 3DGS PLY, in float64."""
    from artifixer_gs.render import read_ply

    ply = read_ply(Path(path))
    quat = ply["quats"].astype(np.float64)  # w, x, y, z, normalised by read_ply
    quat[~np.isfinite(quat).all(1) | (np.linalg.norm(quat, axis=1) < 1e-12)] = (1, 0, 0, 0)  # zero quaternions
    print(f"Gaussians: {len(quat)} from {path}")
    return SimpleNamespace(
        mu=ply["means"].astype(np.float64),
        rot=Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix(),
        scale=ply["scales"].astype(np.float64),
        opacity=ply["opacities"].astype(np.float64),
    )


def object_extent_from_mask(gaussians, mask):
    """Opacity-weighted centre of mass, principal axes and robust half-extents of the masked Gaussians."""
    points, mass = gaussians.mu[mask], gaussians.opacity[mask]
    assert len(points) >= 10, f"only {len(points)} object Gaussians"
    centre = np.average(points, axis=0, weights=mass)
    near = np.linalg.norm(points - centre, axis=1) <= np.percentile(np.linalg.norm(points - centre, axis=1), 99.5)
    points, mass = points[near], mass[near]
    centre = np.average(points, axis=0, weights=mass)
    _, axes = np.linalg.eigh(np.cov((points - centre).T, aweights=mass))
    proj = np.abs((points - centre) @ axes)
    half = np.array([weighted_quantile(proj[:, k], mass, 0.95) for k in range(3)])
    return SimpleNamespace(mask=mask, centre=centre, axes=axes, half=np.maximum(half, 1e-3 * half.max()),
                           radius=float(half.max()), count=int(mask.sum()))


def load_object(gaussians, mask_path):
    """The object of a per-Gaussian mask (object_mask.npy, bool [P] in PLY vertex order). It replaces
    the FlashSplat labels of the original; BEA-GS labels are already vote-filtered, so every masked
    Gaussian counts."""
    mask = np.load(mask_path)
    assert mask.ndim == 1 and len(mask) == len(gaussians.mu), (
        f"{mask_path} has shape {mask.shape}, the PLY {len(gaussians.mu)} Gaussians: another scene?")
    assert mask.dtype == bool or np.isin(mask, (0, 1)).all(), f"{mask_path} must be a boolean mask, got {mask.dtype}"
    print(f"object from {mask_path}: {int(mask.astype(bool).sum())} of {len(mask)} Gaussians")
    return object_extent_from_mask(gaussians, mask.astype(bool))


def ellipsoid_reach(obj_, direction):
    """Distance from the centre to the object's ellipsoid along unit direction(s)."""
    return 1 / np.sqrt((((direction @ obj_.axes) / obj_.half) ** 2).sum(-1))


def look_at(pos, target, up):
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(right, forward), -forward])  # OpenGL: x right, y up, z back


def orbit_pose(c2w, anchor, yaw, pitch, up, max_elev=None):
    """OpenGL camera ``c2w`` rigidly rotated about ``anchor``: by ``yaw`` about the vertical ``up``,
    then by ``pitch`` about the horizontal axis across the anchor->camera direction (positive raises
    the camera; the elevation stays within +-``max_elev``). The anchor keeps its image position, and a
    level camera stays level."""
    offset = c2w[:3, 3] - anchor
    rot = Rotation.from_rotvec(yaw * up).as_matrix()
    offset1 = rot @ offset
    across = np.cross(offset1, up)
    if np.linalg.norm(across) > 1e-9 * np.linalg.norm(offset1) and pitch != 0:
        elev = np.arcsin(np.clip(offset1 @ up / np.linalg.norm(offset1), -1, 1))
        if max_elev is not None:
            pitch = float(np.clip(elev + pitch, -max_elev, max_elev) - elev)
        rot = Rotation.from_rotvec(pitch * across / np.linalg.norm(across)).as_matrix() @ rot
    out = np.eye(4)
    out[:3, :3] = rot @ c2w[:3, :3]
    out[:3, 3] = anchor + rot @ offset
    return out


# ---------------------------------------------------------------------------- photos and base path

def colmap_photo_transforms(colmap_dir, images="images", normalize_near_shared_cameras=False, image_downscale_factor=1):
    """The photo transforms.json that artifixer_gs.prepare writes for this COLMAP scene: one shared
    OPENCV calibration scaled to the size of the files in ``images`` (then by 1/image_downscale_factor,
    as prepare --image_downscale_factor), OpenGL c2w, COLMAP name order."""
    from artifixer_gs.prepare import (
        ColmapScene,
        normalize_near_shared_camera_scene,
        read_colmap_scene,
        require_unique_basenames,
        scale_colmap_scene_to_images,
        scale_image_observations,
        scaled_camera_params,
        shared_camera_intrinsics,
        source_frame,
    )
    from data_processing.camera_trajectories import transforms_json

    colmap_dir = Path(colmap_dir).expanduser().resolve()
    image_dir, sparse_dir = colmap_dir / images, colmap_dir / "sparse" / "0"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing COLMAP images directory: {image_dir}")
    for name in ("cameras.bin", "images.bin"):
        if not (sparse_dir / name).is_file():
            raise FileNotFoundError(f"Missing COLMAP sparse file: {sparse_dir / name}")
    scene = read_colmap_scene(sparse_dir)
    require_unique_basenames(scene.images)
    scene = scale_colmap_scene_to_images(image_dir, scene)
    if normalize_near_shared_cameras:
        scene = normalize_near_shared_camera_scene(scene)
    assert image_downscale_factor >= 1, "--image_downscale_factor must be >= 1"
    if image_downscale_factor != 1:  # as artifixer_gs.prepare.prepare_scene
        factor = 1.0 / image_downscale_factor
        scene = ColmapScene(
            cameras=[
                camera._replace(
                    width=max(1, int(round(camera.width * factor))),
                    height=max(1, int(round(camera.height * factor))),
                    params=scaled_camera_params(camera, factor, factor),
                )
                for camera in scene.cameras
            ],
            images=[scale_image_observations(image, factor, factor) for image in scene.images],
        )
    return transforms_json(shared_camera_intrinsics(scene), [source_frame(image) for image in scene.images])


def check_world_frame(photos, colmap_dir):
    """The photos of a transforms.json are the COLMAP scene's (the PLY's world frame): every photo name
    is in it and the camera positions agree, as the original's render-check setup asserted."""
    from artifixer_gs.prepare import read_colmap_scene, source_frame

    frames = [source_frame(image) for image in read_colmap_scene(Path(colmap_dir).expanduser() / "sparse" / "0").images]
    index = {Path(f["file_path"]).name: f for f in frames}
    missing = [name for name in photos.names if name not in index]
    if missing:
        raise SystemExit(f"{len(missing)} photos of transforms.json are not in {colmap_dir}, e.g. {missing[:3]}")
    poses = np.array([index[name]["transform_matrix"] for name in photos.names], dtype=np.float64)
    error = np.abs(poses[:, :3, 3] - photos.pos).max()
    if not error < 1e-3 * photos.spacing + 1e-5 * np.abs(photos.pos).max():
        raise SystemExit(f"COLMAP poses differ from transforms.json by {error:.3g} scene units: another world frame?")
    return float(error)


def load_photos(transforms):
    """Photo cameras of a transforms.json (path or already-loaded mapping), in file name order."""
    if not isinstance(transforms, dict):
        transforms = json.loads(Path(transforms).read_text())
    assert "applied_transform" not in transforms, "expected a prepared transforms.json without applied_transform"
    frames = sorted(transforms["frames"], key=lambda f: f["file_path"])
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    n = len(c2w)
    assert n >= 2, f"need at least 2 photos, got {n}"
    pos = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    spacing = float(np.median(steps))
    assert spacing > 0, "photo cameras do not move"
    world_up = c2w[:, :3, 1].mean(0)
    return SimpleNamespace(transforms=transforms, names=[Path(f["file_path"]).name for f in frames], c2w=c2w,
                           pos=pos, n=n, steps=steps, spacing=spacing, world_up=world_up / np.linalg.norm(world_up))


def smooth_base(photos, sigma, jump):
    """Photo path smoothed within runs without a jump, and the across-travel direction per photo."""
    n, spacing, pos = photos.n, photos.spacing, photos.pos
    segments = np.split(np.arange(n), np.where(photos.steps > jump * spacing)[0] + 1)
    base = np.zeros((n, 4, 4))
    across = np.zeros((n, 3))
    for seg in segments:
        rot = Rotation.from_matrix(photos.c2w[seg, :3, :3])
        for i in seg:
            w = np.exp(-0.5 * ((seg - i) / sigma) ** 2)
            w /= w.sum()
            base[i] = np.eye(4)
            base[i, :3, :3] = rot.mean(weights=w).as_matrix()
            base[i, :3, 3] = (w[:, None] * pos[seg]).sum(0)
        travel_dir = np.gradient(base[seg, :3, 3], axis=0) if len(seg) > 1 else np.zeros((1, 3))
        for i, t in zip(seg, travel_dir):
            a = np.cross(t, base[i, :3, 1])  # OpenGL camera axes: x right, y up, z back
            across[i] = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-9 * spacing else base[i, :3, 0]
    return base, across, segments


def clearance_for(args, spacing, obj_):
    if args.clearance is not None:
        return args.clearance
    if args.metric_scale:
        return args.clearance_m / args.metric_scale
    return 0.15 * obj_.radius if obj_ else spacing


def free_segments(field, starts, ends, clearance, solid, skip_start=None):
    """Whether each straight segment starts[m] -> ends[m] keeps the clearance (one value, or one per
    segment) from every surface (samples at most a clearance apart, up to 64; the first clearance of a
    segment whose start is flagged in skip_start is not checked, for starts inside a surface)."""
    length = np.linalg.norm(ends - starts, axis=1)
    clearance = np.broadcast_to(np.asarray(clearance, dtype=np.float64), (len(starts),))
    k = int(np.clip(np.ceil((length / clearance).max(initial=0)) + 1, 2, 64))
    t = np.arange(1, k + 1) / k
    samples = starts[:, None] + t[None, :, None] * (ends - starts)[:, None]
    blocked = (field.alpha(samples.reshape(-1, 3), np.repeat(clearance, k)) > solid).reshape(-1, k)
    if skip_start is not None:
        blocked &= ~(skip_start[:, None] & (t[None] * length[:, None] < clearance[:, None]))
    return ~blocked.any(1)


# ---------------------------------------------------------------------------- render checks

class ViewChecker:
    """Render-based view checks (VidSplat Eq. 5) against the photos.

    ``render_fn(c2w_gl [M, 4, 4]) -> (depth [M, h, w], opacity [M, h, w])`` renders camera z-depth
    and opacity (torch tensors) at the check camera ``intr`` = (fx, fy, cx, cy, w, h); pixel centres
    sit at index + 0.5. Every photo is rendered once, so "seen" compares like with like."""

    def __init__(self, render_fn, photo_c2w, intr, solid=0.5, depth_tol=0.05, device="cpu", chunk=64, batch=32):
        import torch

        self.torch, self.render_fn, self.solid, self.tol, self.chunk, self.batch = torch, render_fn, solid, depth_tol, chunk, batch
        self.fx, self.fy, self.cx, self.cy, self.w, self.h = intr
        self.device = device
        depth, opacity = self.render(photo_c2w)
        valid = opacity > solid
        self.photo_depth, self.photo_valid = depth, valid
        big = torch.finfo(depth.dtype).max
        # 3x3 neighbourhood depth range of every photo (pixels without geometry excluded).
        pad = lambda x: torch.nn.functional.max_pool2d(x[:, None], 3, stride=1, padding=1)[:, 0]  # noqa: E731
        self.photo_dmax = pad(torch.where(valid, depth, torch.full_like(depth, -big)))
        self.photo_dmin = -pad(torch.where(valid, -depth, torch.full_like(depth, -big)))
        w2c = np.linalg.inv(np.asarray(photo_c2w) @ FLIP)
        self.w2c = torch.as_tensor(w2c, dtype=torch.float32, device=device)
        v, u = torch.meshgrid(torch.arange(self.h, device=device, dtype=torch.float32),
                              torch.arange(self.w, device=device, dtype=torch.float32), indexing="ij")
        self.rays = torch.stack([(u + 0.5 - self.cx) / self.fx, (v + 0.5 - self.cy) / self.fy, torch.ones_like(u)], -1)

    def render(self, c2w):
        depths, opacities = [], []
        for s in range(0, len(c2w), self.batch):
            d, o = self.render_fn(np.asarray(c2w[s:s + self.batch]))
            depths.append(d.to(self.device).float())
            opacities.append(o.to(self.device).float())
        return self.torch.cat(depths), self.torch.cat(opacities)

    def near_depth(self, c2w=None, index=None, q=0.02):
        """Robust nearest surface (q-quantile of depth over pixels with geometry) of poses or photos."""
        if index is not None:
            depth, valid = self.photo_depth[index][None], self.photo_valid[index][None]
        else:
            depth, opacity = self.render(c2w)
            valid = opacity > self.solid
        out = []
        for d, m in zip(depth, valid):
            out.append(float(self.torch.quantile(d[m], q)) if m.sum() >= 10 else np.inf)
        return np.array(out)

    def anchor(self, index, c2w, patch=0.3):
        """Where the photo's central ray meets the surface: the median depth of the central patch
        (patch x the image size each way, pixels with geometry) along the ray through the principal point."""
        h0, h1 = int(self.h * (0.5 - patch / 2)), int(np.ceil(self.h * (0.5 + patch / 2)))
        w0, w1 = int(self.w * (0.5 - patch / 2)), int(np.ceil(self.w * (0.5 + patch / 2)))
        d = self.photo_depth[index, h0:h1, w0:w1][self.photo_valid[index, h0:h1, w0:w1]]
        if d.numel() < 0.2 * (h1 - h0) * (w1 - w0):
            return None
        return c2w[:3, 3] - float(d.median()) * c2w[:3, 2]  # OpenGL: the camera looks along -z

    def measure(self, c2w, d0, unseen=True):
        """Per pose: near_frac (share of pixels with geometry closer than d0), empty (share without
        geometry) and, with unseen=True, unseen (share whose surface no photo observes, opened 3x3)."""
        torch = self.torch
        depth, opacity = self.render(c2w)
        valid = opacity > self.solid
        d0 = torch.as_tensor(np.broadcast_to(np.asarray(d0, dtype=np.float32), (len(c2w),)).copy(), device=self.device)
        out = {"near_frac": (valid & (depth < d0[:, None, None])).float().mean((1, 2)).cpu().numpy(),
               "empty": (~valid).float().mean((1, 2)).cpu().numpy()}
        if not unseen:
            return out
        c2w_cv = torch.as_tensor(np.asarray(c2w) @ FLIP, dtype=torch.float32, device=self.device)
        fractions = []
        for m in range(len(c2w)):
            points = (self.rays * depth[m][..., None]).reshape(-1, 3) @ c2w_cv[m, :3, :3].T + c2w_cv[m, :3, 3]
            seen = torch.zeros(len(points), dtype=torch.bool, device=self.device)
            for s in range(0, len(self.w2c), self.chunk):
                w2c = self.w2c[s:s + self.chunk]
                cam = torch.einsum("jab,pb->jpa", w2c[:, :3, :3], points) + w2c[:, None, :3, 3]
                z = cam[..., 2]
                safe = z.clamp_min(1e-6)
                u = torch.floor(cam[..., 0] / safe * self.fx + self.cx).long()
                v = torch.floor(cam[..., 1] / safe * self.fy + self.cy).long()
                inside = (z > 1e-6) & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
                j = torch.arange(s, s + len(w2c), device=self.device)[:, None].expand_as(u)
                uc, vc = u.clamp(0, self.w - 1), v.clamp(0, self.h - 1)
                lo, hi = self.photo_dmin[j, vc, uc], self.photo_dmax[j, vc, uc]
                seen |= (inside & (z >= lo - self.tol * z) & (z <= hi + self.tol * z)).any(0)
            miss = (valid[m] & ~seen.reshape(self.h, self.w)).float()[None, None]
            miss = -torch.nn.functional.max_pool2d(-miss, 3, stride=1, padding=1)  # erode
            miss = torch.nn.functional.max_pool2d(miss, 3, stride=1, padding=1)  # dilate
            fractions.append(float(miss.mean()))
        out["unseen"] = np.array(fractions)
        return out


def check_intrinsics(transforms, check_width):
    """(fx, fy, cx, cy, w, h) of the photos' shared camera scaled to check_width pixels wide."""
    w_full, h_full = float(transforms["w"]), float(transforms["h"])
    fx, fy, cx, cy = (float(transforms[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    w = int(check_width)
    h = max(1, int(round(h_full * w / w_full)))
    sx, sy = w / w_full, h / h_full
    # float32 as the 3DGUT check camera's params
    fx, fy, cx, cy = (float(np.float32(v)) for v in (fx * sx, fy * sy, cx * sx, cy * sy))
    return (fx, fy, cx, cy, w, h), (int(w_full), int(h_full))


def gsplat_renderer(ply, photos, check_width):
    """(render_fn, intr, device) rasterizing ``ply`` with gsplat (expected depth + alpha) at
    check_width pixels wide, with the photos' shared pinhole intrinsics (distortion ignored)."""
    import torch
    from gsplat import rasterization

    from artifixer_gs.render import default_eps2d

    eps2d = default_eps2d()
    intr, (w_full, h_full) = check_intrinsics(photos.transforms, check_width)
    fx, fy, cx, cy, w, h = intr
    device = torch.device("cuda")
    gaussians = load_ply_tensors(ply, device)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    print(f"render checks: {w}x{h} pixels (from {w_full}x{h_full}), {len(photos.names)} photos, gsplat")

    def render_fn(c2w_gl):
        viewmats = torch.as_tensor(np.linalg.inv(np.asarray(c2w_gl, dtype=np.float64) @ FLIP), dtype=torch.float32,
                                   device=device)
        with torch.no_grad():
            out, alphas, _ = rasterization(
                **gaussians,
                viewmats=viewmats,
                Ks=K.expand(len(viewmats), 3, 3).contiguous(),
                width=w,
                height=h,
                render_mode="RGB+ED",
                rasterize_mode="classic",
                eps2d=eps2d,
                backgrounds=torch.zeros(len(viewmats), 3, device=device),
            )
        return out[..., 3], alphas[..., 0]

    return render_fn, intr, str(device)


def load_ply_tensors(ply, device):
    """gsplat inputs of a PLY: the geometry, and black colours (only depth and alpha are used)."""
    import torch

    from artifixer_gs.render import read_ply

    g = read_ply(Path(ply))
    quats = g["quats"]
    quats[~np.isfinite(quats).all(1)] = (1, 0, 0, 0)
    tensors = {key: torch.from_numpy(np.ascontiguousarray(g[name])).to(device)
               for key, name in (("means", "means"), ("scales", "scales"), ("opacities", "opacities"))}
    tensors["quats"] = torch.from_numpy(quats).to(device)
    tensors["colors"] = torch.zeros(len(quats), 3, device=device)
    return tensors


# ---------------------------------------------------------------------------- path modes

def offset_path(args, photos, gaussians, obj_, clearance, checker):
    """travel / object mode: the smoothed photo path with per-frame offsets, pulled back where needed."""
    n, spacing = photos.n, photos.spacing
    base, across, segments = smooth_base(photos, args.sigma, args.jump)
    world_up = photos.world_up

    def camera(i, s):
        """Path camera i at offset scale s (0 = base camera)."""
        m = base[i].copy()
        if obj_ is None:
            side = args.side * spacing * np.sin(2 * np.pi * args.periods * i / (n - 1) + 2 * np.pi * args.phase_offset)
            m[:3, 3] += s * (side * across[i] + args.up * spacing * m[:3, 1])
            return m
        phase = 2 * np.pi * args.periods * i / (n - 1) + 2 * np.pi * args.phase_offset
        R, w0 = obj_.radius, base[i, :3, 3] - obj_.centre
        height = w0 @ world_up
        horiz = w0 - height * world_up
        angle = s * args.orbit * R * np.sin(phase) / max(np.linalg.norm(horiz), 0.25 * R)
        horiz = np.cos(angle) * horiz + np.sin(angle) * np.cross(world_up, horiz)
        w = horiz + (height + s * args.rise * R) * world_up
        dist0, dist = np.linalg.norm(w0), np.linalg.norm(w)
        gap0 = dist0 - ellipsoid_reach(obj_, w0 / dist0)  # the base camera's distance from the object
        reach = ellipsoid_reach(obj_, w / dist)
        dist_new = max(dist + s * args.radial * R * np.cos(phase), reach + min(gap0, args.standoff * R))
        m[:3, 3] = obj_.centre + w / dist * dist_new
        aim = Rotation.from_matrix(look_at(m[:3, 3], obj_.centre, world_up))
        photo = Rotation.from_matrix(base[i, :3, :3])
        m[:3, :3] = (Rotation.from_rotvec(args.look_at * (aim * photo.inv()).as_rotvec()) * photo).as_matrix()
        return m

    info = {"offset_scale": None, "segments": len(segments), "pieces": [[int(seg[0]), int(seg[-1])] for seg in segments]}
    scale = np.ones(n)
    if gaussians is None:
        return np.array([camera(i, 1.0) for i in range(n)]), scale, info, None

    field, keep = occupancy_field(gaussians, args.solid, np.concatenate([photos.pos, base[:, :3, 3]]))
    skip_object = obj_.mask[keep] if obj_ else None
    base_blocked = field.alpha(base[:, :3, 3], clearance) > args.solid
    # A base camera already closer than the clearance to a surface (the photographer stood there) may
    # move as long as it comes no closer than it already is: its own clearance is the largest of
    # clearance/2, /4, /8 it keeps. Only base cameras that keep none of them skip the start of (b).
    own = np.full(n, clearance)
    for c in clearance / np.array([2.0, 4.0, 8.0]):
        own[field.alpha(base[:, :3, 3], own) > args.solid] = c
    base_inside = field.alpha(base[:, :3, 3], own) > args.solid

    def sees_object(p):
        """Whether the segment from each camera position to the object is free of other surfaces."""
        v = obj_.centre - p
        dist = np.linalg.norm(v, axis=1)
        v /= dist[:, None]
        free = dist - ellipsoid_reach(obj_, -v) - clearance  # up to just before the object
        k = int(np.clip(np.ceil(free.max() / (0.5 * clearance)), 2, 256))
        along = clearance + np.arange(k)[None] / (k - 1) * np.maximum(free - clearance, 0)[:, None]
        samples = p[:, None] + along[..., None] * v[:, None]
        occluded = field.alpha(samples.reshape(-1, 3), 0.25 * clearance, skip=skip_object) > args.solid
        return ~(occluded.reshape(-1, k) & (along < free[:, None])).any(1)

    # (c) only where the photo saw the object: a path camera must not lose a view the photo had.
    check_view = obj_ is not None and not args.no_visibility
    base_sees = sees_object(base[:, :3, 3]) if check_view else np.zeros(n, dtype=bool)
    if checker is not None:
        d0 = args.d0_m / args.metric_scale if args.metric_scale else clearance
        d0_base = np.minimum(d0, 0.8 * checker.near_depth(base))

    def feasible(cams, owner):
        """Checks (a)-(c) for cameras cams [M, 4, 4] offset from base cameras owner [M]."""
        p = cams[:, :3, 3]
        ok = field.alpha(p, own[owner]) <= args.solid  # (a)
        ok &= free_segments(field, base[owner, :3, 3], p, own[owner], args.solid, base_inside[owner])  # (b)
        if check_view and base_sees[owner].any():
            ok[base_sees[owner]] &= sees_object(p[base_sees[owner]])  # (c)
        if checker is not None and ok.any():
            near = checker.measure(cams[ok], d0_base[owner[ok]], unseen=False)["near_frac"]
            ok[np.flatnonzero(ok)] = near <= args.near_frac  # (c) rendered: no near surface blocks the view
        return ok

    levels = np.linspace(1, 0, max(args.levels, 2))
    grid = np.array([[camera(i, s) for s in levels] for i in range(n)])
    passed = feasible(grid.reshape(-1, 4, 4), np.repeat(np.arange(n), len(levels))).reshape(n, len(levels))
    passed[:, -1] = True  # scale 0: the base camera
    # Largest level whose own check and every smaller level's pass, so any scale below it is safe too.
    prefix = np.flip(np.logical_and.accumulate(np.flip(passed, 1), axis=1), 1)
    allowed = levels[prefix.argmax(1)]
    # Smooth without exceeding any frame's allowance: erode by the Gaussian's reach, then blur.
    radius = int(2.0 * 2 * args.sigma + 0.5)
    for seg in segments:
        eroded = minimum_filter1d(allowed[seg], 2 * radius + 1, mode="nearest")
        scale[seg] = np.minimum(gaussian_filter1d(eroded, 2 * args.sigma, mode="nearest", truncate=2.0), allowed[seg])
    final = np.array([camera(i, s) for i, s in enumerate(scale)])
    bad = np.flatnonzero(~feasible(final, np.arange(n)) & (scale > 0))
    for i in bad:  # between two tested levels: drop to the level below, which passed
        scale[i] = levels[levels <= scale[i] + 1e-9][0]
    path = np.array([camera(i, s) for i, s in enumerate(scale)])
    print(f"offset scale: mean {scale.mean():.2f}, {int((scale < 1).sum())}/{n} frames pulled back, "
          f"{int((scale == 0).sum())} at their base camera; {len(bad)} snapped to a tested level; "
          f"{int(base_blocked.sum())} base cameras closer than the clearance to a surface (kept at least as far as "
          f"they are; {int(base_inside.sum())} inside one)"
          + (f", {int((~base_sees).sum())} whose view of the object is blocked (no view check there)" if check_view else ""))
    if base_inside.mean() > 0.2:
        print("WARNING: many base cameras are inside a surface: raise --solid")
    info.update(offset_scale=np.round(scale, 4).tolist(), base_cameras_touching=int(base_blocked.sum()),
                base_cameras_inside=int(base_inside.sum()),
                render_check=checker is not None)
    collapsed = float(np.mean(scale < 0.25))
    info["collapsed_share"] = collapsed
    if obj_ is not None and collapsed > args.max_collapsed and not args.allow_fallback:
        raise TrajectoryError(
            f"object path collapsed: {collapsed:.0%} of the frames keep under a quarter of their offset "
            f"(limit {args.max_collapsed:.0%}); walls or occluders leave no room for this orbit. Lower --orbit/--radial/"
            f"--rise (TRAJ_ORBIT/TRAJ_RADIAL/TRAJ_RISE) or --clearance_m (TRAJ_CLEARANCE_M), or pass --allow_fallback "
            f"(TRAJ_ALLOW_FALLBACK=1) to accept it", path, info)
    return path, scale, info, (keep, field)


def vidsplat_path(args, photos, gaussians, clearance, checker):
    """VidSplat-style orbits about the surface points the seed photos look at (see the module docstring)."""
    n, L, up = photos.n, args.clip_len, photos.world_up
    if L < 4 or L % 4:
        raise SystemExit("--clip_len must be a positive multiple of 4")
    extents = sorted({np.radians(float(x)) for x in args.orbit_deg.split(",") if x.strip()}, reverse=True)
    budget = args.budget or min(n, args.max_frames or n)
    n_clips = max(1, (budget - 1) // L)
    min_clips = args.min_clips if args.min_clips is not None else max(1, n_clips // 2)
    keyframes = np.arange(3, L, 4)  # 0-based: the last frame of every 4-frame group
    diag = 1 / np.sqrt(2)
    directions = {"left": (1, 0), "right": (-1, 0), "up": (0, 1), "down": (0, -1),
                  "up-left": (diag, diag), "up-right": (-diag, diag), "down-left": (diag, -diag), "down-right": (-diag, -diag)}
    t = np.arange(1, L + 1) / L

    field, keep = occupancy_field(gaussians, args.solid, photos.pos)
    photo_blocked = field.alpha(photos.pos, clearance) > args.solid
    d0_base = args.d0_m / args.metric_scale if args.metric_scale else None
    # Seeds every n/m photos; --phase_offset moves them by that share of one spacing, so loop round k
    # (offset (k-1)/rounds) tries the photos between the earlier rounds' seeds.
    m = min(n, max(args.seeds, 2 * n_clips))
    seeds = np.unique(np.round((np.arange(m) + args.phase_offset) * n / m).astype(int) % n)
    counts = dict.fromkeys(REJECT_REASONS, 0)
    per_seed = {}
    for j in seeds:
        anchor = checker.anchor(j, photos.c2w[j])
        if anchor is None:
            counts["no_anchor"] += 1
            continue
        radius = float(np.linalg.norm(photos.pos[j] - anchor))
        d0 = min(d0_base if d0_base is not None else 0.25 * radius, 0.8 * checker.near_depth(index=j)[0])
        if radius < 2 * d0 or radius < 2 * clearance:
            counts["too_close"] += 1
            continue
        seed_stats = checker.measure(photos.c2w[j][None], d0)
        candidates = []
        for extent in extents:
            for name, (sy, sp) in directions.items():
                poses = np.array([orbit_pose(photos.c2w[j], anchor, sy * extent * tk, sp * extent * tk, up,
                                             np.radians(args.max_elev)) for tk in t])
                starts = np.concatenate([photos.pos[j][None], poses[:-1, :3, 3]])
                ok = (field.alpha(poses[:, :3, 3], clearance) <= args.solid).all()
                ok = ok and free_segments(field, starts, poses[:, :3, 3], clearance, args.solid,
                                          np.r_[photo_blocked[j], np.zeros(L - 1, dtype=bool)]).all()
                if not ok:
                    counts["wall"] += 1
                    continue
                candidates.append(SimpleNamespace(seed=int(j), direction=name, extent=float(np.degrees(extent)),
                                                  poses=poses, anchor=anchor, radius=radius, d0=d0))
        if not candidates:
            continue
        stats = checker.measure(np.concatenate([c.poses[keyframes] for c in candidates]), d0)
        k = len(keyframes)
        valid = []
        for c_i, c in enumerate(candidates):
            s = {key: value[c_i * k:(c_i + 1) * k] for key, value in stats.items()}
            if (s["near_frac"] > args.near_frac).any():
                reason = "near_plane"
            elif (s["unseen"] > args.s_high).any():
                reason = "too_unseen"
            elif (s["empty"] > seed_stats["empty"][0] + args.s_high).any():
                reason = "empty"
            elif s["unseen"][-1] - seed_stats["unseen"][0] < args.s_low:
                reason = "nothing_new"
            else:
                c.stats = {key: np.round(value, 4).tolist() for key, value in s.items()}
                c.gain = float(s["unseen"][-1] - seed_stats["unseen"][0])
                valid.append(c)
                continue
            counts[reason] += 1
        if valid:
            per_seed[int(j)] = sorted(valid, key=lambda c: (c.gain, c.extent), reverse=True)

    # One orbit per stretch of the capture (n_clips equal stretches): the seed whose orbit reveals the
    # most. Stretches without a valid orbit leave their place to the best remaining seeds anywhere, then
    # to a second orbit at least 90 degrees from a chosen seed's first.
    order = sorted(per_seed, key=lambda j: (-per_seed[j][0].gain, j))
    stretch = {j: min(j * n_clips // n, n_clips - 1) for j in order}
    best = {}
    for j in order:
        best.setdefault(stretch[j], j)
    picked = sorted(best.values()) + [j for j in order if j not in best.values()]
    chosen = [per_seed[j][0] for j in picked[:n_clips]]
    if len(chosen) < n_clips:
        vec = lambda c: np.array(directions[c.direction])  # noqa: E731
        for j in order:
            if len(chosen) >= n_clips:
                break
            first = next((c for c in chosen if c.seed == j), None)
            second = next((c for c in per_seed.get(j, []) if first is not None and vec(c) @ vec(first) <= 1e-9), None)
            if second is not None:
                chosen.append(second)
    chosen.sort(key=lambda c: (c.seed, -c.gain))
    info = {"clip_len": L, "clips_wanted": n_clips, "min_clips": min_clips, "seeds_tried": len(seeds),
            "seed_indices": seeds.tolist(), "valid_seeds": sorted(per_seed),
            "rejected": counts, "orbit_deg": [float(np.degrees(e)) for e in extents],
            "clips": [{"seed": photos.names[c.seed], "seed_index": c.seed, "direction": c.direction,
                       "extent_deg": c.extent, "anchor": c.anchor.tolist(), "anchor_distance": c.radius, "d0": c.d0,
                       "keyframes": c.stats} for c in chosen]}
    print(f"vidsplat: {len(chosen)}/{n_clips} orbits of {L} frames from {len(seeds)} seeds; rejected candidates: "
          + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    path = np.concatenate([photos.c2w[chosen[0].seed][None]] + [c.poses for c in chosen]) if chosen else photos.c2w[:1]
    info["anchors"] = [c.anchor.tolist() for c in chosen]
    if len(chosen) < min_clips:
        raise TrajectoryError(
            f"only {len(chosen)} valid VidSplat orbits, {min_clips} needed (of {n_clips}); rejected candidates: "
            + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
            + ". Loosen --s_high/--s_low (TRAJ_S_HIGH/TRAJ_S_LOW), --d0_m (TRAJ_D0_M) or --orbit_deg (TRAJ_ORBIT_DEG), "
            "or lower --min_clips", path, info)
    info["pieces"] = pieces_of(len(path), [1 + k * L for k in range(1, len(chosen))])
    clip_id = np.r_[-1, np.repeat(np.arange(len(chosen)), L)]
    return path, clip_id, info, (keep, field)


# ---------------------------------------------------------------------------- output

def pieces_of(n, starts):
    """[first, last] frame of each continuous piece of an n-frame path whose pieces start at starts."""
    starts = sorted({int(a) for a in starts if 0 < a < n} | {0})
    return [[a, b - 1] for a, b in zip(starts, starts[1:] + [n])]


def jump_lengths(path, pieces, spacing):
    """Camera move across each jump between consecutive pieces, in photo spacings."""
    pos = path[:, :3, 3]
    return [float(np.linalg.norm(pos[b[0]] - pos[a[1]]) / spacing) for a, b in zip(pieces, pieces[1:])]


def write_outputs(args, photos, path, info):
    keys = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")
    out = {k: photos.transforms[k] for k in keys if k in photos.transforms}
    out["frames"] = [{"transform_matrix": m.tolist()} for m in path]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if "failed" not in info:
        args.output.write_text(json.dumps(out, indent=2) + "\n")
    info_path = args.output.with_name(args.output.stem + "_info.json")
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    return info_path


PIECE_COLOURS = ("tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")


def plot_top_view(args, photos, path, colour, colour_label, gaussians, keep, obj_, clearance, pieces, anchors=None, title=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no top-view plot")
        return
    up = photos.world_up
    cams = np.concatenate([photos.pos, path[:, :3, 3]])
    spread = cams - cams.mean(0)
    spread -= np.outer(spread @ up, up)
    h1 = np.linalg.eigh(spread.T @ spread)[1][:, -1]  # main horizontal direction of the cameras
    h2 = np.cross(up, h1)
    flat = lambda x: np.stack([x @ h1, x @ h2], -1)  # noqa: E731
    shown = flat(np.concatenate([cams, np.asarray(anchors).reshape(-1, 3)]) if anchors else cams)
    lo, hi = shown.min(0), shown.max(0)
    lo, hi = lo - 0.3 * (hi - lo).max(), hi + 0.3 * (hi - lo).max()
    fig, ax = plt.subplots(figsize=(10, 10))
    if gaussians is not None:
        band = (cams @ up).min() - clearance, (cams @ up).max() + clearance
        height = gaussians.mu @ up
        solid = keep & (height > band[0]) & (height < band[1])
        if obj_:
            solid &= ~obj_.mask
        xy = flat(gaussians.mu[solid])
        if len(xy):
            ax.hist2d(xy[:, 0], xy[:, 1], bins=400, range=[[lo[0], hi[0]], [lo[1], hi[1]]], cmap="Greys", cmin=1,
                      norm=matplotlib.colors.LogNorm())
    if obj_:
        xy = flat(gaussians.mu[obj_.mask])
        ax.scatter(xy[::max(1, len(xy) // 20000), 0], xy[::max(1, len(xy) // 20000), 1], s=0.2, c="tab:red", alpha=0.3)
        ax.plot(*flat(obj_.centre[None])[0], "r*", ms=15, label="object centre of mass")
    if anchors:
        a = flat(np.asarray(anchors))
        ax.plot(a[:, 0], a[:, 1], "rx", ms=6, label="orbit anchors")
    ax.plot(*flat(photos.pos).T, ".-", c="tab:blue", ms=3, lw=0.5, label="photos")
    p = flat(path[:, :3, 3])
    # One colour per continuous piece, numbered in video order; a dotted black line is a jump between
    # pieces, i.e. a cut in the one ArtiFixer video.
    for k, (a, b) in enumerate(pieces):
        c = PIECE_COLOURS[k % len(PIECE_COLOURS)]
        ax.plot(*p[a:b + 1].T, "-", c=c, lw=2.5, label="path, one colour per continuous piece" if k == 0 else None)
        if k:
            ax.plot(*p[[pieces[k - 1][1], a]].T, ":", c="k", lw=1.0,
                    label="jump between pieces (a cut in the ArtiFixer video)" if k == 1 else None)
        if len(pieces) > 1:
            ax.annotate(str(k + 1), p[a], xytext=(4, 4), textcoords="offset points", color=c, fontsize=9, fontweight="bold")
    sc = ax.scatter(*p.T, c=colour, cmap="viridis", s=6, zorder=3, label=f"path frames (colour: {colour_label})")
    look = flat(-path[:, :3, 2]) * 0.03 * (hi - lo).max()
    step = max(1, len(path) // 60)
    ax.quiver(p[::step, 0], p[::step, 1], look[::step, 0], look[::step, 1], color="0.3", angles="xy",
              scale_units="xy", scale=1, width=0.002)
    fig.colorbar(sc, ax=ax, fraction=0.03, label=colour_label)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(title or f"top view: Gaussians at camera height (grey), clearance {clearance:.3f}; "
                          f"{len(pieces)} path pieces, {len(pieces) - 1} jump{'s' * (len(pieces) != 2)}")
    fig.savefig(args.output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"top view: {args.output.with_suffix('.png')}")


def main(argv=None, renderer=None):
    """renderer: (render_fn, intr[, device]) for the render checks instead of the gsplat renderer (tests)."""
    args = build_parser().parse_args(argv)
    mode = args.mode or ("object" if args.object_mask else "travel")
    if mode == "object" and args.object_mask is None:
        raise SystemExit("--mode object needs --object_mask")
    if mode in ("object", "vidsplat") and args.ply is None:
        raise SystemExit(f"--mode {mode} needs --ply")
    need_render = mode == "vidsplat" or (mode == "object" and args.render_check)

    if args.transforms is None and args.colmap_dir is None:
        raise SystemExit("give the photos: --transforms and/or --colmap_dir")
    if args.transforms is not None:
        photos = load_photos(args.transforms)
        if args.colmap_dir is not None:
            error = check_world_frame(photos, args.colmap_dir)
            print(f"photos: {photos.n} of transforms.json match {args.colmap_dir} (max position error {error:.3g})")
    else:
        photos = load_photos(colmap_photo_transforms(args.colmap_dir, args.images, args.normalize_near_shared_cameras,
                                                     args.image_downscale_factor))
        print(f"photos: {photos.n} from {args.colmap_dir} ({args.images}, {photos.transforms['w']}x{photos.transforms['h']})")
    gaussians = load_gaussians(args.ply) if args.ply else None
    obj_ = load_object(gaussians, args.object_mask) if mode == "object" else None
    clearance = clearance_for(args, photos.spacing, obj_)
    if obj_:
        print(f"object: {obj_.count} Gaussians, centre of mass {np.round(obj_.centre, 3).tolist()}, "
              f"half-extents {np.round(np.sort(obj_.half)[::-1], 3).tolist()}, radius R {obj_.radius:.3f} (scene units)")
    if gaussians is not None:
        print(f"clearance {clearance:.4f} scene units")
    checker = None
    if need_render:
        render_fn, intr, *device = renderer or gsplat_renderer(args.ply, photos, args.check_width)
        observers = photos.c2w  # the photos come first: anchor() and near_depth(index=) index them
        if args.seen_transforms is not None:
            seen = json.loads(args.seen_transforms.read_text())["frames"]
            observers = np.concatenate([photos.c2w, np.array([f["transform_matrix"] for f in seen], dtype=np.float64)])
            print(f"unseen test: {len(photos.c2w)} photos + {len(seen)} earlier generated views")
        checker = ViewChecker(render_fn, observers, intr, args.solid, args.depth_tol, device[0] if device else "cpu")

    info = {"mode": mode, "median_spacing": photos.spacing, "clearance": clearance,
            "collision_checks": gaussians is not None}
    if obj_:
        info["object"] = {"mask": str(args.object_mask), "gaussians": obj_.count, "centre": obj_.centre.tolist(),
                          "axes": obj_.axes.T.tolist(), "half_extents": obj_.half.tolist(), "radius": obj_.radius}
    failure, keep = None, None
    try:
        if mode == "vidsplat":
            path, colour, mode_info, extra = vidsplat_path(args, photos, gaussians, clearance, checker)
            colour_label = "orbit (frame 0: first seed photo)"
        else:
            path, colour, mode_info, extra = offset_path(args, photos, gaussians, obj_, clearance, checker)
            colour_label = "offset scale (1 = full offset, 0 = base camera)"
        info.update(mode_info)
        keep = extra[0] if extra else None
        if mode != "vidsplat" and args.max_frames and len(path) > args.max_frames:
            stride = int(np.ceil(len(path) / args.max_frames))  # frames are checked one by one: any subset is safe
            path, colour = path[::stride], colour[::stride]
            info["frame_stride"] = stride
            info["pieces"] = pieces_of(len(path), [-(-a // stride) for a, _ in info["pieces"]])
            print(f"--max_frames {args.max_frames}: every {stride}th frame, {len(path)} frames")
    except TrajectoryError as error:
        failure = error
        path = error.path if error.path is not None else photos.c2w
        info.update(error.info, failed=error.reason)
        colour, colour_label = np.zeros(len(path)), "failed"

    pieces = info.get("pieces") or [[0, len(path) - 1]]
    if pieces[-1][1] != len(path) - 1:  # a failed vidsplat path is cut short
        pieces = [[0, len(path) - 1]]
    info["pieces"] = pieces
    info["jumps_spacings"] = [round(j, 2) for j in jump_lengths(path, pieces, photos.spacing)]
    info_path = write_outputs(args, photos, path, info)
    if gaussians is not None and keep is None:
        keep = gaussians.opacity >= args.solid
    plot_top_view(args, photos, path, colour, colour_label, gaussians, keep, obj_, clearance, pieces,
                  anchors=info.get("anchors"), title=f"FAILED: {failure.reason[:90]}..." if failure else "")
    if failure is not None:
        print(f"ERROR: {failure.reason}", file=sys.stderr)
        print(f"wrote {info_path} (no trajectory)", file=sys.stderr)
        return 2

    nearest = np.linalg.norm(path[:, None, :3, 3] - photos.pos[None], axis=2).min(1) / photos.spacing
    print(f"photos={photos.n} path_frames={len(path)} median_spacing={photos.spacing:.4f} (scene units)")
    print(f"{len(pieces)} continuous pieces (first frames {[a for a, _ in pieces]}), so {len(pieces) - 1} jump{'s' * (len(pieces) != 2)}"
          + (f" of {', '.join(f'{j:.1f}' for j in info['jumps_spacings'])} photo spacings" if len(pieces) > 1 else ""))
    print(f"path camera to nearest photo camera, in spacings: mean {nearest.mean():.2f} max {nearest.max():.2f}")
    if len(path) == photos.n:
        angle = np.degrees(np.linalg.norm((Rotation.from_matrix(path[:, :3, :3])
                                           * Rotation.from_matrix(photos.c2w[:, :3, :3]).inv()).as_rotvec(), axis=1))
        print(f"rotation change vs. the photo at the same index: mean {angle.mean():.1f} deg, max {angle.max():.1f} deg")
    if obj_:
        gap = np.linalg.norm(path[:, :3, 3] - obj_.centre, axis=1) / obj_.radius
        print(f"path camera to object centre, in R: min {gap.min():.2f} mean {gap.mean():.2f} max {gap.max():.2f}")
    print(f"wrote {args.output} and {info_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
