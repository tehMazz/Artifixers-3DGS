# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-view hole of a removed object from its mesh (removal split --hole_mode mesh).

The mesh (e.g. Gaussian Wrapping multilabel_mesh_extraction hole_<k>.ply: the label-k closure plus its contact
patch) is rasterized with nvdiffrast on every frame that `removal render --depth` rendered. A pixel is in front
when the mesh covers it and either the removed scene is transparent there (background alpha < --background_alpha)
or the mesh depth is at most (1 + --depth_tol) x the removed scene's expected depth (the tolerance keeps the
contact patch, which lies on the surface it touches). Writes <removal_dir>/mesh_front/{i:05d}.png (255 = in front)
and mesh_front.json.

Needs torch + CUDA + nvdiffrast (MESH_PY in scripts/run.sh: the Gaussian Wrapping env has it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from artifixer_gs.render import trajectory_cameras

NEAR, FAR = 0.01, 1.0e4


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices [V,3] float32 and triangles [F,3] int32 of a PLY mesh (extra vertex / face properties ignored)."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    verts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    face = ply["face"]
    key = next(name for name in ("vertex_indices", "vertex_index") if name in face.data.dtype.names)
    faces = face.data[key]
    assert len(faces) > 0, f"{path} has no faces"
    lengths = {len(f) for f in faces}
    assert lengths == {3}, f"{path}: only triangle meshes are supported, got polygons of {sorted(lengths)} vertices"
    return verts, np.stack(faces).astype(np.int32)


def clip_coords(verts, viewmat, K, width: int, height: int, near: float = NEAR, far: float = FAR):
    """nvdiffrast clip coordinates [V,4] and camera z [V] of verts under an OpenCV w2c viewmat and K.

    clip x / w = 2u / W - 1 and y / w = 2v / H - 1 (v down), so row r of the rasterized image is pixel row r."""
    import torch

    cam = verts @ viewmat[:3, :3].T + viewmat[:3, 3]
    x, y, z = cam.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    clip = torch.stack(
        [
            2 * fx / width * x + (2 * cx / width - 1) * z,
            2 * fy / height * y + (2 * cy / height - 1) * z,
            (far + near) / (far - near) * z - 2 * far * near / (far - near),
            z,
        ],
        dim=-1,
    )
    return clip, z


def front_mask(
    mesh_depth: np.ndarray,
    covered: np.ndarray,
    background_depth: np.ndarray,
    background_alpha: np.ndarray,
    depth_tol: float = 0.02,
    background_alpha_threshold: float = 0.5,
) -> np.ndarray:
    """covered & (background alpha < threshold | mesh depth <= (1 + tol) x background depth); alpha in [0, 1]."""
    transparent = background_alpha < background_alpha_threshold
    return covered & (transparent | (mesh_depth <= (1.0 + depth_tol) * background_depth))


def mesh_front(args: argparse.Namespace) -> Path:
    import torch

    try:
        import nvdiffrast.torch as dr
    except ImportError as error:
        raise SystemExit(
            f"--hole_mode mesh needs nvdiffrast, which this Python cannot import ({error}): set MESH_PY to a Python "
            "with torch + nvdiffrast (e.g. the Gaussian Wrapping env), or use HOLE_MODE=alpha"
        ) from error

    removal_dir = args.removal_dir.expanduser().resolve()
    removal = json.loads((removal_dir / "removal.json").read_text())
    depth_dir = removal_dir / "background" / "depth"
    assert depth_dir.is_dir(), f"No {depth_dir}: run removal render with --depth"
    viewmats, Ks, width, height = trajectory_cameras(Path(removal["transforms_path"]))
    assert len(viewmats) == removal["frames"], f"{removal['transforms_path']} changed since removal render"

    device = torch.device("cuda")
    verts, faces = read_mesh(args.mesh)
    verts_t = torch.from_numpy(verts).to(device)
    faces_t = torch.from_numpy(faces).to(device)
    glctx = dr.RasterizeCudaContext(device=device)
    out = removal_dir / "mesh_front"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Rasterizing {args.mesh} ({len(verts)} vertices, {len(faces)} faces) on {len(viewmats)} frames", flush=True)

    fractions = []
    with torch.no_grad():
        for index in range(len(viewmats)):
            viewmat = torch.from_numpy(viewmats[index]).to(device)
            K = torch.from_numpy(Ks[index]).to(device)
            clip, z = clip_coords(verts_t, viewmat, K, width, height)
            rast, _ = dr.rasterize(glctx, clip[None].contiguous(), faces_t, resolution=[height, width])
            depth, _ = dr.interpolate(z[None, :, None].contiguous(), rast, faces_t)
            covered = (rast[0, ..., 3] > 0).cpu().numpy()
            background_alpha = np.asarray(Image.open(removal_dir / "background" / "opacity" / f"{index:05d}.png").convert("L")) / 255.0
            background_depth = np.load(depth_dir / f"{index:05d}.npy")
            front = front_mask(depth[0, ..., 0].cpu().numpy(), covered, background_depth, background_alpha,
                               args.depth_tol, args.background_alpha)
            fractions.append(float(front.mean()))
            Image.fromarray((front * 255).astype(np.uint8)).save(out / f"{index:05d}.png")

    info = {
        "mesh": str(args.mesh.expanduser().resolve()),
        "frames": len(viewmats),
        "depth_tol": args.depth_tol,
        "background_alpha": args.background_alpha,
        "front_fraction_mean": float(np.mean(fractions)),
        "front_views": int(np.count_nonzero(fractions)),
    }
    (removal_dir / "mesh_front.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"Wrote {out}: mesh in front on {info['front_views']}/{len(viewmats)} frames, mean fraction {info['front_fraction_mean']:.4f}",
          flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, required=True, help="Hole mesh PLY (triangles, same frame as the 3DGS PLY).")
    parser.add_argument("--removal_dir", type=Path, required=True, help="removal render --depth output.")
    parser.add_argument("--depth_tol", type=float, default=0.02, help="Relative depth slack (keeps the contact patch).")
    parser.add_argument("--background_alpha", type=float, default=0.5, help="Below this the removed scene is see-through.")
    mesh_front(parser.parse_args(argv))


if __name__ == "__main__":
    main()
