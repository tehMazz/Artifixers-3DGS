<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ArtiFixer-GS

[ArtiFixer](https://research.nvidia.com/labs/sil/projects/artifixer/) (de Lutio et al., SIGGRAPH 2026) for
any 3D Gaussian Splatting method that writes the standard 3DGS PLY: 3DGS, FastGS, Taming-3DGS, and others.

The original repository trains and renders with 3DGUT. The ArtiFixer diffusion model itself only reads
images: RGB renders, opacity maps and the photos along a camera path. This repository keeps the
diffusion model and its data preparation unchanged, renders the PLY with [gsplat](https://github.com/nerfstudio-project/gsplat),
and leaves 3DGS training to the method of your choice. The ArtiFixer3D distillation runs in FastGS
(branch `artifixer`, flag `--lpips_images`). There is no 3DGUT anywhere.

```
your 3DGS PLY ──prepare──► renders + opacity ──ArtiFixer──► fixed frames (video)
                                                              │
               ArtiFixer3D+ video ◄──ArtiFixer── renders ◄── FastGS (photos + fixed frames) ◄──handoff
```

## Pipeline

| stage | command | env | output |
|---|---|---|---|
| prepare | `python -m artifixer_gs.prepare` | ArtiFixer | `input/<id>/` (photos, COLMAP, `transforms.json`), `renders/{photos,trajectory}/{renders,opacity}/`, metric scale, `captions/<id>/caption.h5`, `split.json` |
| infer | `python -m model_eval.run_inference --evalset reconstructed_colmap` | ArtiFixer | `artifixer/.../frames/batch_0000/pred/*.png` + mp4 |
| handoff | `python -m artifixer_gs.distill handoff` | ArtiFixer | `af3d/input/`: COLMAP scene with anchor photos + ArtiFixer frames (PINHOLE), `generated_images.txt` |
| train | FastGS `train.py --lpips_images af3d/input/generated_images.txt` | FastGS | `af3d/fastgs/point_cloud/iteration_N/point_cloud.ply`, `gs_stats.json` |
| plus | `python -m artifixer_gs.distill plus` | ArtiFixer | `af3d/renders/`, `split_artifixer3d_plus.json` |
| infer_plus | `run_inference` on the plus split | ArtiFixer | `artifixer3d_plus/.../pred/*.png` + mp4 |

`scripts/run.sh` runs them all, with one log per stage in `$OUT/logs/<stage>.log`. A finished stage
writes `$OUT/logs/.done_<stage>` (its settings) and is skipped on a rerun with the same settings.
A stage whose settings changed is redone from scratch (`prepare` with `--replace`: new photos,
renders, scale, caption), and so are the stages after it; `FORCE=1` redoes everything. Without
`CAPTION`, an existing `captions/<id>/caption.h5` is kept even then (the Qwen3-VL captioner runs only
when there is none). `IMAGES=images_4` (or
`prepare --images`) picks the photo folder the PLY was trained on; the cameras are scaled to it.

```bash
PLY=.../point_cloud.ply COLMAP=.../scene OUT=output/scene ANCHORS=.../train_images.txt \
CAPTION="A kitchen table ..." MODEL_ID=.../Wan2.1-T2V-1.3B-Diffusers CHECKPOINT_PT=.../artifixer-1.3b.pt \
FASTGS_REPO=.../FastGS FASTGS_PY=.../fastgs/bin/python bash scripts/run.sh
```

There are two kinds of path:

- **Sparse anchors** (`ANCHORS`): the listed photos stay real. The other photo frames are targets that
  ArtiFixer regenerates. The PLY should be trained on the anchors only. `python -m artifixer_gs.subset`
  writes that anchor-only COLMAP scene.
- **Novel path** (`TRAJ`): a `transforms.json` of target cameras, given as OpenGL camera-to-world
  matrices plus intrinsics. Every photo is an anchor.

- **Generated novel path** (`TRAJ_MODE=object|vidsplat|travel`): a `trajectory` stage before
  `prepare` writes `TRAJ=$OUT/trajectory_input.json` with `artifixer_gs.trajectory` (below).
- **Object removal** (`MODE=removal`): the object is deleted from the PLY and ArtiFixer inpaints the
  hole on the photo frames (below).

`STAGES=handoff,train,plus,infer_plus` runs only part of the pipeline. The ArtiFixer stage alone
(PLY in, video out) is `STAGES=prepare,infer`.

## Novel camera paths: `artifixer_gs.trajectory`

Port of ArtiFixer's `scripts/bakerh/make_trajectory.py` (same algorithm, numbers and checks) with the
Gaussians read from the PLY, the object from a per-Gaussian mask and the render checks in gsplat.

```bash
python -m artifixer_gs.trajectory --ply P (--transforms T | --colmap_dir C [--images images_4]) \
    --output O.json --mode object|vidsplat|travel [--object_mask object_mask.npy] [--metric_scale S] [knobs]
```

| mode | path | main knobs (defaults) |
|---|---|---|
| `travel` | smoothed photo path weaving across the direction of travel | `--side 2 --up 0.5` (photo spacings), `--periods 2` |
| `object` | the same path orbiting the object's centre of mass, looking at it | `--orbit 0.6 --radial 0.3 --rise 0.3` (object radii), `--standoff 0.5`, `--render_check`, `--allow_fallback`, `--max_collapsed 0.5` |
| `vidsplat` | 16-frame orbits about seed photos' surface points that reveal unseen surface | `--clip_len 16 --orbit_deg 15,30,45 --s_low 0.03 --s_high 0.4 --seeds 12 [--budget] [--min_clips]` |

Every mode keeps `--clearance_m 0.3` metres (with `--metric_scale`) from any surface, never crosses a
wall (occupancy field of the Gaussians), and `--d0_m 0.5` is the near-surface limit of the render
checks. `--max_frames`, `--phase_offset` and `--seen_transforms` are as in ArtiFixer. It writes the
trajectory JSON (`prepare --trajectory_path` contract: OPENCV intrinsics, frames with
`transform_matrix` only), `<stem>_info.json` (rejections, per-frame/per-orbit stats, failure reason)
and `<stem>.png` (top view); it exits 2 when no acceptable path exists. The render checks (vidsplat,
`--render_check`) need gsplat: `run.sh` runs the stage with `GS_RENDER_PY` when `PY` has no gsplat.

`run.sh` knobs: `TRAJ_MODE`, `OBJECT_MASK` (object mode), `TRAJ_CLEARANCE_M`, `TRAJ_D0_M`,
`TRAJ_MAX_FRAMES`, `TRAJ_ORBIT`, `TRAJ_RADIAL`, `TRAJ_RISE`, `TRAJ_RENDER_CHECK=1`,
`TRAJ_ALLOW_FALLBACK=1`, `TRAJ_CLIP`, `TRAJ_ORBIT_DEG`, `TRAJ_S_LOW`, `TRAJ_S_HIGH`, `TRAJ_BUDGET`,
`TRAJ_SEEDS`, `TRAJ_SIDE`, `TRAJ_UP`, `TRAJ_EXTRA_ARGS`. As in ArtiFixer's `run_removal.sh`, the path
gets `--metric_scale` only from a known scale (`METRIC_SCALE`, ArtiFixer's scene `scale_info.txt`);
without it the clearance comes from the object size and MoGe-2 runs afterwards, in `prepare`. A
changed knob regenerates the path and redoes the later stages. The ArtiFixer3D+ pass of `TRAJ_MODE`
uses `run_inference`'s default reference selection, as ArtiFixer's `vanilla_*` variants
(`TRAJ` / `ANCHORS` keep `--neighbor_selection_mode covisibility`).

## Object removal: `MODE=removal`

ArtiFixer's BakerH removal pipeline (`scripts/bakerh/run_removal.sh` variants `normal` / `bighull`)
on a 3DGS PLY, without 3DGUT or FlashSplat. Inputs: `OBJECT_MASK`, a bool `.npy` [P] aligned with the
PLY vertices (e.g. BEA-GS `labels.pt == k`), and `MASK_DIR`, 2D object masks `<photo stem>.npy|.png`
(2D id/binary map: object `> 0` or `== MASK_LABEL`; `(C,H,W)` BEA-GS mask: object = channel
`MASK_LABEL` (1) `> 0.5`; resized NEAREST).

| stage | command | output |
|---|---|---|
| prepare | `artifixer_gs.prepare --phases prepare,scale,caption` (no render; no caption for bighull) | `input/<id>/`, metric scale, caption |
| removal | `artifixer_gs.removal render --ply --object_mask --split_path $OUT --output_dir $OUT/removal [--hull_trim 99.5] [--hull_expand 0]` | `removed.ply`, `deleted.ply`, `deleted_mask.npy`, `hull_points.npy`, `removal.json`, `background/{renders,opacity}/`, `deleted/{renders,opacity}/` |
| split | `artifixer_gs.removal split --scene_root $OUT --removal_dir $OUT/removal --mask_dir M --output_dir $OUT/removal_split --hole_shape mask\|hull --dilate_px N [--frames a-b,c-d] [--prompt_path P] [--refs render] [--opacity hole0]` | `renders/`, `opacity/` (0 in the hole), `hole/`, `refs/`, `transforms.json`, `selected_indices.json` (all), `frames.json`, `split.json`, `scene/` |
| infer | `run_inference --render_trajectory all_frames` on that split | one rollout over every frame |
| propagate | `artifixer_gs.propagate --scene_root .../scene --removal_dir $OUT/removal_split --seed_pred_dir <pred> --seed_frames auto` | `propagation/{anchors,pred}/`, `order.json` |
| handoff | `artifixer_gs.distill removal_handoff --removal_split ... --propagated_dir ... --output_dir $OUT/af3d/input [--anchor_every 3] [--drop_points_mask]` | FastGS COLMAP scene of the composites, `generated_images.txt`, ArtiFixer3D `split.json` |
| train, plus, infer_plus | as above; `plus` reads `af3d/input/split.json`; `infer_plus` runs `val_frames` (the non-anchors) | ArtiFixer3D PLY, ArtiFixer3D+ frames |

- **Removal rule** (`render`): deleted = the object Gaussians plus every other Gaussian whose centre is
  inside the 3D convex hull of the object centres, trimmed at the 99.5th percentile of their distance
  to the centroid, subsampled to 10k points, every plane pushed out by `--hull_expand` x their mean
  radius. This is ArtiFixer's `--background-convex-hull --convex-hull-trim-percentile 99.5` and the
  `visibility_hull` of `aurafusion_3dgs`; BEA-GS labels come from multi-view votes, so there is no
  FlashSplat contribution filter. Renders go through `artifixer_gs.render` (so `GS_RENDER_PY` works).
- **Hole** (`split`): `(2D mask) | (deleted opacity > --fg_opacity_threshold 0.05)`, its 2D convex
  hull with `--hole_shape hull`, then `binary_dilation` `--dilate_px` times with the 3x3 cross (an L1
  radius). This is `HOLE_MODE=dilate` (default). `HOLE_MODE=alpha` keeps only the pixels of that
  dilated hole where the removed scene is see-through (background alpha < 0.5) or the deleted
  Gaussians show (alpha > 0.05). `HOLE_MODE=mesh HOLE_MESH=<triangle PLY>` adds to that every pixel
  where the mesh (e.g. Gaussian Wrapping's `hole_<k>.ply`) is in front of the removed scene's depth
  (2% slack, so a contact patch counts): the removal stage renders the background depth and runs
  `artifixer_gs.hole_mesh` (nvdiffrast) in `MESH_PY`.
  ArtiFixer gets no mask: the opacity map is 0 in the hole, so those latents start from noise.
  With `--refs render` the reference views are the object-free renders, so nothing ArtiFixer is
  conditioned on shows the object.
- **Variants**: `REMOVAL_VARIANT=normal` (default) = `HOLE_SHAPE=mask DILATE=12`, scene caption
  (`CAPTION`); `REMOVAL_VARIANT=bighull` = `HOLE_SHAPE=hull DILATE=80` and the prompt
  `EMPTY_CAPTION` ("An empty floor without holes: one continuous, clean floor surface with nothing
  standing on it, in an industrial hall.", written by `artifixer_gs.caption` to `$OUT/prompt/caption.h5`).
  Only bighull reads `EMPTY_CAPTION` (as ArtiFixer); normal ignores it and prompts with the scene
  caption. Every other knob can be set alone: `HOLE_SHAPE`, `DILATE`, `MASK_LABEL`, `FRAMES`, `REFS`,
  `HULL_TRIM`, `HULL_EXPAND`.
- **Frames and seeds**: `FRAMES="49-148,284-304"` keeps those frames (0-based, prepared order); each
  contiguous run is a window, and `frames.json` lists `source_indices`, `window_starts` and
  `auto_seeds` (the first 7 frames of each window). `SEEDS=auto` (default) or e.g. `SEEDS=0-6,116-122`
  picks the frames of the single rollout that `propagate` starts from.
- **Propagation** (ArtiFixer's `output/jobs/propagate_removal.py`): the frame order is cut at camera
  jumps above `--jump_factor 4` x the median step (one sequence whose ends meet is cyclic); each round
  every done frame grows a chunk of `--step 7` undone frames in each direction, ArtiFixer generates
  them conditioned on the covisible anchors, and each is composited into the photo (`OUTSIDE=photo`)
  or the object-free render (`OUTSIDE=render`) with a `--feather_sigma 3` feather that never drops
  below the hole. The model is loaded once.
- **Distillation** (ArtiFixer's `output/jobs/build_af3d_split.py`): every FastGS image is a composite
  at its photo pose (source COLMAP pose, checked against the split), PINHOLE cameras; anchors
  (`ANCHOR_EVERY` 3rd frame + the seeds) get 0.8 L1 + 0.2 D-SSIM, the rest LPIPS only.
  `points3D.bin` is copied unchanged, as in ArtiFixer; `DROP_POINTS=1` drops the points inside the
  removed hull (the default; `DROP_POINTS=0` keeps them). ArtiFixer3D+ regenerates the non-anchor frames from the FastGS renders with the 6 most
  covisible anchor composites as references (raw render opacity, no hole).

```bash
MODE=removal REMOVAL_VARIANT=bighull PLY=.../point_cloud.ply COLMAP=.../scene IMAGES=images_4 \
OBJECT_MASK=.../object_mask.npy MASK_DIR=.../masks OUT=output/scene_bighull METRIC_SCALE=0.2 \
MODEL_ID=... CHECKPOINT_PT=... FASTGS_REPO=... FASTGS_PY=... GS_RENDER_PY=... bash scripts/run.sh
```

### What ArtiFixer needs from the 3D model

- `renders/{i:05d}.png`: 8-bit RGB composited on **black**, the same as the training renders.
- `opacity/{i:05d}.png`: accumulated alpha × 255. ArtiFixer uses it as the per-pixel mix between the
  render latent and noise (`model_training/pipeline/pipeline_base.py`).
- `transforms.json` with the poses and intrinsics, and `split.json` pointing at all of the above.

`artifixer_gs.render` writes exactly this from a PLY. It reads the SH degree from the `f_rest` count,
applies sigmoid to opacity, exp to scale and normalises the wxyz quaternion, and uses gsplat's
`classic` mode with eps2d 0.3, which is the 3DGS rasterizer (`EPS2D` / `--eps2d` changes it: 0 for
Gaussian Wrapping, whose rasterizer adds no 2D low-pass and whose 3D filter is baked into the exported PLY). If gsplat is not installed in the
ArtiFixer Python, the render runs as a subprocess in `GS_RENDER_PY`.

## FastGS: ArtiFixer3D training

Another trainer: `TRAIN_CMD` replaces `$FASTGS_PY train.py` (same arguments), and `AF3D_EXPORT`
converts its model into the standard PLY that `plus` renders (Gaussian Wrapping:
`TRAIN_CMD="python gaussian_wrapping/train.py --rasterizer ours"`,
`AF3D_EXPORT="python gaussian_wrapping/export_3dgs_ply.py"`, `FASTGS_REPO` = its checkout).

Branch `artifixer` of FastGS adds `--lpips_images FILE` and `--lambda_lpips 0.1` to `train.py`.
Images listed in the file (the ArtiFixer frames) get `0.1 · LPIPS(vgg)` only. Every other image (the
real photos) keeps FastGS's `0.8 L1 + 0.2 D-SSIM`. This is the 3DGUT ArtiFixer3D recipe
(`colmap_3dgut_sparse_mcmc_lpips`: `lambda_reconlosses_override 0`): frame-to-frame inconsistencies in
the generated frames do not blur the model.

FastGS's multi-view densification and pruning are unchanged. That includes the final pruning of
Gaussians below opacity 0.1, which the MCMC strategy of the 3DGUT version does not have.
`gs_stats.json` reports the Gaussian count and the share of low-opacity Gaussians.

## Deviations from the 3DGUT version

- **Training from scratch only.** There is no `--base_checkpoint` continuation with rescaled schedules.
  ArtiFixer3D trains FastGS from the COLMAP points, as the default 3DGUT ArtiFixer3D does.
- **Pinhole only.** Lens distortion raises an error, so undistort the scene first. 3DGS trainers that
  read PINHOLE COLMAP ignore the principal point and assume the image centre. The renders here use the
  exact `cx, cy` (sub-pixel on mip360).
- **Renderer.** Renders come from gsplat, not 3DGUT. On mip360 kitchen they match FastGS's own renders
  at 49 dB PSNR (`tests/check_render_vs_fastgs.py`).
- **Captions.** `--caption TEXT` writes `caption.h5` with the Wan text encoder and skips the Qwen3-VL
  captioner.
- **Object removal.** The object is a per-Gaussian mask (e.g. BEA-GS labels) instead of FlashSplat's
  labels with its contribution / hit-count visibility filter, and Gaussians no view sees stay in the
  background. The hole footprint is the opacity of everything deleted (object + hull), where ArtiFixer
  used the object alone, so surface the hull removed is regenerated too. The removal split renders at
  the prepared photo size, so ArtiFixer's resize-the-photos-to-the-render step is not needed.

## Files

- `artifixer_gs/`: `prepare.py` (adapted from `data_processing/prepare_colmap_artifixer_inputs.py`),
  `render.py`, `distill.py` (adapted from `data_processing/artifixer3d.py` and
  `output/jobs/build_af3d_split.py`), `subset.py`, `trajectory.py` (`scripts/bakerh/make_trajectory.py`),
  `removal.py` (`render_flashsplat_extraction.py`'s removal rule, `scripts/bakerh/derive_scene.py`,
  `output/jobs/build_removal_split.py`), `propagate.py` (`output/jobs/propagate_removal.py`),
  `caption.py` (`output/jobs/write_caption.py`), and `colmap.py` / `camera_models.py` (copied from
  3DGRUT so no 3DGRUT install is needed).
- `model_eval/`, `model_training/`, `data_processing/`: the ArtiFixer inference code, copied unchanged
  except for those two imports.
- `scripts/run.sh`: the whole pipeline. `scripts/leonardo/`: Leonardo env and the kitchen test job.
- `tests/check_render_vs_fastgs.py`: renderer equivalence. `tests/compare_targets.py`: PSNR per stage
  on the target frames, plus Gaussian opacity stats. `tests/test_trajectory.py`, `tests/test_removal.py`:
  CPU pytest suites (injected renderers, no gsplat or GPU): `python -m pytest tests`.

Models are not included: Wan2.1-T2V-1.3B-Diffusers, `artifixer-1.3b.pt` (`nvidia/ArtiFixer`) and
MoGe-2 (`MOGE_MODEL_PATH`).

## License

Apache-2.0, as ArtiFixer. See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
Please cite ArtiFixer:

```bibtex
@inproceedings{delutio2026artifixer,
    title={ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models},
    author={de Lutio, Riccardo and Fischer, Tobias and Chang, Yen-Yu and Zhang, Yuxuan and
            Wu, Jay Zhangjie and Ren, Xuanchi and Shen, Tianchang and Tothova, Katarina and
            Gojcic, Zan and Turki, Haithem},
    booktitle={SIGGRAPH},
    year={2026}
}
```
