#!/usr/bin/env bash
# PLY -> ArtiFixer video -> ArtiFixer3D (FastGS) -> ArtiFixer3D+ video, on one GPU.
#
# Required:
#   PLY      trained 3DGS PLY (standard INRIA attributes), trained on COLMAP
#   COLMAP   COLMAP scene: $IMAGES/ + sparse/0/{cameras,images,points3D}.bin (undistorted)
#   OUT      output scene root; its basename is the scene id
#   MODEL_ID, CHECKPOINT_PT  Wan2.1-T2V-1.3B-Diffusers dir and artifixer-1.3b.pt
# Path (one of):
#   ANCHORS   file of photo names kept as real anchors; the other photos are ArtiFixer targets
#   TRAJ      transforms.json of novel target cameras (OpenGL c2w + intrinsics); every photo is an anchor
#   TRAJ_MODE object|vidsplat|travel: the `trajectory` stage builds TRAJ=$OUT/trajectory_input.json with
#             artifixer_gs.trajectory (object needs OBJECT_MASK); knobs below
#   MODE=removal  object removal + inpainting on the photo frames (no novel path), see below
# Optional:
#   IMAGES        photo folder in COLMAP the PLY was trained on (images)
#   CAPTION       scene caption (else an existing captions/<id>/caption.h5 is kept, else the Qwen3-VL captioner runs)
#   METRIC_SCALE  metres per scene unit (else MoGe-2 estimates it in prepare)
#   STAGES        subset of the mode's stages (default: all of them; `+` works as a separator)
#   FORCE=1       redo finished stages (a stage is skipped on rerun when its marker
#                 $OUT/logs/.done_<stage> holds the same settings; a stage whose settings changed is redone
#                 from scratch (prepare/trajectory with --replace), and so are the later stages)
#   AF3D_STEPS    FastGS iterations for ArtiFixer3D (30000); FASTGS_ARGS extra train.py flags
#   PY            Python with the ArtiFixer deps (default: python)
#   GS_RENDER_PY  Python with gsplat, when PY has none (renders; also runs the trajectory stage)
#   EPS2D         gsplat 2D covariance low-pass of every render (0.3; 0 for Gaussian Wrapping PLYs)
#   RENDER_CMD    renderer used instead of gsplat for every PLY render (prepare, removal, plus), run in RENDER_CWD
#                 as $RENDER_CMD --ply P --trajectory T --out_dir D [--depth], same layout as artifixer_gs.render
#                 (e.g. Gaussian Wrapping's "python gaussian_wrapping/render_ply_frames.py", RENDER_CWD = the GW repo);
#                 unset = gsplat. gsplat is still used by the trajectory stage's visibility checks
#                 (TRAJ_RENDER_CHECK, and always for TRAJ_MODE=vidsplat)
#   CENTER_PRINCIPAL_POINT=1  every camera read from transforms.json gets its principal point at the image centre
#                 (renders and hole_mesh), the camera model of trainers that ignore COLMAP's cx, cy (Gaussian Wrapping)
#   IGNORE_DISTORTION=1  render / distill SIMPLE_RADIAL, RADIAL, OPENCV cameras as pinhole (their distortion
#                 dropped, as Gaussian Wrapping trains them); unset = such cameras are an error
#   FASTGS_REPO, FASTGS_PY   FastGS checkout with train.py --lpips_images, and its Python
#   TRAIN_CMD     ArtiFixer3D trainer run in FASTGS_REPO (default "$FASTGS_PY train.py"), e.g.
#                 "python gaussian_wrapping/train.py --rasterizer ours"; it gets -s -m -r 1 --iterations ...
#                 --lpips_images like FastGS
#   AF3D_EXPORT   command that turns the trained model into a standard 3DGS PLY, run in FASTGS_REPO as
#                 $AF3D_EXPORT -m <model> --iteration AF3D_STEPS --out $OUT/af3d/export/point_cloud.ply
#                 (e.g. "python gaussian_wrapping/export_3dgs_ply.py"); unset = the trained PLY is standard
#   NUM_VIEWS (6), INFER_ARGS extra run_inference flags
#
# MODE=removal stages: prepare,removal,split,infer,propagate,handoff,train,plus,infer_plus
#   prepare    photos + COLMAP + transforms.json, metric scale, caption (no render)
#   removal    artifixer_gs.removal render: OBJECT_MASK (bool [P] over the PLY) + its 3D hull removed;
#              background / deleted renders of every photo ($OUT/removal)
#   split      artifixer_gs.removal split: hole = 2D mask | deleted opacity, HOLE_SHAPE, DILATE, HOLE_MODE
#              ($OUT/removal_split); with EMPTY_CAPTION, that text is the prompt (prompt/caption.h5)
#   infer      one all_frames ArtiFixer rollout on the hole split
#   propagate  artifixer_gs.propagate from SEEDS of that rollout ($OUT/propagation)
#   handoff    artifixer_gs.distill removal_handoff: composites -> FastGS scene + ArtiFixer3D split
#   train, plus, infer_plus as for the other modes (AF3D+ = val_frames over the non-anchors)
#   Env: OBJECT_MASK, MASK_DIR (2D masks <stem>.npy|.png), MASK_LABEL (object id in them; default > 0 /
#   channel 1), REMOVAL_VARIANT normal (HOLE_SHAPE=mask DILATE=12, scene caption) | bighull
#   (HOLE_SHAPE=hull DILATE=80, EMPTY_CAPTION defaults to the empty-floor prompt; only bighull reads
#   EMPTY_CAPTION, normal always prompts with the scene caption), HOLE_SHAPE, DILATE,
#   FRAMES ("49-148,284-304"), SEEDS (auto = first 7 frames of each FRAMES window; or "0-6,116-122"),
#   OUTSIDE (photo|render), REFS (render), HULL_TRIM (99.5), HULL_EXPAND (0), ANCHOR_EVERY (3),
#   DROP_POINTS=1 (drop the COLMAP points inside the removed hull from the FastGS initialisation; 0 = off),
#   HOLE_MODE dilate (default: the dilated mask) | alpha (its pixels where the removed scene is see-through,
#   background alpha < 0.5, or the deleted Gaussians show, alpha > 0.05) | mesh (alpha | the pixels where
#   HOLE_MESH, a triangle PLY of the removed object's closure, is in front of the removed scene: the removal
#   stage also renders the background depth and runs artifixer_gs.hole_mesh in MESH_PY, a Python with
#   torch + nvdiffrast; default PY, else GS_RENDER_PY, if either imports nvdiffrast) | gavis (its pixels where
#   UNSEEN_DIR, per-view unseen maps <photo stem>.png = 255 * U from BEA-GS beags/unseen_map.py, is above UNSEEN_THRESH
#   (default 0.5), or the deleted Gaussians show: the removed scene there comes from Gaussians the training views never
#   saw from that direction, or from nothing; frames without a map use alpha's rule).
#   CONF_THRESH (unset: off) with CONFIDENCE_DIR (per-view confidence <photo stem>.png, 255 * min(c, 1), e.g. BEA-GS
#   beags_artifixer.py <out>/confidence): HOLE_MODE alpha | mesh also keep the dilated-mask pixels whose confidence is
#   below CONF_THRESH (in (0, 1], e.g. 0.5); frames without a map add nothing; HOLE_MODE dilate keeps them anyway.
# TRAJ_MODE knobs (artifixer_gs.trajectory flags): TRAJ_CLEARANCE_M 0.3, TRAJ_D0_M 0.5, TRAJ_MAX_FRAMES;
#   the path gets --metric_scale only from METRIC_SCALE (as ArtiFixer reads only the scene's existing
#   scale_info.txt); without it the clearance comes from the object size and MoGe-2 runs later, in prepare.
#   The ArtiFixer3D+ pass uses run_inference's default references (no covisibility), as ArtiFixer's vanilla_*.
#   object: TRAJ_ORBIT 0.6, TRAJ_RADIAL 0.3, TRAJ_RISE 0.3, TRAJ_RENDER_CHECK 0, TRAJ_ALLOW_FALLBACK 0;
#   vidsplat: TRAJ_CLIP 16, TRAJ_ORBIT_DEG 15,30,45, TRAJ_S_LOW 0.03, TRAJ_S_HIGH 0.4, TRAJ_BUDGET,
#   TRAJ_SEEDS 12; travel: TRAJ_SIDE 2, TRAJ_UP 0.5; TRAJ_EXTRA_ARGS any other flags.
set -euo pipefail
: "${PLY:?set PLY}" "${COLMAP:?set COLMAP}" "${OUT:?set OUT}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PY=${PY:-python}
IMAGES=${IMAGES:-images}
MODE=${MODE:-}
AF3D_STEPS=${AF3D_STEPS:-30000}
NUM_VIEWS=${NUM_VIEWS:-6}
export EPS2D=${EPS2D:-0.3}  # read by artifixer_gs.render (every gsplat render, here and in GS_RENDER_PY)
: "${MODEL_ID:?set MODEL_ID}" "${CHECKPOINT_PT:?set CHECKPOINT_PT}"
EMPTY_FLOOR="An empty floor without holes: one continuous, clean floor surface with nothing standing on it, in an industrial hall."

if [ "$MODE" = removal ]; then
  : "${OBJECT_MASK:?set OBJECT_MASK (bool .npy over the PLY Gaussians)}" "${MASK_DIR:?set MASK_DIR (2D object masks)}"
  [ -z "${TRAJ:-}${TRAJ_MODE:-}${ANCHORS:-}" ] || { echo "MODE=removal takes no TRAJ, TRAJ_MODE or ANCHORS" >&2; exit 2; }
  case ${REMOVAL_VARIANT:-normal} in
    normal)  # the scene caption is the prompt, as ArtiFixer: an inherited EMPTY_CAPTION is not used
      [ -z "${EMPTY_CAPTION:-}" ] || echo "REMOVAL_VARIANT=normal ignores EMPTY_CAPTION (bighull only)" >&2
      EMPTY_CAPTION=""; HOLE_SHAPE=${HOLE_SHAPE:-mask}; DILATE=${DILATE:-12} ;;
    bighull) HOLE_SHAPE=${HOLE_SHAPE:-hull}; DILATE=${DILATE:-80}; EMPTY_CAPTION=${EMPTY_CAPTION:-$EMPTY_FLOOR} ;;
    *) echo "unknown REMOVAL_VARIANT '$REMOVAL_VARIANT' (normal, bighull)" >&2; exit 2 ;;
  esac
  DROP_POINTS=${DROP_POINTS:-1}
  HOLE_MODE=${HOLE_MODE:-dilate}
  case $HOLE_MODE in
    dilate|alpha) ;;
    mesh) : "${HOLE_MESH:?HOLE_MODE=mesh needs HOLE_MESH (triangle PLY of the object closure)}"
          [ -f "$HOLE_MESH" ] || { echo "HOLE_MESH $HOLE_MESH does not exist" >&2; exit 2; } ;;
    gavis) : "${UNSEEN_DIR:?HOLE_MODE=gavis needs UNSEEN_DIR (per-view unseen maps)}"
           [ -d "$UNSEEN_DIR" ] || { echo "UNSEEN_DIR $UNSEEN_DIR does not exist" >&2; exit 2; }
           UNSEEN_THRESH=${UNSEEN_THRESH:-0.5} ;;
    *) echo "unknown HOLE_MODE '$HOLE_MODE' (dilate, alpha, mesh, gavis)" >&2; exit 2 ;;
  esac
  if [ -n "${CONF_THRESH:-}" ]; then
    : "${CONFIDENCE_DIR:?CONF_THRESH needs CONFIDENCE_DIR (per-view confidence maps)}"
    [ -d "$CONFIDENCE_DIR" ] || { echo "CONFIDENCE_DIR $CONFIDENCE_DIR does not exist" >&2; exit 2; }
    [ "$HOLE_MODE" != dilate ] || echo "CONF_THRESH: HOLE_MODE=dilate keeps the whole dilated mask, the confidence adds nothing" >&2
  fi
  SEEDS=${SEEDS:-auto}; OUTSIDE=${OUTSIDE:-photo}; REFS=${REFS:-render}
  HULL_TRIM=${HULL_TRIM:-99.5}; HULL_EXPAND=${HULL_EXPAND:-0}; ANCHOR_EVERY=${ANCHOR_EVERY:-3}
  ALL_STAGES=prepare,removal,split,infer,propagate,handoff,train,plus,infer_plus
  RENDER_TRAJECTORY=val_frames
elif [ -n "$MODE" ]; then
  echo "unknown MODE '$MODE' (removal, or unset)" >&2; exit 2
elif [ -n "${TRAJ_MODE:-}" ]; then
  case $TRAJ_MODE in object|vidsplat|travel) ;; *) echo "unknown TRAJ_MODE '$TRAJ_MODE' (object, vidsplat, travel)" >&2; exit 2 ;; esac
  [ "$TRAJ_MODE" != object ] || : "${OBJECT_MASK:?TRAJ_MODE=object needs OBJECT_MASK}"
  TRAJ=$OUT/trajectory_input.json
  ALL_STAGES=trajectory,prepare,infer,handoff,train,plus,infer_plus
  RENDER_TRAJECTORY=trajectory
else
  ALL_STAGES=prepare,infer,handoff,train,plus,infer_plus
  if [ -n "${TRAJ:-}" ]; then RENDER_TRAJECTORY=trajectory; else RENDER_TRAJECTORY=val_frames; fi
fi
STAGES=${STAGES:-$ALL_STAGES}; STAGES=${STAGES//+/,}
if [ -n "${TRAJ:-}" ]; then
  PATH_ARGS=(--trajectory_path "$TRAJ")
else
  PATH_ARGS=(${ANCHORS:+--selected_image_names_file "$ANCHORS"})
fi
mkdir -p "$OUT/logs"
REPLACE=  # set per stage by run_stage
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/logs/run.log"; }
INFER=(-m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT"
       --model_id "$MODEL_ID" --num_views "$NUM_VIEWS" --max_neighbors_per_encode 1 --replace_if_exists
       ${INFER_ARGS:-})
pred_dir() {  # newest */frames/batch_0000/pred under $1 (empty if none)
  [ -d "$1" ] || return 0
  find "$1" -type d -path '*/frames/batch_0000/pred' -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
}
scale_file() {  # metres per unit of the prepared root, if known
  [ -f "$OUT/metric_alignment/scale_info.txt" ] || return 0
  sed -n 's/^Scale factor: *\([^ ]*\).*/\1/p' "$OUT/metric_alignment/scale_info.txt" | head -1
}
PREP=(-m artifixer_gs.prepare --ply "$PLY" --colmap_dir "$COLMAP" --images "$IMAGES" --output_root "$OUT"
      ${METRIC_SCALE:+--metric_scale "$METRIC_SCALE"} --text_encoder_model_id "$MODEL_ID")

traj_args() {  # artifixer_gs.trajectory arguments for TRAJ_MODE
  TRAJ_ARGS=(--ply "$PLY" --colmap_dir "$COLMAP" --images "$IMAGES" --output "$TRAJ" --mode "$TRAJ_MODE"
             --clearance_m "${TRAJ_CLEARANCE_M:-0.3}" --d0_m "${TRAJ_D0_M:-0.5}"
             ${METRIC_SCALE:+--metric_scale "$METRIC_SCALE"} ${TRAJ_MAX_FRAMES:+--max_frames "$TRAJ_MAX_FRAMES"})
  case $TRAJ_MODE in
    object)
      TRAJ_ARGS+=(--object_mask "$OBJECT_MASK" --orbit "${TRAJ_ORBIT:-0.6}" --radial "${TRAJ_RADIAL:-0.3}" --rise "${TRAJ_RISE:-0.3}")
      [ "${TRAJ_RENDER_CHECK:-0}" = 0 ] || TRAJ_ARGS+=(--render_check)
      [ "${TRAJ_ALLOW_FALLBACK:-0}" = 0 ] || TRAJ_ARGS+=(--allow_fallback) ;;
    vidsplat)
      TRAJ_ARGS+=(--clip_len "${TRAJ_CLIP:-16}" --orbit_deg "${TRAJ_ORBIT_DEG:-15,30,45}" --s_low "${TRAJ_S_LOW:-0.03}"
                  --s_high "${TRAJ_S_HIGH:-0.4}" --seeds "${TRAJ_SEEDS:-12}" ${TRAJ_BUDGET:+--budget "$TRAJ_BUDGET"}) ;;
    travel)
      TRAJ_ARGS+=(--side "${TRAJ_SIDE:-2}" --up "${TRAJ_UP:-0.5}") ;;
  esac
  TRAJ_ARGS+=(${TRAJ_EXTRA_ARGS:-})
}

# What each stage depends on: a finished stage reruns when this changes.
stage_settings() {
  case $1 in
    trajectory) traj_args; printf '%s\n' "${TRAJ_ARGS[@]}" "eps2d $EPS2D" ;;
    prepare)    echo "$PLY $COLMAP $IMAGES ${ANCHORS:-} ${TRAJ:-} ${METRIC_SCALE:-} ${CAPTION:-} ${EMPTY_CAPTION:-} $EPS2D$([ "$MODE" = removal ] || echo "${RENDER_CMD:+ render $RENDER_CMD}${CENTER_PRINCIPAL_POINT:+ cpp}")" ;;
    removal)    echo "$PLY $OBJECT_MASK $HULL_TRIM $HULL_EXPAND $EPS2D${RENDER_CMD:+ render $RENDER_CMD}${CENTER_PRINCIPAL_POINT:+ cpp} $([ "$HOLE_MODE" != mesh ] || echo "mesh $HOLE_MESH")" ;;
    split)      echo "$MASK_DIR ${MASK_LABEL:-} $HOLE_SHAPE $DILATE ${FRAMES:-} $REFS ${EMPTY_CAPTION:-} $HOLE_MODE${CONF_THRESH:+ conf $CONF_THRESH ${CONFIDENCE_DIR:-}}$([ "$HOLE_MODE" != gavis ] || echo " unseen $UNSEEN_THRESH $UNSEEN_DIR")" ;;
    infer)      echo "$NUM_VIEWS ${INFER_ARGS:-}" ;;
    propagate)  echo "$SEEDS $OUTSIDE $NUM_VIEWS" ;;
    handoff)    echo "${ANCHOR_EVERY:-} ${DROP_POINTS:-0}" ;;
    train)      echo "$AF3D_STEPS ${FASTGS_REPO:-} ${FASTGS_ARGS:-} ${TRAIN_CMD:-} ${AF3D_EXPORT:-}" ;;
    plus)       echo "$EPS2D${RENDER_CMD:+ render $RENDER_CMD}${CENTER_PRINCIPAL_POINT:+ cpp}" ;;
    *)          echo "" ;;
  esac
}

stage_trajectory() {
  local traj_py
  traj_args
  if "$PY" -c 'import gsplat' 2>/dev/null; then traj_py=$PY; else traj_py=${GS_RENDER_PY:-$PY}; fi
  echo "trajectory: $TRAJ_MODE, metric scale ${METRIC_SCALE:-<none: clearance from the object size; MoGe-2 runs in prepare>}, python $traj_py"
  "$traj_py" -m artifixer_gs.trajectory "${TRAJ_ARGS[@]}" \
    || { echo "artifixer_gs.trajectory found no acceptable $TRAJ_MODE path: see ${TRAJ%.json}_info.json and ${TRAJ%.json}.png"; return 1; }
}

stage_prepare() {
  local phases=prepare,render,scale caption
  [ "$MODE" != removal ] || phases=prepare,scale  # removal renders in its own stage
  caption=$(cd "$OUT" && pwd -P); caption=$OUT/captions/$(basename "$caption")/caption.h5
  if [ "$MODE" = removal ] && [ -n "$EMPTY_CAPTION" ]; then
    :  # bighull prompts with EMPTY_CAPTION instead (split stage)
  elif [ -z "${CAPTION:-}" ] && [ -f "$caption" ]; then
    echo "keeping caption $caption (no CAPTION given)"  # also under --replace: never re-caption a given caption.h5
  else
    phases+=,caption
  fi
  "$PY" "${PREP[@]}" --phases "$phases" ${PATH_ARGS[@]+"${PATH_ARGS[@]}"} ${CAPTION:+--caption "$CAPTION"} ${REPLACE:+--replace}
}

mesh_py() {  # a Python with nvdiffrast for artifixer_gs.hole_mesh
  local py
  for py in ${MESH_PY:-} "$PY" ${GS_RENDER_PY:-}; do
    if "$py" -c 'import nvdiffrast.torch' 2>/dev/null; then echo "$py"; return 0; fi
    [ -z "${MESH_PY:-}" ] || break  # an explicit MESH_PY must work
  done
  echo "HOLE_MODE=mesh needs nvdiffrast: none of ${MESH_PY:-PY ($PY) or GS_RENDER_PY (${GS_RENDER_PY:-unset})} imports it; set MESH_PY (e.g. the Gaussian Wrapping env) or use HOLE_MODE=alpha" >&2
  return 1
}

stage_removal() {
  local depth=() py
  if [ "$HOLE_MODE" = mesh ]; then py=$(mesh_py); depth=(--depth); fi
  "$PY" -m artifixer_gs.removal render --ply "$PLY" --object_mask "$OBJECT_MASK" --split_path "$OUT" \
    --output_dir "$OUT/removal" --hull_trim "$HULL_TRIM" --hull_expand "$HULL_EXPAND" ${depth[@]+"${depth[@]}"}
  rm -rf "$OUT/removal/mesh_front" "$OUT/removal/mesh_front.json"
  if [ "$HOLE_MODE" = mesh ]; then
    echo "hole mesh: $HOLE_MESH (python $py)"
    "$py" -m artifixer_gs.hole_mesh --mesh "$HOLE_MESH" --removal_dir "$OUT/removal"
    rm -rf "$OUT/removal/background/depth"  # only hole_mesh reads it
  fi
}

stage_split() {
  local prompt=()
  if [ -n "$EMPTY_CAPTION" ]; then
    echo "caption: $EMPTY_CAPTION"
    # caption.txt records the text of caption.h5: the text encoder only loads when the text changed.
    if [ -n "${FORCE:-}" ] || [ ! -f "$OUT/prompt/caption.h5" ] || [ "$(cat "$OUT/prompt/caption.txt" 2>/dev/null)" != "$EMPTY_CAPTION" ]; then
      rm -f "$OUT/prompt/caption.txt"
      "$PY" -m artifixer_gs.caption --caption "$EMPTY_CAPTION" --output_path "$OUT/prompt/caption.h5" \
        --text_encoder_model_id "$MODEL_ID"
      printf '%s\n' "$EMPTY_CAPTION" > "$OUT/prompt/caption.txt"
    fi
    prompt=(--prompt_path "$OUT/prompt/caption.h5")
  fi
  unseen=()
  [ "$HOLE_MODE" != gavis ] || unseen=(--unseen_dir "$UNSEEN_DIR" --unseen_threshold "$UNSEEN_THRESH")
  "$PY" -m artifixer_gs.removal split --scene_root "$OUT" --removal_dir "$OUT/removal" --mask_dir "$MASK_DIR" \
    ${MASK_LABEL:+--mask_label "$MASK_LABEL"} --output_dir "$OUT/removal_split" --hole_shape "$HOLE_SHAPE" \
    --dilate_px "$DILATE" --hole_mode "$HOLE_MODE" --frames "${FRAMES:-}" --refs "$REFS" --opacity hole0 \
    ${CONF_THRESH:+--confidence_dir "$CONFIDENCE_DIR" --conf_threshold "$CONF_THRESH"} ${prompt[@]+"${prompt[@]}"} \
    ${unseen[@]+"${unseen[@]}"}
}

stage_infer() {
  if [ "$MODE" = removal ]; then  # one rollout over every frame; references = object-free renders
    "$PY" "${INFER[@]}" --save_dir "$OUT/artifixer" --split_path "$OUT/removal_split/split.json" --render_trajectory all_frames
  else
    "$PY" "${INFER[@]}" --save_dir "$OUT/artifixer" --split_path "$OUT/split.json" --render_trajectory "$RENDER_TRAJECTORY"
  fi
}

stage_propagate() {
  local pred
  pred=$(pred_dir "$OUT/artifixer")
  [ -n "$pred" ] || { echo "no single-rollout output under $OUT/artifixer: run the infer stage first"; return 1; }
  echo "seed frames: $SEEDS (from $pred)"
  rm -rf "$OUT/propagation"
  "$PY" -m artifixer_gs.propagate --scene_root "$OUT/removal_split/scene" --removal_dir "$OUT/removal_split" \
    --seed_pred_dir "$pred" --seed_frames "${SEEDS//+/,}" --output_dir "$OUT/propagation" --outside "$OUTSIDE" \
    --num_views "$NUM_VIEWS" --max_neighbors_per_encode 1 --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID"
}

stage_handoff() {
  if [ "$MODE" = removal ]; then
    local drop=()
    [ "$DROP_POINTS" = 0 ] || drop=(--drop_points_mask)
    "$PY" -m artifixer_gs.distill removal_handoff --removal_split "$OUT/removal_split/split.json" \
      --propagated_dir "$OUT/propagation" --output_dir "$OUT/af3d/input" --anchor_every "$ANCHOR_EVERY" \
      ${drop[@]+"${drop[@]}"}
  else
    local pred
    pred=$(pred_dir "$OUT/artifixer")
    [ -n "$pred" ] || { echo "no ArtiFixer predictions under $OUT/artifixer"; return 1; }
    echo "ArtiFixer frames: $pred"
    "$PY" -m artifixer_gs.distill handoff --split_path "$OUT/split.json" --artifixer_frames_dir "$pred" \
      --output_dir "$OUT/af3d/input"
  fi
}

stage_train() {
  : "${FASTGS_REPO:?set FASTGS_REPO}"
  [ -n "${TRAIN_CMD:-}" ] || : "${FASTGS_PY:?set FASTGS_PY (or TRAIN_CMD)}"
  local train
  read -ra train <<< "${TRAIN_CMD:-$FASTGS_PY train.py}"
  # -r 1: keep the prepared resolution (FastGS otherwise shrinks images wider than 1600 px).
  (cd "$FASTGS_REPO" && PYTHONPATH= "${train[@]}" -s "$OUT/af3d/input" -m "$OUT/af3d/fastgs" \
    -r 1 --iterations "$AF3D_STEPS" --save_iterations "$AF3D_STEPS" --test_iterations "$AF3D_STEPS" \
    --lpips_images "$OUT/af3d/input/generated_images.txt" ${FASTGS_ARGS:-})
  if [ -n "${AF3D_EXPORT:-}" ]; then
    local exporter
    read -ra exporter <<< "$AF3D_EXPORT"
    (cd "$FASTGS_REPO" && PYTHONPATH= "${exporter[@]}" -m "$OUT/af3d/fastgs" --iteration "$AF3D_STEPS" --out "$AF3D_PLY")
  fi
}

if [ -n "${AF3D_EXPORT:-}" ]; then
  AF3D_PLY="$OUT/af3d/export/point_cloud.ply"  # the trained PLY is not standard 3DGS: AF3D_EXPORT bakes it
else
  AF3D_PLY="$OUT/af3d/fastgs/point_cloud/iteration_$AF3D_STEPS/point_cloud.ply"
fi
stage_plus() {
  local split=$OUT/split.json
  [ "$MODE" != removal ] || split=$OUT/af3d/input/split.json  # removal_handoff's ArtiFixer3D split
  "$PY" -m artifixer_gs.distill plus --split_path "$split" --ply "$AF3D_PLY" --render_dir "$OUT/af3d/renders" \
    --output_split "$OUT/split_artifixer3d_plus.json"
}

stage_infer_plus() {
  # ArtiFixer3D+: the second pass regenerates only the targets, conditioned on the ArtiFixer3D renders.
  # References: covisibility, except for TRAJ_MODE (ArtiFixer's vanilla_* use run_inference's default).
  local refs=(--neighbor_selection_mode covisibility)
  [ -z "${TRAJ_MODE:-}" ] || refs=()
  "$PY" "${INFER[@]}" --save_dir "$OUT/artifixer3d_plus" --split_path "$OUT/split_artifixer3d_plus.json" \
    --render_trajectory "$RENDER_TRAJECTORY" ${refs[@]+"${refs[@]}"}
}

run_stage() {  # run_stage NAME: stage_NAME with its own log, skipped when its marker is current
  local name=$1 t0=$SECONDS marker=$OUT/logs/.done_$1 settings status later after=0
  settings=$(stage_settings "$name")
  if [ -z "${FORCE:-}" ] && [ -f "$marker" ] && [ "$(cat "$marker")" = "$settings" ]; then
    log "skip $name (finished; FORCE=1 redoes it)"
    return 0
  fi
  # A marker with other settings (or 'stale') means outputs from other settings: redo them (--replace),
  # since prepare would otherwise keep its existing photos, renders and caption.
  REPLACE=${FORCE:-}
  [ ! -f "$marker" ] || REPLACE=1
  IFS=, read -ra later <<< "$ALL_STAGES"
  for s in "${later[@]}"; do  # the stages after this one build on its outputs: redo them too
    if [ "$after" = 1 ] && [ -f "$OUT/logs/.done_$s" ]; then echo stale > "$OUT/logs/.done_$s"; fi
    [ "$s" != "$name" ] || after=1
  done
  rm -f "$marker"
  log "start $name${REPLACE:+ (replacing its outputs)}"
  set +e
  ( set -euo pipefail; "stage_$name" ) 2>&1 | tee "$OUT/logs/$name.log"
  status=${PIPESTATUS[0]}
  set -e
  if [ "$status" -ne 0 ]; then
    log "FAILED $name (exit $status) after $((SECONDS - t0)) s; log $OUT/logs/$name.log"
    exit "$status"
  fi
  printf '%s\n' "$settings" > "$marker"
  log "done $name ($((SECONDS - t0)) s)"
}

IFS=, read -ra stage_list <<< "$STAGES"
for s in "${stage_list[@]}"; do
  [[ ",$ALL_STAGES," == *",$s,"* ]] || { echo "unknown stage '$s' for this mode (stages: $ALL_STAGES)" >&2; exit 2; }
done
log "run mode=${MODE:-${TRAJ_MODE:+trajectory:$TRAJ_MODE}} stages=$STAGES out=$OUT"
IFS=, read -ra all_list <<< "$ALL_STAGES"
for s in "${all_list[@]}"; do  # canonical order, whatever order STAGES lists them in
  # (an `if`, not `||`: a function called from a || list would run with set -e ignored)
  if [[ ",$STAGES," == *",$s,"* ]]; then run_stage "$s"; fi
done

log "outputs: ArtiFixer $(pred_dir "$OUT/artifixer")"
[ "$MODE" != removal ] || log "         propagated composites $OUT/propagation/anchors"
log "         ArtiFixer3D PLY $AF3D_PLY ($(cat "$OUT/af3d/fastgs/gs_stats.json" 2>/dev/null | tr -d '\n ' || echo 'no stats'))"
log "         ArtiFixer3D+ $(pred_dir "$OUT/artifixer3d_plus")"
