#!/usr/bin/env python3
"""
Prepare a TriGSVanilla-compatible NPZ using an exported reference JSON.

This is step 2 of a two-process workflow:
  1) export camera data from pycolmap into JSON in a pycolmap-only process
  2) run this script in a separate RoMa/OpenCV process

Expected dataset layout:
    data_dir/
        images/
            cam00/000001.jpg
            cam01/000001.jpg
            ...

Output NPZ fields:
  positions    [N, 3]
  colors       [N, 3]  float32 in [0, 1]
  velocities   [N, 3]
  times        [N, 1]  normalized to [0, 1]
  durations    [N, 1]
  has_velocity [N]

Pipeline upgrades vs. the original:
  * RoMa TEMPORAL matching within each camera across the keyframe step
    establishes true physical correspondences between frame t and t+step.
    Tracked points get velocity-by-construction (not spatial NN guessing).
  * Fresh cross-camera RoMa triangulation is still run at each keyframe,
    so the final cloud is the UNION of tracked points (priority-kept) and
    fresh points (filled in where tracking failed or is absent).
  * Spatial KNN velocity (paper-style) is used only as a FALLBACK for
    fresh points, with a tight adaptive radius.
  * 2-view candidates are gated by triangulation angle to kill baseline-
    aligned depth-uncertainty smears.
  * Mask filtering projects with full OpenCV distortion, per camera.
  * Camera pair list augments K-nearest with M-farthest wide baselines.
  * Per-point duration can be scaled down for fast-moving points so the
    trainer re-spawns Gaussians instead of stretching them along a wrong
    linear velocity vector.

Performance optimizations (no change to numerical outputs):
  * Fully assembled keyframes are atomically checkpointed to disk and released
    immediately. Restarts reuse valid checkpoints, and final NPZ construction
    streams through memory-mapped arrays instead of retaining every keyframe.
  * RoMa model runs in fp16 on CUDA.  Warp coordinates are cast back to
    fp32 before any math, so downstream results are bit-identical to fp32.
  * Camera intrinsic matrix (_K), distortion coefficients (_D),
    center_world, and extrinsic are all functools.cached_property —
    computed once per Camera instance and never again.
  * bilinear_sample_rgb is fully vectorised; color sampling for an entire
    frame is batched into one numpy call per camera (deferred until after
    the mask filter so rejected points never pay the sampling cost).
  * voxel_downsample uses np.bincount with weights — fully vectorised,
    zero Python loops.
  * Image and mask loading is parallelised with ThreadPoolExecutor.
  * SVD in triangulate_multiview_dlt passes full_matrices=False, skipping
    the large U matrix that is never used.

Dependencies:
  Core: numpy scipy pillow opencv-python torch tqdm
  Spatial backend: romatch (RoMa v1, default) or romav2 (RoMa v2)
  Optional temporal backend: memfof, selected with --temporal-backend memfof
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from rigid_motion import (
    TRACK_BWD,
    TRACK_FWD,
    RigidMotionParams,
    rigid_cluster_motion,
    track_flags_for_rows,
)
from motion_budget import MotionBudgetParams, keyframe_spacing, motion_budget_select, neighbourhood_speed
from roma_backends import (
    MEMFOF_DEFAULT_CHECKPOINT,
    DenseMatch,
    MemfofTemporalBackend,
    create_roma_backend,
)
from scipy.spatial import cKDTree
from tqdm import tqdm

import capture_timing_shim  # noqa: E402  (the ONE shared capture-time implementation, train/trigs/src/trigs/capture_timing.py)

_CT = capture_timing_shim.load()


def rs_tau_frames(residual_frames, sign, readout_frames, coords, extent, reference, content_ref=None):
    """Capture-time offset in FRAMES of each observation vs the canonical frame time.

    reference='midpoint' (shared convention, trigs.capture_timing): residual + sign*readout*clip(s/S - 0.5);
                         residual comes from the camera's MIDPOINT constant (--timing-json).
    reference='content'  (legacy, unchanged): residual + sign*readout*(s - content_ref)/S, where content_ref is
                         the flow-weighted mean coordinate of the moving content and residual comes from sync.json
                         (which absorbed that content's mean readout delay).
    s is the coordinate along the readout axis (raw image pixels; prep works on the raw images)."""
    coords = np.asarray(coords, dtype=np.float64)
    out = np.full(coords.shape, float(residual_frames), dtype=np.float64)
    if float(sign) == 0.0 or float(readout_frames) == 0.0:
        return out
    if reference == 'midpoint':
        frac = _CT.readout_fraction(coords, float(extent))
    elif reference == 'content':
        frac = (coords - float(content_ref)) / max(float(extent), 1.0)
    else:
        raise ValueError(f"rs reference must be 'midpoint' or 'content', got {reference!r}")
    return _CT.capture_offset_s(out, float(sign) * float(readout_frames), frac)


def _rs_axis_index(rs_axis: str) -> int:
    if rs_axis not in ('column', 'row'):
        raise ValueError(f"rs axis must be 'column' or 'row', got {rs_axis!r}")
    return 0 if rs_axis == 'column' else 1


# ---------------------------------------------------------------------------
# Crash-safe per-keyframe checkpointing
# ---------------------------------------------------------------------------


CACHE_SCHEMA_VERSION = 3
# Sub-keyframe frames are tracked over this many frames each way (their rows live ~1 frame).
SUBKEYFRAME_TRACK_DELTA = 1

CACHE_ARRAY_SPECS: Dict[str, Tuple[np.dtype, Tuple[int, ...]]] = {
    "positions": (np.dtype(np.float32), (3,)),
    "colors": (np.dtype(np.float32), (3,)),
    "velocities": (np.dtype(np.float32), (3,)),
    "times": (np.dtype(np.float32), (1,)),
    "durations": (np.dtype(np.float32), (1,)),
    "has_velocity": (np.dtype(bool), ()),
    "source_frames": (np.dtype(np.int32), (1,)),
    "is_static": (np.dtype(bool), ()),
    "dynamic_protect": (np.dtype(bool), ()),
    "dynamic_allowed": (np.dtype(bool), ()),
    "stage_keep": (np.dtype(bool), ()),
    # X(t+1) - 2 X(t) + X(t-1) per video frame^2, from --centred-velocity
    # tracking; zero with has_acceleration=False where t-1 was not tracked.
    "accelerations": (np.dtype(np.float32), (3,)),
    "has_acceleration": (np.dtype(bool), ()),
    # rigid_motion.TRACK_FWD / TRACK_BWD: which endpoints the tracker measured
    # (0 for untracked rows, including --untracked-velocity-from-tracked ones);
    # --rigid-cluster-motion adds RIGID_FWD / RIGID_BWD where it replaced one.
    "track_flags": (np.dtype(np.uint8), ()),
}


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_content_hash(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_signature(path: Path) -> Dict[str, object]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path.resolve()), "missing": True}
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "content_sha256": _file_content_hash(path),
    }


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_savez(path: Path, *, compressed: bool = False, **arrays: np.ndarray) -> None:
    """Write an NPZ without ever exposing a partially-written destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("wb") as f:
            if compressed:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def write_frame_checkpoint(
    path: Path,
    *,
    frame_idx: int,
    run_fingerprint: str,
    input_fingerprint: str,
    velocity_units: str,
    arrays: Mapping[str, np.ndarray],
    stats_delta: Mapping[str, object],
) -> None:
    n_rows = len(np.asarray(arrays["positions"]))
    payload: Dict[str, np.ndarray] = {}
    for name, (dtype, tail_shape) in CACHE_ARRAY_SPECS.items():
        arr = np.asarray(arrays[name])
        if arr.dtype != dtype:
            raise TypeError(f"{name}: expected dtype {dtype}, got {arr.dtype}")
        if arr.shape != (n_rows, *tail_shape):
            raise ValueError(
                f"{name}: expected shape {(n_rows, *tail_shape)}, got {arr.shape}"
            )
        payload[name] = arr
    payload.update(
        {
            "_schema": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int32),
            "_frame_idx": np.asarray(frame_idx, dtype=np.int32),
            "_run_fingerprint": np.asarray(run_fingerprint),
            "_input_fingerprint": np.asarray(input_fingerprint),
            "_velocity_units": np.asarray(velocity_units),
            "_stats_json": np.asarray(json.dumps(stats_delta, sort_keys=True)),
        }
    )
    _atomic_savez(path, compressed=False, **payload)


def load_frame_checkpoint(
    path: Path,
    *,
    frame_idx: int,
    run_fingerprint: str,
    input_fingerprint: str,
    load_arrays: bool = True,
    trust: bool = False,
) -> Optional[Dict[str, object]]:
    """Return None for incomplete, corrupt, stale, or schema-incompatible files.

    trust=True skips the run/input fingerprint equality checks (for reusing
    a cache across an edit that doesn't affect frame output, or across a
    change to how the fingerprints themselves are computed) while still
    rejecting schema mismatches and malformed arrays.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["_schema"]) != CACHE_SCHEMA_VERSION:
                return None
            if int(data["_frame_idx"]) != int(frame_idx):
                return None
            if not trust:
                if str(data["_run_fingerprint"]) != run_fingerprint:
                    return None
                if str(data["_input_fingerprint"]) != input_fingerprint:
                    return None
            n_rows = len(data["positions"])
            arrays: Dict[str, np.ndarray] = {}
            for name, (dtype, tail_shape) in CACHE_ARRAY_SPECS.items():
                arr = data[name]
                if arr.dtype != dtype or arr.shape != (n_rows, *tail_shape):
                    return None
                if load_arrays:
                    arrays[name] = np.asarray(arr)
            return {
                "n_rows": n_rows,
                "arrays": arrays,
                "velocity_units": str(data["_velocity_units"]),
                "stats_delta": json.loads(str(data["_stats_json"])),
                "bytes": int(path.stat().st_size),
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError):
        return None


def subkeyframe_fast_rows(
    arr: Mapping[str, np.ndarray],
    *,
    min_speed_spacings: float,
    spacing_points: int,
    knn: int,
    velocity_per_frame: bool = True,
    frame_delta: int = 1,
    select: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Indices of one frame's dynamic rows at or above min_speed_spacings.

    The motion budget's speed rule (motion_budget.py): neighbourhood-median speed
    over tracked rows, in spacings the frame would have at spacing_points rows.
    ``select`` restricts to those row indices (all of one frame)."""
    static = np.asarray(arr["is_static"]).reshape(-1) > 0.5
    rows = np.flatnonzero(~static) if select is None else np.asarray(select)[~static[np.asarray(select)]]
    if len(rows) < 2:
        return rows[:0]
    pos = np.asarray(arr["positions"][rows], dtype=np.float64)
    spd = np.linalg.norm(np.asarray(arr["velocities"][rows], dtype=np.float64), axis=1)
    if not velocity_per_frame:
        spd = spd / float(max(int(frame_delta), 1))
    hv = np.asarray(arr["has_velocity"][rows]).reshape(-1).astype(bool)
    spacing = keyframe_spacing(pos, int(spacing_points))
    nb = neighbourhood_speed(pos, spd, hv, int(knn)) / max(spacing, 1e-12)
    return rows[nb >= float(min_speed_spacings)]


def trackable_keyframes(frames: Sequence[int], frame_delta: int, input_frame_end: int) -> List[int]:
    """Keyframes whose tracking target t + frame_delta is an available image. The limit is the last
    source image (--input-frame-end), not the clip end: the clip's last keyframe tracks into frames
    past it. (Using --frame-end dropped the last keyframe, e.g. 126 of a 66-126 clip.)"""
    return [int(f) for f in frames if int(f) + int(frame_delta) <= int(input_frame_end)]


def plan_subkeyframe_frames(
    fast_counts: Mapping[int, int],
    *,
    min_rows: int,
    frame_end: int,
    frame_delta: int,
    frame_min: Optional[int] = None,
    frame_max: Optional[int] = None,
) -> Tuple[List[int], List[List[int]]]:
    """Frames between consecutive keyframes whose interval has a bounding keyframe
    with >= min_rows fast rows. Frames without a t+delta image are skipped."""
    kfs = sorted(int(k) for k in fast_counts)
    planned: List[int] = []
    intervals: List[List[int]] = []
    for a, b in zip(kfs[:-1], kfs[1:]):
        if max(int(fast_counts[a]), int(fast_counts[b])) < int(min_rows):
            continue
        added = [
            f for f in range(a + 1, b)
            if (frame_min is None or f >= int(frame_min))
            and (frame_max is None or f <= int(frame_max))
            and f + int(frame_delta) <= int(frame_end)
        ]
        if added:
            intervals.append([a, b])
            planned += added
    return planned, intervals


def subkeyframe_covered_keyframes(
    keyframes: Sequence[int], sub_frames, *, frame_step: int,
) -> List[int]:
    """Keyframes whose fast rows may take the 1-frame lifetime: every frame between this
    keyframe and each neighbouring keyframe is a sub-keyframe frame. A side with no
    neighbouring keyframe (sequence edge) does not count against it. Otherwise the fast
    rows keep frame_step, so they still cover the frames up to the next keyframe."""
    kfs = sorted(int(k) for k in keyframes)
    have = set(int(f) for f in sub_frames)
    out = []
    for i, k in enumerate(kfs):
        sides = []
        if i > 0:
            sides.append(range(kfs[i - 1] + 1, k))
        if i + 1 < len(kfs):
            sides.append(range(k + 1, kfs[i + 1]))
        if sides and all(all(f in have for f in side) for side in sides):
            out.append(k)
    return out


def append_subkeyframe_rows(
    arrays: Mapping[str, np.ndarray],
    keyframes: Sequence[int],
    sub_frames: Mapping[int, Mapping[str, np.ndarray]],
    *,
    frame_step: int,
    fast_rows_fn,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, Dict[str, int]]:
    """Final keyframe arrays + the fast rows of each sub-keyframe frame.

    Returns (arrays, local_stride, keyframe rows marked fast, rows added per sub frame).
    local_stride is 1 for fast keyframe rows and every added row, frame_step otherwise.
    Keyframe rows keep their order and come first."""
    sf_all = np.asarray(arrays["source_frames"]).reshape(-1)
    local_stride = np.full(len(sf_all), int(frame_step), dtype=np.int32)
    for k in subkeyframe_covered_keyframes(keyframes, sub_frames.keys(), frame_step=frame_step):
        local_stride[fast_rows_fn(arrays, np.flatnonzero(sf_all == int(k)))] = 1
    n_kf_fast = int((local_stride == 1).sum())
    parts: Dict[str, List[np.ndarray]] = {name: [np.asarray(arrays[name])] for name in arrays}
    per_sub: Dict[str, int] = {}
    for f in sorted(sub_frames):
        arr_f = sub_frames[f]
        keep = fast_rows_fn(arr_f, None)
        per_sub[str(int(f))] = int(len(keep))
        for name in arrays:
            parts[name].append(np.asarray(arr_f[name])[keep])
    n_sub = int(sum(per_sub.values()))
    out = {name: np.concatenate(v, axis=0) for name, v in parts.items()} if n_sub else {k: np.asarray(v) for k, v in arrays.items()}
    local_stride = np.concatenate([local_stride, np.ones(n_sub, dtype=np.int32)])
    return out, local_stride, n_kf_fast, per_sub


def build_run_fingerprint(args: argparse.Namespace) -> Tuple[str, Dict[str, object]]:
    operational = {
        "cache_dir",
        "no_resume",
        "log_memory",
        "stats_path",
        "output_path",
        "solve_offsets_out",
        "force_run_fingerprint",
        "trust_cache",
        "keep_cache",
    }
    # --rigid-* and --motion-budget-* only change finalize-time post-passes,
    # never a per-frame checkpoint, so tuning them must not invalidate the cache.
    # --motion-budget itself does (assembly then skips the per-frame cap), and
    # with it on, --target-points-per-frame is only used at finalize.
    finalize_only = {"target_points_per_frame"} if getattr(args, "motion_budget", False) else set()
    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in operational
        and key not in finalize_only
        and not key.startswith("rigid_")
        and not key.startswith("motion_budget_")
        # Sub-keyframe selection only decides WHICH frames are reconstructed;
        # each frame's checkpoint is the same as in a run without it.
        and not key.startswith("subkeyframe_")
    }
    script_path = Path(__file__).resolve()
    backend_path = script_path.with_name("roma_backends.py")
    descriptor: Dict[str, object] = {
        "schema": CACHE_SCHEMA_VERSION,
        "arguments": arguments,
        "script_sha256": _file_content_hash(script_path),
        "backend_sha256": _file_content_hash(backend_path),
        "reference_sha256": _file_content_hash(Path(args.reference_json)),
        "sync_sha256": (
            _file_content_hash(Path(args.sync_json)) if args.sync_json else None
        ),
        "depth_guide_plan_sha256": (
            _file_content_hash(Path(args.depth_guide_plan))
            if getattr(args, "depth_guide_plan", None)
            else None
        ),
        "rotation_sha256": (
            _file_content_hash(Path(args.rotation_json)) if args.rotation_json else None
        ),
    }
    if getattr(args, "timing_json", None):  # only when used, so legacy descriptors are unchanged
        descriptor["timing_sha256"] = _file_content_hash(Path(args.timing_json))
        descriptor["capture_timing_sha256"] = _file_content_hash(
            Path(capture_timing_shim._PATH))
    computed_fingerprint = _json_hash(descriptor)
    forced = getattr(args, "force_run_fingerprint", None)
    if forced:
        print(
            f"[cache] WARNING: --force-run-fingerprint pins the run fingerprint "
            f"to {forced[:16]} instead of the computed {computed_fingerprint[:16]}; "
            f"per-frame checkpoints are still validated against current "
            f"image/mask content, but the top-level script/argument/"
            f"reference-file identity check is bypassed."
        )
        return forced, descriptor
    return computed_fingerprint, descriptor


def _stats_snapshot(stats: Mapping[str, object]) -> Dict[str, object]:
    static = stats.get("static_dynamic", {})
    volume = stats.get("dynamic_volume_filter", {})
    temporal = stats.get("temporal_tracking", {})
    return {
        "per_frame_len": len(stats.get("per_frame", [])),
        "per_velocity_pair_len": len(stats.get("per_velocity_pair", [])),
        "temporal_pair_len": len(temporal.get("per_pair", [])),
        "static": {
            key: int(static.get(key, 0))
            for key in ("static", "dynamic", "protected")
        },
        "volume": {
            key: int(volume.get(key, 0))
            for key in (
                "dynamic_allowed",
                "outside_dynamic_allowed",
                "stage_keep",
                "outside_stage_keep",
            )
        },
    }


def _stats_delta(
    stats: Mapping[str, object], before: Mapping[str, object]
) -> Dict[str, object]:
    temporal = stats.get("temporal_tracking", {})
    now = _stats_snapshot(stats)
    return {
        "per_frame": list(stats.get("per_frame", []))[int(before["per_frame_len"]):],
        "per_velocity_pair": list(stats.get("per_velocity_pair", []))[
            int(before["per_velocity_pair_len"]):
        ],
        "temporal_pairs": list(temporal.get("per_pair", []))[
            int(before["temporal_pair_len"]):
        ],
        "static": {
            key: int(now["static"][key]) - int(before["static"][key])
            for key in now["static"]
        },
        "volume": {
            key: int(now["volume"][key]) - int(before["volume"][key])
            for key in now["volume"]
        },
    }


def _apply_stats_delta(stats: Dict[str, object], delta: Mapping[str, object]) -> None:
    stats["per_frame"].extend(delta.get("per_frame", []))
    stats["per_velocity_pair"].extend(delta.get("per_velocity_pair", []))
    stats["temporal_tracking"]["per_pair"].extend(delta.get("temporal_pairs", []))
    static = stats.setdefault(
        "static_dynamic", {"static": 0, "dynamic": 0, "protected": 0}
    )
    for key, value in delta.get("static", {}).items():
        static[key] = int(static.get(key, 0)) + int(value)
    volume = stats.setdefault(
        "dynamic_volume_filter",
        {
            "dynamic_allowed": 0,
            "outside_dynamic_allowed": 0,
            "stage_keep": 0,
            "outside_stage_keep": 0,
        },
    )
    for key, value in delta.get("volume", {}).items():
        volume[key] = int(volume.get(key, 0)) + int(value)


def _create_memmap_arrays(
    directory: Path, prefix: str, n_rows: int
) -> Dict[str, np.memmap]:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.memmap] = {}
    for name, (dtype, tail_shape) in CACHE_ARRAY_SPECS.items():
        arrays[name] = np.lib.format.open_memmap(
            directory / f"{prefix}-{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(n_rows, *tail_shape),
        )
    return arrays


def _flush_memmaps(arrays: Mapping[str, np.ndarray]) -> None:
    for arr in arrays.values():
        if isinstance(arr, np.memmap):
            arr.flush()


def _close_memmaps(*groups: Mapping[str, np.ndarray]) -> None:
    seen: set[int] = set()
    for group in groups:
        for arr in group.values():
            if not isinstance(arr, np.memmap) or id(arr) in seen:
                continue
            seen.add(id(arr))
            arr.flush()
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _cleanup_completed_cache(active_cache: Path, cache_root: Path) -> bool:
    """Best-effort cleanup after outputs are committed; never fail the run."""
    try:
        shutil.rmtree(active_cache)
    except OSError as exc:
        print(
            f"[cache] WARNING: output is complete, but cache cleanup failed: "
            f"{active_cache}: {exc}"
        )
        return False
    try:
        cache_root.rmdir()
    except OSError:
        pass
    print(f"[cache] removed completed run cache {active_cache}")
    return True


def _finish_completed_cache(
    active_cache: Path, cache_root: Path, *, keep_cache: bool
) -> bool:
    """Keep or remove a completed run cache according to the CLI policy."""
    if keep_cache:
        print(f"[cache] retained completed run cache {active_cache}")
        return True
    return _cleanup_completed_cache(active_cache, cache_root)


# ---------------------------------------------------------------------------
# Camera + observation containers
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    name: str
    folder: str
    R: np.ndarray            # [3, 3], world->camera
    t: np.ndarray            # [3], world->camera
    camera_model_name: str
    width: int
    height: int
    params: np.ndarray

    # ------------------------------------------------------------------
    # Cached geometric properties — computed once, never reallocated.
    # functools.cached_property stores results in the instance __dict__,
    # which is fully supported by (non-frozen) dataclasses.
    # ------------------------------------------------------------------

    @functools.cached_property
    def extrinsic(self) -> np.ndarray:
        return np.concatenate([self.R, self.t.reshape(3, 1)], axis=1)

    @functools.cached_property
    def center_world(self) -> np.ndarray:
        return -self.R.T @ self.t

    @functools.cached_property
    def _K(self) -> np.ndarray:
        """Camera intrinsic matrix [3, 3] float64."""
        m = self.camera_model_name.upper()
        p = self.params.astype(np.float64)
        if m == 'SIMPLE_PINHOLE':
            f, cx, cy = p[:3]
            fx = fy = f
        elif m in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV'):
            fx, fy, cx, cy = p[:4]
        elif m in ('SIMPLE_RADIAL', 'RADIAL'):
            f, cx, cy = p[:3]
            fx = fy = f
        else:
            raise NotImplementedError(
                f'Unsupported camera model for K: {self.camera_model_name}'
            )
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @functools.cached_property
    def _D(self) -> np.ndarray:
        """OpenCV distortion coefficient vector."""
        m = self.camera_model_name.upper()
        p = self.params.astype(np.float64)
        if m in ('SIMPLE_PINHOLE', 'PINHOLE'):
            return np.zeros((4, 1), dtype=np.float64)
        if m == 'SIMPLE_RADIAL':
            k1 = p[3]
            return np.array([[k1], [0.0], [0.0], [0.0]], dtype=np.float64)
        if m == 'RADIAL':
            k1, k2 = p[3:5]
            return np.array([[k1], [k2], [0.0], [0.0]], dtype=np.float64)
        if m == 'OPENCV':
            k1, k2, p1, p2 = p[4:8]
            return np.array([[k1], [k2], [p1], [p2]], dtype=np.float64)
        if m == 'FULL_OPENCV':
            return p[4:].reshape(-1, 1).astype(np.float64)
        if m == 'OPENCV_FISHEYE':
            return p[4:8].reshape(-1, 1).astype(np.float64)
        raise NotImplementedError(
            f'Unsupported camera model for distortion: {self.camera_model_name}'
        )

    def unproject_image_points(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        m = self.camera_model_name.upper()
        if m == 'OPENCV_FISHEYE':
            und = cv2.fisheye.undistortPoints(uv, self._K, self._D, P=None)
        else:
            und = cv2.undistortPoints(uv, self._K, self._D, P=None)
        return und.reshape(-1, 2)

    def undistort_image_points_pixel(self, uv: np.ndarray) -> np.ndarray:
        """Map distorted observations to this camera's ideal pixel plane.

        ``unproject_image_points`` returns normalized coordinates, which are
        ideal for calibrated triangulation.  Fundamental-matrix RANSAC uses a
        pixel threshold, so return the same rays expressed through K instead.
        """
        xy = self.unproject_image_points(uv)
        out = np.empty_like(xy, dtype=np.float64)
        out[:, 0] = self._K[0, 0] * xy[:, 0] + self._K[0, 2]
        out[:, 1] = self._K[1, 1] * xy[:, 1] + self._K[1, 2]
        return out

    def project_world_point(self, X_world: np.ndarray) -> Optional[np.ndarray]:
        xcam = self.R @ X_world + self.t
        if xcam[2] <= 1e-9:
            return None
        obj = xcam.reshape(1, 1, 3).astype(np.float64)
        zero_rvec = np.zeros((3, 1), dtype=np.float64)
        zero_tvec = np.zeros((3, 1), dtype=np.float64)
        m = self.camera_model_name.upper()
        try:
            if m == 'OPENCV_FISHEYE':
                uv, _ = cv2.fisheye.projectPoints(
                    obj, zero_rvec, zero_tvec, self._K, self._D
                )
            else:
                uv, _ = cv2.projectPoints(
                    obj, zero_rvec, zero_tvec, self._K, self._D
                )
        except cv2.error:
            return None
        uv = uv.reshape(-1, 2)[0]
        if not np.all(np.isfinite(uv)):
            return None
        return uv

    def project_world_points_batch(
        self, Xs_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch-project [N, 3] world points into this camera WITH distortion.
        Returns:
          uvs   [N, 2] float64 pixel coordinates (NaN where invalid)
          valid [N]    bool mask (depth > 0 AND projection finite)
        """
        N = len(Xs_world)
        if N == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

        xcam = (self.R @ Xs_world.T).T + self.t[None, :]  # [N, 3]
        valid_depth = xcam[:, 2] > 1e-6

        uvs = np.full((N, 2), np.nan, dtype=np.float64)

        if not valid_depth.any():
            return uvs, valid_depth

        xcam_valid = xcam[valid_depth].astype(np.float64)
        zero_rvec = np.zeros((3, 1), dtype=np.float64)
        zero_tvec = np.zeros((3, 1), dtype=np.float64)

        m = self.camera_model_name.upper()
        try:
            if m == 'OPENCV_FISHEYE':
                projected, _ = cv2.fisheye.projectPoints(
                    xcam_valid.reshape(-1, 1, 3),
                    zero_rvec, zero_tvec, self._K, self._D,
                )
            else:
                projected, _ = cv2.projectPoints(
                    xcam_valid.reshape(-1, 1, 3),
                    zero_rvec, zero_tvec, self._K, self._D,
                )
        except cv2.error:
            return uvs, np.zeros((N,), dtype=bool)

        projected = projected.reshape(-1, 2)
        finite = np.all(np.isfinite(projected), axis=1)

        valid_indices = np.where(valid_depth)[0][finite]
        uvs[valid_indices] = projected[finite]

        valid_mask = np.zeros(N, dtype=bool)
        valid_mask[valid_indices] = True
        return uvs, valid_mask


@dataclass
class FramePointCloud:
    frame_idx: int
    positions: np.ndarray
    colors: np.ndarray
    # Per-point list of (cam_idx, uv) observations for temporal flow warping.
    observations: List[List[Tuple[int, np.ndarray]]] = field(default_factory=list)
    # Actual integer source frame selected for each camera. This is needed to
    # compute the residual sub-frame capture time after nearest-frame selection.
    camera_frame_indices: Optional[np.ndarray] = None
    debug: Dict[str, float] = field(default_factory=dict)


@dataclass
class Observation:
    cam_idx: int
    uv: np.ndarray
    # Pixel shifted to the canonical keyframe time (sync motion compensation).
    # None means uncompensated; geometry falls back to `uv`. Color/mask/warp
    # sampling must always use `uv` — that is where the image content actually is.
    uv_canon: Optional[np.ndarray] = None
    # Local image motion magnitude in px per video frame (0 when unknown).
    flow_px: float = 0.0


def observation_geom_uv(obs: Observation) -> np.ndarray:
    return obs.uv if obs.uv_canon is None else obs.uv_canon


# ---------------------------------------------------------------------------
# Union-find for cross-camera match linking
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Data loading (parallelised with ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def load_reference_export(
    path: Path,
) -> Tuple[List[Camera], Optional[Dict[str, object]]]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    cameras: List[Camera] = []
    for item in data['cameras']:
        cameras.append(
            Camera(
                name=item['name'],
                folder=item['folder'],
                R=np.asarray(item['R'], dtype=np.float64),
                t=np.asarray(item['t'], dtype=np.float64),
                camera_model_name=str(item['camera_model_name']),
                width=int(item['width']),
                height=int(item['height']),
                params=np.asarray(item['params'], dtype=np.float64),
            )
        )
    return cameras, data.get('scene_meta')


def load_depth_guide_plan(
    path: Path,
    cameras: Sequence[Camera],
    reference_json: Path,
) -> Dict[str, object]:
    """Load an automatic depth coverage plan without importing depth geometry.

    The plan may select only crop rectangles and camera pairs. Its reference
    hash and camera identities are checked here so stale plans cannot silently
    feed crops from a different calibration into triangulation.
    """
    with path.open('r', encoding='utf-8') as handle:
        plan = json.load(handle)
    if plan.get('format') != 'trigs-depth-guided-roma-plan-v1':
        raise ValueError(
            f"Unsupported --depth-guide-plan format: {plan.get('format')!r}"
        )
    expected_hash = _file_content_hash(reference_json)
    if plan.get('reference_sha256') != expected_hash:
        raise ValueError(
            '--depth-guide-plan was built from a different reference JSON'
        )
    entries = plan.get('entries')
    if not isinstance(entries, list):
        raise ValueError('--depth-guide-plan entries must be a list')
    normalized: List[Dict[str, object]] = []
    for row_idx, row in enumerate(entries):
        if not isinstance(row, dict):
            raise ValueError(f'depth guide entry {row_idx} is not an object')
        cam_a = int(row['cam_a'])
        cam_b = int(row['cam_b'])
        if not (0 <= cam_a < len(cameras) and 0 <= cam_b < len(cameras)):
            raise ValueError(f'depth guide entry {row_idx} has invalid cameras')
        if cam_a == cam_b:
            raise ValueError(f'depth guide entry {row_idx} pairs a camera to itself')
        if str(row.get('camera_a')) != str(cameras[cam_a].folder):
            raise ValueError(f'depth guide entry {row_idx} camera_a mismatch')
        if str(row.get('camera_b')) != str(cameras[cam_b].folder):
            raise ValueError(f'depth guide entry {row_idx} camera_b mismatch')
        normalized_row: Dict[str, object] = dict(row)
        for key, cam_idx in (('crop_a', cam_a), ('crop_b', cam_b)):
            crop = [int(v) for v in row[key]]
            if len(crop) != 4:
                raise ValueError(f'depth guide entry {row_idx} {key} is not xyxy')
            x0, y0, x1, y1 = crop
            camera = cameras[cam_idx]
            if not (
                0 <= x0 < x1 <= camera.width
                and 0 <= y0 < y1 <= camera.height
                and x1 - x0 >= 8
                and y1 - y0 >= 8
            ):
                raise ValueError(
                    f'depth guide entry {row_idx} {key} is outside '
                    f'{camera.width}x{camera.height}: {crop}'
                )
            normalized_row[key] = crop
        normalized_row['cam_a'] = cam_a
        normalized_row['cam_b'] = cam_b
        normalized.append(normalized_row)
    plan['entries'] = normalized
    plan['path'] = str(path)
    return plan


def image_path(
    images_root: Path, cam: Camera, frame_idx: int, image_pattern: str
) -> Path:
    return images_root / cam.folder / image_pattern.format(frame=frame_idx)


def load_frame_images(
    images_root: Path,
    cameras: Sequence[Camera],
    frame_idx: int,
    image_pattern: str,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Load all camera images for a single frame in parallel."""
    paths = [image_path(images_root, cam, frame_idx, image_pattern) for cam in cameras]

    def _load(p: Path) -> np.ndarray:
        if not p.exists():
            raise FileNotFoundError(f'Missing image for frame {frame_idx}: {p}')
        return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)

    n_workers = min(len(cameras), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        images = list(ex.map(_load, paths))
    return paths, images


# ---------------------------------------------------------------------------
# Camera pair selection (nearest + farthest baselines)
# ---------------------------------------------------------------------------


def choose_camera_pairs(
    cameras: Sequence[Camera],
    neighbors_per_camera: int,
    wide_baseline_per_camera: int = 0,
) -> List[Tuple[int, int]]:
    """
    K nearest neighbours per camera (short baseline, reliable matching) plus
    M farthest cameras per camera (wide baseline, strong depth constraint).
    The wide-baseline pairs are what kill the baseline-aligned smear.
    """
    n = len(cameras)
    if n < 2:
        return []

    if neighbors_per_camera <= 0 or neighbors_per_camera >= (n - 1):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    # center_world is now a cached_property — free after first access.
    centers = np.stack([cam.center_world for cam in cameras], axis=0)
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)

    pairs = set()
    k_near = min(neighbors_per_camera, n - 1)
    for i in range(n):
        order = np.argsort(dists[i])
        for j in order[:k_near]:
            a, b = sorted((i, int(j)))
            pairs.add((a, b))

    if wide_baseline_per_camera > 0:
        for i in range(n):
            finite = np.where(np.isfinite(dists[i]))[0]
            if len(finite) == 0:
                continue
            order = finite[np.argsort(-dists[i, finite])]
            m_far = min(wide_baseline_per_camera, len(order))
            for j in order[:m_far]:
                a, b = sorted((i, int(j)))
                pairs.add((a, b))

    return sorted(pairs)


def choose_detail_sphere_camera_pairs(
    cameras: Sequence[Camera],
    center: np.ndarray,
    *,
    min_image_margin_px: float = 0.0,
    max_view_angle_deg: float = 180.0,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Build an overlap-aware pair graph among cameras that see a detail center.

    The normal pair graph is based on camera-center distance. That works for
    scene regions visible in most cameras, but fails for a small detail region
    seen only by a sparse mix of full-body and close-up cameras. Projecting the
    requested world-space detail center identifies cameras with image coverage.

    Merely seeing the center is not sufficient for a convex detail such as a
    head: cameras on opposite sides see disjoint surface patches, and dense
    semantic matching can confidently associate different physical points.
    Restricting the angle between center-to-camera rays approximates common
    surface visibility. 180 degrees preserves the old complete-graph behavior.
    """
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    margin = max(float(min_image_margin_px), 0.0)
    visible: List[int] = []
    for cam_idx, camera in enumerate(cameras):
        uv = camera.project_world_point(center_arr)
        if uv is None or not np.all(np.isfinite(uv)):
            continue
        u, v = float(uv[0]), float(uv[1])
        if (
            margin <= u < float(camera.width) - margin
            and margin <= v < float(camera.height) - margin
        ):
            visible.append(int(cam_idx))

    max_angle = float(np.clip(max_view_angle_deg, 0.0, 180.0))
    camera_rays: Dict[int, np.ndarray] = {}
    for cam_idx in visible:
        ray = np.asarray(cameras[cam_idx].center_world, dtype=np.float64) - center_arr
        norm = float(np.linalg.norm(ray))
        if norm > 1e-9:
            camera_rays[int(cam_idx)] = ray / norm

    pairs: List[Tuple[int, int]] = []
    for i in range(len(visible)):
        for j in range(i + 1, len(visible)):
            cam_a, cam_b = int(visible[i]), int(visible[j])
            ray_a, ray_b = camera_rays.get(cam_a), camera_rays.get(cam_b)
            if ray_a is None or ray_b is None:
                continue
            angle_deg = math.degrees(
                math.acos(float(np.clip(ray_a @ ray_b, -1.0, 1.0)))
            )
            if angle_deg <= max_angle + 1e-9:
                pairs.append((cam_a, cam_b))
    return visible, pairs


def project_detail_sphere_crop_bounds(
    camera: Camera,
    center: np.ndarray,
    radius: float,
    *,
    scale: float = 1.25,
    min_size_px: int = 256,
) -> Optional[Tuple[int, int, int, int]]:
    """Return a square image crop covering a projected world-space sphere.

    The surface is sampled densely enough to account for perspective and lens
    distortion.  A square crop is shifted inside the image rather than merely
    clipped at an edge, preserving useful context for centers near the frame
    boundary.
    """
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    center_uv = camera.project_world_point(center_arr)
    if center_uv is None or not np.all(np.isfinite(center_uv)):
        return None
    if not (
        0.0 <= float(center_uv[0]) < float(camera.width)
        and 0.0 <= float(center_uv[1]) < float(camera.height)
    ):
        return None

    # Fibonacci sphere plus the center. The guard discards pathological lens
    # projections far outside the sensor while retaining partially visible
    # spheres at image edges.
    n_surface = 96
    idx = np.arange(n_surface, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_surface)
    phi = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    xy_radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    directions = np.stack(
        [xy_radius * np.cos(phi), xy_radius * np.sin(phi), z], axis=1
    )
    surface = center_arr[None, :] + float(radius) * directions
    uvs, valid = camera.project_world_points_batch(surface)
    guard = valid & np.all(np.isfinite(uvs), axis=1)
    guard &= (uvs[:, 0] >= -float(camera.width))
    guard &= (uvs[:, 0] <= 2.0 * float(camera.width))
    guard &= (uvs[:, 1] >= -float(camera.height))
    guard &= (uvs[:, 1] <= 2.0 * float(camera.height))
    projected = uvs[guard]
    if len(projected) == 0:
        projected = np.asarray(center_uv, dtype=np.float64).reshape(1, 2)
    else:
        projected = np.concatenate(
            [projected, np.asarray(center_uv, dtype=np.float64).reshape(1, 2)],
            axis=0,
        )

    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    crop_center = 0.5 * (lo + hi)
    span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    size = int(math.ceil(max(span * max(float(scale), 1.0), float(min_size_px))))
    size = max(8, min(size, int(camera.width), int(camera.height)))

    x0 = int(round(float(crop_center[0]) - 0.5 * size))
    y0 = int(round(float(crop_center[1]) - 0.5 * size))
    x0 = min(max(x0, 0), int(camera.width) - size)
    y0 = min(max(y0, 0), int(camera.height) - size)
    return x0, y0, x0 + size, y0 + size


# ---------------------------------------------------------------------------
# RoMa match wrappers
# ---------------------------------------------------------------------------


def _apply_masks_to_certainty(
    certainty: torch.Tensor,
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    mask_min_fg_alpha: float,
    device: str,
) -> torch.Tensor:
    """
    Multiply RoMa's certainty tensor by foreground masks so the sampler
    strongly prefers matches whose A-pixel AND B-pixel land on the subject.
    """
    c = _certainty_2d(certainty)   # [H_warp, 2*W_warp]
    H_warp = c.shape[0]
    W2 = c.shape[1]
    W_warp = W2 // 2

    fg_floor = 1e-4

    def _build_mask_factor(mask_np: np.ndarray, H: int, W: int) -> torch.Tensor:
        resized = cv2.resize(
            mask_np.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR,
        )
        t = torch.from_numpy(resized).to(device=c.device, dtype=c.dtype)
        return torch.where(t >= mask_min_fg_alpha, t, torch.full_like(t, fg_floor))

    c_a = c[:, :W_warp] * _build_mask_factor(mask_a, H_warp, W_warp) if mask_a is not None else c[:, :W_warp]
    c_b = c[:, W_warp:] * _build_mask_factor(mask_b, H_warp, W_warp) if mask_b is not None else c[:, W_warp:]

    c_weighted = torch.cat([c_a, c_b], dim=1)

    if certainty.dim() == 3 and certainty.shape[0] == 1:
        return c_weighted.unsqueeze(0)
    return c_weighted


def run_roma_pair(
    model,
    path_a: Path,
    path_b: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    num_matches: int,
    min_certainty: float,
    max_ransac_reproj_px: float,
    min_inliers: int,
    device: str,
    use_fundamental_ransac: bool = True,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    mask_min_fg_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-camera matching through the selected RoMa backend."""
    del device  # Backend owns its device.
    empty_kpts = np.empty((0, 2), np.float32)
    empty_cert = np.empty((0,), np.float32)

    try:
        kpts_a, kpts_b, cert = model.match_points(
            path_a,
            path_b,
            image_a,
            image_b,
            num=int(num_matches),
            mask_a=mask_a,
            mask_b=mask_b,
            mask_min_fg_alpha=float(mask_min_fg_alpha),
        )
    except Exception as e:
        print(f'  [warn] RoMa pair failed {path_a.name} <-> {path_b.name}: {e}')
        return empty_kpts, empty_kpts, empty_cert

    kpts_a = np.asarray(kpts_a, dtype=np.float32)
    kpts_b = np.asarray(kpts_b, dtype=np.float32)
    cert = np.asarray(cert, dtype=np.float32).reshape(-1)

    keep = cert >= float(min_certainty)
    keep &= np.all(np.isfinite(kpts_a), axis=1)
    keep &= np.all(np.isfinite(kpts_b), axis=1)
    keep &= np.isfinite(cert)
    kpts_a = np.ascontiguousarray(kpts_a[keep], dtype=np.float32)
    kpts_b = np.ascontiguousarray(kpts_b[keep], dtype=np.float32)
    cert = np.ascontiguousarray(cert[keep], dtype=np.float32)

    if len(kpts_a) < max(8, min_inliers):
        return empty_kpts, empty_kpts, empty_cert

    pair_key = np.round(np.concatenate([kpts_a, kpts_b], axis=1), decimals=2)
    _, unique_idx = np.unique(pair_key, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    kpts_a = np.ascontiguousarray(kpts_a[unique_idx], dtype=np.float32)
    kpts_b = np.ascontiguousarray(kpts_b[unique_idx], dtype=np.float32)
    cert = np.ascontiguousarray(cert[unique_idx], dtype=np.float32)

    if len(kpts_a) < max(8, min_inliers):
        return empty_kpts, empty_kpts, empty_cert

    if use_fundamental_ransac:
        Fm = None
        mask = None
        try:
            Fm, mask = cv2.findFundamentalMat(
                kpts_a,
                kpts_b,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=max_ransac_reproj_px,
                confidence=0.999,
                maxIters=10000,
            )
        except cv2.error as e:
            print(f'  [warn] USAC_MAGSAC failed for {path_a.name}<->{path_b.name}: {e}')
            try:
                Fm, mask = cv2.findFundamentalMat(
                    kpts_a,
                    kpts_b,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=max_ransac_reproj_px,
                    confidence=0.999,
                )
            except cv2.error as e2:
                print(f'  [warn] FM_RANSAC fallback failed: {e2}')
                return empty_kpts, empty_kpts, empty_cert

        if Fm is None or mask is None or np.asarray(Fm).size == 0:
            return empty_kpts, empty_kpts, empty_cert
        inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
        if len(inlier_mask) != len(kpts_a):
            return empty_kpts, empty_kpts, empty_cert
        kpts_a = np.ascontiguousarray(kpts_a[inlier_mask], dtype=np.float32)
        kpts_b = np.ascontiguousarray(kpts_b[inlier_mask], dtype=np.float32)
        cert = np.ascontiguousarray(cert[inlier_mask], dtype=np.float32)
        if len(kpts_a) < min_inliers:
            return empty_kpts, empty_kpts, empty_cert

    return kpts_a, kpts_b, cert


def sample_flow_per_frame(
    model,
    dense: Optional[DenseMatch],
    uvs: np.ndarray,
    temporal_frame_delta: int,
    min_cert: float = 0.05,
) -> np.ndarray:
    """Local image velocity at `uvs` in px per video frame, from a t->t+delta
    dense warp of the same camera. Rows where the warp is unusable fall back
    to zero flow (== no compensation, no slack)."""
    flow = np.zeros((len(uvs), 2), dtype=np.float64)
    if dense is None or len(uvs) == 0:
        return flow
    uvs64 = np.asarray(uvs, dtype=np.float64)
    uvs_b, certs = sample_warp_at_uvs(model, dense, np.asarray(uvs, dtype=np.float32))
    uvs_b = np.asarray(uvs_b, dtype=np.float64)
    certs = np.asarray(certs, dtype=np.float64).reshape(-1)
    ok = np.isfinite(uvs_b).all(axis=1) & np.isfinite(certs) & (certs >= float(min_cert))
    ok &= (uvs_b[:, 0] >= 0) & (uvs_b[:, 0] <= dense.w_b - 1)
    ok &= (uvs_b[:, 1] >= 0) & (uvs_b[:, 1] <= dense.h_b - 1)
    flow[ok] = (uvs_b[ok] - uvs64[ok]) / float(max(int(temporal_frame_delta), 1))
    return flow


def symmetric_epipolar_distance_px(
    F: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> np.ndarray:
    """max(point-to-epiline distance in image b, same in image a), in pixels.
    Follows the cv2.findFundamentalMat convention x_b^T F x_a = 0."""
    a = np.concatenate([pts_a, np.ones((len(pts_a), 1))], axis=1)
    b = np.concatenate([pts_b, np.ones((len(pts_b), 1))], axis=1)
    lines_b = a @ F.T   # epilines in image b
    lines_a = b @ F     # epilines in image a
    num = np.abs(np.sum(b * lines_b, axis=1))
    d_b = num / np.maximum(np.linalg.norm(lines_b[:, :2], axis=1), 1e-12)
    d_a = num / np.maximum(np.linalg.norm(lines_a[:, :2], axis=1), 1e-12)
    return np.maximum(d_a, d_b)


def calibrated_fundamental(camera_a: Camera, camera_b: Camera) -> np.ndarray:
    """Fundamental matrix on the cameras' ideal, undistorted pixel planes."""
    r_rel = camera_b.R @ camera_a.R.T
    t_rel = camera_b.t - r_rel @ camera_a.t
    tx, ty, tz = np.asarray(t_rel, dtype=np.float64).reshape(3)
    skew_t = np.asarray(
        [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]],
        dtype=np.float64,
    )
    essential = skew_t @ r_rel
    fundamental = np.linalg.inv(camera_b._K).T @ essential @ np.linalg.inv(camera_a._K)
    norm = float(np.linalg.norm(fundamental))
    return fundamental / max(norm, 1e-12)


def calibrated_pair_epipolar_distance_px(
    camera_a: Camera,
    camera_b: Camera,
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    """Score raw image correspondences against the supplied camera model."""
    ideal_a = camera_a.undistort_image_points_pixel(points_a)
    ideal_b = camera_b.undistort_image_points_pixel(points_b)
    fundamental = calibrated_fundamental(camera_a, camera_b)
    return symmetric_epipolar_distance_px(fundamental, ideal_a, ideal_b)


def calibrated_gate_px(
    base_px: float,
    flow_mag: Optional[np.ndarray],
    motion_slack: float,
    motion_slack_max_px: float,
) -> np.ndarray | float:
    """Per-match tolerance for the calibrated epipolar check.

    Same rule as the pair-RANSAC rescue: matches on content moving more than
    1 px/frame get `motion_slack * flow` extra pixels, capped at
    `motion_slack_max_px`; everything else keeps `base_px`. A fixed gate
    rejects fast limbs that are only a few ms apart between cameras even with
    a good sync (example capture keyframe 321: the swinging hand survived
    RANSAC and was then removed here)."""
    if flow_mag is None or motion_slack <= 0.0:
        return float(base_px)
    flow = np.asarray(flow_mag, dtype=np.float64)
    slack = np.where(
        flow > 1.0, np.minimum(motion_slack * flow, motion_slack_max_px), 0.0
    )
    return float(base_px) + slack


def pair_ransac_with_motion_slack(
    kpts_a_geom: np.ndarray,
    kpts_b_geom: np.ndarray,
    flow_mag: Optional[np.ndarray],
    max_ransac_reproj_px: float,
    min_inliers: int,
    motion_slack: float,
    motion_slack_max_px: float,
    pair_label: str = '',
    camera_a: Optional[Camera] = None,
    camera_b: Optional[Camera] = None,
) -> Optional[Tuple[np.ndarray, int]]:
    """Fundamental RANSAC on (possibly sync-compensated) match coordinates,
    then readmit fast-moving rejects whose epipolar residual fits within a
    flow-scaled slack. The slack only opens up where the image actually moves
    (>1 px/frame), so static matches keep the strict gate.

    When cameras are supplied, fit and score the epipolar model on their
    undistorted ideal pixel planes.  A fundamental matrix is not valid over
    raw radially distorted coordinates; using raw pixels disproportionately
    rejects matches near image edges (often faces/heads in close camera rigs).

    Returns (keep_mask, n_rescued), or None when the pair fails entirely."""
    ransac_a = np.asarray(kpts_a_geom, dtype=np.float64)
    ransac_b = np.asarray(kpts_b_geom, dtype=np.float64)
    if camera_a is not None and camera_b is not None:
        try:
            undistorted_a = camera_a.undistort_image_points_pixel(ransac_a)
            undistorted_b = camera_b.undistort_image_points_pixel(ransac_b)
            ransac_a, ransac_b = undistorted_a, undistorted_b
        except (cv2.error, ValueError, NotImplementedError) as e:
            print(
                f'  [warn] distortion-aware pair geometry failed for '
                f'{pair_label}; using raw pixels: {e}'
            )

    Fm = None
    mask = None
    try:
        Fm, mask = cv2.findFundamentalMat(
            ransac_a.astype(np.float32),
            ransac_b.astype(np.float32),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=max_ransac_reproj_px,
            confidence=0.999,
            maxIters=10000,
        )
    except cv2.error as e:
        print(f'  [warn] USAC_MAGSAC failed for {pair_label}: {e}')
        try:
            Fm, mask = cv2.findFundamentalMat(
                ransac_a.astype(np.float32),
                ransac_b.astype(np.float32),
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=max_ransac_reproj_px,
                confidence=0.999,
            )
        except cv2.error as e2:
            print(f'  [warn] FM_RANSAC fallback failed: {e2}')
            return None

    if Fm is None or mask is None or np.asarray(Fm).size == 0:
        return None
    keep = np.asarray(mask).reshape(-1).astype(bool)
    if len(keep) != len(kpts_a_geom):
        return None

    n_rescued = 0
    if motion_slack > 0.0 and flow_mag is not None:
        F = np.asarray(Fm, dtype=np.float64)[:3, :3]
        slack = np.minimum(motion_slack * flow_mag, motion_slack_max_px)
        candidates = (~keep) & (flow_mag > 1.0)
        if candidates.any():
            d = symmetric_epipolar_distance_px(
                F, ransac_a[candidates], ransac_b[candidates]
            )
            rescued_local = d <= (max_ransac_reproj_px + slack[candidates])
            idx = np.flatnonzero(candidates)[rescued_local]
            keep[idx] = True
            n_rescued = int(len(idx))

    if int(keep.sum()) < min_inliers:
        return None
    return keep, n_rescued


def build_roma_warp(
    model,
    path_a: Path,
    path_b: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    device: str,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    mask_min_fg_alpha: float = 0.5,
) -> DenseMatch:
    del device
    return model.build_dense(
        path_a,
        path_b,
        image_a,
        image_b,
        mask_a=mask_a,
        mask_b=mask_b,
        mask_min_fg_alpha=float(mask_min_fg_alpha),
    )


def sample_warp_at_uvs(
    model,
    dense: DenseMatch,
    uvs_a: np.ndarray,
    *unused,
    reverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    del unused
    return model.sample_at_uvs(dense, uvs_a, reverse=reverse)


def validate_warp_layout(
    model,
    dense: DenseMatch,
    *unused,
    tolerance_px: float = 3.0,
) -> bool:
    del unused
    try:
        kpts_a, kpts_b, _ = model.sample_points_from_dense(dense, num=128)
        our_b, _ = model.sample_at_uvs(dense, kpts_a, reverse=False)
    except Exception as e:
        print(f'  [warp-validate] backend validation raised ({e}); skipping validation.')
        return True
    if len(kpts_a) == 0:
        return True
    diffs = np.linalg.norm(our_b - kpts_b, axis=1)
    median_diff = float(np.median(diffs))
    max_diff = float(np.max(diffs))
    print(
        f'  [warp-validate] backend={getattr(model, "name", "unknown")} '
        f'median={median_diff:.3f}px max={max_diff:.3f}px '
        f'tol={tolerance_px:.1f}px {"OK" if median_diff < tolerance_px else "FAIL"}'
    )
    return median_diff < tolerance_px


# ---------------------------------------------------------------------------
# Multi-view components, triangulation, geometry checks
# ---------------------------------------------------------------------------


def build_multiview_components(
    pair_matches: Sequence[
        Tuple[int, int, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    ],
    same_camera_merge_px: float,
) -> List[List[Observation]]:
    observations: List[Observation] = []
    pair_edges: List[Tuple[int, int]] = []
    obs_ids_per_camera: Dict[int, List[int]] = {}
    for cam_a, cam_b, kpts_a, kpts_b, canon_a, canon_b, flow_a, flow_b in pair_matches:
        for k in range(len(kpts_a)):
            id_a = len(observations)
            observations.append(Observation(
                cam_idx=cam_a,
                uv=kpts_a[k].astype(np.float64),
                uv_canon=None if canon_a is None else canon_a[k].astype(np.float64),
                flow_px=0.0 if flow_a is None else float(flow_a[k]),
            ))
            id_b = len(observations)
            observations.append(Observation(
                cam_idx=cam_b,
                uv=kpts_b[k].astype(np.float64),
                uv_canon=None if canon_b is None else canon_b[k].astype(np.float64),
                flow_px=0.0 if flow_b is None else float(flow_b[k]),
            ))
            pair_edges.append((id_a, id_b))
            obs_ids_per_camera.setdefault(cam_a, []).append(id_a)
            obs_ids_per_camera.setdefault(cam_b, []).append(id_b)
    if not observations:
        return []
    uf = UnionFind(len(observations))
    for a, b in pair_edges:
        uf.union(a, b)
    radius_sq = float(same_camera_merge_px) ** 2
    for cam_idx, obs_ids in obs_ids_per_camera.items():
        if len(obs_ids) < 2:
            continue
        uv = np.stack([observations[i].uv for i in obs_ids], axis=0)
        tree = cKDTree(uv)
        for i, j in tree.query_pairs(float(same_camera_merge_px)):
            da = uv[i] - uv[j]
            if float(da @ da) <= radius_sq:
                uf.union(obs_ids[i], obs_ids[j])
    components: Dict[int, List[Observation]] = {}
    for idx, obs in enumerate(observations):
        root = uf.find(idx)
        components.setdefault(root, []).append(obs)
    return list(components.values())


def average_observations_per_camera(
    component: Sequence[Observation],
) -> List[Observation]:
    grouped: Dict[int, List[Observation]] = {}
    for obs in component:
        grouped.setdefault(obs.cam_idx, []).append(obs)
    reduced: List[Observation] = []
    for cam_idx, obs_list in grouped.items():
        uv = np.mean(np.stack([o.uv for o in obs_list], axis=0), axis=0)
        any_canon = any(o.uv_canon is not None for o in obs_list)
        uv_canon = (
            np.mean(np.stack([observation_geom_uv(o) for o in obs_list], axis=0), axis=0)
            if any_canon
            else None
        )
        flow_px = float(np.mean([o.flow_px for o in obs_list]))
        reduced.append(
            Observation(cam_idx=cam_idx, uv=uv, uv_canon=uv_canon, flow_px=flow_px)
        )
    return reduced

def solve_camera_offsets(
    keyframe_records,          # list of {"cloud": FramePointCloud, "X":[M,3], "V":[M,3], "src_idx":[M]}
    cameras,
    fps: float,
    *,
    min_image_speed_px: float = 0.05,  # a view with <this px/frame motion carries no timing signal
    min_cam_obs: int = 30,
    min_views: int = 2,
    huber_px: float = 2.0,
    n_iters: int = 5,
    ridge_frac: float = 1e-2,
    fd_eps_frac: float = 1e-3,
):
    """Returns per-camera offset in SECONDS (mean-zero). V must be world-units-per-video-frame."""
    import numpy as np
    n_cams = len(cameras)

    pts = []  # (X[3], V[3], [(cam_idx, uv[2]), ...])
    for rec in keyframe_records:
        cloud, X_all, V_all, src = rec["cloud"], rec["X"], rec["V"], rec["src_idx"]
        for p in range(len(src)):
            obs = cloud.observations[int(src[p])]
            if len(obs) >= min_views:
                pts.append((X_all[p].astype(np.float64), V_all[p].astype(np.float64), obs))
    if not pts:
        raise RuntimeError("solve_camera_offsets: no usable multi-view points")

    cam_obs_count = np.zeros(n_cams, dtype=np.int64)
    for _, _, obs in pts:
        for ci, _ in obs:
            cam_obs_count[int(ci)] += 1

    def proj(cam, X):
        uv = cam.project_world_point(X)
        return None if uv is None else np.asarray(uv, np.float64)

    def jac(cam, X):  # central finite-diff dπ/dX  -> [2,3]
        eps = fd_eps_frac * (np.linalg.norm(X - cam.center_world) + 1e-6)
        J = np.empty((2, 3))
        for a in range(3):
            d = np.zeros(3); d[a] = eps
            up, um = proj(cam, X + d), proj(cam, X - d)
            if up is None or um is None:
                return None
            J[:, a] = (up - um) / (2.0 * eps)
        return J

    delta = np.zeros(n_cams)  # offsets in FRAMES

    def rms(apply):
        s = []
        for X, V, obs in pts:
            for ci, uv in obs:
                pv = proj(cameras[ci], X + (delta[ci] if apply else 0.0) * V)
                if pv is not None:
                    s.append(float(np.sum((np.asarray(uv, np.float64) - pv) ** 2)))
        return float(np.sqrt(np.mean(s))) if s else float("nan")

    rms0 = rms(False)
    for _ in range(n_iters):
        S = np.zeros((n_cams, n_cams)); g = np.zeros(n_cams); used = 0
        for X, V, obs in pts:
            rows = []
            for ci, uv in obs:
                cam = cameras[ci]
                Xc = X + delta[ci] * V
                pi = proj(cam, Xc); J = jac(cam, Xc) if pi is not None else None
                if J is None:
                    continue
                v = J @ V
                if np.linalg.norm(v) < min_image_speed_px:
                    continue
                r = np.asarray(uv, np.float64) - pi
                w = 1.0 if np.linalg.norm(r) <= huber_px else huber_px / (np.linalg.norm(r) + 1e-9)
                rows.append((ci, J, v, r, w))
            if len(rows) < min_views:
                continue
            used += 1
            lc = [r[0] for r in rows]; m = len(lc)
            A = np.zeros((2 * m, 3)); B = np.zeros((2 * m, m)); rv = np.zeros(2 * m); Wd = np.zeros(2 * m)
            for i, (_, J, v, r, w) in enumerate(rows):
                A[2*i:2*i+2] = J; B[2*i:2*i+2, i] = -v; rv[2*i:2*i+2] = r; Wd[2*i:2*i+2] = w
            WA, WB, Wr = Wd[:, None] * A, Wd[:, None] * B, Wd * rv
            AtA = A.T @ WA + 1e-9 * np.eye(3)
            AtAinv = np.linalg.inv(AtA)
            AtB, Atr = A.T @ WB, A.T @ Wr
            S_loc = B.T @ WB - AtB.T @ AtAinv @ AtB
            g_loc = B.T @ Wr - AtB.T @ AtAinv @ Atr
            for ii, ci in enumerate(lc):
                g[ci] += g_loc[ii]
                for jj, cj in enumerate(lc):
                    S[ci, cj] += S_loc[ii, jj]
        S += ridge_frac * (np.trace(S) / n_cams) * np.eye(n_cams)
        dd = np.linalg.solve(S, g)
        dd -= dd.mean()
        delta += dd
        if np.max(np.abs(dd)) < 1e-4:
            break

    delta -= delta.mean()
    
    under = cam_obs_count < min_cam_obs
    if under.any():
        print(f"[solve_offsets] pinning {int(under.sum())} under-observed cameras to 0 "
              f"(min_cam_obs={min_cam_obs})")
        delta[under] = 0.0
        if (~under).any():
            delta[~under] -= delta[~under].mean()   # re-gauge on solved cams only

    rms1 = rms(True)
    print(f"[solve_offsets] points used={used}  reproj RMS {rms0:.3f}px -> {rms1:.3f}px")
    for ci, cam in enumerate(cameras):
        print(f"  cam {ci:2d} ({cam.folder}): {delta[ci]/fps*1000:+.3f} ms ({delta[ci]:+.3f} frame)")
    return {f"{int(cam.folder):04d}": float(delta[ci] / fps) for ci, cam in enumerate(cameras)}

def triangulate_multiview_dlt(
    observations: Sequence[Observation], cameras: Sequence[Camera]
) -> Optional[np.ndarray]:
    if len(observations) < 2:
        return None
    A_rows = []
    for obs in observations:
        cam = cameras[obs.cam_idx]
        uv_cam = cam.unproject_image_points(obs.uv.reshape(1, 2))[0]
        P = cam.extrinsic   # cached_property — no reallocation
        u, v = float(uv_cam[0]), float(uv_cam[1])
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])
    A = np.stack(A_rows, axis=0)
    # full_matrices=False skips the large [2*N, 2*N] U matrix we never use.
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    if not np.all(np.isfinite(X)):
        return None
    return X.astype(np.float64)


def compute_triangulation_angle_deg(
    X: np.ndarray,
    observations: Sequence[Observation],
    cameras: Sequence[Camera],
) -> float:
    """Largest angle between any two viewing rays to X."""
    if len(observations) < 2:
        return 0.0
    rays = []
    for obs in observations:
        cam = cameras[obs.cam_idx]
        d = X - cam.center_world   # cached_property
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        rays.append(d / n)
    if len(rays) < 2:
        return 0.0
    rays_np = np.stack(rays, axis=0)
    cos_mat = np.clip(rays_np @ rays_np.T, -1.0, 1.0)
    np.fill_diagonal(cos_mat, 1.0)
    min_cos = float(cos_mat.min())
    return math.degrees(math.acos(min_cos))


def compute_reprojection_errors(
    X: np.ndarray,
    observations: Sequence[Observation],
    cameras: Sequence[Camera],
) -> Tuple[np.ndarray, np.ndarray]:
    errors = []
    depths = []
    for obs in observations:
        cam = cameras[obs.cam_idx]
        xcam = cam.R @ X + cam.t
        depths.append(float(xcam[2]))
        uv = cam.project_world_point(X)
        if uv is None:
            errors.append(np.inf)
        else:
            errors.append(float(np.linalg.norm(uv - obs.uv)))
    return np.asarray(errors, dtype=np.float64), np.asarray(depths, dtype=np.float64)


# ---------------------------------------------------------------------------
# Batched geometry over ragged observation lists
#
# A batch of M points stores its observations point-major and flat:
# ``obs_cam [K]`` camera index, ``obs_uv [K, 2]`` pixel, and ``counts [M]``
# observations per point (point i owns rows offsets[i]:offsets[i]+counts[i]).
# Each batched function returns what the scalar function of the same name
# returns for every point, up to floating-point rounding.
# ---------------------------------------------------------------------------


def _ragged_offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
    return offsets


def _ragged_groups(counts: np.ndarray):
    """Yield ``(n, point_idx [m], obs_idx [m, n])`` for each distinct count."""
    counts = np.asarray(counts, dtype=np.int64)
    offsets = _ragged_offsets(counts)
    for n in np.unique(counts):
        n = int(n)
        points = np.flatnonzero(counts == n)
        obs = offsets[points, None] + np.arange(n, dtype=np.int64)[None, :]
        yield n, points, obs


def _ragged_take(counts: np.ndarray, point_idx: np.ndarray) -> np.ndarray:
    """Flat observation rows of the selected points, point-major."""
    counts = np.asarray(counts, dtype=np.int64)
    point_idx = np.asarray(point_idx, dtype=np.int64)
    sub_counts = counts[point_idx]
    total = int(sub_counts.sum())
    if total == 0:
        return np.zeros((0,), dtype=np.int64)
    starts = _ragged_offsets(counts)[point_idx]
    sub_offsets = _ragged_offsets(sub_counts)
    return np.repeat(starts - sub_offsets, sub_counts) + np.arange(total, dtype=np.int64)


def _ragged_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-point ``np.mean`` (rows reduce exactly like a 1-D ``np.mean``)."""
    out = np.full((len(counts),), np.nan, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    for n, points, obs in _ragged_groups(counts):
        if n > 0:
            out[points] = values[obs].mean(axis=1)
    return out


def _ragged_max(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full((len(counts),), np.nan, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    for n, points, obs in _ragged_groups(counts):
        if n > 0:
            out[points] = values[obs].max(axis=1)
    return out


def _ragged_any(mask: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.zeros((len(counts),), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    for n, points, obs in _ragged_groups(counts):
        if n > 0:
            out[points] = mask[obs].any(axis=1)
    return out


def _rotate_rows(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``R @ x`` for every row x, bit-identical to the per-point product
    (a plain ``X @ R.T`` takes a different BLAS path and rounds differently)."""
    return np.matmul(R[None], X[:, :, None])[:, :, 0]


def _row_norms(d: np.ndarray) -> np.ndarray:
    """``np.linalg.norm`` of every row (same summation order, bit-identical)."""
    return np.linalg.norm(d, axis=1)


def _py_min(a, b) -> np.ndarray:
    """Elementwise builtin ``min(a, b)``: ``b`` only where ``b < a`` (NaN-faithful)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.where(b < a, b, a)


def _project_camera_points(cam: Camera, xcam: np.ndarray) -> np.ndarray:
    """Distorted pixels of camera-frame points [N, 3]; raises cv2.error."""
    obj = np.ascontiguousarray(xcam, dtype=np.float64).reshape(-1, 1, 3)
    zero_rvec = np.zeros((3, 1), dtype=np.float64)
    zero_tvec = np.zeros((3, 1), dtype=np.float64)
    if cam.camera_model_name.upper() == 'OPENCV_FISHEYE':
        uv, _ = cv2.fisheye.projectPoints(obj, zero_rvec, zero_tvec, cam._K, cam._D)
    else:
        uv, _ = cv2.projectPoints(obj, zero_rvec, zero_tvec, cam._K, cam._D)
    return uv.reshape(-1, 2)


def triangulate_multiview_dlt_batch(
    obs_cam: np.ndarray,
    obs_uv: np.ndarray,
    counts: np.ndarray,
    cameras: Sequence[Camera],
) -> Tuple[np.ndarray, np.ndarray]:
    """``triangulate_multiview_dlt`` for M points at once.

    Unprojects once per camera and solves every DLT system of the same size
    with one stacked SVD. Returns ``X [M, 3]`` (NaN where it failed) and
    ``ok [M]``, True exactly where the scalar function returns a point.
    """
    obs_cam = np.asarray(obs_cam, dtype=np.int64).reshape(-1)
    obs_uv = np.asarray(obs_uv, dtype=np.float64).reshape(-1, 2)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    n_points = len(counts)
    X = np.full((n_points, 3), np.nan, dtype=np.float64)
    ok = np.zeros((n_points,), dtype=bool)
    if n_points == 0 or len(obs_cam) == 0:
        return X, ok

    norm_uv = np.empty_like(obs_uv)
    for cam_idx in np.unique(obs_cam):
        sel = obs_cam == cam_idx
        norm_uv[sel] = cameras[int(cam_idx)].unproject_image_points(obs_uv[sel])

    used = np.unique(obs_cam)
    P = np.zeros((int(used.max()) + 1, 3, 4), dtype=np.float64)
    for cam_idx in used:
        P[int(cam_idx)] = cameras[int(cam_idx)].extrinsic
    P_obs = P[obs_cam]
    rows = np.empty((len(obs_cam), 2, 4), dtype=np.float64)
    rows[:, 0] = norm_uv[:, 0:1] * P_obs[:, 2] - P_obs[:, 0]
    rows[:, 1] = norm_uv[:, 1:2] * P_obs[:, 2] - P_obs[:, 1]
    rows_finite = np.isfinite(rows).all(axis=(1, 2))

    for n, points, obs in _ragged_groups(counts):
        if n < 2:
            continue
        finite = rows_finite[obs].all(axis=1)
        for p_local in np.flatnonzero(~finite):
            # LAPACK rejects non-finite systems; keep the scalar behaviour.
            point_obs = [
                Observation(cam_idx=int(obs_cam[k]), uv=obs_uv[k])
                for k in obs[p_local]
            ]
            Xp = triangulate_multiview_dlt(point_obs, cameras)
            if Xp is not None:
                X[points[p_local]] = Xp
                ok[points[p_local]] = True
        if not finite.any():
            continue
        good = points[finite]
        A = rows[obs[finite]].reshape(len(good), 2 * n, 4)
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        Xh = vt[:, -1, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            Xg = Xh[:, :3] / Xh[:, 3:4]
        good_ok = ~(np.abs(Xh[:, 3]) < 1e-12) & np.isfinite(Xg).all(axis=1)
        X[good[good_ok]] = Xg[good_ok]
        ok[good[good_ok]] = True
    return X, ok


def compute_reprojection_errors_batch(
    X: np.ndarray,
    obs_cam: np.ndarray,
    obs_uv: np.ndarray,
    counts: np.ndarray,
    cameras: Sequence[Camera],
) -> Tuple[np.ndarray, np.ndarray]:
    """``compute_reprojection_errors`` for M points: flat ``errors [K]`` and
    ``depths [K]`` in observation order (inf where the point does not project)."""
    obs_cam = np.asarray(obs_cam, dtype=np.int64).reshape(-1)
    obs_uv = np.asarray(obs_uv, dtype=np.float64).reshape(-1, 2)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    n_obs = len(obs_cam)
    errors = np.full((n_obs,), np.inf, dtype=np.float64)
    depths = np.empty((n_obs,), dtype=np.float64)
    if n_obs == 0:
        return errors, depths
    X_obs = np.repeat(np.asarray(X, dtype=np.float64).reshape(-1, 3), counts, axis=0)
    for cam_idx in np.unique(obs_cam):
        cam = cameras[int(cam_idx)]
        rows = np.flatnonzero(obs_cam == cam_idx)
        xcam = _rotate_rows(cam.R, X_obs[rows]) + cam.t
        depths[rows] = xcam[:, 2]
        front = xcam[:, 2] > 1e-9
        rows_front = rows[front]
        if len(rows_front) == 0:
            continue
        try:
            projected = _project_camera_points(cam, xcam[front])
        except cv2.error:
            for k in rows_front:
                uv = cam.project_world_point(X_obs[k])
                if uv is not None:
                    errors[k] = float(np.linalg.norm(uv - obs_uv[k]))
            continue
        finite = np.isfinite(projected).all(axis=1)
        errors[rows_front[finite]] = _row_norms(
            projected[finite] - obs_uv[rows_front[finite]]
        )
    return errors, depths


def compute_triangulation_angle_deg_batch(
    X: np.ndarray,
    obs_cam: np.ndarray,
    counts: np.ndarray,
    cameras: Sequence[Camera],
) -> np.ndarray:
    """``compute_triangulation_angle_deg`` for M points: ``angles [M]``."""
    obs_cam = np.asarray(obs_cam, dtype=np.int64).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    angles = np.zeros((len(counts),), dtype=np.float64)
    if len(obs_cam) == 0:
        return angles
    used = np.unique(obs_cam)
    centers = np.zeros((int(used.max()) + 1, 3), dtype=np.float64)
    for cam_idx in used:
        centers[int(cam_idx)] = cameras[int(cam_idx)].center_world
    X_obs = np.repeat(np.asarray(X, dtype=np.float64).reshape(-1, 3), counts, axis=0)
    d = X_obs - centers[obs_cam]
    norms = _row_norms(d)
    ray_ok = ~(norms < 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        rays = d / norms[:, None]
    for n, points, obs in _ragged_groups(counts):
        if n < 2:
            continue
        valid = ray_ok[obs]                      # [m, n]
        enough = valid.sum(axis=1) >= 2
        if not enough.any():
            continue
        r = np.where(valid[:, :, None], rays[obs], 0.0)[enough]
        v = valid[enough]
        cos = np.clip(np.matmul(r, r.transpose(0, 2, 1)), -1.0, 1.0)
        # Skipped rays and the diagonal never lower the minimum.
        cos[~(v[:, :, None] & v[:, None, :])] = 1.0
        idx = np.arange(n)
        cos[:, idx, idx] = 1.0
        min_cos = cos.reshape(len(r), -1).min(axis=1)
        angles[points[enough]] = np.degrees(np.arccos(min_cos))
    return angles


def within_scene_bounds_batch(
    X: np.ndarray,
    scene_meta: Optional[Dict[str, object]],
    scene_radius_mult: float,
) -> np.ndarray:
    """``within_scene_bounds`` for points [M, 3]."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    if scene_meta is None:
        return np.ones((len(X),), dtype=bool)
    center = np.asarray(
        scene_meta.get('scene_center', [0.0, 0.0, 0.0]), dtype=np.float64
    )
    radius = float(
        scene_meta.get('scene_radius_p95', scene_meta.get('scene_radius_max', 0.0))
    )
    if radius <= 0:
        return np.ones((len(X),), dtype=bool)
    return _row_norms(X - center[None, :]) <= float(scene_radius_mult * radius)


# ---------------------------------------------------------------------------
# Vectorised bilinear colour sampling
# ---------------------------------------------------------------------------


def bilinear_sample_rgb_batch(image: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """
    Batch bilinear sampling.

    Args:
        image: [H, W, 3] uint8
        uvs:   [N, 2] float32 pixel coordinates

    Returns:
        [N, 3] float32 in [0, 1]
    """
    h, w = image.shape[:2]
    u = np.clip(uvs[:, 0], 0.0, w - 1.0)
    v = np.clip(uvs[:, 1], 0.0, h - 1.0)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (u - x0).astype(np.float32)[:, None]  # [N, 1] for broadcasting
    dy = (v - y0).astype(np.float32)[:, None]
    c00 = image[y0, x0].astype(np.float32)
    c10 = image[y0, x1].astype(np.float32)
    c01 = image[y1, x0].astype(np.float32)
    c11 = image[y1, x1].astype(np.float32)
    c = (c00 * (1.0 - dx) + c10 * dx) * (1.0 - dy) + \
        (c01 * (1.0 - dx) + c11 * dx) * dy
    return (c / 255.0).astype(np.float32)


def bilinear_sample_rgb(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Scalar convenience wrapper — calls the batch implementation."""
    return bilinear_sample_rgb_batch(image, np.asarray(uv, dtype=np.float32).reshape(1, 2))[0]


def _batch_compute_colors(
    obs_lists: List[List[Tuple[int, np.ndarray]]],
    images: Sequence[np.ndarray],
    mode: str = 'medoid',
    trim_frac: float = 0.25,
) -> np.ndarray:
    """
    Batch color sampling across N points.

    Modes:
      mean         old behavior; averages all observing-camera colors
      median       per-channel median RGB
      medoid       choose the real observed RGB closest to median RGB
      trimmed-mean drop farthest colors from median, then average

    For TriGS/3DGS init, medoid is usually safer than mean for glossy
    surfaces because it avoids blending view-dependent highlights into
    the base diffuse color.
    """
    n = len(obs_lists)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    mode = str(mode).lower()
    if mode not in ('mean', 'median', 'medoid', 'trimmed-mean'):
        raise ValueError(f'Unknown color mode: {mode}')

    # Group all (point_index, uv) pairs by camera.
    cam_pt_indices: Dict[int, List[int]] = {}
    cam_uvs: Dict[int, List[np.ndarray]] = {}

    for pt_idx, obs_list in enumerate(obs_lists):
        for cam_idx, uv in obs_list:
            cam_pt_indices.setdefault(cam_idx, []).append(pt_idx)
            cam_uvs.setdefault(cam_idx, []).append(np.asarray(uv, dtype=np.float32))

    # Store sampled colors per point. Per-frame point count is modest enough
    # that this is fine, and it avoids global [N, max_views, 3] padding.
    per_point_cols: List[List[np.ndarray]] = [[] for _ in range(n)]

    for cam_idx, pt_idx_list in cam_pt_indices.items():
        uvs_arr = np.stack(cam_uvs[cam_idx], axis=0)
        cols = bilinear_sample_rgb_batch(images[cam_idx], uvs_arr).astype(np.float32)

        for pt_idx, col in zip(pt_idx_list, cols):
            per_point_cols[int(pt_idx)].append(col)

    out = np.zeros((n, 3), dtype=np.float32)

    for i, cols_list in enumerate(per_point_cols):
        if not cols_list:
            continue

        cols = np.stack(cols_list, axis=0).astype(np.float32)

        if len(cols) == 1:
            out[i] = cols[0]
            continue

        if mode == 'mean':
            out[i] = cols.mean(axis=0)

        elif mode == 'median':
            out[i] = np.median(cols, axis=0)

        elif mode == 'medoid':
            med = np.median(cols, axis=0)
            d2 = np.sum((cols - med[None, :]) ** 2, axis=1)
            out[i] = cols[int(np.argmin(d2))]

        elif mode == 'trimmed-mean':
            med = np.median(cols, axis=0)
            d2 = np.sum((cols - med[None, :]) ** 2, axis=1)

            k = len(cols)
            drop = int(round(float(trim_frac) * k))
            keep_n = max(1, k - drop)

            keep_idx = np.argsort(d2)[:keep_n]
            out[i] = cols[keep_idx].mean(axis=0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Mask loading + distortion-aware mask filter
# ---------------------------------------------------------------------------


def mask_path_for_frame(
    masks_root: Path, cam: Camera, frame_idx: int, mask_pattern: str
) -> Path:
    return masks_root / cam.folder / mask_pattern.format(frame=frame_idx)


def load_frame_masks(
    masks_root: Path,
    cameras: Sequence[Camera],
    frame_idx: int,
    mask_pattern: str,
) -> List[Optional[np.ndarray]]:
    """Load all camera masks for a single frame in parallel."""

    def _load_mask(cam: Camera) -> Optional[np.ndarray]:
        p = mask_path_for_frame(masks_root, cam, frame_idx, mask_pattern)
        if not p.exists():
            return None
        try:
            raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
        if raw is None:
            return None
        return raw.astype(np.float32) / 255.0

    n_workers = min(len(cameras), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        result = list(ex.map(_load_mask, cameras))
    return result


def load_frame_images_for_camera_frames(
    images_root: Path,
    cameras: Sequence[Camera],
    frame_indices: Sequence[int],
    image_pattern: str,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Load one frame per camera. This is the sync-aware counterpart to
    load_frame_images(), where camera c may use a different source frame.
    """
    if len(frame_indices) != len(cameras):
        raise ValueError(
            f"frame_indices length {len(frame_indices)} != cameras {len(cameras)}"
        )

    paths = [
        image_path(images_root, cam, int(frame_indices[i]), image_pattern)
        for i, cam in enumerate(cameras)
    ]

    def _load(item: Tuple[int, Path]) -> np.ndarray:
        i, p = item
        if not p.exists():
            raise FileNotFoundError(
                f"Missing sync-aware image for camera {cameras[i].folder} "
                f"frame {int(frame_indices[i])}: {p}"
            )
        return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)

    n_workers = min(len(cameras), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        images = list(ex.map(_load, list(enumerate(paths))))
    return paths, images


def load_frame_masks_for_camera_frames(
    masks_root: Path,
    cameras: Sequence[Camera],
    frame_indices: Sequence[int],
    mask_pattern: str,
) -> List[Optional[np.ndarray]]:
    """Load one mask per camera, matching sync-aware source frames."""
    if len(frame_indices) != len(cameras):
        raise ValueError(
            f"frame_indices length {len(frame_indices)} != cameras {len(cameras)}"
        )

    def _load_mask(item: Tuple[int, Camera]) -> Optional[np.ndarray]:
        i, cam = item
        p = mask_path_for_frame(masks_root, cam, int(frame_indices[i]), mask_pattern)
        if not p.exists():
            return None
        try:
            raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
        if raw is None:
            return None
        return raw.astype(np.float32) / 255.0

    n_workers = min(len(cameras), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_load_mask, list(enumerate(cameras))))


def _lookup_sync_offset_for_camera(
    sync_data: Dict[str, float],
    cameras: Sequence[Camera],
    cam_idx: int,
) -> Tuple[float, Optional[str]]:
    """Resolve a camera offset from explicit camera identities in sync.json.

    Camera-list indices are deliberately not candidates. They are zero-based,
    while capture folder identifiers are commonly one-based, so using an index
    as a fallback can silently assign a neighboring camera's offset when the
    correct folder key is absent.
    """
    cam = cameras[int(cam_idx)]
    candidates: List[str] = []
    candidates.extend([str(cam.folder), str(cam.name)])

    # Look for the exact padded folder name (e.g. "01" -> "0001")
    if str(cam.folder).isdigit():
        candidates.append(f"{int(cam.folder):04d}")

    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in sync_data:
            return float(sync_data[key]), key
    return 0.0, None


def load_sync_offsets_for_cameras(
    sync_json: Optional[str],
    cameras: Sequence[Camera],
) -> Tuple[np.ndarray, List[Optional[str]]]:
    if not sync_json:
        return np.zeros((len(cameras),), dtype=np.float64), [None] * len(cameras)

    p = Path(sync_json)
    if not p.exists():
        raise FileNotFoundError(f"--sync-json not found: {p}")

    with p.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    sync_data = {str(k): float(v) for k, v in raw.items()}

    offsets = np.zeros((len(cameras),), dtype=np.float64)
    keys: List[Optional[str]] = []
    key_owners: Dict[str, int] = {}
    for cam_idx in range(len(cameras)):
        off, key = _lookup_sync_offset_for_camera(sync_data, cameras, cam_idx)
        if key is not None and key in key_owners:
            other_idx = key_owners[key]
            raise ValueError(
                f"sync key {key!r} matched multiple cameras: "
                f"index {other_idx} folder {cameras[other_idx].folder!r} and "
                f"index {cam_idx} folder {cameras[cam_idx].folder!r}"
            )
        if key is not None:
            key_owners[key] = cam_idx
        offsets[cam_idx] = float(off)
        keys.append(key)

    missing = [i for i, k in enumerate(keys) if k is None]
    if missing:
        print(
            f"[sync] WARNING: missing offsets for {len(missing)} cameras; "
            f"using 0.0 for first few {missing[:8]}"
        )

    preview = [
        {
            'cam_idx': int(i),
            'folder': str(cameras[i].folder),
            'sync_key': keys[i],
            'offset_ms': round(float(offsets[i]) * 1000.0, 3),
        }
        for i in range(min(8, len(cameras)))
    ]
    print(f"[sync] loaded {p}; mapping preview: {preview}")
    return offsets, keys


def sync_aware_camera_frames(
    global_source_frame: int,
    *,
    fps: float,
    offsets_sec: np.ndarray,
    frame_start: int,
    frame_end: int,
) -> np.ndarray:
    """Map a desired global source-frame time to a nearest source frame
    for each camera.

    Convention: physical_time_seconds = (source_frame - frame_start) / fps
    + camera_offset_seconds. Therefore the frame closest to a global time is
    source_frame ~= global_source_frame - camera_offset_seconds * fps.
    """
    fps = max(float(fps), 1e-6)
    raw = float(global_source_frame) - offsets_sec.astype(np.float64) * fps
    frames = np.rint(raw).astype(np.int64)

    bad = (frames < int(frame_start)) | (frames > int(frame_end))
    if bad.any():
        frames = np.clip(frames, int(frame_start), int(frame_end))
    return frames


def camera_capture_residual_frames(
    global_source_frame: int,
    camera_frame_indices: np.ndarray,
    *,
    fps: float,
    offsets_sec: np.ndarray,
) -> np.ndarray:
    """Physical capture time relative to the requested global frame, in frames.

    With physical_frame_time = source_frame + offset_seconds * fps, the
    residual after nearest-integer frame selection is:
      selected_frame - requested_frame + offset_seconds * fps.
    It normally lies in [-0.5, 0.5], except where frame-range clipping occurs.
    """
    return (
        np.asarray(camera_frame_indices, dtype=np.float64)
        - float(global_source_frame)
        + np.asarray(offsets_sec, dtype=np.float64) * float(fps)
    )


def frame_input_fingerprint(
    *,
    frame_idx: int,
    next_frame_idx: int,
    images_root: Path,
    masks_root: Optional[Path],
    cameras: Sequence[Camera],
    image_pattern: str,
    mask_pattern: str,
    fps: float,
    offsets_sec: np.ndarray,
    sync_enabled: bool,
    frame_start: int,
    frame_end: int,
) -> str:
    signatures: List[Dict[str, object]] = []
    for logical_frame in (int(frame_idx), int(next_frame_idx)):
        selected = (
            sync_aware_camera_frames(
                logical_frame,
                fps=fps,
                offsets_sec=offsets_sec,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            if sync_enabled
            else np.full((len(cameras),), logical_frame, dtype=np.int64)
        )
        for cam_idx, cam in enumerate(cameras):
            source_frame = int(selected[cam_idx])
            signatures.append(
                {
                    "kind": "image",
                    "logical_frame": logical_frame,
                    "camera": int(cam_idx),
                    **_file_signature(
                        image_path(images_root, cam, source_frame, image_pattern)
                    ),
                }
            )
            if masks_root is not None:
                signatures.append(
                    {
                        "kind": "mask",
                        "logical_frame": logical_frame,
                        "camera": int(cam_idx),
                        **_file_signature(
                            mask_path_for_frame(
                                masks_root, cam, source_frame, mask_pattern
                            )
                        ),
                    }
                )
    return _json_hash(signatures)


def load_rs_signs_for_cameras(
    rotation_json: str,
    cameras: Sequence[Camera],
    flip_sign: bool = False,
) -> np.ndarray:
    """Per-camera rolling-shutter time direction along +u in the delivered
    (rotated-to-portrait) images.

    Cameras record landscape; extraction rotates ±90°, so the sensor readout
    lines become portrait image COLUMNS and the readout time axis runs
    horizontally with a per-camera sign. Convention (empirically calibrated on
    example capture by regressing trained camera-time offsets against
    rotation-signed subject column): rotation 'l' -> capture time increases
    with +u (sign +1), rotation 'r' -> decreases (sign -1). Use
    ``flip_sign`` if a capture rig inverts this mapping.

    Accepted keys per camera folder ``F``: "Camera {int(F)}", "{int(F)}",
    "{int(F):02d}", "{int(F):04d}", or the folder string itself. Cameras
    without an entry get sign 0 (RS compensation disabled for them).
    """
    raw = json.loads(Path(rotation_json).read_text(encoding='utf-8'))
    data = {str(k): str(v).strip().lower() for k, v in raw.items()}
    signs = np.zeros((len(cameras),), dtype=np.float64)
    missing = []
    for i, cam in enumerate(cameras):
        folder = str(cam.folder)
        candidates = [folder]
        if folder.isdigit():
            n = int(folder)
            candidates += [f'Camera {n}', str(n), f'{n:02d}', f'{n:04d}']
        val = None
        for key in candidates:
            if key in data:
                val = data[key]
                break
        if val in ('l', 'left'):
            signs[i] = 1.0
        elif val in ('r', 'right'):
            signs[i] = -1.0
        else:
            missing.append(folder)
    if flip_sign:
        signs = -signs
    if missing:
        print(
            f'[rs] WARNING: no rotation entry for {len(missing)} cameras '
            f'({missing[:6]}...); RS compensation disabled for them.'
        )
    n_l = int(np.sum(signs > 0)) if not flip_sign else int(np.sum(signs < 0))
    print(f'[rs] rotation signs loaded: {n_l} left, {len(cameras) - n_l - len(missing)} right'
          + (' (flipped)' if flip_sign else ''))
    return signs


def batch_mask_filter_distorted(
    candidate_Xs: np.ndarray,
    candidate_obs_uvs: List[Optional[Dict[int, np.ndarray]]],
    cameras: Sequence[Camera],
    masks: Sequence[Optional[np.ndarray]],
    min_fg_alpha: float,
    min_fg_fraction: float,
) -> np.ndarray:
    """
    Distortion-aware mask voting. For each camera that has a mask:
      - project all N points through cv2.projectPoints with full distortion
      - override with stored observation uv where available
      - sample the mask with bilinear interpolation, vote fg/bg
    """
    N = len(candidate_Xs)
    if N == 0:
        return np.ones(0, dtype=bool)

    Xs = candidate_Xs.astype(np.float64)
    has_any_mask = any(m is not None for m in masks)
    if not has_any_mask:
        return np.ones(N, dtype=bool)

    fg_votes = np.zeros(N, dtype=np.int32)
    total_votes = np.zeros(N, dtype=np.int32)

    # Invert the stored observations once: {cam_idx: (rows, uvs)}. Scanning
    # every candidate's dict again inside the per-camera loop costs
    # n_cameras * N Python iterations, which dominates at million-point scale.
    overrides: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if candidate_obs_uvs is not None:
        rows_by_cam: Dict[int, List[int]] = {}
        uvs_by_cam: Dict[int, List[np.ndarray]] = {}
        for n, d in enumerate(candidate_obs_uvs):
            if not d:
                continue
            for cam_key, uv in d.items():
                rows_by_cam.setdefault(int(cam_key), []).append(n)
                uvs_by_cam.setdefault(int(cam_key), []).append(uv)
        overrides = {
            cam_key: (
                np.asarray(rows, dtype=np.int64),
                np.asarray(uvs_by_cam[cam_key], dtype=np.float64).reshape(-1, 2),
            )
            for cam_key, rows in rows_by_cam.items()
        }

    for cam_idx, (cam, mask) in enumerate(zip(cameras, masks)):
        if mask is None:
            continue

        uvs_proj, valid = cam.project_world_points_batch(Xs)

        override = overrides.get(cam_idx)
        if override is not None:
            rows, uvs_obs = override
            uvs_proj[rows] = uvs_obs
            valid[rows] = True

        if not valid.any():
            continue

        W, H = cam.width, cam.height
        in_bounds = valid & (
            (uvs_proj[:, 0] >= 0) & (uvs_proj[:, 0] <= W - 1) &
            (uvs_proj[:, 1] >= 0) & (uvs_proj[:, 1] <= H - 1)
        )
        if not in_bounds.any():
            continue

        idx = np.where(in_bounds)[0]
        u = uvs_proj[idx, 0]
        v = uvs_proj[idx, 1]
        mh, mw = mask.shape[:2]
        if mw != W or mh != H:
            u = u * (mw / W)
            v = v * (mh / H)
        u = np.clip(u, 0.0, mw - 1.0)
        v = np.clip(v, 0.0, mh - 1.0)
        x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
        x1 = np.minimum(x0 + 1, mw - 1);  y1 = np.minimum(y0 + 1, mh - 1)
        dx = (u - x0).astype(np.float32);  dy = (v - y0).astype(np.float32)
        a00 = mask[y0, x0]; a10 = mask[y0, x1]
        a01 = mask[y1, x0]; a11 = mask[y1, x1]
        a0 = a00 * (1.0 - dx) + a10 * dx
        a1 = a01 * (1.0 - dx) + a11 * dx
        alphas = a0 * (1.0 - dy) + a1 * dy

        total_votes[idx] += 1
        fg_votes[idx[alphas >= min_fg_alpha]] += 1

    no_data = total_votes == 0
    fraction = np.where(
        total_votes > 0,
        fg_votes.astype(np.float32) / total_votes.astype(np.float32),
        0.0,
    )
    return no_data | (fraction >= min_fg_fraction)


# ---------------------------------------------------------------------------
# Voxel dedup
# ---------------------------------------------------------------------------

def _voxel_merge_static_rows(
    positions: np.ndarray,
    colors: np.ndarray,
    radius: float,
    max_points: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(positions) == 0 or radius <= 0.0:
        return positions, colors

    coords = np.floor(positions / radius).astype(np.int64)
    mins = coords.min(axis=0)
    coords = coords - mins

    maxs = coords.max(axis=0) + 1
    keys = (
        coords[:, 0] * (maxs[1] * maxs[2])
        + coords[:, 1] * maxs[2]
        + coords[:, 2]
    )

    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    pos_sorted = positions[order]
    col_sorted = colors[order]

    starts = np.r_[0, np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(keys_sorted)]).astype(np.float32)

    pos_out = np.add.reduceat(pos_sorted, starts, axis=0) / counts[:, None]
    col_out = np.add.reduceat(col_sorted, starts, axis=0) / counts[:, None]

    if max_points > 0 and len(pos_out) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pos_out), size=max_points, replace=False)
        keep.sort()
        pos_out = pos_out[keep]
        col_out = col_out[keep]

    return pos_out.astype(np.float32), col_out.astype(np.float32)


def _disk_backed_static_merge(
    arrays: Mapping[str, np.ndarray],
    *,
    work_dir: Path,
    radius: float,
    max_points: int,
    seed: int,
    source_frame_start: int,
    block_rows: int = 250_000,
) -> Dict[str, np.memmap]:
    """Merge static voxels through SQLite while streaming dynamic rows.

    SQLite is used only for the optional global static merge. Per-block numpy
    aggregation keeps the number of database updates close to the number of
    occupied voxels rather than the number of input points.
    """
    n_rows = len(arrays["positions"])
    db_path = work_dir / "static-voxels.sqlite3"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE voxels (
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            z INTEGER NOT NULL,
            n INTEGER NOT NULL,
            px REAL NOT NULL,
            py REAL NOT NULL,
            pz REAL NOT NULL,
            cr REAL NOT NULL,
            cg REAL NOT NULL,
            cb REAL NOT NULL,
            PRIMARY KEY (x, y, z)
        ) WITHOUT ROWID
        """
    )
    upsert = """
        INSERT INTO voxels (x,y,z,n,px,py,pz,cr,cg,cb)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(x,y,z) DO UPDATE SET
            n=n+excluded.n,
            px=px+excluded.px,
            py=py+excluded.py,
            pz=pz+excluded.pz,
            cr=cr+excluded.cr,
            cg=cg+excluded.cg,
            cb=cb+excluded.cb
    """
    dynamic_count = 0
    try:
        for start in range(0, n_rows, block_rows):
            stop = min(start + block_rows, n_rows)
            static_mask = np.asarray(arrays["is_static"][start:stop], dtype=bool)
            dynamic_count += int((~static_mask).sum())
            if not static_mask.any():
                continue
            positions = np.asarray(arrays["positions"][start:stop][static_mask])
            colors = np.asarray(arrays["colors"][start:stop][static_mask])
            coords = np.floor(positions / float(radius)).astype(np.int64)
            unique, inverse = np.unique(coords, axis=0, return_inverse=True)
            counts = np.bincount(inverse).astype(np.int64)
            pos_sums = np.stack(
                [
                    np.bincount(inverse, weights=positions[:, axis].astype(np.float64))
                    for axis in range(3)
                ],
                axis=1,
            )
            col_sums = np.stack(
                [
                    np.bincount(inverse, weights=colors[:, axis].astype(np.float64))
                    for axis in range(3)
                ],
                axis=1,
            )
            rows = [
                (
                    int(unique[i, 0]),
                    int(unique[i, 1]),
                    int(unique[i, 2]),
                    int(counts[i]),
                    float(pos_sums[i, 0]),
                    float(pos_sums[i, 1]),
                    float(pos_sums[i, 2]),
                    float(col_sums[i, 0]),
                    float(col_sums[i, 1]),
                    float(col_sums[i, 2]),
                )
                for i in range(len(unique))
            ]
            connection.executemany(upsert, rows)
            connection.commit()

        voxel_count = int(
            connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0]
        )
        keep_indices: Optional[set[int]] = None
        static_count = voxel_count
        if max_points > 0 and voxel_count > max_points:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(voxel_count, size=max_points, replace=False)
            keep_indices = set(int(x) for x in chosen.tolist())
            static_count = int(max_points)

        output = _create_memmap_arrays(
            work_dir, "final", static_count + dynamic_count
        )
        cursor = connection.execute(
            "SELECT n,px,py,pz,cr,cg,cb FROM voxels ORDER BY x,y,z"
        )
        out_idx = 0
        voxel_idx = 0
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for n, px, py, pz, cr, cg, cb in rows:
                if keep_indices is not None and voxel_idx not in keep_indices:
                    voxel_idx += 1
                    continue
                output["positions"][out_idx] = (px / n, py / n, pz / n)
                output["colors"][out_idx] = (cr / n, cg / n, cb / n)
                out_idx += 1
                voxel_idx += 1

        static_slice = slice(0, static_count)
        output["velocities"][static_slice] = 0.0
        output["accelerations"][static_slice] = 0.0
        output["has_acceleration"][static_slice] = False
        output["track_flags"][static_slice] = 0
        output["times"][static_slice] = 0.5
        output["durations"][static_slice] = 1.0
        output["has_velocity"][static_slice] = False
        output["source_frames"][static_slice] = int(source_frame_start)
        output["is_static"][static_slice] = True
        output["dynamic_protect"][static_slice] = False
        output["dynamic_allowed"][static_slice] = False
        output["stage_keep"][static_slice] = True

        out_idx = static_count
        for start in range(0, n_rows, block_rows):
            stop = min(start + block_rows, n_rows)
            dynamic_mask = ~np.asarray(
                arrays["is_static"][start:stop], dtype=bool
            )
            count = int(dynamic_mask.sum())
            if count == 0:
                continue
            target = slice(out_idx, out_idx + count)
            for name in CACHE_ARRAY_SPECS:
                output[name][target] = arrays[name][start:stop][dynamic_mask]
            output["is_static"][target] = False
            out_idx += count
        _flush_memmaps(output)
        return output
    finally:
        connection.close()


def voxel_downsample(
    positions: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxel-grid downsampling with per-voxel mean (vectorised, no Python loop).
    Uses np.bincount with weights instead of np.split + loop.
    """
    if len(positions) == 0 or voxel_size <= 0:
        return positions, colors
    coords = np.floor(positions / voxel_size).astype(np.int64)
    coords -= coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    keys = (
        coords[:, 0] * (maxs[1] * maxs[2]) +
        coords[:, 1] * maxs[2] +
        coords[:, 2]
    )
    # np.unique with return_inverse assigns each point to a voxel label in
    # [0, n_voxels).  Labels are stable across numpy versions.
    _, labels = np.unique(keys, return_inverse=True)
    n_voxels = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_voxels).astype(np.float32)

    out_pos = np.stack([
        np.bincount(labels, weights=positions[:, i].astype(np.float64), minlength=n_voxels)
        for i in range(3)
    ], axis=1).astype(np.float32)
    out_col = np.stack([
        np.bincount(labels, weights=colors[:, i].astype(np.float64), minlength=n_voxels)
        for i in range(3)
    ], axis=1).astype(np.float32)
    return out_pos / counts[:, None], out_col / counts[:, None]


def voxel_downsample_with_priority(
    positions: np.ndarray,
    colors: np.ndarray,
    priorities: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Within each voxel keep the point with the HIGHEST priority
    (ties broken by stable sort order). Tracked points (priority=1) win
    over fresh points (priority=0).

    Returns (positions, colors, priorities) for the voxel representatives.
    """
    n = len(positions)
    if n == 0 or voxel_size <= 0:
        return positions, colors, priorities
    coords = np.floor(positions / voxel_size).astype(np.int64)
    mins = coords.min(axis=0)
    coords -= mins
    maxs = coords.max(axis=0) + 1
    keys = (
        coords[:, 0] * (maxs[1] * maxs[2]) +
        coords[:, 1] * maxs[2] +
        coords[:, 2]
    )
    # Sort by (key ascending, priority descending) — first occurrence of each
    # key in keys_sorted is the highest-priority point in that voxel.
    order = np.lexsort((-priorities, keys))
    keys_sorted = keys[order]
    _, first_idx = np.unique(keys_sorted, return_index=True)
    kept_orig_idx = order[first_idx]
    return positions[kept_orig_idx], colors[kept_orig_idx], priorities[kept_orig_idx]


def auto_max_voxel_size(
    positions: np.ndarray, target_points: int, safety_factor: float = 4.0
) -> float:
    if len(positions) == 0 or target_points <= 0:
        return 1.0
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    extents = np.maximum(hi - lo, 1e-6)
    bbox_volume = float(extents[0] * extents[1] * extents[2])
    v_min = (bbox_volume / max(target_points, 1)) ** (1.0 / 3.0)
    return float(v_min * safety_factor)


def deduplicate_to_target_with_priority(
    positions: np.ndarray,
    colors: np.ndarray,
    priorities: np.ndarray,
    target_points: int,
    min_radius: float,
    max_radius: float = 0.0,
    max_iters: int = 20,
    tolerance_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(positions)
    if n == 0 or target_points <= 0 or n <= target_points:
        return positions, colors, priorities, 0.0

    if max_radius <= 0.0:
        max_radius = auto_max_voxel_size(positions, target_points)
        print(f'  [dedup] auto max_radius = {max_radius:.5f} (bbox-derived)')

    pos0, col0, pri0 = voxel_downsample_with_priority(positions, colors, priorities, min_radius)
    if len(pos0) <= target_points:
        return pos0, col0, pri0, min_radius

    lo, hi = min_radius, max(min_radius * 2.0, 1e-6)
    pos_hi, col_hi, pri_hi = voxel_downsample_with_priority(positions, colors, priorities, hi)
    while len(pos_hi) > target_points and hi < max_radius:
        lo = hi
        hi = min(hi * 2.0, max_radius)
        pos_hi, col_hi, pri_hi = voxel_downsample_with_priority(positions, colors, priorities, hi)

    if len(pos_hi) > target_points:
        print(
            f'  [dedup] WARNING: could not reach target {target_points} pts '
            f'with max voxel {max_radius:.5f}; returning {len(pos_hi)} pts.'
        )
        return pos_hi, col_hi, pri_hi, hi

    best_pos, best_col, best_pri, best_r = pos_hi, col_hi, pri_hi, hi
    best_err = abs(len(pos_hi) - target_points)
    tol = max(1, int(target_points * tolerance_ratio))

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        pos_mid, col_mid, pri_mid = voxel_downsample_with_priority(positions, colors, priorities, mid)
        n_mid = len(pos_mid)
        err = abs(n_mid - target_points)
        if err < best_err:
            best_pos, best_col, best_pri, best_r = pos_mid, col_mid, pri_mid, mid
            best_err = err
        if err <= tol:
            return pos_mid, col_mid, pri_mid, mid
        if n_mid > target_points:
            lo = mid
        else:
            hi = mid

    return best_pos, best_col, best_pri, best_r


# ---------------------------------------------------------------------------
# Scene bounds
# ---------------------------------------------------------------------------


def within_scene_bounds(
    X: np.ndarray,
    scene_meta: Optional[Dict[str, object]],
    scene_radius_mult: float,
) -> bool:
    if scene_meta is None:
        return True
    center = np.asarray(
        scene_meta.get('scene_center', [0.0, 0.0, 0.0]), dtype=np.float64
    )
    radius = float(
        scene_meta.get('scene_radius_p95', scene_meta.get('scene_radius_max', 0.0))
    )
    if radius <= 0:
        return True
    return float(np.linalg.norm(X - center)) <= float(scene_radius_mult * radius)


def filter_two_view_by_three_view_core(
    positions_by_endpoint: Sequence[np.ndarray],
    support_counts: np.ndarray,
    *,
    enabled: bool,
    radius: float,
    radius_factor: float,
    support_knn: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Reject spatially unsupported 2-view points without thinning surfaces.

    Every 3+-view point is retained. For each endpoint, the k-th-neighbour
    distance is measured against the *full* cloud, including other 2-view
    points. Consequently a dense surface made entirely of 2-view points can
    survive; it does not need a 3-view point beside every output point.

    The trusted 3+-view rows only calibrate what normal local point spacing
    looks like. Sparse trails and islands whose k-neighbour distance is much
    larger than the trusted-surface median are rejected. Passing both t and
    t+delta prevents a locally plausible point from flying into sparse empty
    space at the temporal target.

    ``radius <= 0`` selects an adaptive radius independently at each endpoint:
    ``radius_factor * median k-neighbour spacing`` at trusted 3+-view rows.
    """
    support = np.asarray(support_counts, dtype=np.int32).reshape(-1)
    n = len(support)
    endpoints = [np.asarray(p, dtype=np.float64) for p in positions_by_endpoint]
    if any(len(p) != n for p in endpoints):
        raise ValueError("positions/support length mismatch in 3-view-core filter")

    trusted = support >= 3
    two_view = support == 2
    keep = np.ones(n, dtype=bool)
    info: Dict[str, object] = {
        'enabled': int(bool(enabled)),
        'trusted_three_plus': int(trusted.sum()),
        'two_view_candidates': int(two_view.sum()),
        'two_view_kept': int(two_view.sum()),
        'two_view_rejected': 0,
        'radii': [],
        'trusted_median_knn': [],
        'support_knn': int(support_knn),
    }
    if not enabled or not two_view.any():
        return keep, info

    candidate_idx = np.flatnonzero(two_view)
    candidate_supported = np.ones(len(candidate_idx), dtype=bool)
    used_radii: List[float] = []
    trusted_spacings: List[float] = []
    for endpoint in endpoints:
        k = min(max(int(support_knn), 1), len(endpoint) - 1)
        if k < 1:
            candidate_supported[:] = False
            used_radii.append(0.0)
            trusted_spacings.append(0.0)
            continue

        tree = cKDTree(endpoint)
        # query includes the row itself at column zero.
        dists, _ = tree.query(endpoint, k=k + 1)
        kth = np.asarray(dists[:, k], dtype=np.float64)
        calibration = kth[trusted & np.isfinite(kth) & (kth > 1e-12)]
        if len(calibration) == 0:
            # Explicit radii remain usable without a 3-view core. Automatic
            # mode deliberately rejects because it has no trusted scale.
            spacing = 0.0
        else:
            spacing = float(np.median(calibration))
        used_radius = float(radius)
        if used_radius <= 0.0:
            used_radius = float(radius_factor) * spacing
        if not np.isfinite(used_radius) or used_radius <= 0.0:
            candidate_supported[:] = False
            used_radius = 0.0
        else:
            candidate_supported &= kth[candidate_idx] <= used_radius
        used_radii.append(float(used_radius))
        trusted_spacings.append(float(spacing))

    keep[candidate_idx] = candidate_supported
    n_kept = int(candidate_supported.sum())
    info['two_view_kept'] = n_kept
    info['two_view_rejected'] = int(len(candidate_idx) - n_kept)
    info['radii'] = used_radii
    info['trusted_median_knn'] = trusted_spacings
    return keep, info


# ---------------------------------------------------------------------------
# Fresh-frame reconstruction (cross-camera pair triangulation)
# ---------------------------------------------------------------------------


_MATCH_DEBUG_STAGES = (
    'raw',
    'ransac_kept',
    'calibrated_rejected',
    'calibrated_kept',
    'final_accepted',
)


def _make_match_debug_grids(
    cameras: Sequence[Camera], cell_px: int
) -> Dict[str, List[np.ndarray]]:
    cell = max(int(cell_px), 1)
    return {
        stage: [
            np.zeros(
                (
                    int(math.ceil(float(cam.height) / cell)),
                    int(math.ceil(float(cam.width) / cell)),
                ),
                dtype=np.uint32,
            )
            for cam in cameras
        ]
        for stage in _MATCH_DEBUG_STAGES
    }


def _accumulate_match_debug(
    grid: np.ndarray, uv: np.ndarray, cell_px: int
) -> None:
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return
    cell = max(int(cell_px), 1)
    x = np.floor(points[:, 0] / cell).astype(np.int64)
    y = np.floor(points[:, 1] / cell).astype(np.int64)
    valid = (
        np.isfinite(points).all(axis=1)
        & (x >= 0)
        & (y >= 0)
        & (x < grid.shape[1])
        & (y < grid.shape[0])
    )
    np.add.at(grid, (y[valid], x[valid]), 1)


def _write_match_debug_grids(
    output_dir: Path,
    cameras: Sequence[Camera],
    images: Sequence[np.ndarray],
    grids: Dict[str, List[np.ndarray]],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_camera: Dict[str, Dict[str, int]] = {}
    for cam_idx, (cam, image) in enumerate(zip(cameras, images)):
        panels: List[np.ndarray] = []
        counts: Dict[str, int] = {}
        for stage in _MATCH_DEBUG_STAGES:
            grid = grids[stage][cam_idx]
            count = int(grid.sum(dtype=np.uint64))
            counts[stage] = count
            base = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                (grid.shape[1], grid.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            density = np.log1p(grid.astype(np.float32))
            positive = density[density > 0]
            scale = float(np.percentile(positive, 99.0)) if len(positive) else 1.0
            normalized = np.clip(density / max(scale, 1e-6), 0.0, 1.0)
            heat = cv2.applyColorMap(
                np.round(normalized * 255.0).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )
            occupied = (grid > 0)[..., None]
            panel = np.where(
                occupied,
                cv2.addWeighted(base, 0.35, heat, 0.65, 0.0),
                np.round(base.astype(np.float32) * 0.45).astype(np.uint8),
            )
            panel = cv2.resize(
                panel,
                (grid.shape[1] * 2, grid.shape[0] * 2),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(
                panel,
                f'{stage}: {count:,}',
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        cv2.imwrite(str(output_dir / f'{cam.folder}.jpg'), np.hstack(panels))
        per_camera[cam.folder] = counts

    report = {
        'format': 'roma-match-stage-coverage-v1',
        'stages': list(_MATCH_DEBUG_STAGES),
        'per_camera': per_camera,
    }
    (output_dir / 'coverage.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8'
    )
    return report


_SPATIAL_AUDIT_STATUSES = (
    'min_views',
    'triangulation_failed',
    'outside_bounds',
    'negative_depth',
    'reprojection',
    'triangulation_angle',
    'mask',
    'two_view_core',
    'final_accepted',
)


def _points_in_roi(uv: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    """Return a finite, inclusive pixel-ROI mask without changing the points."""
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x0, y0, x1, y1 = [float(value) for value in roi]
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y1)
    )


def _write_spatial_audit(
    output_dir: Path,
    *,
    frame_idx: int,
    camera_idx: int,
    camera: Camera,
    roi: Sequence[float],
    pair_observation_counts: Mapping[str, int],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Persist an observational audit. This function never filters geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'spatial-audit-camera{int(camera_idx):04d}'
    status_to_code = {
        status: code for code, status in enumerate(_SPATIAL_AUDIT_STATUSES)
    }
    status_counts = {status: 0 for status in _SPATIAL_AUDIT_STATUSES}
    for record in records:
        status = str(record['status'])
        status_counts[status] = status_counts.get(status, 0) + 1

    n_records = len(records)
    positions = np.full((n_records, 3), np.nan, dtype=np.float32)
    uvs = np.full((n_records, 2), np.nan, dtype=np.float32)
    support = np.zeros((n_records,), dtype=np.int16)
    component_ids = np.zeros((n_records,), dtype=np.int64)
    status_codes = np.full((n_records,), -1, dtype=np.int16)
    mean_reprojection_px = np.full((n_records,), np.nan, dtype=np.float32)
    max_reprojection_px = np.full((n_records,), np.nan, dtype=np.float32)
    reprojection_allowed_px = np.full((n_records,), np.nan, dtype=np.float32)
    triangulation_angle_deg = np.full((n_records,), np.nan, dtype=np.float32)
    for row, record in enumerate(records):
        component_ids[row] = int(record['component_id'])
        uvs[row] = np.asarray(record['uv'], dtype=np.float32)
        support[row] = int(record['support'])
        status_codes[row] = int(status_to_code[str(record['status'])])
        position = record.get('position')
        if position is not None:
            positions[row] = np.asarray(position, dtype=np.float32)
        for key, target in (
            ('mean_reprojection_px', mean_reprojection_px),
            ('max_reprojection_px', max_reprojection_px),
            ('reprojection_allowed_px', reprojection_allowed_px),
            ('triangulation_angle_deg', triangulation_angle_deg),
        ):
            value = record.get(key)
            if value is not None:
                target[row] = float(value)

    np.savez_compressed(
        output_dir / f'{prefix}.npz',
        component_id=component_ids,
        uv=uvs,
        support=support,
        position=positions,
        status_code=status_codes,
        mean_reprojection_px=mean_reprojection_px,
        max_reprojection_px=max_reprojection_px,
        reprojection_allowed_px=reprojection_allowed_px,
        triangulation_angle_deg=triangulation_angle_deg,
    )
    report: Dict[str, object] = {
        'format': 'roma-spatial-component-audit-v1',
        'frame_idx': int(frame_idx),
        'camera_index': int(camera_idx),
        'camera_folder': str(camera.folder),
        'image_size': [int(camera.width), int(camera.height)],
        'roi_xyxy_inclusive': [float(value) for value in roi],
        'pair_observations_in_roi': {
            key: int(value) for key, value in pair_observation_counts.items()
        },
        'components_in_roi': int(n_records),
        'component_status_counts': status_counts,
        'status_codes': status_to_code,
        'component_arrays': f'{prefix}.npz',
    }
    (output_dir / f'{prefix}.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8'
    )
    return report


def reconstruct_frame_cloud(
    frame_idx: int,
    images_root: Path,
    cameras: Sequence[Camera],
    image_pattern: str,
    model,
    device: str,
    pair_list: Sequence[Tuple[int, int]],
    roma_num_matches: int,
    roma_min_certainty: float,
    pair_ransac_px: float,
    pair_calibrated_max_error_px: float,
    min_pair_inliers: int,
    same_camera_merge_px: float,
    min_track_views: int,
    reproj_thresh_px: float,
    min_triangulation_angle_deg: float,
    scene_meta: Optional[Dict[str, object]],
    scene_radius_mult: float,
    filter_two_view_by_core: bool,
    two_view_support_radius: float,
    two_view_support_radius_factor: float,
    two_view_support_knn: int,
    masks_root: Optional[Path],
    mask_pattern: str,
    mask_min_fg_alpha: float,
    mask_min_fg_fraction: float,
    color_mode: str = 'medoid',
    color_trim_frac: float = 0.25,
    images_override: Optional[List[np.ndarray]] = None,
    paths_override: Optional[List[Path]] = None,
    masks_override: Optional[List[Optional[np.ndarray]]] = None,
    images_next_override: Optional[List[np.ndarray]] = None,
    paths_next_override: Optional[List[Path]] = None,
    masks_next_override: Optional[List[Optional[np.ndarray]]] = None,
    capture_residual_frames: Optional[np.ndarray] = None,
    temporal_frame_delta: int = 1,
    fresh_motion_compensate: bool = False,
    motion_slack: float = 0.0,
    motion_slack_max_px: float = 12.0,
    calibrated_motion_slack: bool = False,
    warp_cache_out: Optional[List[Optional[DenseMatch]]] = None,
    rs_signs: Optional[np.ndarray] = None,
    rs_readout_frames: float = 0.0,
    rs_axis: str = 'column',
    rs_reference: str = 'content',
    detail_sphere_center: Optional[np.ndarray] = None,
    detail_sphere_radius: float = 0.0,
    detail_sphere_crop_matching: bool = False,
    detail_sphere_crop_radius: float = 0.0,
    detail_sphere_crop_scale: float = 1.25,
    detail_sphere_crop_min_size_px: int = 256,
    detail_sphere_crop_num_matches: int = 6000,
    detail_sphere_camera_max_view_angle_deg: float = 45.0,
    depth_guide_entries: Optional[Sequence[Mapping[str, object]]] = None,
    depth_guide_num_matches: int = 6000,
    match_debug_dir: Optional[Path] = None,
    match_debug_cell_px: int = 16,
    spatial_audit_camera: Optional[int] = None,
    spatial_audit_roi: Optional[Tuple[float, float, float, float]] = None,
) -> FramePointCloud:
    """
    Build a fresh per-frame point cloud by cross-camera RoMa pair matching
    + multiview DLT triangulation. Applies geometry, reprojection,
    triangulation-angle, scene-bounds, and mask filters.

    When `fresh_motion_compensate` is enabled and per-camera capture residuals
    plus next-frame images are supplied, each cross-camera match is shifted to
    the canonical keyframe time (uv - residual * local_image_flow) before pair
    RANSAC and triangulation, so fast-moving surfaces are not rejected as
    epipolar outliers by the sub-frame camera skew. `motion_slack` additionally
    widens the RANSAC/reprojection gates proportionally to local image motion
    to absorb blur-induced matching noise. Dense warps are sampled and released
    one camera at a time so GPU memory is independent of camera count.

    Color sampling is deferred until after the mask filter to avoid work
    for rejected points. Returns a FramePointCloud with per-point
    observations retained so callers can use them for temporal flow warping.
    """
    if images_override is not None and paths_override is not None:
        images, paths = images_override, paths_override
    else:
        paths, images = load_frame_images(images_root, cameras, frame_idx, image_pattern)

    if masks_override is not None:
        masks = masks_override
    else:
        masks = (
            load_frame_masks(masks_root, cameras, frame_idx, mask_pattern)
            if masks_root is not None
            else [None] * len(cameras)
        )

    match_debug = (
        _make_match_debug_grids(cameras, match_debug_cell_px)
        if match_debug_dir is not None
        else None
    )
    audit_active = (
        spatial_audit_camera is not None and spatial_audit_roi is not None
    )
    audit_camera_idx = (
        int(spatial_audit_camera) if spatial_audit_camera is not None else -1
    )
    audit_pair_counts = {
        stage: 0
        for stage in (
            'raw',
            'ransac_kept',
            'calibrated_rejected',
            'calibrated_survived',
            'calibrated_kept',
        )
    }
    audit_records: List[Dict[str, object]] = []

    def _audit_pair_observations(
        stage: str, cam_idx: int, points: np.ndarray
    ) -> None:
        if audit_active and int(cam_idx) == audit_camera_idx:
            audit_pair_counts[stage] += int(
                _points_in_roi(points, spatial_audit_roi).sum()
            )

    # Stage 1: RoMa pair matching (certainty filter + dedup only; RANSAC is
    # deferred so matches can be sync-compensated first).
    raw_pairs: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for cam_a, cam_b in tqdm(pair_list, desc=f'frame {frame_idx} pairs', leave=False):
        kpts_a, kpts_b, _cert = run_roma_pair(
            model, paths[cam_a], paths[cam_b],
            images[cam_a], images[cam_b],
            roma_num_matches, roma_min_certainty,
            pair_ransac_px, min_pair_inliers, device,
            use_fundamental_ransac=False,
            mask_a=masks[cam_a],
            mask_b=masks[cam_b],
            mask_min_fg_alpha=mask_min_fg_alpha,
        )
        if len(kpts_a) == 0:
            continue
        raw_pairs.append((cam_a, cam_b, kpts_a, kpts_b))
        _audit_pair_observations('raw', cam_a, kpts_a)
        _audit_pair_observations('raw', cam_b, kpts_b)
        if match_debug is not None:
            _accumulate_match_debug(
                match_debug['raw'][cam_a], kpts_a, match_debug_cell_px
            )
            _accumulate_match_debug(
                match_debug['raw'][cam_b], kpts_b, match_debug_cell_px
            )

    # Depth-guided matching is deliberately only a crop/pair proposal pass.
    # Raw L3MHET depth never enters the point cloud. Matches from these crops
    # are mapped back to full-image coordinates and then travel through the
    # exact same sync, RANSAC, calibrated epipolar, multiview triangulation,
    # reprojection, angle, bounds, mask, and two-view-core gates below.
    depth_guide_attempted = 0
    depth_guide_succeeded = 0
    depth_guide_matches = 0
    guide_entries = list(depth_guide_entries or [])
    if guide_entries and int(depth_guide_num_matches) > 0:
        with tempfile.TemporaryDirectory(prefix='trigs-depth-guide-') as temp_name:
            temp_dir = Path(temp_name)
            for entry_index, entry in enumerate(
                tqdm(
                    guide_entries,
                    desc=f'frame {frame_idx} depth-guide crops',
                    leave=False,
                )
            ):
                cam_a = int(entry['cam_a'])
                cam_b = int(entry['cam_b'])
                x0a, y0a, x1a, y1a = [int(v) for v in entry['crop_a']]
                x0b, y0b, x1b, y1b = [int(v) for v in entry['crop_b']]
                image_a = np.ascontiguousarray(images[cam_a][y0a:y1a, x0a:x1a])
                image_b = np.ascontiguousarray(images[cam_b][y0b:y1b, x0b:x1b])
                mask_a = (
                    np.ascontiguousarray(masks[cam_a][y0a:y1a, x0a:x1a])
                    if masks[cam_a] is not None
                    else None
                )
                mask_b = (
                    np.ascontiguousarray(masks[cam_b][y0b:y1b, x0b:x1b])
                    if masks[cam_b] is not None
                    else None
                )
                if min(*image_a.shape[:2], *image_b.shape[:2]) < 8:
                    continue
                path_a = temp_dir / f'{entry_index:05d}-a-{cam_a:04d}.png'
                path_b = temp_dir / f'{entry_index:05d}-b-{cam_b:04d}.png'
                Image.fromarray(image_a).save(path_a)
                Image.fromarray(image_b).save(path_b)
                depth_guide_attempted += 1
                kpts_a, kpts_b, _cert = run_roma_pair(
                    model,
                    path_a,
                    path_b,
                    image_a,
                    image_b,
                    int(depth_guide_num_matches),
                    roma_min_certainty,
                    pair_ransac_px,
                    min_pair_inliers,
                    device,
                    use_fundamental_ransac=False,
                    mask_a=mask_a,
                    mask_b=mask_b,
                    mask_min_fg_alpha=mask_min_fg_alpha,
                )
                if len(kpts_a) == 0:
                    continue
                kpts_a = np.ascontiguousarray(
                    kpts_a + np.asarray([x0a, y0a], dtype=np.float32)[None]
                )
                kpts_b = np.ascontiguousarray(
                    kpts_b + np.asarray([x0b, y0b], dtype=np.float32)[None]
                )
                raw_pairs.append((cam_a, cam_b, kpts_a, kpts_b))
                _audit_pair_observations('raw', cam_a, kpts_a)
                _audit_pair_observations('raw', cam_b, kpts_b)
                if match_debug is not None:
                    _accumulate_match_debug(
                        match_debug['raw'][cam_a], kpts_a, match_debug_cell_px
                    )
                    _accumulate_match_debug(
                        match_debug['raw'][cam_b], kpts_b, match_debug_cell_px
                    )
                depth_guide_succeeded += 1
                depth_guide_matches += int(len(kpts_a))
        print(
            f'  [depth-guide] pairs={depth_guide_succeeded}/'
            f'{depth_guide_attempted} matches={depth_guide_matches:,}'
        )

    # A second, detail-only pass gives small/edge image regions their own RoMa
    # inference resolution. Global dense matching can have valid coverage yet
    # still allocate too few feature cells to a smooth face surrounded by a
    # large textured body/floor. Crop matches are mapped back to the original
    # distorted pixel plane before the normal sync, RANSAC, triangulation, and
    # reprojection stages, so this adds observations without bypassing gates.
    detail_crop_attempted = 0
    detail_crop_succeeded = 0
    detail_crop_matches = 0
    detail_crop_cameras = 0
    detail_crop_sizes: List[int] = []
    effective_crop_radius = (
        float(detail_sphere_crop_radius)
        if float(detail_sphere_crop_radius) > 0.0
        else float(detail_sphere_radius)
    )
    crop_active = (
        bool(detail_sphere_crop_matching)
        and detail_sphere_center is not None
        and effective_crop_radius > 0.0
        and int(detail_sphere_crop_num_matches) > 0
    )
    if crop_active:
        center_arr = np.asarray(detail_sphere_center, dtype=np.float64).reshape(3)
        with tempfile.TemporaryDirectory(prefix='trigs-detail-crops-') as temp_name:
            temp_dir = Path(temp_name)
            crop_sources: Dict[
                int,
                Tuple[Path, np.ndarray, Optional[np.ndarray], np.ndarray],
            ] = {}
            for cam_idx, camera in enumerate(cameras):
                bounds = project_detail_sphere_crop_bounds(
                    camera,
                    center_arr,
                    effective_crop_radius,
                    scale=float(detail_sphere_crop_scale),
                    min_size_px=int(detail_sphere_crop_min_size_px),
                )
                if bounds is None:
                    continue
                x0, y0, x1, y1 = bounds
                crop_image = np.ascontiguousarray(images[cam_idx][y0:y1, x0:x1])
                if crop_image.shape[0] < 8 or crop_image.shape[1] < 8:
                    continue
                crop_mask = (
                    np.ascontiguousarray(masks[cam_idx][y0:y1, x0:x1])
                    if masks[cam_idx] is not None
                    else None
                )
                crop_path = temp_dir / f'cam-{cam_idx:04d}.png'
                Image.fromarray(crop_image).save(crop_path)
                crop_sources[cam_idx] = (
                    crop_path,
                    crop_image,
                    crop_mask,
                    np.asarray([x0, y0], dtype=np.float32),
                )
                detail_crop_sizes.append(int(crop_image.shape[0]))

            detail_crop_cameras = int(len(crop_sources))
            # Detail crops must not inherit every global near/far pair. A
            # complete graph across a convex head includes opposite cameras
            # with no common visible surface and was the direct source of the
            # geometrically invalid head matches. Use the same center-ray
            # overlap gate as detail pair augmentation.
            _, overlap_pairs = choose_detail_sphere_camera_pairs(
                cameras,
                center_arr,
                max_view_angle_deg=float(
                    detail_sphere_camera_max_view_angle_deg
                ),
            )
            overlap_pair_set = set(overlap_pairs)
            crop_pairs = [
                (cam_a, cam_b)
                for cam_a, cam_b in pair_list
                if cam_a in crop_sources
                and cam_b in crop_sources
                and (cam_a, cam_b) in overlap_pair_set
            ]
            for cam_a, cam_b in tqdm(
                crop_pairs,
                desc=f'frame {frame_idx} detail crops',
                leave=False,
            ):
                detail_crop_attempted += 1
                path_a, image_a, mask_a, offset_a = crop_sources[cam_a]
                path_b, image_b, mask_b, offset_b = crop_sources[cam_b]
                kpts_a, kpts_b, _cert = run_roma_pair(
                    model,
                    path_a,
                    path_b,
                    image_a,
                    image_b,
                    int(detail_sphere_crop_num_matches),
                    roma_min_certainty,
                    pair_ransac_px,
                    min_pair_inliers,
                    device,
                    use_fundamental_ransac=False,
                    mask_a=mask_a,
                    mask_b=mask_b,
                    mask_min_fg_alpha=mask_min_fg_alpha,
                )
                if len(kpts_a) == 0:
                    continue
                kpts_a = np.ascontiguousarray(kpts_a + offset_a[None, :])
                kpts_b = np.ascontiguousarray(kpts_b + offset_b[None, :])
                raw_pairs.append((cam_a, cam_b, kpts_a, kpts_b))
                _audit_pair_observations('raw', cam_a, kpts_a)
                _audit_pair_observations('raw', cam_b, kpts_b)
                detail_crop_succeeded += 1
                detail_crop_matches += int(len(kpts_a))

        print(
            f'  [detail-crops] cameras={detail_crop_cameras} '
            f'pairs={detail_crop_succeeded}/{detail_crop_attempted} '
            f'matches={detail_crop_matches:,} '
            f'size={min(detail_crop_sizes, default=0)}..'
            f'{max(detail_crop_sizes, default=0)}px'
        )

    # Stage 2: per-camera temporal warps (t -> t+delta) for sync motion
    # compensation and flow-scaled gate slack. A dense RoMa v2 prediction can
    # occupy well over 100 MiB. Keeping one for every camera exhausts even a
    # large GPU on dense rigs, so each camera is built, sampled for all pair
    # endpoints that use it, and freed before the next camera.
    n_cams = len(cameras)
    rs_requested = (
        rs_signs is not None
        and abs(float(rs_readout_frames)) > 1e-9
        and bool(np.any(np.asarray(rs_signs) != 0))
    )
    compensate = bool(fresh_motion_compensate) and (
        (
            capture_residual_frames is not None
            and float(np.max(np.abs(np.asarray(capture_residual_frames)))) > 1e-6
        )
        or rs_requested
    )
    need_flow = compensate or float(motion_slack) > 0.0
    have_next = images_next_override is not None and paths_next_override is not None
    pair_flows: List[List[Optional[np.ndarray]]] = [
        [None, None] for _ in raw_pairs
    ]
    any_warp_ok = False
    if need_flow and have_next:
        pair_queries_by_cam: Dict[
            int, List[Tuple[int, int, np.ndarray]]
        ] = {}
        for pair_idx, (cam_a, cam_b, kpts_a, kpts_b) in enumerate(raw_pairs):
            pair_queries_by_cam.setdefault(cam_a, []).append(
                (pair_idx, 0, kpts_a)
            )
            pair_queries_by_cam.setdefault(cam_b, []).append(
                (pair_idx, 1, kpts_b)
            )

        layout_validated = False
        for c in tqdm(range(n_cams), desc=f'frame {frame_idx} temporal warps', leave=False):
            queries = pair_queries_by_cam.get(c, [])
            if not queries:
                continue
            dense = None
            try:
                mask_b_c = masks_next_override[c] if masks_next_override is not None else None
                dense = build_roma_warp(
                    model, paths[c], paths_next_override[c],
                    images[c], images_next_override[c], device,
                    mask_a=masks[c], mask_b=mask_b_c,
                    mask_min_fg_alpha=mask_min_fg_alpha,
                )
                if not layout_validated:
                    if not validate_warp_layout(model, dense):
                        raise RuntimeError(
                            'RoMa dense-warp tensor layout does not match the '
                            'expected forward-warp convention.'
                        )
                    layout_validated = True

                all_uvs = np.concatenate(
                    [np.asarray(item[2], dtype=np.float32) for item in queries],
                    axis=0,
                )
                all_flows = sample_flow_per_frame(
                    model, dense, all_uvs, temporal_frame_delta
                )
                cursor = 0
                for pair_idx, endpoint, uvs in queries:
                    count = len(uvs)
                    pair_flows[pair_idx][endpoint] = all_flows[
                        cursor:cursor + count
                    ]
                    cursor += count
                any_warp_ok = True
            except Exception as e:
                print(f'  [warn] fresh-comp warp failed for cam {c}: {e}')
            finally:
                dense = None
                if device is None or str(device).startswith('cuda'):
                    torch.cuda.empty_cache()

        # Dense tensors are intentionally never returned: doing so recreates
        # an O(number_of_cameras) GPU allocation. The temporal tracker builds
        # and consumes one camera warp at a time below.
        if warp_cache_out is not None:
            warp_cache_out.clear()
    elif need_flow and not have_next:
        print(
            f'  [warn] frame {frame_idx}: motion compensation/slack requested '
            'but no next-frame images supplied; falling back to uncompensated matching.'
        )
        compensate = False

    residual = (
        np.asarray(capture_residual_frames, dtype=np.float64)
        if capture_residual_frames is not None
        else np.zeros((n_cams,), dtype=np.float64)
    )

    # Stage 3: sync/RS-compensate matches, then pair RANSAC (+ motion-slack rescue).
    pair_matches: List[Tuple[int, int, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = []
    total_kept_pair_matches = 0
    total_rescued = 0
    total_rejected_calibrated = 0
    comp_shift_sq_sum = 0.0
    comp_shift_count = 0
    comp_shift_max = 0.0

    rs_active = rs_requested and any_warp_ok

    # Pre-sample per-pair flows. For rolling shutter, accumulate a
    # flow-weighted mean column per camera: the RS term is referenced to the
    # moving content's mean column, so it is zero-mean over the subject and
    # composes with sync offsets (which already absorb each camera's mean
    # subject-column readout delay — they were solved from the same content).
    rs_ax = _rs_axis_index(rs_axis)
    rs_extent = np.asarray([
        float(cameras[c].width if rs_ax == 0 else cameras[c].height) for c in range(n_cams)
    ])
    rs_u_wsum = np.zeros((n_cams,), dtype=np.float64)
    rs_w_sum = np.zeros((n_cams,), dtype=np.float64)
    for pair_idx, (cam_a, cam_b, kpts_a, kpts_b) in enumerate(raw_pairs):
        fa, fb = pair_flows[pair_idx]
        if fa is not None and fb is not None:
            if rs_active and rs_reference == 'content':
                wa = np.linalg.norm(fa, axis=1)
                wb = np.linalg.norm(fb, axis=1)
                rs_u_wsum[cam_a] += float(np.sum(wa * kpts_a[:, rs_ax].astype(np.float64)))
                rs_w_sum[cam_a] += float(np.sum(wa))
                rs_u_wsum[cam_b] += float(np.sum(wb * kpts_b[:, rs_ax].astype(np.float64)))
                rs_w_sum[cam_b] += float(np.sum(wb))
        else:
            fa = fb = None
            pair_flows[pair_idx] = [None, None]

    rs_u_ref = 0.5 * rs_extent
    if rs_active and rs_reference == 'content':
        has_w = rs_w_sum > 1e-6
        rs_u_ref[has_w] = rs_u_wsum[has_w] / rs_w_sum[has_w]

    def _pixel_tau_frames(cam_idx: int, kpts: np.ndarray) -> np.ndarray:
        """Capture-time offset (frames) of each pixel vs the canonical frame
        time: per-camera sync residual + rolling-shutter term (rs_tau_frames)."""
        sign = float(rs_signs[cam_idx]) if (rs_active and rs_signs is not None) else 0.0
        return rs_tau_frames(
            residual[cam_idx], sign, rs_readout_frames,
            np.asarray(kpts, dtype=np.float64)[:, rs_ax], rs_extent[cam_idx],
            rs_reference, content_ref=rs_u_ref[cam_idx],
        )

    for (cam_a, cam_b, kpts_a, kpts_b), (flow_a, flow_b) in zip(raw_pairs, pair_flows):
        canon_a = canon_b = None
        flow_mag = None
        if flow_a is not None and flow_b is not None:
            flow_mag = 0.5 * (
                np.linalg.norm(flow_a, axis=1) + np.linalg.norm(flow_b, axis=1)
            )
            if compensate:
                shift_a = -_pixel_tau_frames(cam_a, kpts_a)[:, None] * flow_a
                shift_b = -_pixel_tau_frames(cam_b, kpts_b)[:, None] * flow_b
                canon_a = kpts_a.astype(np.float64) + shift_a
                canon_b = kpts_b.astype(np.float64) + shift_b
                shifts = np.concatenate([
                    np.linalg.norm(shift_a, axis=1),
                    np.linalg.norm(shift_b, axis=1),
                ])
                comp_shift_sq_sum += float(np.sum(shifts ** 2))
                comp_shift_count += int(len(shifts))
                comp_shift_max = max(comp_shift_max, float(np.max(shifts, initial=0.0)))

        geom_a = canon_a if canon_a is not None else kpts_a.astype(np.float64)
        geom_b = canon_b if canon_b is not None else kpts_b.astype(np.float64)
        result = pair_ransac_with_motion_slack(
            geom_a, geom_b, flow_mag,
            pair_ransac_px, min_pair_inliers,
            float(motion_slack), float(motion_slack_max_px),
            pair_label=f'{paths[cam_a].name}<->{paths[cam_b].name}',
            camera_a=cameras[cam_a],
            camera_b=cameras[cam_b],
        )
        if result is None:
            continue
        keep, n_rescued = result
        _audit_pair_observations('ransac_kept', cam_a, kpts_a[keep])
        _audit_pair_observations('ransac_kept', cam_b, kpts_b[keep])
        if match_debug is not None:
            _accumulate_match_debug(
                match_debug['ransac_kept'][cam_a],
                kpts_a[keep],
                match_debug_cell_px,
            )
            _accumulate_match_debug(
                match_debug['ransac_kept'][cam_b],
                kpts_b[keep],
                match_debug_cell_px,
            )
        if float(pair_calibrated_max_error_px) > 0.0:
            calibrated_residual = calibrated_pair_epipolar_distance_px(
                cameras[cam_a], cameras[cam_b], geom_a, geom_b
            )
            calibrated_keep = calibrated_residual <= (
                calibrated_gate_px(
                    pair_calibrated_max_error_px, flow_mag,
                    motion_slack, motion_slack_max_px,
                )
                if calibrated_motion_slack
                else float(pair_calibrated_max_error_px)
            )
            calibrated_rejected = keep & ~calibrated_keep
            _audit_pair_observations(
                'calibrated_rejected', cam_a, kpts_a[calibrated_rejected]
            )
            _audit_pair_observations(
                'calibrated_rejected', cam_b, kpts_b[calibrated_rejected]
            )
            calibrated_survived = keep & calibrated_keep
            _audit_pair_observations(
                'calibrated_survived', cam_a, kpts_a[calibrated_survived]
            )
            _audit_pair_observations(
                'calibrated_survived', cam_b, kpts_b[calibrated_survived]
            )
            if match_debug is not None:
                _accumulate_match_debug(
                    match_debug['calibrated_rejected'][cam_a],
                    kpts_a[calibrated_rejected],
                    match_debug_cell_px,
                )
                _accumulate_match_debug(
                    match_debug['calibrated_rejected'][cam_b],
                    kpts_b[calibrated_rejected],
                    match_debug_cell_px,
                )
            total_rejected_calibrated += int(np.sum(keep & ~calibrated_keep))
            keep &= calibrated_keep
            if int(keep.sum()) < int(min_pair_inliers):
                continue
        else:
            _audit_pair_observations(
                'calibrated_survived', cam_a, kpts_a[keep]
            )
            _audit_pair_observations(
                'calibrated_survived', cam_b, kpts_b[keep]
            )
        _audit_pair_observations('calibrated_kept', cam_a, kpts_a[keep])
        _audit_pair_observations('calibrated_kept', cam_b, kpts_b[keep])
        if match_debug is not None:
            _accumulate_match_debug(
                match_debug['calibrated_kept'][cam_a],
                kpts_a[keep],
                match_debug_cell_px,
            )
            _accumulate_match_debug(
                match_debug['calibrated_kept'][cam_b],
                kpts_b[keep],
                match_debug_cell_px,
            )
        total_rescued += n_rescued
        total_kept_pair_matches += int(keep.sum())
        pair_matches.append((
            cam_a, cam_b,
            kpts_a[keep], kpts_b[keep],
            canon_a[keep] if canon_a is not None else None,
            canon_b[keep] if canon_b is not None else None,
            np.linalg.norm(flow_a[keep], axis=1) if flow_a is not None else None,
            np.linalg.norm(flow_b[keep], axis=1) if flow_b is not None else None,
        ))

    components = build_multiview_components(pair_matches, same_camera_merge_px)

    # Accumulate positions and observations; defer color sampling until after
    # the mask filter so masked-out points never pay the sampling cost.
    out_positions: List[np.ndarray] = []
    out_observations: List[List[Tuple[int, np.ndarray]]] = []
    out_obs_uvs_for_mask: List[Optional[Dict[int, np.ndarray]]] = []
    out_support_counts: List[int] = []
    out_audit_record_indices: List[int] = []

    accepted = 0
    rej_min_views = rej_triang = rej_depth = rej_reproj = rej_angle = rej_bounds = rej_mask = 0
    support_hist: Dict[str, int] = {}
    accepted_support_hist: Dict[str, int] = {}

    cand_obs: List[List[Observation]] = []
    cand_audit: List[int] = []
    cand_cam: List[int] = []
    cand_uv: List[np.ndarray] = []
    cand_flow: List[float] = []
    for component_id, comp in enumerate(components):
        obs = average_observations_per_camera(comp)
        n_obs = len(obs)
        support_hist[str(n_obs)] = support_hist.get(str(n_obs), 0) + 1
        audit_record_idx = -1
        if audit_active:
            camera_observation = next(
                (o for o in obs if int(o.cam_idx) == audit_camera_idx), None
            )
            if (
                camera_observation is not None
                and bool(
                    _points_in_roi(
                        np.asarray(camera_observation.uv)[None],
                        spatial_audit_roi,
                    )[0]
                )
            ):
                audit_record_idx = len(audit_records)
                audit_records.append({
                    'component_id': int(component_id),
                    'uv': np.asarray(camera_observation.uv, dtype=np.float64).tolist(),
                    'support': int(n_obs),
                    'status': 'min_views',
                })

        if n_obs < min_track_views:
            rej_min_views += 1
            continue

        cand_obs.append(obs)
        cand_audit.append(audit_record_idx)
        for o in obs:
            cand_cam.append(int(o.cam_idx))
            # Geometry runs on sync-compensated pixels; color/mask/tracking keep raw.
            cand_uv.append(observation_geom_uv(o))
            cand_flow.append(float(o.flow_px))

    # Geometry for every surviving component at once; the gates below apply
    # it in the same order as the per-point checks they replace.
    cand_counts = np.fromiter(
        (len(o) for o in cand_obs), dtype=np.int64, count=len(cand_obs)
    )
    cand_cam_arr = np.asarray(cand_cam, dtype=np.int64)
    cand_uv_arr = np.asarray(cand_uv, dtype=np.float64).reshape(-1, 2)
    mean_flow = _ragged_mean(np.asarray(cand_flow, dtype=np.float64), cand_counts)
    cand_reproj_allowed = reproj_thresh_px + _py_min(
        float(motion_slack) * mean_flow, float(motion_slack_max_px)
    )
    cand_X, cand_tri_ok = triangulate_multiview_dlt_batch(
        cand_cam_arr, cand_uv_arr, cand_counts, cameras
    )
    cand_in_bounds = np.zeros((len(cand_obs),), dtype=bool)
    cand_in_bounds[cand_tri_ok] = within_scene_bounds_batch(
        cand_X[cand_tri_ok], scene_meta, scene_radius_mult
    )
    alive = np.flatnonzero(cand_tri_ok & cand_in_bounds)
    alive_rows = _ragged_take(cand_counts, alive)
    reproj_flat, depths_flat = compute_reprojection_errors_batch(
        cand_X[alive], cand_cam_arr[alive_rows], cand_uv_arr[alive_rows],
        cand_counts[alive], cameras,
    )
    cand_mean_reproj = np.full((len(cand_obs),), np.nan, dtype=np.float64)
    cand_max_reproj = np.full((len(cand_obs),), np.nan, dtype=np.float64)
    cand_neg_depth = np.zeros((len(cand_obs),), dtype=bool)
    cand_mean_reproj[alive] = _ragged_mean(reproj_flat, cand_counts[alive])
    cand_max_reproj[alive] = _ragged_max(reproj_flat, cand_counts[alive])
    cand_neg_depth[alive] = _ragged_any(depths_flat <= 0, cand_counts[alive])
    cand_angle = np.zeros((len(cand_obs),), dtype=np.float64)
    if min_triangulation_angle_deg > 0.0:
        angle_idx = alive[
            ~cand_neg_depth[alive]
            & ~(cand_mean_reproj[alive] > cand_reproj_allowed[alive])
        ]
        angle_rows = _ragged_take(cand_counts, angle_idx)
        cand_angle[angle_idx] = compute_triangulation_angle_deg_batch(
            cand_X[angle_idx], cand_cam_arr[angle_rows],
            cand_counts[angle_idx], cameras,
        )

    tri_ok_list = cand_tri_ok.tolist()
    in_bounds_list = cand_in_bounds.tolist()
    neg_depth_list = cand_neg_depth.tolist()
    mean_reproj_list = cand_mean_reproj.tolist()
    max_reproj_list = cand_max_reproj.tolist()
    allowed_list = cand_reproj_allowed.tolist()
    angle_list = cand_angle.tolist()
    for cand_idx, obs in enumerate(cand_obs):
        audit_record_idx = cand_audit[cand_idx]
        n_obs = len(obs)
        reproj_allowed = allowed_list[cand_idx]

        if not tri_ok_list[cand_idx]:
            if audit_record_idx >= 0:
                audit_records[audit_record_idx]['status'] = 'triangulation_failed'
            rej_triang += 1
            continue
        X = cand_X[cand_idx]

        if audit_record_idx >= 0:
            audit_records[audit_record_idx]['position'] = (
                np.asarray(X, dtype=np.float64).tolist()
            )
            audit_records[audit_record_idx]['reprojection_allowed_px'] = float(
                reproj_allowed
            )

        if not in_bounds_list[cand_idx]:
            if audit_record_idx >= 0:
                audit_records[audit_record_idx]['status'] = 'outside_bounds'
            rej_bounds += 1
            continue

        if audit_record_idx >= 0:
            audit_records[audit_record_idx]['mean_reprojection_px'] = float(
                mean_reproj_list[cand_idx]
            )
            audit_records[audit_record_idx]['max_reprojection_px'] = float(
                max_reproj_list[cand_idx]
            )
        if neg_depth_list[cand_idx]:
            if audit_record_idx >= 0:
                audit_records[audit_record_idx]['status'] = 'negative_depth'
            rej_depth += 1
            continue
        if mean_reproj_list[cand_idx] > reproj_allowed:
            if audit_record_idx >= 0:
                audit_records[audit_record_idx]['status'] = 'reprojection'
            rej_reproj += 1
            continue

        if min_triangulation_angle_deg > 0.0:
            ang = angle_list[cand_idx]
            if audit_record_idx >= 0:
                audit_records[audit_record_idx]['triangulation_angle_deg'] = float(ang)
            if ang < min_triangulation_angle_deg:
                if audit_record_idx >= 0:
                    audit_records[audit_record_idx]['status'] = (
                        'triangulation_angle'
                    )
                rej_angle += 1
                continue

        out_positions.append(X.astype(np.float32))
        out_observations.append([(o.cam_idx, o.uv.astype(np.float32)) for o in obs])
        out_obs_uvs_for_mask.append({o.cam_idx: o.uv for o in obs})
        out_support_counts.append(int(n_obs))
        out_audit_record_indices.append(int(audit_record_idx))
        accepted += 1
        accepted_support_hist[str(n_obs)] = accepted_support_hist.get(str(n_obs), 0) + 1

    # Distortion-aware mask filter (before color sampling saves wasted work).
    if masks_root is not None and out_positions:
        Xs = np.stack(out_positions, axis=0)
        keep = batch_mask_filter_distorted(
            Xs, out_obs_uvs_for_mask, cameras, masks,
            mask_min_fg_alpha, mask_min_fg_fraction,
        )
        for record_idx, kept in zip(out_audit_record_indices, keep):
            if record_idx >= 0 and not bool(kept):
                audit_records[record_idx]['status'] = 'mask'
        out_positions = [p for p, k in zip(out_positions, keep) if k]
        out_observations = [o for o, k in zip(out_observations, keep) if k]
        out_support_counts = [s for s, k in zip(out_support_counts, keep) if k]
        out_audit_record_indices = [
            r for r, k in zip(out_audit_record_indices, keep) if k
        ]
        rej_mask = int((~keep).sum())
        accepted = len(out_positions)

    core_filter_stats: Dict[str, object] = {
        'enabled': int(bool(filter_two_view_by_core)),
        'trusted_three_plus': int(sum(s >= 3 for s in out_support_counts)),
        'two_view_candidates': int(sum(s == 2 for s in out_support_counts)),
        'two_view_kept': int(sum(s == 2 for s in out_support_counts)),
        'two_view_rejected': 0,
        'radii': [],
        'trusted_median_knn': [],
        'support_knn': int(two_view_support_knn),
    }
    if out_positions and filter_two_view_by_core:
        support_arr = np.asarray(out_support_counts, dtype=np.int32)
        keep, core_filter_stats = filter_two_view_by_three_view_core(
            [np.stack(out_positions, axis=0)],
            support_arr,
            enabled=True,
            radius=float(two_view_support_radius),
            radius_factor=float(two_view_support_radius_factor),
            support_knn=int(two_view_support_knn),
        )
        for record_idx, kept in zip(out_audit_record_indices, keep):
            if record_idx >= 0 and not bool(kept):
                audit_records[record_idx]['status'] = 'two_view_core'
        out_positions = [p for p, k in zip(out_positions, keep) if k]
        out_observations = [o for o, k in zip(out_observations, keep) if k]
        out_support_counts = [s for s, k in zip(out_support_counts, keep) if k]
        out_audit_record_indices = [
            r for r, k in zip(out_audit_record_indices, keep) if k
        ]
        accepted = len(out_positions)
        print(
            f"  [two-view-support:fresh] core={core_filter_stats['trusted_three_plus']:,} "
            f"two_view={core_filter_stats['two_view_candidates']:,} "
            f"kept={core_filter_stats['two_view_kept']:,} "
            f"rejected={core_filter_stats['two_view_rejected']:,} "
            f"radius={core_filter_stats['radii']}"
        )

    accepted_support_hist_pre_core = dict(accepted_support_hist)
    accepted_support_hist = {}
    for support in out_support_counts:
        key = str(int(support))
        accepted_support_hist[key] = accepted_support_hist.get(key, 0) + 1

    for record_idx in out_audit_record_indices:
        if record_idx >= 0:
            audit_records[record_idx]['status'] = 'final_accepted'

    spatial_audit_report: Optional[Dict[str, object]] = None
    if audit_active:
        if match_debug_dir is None:
            raise ValueError(
                'A spatial audit requires match_debug_dir for its output files.'
            )
        spatial_audit_report = _write_spatial_audit(
            match_debug_dir,
            frame_idx=frame_idx,
            camera_idx=audit_camera_idx,
            camera=cameras[audit_camera_idx],
            roi=spatial_audit_roi,
            pair_observation_counts=audit_pair_counts,
            records=audit_records,
        )

    match_debug_report: Optional[Dict[str, object]] = None
    if match_debug is not None:
        for observations in out_observations:
            for cam_idx, uv in observations:
                _accumulate_match_debug(
                    match_debug['final_accepted'][cam_idx],
                    np.asarray(uv)[None],
                    match_debug_cell_px,
                )
        match_debug_report = _write_match_debug_grids(
            match_debug_dir,
            cameras,
            images,
            match_debug,
        )

    # Batch color sampling: one numpy call per camera across all accepted points.
    if out_positions:
        positions = np.stack(out_positions, axis=0)
        colors = _batch_compute_colors(
            out_observations,
            images,
            mode=color_mode,
            trim_frac=color_trim_frac,
        )
    else:
        positions = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)

    debug = {
        'pair_count': int(len(pair_list)),
        'pair_pass_count': int(len(raw_pairs)),
        'pair_matches_after_geom': int(total_kept_pair_matches),
        'pair_matches_rescued': int(total_rescued),
        'pair_calibrated_max_error_px': float(pair_calibrated_max_error_px),
        'pair_matches_rejected_calibrated': int(total_rejected_calibrated),
        'pair_geometry_undistorted': 1,
        'detail_crop_enabled': int(bool(crop_active)),
        'detail_crop_radius': float(effective_crop_radius),
        'detail_crop_cameras': int(detail_crop_cameras),
        'detail_crop_pairs_attempted': int(detail_crop_attempted),
        'detail_crop_pairs_succeeded': int(detail_crop_succeeded),
        'detail_crop_matches_before_geom': int(detail_crop_matches),
        'detail_crop_size_min_px': int(min(detail_crop_sizes, default=0)),
        'detail_crop_size_max_px': int(max(detail_crop_sizes, default=0)),
        'depth_guide_enabled': int(bool(guide_entries)),
        'depth_guide_entries': int(len(guide_entries)),
        'depth_guide_pairs_attempted': int(depth_guide_attempted),
        'depth_guide_pairs_succeeded': int(depth_guide_succeeded),
        'depth_guide_matches_before_geom': int(depth_guide_matches),
        'sync_comp_enabled': int(bool(compensate)),
        'rs_comp_enabled': int(bool(rs_active)),
        'sync_comp_rms_shift_px': (
            float(np.sqrt(comp_shift_sq_sum / comp_shift_count))
            if comp_shift_count > 0 else 0.0
        ),
        'sync_comp_max_shift_px': float(comp_shift_max),
        'components_total': int(len(components)),
        'components_accepted': int(accepted),
        'match_debug': match_debug_report,
        'spatial_audit': spatial_audit_report,
        'rej_min_track_views': int(rej_min_views),
        'rej_triangulation': int(rej_triang),
        'rej_negative_depth': int(rej_depth),
        'rej_reprojection': int(rej_reproj),
        'rej_triangulation_angle': int(rej_angle),
        'rej_bounds': int(rej_bounds),
        'rej_mask': int(rej_mask),
        'support_hist': support_hist,
        'accepted_support_hist': accepted_support_hist,
        'accepted_support_hist_pre_core': accepted_support_hist_pre_core,
        'two_view_core_filter': core_filter_stats,
        'color_mode': str(color_mode),
        'color_trim_frac': float(color_trim_frac),
    }
    return FramePointCloud(
        frame_idx=frame_idx,
        positions=positions,
        colors=colors,
        observations=out_observations,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# RoMa temporal tracking
# ---------------------------------------------------------------------------


def track_points_to_next_keyframe(
    cloud_t: FramePointCloud,
    cameras: Sequence[Camera],
    model,
    device: str,
    paths_t: Sequence[Path],
    images_t: Sequence[np.ndarray],
    paths_tnext: Sequence[Path],
    images_tnext: Sequence[np.ndarray],
    temporal_min_certainty: float,
    temporal_reproj_thresh_px: float,
    min_tracked_views: int,
    min_triangulation_angle_deg: float,
    filter_two_view_by_core: bool,
    two_view_support_radius: float,
    two_view_support_radius_factor: float,
    two_view_support_knn: int,
    scene_meta: Optional[Dict[str, object]],
    scene_radius_mult: float,
    masks_t: Optional[Sequence[Optional[np.ndarray]]],
    masks_tnext: Optional[Sequence[Optional[np.ndarray]]],
    mask_min_fg_alpha: float,
    mask_min_fg_fraction: float,
    capture_residual_t: Optional[np.ndarray] = None,
    capture_residual_tnext: Optional[np.ndarray] = None,
    temporal_frame_delta: int = 1,
    temporal_cycle_thresh_px: float = 0.0,
    sync_motion_compensate: bool = True,
    warps_override: Optional[List[Optional[DenseMatch]]] = None,
    motion_slack: float = 0.0,
    motion_slack_max_px: float = 12.0,
    temporal_min_certainty_fast: float = 0.0,
    temporal_fast_flow_px: float = 8.0,
    rs_signs: Optional[np.ndarray] = None,
    rs_readout_frames: float = 0.0,
    rs_axis: str = 'column',
    rs_reference: str = 'content',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    For each 3D point in cloud_t with observations in cameras [c1, c2, ...]:
      - for each observing camera c, warp observation pixel uv_c(t) to
        uv_c(t+step) using the (c_t -> c_{t+step}) RoMa dense warp
      - keep observations whose temporal certainty >= threshold
      - if at least min_tracked_views warped observations survive, triangulate
        them to get X(t+step) and validate (reproj, angle, bounds, mask)

    Returns parallel arrays:
      positions_t_tracked    [M, 3]
      positions_tnext_tracked[M, 3]
      colors_tracked         [M, 3]  colors from keyframe t
      src_indices_tracked     [M] source indices into cloud_t.positions
      stats                   dict
    """
    n_cams = len(cameras)
    if len(cloud_t.positions) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            {},
        )

    stats = {
        'points_input': int(len(cloud_t.positions)),
        'rej_no_warp': 0,
        'rej_low_certainty': 0,
        'rej_cycle': 0,
        'rej_min_views': 0,
        'rej_triangulation': 0,
        'rej_reprojection': 0,
        'rej_angle': 0,
        'rej_bounds': 0,
        'rej_mask': 0,
        'accepted': 0,
    }

    # Batch RoMa sampling per camera.
    per_cam_queries: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for p_idx, obs_list in enumerate(cloud_t.observations):
        for cam_idx, uv in obs_list:
            per_cam_queries.setdefault(cam_idx, []).append((p_idx, uv))

    # cam -> (canonical uv at t, canonical uv at t+delta, actual uv in target image, certainty)
    warped_obs: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = {}
    # Magnitude for per-frame flow; the signed value orders the two capture
    # times (negative when tracking back to t - delta for centred velocity).
    delta_nominal = max(abs(int(temporal_frame_delta)), 1)
    delta_signed = int(temporal_frame_delta) if int(temporal_frame_delta) != 0 else 1
    residual_t = (
        np.asarray(capture_residual_t, dtype=np.float64)
        if capture_residual_t is not None
        else np.zeros((n_cams,), dtype=np.float64)
    )
    residual_tnext = (
        np.asarray(capture_residual_tnext, dtype=np.float64)
        if capture_residual_tnext is not None
        else np.zeros((n_cams,), dtype=np.float64)
    )

    rs_active = (
        rs_signs is not None
        and abs(float(rs_readout_frames)) > 1e-9
        and bool(np.any(np.asarray(rs_signs) != 0))
    )

    layout_validated = False

    for cam_idx, queries in tqdm(
        per_cam_queries.items(),
        desc=f'temporal warps {cloud_t.frame_idx}->next',
        leave=False,
        total=len(per_cam_queries),
    ):
        dense: Optional[DenseMatch] = None
        try:
            if (
                warps_override is not None
                and cam_idx < len(warps_override)
                and warps_override[cam_idx] is not None
            ):
                dense = warps_override[cam_idx]
            else:
                mask_a = masks_t[cam_idx] if masks_t is not None else None
                mask_b = masks_tnext[cam_idx] if masks_tnext is not None else None
                dense = build_roma_warp(
                    model,
                    paths_t[cam_idx],
                    paths_tnext[cam_idx],
                    images_t[cam_idx],
                    images_tnext[cam_idx],
                    device,
                    mask_a=mask_a,
                    mask_b=mask_b,
                    mask_min_fg_alpha=mask_min_fg_alpha,
                )
            if not layout_validated:
                if not validate_warp_layout(model, dense):
                    raise RuntimeError(
                        'RoMa dense-warp tensor layout does not match the '
                        'expected forward-warp convention.'
                    )
                layout_validated = True

            dense_h_b, dense_w_b = int(dense.h_b), int(dense.w_b)
            uvs_a = np.stack([q[1] for q in queries], axis=0).astype(np.float32)
            uvs_b, certs = sample_warp_at_uvs(model, dense, uvs_a)

            # Rolling-shutter term for this camera (rs_tau_frames). Legacy 'content' reference: flow-weighted
            # mean coordinate of the tracked observations; 'midpoint': the sensor middle (shared convention).
            rs_ax = _rs_axis_index(rs_axis)
            rs_sign_c = float(rs_signs[cam_idx]) if (rs_active and rs_signs is not None) else 0.0
            rs_extent_c = float(cameras[cam_idx].width if rs_ax == 0 else cameras[cam_idx].height)
            rs_coef = 0.0  # nonzero <=> a rolling-shutter term applies to this camera
            rs_u_ref_c = 0.5 * rs_extent_c
            if rs_sign_c != 0.0:
                rs_coef = rs_sign_c * float(rs_readout_frames) / max(rs_extent_c, 1.0)
                if rs_reference == 'content':
                    w_rs = np.linalg.norm(
                        np.asarray(uvs_b, dtype=np.float64)
                        - np.asarray(uvs_a, dtype=np.float64),
                        axis=1,
                    )
                    if float(np.sum(w_rs)) > 1e-6:
                        rs_u_ref_c = float(
                            np.sum(w_rs * uvs_a[:, rs_ax].astype(np.float64))
                            / np.sum(w_rs)
                        )

            cycle_ok = np.ones((len(uvs_a),), dtype=bool)
            if (
                float(temporal_cycle_thresh_px) > 0.0
                and getattr(model, 'supports_reverse', False)
            ):
                uvs_back, _ = sample_warp_at_uvs(
                    model, dense, uvs_b, reverse=True
                )
                cycle_err = np.linalg.norm(uvs_back - uvs_a, axis=1)
                flow_pf = np.linalg.norm(
                    np.asarray(uvs_b, dtype=np.float64)
                    - np.asarray(uvs_a, dtype=np.float64),
                    axis=1,
                ) / float(delta_nominal)
                cycle_allowed = float(temporal_cycle_thresh_px) + np.minimum(
                    float(motion_slack) * flow_pf,
                    float(motion_slack_max_px),
                )
                cycle_ok = cycle_err <= cycle_allowed
        except Exception as e:
            print(f'  [warn] temporal warp failed for cam {cam_idx}: {e}')
            continue
        finally:
            dense = None
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        for row_idx, ((p_idx, uv_a), uv_b, c) in enumerate(zip(queries, uvs_b, certs)):
            if (
                uv_b[0] < 0
                or uv_b[0] > dense_w_b - 1
                or uv_b[1] < 0
                or uv_b[1] > dense_h_b - 1
            ):
                continue
            uv_a64 = np.asarray(uv_a, dtype=np.float64)
            uv_b64 = np.asarray(uv_b, dtype=np.float64)
            obs_flow_pf = float(
                np.linalg.norm(uv_b64 - uv_a64) / float(delta_nominal)
            )
            # Motion blur legitimately lowers warp certainty on fast pixels;
            # let them in at a lower floor instead of losing the whole limb.
            cert_floor = float(temporal_min_certainty)
            if (
                float(temporal_min_certainty_fast) > 0.0
                and obs_flow_pf >= float(temporal_fast_flow_px)
            ):
                cert_floor = min(cert_floor, float(temporal_min_certainty_fast))
            if c < cert_floor:
                stats['rej_low_certainty'] += 1
                continue
            if not cycle_ok[row_idx]:
                stats['rej_cycle'] += 1
                continue
            uv_t_canon = uv_a64
            uv_tnext_canon = uv_b64

            if sync_motion_compensate and (
                capture_residual_t is not None
                or capture_residual_tnext is not None
                or rs_coef != 0.0
            ):
                tau_a = float(rs_tau_frames(residual_t[cam_idx], rs_sign_c, rs_readout_frames,
                                            uv_a64[rs_ax], rs_extent_c, rs_reference, rs_u_ref_c))
                tau_b = float(rs_tau_frames(residual_tnext[cam_idx], rs_sign_c, rs_readout_frames,
                                            uv_b64[rs_ax], rs_extent_c, rs_reference, rs_u_ref_c))
                observed_interval = float(delta_signed) + tau_b - tau_a
                if abs(observed_interval) > 1e-4:
                    image_velocity = (uv_b64 - uv_a64) / observed_interval
                    # Shift each asynchronous observation to the requested global times.
                    uv_t_canon = uv_a64 - tau_a * image_velocity
                    uv_tnext_canon = uv_b64 - tau_b * image_velocity

            warped_obs.setdefault(p_idx, {})[cam_idx] = (
                uv_t_canon, uv_tnext_canon, uv_b64, float(c), obs_flow_pf
            )

    tracked_X_t: List[np.ndarray] = []
    tracked_X_tnext: List[np.ndarray] = []
    tracked_src_indices: List[int] = []
    tracked_obs_uv_tnext: List[Dict[int, np.ndarray]] = []
    tracked_support_counts: List[int] = []

    # Gate each point on its warped views, then triangulate and validate all
    # candidates at once with the same gates and counters as the per-point
    # checks this replaces. Both endpoints use the identical camera subset.
    # When sync offsets are supplied, observations have already been shifted
    # to common global times using the measured image displacement.
    cand_src: List[int] = []
    cand_maps: List[Dict[int, Tuple]] = []
    cand_cam: List[int] = []
    cand_uv_t: List[np.ndarray] = []
    cand_uv_tnext: List[np.ndarray] = []
    cand_flow: List[float] = []
    for p_idx in range(len(cloud_t.positions)):
        obs_map = warped_obs.get(p_idx, {})
        if len(obs_map) == 0:
            stats['rej_no_warp'] += 1
            continue
        if len(obs_map) < min_tracked_views:
            stats['rej_min_views'] += 1
            continue
        cand_src.append(int(p_idx))
        cand_maps.append(obs_map)
        for c, values in obs_map.items():
            cand_cam.append(int(c))
            cand_uv_t.append(values[0])
            cand_uv_tnext.append(values[1])
            cand_flow.append(values[4])

    cand_counts = np.fromiter(
        (len(m) for m in cand_maps), dtype=np.int64, count=len(cand_maps)
    )
    cand_cam_arr = np.asarray(cand_cam, dtype=np.int64)
    cand_uv_t_arr = np.asarray(cand_uv_t, dtype=np.float64).reshape(-1, 2)
    cand_uv_tnext_arr = np.asarray(cand_uv_tnext, dtype=np.float64).reshape(-1, 2)
    X_t_all, ok_t = triangulate_multiview_dlt_batch(
        cand_cam_arr, cand_uv_t_arr, cand_counts, cameras
    )
    X_tnext_all, ok_tnext = triangulate_multiview_dlt_batch(
        cand_cam_arr, cand_uv_tnext_arr, cand_counts, cameras
    )
    tri_ok = ok_t & ok_tnext
    stats['rej_triangulation'] += int(np.count_nonzero(~tri_ok))

    in_bounds = np.zeros((len(cand_maps),), dtype=bool)
    in_bounds[tri_ok] = (
        within_scene_bounds_batch(X_t_all[tri_ok], scene_meta, scene_radius_mult)
        & within_scene_bounds_batch(X_tnext_all[tri_ok], scene_meta, scene_radius_mult)
    )
    stats['rej_bounds'] += int(np.count_nonzero(tri_ok & ~in_bounds))
    alive = np.flatnonzero(tri_ok & in_bounds)

    # Widen the reprojection gate proportionally to local image motion so
    # blur-noisy fast points survive; static points keep the strict gate.
    mean_flow_pf = _ragged_mean(np.asarray(cand_flow, dtype=np.float64), cand_counts)
    reproj_allowed = temporal_reproj_thresh_px + _py_min(
        float(motion_slack) * mean_flow_pf, float(motion_slack_max_px)
    )
    rows = _ragged_take(cand_counts, alive)
    sub_counts = cand_counts[alive]
    reproj_t, depths_t = compute_reprojection_errors_batch(
        X_t_all[alive], cand_cam_arr[rows], cand_uv_t_arr[rows], sub_counts, cameras
    )
    reproj, depths = compute_reprojection_errors_batch(
        X_tnext_all[alive], cand_cam_arr[rows], cand_uv_tnext_arr[rows], sub_counts, cameras
    )
    allowed_alive = reproj_allowed[alive]
    reproj_ok = ~(
        _ragged_any(depths_t <= 0, sub_counts)
        | (_ragged_mean(reproj_t, sub_counts) > allowed_alive)
        | _ragged_any(depths <= 0, sub_counts)
        | (_ragged_mean(reproj, sub_counts) > allowed_alive)
    )
    stats['rej_reprojection'] += int(np.count_nonzero(~reproj_ok))
    alive = alive[reproj_ok]

    if min_triangulation_angle_deg > 0.0:
        rows = _ragged_take(cand_counts, alive)
        sub_counts = cand_counts[alive]
        ang_t = compute_triangulation_angle_deg_batch(
            X_t_all[alive], cand_cam_arr[rows], sub_counts, cameras
        )
        ang_tnext = compute_triangulation_angle_deg_batch(
            X_tnext_all[alive], cand_cam_arr[rows], sub_counts, cameras
        )
        angle_ok = ~(_py_min(ang_t, ang_tnext) < min_triangulation_angle_deg)
        stats['rej_angle'] += int(np.count_nonzero(~angle_ok))
        alive = alive[angle_ok]

    # The mask vote runs as one batched call after this loop.
    tracked_X_t.extend(X_t_all[alive].astype(np.float32))
    tracked_X_tnext.extend(X_tnext_all[alive].astype(np.float32))
    for cand_idx in alive.tolist():
        obs_map = cand_maps[cand_idx]
        tracked_src_indices.append(cand_src[cand_idx])
        tracked_obs_uv_tnext.append({c: values[2] for c, values in obs_map.items()})
        tracked_support_counts.append(int(len(obs_map)))
        stats['accepted'] += 1

    pos_t = np.zeros((0, 3), dtype=np.float32)
    pos_tnext = np.zeros((0, 3), dtype=np.float32)
    src_idx = np.zeros((0,), dtype=np.int64)
    support_arr = np.zeros((0,), dtype=np.int32)

    if tracked_X_t:
        pos_t = np.stack(tracked_X_t, axis=0)
        pos_tnext = np.stack(tracked_X_tnext, axis=0)
        src_idx = np.asarray(tracked_src_indices, dtype=np.int64)
        support_arr = np.asarray(tracked_support_counts, dtype=np.int32)

        # One batched mask vote for every survivor, matching how the fresh
        # path filters its cloud. Per-point calls project the full camera rig
        # for a single point and cost hours at million-point scale.
        if masks_tnext is not None:
            keep_fg = batch_mask_filter_distorted(
                pos_tnext,
                tracked_obs_uv_tnext,
                cameras,
                masks_tnext,
                mask_min_fg_alpha,
                mask_min_fg_fraction,
            )
            stats['rej_mask'] += int((~keep_fg).sum())
            pos_t = pos_t[keep_fg]
            pos_tnext = pos_tnext[keep_fg]
            src_idx = src_idx[keep_fg]
            support_arr = support_arr[keep_fg]

    if len(pos_t):
        keep, core_filter_stats = filter_two_view_by_three_view_core(
            [pos_t, pos_tnext],
            support_arr,
            enabled=bool(filter_two_view_by_core),
            radius=float(two_view_support_radius),
            radius_factor=float(two_view_support_radius_factor),
            support_knn=int(two_view_support_knn),
        )
        if filter_two_view_by_core:
            pos_t = pos_t[keep]
            pos_tnext = pos_tnext[keep]
            src_idx = src_idx[keep]
            print(
                f"  [two-view-support:temporal] core={core_filter_stats['trusted_three_plus']:,} "
                f"two_view={core_filter_stats['two_view_candidates']:,} "
                f"kept={core_filter_stats['two_view_kept']:,} "
                f"rejected={core_filter_stats['two_view_rejected']:,} "
                f"radii={core_filter_stats['radii']}"
            )
        stats['two_view_core_filter'] = core_filter_stats
        stats['accepted'] = int(len(pos_t))

        # Use the keyframe-t colors because pos_t is the canonical point at time t.
        cols = cloud_t.colors[src_idx].astype(np.float32, copy=False)
    else:
        cols = np.zeros((0, 3), dtype=np.float32)
        # Everything triangulated may still have been voted out by the masks.
        stats['accepted'] = 0
        stats['two_view_core_filter'] = {
            'enabled': int(bool(filter_two_view_by_core)),
            'trusted_three_plus': 0,
            'two_view_candidates': 0,
            'two_view_kept': 0,
            'two_view_rejected': 0,
            'radii': [],
            'trusted_median_knn': [],
            'support_knn': int(two_view_support_knn),
        }
    return pos_t, pos_tnext, cols, src_idx, stats


# ---------------------------------------------------------------------------
# Spatial KNN velocity (fallback for fresh points that weren't tracked)
# ---------------------------------------------------------------------------


def median_nn_distance(points: np.ndarray, sample_cap: int = 20000) -> float:
    if len(points) < 2:
        return 0.0
    pts = (
        points[np.random.choice(len(points), size=sample_cap, replace=False)]
        if len(points) > sample_cap
        else points
    )
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return float(np.median(dists[:, 1]))


def mutual_nn_velocities(
    pos_t: np.ndarray, pos_t1: np.ndarray, max_radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    velocities = np.zeros_like(pos_t, dtype=np.float32)
    valid = np.zeros(len(pos_t), dtype=bool)
    if len(pos_t) == 0 or len(pos_t1) == 0:
        return velocities, valid
    tree_t1 = cKDTree(pos_t1)
    tree_t = cKDTree(pos_t)
    d01, i01 = tree_t1.query(pos_t, k=1)
    d10, i10 = tree_t.query(pos_t1, k=1)
    for i, (d, j) in enumerate(zip(d01, i01)):
        if not np.isfinite(d) or d > max_radius:
            continue
        if i10[j] != i:
            continue
        velocities[i] = (pos_t1[j] - pos_t[i]).astype(np.float32)
        valid[i] = True
    return velocities, valid


def compute_adaptive_velocity_radius(
    pos_t: np.ndarray,
    pos_t1: np.ndarray,
    radius_factor: float,
    min_radius: float,
    max_radius: float,
) -> float:
    scale = max(median_nn_distance(pos_t), median_nn_distance(pos_t1))
    if scale <= 0:
        return min_radius
    return float(np.clip(scale * radius_factor, min_radius, max_radius))


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Prepare TriGSVanilla COLMAP NPZ with RoMa temporal tracking.'
    )
    p.add_argument('--reference-json', type=str, required=True)
    p.add_argument('--images-root', type=str, required=True)
    p.add_argument('--image-pattern', type=str, default=None,
                   help="Per-camera image filename pattern. Default: '{frame:06d}.jpg' or '{frame:06d}.png', "
                        'whichever exists under --images-root.')
    p.add_argument('--output-path', type=str, required=True)
    p.add_argument('--stats-path', type=str, default=None)
    p.add_argument('--frame-start', type=int, required=True)
    p.add_argument('--frame-end', type=int, required=True)
    p.add_argument('--frame-step', type=int, default=5)
    p.add_argument(
        '--input-frame-start',
        type=int,
        default=None,
        help=(
            'First available source-image frame. Default: the first frame present for every camera '
            'under --images-root (else --frame-start). '
            'Set this earlier than the keyframe when sync offsets may select '
            'an adjacent source frame.'
        ),
    )
    p.add_argument(
        '--input-frame-end',
        type=int,
        default=None,
        help=(
            'Last available source-image frame. Default: the last frame present for every camera '
            'under --images-root (else --frame-end). '
            'This may extend past the last requested keyframe/temporal target.'
        ),
    )
    p.add_argument(
        '--match-debug-dir',
        type=str,
        default=None,
        help=(
            'Optional directory for per-camera stage-coverage panels showing '
            'raw RoMa observations, RANSAC survivors, calibrated rejects, '
            'calibrated survivors, and final accepted observations.'
        ),
    )
    p.add_argument(
        '--match-debug-cell-px',
        type=int,
        default=16,
        help='Pixel bin size used by --match-debug-dir coverage panels.',
    )
    p.add_argument(
        '--depth-guide-plan',
        type=str,
        default=None,
        help=(
            'Optional trigs-depth-guided-roma-plan-v1 JSON. The plan only '
            'adds RoMa crop inference passes for automatically detected '
            'coverage holes; depth is never imported as point geometry.'
        ),
    )
    p.add_argument(
        '--depth-guide-num-matches',
        type=int,
        default=6000,
        help='RoMa samples requested for each depth-guide crop pair.',
    )
    p.add_argument(
        '--depth-guide-apply-all-frames',
        action='store_true',
        help=(
            'Apply a static depth guide to every keyframe. By default it is '
            'used only for the plan target_frame, preventing stale crops on '
            'moving sequences.'
        ),
    )
    p.add_argument(
        '--spatial-audit-camera',
        type=int,
        default=None,
        help=(
            'Optional zero-based camera index to audit. Requires '
            '--spatial-audit-roi and --match-debug-dir. This only records '
            'diagnostics and never changes matching or filtering.'
        ),
    )
    p.add_argument(
        '--spatial-audit-roi',
        type=str,
        default=None,
        help=(
            "Inclusive image-space audit rectangle 'x0,y0,x1,y1'. Requires "
            '--spatial-audit-camera and --match-debug-dir.'
        ),
    )
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument(
        '--sync-json',
        type=str,
        default=None,
        help=(
            'Optional sync.json of per-camera offsets in seconds. Convention: '
            'physical_time=(source_frame-frame_start)/fps+offset. RoMa init '
            'will choose the nearest source frame per camera for each global time.'
        ),
    )
    p.add_argument('--solve-offsets-out', type=str, default=None,
               help='If set, solve per-camera sub-frame offsets from tracked '
                    'trajectories and write them (seconds, mean-zero) to this JSON.')
    p.add_argument('--device', type=str, default='cuda')

    # TriGSVanilla compatibility
    p.add_argument('--trigs-logical-frame-start', type=int, default=0)
    p.add_argument('--trigs-velocity-per-frame', action='store_true', default=True)
    p.add_argument(
        '--no-trigs-velocity-per-frame',
        action='store_false',
        dest='trigs_velocity_per_frame',
    )

    # Cross-camera pair selection
    p.add_argument('--camera-neighbors', type=int, default=4)
    p.add_argument('--wide-baseline-per-camera', type=int, default=1)

    # RoMa matching
    p.add_argument('--roma-backend', choices=('v1', 'v2'), default='v2',
                   help='RoMa implementation. v1 uses romatch; v2 uses romav2/DINOv3.')
    p.add_argument('--roma-model', choices=('outdoor', 'indoor', 'tiny-outdoor'), default='outdoor',
                   help='RoMa v1 model family. Ignored by v2.')
    p.add_argument('--roma-v2-setting',
                   choices=('turbo', 'fast', 'base', 'precise', 'mega1500', 'scannet1500', 'wxbs', 'satast'),
                   default='fast', help='Official RoMa v2 inference setting.')
    p.add_argument('--roma-symmetric', action=argparse.BooleanOptionalAction, default=True,
                   help='Enable bidirectional/symmetric RoMa v1 predictions.')
    p.add_argument('--roma-upsample-preds', action=argparse.BooleanOptionalAction, default=True,
                   help='Enable RoMa v1 high-resolution refinement.')
    p.add_argument('--roma-v2-mask-oversample', type=float, default=2.0,
                   help='Oversample v2 correspondences before endpoint mask filtering.')
    p.add_argument(
        '--roma-v2-spatial-grid',
        type=int,
        default=8,
        help=(
            'Reserve deterministic RoMa v2 source anchors across an NxN image '
            'grid so small or border regions are not starved by a large '
            'textured floor/background, while identical source pixels can '
            'merge across camera pairs into multiview tracks. 0 disables.'
        ),
    )
    p.add_argument(
        '--roma-v2-spatial-fraction',
        type=float,
        default=0.5,
        help=(
            'Fraction of --roma-num-matches reserved for spatial-grid sampling; '
            'the remainder uses RoMa v2 native global confidence sampling.'
        ),
    )
    p.add_argument('--roma-num-matches', type=int, default=6000)
    p.add_argument('--roma-min-certainty', type=float, default=0.05)
    p.add_argument('--pair-ransac-px', type=float, default=0.85)
    p.add_argument(
        '--pair-calibrated-max-error-px',
        type=float,
        default=2.5,
        help=(
            'After pair-fitted RANSAC, also require each correspondence to '
            'agree with the supplied camera calibration by at most this many '
            'pixels on the undistorted image planes. 0 disables. This is '
            'recommended when admitting 2-view detail points.'
        ),
    )
    p.add_argument('--min-pair-inliers', type=int, default=60)

    # Cross-camera geometry
    p.add_argument('--same-camera-merge-px', type=float, default=0.8)
    p.add_argument('--min-track-views', type=int, default=2)
    p.add_argument('--reproj-thresh-px', type=float, default=2.5)
    p.add_argument('--min-triangulation-angle-deg', type=float, default=5.0)
    p.add_argument('--scene-radius-mult', type=float, default=1.5)
    p.add_argument(
        '--filter-two-view-by-three-view-core',
        action=argparse.BooleanOptionalAction,
        help=(
            'Keep all 3+-view points and retain 2-view density only where its '
            'local kNN spacing is comparable to the trusted 3+-view surface. '
            'Applied to fresh geometry and both temporal endpoints.'
        ),
        default=True,
    )
    p.add_argument(
        '--two-view-support-radius',
        type=float,
        default=0.0,
        help=(
            'Maximum k-th-neighbour distance for a 2-view point. 0 chooses it '
            'automatically from the trusted 3+-view spacing.'
        ),
    )
    p.add_argument(
        '--two-view-support-radius-factor',
        type=float,
        default=2.5,
        help='Automatic support radius = this factor times trusted median kNN spacing.',
    )
    p.add_argument(
        '--two-view-support-knn',
        type=int,
        default=8,
        help='Neighbour rank used to identify sparse 2-view trails (default: 8).',
    )

    # Dedup
    p.add_argument('--dedup-radius', type=float, default=0.0)
    p.add_argument('--target-points-per-frame', type=int, default=70000)
    p.add_argument('--dedup-radius-max', type=float, default=0.10)
    p.add_argument(
        '--detail-sphere-center',
        type=str,
        default=None,
        help=(
            "Optional world-space detail sphere center as 'x,y,z'. When used "
            "with --detail-sphere-radius and --detail-sphere-target-fraction, "
            "the target-point cap reserves a fixed share for this region."
        ),
    )
    p.add_argument(
        '--detail-sphere-radius',
        type=float,
        default=0.0,
        help='World-space radius of the reserved detail region. 0 disables.',
    )
    p.add_argument(
        '--detail-sphere-target-fraction',
        type=float,
        default=0.0,
        help=(
            'Fraction of --target-points-per-frame reserved for the detail '
            'sphere. Unused quota is returned to the rest of the scene.'
        ),
    )
    p.add_argument(
        '--detail-sphere-camera-pairs',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'When a detail sphere is active, add overlap-aware RoMa pairs '
            'among cameras whose images contain its projected center. Pair '
            'overlap is controlled by --detail-sphere-camera-max-view-angle-deg. '
            'Enabled by default; use --no-detail-sphere-camera-pairs to disable.'
        ),
    )
    p.add_argument(
        '--detail-sphere-camera-min-margin-px',
        type=float,
        default=0.0,
        help=(
            'Require the projected detail center to be this many pixels '
            'inside each image before adding that camera to the detail graph.'
        ),
    )
    p.add_argument(
        '--detail-sphere-camera-max-view-angle-deg',
        type=float,
        default=45.0,
        help=(
            'Maximum angle between detail-center-to-camera rays for detail '
            'pair augmentation and crop matching. This rejects opposite-side '
            'views that do not share the same convex surface (default: 45). '
            'Use 180 to restore the old complete graph.'
        ),
    )
    p.add_argument(
        '--detail-sphere-crop-matching',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Run an additional RoMa pass on per-camera crops covering the '
            'projected detail sphere, then map matches back to full-image '
            'coordinates. Enabled by default when a detail sphere is active.'
        ),
    )
    p.add_argument(
        '--detail-sphere-crop-scale',
        type=float,
        default=1.25,
        help='Scale applied to the projected-sphere crop bounds (default: 1.25).',
    )
    p.add_argument(
        '--detail-sphere-crop-radius',
        type=float,
        default=0.0,
        help=(
            'World-space radius used only to construct detail image crops. '
            '0 reuses --detail-sphere-radius. A smaller value can focus RoMa '
            'on an inner face/head region without changing output sampling.'
        ),
    )
    p.add_argument(
        '--detail-sphere-crop-min-size-px',
        type=int,
        default=256,
        help='Minimum square detail crop size before RoMa inference.',
    )
    p.add_argument(
        '--detail-sphere-crop-num-matches',
        type=int,
        default=6000,
        help='RoMa samples requested per detail-crop camera pair; 0 disables.',
    )

    # Masks
    p.add_argument('--masks-root', type=str, default=None)
    p.add_argument('--mask-pattern', type=str, default='{frame:06d}.png')
    p.add_argument('--mask-min-fg-alpha', type=float, default=0.5)
    p.add_argument('--mask-min-fg-fraction', type=float, default=0.75)

    # Temporal tracking
    p.add_argument('--disable-temporal-tracking', action='store_true')
    p.add_argument('--temporal-min-certainty', type=float, default=0.15)
    p.add_argument('--temporal-reproj-thresh-px', type=float, default=2.5)
    p.add_argument('--min-tracked-views', type=int, default=2)
    p.add_argument('--temporal-cycle-thresh-px', type=float, default=2.0,
                   help='Forward/backward dense-warp cycle threshold. 0 disables.')
    p.add_argument('--sync-motion-compensate', action=argparse.BooleanOptionalAction, default=True,
                   help='Shift asynchronous per-camera temporal observations to common global times before triangulation.')
    p.add_argument('--fresh-motion-compensate', action=argparse.BooleanOptionalAction, default=True,
                   help='Sync-compensate cross-camera FRESH matches: shift each matched pixel to '
                        'the canonical keyframe time using the local t->t+delta image flow before '
                        'pair RANSAC and triangulation, so fast-moving surfaces are not rejected '
                        'as epipolar outliers by sub-frame camera skew. Requires --sync-json; '
                        'no-op without it. Dense warps are processed one camera at a time.')
    p.add_argument('--motion-slack', type=float, default=1.0,
                   help='Extra px of geometric tolerance per px/frame of local image motion, '
                        'applied to pair-RANSAC rescue, fresh reprojection, temporal cycle, and '
                        'temporal reprojection gates. Static pixels (flow ~0) keep the base '
                        'thresholds. Absorbs motion-blur matching noise; 0 disables.')
    p.add_argument('--motion-slack-max-px', type=float, default=12.0,
                   help='Cap on the motion-slack tolerance added to any single gate.')
    p.add_argument('--calibrated-motion-slack', action=argparse.BooleanOptionalAction, default=True,
                   help='Also apply --motion-slack to the calibrated epipolar check '
                        '(--pair-calibrated-max-error-px). Off by default, that gate is '
                        'fixed and removes fast limbs whose cameras are a few ms apart.')
    p.add_argument('--temporal-min-certainty-fast', type=float, default=0.08,
                   help='Second-tier certainty floor for fast-moving observations '
                        '(image flow >= --temporal-fast-flow-px). Motion blur legitimately '
                        'lowers RoMa warp certainty on fast limbs; this keeps those '
                        'observations while static pixels still require '
                        '--temporal-min-certainty. 0 disables the tier.')
    p.add_argument('--temporal-fast-flow-px', type=float, default=8.0,
                   help='Image flow (px per video frame) above which '
                        '--temporal-min-certainty-fast applies.')
    p.add_argument('--untracked-velocity-from-tracked', action=argparse.BooleanOptionalAction, default=True,
                   help='Keep fresh points that failed temporal tracking, giving them velocity '
                        'inherited from the k nearest TRACKED points of the same keyframe '
                        '(inverse-distance weighted). Rows with no tracked neighbor within '
                        '--untracked-velocity-radius are dropped. Rescues blur-peak limbs where '
                        'the t->t+delta warp cannot follow the motion. Independent of the '
                        '--include-untracked-fresh ablation (which keeps ALL untracked rows).')
    p.add_argument('--untracked-velocity-radius', type=float, default=0.2,
                   help='Max distance (world units) to the nearest tracked point for velocity '
                        'inheritance eligibility.')
    p.add_argument('--untracked-velocity-knn', type=int, default=8,
                   help='Number of tracked neighbors used for inherited velocity.')
    p.add_argument('--rotation-json', type=str, default=None,
                   help='JSON mapping each camera to its extract rotation ("l"/"r"), e.g. '
                        '{"Camera 1": "l", ...}. Cameras record landscape and are rotated '
                        'to portrait, so the rolling-shutter readout runs along image '
                        'COLUMNS with a per-camera sign. Required for --rs-readout-ms.')
    p.add_argument('--rs-readout-ms', type=float, default=0.0,
                   help='Rolling-shutter readout duration in ms across the full column '
                        'span of the delivered images (Sony RX0 II @1080p60 measured '
                        '~7 ms on example capture). Adds a per-pixel capture-time term '
                        's_c*readout*(u-u_ref)/W to the sync motion compensation, '
                        'referenced to the flow-weighted mean column so it composes '
                        'with sync.json offsets. 0 disables.')
    p.add_argument('--rs-flip-sign', action='store_true',
                   help='Invert the l/r -> readout-direction convention if a rig\'s '
                        'extract rotation maps sensor lines the other way.')
    p.add_argument('--rs-axis', choices=('column', 'row'), default='column',
                   help='Readout axis in the delivered images: column (extracts rotated +-90, legacy default) '
                        'or row (unrotated landscape capture).')
    p.add_argument('--rs-reference', choices=('content', 'midpoint'), default='content',
                   help='Time origin of the rolling-shutter term: content (legacy: flow-weighted mean '
                        'coordinate of the moving content, composes with sync.json) or midpoint (sensor middle; '
                        'requires --timing-json midpoint constants). --timing-json forces midpoint.')
    p.add_argument('--timing-json', type=str, default=None,
                   help='Candidate capture-timing file (convention_id midpoint_v1, trigs.capture_timing): '
                        'per-camera MIDPOINT constants, per-camera rolling-shutter flags, readout and axis. '
                        'Replaces --sync-json and --rs-readout-ms (mutually exclusive with both).')

    # Velocity model tuning
    p.add_argument('--velocity-radius-factor', type=float, default=8.0)
    p.add_argument('--velocity-min-radius', type=float, default=0.01)
    p.add_argument('--velocity-max-radius', type=float, default=0.25)
    p.add_argument('--speed-adaptive-duration', action='store_true', default=True)
    p.add_argument(
        '--no-speed-adaptive-duration',
        action='store_false',
        dest='speed_adaptive_duration',
    )
    p.add_argument('--velocity-noise-floor', type=float, default=0.0,
               help='Velocities with magnitude below this (world units, '
                    'same scale as positions) are demoted to has_velocity=False '
                    'and zeroed. Set ~1-2x your triangulation depth noise; '
                    '0 disables.')
    p.add_argument('--duration-min-scale', type=float, default=0.2)
    p.add_argument('--duration-max-scale', type=float, default=1.0)

    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--disable-knn-velocity-fallback', action=argparse.BooleanOptionalAction,
               help='Skip spatial KNN fallback for points without RoMa-tracked '
                    'velocity. Untracked points stay has_velocity=False. '
                    'Eliminates fake velocities on static surfaces.', default=True)

    # Diagnostics / memory behavior
    p.add_argument(
        '--log-memory',
        action='store_true',
        default=False,
        help='Print process RSS at major pipeline stages. Default: disabled.',
    )
    p.add_argument(
        '--no-log-memory',
        action='store_false',
        dest='log_memory',
        default=False,
        help='Disable process RSS logging.',
    )
    p.add_argument(
        '--temporal-frame-delta',
        type=int,
        default=None,
        help='Track each keyframe t to t + temporal_frame_delta. Use 1 for adjacent-frame velocity.',
    )
    p.add_argument(
        '--temporal-backend',
        choices=('roma', 'memfof'),
        default='memfof',
        help=(
            'Dense matcher for the temporal tracks (spatial matching always uses '
            'RoMa). "memfof" runs MEMFOF multi-frame optical flow (BSD-3 code and '
            'weights) on (t-d, t, t+d) per camera: one inference gives both '
            'directions. It has no confidence output, so certainty gates and the '
            'dense-warp cycle check do not apply; triangulation, reprojection, mask '
            'and two-view-core gates do. Requires --centred-velocity.'
        ),
    )
    p.add_argument('--memfof-checkpoint', default=MEMFOF_DEFAULT_CHECKPOINT,
                   help='HuggingFace id of the MEMFOF weights.')
    p.add_argument(
        '--memfof-compose-hops',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'With --temporal-backend memfof: obtain the t -> t+-delta destinations by composing '
            'delta one-frame hops (hop c -> c+-1 uses the forward/backward flow of the MEMFOF '
            'triplet (c-1, c, c+1)), each sampled at the point\'s current pixel, instead of one '
            '(t-delta, t, t+delta) triplet. Capture-time compensation and every tracker gate are '
            'unchanged. Costs about 2*delta-1 MEMFOF inferences per camera and keyframe instead of '
            'one, and keeps the hop flows of one keyframe in memory (~2 GB for 27 cameras at 1080p).'
        ),
    )
    p.add_argument('--memfof-max-side', type=int, default=1920,
                   help='Images are resized so the long side is at most this for '
                        'MEMFOF (trained for 1080p); flows are rescaled to full resolution.')
    p.add_argument(
        '--rigid-cluster-motion',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Finalize-time piecewise-rigid repair of keyframe motion (rigid_motion.py): '
            'per small spatial cluster, gate tracked endpoints on the neighbour '
            "keyframe's reconstructed points, fit one rigid motion (RANSAC + ICP) and "
            'replace endpoints that are missing, off-surface or inconsistent with it. '
            'Catches tracks that stick to the background on fast, blurred parts (which '
            'pass every multi-view gate). Requires --centred-velocity and '
            '--temporal-frame-delta equal to --frame-step.'
        ),
    )
    p.add_argument('--rigid-cluster-size', type=float, default=0.0,
                   help='Cluster voxel size in world units; 0 = --rigid-cluster-size-spacings x '
                        'the keyframe median point spacing.')
    p.add_argument('--rigid-cluster-size-spacings', type=float, default=10.0)
    p.add_argument('--rigid-min-cluster-points', type=int, default=30)
    p.add_argument('--rigid-min-tracks', type=int, default=12)
    p.add_argument('--rigid-gate-spacings', type=float, default=3.0,
                   help='Endpoint must land within this many spacings of a neighbour-keyframe point.')
    p.add_argument('--rigid-inlier-spacings', type=float, default=2.0)
    p.add_argument('--rigid-min-inlier-frac', type=float, default=0.3)
    p.add_argument('--rigid-min-motion-spacings', type=float, default=1.0,
                   help='Fits moving the cluster less than this are skipped (never overwrite motion with a static consensus).')
    p.add_argument('--rigid-ransac-iters', type=int, default=256)
    p.add_argument('--rigid-icp-iters', type=int, default=20)
    p.add_argument('--rigid-fill-hops', type=int, default=3)
    p.add_argument(
        '--rigid-keep-inherited-on-vote',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When a row's rigid fit loses the local-support vote (its neighbours' tracks agree "
            "more with the row's current motion than with the fit), keep that current endpoint "
            'in the velocity rebuild instead of extrapolating from the other endpoint alone '
            '(both directions). Measured on example capture frames 81..106 with composed MEMFOF.'
        ),
    )
    p.add_argument(
        '--motion-budget',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Never thin moving areas (motion_budget.py): --target-points-per-frame '
            'then caps only the still rows of each keyframe, and moving rows are all '
            'kept, so keyframes with motion can exceed the cap. Applied at finalize '
            '(after --rigid-cluster-motion), so assembly keeps every row; the cap and '
            'the --motion-budget-* settings can be changed without invalidating the '
            'per-frame cache. Static rows are untouched.'
        ),
    )
    p.add_argument('--motion-budget-moving-per-frame', type=int, default=0,
                   help='Optional cap on moving rows per keyframe; 0 = keep them all.')
    p.add_argument('--motion-budget-min-speed', type=float, default=0.2,
                   help='A row moves when its neighbourhood-median speed is at least this '
                        'many point spacings per video frame (spacing at '
                        '--motion-budget-spacing-points density). Tracking noise measured '
                        '0.04-0.15 at p99 on example capture / another capture.')
    p.add_argument('--motion-budget-spacing-points', type=int, default=70000,
                   help='Keyframe density (rows) at which the speed threshold spacing is '
                        'measured; fixed so the still cap does not shift the threshold.')
    p.add_argument('--motion-budget-knn', type=int, default=16,
                   help='Neighbours (besides the row itself) in the speed median.')
    p.add_argument('--subkeyframe-min-speed', type=float, default=None,
                   help='Speed-aware sub-keyframes (0 = off). After the --frame-step keyframes, '
                        'every keyframe interval bounded by a keyframe with at least '
                        '--subkeyframe-min-rows rows at or above this speed (point spacings per '
                        'video frame, measured as for --motion-budget-min-speed) is also '
                        'reconstructed at each frame in between, and those frames contribute '
                        'only their rows at or above this speed. Fast rows (keyframe and in-between) '
                        'get local_stride 1 in the NPZ, every other row --frame-step; the trainer '
                        'keeps in-between rows on their own grid with sigma_t scaled by '
                        'local_stride / --frame-step and outside --config.target-count.')
    p.add_argument('--subkeyframe-min-rows', type=int, default=200,
                   help='Fast rows a bounding keyframe needs before its interval gets sub-keyframes.')
    p.add_argument('--subkeyframe-frame-min', type=int, default=None,
                   help='Only add sub-keyframes at frames >= this (default: no limit).')
    p.add_argument('--subkeyframe-frame-max', type=int, default=None,
                   help='Only add sub-keyframes at frames <= this (default: no limit).')
    p.add_argument(
        '--centred-velocity',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Also track each keyframe back to t - temporal_frame_delta. Points '
            'tracked both ways get the centred velocity (X(t+d) - X(t-d)) / 2d '
            'instead of the forward difference (which is the velocity half a '
            'step later), plus a measured acceleration saved as '
            '"accelerations" / "has_acceleration". Keyframes with no t-d frame '
            'keep forward velocities.'
        ),
    )
    p.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help=(
            'Directory for crash-safe per-keyframe checkpoints. Defaults to '
            '<output-path>.cache. The active cache is removed after all requested '
            'outputs are committed successfully unless --keep-cache is set.'
        ),
    )
    p.add_argument(
        '--no-resume',
        action='store_true',
        help='Recompute every keyframe while still using disk-backed checkpoints.',
    )
    p.add_argument(
        '--keep-cache',
        action='store_true',
        help=(
            'Retain the completed run cache after all requested outputs are '
            'committed, allowing later identical runs to reuse every keyframe.'
        ),
    )
    p.add_argument(
        '--force-run-fingerprint',
        type=str,
        default=None,
        help=(
            'Pin the run fingerprint to this exact value instead of computing '
            'it from the current script/backend/argument/reference-file '
            'hashes. Points --cache-dir at a specific prior cache namespace '
            '(copy the value from that run\'s manifest.json "run_fingerprint" '
            'field) so it gets reused even though something irrelevant to '
            'frame output changed since it was written (e.g. you edited this '
            'script or bumped an unrelated flag). Per-frame checkpoints are '
            'still checked against current image/mask content, so frames '
            'that genuinely changed are still recomputed.'
        ),
    )
    p.add_argument(
        '--trust-cache',
        action='store_true',
        help=(
            'Reuse existing checkpoints without checking their stored run/'
            'input fingerprint values, only schema and array-shape integrity. '
            'Needed alongside --force-run-fingerprint when the fingerprint '
            'computation itself changed (e.g. after a caching-logic edit to '
            'this script) so old checkpoints can never match a freshly '
            'computed fingerprint even though the underlying files did not '
            'change. Use with care: a checkpoint whose inputs genuinely '
            'changed will be reused anyway.'
        ),
    )

    p.add_argument(
        '--include-untracked-fresh',
        action='store_true',
        default=False,
        help='Also include untracked fresh triangulated points with zero/KNN velocity. Leave OFF for clean velocity initialization.',
    )

    p.add_argument(
        '--dedup-when-no-target',
        action='store_true',
        default=False,
        help='Apply dedup-radius even when target-points-per-frame <= 0. Default is no dedup unless target cap is active.',
    )
    p.add_argument(
        '--color-mode',
        type=str,
        default='medoid',
        choices=('mean', 'median', 'medoid', 'trimmed-mean'),
        help='How to initialize per-point RGB from multi-view observations.',
    )
    p.add_argument(
        '--color-trim-frac',
        type=float,
        default=0.25,
        help='Fraction of farthest observed colors to drop for --color-mode trimmed-mean.',
    )
    p.add_argument(
        "--static-motion-floor",
        type=float,
        default=0.0,
        help=(
            "World-units per video frame. Tracked points with speed below this "
            "are eligible to become static 3D rows. 0 disables static tagging."
        ),
    )
    p.add_argument(
        "--static-allow-protected",
        action="store_true",
        help=(
            "Allow tracked rows below --static-motion-floor to become static "
            "even when they are inside a dynamic-protection mask (for example "
            "--stage-capsule-protect). Use only for a known-static capture or "
            "calibration interval; otherwise protected slow actor geometry can "
            "be frozen into the persistent static pool."
        ),
    )
    p.add_argument(
        "--static-spatial-dedup-radius",
        type=float,
        default=0.00018,
        help=(
            "Voxel radius for merging static rows across all keyframes. "
            "Use this to collapse static walls into a persistent 3D pool."
        ),
    )
    p.add_argument(
        "--static-max-points",
        type=int,
        default=0,
        help="Optional cap after static voxel merge. 0 disables.",
    )
    p.add_argument(
        "--dynamic-center-protect-frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of observing views where a point must be near image center "
            "to be protected from static tagging/demotion."
        ),
    )
    p.add_argument("--center-protect-margin-x", type=float, default=0.22)
    p.add_argument("--center-protect-margin-y", type=float, default=0.12)
    p.add_argument(
        "--disable-image-center-protect",
        action=argparse.BooleanOptionalAction,
        help="Disable camera-image center protection. Use with stage capsule protection.",
        default=True,
    )
    
    # Broader dynamic-allow capsule.
    # Rows outside this capsule cannot remain normal dynamic rows.
    p.add_argument(
        "--dynamic-allow-capsule",
        action="store_true",
        help=(
            "Enable a broad world-space capsule where dynamic rows are allowed. "
            "Rows outside this capsule are forced static or dropped if moving."
        ),
    )
    p.add_argument(
        "--dynamic-allow-capsule-a",
        type=str,
        default="0,-1.50,0",
        help="Dynamic-allow capsule endpoint A as 'x,y,z'. Usually lower/wider than protect capsule.",
    )
    p.add_argument(
        "--dynamic-allow-capsule-b",
        type=str,
        default="0,1.35,0",
        help="Dynamic-allow capsule endpoint B as 'x,y,z'. Usually upper actor motion envelope.",
    )
    p.add_argument(
        "--dynamic-allow-capsule-radius",
        type=float,
        default=0.90,
        help="World-space radius for the broad dynamic-allow capsule.",
    )
    p.add_argument(
        "--dynamic-allow-auto-xz",
        action="store_true",
        help="Auto-center dynamic-allow capsule X/Z from moving tracked rows.",
    )
    p.add_argument(
        "--dynamic-allow-auto-seed-motion-floor",
        type=float,
        default=0.006,
        help="Speed floor used to auto-center dynamic-allow capsule.",
    )
    p.add_argument(
        "--dynamic-allow-auto-min-seeds",
        type=int,
        default=100,
        help="Minimum moving seeds required before dynamic-allow auto X/Z is used.",
    )
    p.add_argument(
        "--outside-dynamic-mode",
        type=str,
        default="drop_moving_force_static",
        choices=("force_static", "drop_moving_force_static", "drop_all"),
        help=(
            "Behavior outside --dynamic-allow-capsule. "
            "force_static: keep outside rows but force static. "
            "drop_moving_force_static: drop moving outside rows, force low-motion outside rows static. "
            "drop_all: drop all rows outside dynamic-allow capsule."
        ),
    )
    p.add_argument(
        "--outside-dynamic-motion-drop-thresh",
        type=float,
        default=0.003,
        help=(
            "World units per video frame. With outside-dynamic-mode=drop_moving_force_static, "
            "outside dynamic-allow rows above this speed are dropped as shadow/reflection artifacts."
        ),
    )

    # Hard stage/world keep capsule.
    p.add_argument(
        "--stage-keep-capsule",
        action="store_true",
        help=(
            "Enable a hard world-space crop capsule. Surviving rows outside are "
            "dropped before target-point downsampling; rows inside must still "
            "pass reconstruction and temporal tracking."
        ),
    )
    p.add_argument(
        "--stage-keep-capsule-a",
        type=str,
        default="0,-1.75,0",
        help="Stage keep capsule endpoint A as 'x,y,z'.",
    )
    p.add_argument(
        "--stage-keep-capsule-b",
        type=str,
        default="0,1.50,0",
        help="Stage keep capsule endpoint B as 'x,y,z'.",
    )
    p.add_argument(
        "--stage-keep-capsule-radius",
        type=float,
        default=1.50,
        help="World-space radius for hard keep capsule.",
    )
    
    p.add_argument(
        "--stage-capsule-protect",
        action="store_true",
        help=(
            "Protect reconstructed/tracked rows inside a 3D stage/world capsule "
            "from becoming static. This does not restore rows rejected during "
            "matching, triangulation, or temporal tracking, does not exempt rows "
            "from target-point downsampling, and is a no-op when static "
            "classification is disabled."
        ),
    )
    p.add_argument(
        "--stage-capsule-a",
        type=str,
        default="0,-1.35,0",
        help="World-space capsule endpoint A as 'x,y,z'. For Y-up, put this near/below feet.",
    )
    p.add_argument(
        "--stage-capsule-b",
        type=str,
        default="0,1.20,0",
        help="World-space capsule endpoint B as 'x,y,z'. For Y-up, put this near head/upper body.",
    )
    p.add_argument(
        "--stage-capsule-radius",
        type=float,
        default=0.55,
        help="World-space capsule radius around the actor centerline.",
    )
    p.add_argument(
        "--stage-capsule-auto-xz",
        action="store_true",
        help="Auto-center the stage capsule X/Z from moving tracked rows.",
    )
    p.add_argument(
        "--stage-capsule-auto-seed-motion-floor",
        type=float,
        default=0.006,
        help="Speed floor used to find moving actor seeds for auto X/Z capsule centering.",
    )
    p.add_argument(
        "--stage-capsule-auto-min-seeds",
        type=int,
        default=100,
        help="Minimum moving seeds required before auto X/Z capsule centering is used.",
    )
    p.add_argument("--actor-hull-protect", action="store_true")
    p.add_argument("--actor-seed-motion-floor", type=float, default=0.012)
    p.add_argument("--actor-hull-pad-x-frac", type=float, default=0.05)
    p.add_argument("--actor-hull-pad-top-frac", type=float, default=0.04)
    p.add_argument("--actor-hull-pad-bottom-frac", type=float, default=0.14)
    p.add_argument("--actor-hull-down-extend-frac", type=float, default=0.65)
    p.add_argument("--actor-hull-min-frac", type=float, default=0.35)
    p.add_argument("--actor-hull-min-seed-points", type=int, default=50)
    p.add_argument("--actor-hull-min-seed-points-per-cam", type=int, default=10)
    p.add_argument("--actor-hull-max-area-frac", type=float, default=0.35)
    p.add_argument(
        "--actor-hull-extend-to-bottom",
        action="store_true",
        help="Extend each accepted actor hull down to the image bottom to protect stationary feet/legs.",
    )
    p.add_argument(
        "--actor-hull-min-votes",
        type=int,
        default=2,
        help="Protect a point if it falls inside the actor hull in at least this many cameras.",
    )
    p.add_argument(
        "--actor-3d-dilate-protect",
        action="store_true",
        help="Protect low-motion rows near moving actor rows in 3D so stationary feet/hands do not become static.",
    )
    p.add_argument(
        "--actor-3d-dilate-radius",
        type=float,
        default=0.10,
        help="World-space radius for actor 3D protection dilation.",
    )
    p.add_argument(
        "--actor-3d-dilate-iters",
        type=int,
        default=2,
        help="Number of 3D dilation iterations from moving actor seeds.",
    )
    p.add_argument(
        "--actor-3d-seed-motion-floor",
        type=float,
        default=0.018,
        help="World units per video frame used to seed actor 3D dilation.",
    )
    p.add_argument(
        "--actor-column-protect",
        action="store_true",
        help="Protect rows that project into the central actor column, ignoring vertical image position.",
    )
    p.add_argument(
        "--actor-column-margin-x",
        type=float,
        default=0.36,
        help="Horizontal normalized image margin for actor-column protection. 0.36 means x in [0.36, 0.64].",
    )
    p.add_argument(
        "--actor-column-min-frac",
        type=float,
        default=0.35,
        help="Fraction of observing cameras that must vote inside the actor column.",
    )
    args = p.parse_args()
    resolve_auto_defaults(args)
    return args


SUBKEYFRAME_MIN_SPEED_DEFAULT = 4.0


def _frame_regex(pattern: str):
    """Regex matching filenames produced by a '{frame:0Nd}'/'{frame}' pattern, or None."""
    m = re.fullmatch(r'(.*)\{frame(?::0?(\d+)d)?\}(.*)', pattern)
    if m is None:
        return None
    digits = r'\d{%s}' % m.group(2) if m.group(2) else r'\d+'
    return re.compile('^' + re.escape(m.group(1)) + '(' + digits + ')' + re.escape(m.group(3)) + '$')


def _camera_dirs(images_root: str) -> List[Path]:
    root = Path(images_root)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.'))


def resolve_auto_defaults(args: argparse.Namespace) -> None:
    """Fill the defaults that depend on the data or on other flags (before the cache fingerprint)."""
    cams = _camera_dirs(args.images_root)
    if args.image_pattern is None:
        args.image_pattern = '{frame:06d}.png'
        for ext in ('.jpg', '.png'):
            rx = _frame_regex('{frame:06d}' + ext)
            if cams and any(rx.match(f.name) for f in cams[0].iterdir()):
                args.image_pattern = '{frame:06d}' + ext
                break
        print(f"[defaults] --image-pattern {args.image_pattern}")
    if args.input_frame_start is None or args.input_frame_end is None:
        rx = _frame_regex(args.image_pattern)
        spans = []
        if rx is not None:
            for d in cams:
                frames = [int(m.group(1)) for f in d.iterdir() if (m := rx.match(f.name))]
                if frames:
                    spans.append((min(frames), max(frames)))
        if spans:
            lo, hi = max(a for a, _ in spans), min(b for _, b in spans)
            if lo <= int(args.frame_start) and hi >= int(args.frame_end):
                if args.input_frame_start is None:
                    args.input_frame_start = lo
                if args.input_frame_end is None:
                    args.input_frame_end = hi
                print(f"[defaults] source images available for every camera: frames {lo}..{hi} "
                      f"(--input-frame-start {args.input_frame_start} --input-frame-end {args.input_frame_end})")
    if args.temporal_frame_delta is None:
        args.temporal_frame_delta = int(args.frame_step)
    if args.memfof_compose_hops is None:
        args.memfof_compose_hops = args.temporal_backend == 'memfof'
    if args.subkeyframe_min_speed is None:
        usable = (not args.disable_temporal_tracking
                  and bool(getattr(args, 'trigs_velocity_per_frame', True))
                  and not args.solve_offsets_out)
        args.subkeyframe_min_speed = SUBKEYFRAME_MIN_SPEED_DEFAULT if usable else 0.0
        if not usable:
            print("[defaults] speed-aware sub-keyframes off (they need temporal tracking, per-frame "
                  "velocities and no --solve-offsets-out)")


def scale_duration_by_speed(
    velocities: np.ndarray,
    has_velocity: np.ndarray,
    base_duration: float,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    n = len(velocities)
    scales = np.ones(n, dtype=np.float32) * max_scale
    if has_velocity.any():
        speeds = np.linalg.norm(velocities, axis=1)
        med = float(np.median(speeds[has_velocity]))
        if med > 0:
            s_norm = speeds / med
            scales_tracked = np.clip(1.0 / (1.0 + s_norm), min_scale, max_scale)
            scales[has_velocity] = scales_tracked[has_velocity]
    return (scales * base_duration).reshape(-1, 1).astype(np.float32)


def _rss_gb() -> float:
    """Return current process resident set size in GiB without extra deps."""
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # VmRSS is reported in KiB.
                    return float(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.
        if os.uname().sysname == "Darwin":
            return rss / (1024.0 ** 3)
        return rss / (1024.0 ** 2)
    except (ImportError, OSError, AttributeError):
        pass
    return -1.0


def _log_mem(enabled: bool, tag: str) -> None:
    if not enabled:
        return
    rss = _rss_gb()
    fields = []
    if rss >= 0:
        fields.append(f'rss={rss:.2f} GiB')
    if torch.cuda.is_available():
        fields.append(
            f'cuda_alloc={torch.cuda.memory_allocated() / (1024.0 ** 3):.2f} GiB'
        )
        fields.append(
            f'cuda_reserved={torch.cuda.memory_reserved() / (1024.0 ** 3):.2f} GiB'
        )
    if fields:
        print(f"[mem] {tag}: {', '.join(fields)}", flush=True)


def _empty_tracked_pair():
    return (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def combine_bidirectional_tracks(
    pos_t_fwd: np.ndarray,
    pos_tnext: np.ndarray,
    src_fwd: np.ndarray,
    pos_t_bwd: np.ndarray,
    pos_tprev: np.ndarray,
    src_bwd: np.ndarray,
    frame_delta: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre the forward-difference velocity with a t -> t-delta track.

    X(t+d) - X(t) is the mean velocity over [t, t+d], i.e. the velocity at
    t + d/2; at a stop or reversal that leads the motion by half a step. For
    points tracked both ways the forward endpoint is moved so the assembled
    velocity becomes the centred (X(t+d) - X(t-d)) / 2d, and the acceleration
    (X(t+d) - 2 X(t) + X(t-d)) / d^2 is returned alongside. Each direction is
    differenced against its own X(t) (the two triangulations use different
    camera subsets). Forward-only points keep their forward velocity and get
    has_acceleration=False.

    Returns (pos_tnext_out [M,3], accelerations [M,3] per frame^2,
    has_acceleration [M]) aligned with the forward arrays.
    """
    d = float(max(int(frame_delta), 1))
    pos_out = np.asarray(pos_tnext, dtype=np.float32).copy()
    accel = np.zeros((len(pos_out), 3), dtype=np.float32)
    has_acc = np.zeros((len(pos_out),), dtype=bool)
    if len(pos_out) == 0 or len(src_bwd) == 0:
        return pos_out, accel, has_acc
    bwd_row = {int(s): i for i, s in enumerate(np.asarray(src_bwd).tolist())}
    fwd_rows = [i for i, s in enumerate(np.asarray(src_fwd).tolist()) if int(s) in bwd_row]
    if not fwd_rows:
        return pos_out, accel, has_acc
    fi = np.asarray(fwd_rows, dtype=np.int64)
    bi = np.asarray([bwd_row[int(src_fwd[i])] for i in fwd_rows], dtype=np.int64)
    step_fwd = (np.asarray(pos_tnext, np.float64)[fi] - np.asarray(pos_t_fwd, np.float64)[fi]) / d
    step_bwd = (np.asarray(pos_t_bwd, np.float64)[bi] - np.asarray(pos_tprev, np.float64)[bi]) / d
    v_centred = 0.5 * (step_fwd + step_bwd)
    pos_out[fi] = (np.asarray(pos_t_fwd, np.float64)[fi] + v_centred * d).astype(np.float32)
    accel[fi] = ((step_fwd - step_bwd) / d).astype(np.float32)
    has_acc[fi] = True
    return pos_out, accel, has_acc


def rigid_params_from_args(args: argparse.Namespace) -> RigidMotionParams:
    return RigidMotionParams(
        cluster_size=float(args.rigid_cluster_size),
        cluster_size_spacings=float(args.rigid_cluster_size_spacings),
        min_cluster_points=int(args.rigid_min_cluster_points),
        min_tracks=int(args.rigid_min_tracks),
        gate_spacings=float(args.rigid_gate_spacings),
        inlier_spacings=float(args.rigid_inlier_spacings),
        min_inlier_frac=float(args.rigid_min_inlier_frac),
        min_motion_spacings=float(args.rigid_min_motion_spacings),
        ransac_iters=int(args.rigid_ransac_iters),
        icp_iters=int(args.rigid_icp_iters),
        fill_hops=int(args.rigid_fill_hops),
        seed=int(args.seed),
        keep_inherited_on_vote=bool(getattr(args, 'rigid_keep_inherited_on_vote', False)),
    )


def apply_rigid_cluster_motion(
    arrays: Mapping[str, np.ndarray],
    *,
    frame_delta: int,
    velocity_per_frame: bool,
    speed_adaptive_duration: bool,
    one_frame_duration: float,
    duration_min_scale: float,
    duration_max_scale: float,
    params: RigidMotionParams,
) -> Dict[str, object]:
    """Run rigid_motion.rigid_cluster_motion on the assembled rows, in place.

    Stored velocities are per video frame (or per temporal delta without
    --trigs-velocity-per-frame) and accelerations in the matching unit squared;
    the pass works per frame and writes back in the stored units. Durations of
    keyframes whose rows changed are recomputed with the same speed-adaptive
    rule the assembly used.
    """
    d = float(frame_delta)
    unit = 1.0 if velocity_per_frame else d
    vel_pf = np.asarray(arrays["velocities"], np.float64) / unit
    acc_pf = np.asarray(arrays["accelerations"], np.float64) / (unit * unit)
    vel, acc, has_acc, flags, changed, rstats = rigid_cluster_motion(
        positions=np.asarray(arrays["positions"]),
        colors=np.asarray(arrays["colors"]),
        velocities=vel_pf,
        accelerations=acc_pf,
        has_acceleration=np.asarray(arrays["has_acceleration"]),
        track_flags=np.asarray(arrays["track_flags"]),
        source_frames=np.asarray(arrays["source_frames"]).reshape(-1),
        is_static=np.asarray(arrays["is_static"]),
        frame_delta=int(frame_delta),
        params=params,
    )
    idx = np.nonzero(changed)[0]
    if len(idx):
        arrays["velocities"][idx] = (vel[idx] * unit).astype(np.float32)
        arrays["accelerations"][idx] = (acc[idx] * unit * unit).astype(np.float32)
        arrays["has_acceleration"][idx] = has_acc[idx]
        arrays["has_velocity"][idx] = True
        arrays["track_flags"][idx] = flags[idx]
        if speed_adaptive_duration:
            sf = np.asarray(arrays["source_frames"]).reshape(-1)
            for k in np.unique(sf[idx]):
                rows_k = np.nonzero(sf == k)[0]
                arrays["durations"][rows_k] = scale_duration_by_speed(
                    np.asarray(arrays["velocities"][rows_k]),
                    np.asarray(arrays["has_velocity"][rows_k]),
                    one_frame_duration,
                    duration_min_scale,
                    duration_max_scale,
                )
    out = rstats.as_dict()
    print(
        f"[rigid-cluster-motion] keyframes={out['keyframes']} clusters={out['clusters']} "
        f"fits fwd/bwd={out['fits']['fwd']}/{out['fits']['bwd']} "
        f"replaced fwd/bwd={out['replaced']['fwd']}/{out['replaced']['bwd']} "
        f"filled fwd/bwd={out['filled_untracked']['fwd']}/{out['filled_untracked']['bwd']} "
        f"rows changed={out['rows_changed']:,}"
    )
    return out


def motion_budget_params_from_args(args: argparse.Namespace) -> MotionBudgetParams:
    return MotionBudgetParams(
        moving_per_frame=int(args.motion_budget_moving_per_frame),
        still_per_frame=int(args.target_points_per_frame),
        min_speed_spacings=float(args.motion_budget_min_speed),
        knn=int(args.motion_budget_knn),
        seed=int(args.seed),
        spacing_points=int(args.motion_budget_spacing_points),
    )


def apply_motion_budget(
    arrays: Mapping[str, np.ndarray],
    *,
    frame_delta: int,
    velocity_per_frame: bool,
    dedup_radius: float,
    dedup_radius_max: float,
    params: MotionBudgetParams,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Run motion_budget.motion_budget_select on the assembled rows.

    Returns new in-memory arrays holding only the kept rows (row order
    preserved) and the stats. Stored velocities are per video frame, or per
    temporal delta without --trigs-velocity-per-frame.
    """
    unit = 1.0 if velocity_per_frame else float(frame_delta)
    speed = np.linalg.norm(np.asarray(arrays["velocities"], np.float64), axis=1) / unit
    flags = np.asarray(arrays["track_flags"])
    # Same priority the per-frame cap uses: tracked rows over fresh/inherited ones.
    priorities = ((flags & (TRACK_FWD | TRACK_BWD)) > 0).astype(np.float32)

    def thin(pos: np.ndarray, col: np.ndarray, pri: np.ndarray, target: int) -> np.ndarray:
        max_r = dedup_radius_max if dedup_radius_max > 0.10 else auto_max_voxel_size(pos, target)
        return _priority_voxel_to_target_with_indices(
            pos, col, pri, target_points=int(target),
            min_radius=float(dedup_radius), max_radius=float(max_r),
        )[3]

    keep_idx, mstats = motion_budget_select(
        positions=np.asarray(arrays["positions"]),
        colors=np.asarray(arrays["colors"]),
        priorities=priorities,
        speed_per_frame=speed,
        has_velocity=np.asarray(arrays["has_velocity"]),
        source_frames=np.asarray(arrays["source_frames"]).reshape(-1),
        is_static=np.asarray(arrays["is_static"]),
        params=params,
        thin_fn=thin,
    )
    out = {name: np.asarray(arr[keep_idx]) for name, arr in arrays.items()}
    kp = mstats["per_keyframe"]
    lo, mid, hi = mstats["kept_per_keyframe_min_p50_max"]
    print(
        f"[motion-budget] rows {mstats['rows_in']:,} -> {mstats['rows_out']:,} "
        f"(static {mstats['static_rows']:,} untouched); kept per keyframe "
        f"min/p50/max {lo:,}/{mid:,}/{hi:,}"
    )
    for p_ in kp[:: max(1, len(kp) // 12)]:
        print(
            f"  [motion-budget] keyframe {p_['keyframe']}: in {p_['rows_in']:,} "
            f"(moving {p_['moving_in']:,}) -> moving {p_['moving_kept']:,} + "
            f"still {p_['still_kept']:,}"
        )
    return out, mstats


def accelerations_for_rows(
    row_positions: np.ndarray,
    tracked_positions: np.ndarray,
    tracked_accel: np.ndarray,
    tracked_has_accel: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Look up each assembled row's acceleration by its exact tracked X(t).

    Assembly filters, dedups and caps rows without exposing the kept
    indices; tracked rows keep their X(t) bit-for-bit, so an exact position
    match recovers them. Rows with no match (fresh rows) get zero and False.
    """
    n = len(row_positions)
    accel = np.zeros((n, 3), dtype=np.float32)
    has = np.zeros((n,), dtype=bool)
    if n == 0 or len(tracked_positions) == 0:
        return accel, has
    dist, idx = cKDTree(np.asarray(tracked_positions, np.float64)).query(
        np.asarray(row_positions, np.float64), k=1
    )
    hit = dist <= tol
    accel[hit] = np.asarray(tracked_accel)[idx[hit]]
    has[hit] = np.asarray(tracked_has_accel, dtype=bool)[idx[hit]]
    return accel, has


def _maybe_auto_center_capsule_xz(
    *,
    name: str,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    positions: np.ndarray,
    speed_per_frame: np.ndarray,
    enabled: bool,
    seed_motion_floor: float,
    min_seeds: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optionally replace endpoint A/B x,z by median x,z of moving rows."""
    a = np.asarray(endpoint_a, dtype=np.float32).copy()
    b = np.asarray(endpoint_b, dtype=np.float32).copy()

    if not enabled or len(positions) == 0:
        return a, b

    seed_mask = np.asarray(speed_per_frame).reshape(-1) >= float(seed_motion_floor)
    n_seed = int(seed_mask.sum())

    if n_seed < int(min_seeds):
        print(
            f"  [{name}] auto_xz skipped: seeds={n_seed:,} "
            f"< min={int(min_seeds):,}; using explicit endpoints"
        )
        return a, b

    seed_pos = np.asarray(positions[seed_mask], dtype=np.float32)
    center_x = float(np.median(seed_pos[:, 0]))
    center_z = float(np.median(seed_pos[:, 2]))

    a[0] = center_x
    b[0] = center_x
    a[2] = center_z
    b[2] = center_z

    print(
        f"  [{name}] auto_xz seeds={n_seed:,} "
        f"center_x={center_x:.4f} center_z={center_z:.4f} "
        f"radius={float(radius):.3f}"
    )

    return a, b

def _center_protect_mask(
    cloud: FramePointCloud,
    src_idx: np.ndarray,
    cameras: Sequence[Camera],
    margin_x: float,
    margin_y: float,
    min_frac: float,
) -> np.ndarray:
    if len(src_idx) == 0:
        return np.zeros((0,), dtype=bool)

    out = np.zeros((len(src_idx),), dtype=bool)
    for j, pi in enumerate(src_idx):
        obs = cloud.observations[int(pi)]
        if not obs:
            continue

        votes = 0
        total = 0
        for cam_idx, uv in obs:
            cam = cameras[int(cam_idx)]
            u, v = float(uv[0]), float(uv[1])
            x = u / max(float(cam.width - 1), 1.0)
            y = v / max(float(cam.height - 1), 1.0)
            total += 1
            if (
                margin_x <= x <= 1.0 - margin_x
                and margin_y <= y <= 1.0 - margin_y
            ):
                votes += 1

        if total > 0 and (votes / total) >= min_frac:
            out[j] = True

    return out


def _actor_column_protect_mask(
    cloud: FramePointCloud,
    src_idx: np.ndarray,
    cameras: Sequence[Camera],
    margin_x: float,
    min_frac: float,
) -> np.ndarray:
    """Protect observations in a central horizontal image band."""
    if len(src_idx) == 0:
        return np.zeros((0,), dtype=bool)
    out = np.zeros((len(src_idx),), dtype=bool)
    for j, pi in enumerate(src_idx):
        obs = cloud.observations[int(pi)]
        if not obs:
            continue
        votes = 0
        for cam_idx, uv in obs:
            cam = cameras[int(cam_idx)]
            x = float(uv[0]) / max(float(cam.width - 1), 1.0)
            votes += int(float(margin_x) <= x <= 1.0 - float(margin_x))
        out[j] = (votes / len(obs)) >= float(min_frac)
    return out


def _parse_vec3_arg(value: str, name: str) -> np.ndarray:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must be 'x,y,z', got: {value!r}")

    try:
        return np.asarray(
            [float(parts[0]), float(parts[1]), float(parts[2])],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain three floats, got: {value!r}"
        ) from exc


def _parse_roi_arg(
    value: str, name: str
) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in str(value).split(',')]
    if len(parts) != 4:
        raise ValueError(f"{name} must be 'x0,y0,x1,y1', got: {value!r}")
    try:
        roi = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f'{name} must contain four numbers, got: {value!r}'
        ) from exc
    if not np.isfinite(np.asarray(roi, dtype=np.float64)).all():
        raise ValueError(f'{name} values must be finite, got: {value!r}')
    if roi[2] < roi[0] or roi[3] < roi[1]:
        raise ValueError(f'{name} requires x1 >= x0 and y1 >= y0')
    return roi


def _stage_capsule_protect_mask(
    positions: np.ndarray,
    *,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Protect rows inside a 3D capsule around the center-stage actor.

    This uses reconstructed world coordinates, not camera-image projection.
    A point is protected when its closest distance to segment A-B is <= radius.
    """
    positions = np.asarray(positions, dtype=np.float32)
    n = int(len(positions))

    if n == 0 or radius <= 0.0:
        return np.zeros((n,), dtype=bool)

    a = np.asarray(endpoint_a, dtype=np.float32).reshape(1, 3)
    b = np.asarray(endpoint_b, dtype=np.float32).reshape(1, 3)

    ab = b - a
    ab_len2 = float(np.sum(ab * ab))

    if ab_len2 <= 1e-12:
        d2 = np.sum((positions - a) ** 2, axis=1)
        return d2 <= float(radius) ** 2

    ap = positions - a
    t = np.sum(ap * ab, axis=1, keepdims=True) / ab_len2
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab
    d2 = np.sum((positions - closest) ** 2, axis=1)

    return d2 <= float(radius) ** 2

def _actor_hull_protect_mask(
    *,
    cloud: FramePointCloud,
    tracked_src_idx: np.ndarray,
    tracked_speed_per_frame: np.ndarray,
    center_protect_tracked: np.ndarray,
    cameras: Sequence[Camera],
    actor_seed_motion_floor: float,
    actor_hull_pad_x_frac: float,
    actor_hull_pad_top_frac: float,
    actor_hull_pad_bottom_frac: float,
    actor_hull_down_extend_frac: float,
    actor_hull_min_frac: float,
    actor_hull_min_seed_points: int,
    actor_hull_min_seed_points_per_cam: int,
    actor_hull_max_area_frac: float,
    actor_hull_extend_to_bottom: bool,
    actor_hull_min_votes: int,
) -> np.ndarray:
    n = len(tracked_src_idx)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    # High-confidence actor seeds: moving or already center-protected.
    seed = tracked_speed_per_frame >= actor_seed_motion_floor

    if int(seed.sum()) < actor_hull_min_seed_points:
        return np.zeros((n,), dtype=bool)

    # Per-camera seed bboxes.
    cam_boxes = {}
    for row_i, src_i in enumerate(tracked_src_idx):
        if not seed[row_i]:
            continue

        for cam_idx, uv in cloud.observations[int(src_i)]:
            cam_idx = int(cam_idx)
            cam = cameras[cam_idx]
            u, v = float(uv[0]), float(uv[1])

            if cam_idx not in cam_boxes:
                cam_boxes[cam_idx] = [u, v, u, v, 1]
            else:
                b = cam_boxes[cam_idx]
                b[0] = min(b[0], u)
                b[1] = min(b[1], v)
                b[2] = max(b[2], u)
                b[3] = max(b[3], v)
                b[4] += 1

    # Pad bboxes. Use asymmetric padding: feet are often below the moving
    # torso/arm seed bbox, while walls/stands should not explode the box.
    padded_boxes = {}

    for cam_idx, b in cam_boxes.items():
        if b[4] < actor_hull_min_seed_points_per_cam:
            continue

        cam = cameras[cam_idx]

        box_h = max(float(b[3] - b[1]), 1.0)

        # IMPORTANT:
        # Use the function parameters here, not args.*.
        # This helper does not receive args and must stay self-contained.
        pad_x = float(actor_hull_pad_x_frac) * float(cam.width)
        pad_top = float(actor_hull_pad_top_frac) * float(cam.height)

        pad_bottom = max(
            float(actor_hull_pad_bottom_frac) * float(cam.height),
            float(actor_hull_down_extend_frac) * box_h,
        )

        x0 = max(0.0, float(b[0]) - pad_x)
        y0 = max(0.0, float(b[1]) - pad_top)
        x1 = min(float(cam.width - 1), float(b[2]) + pad_x)
        if actor_hull_extend_to_bottom:
            y1 = float(cam.height - 1)
        else:
            y1 = min(float(cam.height - 1), float(b[3]) + pad_bottom)

        area_frac = ((x1 - x0) * (y1 - y0)) / max(
            float(cam.width * cam.height),
            1.0,
        )

        if area_frac > actor_hull_max_area_frac:
            continue

        padded_boxes[cam_idx] = (x0, y0, x1, y1)

    if not padded_boxes:
        return np.zeros((n,), dtype=bool)

    out = np.zeros((n,), dtype=bool)

    for row_i, src_i in enumerate(tracked_src_idx):
        votes = 0
        total = 0

        for cam_idx, uv in cloud.observations[int(src_i)]:
            cam_idx = int(cam_idx)
            if cam_idx not in padded_boxes:
                continue

            total += 1
            u, v = float(uv[0]), float(uv[1])
            x0, y0, x1, y1 = padded_boxes[cam_idx]

            if x0 <= u <= x1 and y0 <= v <= y1:
                votes += 1

        if votes >= actor_hull_min_votes:
            out[row_i] = True
        elif total > 0 and (votes / total) >= actor_hull_min_frac:
            out[row_i] = True

    return out

def _actor_3d_dilate_protect_mask(
    *,
    positions: np.ndarray,
    speed_per_frame: np.ndarray,
    seed_motion_floor: float,
    radius: float,
    iters: int,
) -> np.ndarray:
    """Protect rows connected in 3D to moving actor rows.

    This catches temporarily-stationary body parts such as feet/hands/head
    without expanding a 2D image hull across the background.
    """
    positions = np.asarray(positions, dtype=np.float32)
    speed_per_frame = np.asarray(speed_per_frame, dtype=np.float32).reshape(-1)

    n = int(len(positions))
    if n == 0:
        return np.zeros((0,), dtype=bool)

    if len(speed_per_frame) != n:
        raise ValueError(
            "_actor_3d_dilate_protect_mask row mismatch: "
            f"positions={n}, speed_per_frame={len(speed_per_frame)}"
        )

    if radius <= 0.0 or iters <= 0:
        return np.zeros((n,), dtype=bool)

    seed = speed_per_frame >= float(seed_motion_floor)
    if int(seed.sum()) == 0:
        return np.zeros((n,), dtype=bool)

    tree = cKDTree(positions)

    protected = seed.copy()
    frontier = np.flatnonzero(seed).astype(np.int64)

    for _ in range(int(iters)):
        if len(frontier) == 0:
            break

        # scipy may return a list-of-lists or an object ndarray of lists.
        neighbor_lists = tree.query_ball_point(
            positions[frontier],
            r=float(radius),
        )

        if isinstance(neighbor_lists, np.ndarray):
            neighbor_lists = neighbor_lists.tolist()

        flat_neighbors = []
        for group in neighbor_lists:
            if group:
                flat_neighbors.extend(group)

        if len(flat_neighbors) == 0:
            break

        nb_idx = np.asarray(flat_neighbors, dtype=np.int64)

        new_mask = np.zeros((n,), dtype=bool)
        new_mask[nb_idx] = True
        new_mask &= ~protected

        if not new_mask.any():
            break

        protected |= new_mask
        frontier = np.flatnonzero(new_mask).astype(np.int64)

    return protected

def _priority_voxel_once_with_indices(
    positions: np.ndarray,
    colors: np.ndarray,
    priorities: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(positions) == 0 or radius <= 0.0:
        idx = np.arange(len(positions), dtype=np.int64)
        return positions, colors, priorities, idx
    coords = np.floor(positions / float(radius)).astype(np.int64)
    coords -= coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    keys = coords[:, 0] * (maxs[1] * maxs[2]) + coords[:, 1] * maxs[2] + coords[:, 2]
    order = np.lexsort((-priorities, keys))
    _, first = np.unique(keys[order], return_index=True)
    kept = order[first].astype(np.int64, copy=False)
    return positions[kept], colors[kept], priorities[kept], kept


def _append_assembled_frame(
    *,
    args: argparse.Namespace,
    frames: Sequence[int],
    cameras: Sequence[Camera],   # ADD THIS
    cloud: FramePointCloud,
    idx: int,
    next_cloud: Optional[FramePointCloud],
    tracked_pair: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    denom: int,
    one_frame_duration: float,
    all_positions: List[np.ndarray],
    all_colors: List[np.ndarray],
    all_velocities: List[np.ndarray],
    all_times: List[np.ndarray],
    all_durations: List[np.ndarray],
    all_has_velocity: List[np.ndarray],
    all_source_frames: List[np.ndarray],
    stats: Dict[str, object],
    all_is_static: List[np.ndarray],
    all_dynamic_protect: List[np.ndarray],
    all_dynamic_allowed: List[np.ndarray],
    all_stage_keep: List[np.ndarray],
    sync_offsets_frames: Optional[np.ndarray] = None,
) -> str:
    """
    Assemble one keyframe and append its final arrays to the output lists.

    Default behavior is tracked-only:
      positions  = X(t)
      velocities = X(t+delta) - X(t), optionally divided by delta
      colors     = colors at keyframe t
      has_velocity = True for all saved rows

    Set --include-untracked-fresh only for ablations.
    """
    tracked_pos_t, tracked_pos_tn, tracked_cols, tracked_src_idx = tracked_pair

    include_untracked_fresh = bool(getattr(args, 'include_untracked_fresh', False))
    inherit_vel = bool(getattr(args, 'untracked_velocity_from_tracked', False))

    if len(tracked_pos_t) > 0:
        tracked_vel = (tracked_pos_tn - tracked_pos_t).astype(np.float32, copy=False)
    else:
        tracked_vel = np.zeros((0, 3), dtype=np.float32)

    if include_untracked_fresh or inherit_vel:
        # Tracked rows replace their original fresh rows. Keeping both creates
        # duplicate positions with conflicting velocity/has_velocity values.
        fresh_keep = np.ones((len(cloud.positions),), dtype=bool)
        valid_src = np.asarray(tracked_src_idx, dtype=np.int64)
        valid_src = valid_src[(valid_src >= 0) & (valid_src < len(fresh_keep))]
        fresh_keep[np.unique(valid_src)] = False
        fresh_src_idx = np.flatnonzero(fresh_keep).astype(np.int64)
        fresh_pos = cloud.positions[fresh_src_idx].astype(np.float32, copy=False)
        fresh_col = cloud.colors[fresh_src_idx].astype(np.float32, copy=False)

        fresh_vel = np.zeros((len(fresh_pos), 3), dtype=np.float32)
        fresh_has_vel = np.zeros((len(fresh_pos),), dtype=bool)
        n_inherited = 0
        if inherit_vel and len(fresh_pos) > 0 and len(tracked_pos_t) > 0:
            # Untracked fresh points (typically blur-peak limbs the t->t+delta
            # warp could not follow) inherit velocity from the nearest tracked
            # points at the same keyframe, inverse-distance weighted.
            k = min(int(getattr(args, 'untracked_velocity_knn', 8)), len(tracked_pos_t))
            tree = cKDTree(tracked_pos_t)
            dists, nn = tree.query(fresh_pos, k=k)
            dists = np.atleast_2d(np.asarray(dists, dtype=np.float64).reshape(len(fresh_pos), -1))
            nn = np.atleast_2d(np.asarray(nn, dtype=np.int64).reshape(len(fresh_pos), -1))
            w = 1.0 / np.maximum(dists, 1e-6)
            w /= np.sum(w, axis=1, keepdims=True)
            fresh_vel = np.sum(
                tracked_vel[nn].astype(np.float64) * w[..., None], axis=1
            ).astype(np.float32)
            near = dists[:, 0] <= float(getattr(args, 'untracked_velocity_radius', 0.15))
            fresh_vel[~near] = 0.0
            fresh_has_vel = near
            n_inherited = int(near.sum())
            if not include_untracked_fresh:
                # Inherit-only mode: keep just the rows that found a tracked
                # neighbor; isolated fresh rows are likelier junk.
                fresh_src_idx = fresh_src_idx[near]
                fresh_pos = fresh_pos[near]
                fresh_col = fresh_col[near]
                fresh_vel = fresh_vel[near]
                fresh_has_vel = fresh_has_vel[near]
        elif not include_untracked_fresh:
            # inherit_vel requested but nothing tracked to inherit from.
            fresh_src_idx = np.zeros((0,), dtype=np.int64)
            fresh_pos = np.zeros((0, 3), dtype=np.float32)
            fresh_col = np.zeros((0, 3), dtype=np.float32)
            fresh_vel = np.zeros((0, 3), dtype=np.float32)
            fresh_has_vel = np.zeros((0,), dtype=bool)

        if inherit_vel:
            print(
                f'  [untracked-vel] frame {cloud.frame_idx}: '
                f'{n_inherited}/{len(fresh_keep.nonzero()[0])} untracked fresh rows '
                f'inherited tracked velocity (kept {len(fresh_pos)})'
            )

        combined_pos = np.concatenate([tracked_pos_t, fresh_pos], axis=0).astype(np.float32, copy=False)
        combined_col = np.concatenate([tracked_cols, fresh_col], axis=0).astype(np.float32, copy=False)
        combined_pri = np.concatenate([
            np.ones(len(tracked_pos_t), dtype=np.float32),
            np.zeros(len(fresh_pos), dtype=np.float32),
        ])
        vel_arr = np.concatenate([tracked_vel, fresh_vel], axis=0)
        has_vel_arr = np.concatenate([
            np.ones(len(tracked_pos_t), dtype=bool),
            fresh_has_vel,
        ])

    else:
        fresh_src_idx = np.zeros((0,), dtype=np.int64)
        combined_pos = tracked_pos_t.astype(np.float32, copy=False)
        combined_col = tracked_cols.astype(np.float32, copy=False)
        combined_pri = np.ones(len(combined_pos), dtype=np.float32)
        vel_arr = tracked_vel.astype(np.float32, copy=False)
        has_vel_arr = np.ones(len(combined_pos), dtype=bool)

    point_offsets_frames = np.zeros(len(combined_pos), dtype=np.float32)
    if sync_offsets_frames is not None:
        if cloud.camera_frame_indices is not None:
            capture_residual = (
                np.asarray(cloud.camera_frame_indices, dtype=np.float64)
                - float(cloud.frame_idx)
                + np.asarray(sync_offsets_frames, dtype=np.float64)
            )
        else:
            capture_residual = np.asarray(sync_offsets_frames, dtype=np.float64)

        # Motion-compensated tracked observations were shifted to the nominal
        # global keyframe time before triangulation, so their timestamp is exact.
        if not bool(getattr(args, 'sync_motion_compensate', True)):
            for i in range(len(tracked_pos_t)):
                obs = cloud.observations[int(tracked_src_idx[i])]
                cams = [int(o[0]) for o in obs]
                if cams:
                    point_offsets_frames[i] = float(np.mean(capture_residual[cams]))

        # Fresh rows triangulated with fresh-path sync compensation are already
        # at the canonical keyframe time; only uncompensated fresh rows carry a
        # mean capture-residual timestamp.
        if len(fresh_src_idx) > 0 and not bool(cloud.debug.get('sync_comp_enabled', 0)):
            offset_idx = len(tracked_pos_t)
            for i, src_i in enumerate(fresh_src_idx):
                obs = cloud.observations[int(src_i)]
                cams = [int(o[0]) for o in obs]
                if cams:
                    point_offsets_frames[offset_idx + i] = float(np.mean(capture_residual[cams]))
    # Row-aligned dynamic protection mask.
    # First rows are tracked_pos_t; optional trailing rows are fresh_pos.

    # 1) Optional legacy image-center protection.
    # This is camera-image based, not stage/world based. Disable it when using
    # --stage-roi-protect to avoid protecting wall rays that pass through image center.
    if (
        not args.disable_image_center_protect
        and len(tracked_src_idx) == len(tracked_pos_t)
    ):
        center_protect_tracked = _center_protect_mask(
            cloud=cloud,
            src_idx=tracked_src_idx,
            cameras=cameras,
            margin_x=float(args.center_protect_margin_x),
            margin_y=float(args.center_protect_margin_y),
            min_frac=float(args.dynamic_center_protect_frac),
        )
    else:
        center_protect_tracked = np.zeros((len(tracked_pos_t),), dtype=bool)

    if bool(getattr(args, 'actor_column_protect', False)) and len(tracked_src_idx) == len(tracked_pos_t):
        actor_column_protect_tracked = _actor_column_protect_mask(
            cloud=cloud,
            src_idx=tracked_src_idx,
            cameras=cameras,
            margin_x=float(args.actor_column_margin_x),
            min_frac=float(args.actor_column_min_frac),
        )
    else:
        actor_column_protect_tracked = np.zeros((len(tracked_pos_t),), dtype=bool)

    # 2) Frame delta and tracked speed, computed once and reused by optional
    # motion-based actor protections.
    if next_cloud is not None:
        frame_delta = int(next_cloud.frame_idx - cloud.frame_idx)
    else:
        frame_delta = int(getattr(args, "temporal_frame_delta", 1))
    frame_delta = max(frame_delta, 1)

    tracked_speed_per_frame = (
        np.linalg.norm(tracked_vel, axis=1).astype(np.float32)
        / float(frame_delta)
        if len(tracked_vel) > 0
        else np.zeros((0,), dtype=np.float32)
    )

    # 3) Optional 3D stage/world ROI protection.
    # This is the correct center-stage protection. It uses actual world-space
    # point positions, not camera projection.
    if getattr(args, "stage_capsule_protect", False):
        stage_capsule_a = _parse_vec3_arg(
            args.stage_capsule_a,
            "--stage-capsule-a",
        )
        stage_capsule_b = _parse_vec3_arg(
            args.stage_capsule_b,
            "--stage-capsule-b",
        )

        stage_capsule_a, stage_capsule_b = _maybe_auto_center_capsule_xz(
            name="stage-capsule-protect",
            endpoint_a=stage_capsule_a,
            endpoint_b=stage_capsule_b,
            positions=tracked_pos_t,
            speed_per_frame=tracked_speed_per_frame,
            enabled=bool(getattr(args, "stage_capsule_auto_xz", False)),
            seed_motion_floor=float(args.stage_capsule_auto_seed_motion_floor),
            min_seeds=int(args.stage_capsule_auto_min_seeds),
            radius=float(args.stage_capsule_radius),
        )

        stage_capsule_protect_tracked = _stage_capsule_protect_mask(
            tracked_pos_t,
            endpoint_a=stage_capsule_a,
            endpoint_b=stage_capsule_b,
            radius=float(args.stage_capsule_radius),
        )

        print(
            f"  [stage-capsule] protected={int(stage_capsule_protect_tracked.sum()):,} "
            f"/ {len(stage_capsule_protect_tracked):,} "
            f"a=({stage_capsule_a[0]:.3f},{stage_capsule_a[1]:.3f},{stage_capsule_a[2]:.3f}) "
            f"b=({stage_capsule_b[0]:.3f},{stage_capsule_b[1]:.3f},{stage_capsule_b[2]:.3f})"
        )
    else:
        stage_capsule_protect_tracked = np.zeros((len(tracked_pos_t),), dtype=bool)

    # 4) Optional actor hull protection. Leave off for the stage-ROI diagnostic.
    if args.actor_hull_protect and len(tracked_src_idx) == len(tracked_pos_t):
        actor_protect_tracked = _actor_hull_protect_mask(
            cloud=cloud,
            tracked_src_idx=tracked_src_idx,
            tracked_speed_per_frame=tracked_speed_per_frame,
            center_protect_tracked=center_protect_tracked,
            cameras=cameras,
            actor_seed_motion_floor=float(args.actor_seed_motion_floor),
            actor_hull_pad_x_frac=float(args.actor_hull_pad_x_frac),
            actor_hull_pad_top_frac=float(args.actor_hull_pad_top_frac),
            actor_hull_pad_bottom_frac=float(args.actor_hull_pad_bottom_frac),
            actor_hull_down_extend_frac=float(args.actor_hull_down_extend_frac),
            actor_hull_min_frac=float(args.actor_hull_min_frac),
            actor_hull_min_seed_points=int(args.actor_hull_min_seed_points),
            actor_hull_min_seed_points_per_cam=int(args.actor_hull_min_seed_points_per_cam),
            actor_hull_max_area_frac=float(args.actor_hull_max_area_frac),
            actor_hull_extend_to_bottom=bool(args.actor_hull_extend_to_bottom),
            actor_hull_min_votes=int(args.actor_hull_min_votes),
        )
    else:
        actor_protect_tracked = np.zeros((len(tracked_pos_t),), dtype=bool)

    # 5) Optional local 3D dilation. Leave off for the stage-ROI diagnostic.
    if args.actor_3d_dilate_protect and len(tracked_pos_t) == len(tracked_speed_per_frame):
        actor_3d_protect_tracked = _actor_3d_dilate_protect_mask(
            positions=tracked_pos_t,
            speed_per_frame=tracked_speed_per_frame,
            seed_motion_floor=float(args.actor_3d_seed_motion_floor),
            radius=float(args.actor_3d_dilate_radius),
            iters=int(args.actor_3d_dilate_iters),
        )
    else:
        actor_3d_protect_tracked = np.zeros((len(tracked_pos_t),), dtype=bool)

    protected_tracked = (
        center_protect_tracked
        | actor_column_protect_tracked
        | stage_capsule_protect_tracked
        | actor_protect_tracked
        | actor_3d_protect_tracked
    )

    if include_untracked_fresh or inherit_vel:
        # World-based stage-capsule protection applies to fresh rows too;
        # image/hull protections are tracked-only and stay False for fresh.
        if getattr(args, "stage_capsule_protect", False) and len(fresh_pos) > 0:
            protected_fresh = _stage_capsule_protect_mask(
                fresh_pos,
                endpoint_a=stage_capsule_a,
                endpoint_b=stage_capsule_b,
                radius=float(args.stage_capsule_radius),
            )
        else:
            protected_fresh = np.zeros((len(fresh_pos),), dtype=bool)
        dynamic_protect_arr = np.concatenate(
            [protected_tracked, protected_fresh],
            axis=0,
        )
    else:
        dynamic_protect_arr = protected_tracked

    # ------------------------------------------------------------------
    # New dynamic ROI filtering:
    #
    #   protect capsule      = narrow actor protection; protected dynamic rows.
    #   dynamic allow capsule = broader actor motion envelope.
    #   stage keep capsule    = hard world/stage bound; outside is dropped.
    #
    # Important: this happens before target-per-frame dedup/capping so shadow
    # / wall / stage-edge artifacts do not consume the dynamic budget.
    # ------------------------------------------------------------------

    # Speed in world units per VIDEO frame.
    speed_per_video_frame_all = (
        np.linalg.norm(vel_arr, axis=1).astype(np.float32)
        / float(frame_delta)
        if len(vel_arr) > 0
        else np.zeros((0,), dtype=np.float32)
    )

    # Broad dynamic-allowed region.
    if getattr(args, "dynamic_allow_capsule", False):
        dynamic_allow_a = _parse_vec3_arg(
            args.dynamic_allow_capsule_a,
            "--dynamic-allow-capsule-a",
        )
        dynamic_allow_b = _parse_vec3_arg(
            args.dynamic_allow_capsule_b,
            "--dynamic-allow-capsule-b",
        )

        dynamic_allow_a, dynamic_allow_b = _maybe_auto_center_capsule_xz(
            name="dynamic-allow-capsule",
            endpoint_a=dynamic_allow_a,
            endpoint_b=dynamic_allow_b,
            positions=combined_pos,
            speed_per_frame=speed_per_video_frame_all,
            enabled=bool(getattr(args, "dynamic_allow_auto_xz", False)),
            seed_motion_floor=float(args.dynamic_allow_auto_seed_motion_floor),
            min_seeds=int(args.dynamic_allow_auto_min_seeds),
            radius=float(args.dynamic_allow_capsule_radius),
        )

        dynamic_allowed_arr = _stage_capsule_protect_mask(
            combined_pos,
            endpoint_a=dynamic_allow_a,
            endpoint_b=dynamic_allow_b,
            radius=float(args.dynamic_allow_capsule_radius),
        )

        # Existing protected actor rows are always dynamic-allowed.
        dynamic_allowed_arr |= dynamic_protect_arr

        print(
            f"  [dynamic-allow] inside={int(dynamic_allowed_arr.sum()):,} "
            f"/ {len(dynamic_allowed_arr):,} "
            f"a=({dynamic_allow_a[0]:.3f},{dynamic_allow_a[1]:.3f},{dynamic_allow_a[2]:.3f}) "
            f"b=({dynamic_allow_b[0]:.3f},{dynamic_allow_b[1]:.3f},{dynamic_allow_b[2]:.3f}) "
            f"radius={float(args.dynamic_allow_capsule_radius):.3f}"
        )
    else:
        dynamic_allowed_arr = np.ones((len(combined_pos),), dtype=bool)

    # Hard stage/world keep region.
    if getattr(args, "stage_keep_capsule", False):
        stage_keep_a = _parse_vec3_arg(
            args.stage_keep_capsule_a,
            "--stage-keep-capsule-a",
        )
        stage_keep_b = _parse_vec3_arg(
            args.stage_keep_capsule_b,
            "--stage-keep-capsule-b",
        )

        stage_keep_arr = _stage_capsule_protect_mask(
            combined_pos,
            endpoint_a=stage_keep_a,
            endpoint_b=stage_keep_b,
            radius=float(args.stage_keep_capsule_radius),
        )

        print(
            f"  [stage-keep] inside={int(stage_keep_arr.sum()):,} "
            f"/ {len(stage_keep_arr):,} "
            f"a=({stage_keep_a[0]:.3f},{stage_keep_a[1]:.3f},{stage_keep_a[2]:.3f}) "
            f"b=({stage_keep_b[0]:.3f},{stage_keep_b[1]:.3f},{stage_keep_b[2]:.3f}) "
            f"radius={float(args.stage_keep_capsule_radius):.3f}"
        )
    else:
        stage_keep_arr = np.ones((len(combined_pos),), dtype=bool)

    outside_dynamic_allowed = ~dynamic_allowed_arr
    outside_stage_keep = ~stage_keep_arr

    outside_mode = str(getattr(args, "outside_dynamic_mode", "drop_moving_force_static"))
    moving_outside_dynamic = (
        outside_dynamic_allowed
        & has_vel_arr.astype(bool)
        & (
            speed_per_video_frame_all
            > float(getattr(args, "outside_dynamic_motion_drop_thresh", 0.003))
        )
    )

    if outside_mode == "force_static":
        drop_roi_rows = outside_stage_keep
        force_static_roi_rows = outside_dynamic_allowed & (~drop_roi_rows)

    elif outside_mode == "drop_moving_force_static":
        drop_roi_rows = outside_stage_keep | moving_outside_dynamic
        force_static_roi_rows = outside_dynamic_allowed & (~drop_roi_rows)

    elif outside_mode == "drop_all":
        drop_roi_rows = outside_stage_keep | outside_dynamic_allowed
        force_static_roi_rows = np.zeros((len(combined_pos),), dtype=bool)

    else:
        raise ValueError(f"Unknown outside_dynamic_mode: {outside_mode}")

    print(
        "  [dynamic-volume-filter] "
        f"before={len(combined_pos):,} "
        f"dynamic_allowed={int(dynamic_allowed_arr.sum()):,} "
        f"stage_keep={int(stage_keep_arr.sum()):,} "
        f"force_static_outside={int(force_static_roi_rows.sum()):,} "
        f"drop_moving_outside={int((drop_roi_rows & moving_outside_dynamic).sum()):,} "
        f"drop_outside_keep={int(outside_stage_keep.sum()):,} "
        f"drop_total={int(drop_roi_rows.sum()):,}"
    )

    if drop_roi_rows.any():
        keep_roi_rows = ~drop_roi_rows

        combined_pos = combined_pos[keep_roi_rows]
        combined_col = combined_col[keep_roi_rows]
        combined_pri = combined_pri[keep_roi_rows]
        vel_arr = vel_arr[keep_roi_rows]
        has_vel_arr = has_vel_arr[keep_roi_rows]
        dynamic_protect_arr = dynamic_protect_arr[keep_roi_rows]
        dynamic_allowed_arr = dynamic_allowed_arr[keep_roi_rows]
        stage_keep_arr = stage_keep_arr[keep_roi_rows]
        force_static_roi_rows = force_static_roi_rows[keep_roi_rows]
        point_offsets_frames = point_offsets_frames[keep_roi_rows]

    if force_static_roi_rows.any():
        vel_arr[force_static_roi_rows] = 0.0
        has_vel_arr[force_static_roi_rows] = False

    n_total = len(combined_pos)

    if n_total == 0:
        return 'world_per_video_frame' if args.trigs_velocity_per_frame else 'world_per_keyframe_interval'

    # Drop very tiny tracked velocities in tracked-only mode instead of saving
    # zero-velocity rows into TriGS.
    noise_floor = float(getattr(args, 'velocity_noise_floor', 0.0))
    if noise_floor > 0.0 and not (include_untracked_fresh or inherit_vel):
        mag = np.linalg.norm(vel_arr, axis=1)
        keep = mag >= noise_floor

        combined_pos = combined_pos[keep]
        combined_col = combined_col[keep]
        combined_pri = combined_pri[keep]
        vel_arr = vel_arr[keep]
        has_vel_arr = has_vel_arr[keep]
        dynamic_protect_arr = dynamic_protect_arr[keep]
        dynamic_allowed_arr = dynamic_allowed_arr[keep]
        stage_keep_arr = stage_keep_arr[keep]
        n_total = len(combined_pos)
        point_offsets_frames = point_offsets_frames[keep]

        if n_total == 0:
            return 'world_per_video_frame' if args.trigs_velocity_per_frame else 'world_per_keyframe_interval'

    # With --motion-budget the per-frame cap is applied at finalize, to still rows
    # only (moving rows are never thinned), so assembly keeps every row.
    assembly_cap = 0 if bool(getattr(args, 'motion_budget', False)) else int(args.target_points_per_frame)
    if (
        assembly_cap <= 0
        and not bool(getattr(args, 'motion_budget', False))
        and bool(getattr(args, 'dedup_when_no_target', False))
        and float(args.dedup_radius) > 0.0
    ):
        kept_pos, kept_col, _kept_pri, kept_idx = _priority_voxel_once_with_indices(
            combined_pos, combined_col, combined_pri, float(args.dedup_radius)
        )
        combined_pos = kept_pos.astype(np.float32, copy=False)
        combined_col = kept_col.astype(np.float32, copy=False)
        vel_arr = vel_arr[kept_idx]
        has_vel_arr = has_vel_arr[kept_idx]
        dynamic_protect_arr = dynamic_protect_arr[kept_idx]
        dynamic_allowed_arr = dynamic_allowed_arr[kept_idx]
        stage_keep_arr = stage_keep_arr[kept_idx]
        point_offsets_frames = point_offsets_frames[kept_idx]
        combined_pri = combined_pri[kept_idx]
        n_total = len(combined_pos)

    if assembly_cap > 0 and n_total > assembly_cap:
        detail_active = (
            getattr(args, 'detail_sphere_center', None) is not None
            and float(getattr(args, 'detail_sphere_radius', 0.0)) > 0.0
            and float(getattr(args, 'detail_sphere_target_fraction', 0.0)) > 0.0
        )
        if detail_active:
            detail_center = _parse_vec3_arg(
                args.detail_sphere_center,
                '--detail-sphere-center',
            )
            kept_pos, kept_col, _kept_pri, kept_idx, detail_info = (
                _priority_voxel_to_target_with_detail_sphere(
                    combined_pos,
                    combined_col,
                    combined_pri,
                    target_points=int(args.target_points_per_frame),
                    min_radius=float(args.dedup_radius),
                    max_radius=(
                        float(args.dedup_radius_max)
                        if float(args.dedup_radius_max) > 0.10
                        else 0.0
                    ),
                    center=detail_center,
                    radius=float(args.detail_sphere_radius),
                    target_fraction=float(args.detail_sphere_target_fraction),
                )
            )
            print(
                '  [detail-sphere] '
                f"input={detail_info['input_detail']:,}/"
                f"{detail_info['input_total']:,} "
                f"quota={detail_info['detail_quota']:,} "
                f"kept={detail_info['kept_detail']:,}/"
                f"{detail_info['kept_total']:,}"
            )
            stats.setdefault('per_detail_sampling', []).append({
                'frame_idx': int(cloud.frame_idx),
                **detail_info,
            })
        else:
            kept_pos, kept_col, _kept_pri, kept_idx = _priority_voxel_to_target_with_indices(
                combined_pos,
                combined_col,
                combined_pri,
                target_points=args.target_points_per_frame,
                min_radius=args.dedup_radius,
                max_radius=args.dedup_radius_max if args.dedup_radius_max > 0.10 else 0.0,
            )

        combined_pos = kept_pos.astype(np.float32, copy=False)
        combined_col = kept_col.astype(np.float32, copy=False)
        vel_arr = vel_arr[kept_idx]
        has_vel_arr = has_vel_arr[kept_idx]
        dynamic_protect_arr = dynamic_protect_arr[kept_idx]
        dynamic_allowed_arr = dynamic_allowed_arr[kept_idx]
        stage_keep_arr = stage_keep_arr[kept_idx]
        point_offsets_frames = point_offsets_frames[kept_idx]

    # KNN fallback is allowed only in the explicit ablation mode.
    if (
        include_untracked_fresh
        and next_cloud is not None
        and (~has_vel_arr).any()
        and not args.disable_knn_velocity_fallback
    ):
        radius = compute_adaptive_velocity_radius(
            combined_pos[~has_vel_arr],
            next_cloud.positions,
            args.velocity_radius_factor,
            args.velocity_min_radius,
            args.velocity_max_radius,
        )

        fresh_vel, fresh_has = mutual_nn_velocities(
            combined_pos[~has_vel_arr],
            next_cloud.positions,
            radius,
        )

        fresh_mask = ~has_vel_arr
        fresh_indices = np.where(fresh_mask)[0]

        vel_arr[fresh_indices] = fresh_vel
        has_vel_arr[fresh_indices] = fresh_has

        stats['per_velocity_pair'].append({
            'frame_idx': int(cloud.frame_idx),
            'next_frame_idx': int(next_cloud.frame_idx),
            'radius_knn': float(radius),
            'knn_matches': int(fresh_has.sum()),
            'knn_candidates': int(fresh_mask.sum()),
            'tracked_already': int((~fresh_mask).sum()),
        })

    # Optional noise floor for modes that carry untracked fresh rows.
    if noise_floor > 0.0 and (include_untracked_fresh or inherit_vel):
        mag = np.linalg.norm(vel_arr, axis=1)
        low = has_vel_arr & (mag < noise_floor)
        vel_arr[low] = 0.0
        has_vel_arr[low] = False

    # Anchor the point to its actual physical sub-frame time
    time_value = float((int(cloud.frame_idx) - int(args.frame_start)) / float(max(denom, 1)))
    point_times = time_value + (point_offsets_frames / float(max(denom, 1)))
    times = point_times.reshape(-1, 1).astype(np.float32)

    if next_cloud is not None:
        frame_delta = int(next_cloud.frame_idx - cloud.frame_idx)
    else:
        frame_delta = int(getattr(args, 'temporal_frame_delta', 1))
    frame_delta = max(frame_delta, 1)

    if args.trigs_velocity_per_frame:
        vel_out = (vel_arr.astype(np.float32, copy=False) / float(frame_delta)).astype(np.float32, copy=False)
        velocity_units = 'world_per_video_frame'
    else:
        vel_out = vel_arr.astype(np.float32, copy=False)
        velocity_units = 'world_per_keyframe_interval'

    # Static classification uses world units per VIDEO frame, regardless of
    # whether the NPZ stores velocity per video frame or per keyframe interval.
    speed_per_video_frame = (
        np.linalg.norm(vel_arr, axis=1).astype(np.float32)
        / float(frame_delta)
    )

    static_floor = float(getattr(args, "static_motion_floor", 0.0))
    static_allow_protected = bool(
        getattr(args, "static_allow_protected", False)
    )
    if static_floor > 0.0:
        is_static_arr = (
            has_vel_arr
            & (speed_per_video_frame <= static_floor)
        )
        if not static_allow_protected:
            is_static_arr &= ~dynamic_protect_arr
    else:
        is_static_arr = np.zeros((len(combined_pos),), dtype=bool)

    # ROI rule: rows outside dynamic-allow that survived are persistent static.
    if "dynamic_allowed_arr" in locals():
        is_static_arr |= (~dynamic_allowed_arr)

    # Protected actor rows normally win. A known-static capture may explicitly
    # opt into classifying low-motion protected rows as persistent geometry.
    if not static_allow_protected:
        is_static_arr[dynamic_protect_arr] = False

    # Static rows should not seed dynamic motion.
    if is_static_arr.any():
        vel_arr[is_static_arr] = 0.0
        has_vel_arr[is_static_arr] = False

        if args.trigs_velocity_per_frame:
            vel_out = (
                vel_arr.astype(np.float32, copy=False) / float(frame_delta)
            ).astype(np.float32, copy=False)
        else:
            vel_out = vel_arr.astype(np.float32, copy=False)

    if args.speed_adaptive_duration:
        durations = scale_duration_by_speed(
            vel_out,
            has_vel_arr,
            one_frame_duration,
            args.duration_min_scale,
            args.duration_max_scale,
        )
    else:
        durations = np.full((len(combined_pos), 1), one_frame_duration, dtype=np.float32)

    source_frames = np.full((len(combined_pos), 1), int(cloud.frame_idx), dtype=np.int32)

    all_positions.append(combined_pos.astype(np.float32, copy=False))
    all_colors.append(combined_col.astype(np.float32, copy=False))
    all_velocities.append(vel_out.astype(np.float32, copy=False))
    all_times.append(times.astype(np.float32, copy=False))
    all_durations.append(durations.astype(np.float32, copy=False))
    all_has_velocity.append(has_vel_arr.astype(bool, copy=False))
    all_source_frames.append(source_frames.astype(np.int32, copy=False))

    all_is_static.append(is_static_arr.astype(bool, copy=False))
    all_dynamic_protect.append(dynamic_protect_arr.astype(bool, copy=False))
    all_dynamic_allowed.append(dynamic_allowed_arr.astype(bool, copy=False))
    all_stage_keep.append(stage_keep_arr.astype(bool, copy=False))

    stats.setdefault("static_dynamic", {"static": 0, "dynamic": 0, "protected": 0})
    stats["static_dynamic"]["static"] += int(is_static_arr.sum())
    stats["static_dynamic"]["dynamic"] += int((~is_static_arr).sum())
    stats["static_dynamic"]["protected"] += int(dynamic_protect_arr.sum())

    stats.setdefault("dynamic_volume_filter", {
        "dynamic_allowed": 0,
        "outside_dynamic_allowed": 0,
        "stage_keep": 0,
        "outside_stage_keep": 0,
    })
    stats["dynamic_volume_filter"]["dynamic_allowed"] += int(dynamic_allowed_arr.sum())
    stats["dynamic_volume_filter"]["outside_dynamic_allowed"] += int((~dynamic_allowed_arr).sum())
    stats["dynamic_volume_filter"]["stage_keep"] += int(stage_keep_arr.sum())
    stats["dynamic_volume_filter"]["outside_stage_keep"] += int((~stage_keep_arr).sum())

    return velocity_units

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but torch.cuda.is_available() is False.')

    temporal_frame_delta = int(getattr(args, 'temporal_frame_delta', 1))
    if temporal_frame_delta < 1:
        raise ValueError('--temporal-frame-delta must be >= 1')
    if args.temporal_backend == 'memfof':
        if args.disable_temporal_tracking:
            raise ValueError('--temporal-backend memfof needs temporal tracking enabled')
        if not args.centred_velocity:
            raise ValueError(
                '--temporal-backend memfof requires --centred-velocity (MEMFOF flows '
                'come from the (t-d, t, t+d) frame triplet)'
            )
    if getattr(args, 'memfof_compose_hops', False) and args.temporal_backend != 'memfof':
        raise ValueError('--memfof-compose-hops requires --temporal-backend memfof')
    if args.motion_budget and int(args.motion_budget_moving_per_frame) < 0:
        raise ValueError('--motion-budget-moving-per-frame must be >= 0 (0 = no cap)')
    if args.rigid_cluster_motion:
        if not args.centred_velocity:
            raise ValueError('--rigid-cluster-motion requires --centred-velocity')
        if temporal_frame_delta != int(args.frame_step):
            raise ValueError(
                '--rigid-cluster-motion fits each keyframe to its neighbour keyframes, '
                f'so --temporal-frame-delta ({temporal_frame_delta}) must equal '
                f'--frame-step ({int(args.frame_step)})'
            )
    if int(args.two_view_support_knn) < 1:
        raise ValueError('--two-view-support-knn must be >= 1')
    if float(args.two_view_support_radius) < 0.0:
        raise ValueError('--two-view-support-radius must be >= 0')
    if float(args.two_view_support_radius_factor) <= 0.0:
        raise ValueError('--two-view-support-radius-factor must be > 0')
    if float(args.pair_calibrated_max_error_px) < 0.0:
        raise ValueError('--pair-calibrated-max-error-px must be >= 0')
    if int(args.roma_v2_spatial_grid) < 0:
        raise ValueError('--roma-v2-spatial-grid must be >= 0')
    if not 0.0 <= float(args.roma_v2_spatial_fraction) <= 1.0:
        raise ValueError('--roma-v2-spatial-fraction must be in [0, 1]')
    detail_radius = float(args.detail_sphere_radius)
    detail_fraction = float(args.detail_sphere_target_fraction)
    detail_crop_radius = float(args.detail_sphere_crop_radius)
    if detail_radius < 0.0:
        raise ValueError('--detail-sphere-radius must be >= 0')
    if not 0.0 <= detail_fraction < 1.0:
        raise ValueError('--detail-sphere-target-fraction must be in [0, 1)')
    if detail_fraction > 0.0 and (
        args.detail_sphere_center is None or detail_radius <= 0.0
    ):
        raise ValueError(
            'A positive --detail-sphere-target-fraction requires '
            '--detail-sphere-center and a positive --detail-sphere-radius'
        )
    if detail_radius > 0.0 and args.detail_sphere_center is None:
        raise ValueError(
            '--detail-sphere-radius requires --detail-sphere-center'
        )
    if detail_crop_radius > 0.0 and args.detail_sphere_center is None:
        raise ValueError(
            '--detail-sphere-crop-radius requires --detail-sphere-center'
        )
    if args.detail_sphere_center is not None:
        _parse_vec3_arg(args.detail_sphere_center, '--detail-sphere-center')
    if float(args.detail_sphere_camera_min_margin_px) < 0.0:
        raise ValueError('--detail-sphere-camera-min-margin-px must be >= 0')
    if float(args.detail_sphere_crop_scale) < 1.0:
        raise ValueError('--detail-sphere-crop-scale must be >= 1')
    if detail_crop_radius < 0.0:
        raise ValueError('--detail-sphere-crop-radius must be >= 0')
    if int(args.detail_sphere_crop_min_size_px) < 8:
        raise ValueError('--detail-sphere-crop-min-size-px must be >= 8')
    if int(args.detail_sphere_crop_num_matches) < 0:
        raise ValueError('--detail-sphere-crop-num-matches must be >= 0')
    if int(args.depth_guide_num_matches) < 0:
        raise ValueError('--depth-guide-num-matches must be >= 0')
    if not 0.0 < float(args.detail_sphere_camera_max_view_angle_deg) <= 180.0:
        raise ValueError(
            '--detail-sphere-camera-max-view-angle-deg must be in (0, 180]'
        )

    audit_requested = (
        args.spatial_audit_camera is not None
        or args.spatial_audit_roi is not None
    )
    if audit_requested and (
        args.spatial_audit_camera is None
        or args.spatial_audit_roi is None
        or args.match_debug_dir is None
    ):
        raise ValueError(
            '--spatial-audit-camera and --spatial-audit-roi must be used '
            'together with --match-debug-dir'
        )
    spatial_audit_roi = (
        _parse_roi_arg(args.spatial_audit_roi, '--spatial-audit-roi')
        if audit_requested
        else None
    )

    reference_json_path = Path(args.reference_json)
    cameras, scene_meta = load_reference_export(reference_json_path)
    depth_guide_plan = (
        load_depth_guide_plan(
            Path(args.depth_guide_plan),
            cameras,
            reference_json_path,
        )
        if args.depth_guide_plan is not None
        else None
    )
    if audit_requested and not (
        0 <= int(args.spatial_audit_camera) < len(cameras)
    ):
        raise ValueError(
            f'--spatial-audit-camera must be in [0, {len(cameras) - 1}]'
        )
    if audit_requested:
        audit_camera = cameras[int(args.spatial_audit_camera)]
        x0, y0, x1, y1 = spatial_audit_roi
        if x0 < 0 or y0 < 0 or x1 >= audit_camera.width or y1 >= audit_camera.height:
            raise ValueError(
                '--spatial-audit-roi must lie inside camera '
                f'{int(args.spatial_audit_camera)} image bounds '
                f'{audit_camera.width}x{audit_camera.height}'
            )

    input_frame_start = int(
        args.frame_start if args.input_frame_start is None else args.input_frame_start
    )
    input_frame_end = int(
        args.frame_end if args.input_frame_end is None else args.input_frame_end
    )
    if input_frame_start > int(args.frame_start):
        raise ValueError('--input-frame-start cannot be later than --frame-start')
    if input_frame_end < int(args.frame_end):
        raise ValueError('--input-frame-end cannot be earlier than --frame-end')

    frames = list(range(args.frame_start, args.frame_end + 1, args.frame_step))
    if not frames:
        raise ValueError('No keyframes selected. Check --frame-start, --frame-end, and --frame-step.')

    # Dynamic initialization needs a valid t+delta image for every keyframe.
    # A static initialization does not: --disable-temporal-tracking must be
    # able to reconstruct a single frame without reading a nonexistent
    # adjacent frame.
    if not args.disable_temporal_tracking:
        frames = trackable_keyframes(frames, temporal_frame_delta, input_frame_end)
        if not frames:
            raise ValueError(
                'No keyframes have a valid temporal target frame. '
                'Need frame_idx + --temporal-frame-delta <= --input-frame-end.'
            )

    detail_center_arr = (
        _parse_vec3_arg(args.detail_sphere_center, '--detail-sphere-center')
        if args.detail_sphere_center is not None
        else None
    )
    pair_list = choose_camera_pairs(cameras, args.camera_neighbors, args.wide_baseline_per_camera)
    base_pair_count = int(len(pair_list))
    detail_visible_cameras: List[int] = []
    detail_pair_count = 0
    detail_pairs_added = 0
    if (
        args.detail_sphere_center is not None
        and bool(args.detail_sphere_camera_pairs)
    ):
        detail_visible_cameras, detail_pairs = choose_detail_sphere_camera_pairs(
            cameras,
            detail_center_arr,
            min_image_margin_px=float(args.detail_sphere_camera_min_margin_px),
            max_view_angle_deg=float(
                args.detail_sphere_camera_max_view_angle_deg
            ),
        )
        detail_pair_count = int(len(detail_pairs))
        pair_set = set(pair_list)
        before = len(pair_set)
        pair_set.update(detail_pairs)
        detail_pairs_added = int(len(pair_set) - before)
        pair_list = sorted(pair_set)
        print(
            '[detail-sphere-pairs] '
            f'visible_cameras={len(detail_visible_cameras)} '
            f'overlap_pairs={detail_pair_count} '
            f'added={detail_pairs_added} total={len(pair_list)} '
            f'margin={float(args.detail_sphere_camera_min_margin_px):.1f}px '
            f'max_view_angle='
            f'{float(args.detail_sphere_camera_max_view_angle_deg):.1f}deg'
        )
    if not pair_list:
        raise ValueError('Need at least two cameras to form camera pairs.')

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_fingerprint, run_descriptor = build_run_fingerprint(args)
    cache_root = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_path.with_name(output_path.name + ".cache")
    )
    active_cache = cache_root / run_fingerprint[:16]
    active_cache.mkdir(parents=True, exist_ok=True)
    manifest_path = active_cache / "manifest.json"
    if not manifest_path.exists():
        _atomic_write_json(
            manifest_path,
            {
                "schema": CACHE_SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "descriptor": run_descriptor,
            },
        )

    print(f'Cameras: {len(cameras)} | Camera pairs (near+far): {len(pair_list)}')
    print(
        f'Keyframes: {frames[0]}..{frames[-1]} step={args.frame_step} '
        f'| temporal delta={temporal_frame_delta}'
    )

    model = create_roma_backend(
        backend=args.roma_backend,
        device=args.device,
        model=args.roma_model,
        v2_setting=args.roma_v2_setting,
        symmetric=args.roma_symmetric,
        upsample_preds=args.roma_upsample_preds,
        v2_mask_oversample=args.roma_v2_mask_oversample,
        v2_spatial_grid=args.roma_v2_spatial_grid,
        v2_spatial_fraction=args.roma_v2_spatial_fraction,
    )
    print(f'RoMa backend: {model.name} | reverse={getattr(model, "supports_reverse", False)}')
    temporal_model = model
    if args.temporal_backend == 'memfof':
        temporal_model = MemfofTemporalBackend(
            device=args.device,
            checkpoint=args.memfof_checkpoint,
            max_side=int(args.memfof_max_side),
        )
        print(
            f'Temporal backend: {temporal_model.name} (max side {int(args.memfof_max_side)} px)'
            + (' | composed one-frame hops' if args.memfof_compose_hops else '')
        )
    if args.roma_backend == 'v2' and int(args.roma_v2_spatial_grid) > 0:
        spatial_matches = int(round(
            int(args.roma_num_matches) * float(args.roma_v2_spatial_fraction)
        ))
        print(
            f'[roma-spatial] grid={int(args.roma_v2_spatial_grid)}x'
            f'{int(args.roma_v2_spatial_grid)} '
            f'spatial={spatial_matches:,} global='
            f'{int(args.roma_num_matches) - spatial_matches:,} matches/pair'
        )

    images_root = Path(args.images_root)
    masks_root = Path(args.masks_root) if args.masks_root else None

    timing_candidate = None
    if getattr(args, 'timing_json', None):
        if args.sync_json or float(args.rs_readout_ms) != 0.0:
            raise ValueError('--timing-json replaces --sync-json and --rs-readout-ms; pass only --timing-json.')
        timing_candidate = _CT.load_timing_candidate(args.timing_json)
        if abs(float(timing_candidate.fps) - float(args.fps)) > 1e-6:
            raise ValueError(f'{args.timing_json} fps {timing_candidate.fps} != --fps {args.fps}')
        entries = [timing_candidate.lookup(str(cam.folder)) for cam in cameras]
        missing = [str(cam.folder) for cam, e in zip(cameras, entries) if e is None]
        if missing:
            raise ValueError(f'{args.timing_json} has no entry for cameras {missing}')
        sync_offsets_sec = np.asarray([e.c_s for e in entries], dtype=np.float64)
        sync_keys = [str(cam.folder) for cam in cameras]
        sync_enabled = True
        rs_axis_eff, rs_reference_eff = timing_candidate.axis, 'midpoint'
        if rs_axis_eff == 'row':
            if args.rotation_json:
                raise ValueError('row-axis --timing-json does not use --rotation-json')
            rs_signs_arr = np.asarray([1.0 if e.rs else 0.0 for e in entries], dtype=np.float64)
        else:
            if not args.rotation_json:
                raise ValueError('column-axis --timing-json still needs --rotation-json for the per-camera sign')
            rs_signs_arr = load_rs_signs_for_cameras(
                args.rotation_json, cameras, flip_sign=bool(args.rs_flip_sign)) * np.asarray(
                [1.0 if e.rs else 0.0 for e in entries])
        rs_readout_frames = float(timing_candidate.readout_ms) * 1e-3 * float(args.fps)
        print(f'[timing] {args.timing_json}: midpoint constants for {len(cameras)} cameras '
              f'(sync.json NOT read); readout {timing_candidate.readout_ms:.2f} ms along {rs_axis_eff}s; '
              f'rs on {int(np.sum(rs_signs_arr != 0))} / off {int(np.sum(rs_signs_arr == 0))}; statuses '
              + str({st: sum(1 for e in entries if e.status == st) for st in sorted({e.status for e in entries})}))
    else:
        sync_offsets_sec, sync_keys = load_sync_offsets_for_cameras(args.sync_json, cameras)
        sync_enabled = bool(args.sync_json)
        rs_axis_eff, rs_reference_eff = str(args.rs_axis), str(args.rs_reference)
        if rs_reference_eff == 'midpoint':
            raise ValueError(
                '--rs-reference midpoint needs --timing-json midpoint constants: sync.json offsets already '
                'contain each camera\'s content-dependent readout delay, so adding a midpoint term would '
                'double-count it.')
        if rs_axis_eff == 'row':
            rs_signs_arr = np.ones((len(cameras),), dtype=np.float64)
        else:
            rs_signs_arr = (
                load_rs_signs_for_cameras(
                    args.rotation_json, cameras, flip_sign=bool(args.rs_flip_sign)
                )
                if args.rotation_json
                else None
            )
        rs_readout_frames = float(args.rs_readout_ms) * 1e-3 * float(args.fps)
        if rs_readout_frames != 0.0 and rs_signs_arr is None:
            raise ValueError(
                '--rs-readout-ms requires --rotation-json to know each camera\'s '
                'readout direction in the rotated images.'
            )
    if rs_signs_arr is not None and rs_readout_frames != 0.0:
        print(
            f'[rs] rolling-shutter compensation active: readout '
            f'{rs_readout_frames / float(args.fps) * 1000.0:.2f} ms full-span = {rs_readout_frames:.4f} frames, '
            f'axis {rs_axis_eff}, reference {rs_reference_eff}'
        )
    # Create a numpy array of frame-unit offsets mapped to the camera indices
    sync_offsets_frames_arr = sync_offsets_sec * float(args.fps) if sync_enabled else None

    stats: Dict[str, object] = {
        'fps': float(args.fps),
        'frames': [int(f) for f in frames],
        'input_frame_start': input_frame_start,
        'input_frame_end': input_frame_end,
        'camera_count': int(len(cameras)),
        'camera_pairs': [[int(a), int(b)] for a, b in pair_list],
        'reference_json': str(args.reference_json),
        'roma_backend': str(model.name),
        'roma_sampling': {
            'v2_spatial_grid': int(args.roma_v2_spatial_grid),
            'v2_spatial_fraction': float(args.roma_v2_spatial_fraction),
        },
        'pair_geometry': {
            'fitted_ransac_px': float(args.pair_ransac_px),
            'calibrated_max_error_px': float(
                args.pair_calibrated_max_error_px
            ),
        },
        'detail_sampling': {
            'center': args.detail_sphere_center,
            'radius': float(args.detail_sphere_radius),
            'target_fraction': float(args.detail_sphere_target_fraction),
            'camera_pair_augmentation': bool(args.detail_sphere_camera_pairs),
            'camera_min_margin_px': float(args.detail_sphere_camera_min_margin_px),
            'camera_max_view_angle_deg': float(
                args.detail_sphere_camera_max_view_angle_deg
            ),
            'crop_matching': bool(args.detail_sphere_crop_matching),
            'crop_radius': float(args.detail_sphere_crop_radius),
            'crop_scale': float(args.detail_sphere_crop_scale),
            'crop_min_size_px': int(args.detail_sphere_crop_min_size_px),
            'crop_num_matches': int(args.detail_sphere_crop_num_matches),
            'visible_cameras': [int(v) for v in detail_visible_cameras],
            'base_pair_count': int(base_pair_count),
            'eligible_detail_pair_count': int(detail_pair_count),
            'detail_pairs_added': int(detail_pairs_added),
            'final_pair_count': int(len(pair_list)),
        },
        'depth_guide': {
            'enabled': depth_guide_plan is not None,
            'plan': str(args.depth_guide_plan) if args.depth_guide_plan else None,
            'target_frame': (
                int(depth_guide_plan['target_frame'])
                if depth_guide_plan is not None
                else None
            ),
            'entries': (
                int(len(depth_guide_plan['entries']))
                if depth_guide_plan is not None
                else 0
            ),
            'num_matches': int(args.depth_guide_num_matches),
            'apply_all_frames': bool(args.depth_guide_apply_all_frames),
            'imports_depth_points': False,
        },
        'memory_mode': 'disk_checkpointed_keyframes',
        'cache': {
            'schema': CACHE_SCHEMA_VERSION,
            'run_fingerprint': run_fingerprint,
            'directory': str(active_cache),
            'resume_enabled': not bool(args.no_resume),
            'hits': 0,
            'recomputed': 0,
            'stale_or_corrupt': 0,
            'checkpoint_bytes': 0,
        },
        'temporal_frame_delta': int(temporal_frame_delta),
        'include_untracked_fresh': bool(getattr(args, 'include_untracked_fresh', False)),
        'untracked_velocity_from_tracked': bool(getattr(args, 'untracked_velocity_from_tracked', False)),
        'untracked_velocity_radius': float(getattr(args, 'untracked_velocity_radius', 0.15)),
        'temporal_min_certainty_fast': float(getattr(args, 'temporal_min_certainty_fast', 0.0)),
        'temporal_fast_flow_px': float(getattr(args, 'temporal_fast_flow_px', 8.0)),
        'two_view_core_filter': {
            'enabled': bool(args.filter_two_view_by_three_view_core),
            'radius': float(args.two_view_support_radius),
            'radius_factor': float(args.two_view_support_radius_factor),
            'knn': int(args.two_view_support_knn),
        },
        'static_classification': {
            'motion_floor': float(args.static_motion_floor),
            'allow_protected': bool(args.static_allow_protected),
            'spatial_dedup_radius': float(args.static_spatial_dedup_radius),
            'max_points': int(args.static_max_points),
        },
        'per_frame': [],
        'per_velocity_pair': [],
        'temporal_tracking': {
            'enabled': not args.disable_temporal_tracking,
            'per_pair': [],
        },
        'sync': {
            'enabled': bool(sync_enabled),
            'sync_json': str(args.sync_json) if args.sync_json else None,
            'offsets_sec': [float(x) for x in sync_offsets_sec.tolist()],
            'keys': [None if k is None else str(k) for k in sync_keys],
            'fresh_motion_compensate': bool(args.fresh_motion_compensate) and sync_enabled,
            'motion_slack': float(args.motion_slack),
            'motion_slack_max_px': float(args.motion_slack_max_px),
            'rolling_shutter': {
                'enabled': bool(rs_signs_arr is not None and rs_readout_frames != 0.0),
                'rotation_json': str(args.rotation_json) if args.rotation_json else None,
                'readout_ms': float(rs_readout_frames) / float(args.fps) * 1000.0,
                'readout_frames': float(rs_readout_frames),
                'flip_sign': bool(args.rs_flip_sign),
                'axis': rs_axis_eff,
                'reference': rs_reference_eff,
                'convention': _CT.CONVENTION if rs_reference_eff == 'midpoint' else 'legacy content-referenced',
                'timing_json': str(args.timing_json) if getattr(args, 'timing_json', None) else None,
                'timing_sha256': (_file_content_hash(Path(args.timing_json))
                                  if getattr(args, 'timing_json', None) else None),
                'timing_status': ({str(cam.folder): timing_candidate.lookup(str(cam.folder)).status for cam in cameras}
                                  if timing_candidate is not None else None),
                'signs': (
                    [int(s) for s in rs_signs_arr.tolist()]
                    if rs_signs_arr is not None else None
                ),
            },
        },
    }

    # Normalize times over the full source video range, not just the last keyframe.
    denom = max(int(args.frame_end) - int(args.frame_start), 1)
    one_frame_duration = float(args.frame_step / float(denom))

    velocity_units = (
        'world_per_video_frame'
        if args.trigs_velocity_per_frame
        else 'world_per_keyframe_interval'
    )

    def build_fresh(
        frame_idx: int,
        *,
        reconstruct: bool = True,
        images_next: Optional[List[np.ndarray]] = None,
        paths_next: Optional[List[Path]] = None,
        masks_next: Optional[List[Optional[np.ndarray]]] = None,
        warp_cache_out: Optional[List[Optional[DenseMatch]]] = None,
        frame_delta: Optional[int] = None,
    ) -> Tuple[List[Path], List[np.ndarray], Optional[List[Optional[np.ndarray]]], FramePointCloud]:
        _log_mem(args.log_memory, f'before load/reconstruct frame {frame_idx}')

        if sync_enabled:
            camera_frame_indices = sync_aware_camera_frames(
                frame_idx,
                fps=args.fps,
                offsets_sec=sync_offsets_sec,
                frame_start=input_frame_start,
                frame_end=input_frame_end,
            )
            paths, images = load_frame_images_for_camera_frames(
                images_root,
                cameras,
                camera_frame_indices,
                args.image_pattern,
            )
            masks = (
                load_frame_masks_for_camera_frames(
                    masks_root, cameras, camera_frame_indices, args.mask_pattern
                )
                if masks_root is not None
                else None
            )
        else:
            camera_frame_indices = np.full((len(cameras),), int(frame_idx), dtype=np.int64)
            paths, images = load_frame_images(
                images_root,
                cameras,
                frame_idx,
                args.image_pattern,
            )

            masks = (
                load_frame_masks(masks_root, cameras, frame_idx, args.mask_pattern)
                if masks_root is not None
                else None
            )

        if not reconstruct:
            cloud = FramePointCloud(
                frame_idx=int(frame_idx),
                positions=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
                observations=[],
                camera_frame_indices=np.asarray(camera_frame_indices, dtype=np.int64),
                debug={'inputs_only': 1.0},
            )
            _log_mem(args.log_memory, f'after load-only frame {frame_idx}')
            return paths, images, masks, cloud

        capture_residual = (
            camera_capture_residual_frames(
                frame_idx,
                camera_frame_indices,
                fps=args.fps,
                offsets_sec=sync_offsets_sec,
            )
            if sync_enabled
            else None
        )

        cloud = reconstruct_frame_cloud(
            frame_idx=frame_idx,
            images_root=images_root,
            cameras=cameras,
            image_pattern=args.image_pattern,
            model=model,
            device=args.device,
            pair_list=pair_list,
            roma_num_matches=args.roma_num_matches,
            roma_min_certainty=args.roma_min_certainty,
            pair_ransac_px=args.pair_ransac_px,
            pair_calibrated_max_error_px=float(
                args.pair_calibrated_max_error_px
            ),
            min_pair_inliers=args.min_pair_inliers,
            same_camera_merge_px=args.same_camera_merge_px,
            min_track_views=args.min_track_views,
            reproj_thresh_px=args.reproj_thresh_px,
            min_triangulation_angle_deg=args.min_triangulation_angle_deg,
            scene_meta=scene_meta,
            scene_radius_mult=args.scene_radius_mult,
            filter_two_view_by_core=bool(args.filter_two_view_by_three_view_core),
            two_view_support_radius=float(args.two_view_support_radius),
            two_view_support_radius_factor=float(args.two_view_support_radius_factor),
            two_view_support_knn=int(args.two_view_support_knn),
            masks_root=masks_root,
            mask_pattern=args.mask_pattern,
            mask_min_fg_alpha=args.mask_min_fg_alpha,
            mask_min_fg_fraction=args.mask_min_fg_fraction,
            color_mode=args.color_mode,
            color_trim_frac=args.color_trim_frac,
            images_override=images,
            paths_override=paths,
            masks_override=masks,
            images_next_override=images_next,
            paths_next_override=paths_next,
            masks_next_override=masks_next,
            capture_residual_frames=capture_residual,
            temporal_frame_delta=int(frame_delta) if frame_delta is not None else temporal_frame_delta,
            fresh_motion_compensate=bool(args.fresh_motion_compensate),
            motion_slack=float(args.motion_slack),
            motion_slack_max_px=float(args.motion_slack_max_px),
            calibrated_motion_slack=bool(args.calibrated_motion_slack),
            warp_cache_out=warp_cache_out,
            rs_signs=rs_signs_arr,
            rs_readout_frames=rs_readout_frames,
            rs_axis=rs_axis_eff,
            rs_reference=rs_reference_eff,
            detail_sphere_center=detail_center_arr,
            detail_sphere_radius=float(args.detail_sphere_radius),
            detail_sphere_crop_matching=bool(args.detail_sphere_crop_matching),
            detail_sphere_crop_radius=float(args.detail_sphere_crop_radius),
            detail_sphere_crop_scale=float(args.detail_sphere_crop_scale),
            detail_sphere_crop_min_size_px=int(args.detail_sphere_crop_min_size_px),
            detail_sphere_crop_num_matches=int(args.detail_sphere_crop_num_matches),
            detail_sphere_camera_max_view_angle_deg=float(
                args.detail_sphere_camera_max_view_angle_deg
            ),
            depth_guide_entries=(
                depth_guide_plan['entries']
                if depth_guide_plan is not None
                and (
                    bool(args.depth_guide_apply_all_frames)
                    or int(frame_idx) == int(depth_guide_plan['target_frame'])
                )
                else None
            ),
            depth_guide_num_matches=int(args.depth_guide_num_matches),
            match_debug_dir=(
                Path(args.match_debug_dir) / f'frame-{int(frame_idx):06d}'
                if args.match_debug_dir is not None
                else None
            ),
            match_debug_cell_px=int(args.match_debug_cell_px),
            spatial_audit_camera=(
                int(args.spatial_audit_camera) if audit_requested else None
            ),
            spatial_audit_roi=spatial_audit_roi,
        )

        cloud.camera_frame_indices = np.asarray(camera_frame_indices, dtype=np.int64)

        stats['per_frame'].append({
            'frame_idx': int(frame_idx),
            'camera_frame_min': int(np.min(camera_frame_indices)),
            'camera_frame_max': int(np.max(camera_frame_indices)),
            'camera_frames': [int(x) for x in camera_frame_indices.tolist()],
            **cloud.debug,
        })

        print(
            f"Frame {frame_idx} | pts={len(cloud.positions):,} "
            f"accepted={cloud.debug['components_accepted']} "
            f"rej_angle={cloud.debug['rej_triangulation_angle']} "
            f"rej_mask={cloud.debug['rej_mask']} "
            f"rej_reproj={cloud.debug['rej_reprojection']}"
        )

        _log_mem(args.log_memory, f'after reconstruct frame {frame_idx} pts={len(cloud.positions):,}')
        return paths, images, masks, cloud

    def _checkpoint_arrays(frame_idx: int, delta: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Arrays of a committed per-frame checkpoint (keyframe or sub-keyframe)."""
        d = temporal_frame_delta if delta is None else int(delta)
        nf = int(frame_idx) if args.disable_temporal_tracking else int(frame_idx + d)
        fp = frame_input_fingerprint(
            frame_idx=int(frame_idx), next_frame_idx=nf, images_root=images_root, masks_root=masks_root,
            cameras=cameras, image_pattern=args.image_pattern, mask_pattern=args.mask_pattern, fps=args.fps,
            offsets_sec=sync_offsets_sec, sync_enabled=sync_enabled, frame_start=input_frame_start,
            frame_end=input_frame_end,
        )
        path = active_cache / f"frame-{int(frame_idx):010d}.npz"
        loaded = load_frame_checkpoint(
            path, frame_idx=int(frame_idx), run_fingerprint=run_fingerprint, input_fingerprint=fp,
            load_arrays=True, trust=bool(args.trust_cache),
        )
        if loaded is None:
            raise RuntimeError(f"Checkpoint invalid while reading sub-keyframe inputs: {path}")
        return loaded["arrays"]

    def _fast_rows(arr: Mapping[str, np.ndarray], select: Optional[np.ndarray] = None) -> np.ndarray:
        return subkeyframe_fast_rows(
            arr, min_speed_spacings=float(args.subkeyframe_min_speed),
            spacing_points=int(args.motion_budget_spacing_points), knn=int(args.motion_budget_knn),
            velocity_per_frame=bool(getattr(args, 'trigs_velocity_per_frame', True)),
            frame_delta=int(temporal_frame_delta), select=select,
        )

    def _plan_subkeyframe_frames() -> List[int]:
        """Frames between keyframes that need their own fast rows (see --subkeyframe-min-speed)."""
        kfs = sorted(int(k) for k in assembled_keyframes)
        fast_counts = {k: int(len(_fast_rows(_checkpoint_arrays(k)))) for k in kfs}
        planned, intervals = plan_subkeyframe_frames(
            fast_counts, min_rows=int(args.subkeyframe_min_rows), frame_end=int(input_frame_end),
            frame_delta=SUBKEYFRAME_TRACK_DELTA, frame_min=args.subkeyframe_frame_min,
            frame_max=args.subkeyframe_frame_max,
        )
        stats['subkeyframes'] = {
            'min_speed_spacings': float(args.subkeyframe_min_speed),
            'min_rows': int(args.subkeyframe_min_rows),
            'keyframe_fast_rows': {str(k): v for k, v in fast_counts.items()},
            'intervals': intervals,
            'planned_frames': [int(f) for f in planned],
        }
        print(
            f"[subkeyframes] keyframe rows >= {float(args.subkeyframe_min_speed):g} spacings/frame: "
            + ", ".join(f"{k}:{v}" for k, v in fast_counts.items())
        )
        print(
            f"[subkeyframes] {len(intervals)} of {max(len(kfs) - 1, 0)} keyframe intervals need sub-keyframes "
            f"(a bounding keyframe has >= {int(args.subkeyframe_min_rows)} fast rows): "
            f"{len(planned)} extra frames {planned}"
        )
        return planned

    assembled_keyframes: List[int] = []
    # Offset solving needs the full per-keyframe observation graphs after the
    # main loop.  They are large (nested per-point/per-camera arrays), so do
    # not retain them for the normal NPZ-only path.
    records = [] if args.solve_offsets_out else None
    if args.solve_offsets_out:
        print(
            "[cache] WARNING: --solve-offsets-out keeps observation graphs in RAM "
            "and disables frame skipping; bounded-memory resume does not cover "
            "offset-solving runs."
        )

    progress_desc = (
        'keyframe reconstruct+assemble'
        if args.disable_temporal_tracking
        else 'keyframe reconstruct+track(t->t+delta)+assemble'
    )
    sub_enabled = float(getattr(args, 'subkeyframe_min_speed', 0.0) or 0.0) > 0.0
    if sub_enabled and args.disable_temporal_tracking:
        raise ValueError('--subkeyframe-min-speed needs temporal tracking (row speeds)')
    if sub_enabled and not bool(getattr(args, 'trigs_velocity_per_frame', True)):
        raise ValueError('--subkeyframe-min-speed needs per-video-frame velocities (sub-keyframes track 1 frame)')
    if sub_enabled and args.solve_offsets_out:
        raise ValueError('--subkeyframe-min-speed cannot be combined with --solve-offsets-out')
    assembled_subframes: List[int] = []
    work_items: List[Tuple[int, int]] = [(i, int(f)) for i, f in enumerate(frames)]
    n_keyframe_items = len(work_items)
    subframes_planned = not sub_enabled
    pbar = tqdm(total=len(work_items), desc=progress_desc)
    work_pos = 0
    while True:
        if not subframes_planned and work_pos == n_keyframe_items:
            subframes_planned = True
            planned = _plan_subkeyframe_frames()
            work_items += [(n_keyframe_items + j, int(f)) for j, f in enumerate(planned)]
            pbar.total = len(work_items)
            pbar.set_description('sub-keyframe reconstruct+track+assemble')
            pbar.refresh()
        if work_pos >= len(work_items):
            break
        idx, frame_idx = work_items[work_pos]
        is_subframe = work_pos >= n_keyframe_items
        # Sub-keyframe rows live ~1 frame (local_stride 1), so they are tracked over one
        # frame each way (triplet t-1, t, t+1); keyframes keep --temporal-frame-delta.
        frame_delta = SUBKEYFRAME_TRACK_DELTA if is_subframe else temporal_frame_delta
        work_pos += 1
        pbar.update(1)
        frame_idx = int(frame_idx)
        next_frame = (
            frame_idx
            if args.disable_temporal_tracking
            else int(frame_idx + frame_delta)
        )
        input_fingerprint = frame_input_fingerprint(
            frame_idx=frame_idx,
            next_frame_idx=next_frame,
            images_root=images_root,
            masks_root=masks_root,
            cameras=cameras,
            image_pattern=args.image_pattern,
            mask_pattern=args.mask_pattern,
            fps=args.fps,
            offsets_sec=sync_offsets_sec,
            sync_enabled=sync_enabled,
            frame_start=input_frame_start,
            frame_end=input_frame_end,
        )
        checkpoint_path = active_cache / f"frame-{frame_idx:010d}.npz"
        cached = None
        can_skip = not args.no_resume and not args.solve_offsets_out
        if can_skip and checkpoint_path.exists():
            cached = load_frame_checkpoint(
                checkpoint_path,
                frame_idx=frame_idx,
                run_fingerprint=run_fingerprint,
                input_fingerprint=input_fingerprint,
                load_arrays=False,
                trust=bool(args.trust_cache),
            )
            if cached is None:
                stats["cache"]["stale_or_corrupt"] += 1
                print(f"[cache] stale/corrupt frame {frame_idx}; recomputing")
        if cached is not None:
            _apply_stats_delta(stats, cached["stats_delta"])
            stats["cache"]["hits"] += 1
            stats["cache"]["checkpoint_bytes"] += int(cached["bytes"])
            (assembled_subframes if is_subframe else assembled_keyframes).append(frame_idx)
            velocity_units = str(cached["velocity_units"])
            print(
                f"[cache] hit frame {frame_idx}: {int(cached['n_rows']):,} rows, "
                f"{int(cached['bytes']) / (1024 ** 2):.1f} MiB"
            )
            _log_mem(args.log_memory, f"after cache hit frame {frame_idx}")
            continue

        stats_before = _stats_snapshot(stats)
        frame_lists: Dict[str, List[np.ndarray]] = {
            name: [] for name in CACHE_ARRAY_SPECS
        }

        if args.disable_temporal_tracking:
            # Genuine single-frame static reconstruction: do not load t+delta
            # and do not request fresh-path temporal compensation warps.
            paths_n = None
            images_n = None
            masks_n = None
            cloud_n = None
            paths_t, images_t, masks_t, cloud_t = build_fresh(frame_idx)
        else:
            # The temporal tracker only needs the next images. Reconstruct the
            # next cloud only when the explicit fresh-point KNN fallback uses
            # it. Load it before t so its images can drive fresh-path sync
            # motion compensation (t->t+delta flow).
            need_next_cloud = bool(
                getattr(args, 'include_untracked_fresh', False)
            ) and not bool(args.disable_knn_velocity_fallback)
            paths_n, images_n, masks_n, cloud_n = build_fresh(
                next_frame, reconstruct=need_next_cloud
            )

            # Build keyframe cloud at t. Dense compensation warps are sampled
            # and released one camera at a time inside reconstruction.
            paths_t, images_t, masks_t, cloud_t = build_fresh(
                frame_idx,
                frame_delta=frame_delta,
                images_next=images_n,
                paths_next=paths_n,
                masks_next=masks_n,
            )

        if not args.disable_temporal_tracking:
            assert cloud_n is not None
            assert paths_n is not None
            assert images_n is not None
            _log_mem(args.log_memory, f'before temporal {frame_idx}->{next_frame}')

            residual_t = camera_capture_residual_frames(
                frame_idx,
                cloud_t.camera_frame_indices,
                fps=args.fps,
                offsets_sec=sync_offsets_sec,
            ) if sync_enabled else None
            residual_n = camera_capture_residual_frames(
                next_frame,
                cloud_n.camera_frame_indices,
                fps=args.fps,
                offsets_sec=sync_offsets_sec,
            ) if sync_enabled else None

            back_frame = int(frame_idx - frame_delta)
            do_backward = (
                bool(getattr(args, 'centred_velocity', False))
                and back_frame >= int(input_frame_start)
            )
            back_loaded = None
            fwd_warps: Optional[List[Optional[DenseMatch]]] = None
            bwd_warps: Optional[List[Optional[DenseMatch]]] = None
            if args.temporal_backend == 'memfof':
                # One MEMFOF inference per camera on (t-d, t, t+d) gives both the
                # forward and the backward warp. At the first keyframe there is no
                # t-d frame: t stands in for it and only the forward flow is used.
                if do_backward:
                    back_loaded = build_fresh(back_frame, reconstruct=False)
                images_prev = back_loaded[1] if back_loaded is not None else images_t
                fwd_warps, bwd_warps = [], []
                for cam_idx in range(len(cameras)):
                    bwd_flow, fwd_flow, flow_scale = temporal_model.triplet_flows(
                        images_prev[cam_idx], images_t[cam_idx], images_n[cam_idx]
                    )
                    h_img, w_img = images_t[cam_idx].shape[:2]
                    fwd_warps.append(MemfofTemporalBackend.make_dense(fwd_flow, flow_scale, h_img, w_img))
                    bwd_warps.append(
                        MemfofTemporalBackend.make_dense(bwd_flow, flow_scale, h_img, w_img)
                        if back_loaded is not None else None
                    )
                if args.memfof_compose_hops:
                    # Replace each warp's sampling by delta one-frame hops: hop c -> c+-1 uses the
                    # forward/backward flow of triplet (c-1, c, c+1). Frames are chosen per camera
                    # with the same sync-aware rule as the keyframe images; the direct flows above
                    # stay in the payload for layout validation only.
                    d_hop = abs(int(frame_delta))
                    # Forward hops read frames t-1 .. t+d (the first hop's triplet is (t-1, t, t+1)),
                    # backward hops t-d .. t+1. Frames outside the input range are clipped, as the
                    # sync-aware selection does, so at the sequence start t stands in for t-1.
                    lo_g = frame_idx - d_hop if back_loaded is not None else frame_idx - 1
                    hop_frames = {
                        g: (
                            sync_aware_camera_frames(
                                g, fps=args.fps, offsets_sec=sync_offsets_sec,
                                frame_start=input_frame_start, frame_end=input_frame_end,
                            )
                            if sync_enabled
                            else np.full(
                                (len(cameras),), int(np.clip(g, input_frame_start, input_frame_end)), dtype=np.int64
                            )
                        )
                        for g in range(lo_g, frame_idx + d_hop + 1)
                    }
                    centres_f = list(range(frame_idx, frame_idx + d_hop))
                    centres_b = (
                        list(range(frame_idx, frame_idx - d_hop, -1)) if back_loaded is not None else []
                    )
                    for cam_idx in range(len(cameras)):
                        known = {frame_idx: images_t[cam_idx], frame_idx + d_hop: images_n[cam_idx]}
                        if back_loaded is not None:
                            known[frame_idx - d_hop] = images_prev[cam_idx]

                        def hop_image(g: int) -> np.ndarray:
                            if g not in known:
                                path_g = image_path(
                                    images_root, cameras[cam_idx], int(hop_frames[g][cam_idx]),
                                    args.image_pattern,
                                )
                                known[g] = np.asarray(Image.open(path_g).convert('RGB'), dtype=np.uint8)
                            return known[g]

                        hop_flows = {
                            c: temporal_model.triplet_flows(hop_image(c - 1), hop_image(c), hop_image(c + 1))
                            for c in sorted(set(centres_f) | set(centres_b))
                        }
                        h_img, w_img = images_t[cam_idx].shape[:2]
                        fwd_warps[cam_idx] = MemfofTemporalBackend.make_composed_dense(
                            fwd_warps[cam_idx].payload["flow"], fwd_warps[cam_idx].payload["scale"],
                            h_img, w_img, [(hop_flows[c][1], hop_flows[c][2]) for c in centres_f],
                        )
                        if back_loaded is not None:
                            bwd_warps[cam_idx] = MemfofTemporalBackend.make_composed_dense(
                                bwd_warps[cam_idx].payload["flow"], bwd_warps[cam_idx].payload["scale"],
                                h_img, w_img, [(hop_flows[c][0], hop_flows[c][2]) for c in centres_b],
                            )
                        del known, hop_flows
                del images_prev
                _log_mem(args.log_memory, f'after memfof flows {frame_idx}')

            tracked_pair = track_points_to_next_keyframe(
                cloud_t=cloud_t,
                cameras=cameras,
                model=temporal_model,
                device=args.device,
                paths_t=paths_t,
                images_t=images_t,
                paths_tnext=paths_n,
                images_tnext=images_n,
                temporal_min_certainty=args.temporal_min_certainty,
                temporal_reproj_thresh_px=args.temporal_reproj_thresh_px,
                min_tracked_views=args.min_tracked_views,
                min_triangulation_angle_deg=args.min_triangulation_angle_deg,
                filter_two_view_by_core=bool(args.filter_two_view_by_three_view_core),
                two_view_support_radius=float(args.two_view_support_radius),
                two_view_support_radius_factor=float(args.two_view_support_radius_factor),
                two_view_support_knn=int(args.two_view_support_knn),
                scene_meta=scene_meta,
                scene_radius_mult=args.scene_radius_mult,
                masks_t=masks_t,
                masks_tnext=masks_n,
                mask_min_fg_alpha=args.mask_min_fg_alpha,
                mask_min_fg_fraction=args.mask_min_fg_fraction,
                capture_residual_t=residual_t,
                capture_residual_tnext=residual_n,
                temporal_frame_delta=frame_delta,
                temporal_cycle_thresh_px=args.temporal_cycle_thresh_px,
                sync_motion_compensate=bool(args.sync_motion_compensate),
                warps_override=fwd_warps,
                motion_slack=float(args.motion_slack),
                motion_slack_max_px=float(args.motion_slack_max_px),
                temporal_min_certainty_fast=float(args.temporal_min_certainty_fast),
                temporal_fast_flow_px=float(args.temporal_fast_flow_px),
                rs_signs=rs_signs_arr,
                rs_readout_frames=rs_readout_frames,
                rs_axis=rs_axis_eff,
                rs_reference=rs_reference_eff,
            )

            pos_t, pos_tn, cols, src_idx, ttl_stats = tracked_pair
            tracked_accel = np.zeros((len(pos_t), 3), dtype=np.float32)
            tracked_has_accel = np.zeros((len(pos_t),), dtype=bool)

            if do_backward:
                _log_mem(args.log_memory, f'before temporal {frame_idx}->{back_frame}')
                paths_p, images_p, masks_p, cloud_p = (
                    back_loaded if back_loaded is not None
                    else build_fresh(back_frame, reconstruct=False)
                )
                back_loaded = None
                residual_p = camera_capture_residual_frames(
                    back_frame,
                    cloud_p.camera_frame_indices,
                    fps=args.fps,
                    offsets_sec=sync_offsets_sec,
                ) if sync_enabled else None
                pos_t_b, pos_tp, _, src_idx_b, back_stats = track_points_to_next_keyframe(
                    cloud_t=cloud_t,
                    cameras=cameras,
                    model=temporal_model,
                    device=args.device,
                    paths_t=paths_t,
                    images_t=images_t,
                    paths_tnext=paths_p,
                    images_tnext=images_p,
                    temporal_min_certainty=args.temporal_min_certainty,
                    temporal_reproj_thresh_px=args.temporal_reproj_thresh_px,
                    min_tracked_views=args.min_tracked_views,
                    min_triangulation_angle_deg=args.min_triangulation_angle_deg,
                    filter_two_view_by_core=bool(args.filter_two_view_by_three_view_core),
                    two_view_support_radius=float(args.two_view_support_radius),
                    two_view_support_radius_factor=float(args.two_view_support_radius_factor),
                    two_view_support_knn=int(args.two_view_support_knn),
                    scene_meta=scene_meta,
                    scene_radius_mult=args.scene_radius_mult,
                    masks_t=masks_t,
                    masks_tnext=masks_p,
                    mask_min_fg_alpha=args.mask_min_fg_alpha,
                    mask_min_fg_fraction=args.mask_min_fg_fraction,
                    capture_residual_t=residual_t,
                    capture_residual_tnext=residual_p,
                    temporal_frame_delta=-frame_delta,
                    temporal_cycle_thresh_px=args.temporal_cycle_thresh_px,
                    sync_motion_compensate=bool(args.sync_motion_compensate),
                    warps_override=bwd_warps,
                    motion_slack=float(args.motion_slack),
                    motion_slack_max_px=float(args.motion_slack_max_px),
                    temporal_min_certainty_fast=float(args.temporal_min_certainty_fast),
                    temporal_fast_flow_px=float(args.temporal_fast_flow_px),
                    rs_signs=rs_signs_arr,
                    rs_readout_frames=rs_readout_frames,
                    rs_axis=rs_axis_eff,
                    rs_reference=rs_reference_eff,
                )
                pos_tn, tracked_accel, tracked_has_accel = combine_bidirectional_tracks(
                    pos_t, pos_tn, src_idx, pos_t_b, pos_tp, src_idx_b, frame_delta,
                )
                print(
                    f"Temporal {frame_idx}->{back_frame} | tracked={back_stats.get('accepted', 0):,} "
                    f"centred={int(tracked_has_accel.sum()):,}/{len(pos_t):,}"
                )
                stats['temporal_tracking'].setdefault('backward_per_pair', []).append(
                    {'frame_t': int(frame_idx), 'frame_tprev': int(back_frame),
                     'centred': int(tracked_has_accel.sum()), **back_stats}
                )
                del paths_p, images_p, masks_p, cloud_p
                _log_mem(args.log_memory, f'after temporal {frame_idx}->{back_frame}')
            del fwd_warps, bwd_warps, back_loaded

            tracked_pair_arrays = (pos_t, pos_tn, cols, src_idx)

            if records is not None:
                records.append({
                    "cloud": cloud_t,
                    "X": pos_t,
                    "V": (pos_tn - pos_t) / float(frame_delta),   # world units per video frame
                    "src_idx": src_idx,
                })

            print(
                f"Temporal {frame_idx}->{next_frame} | tracked={ttl_stats.get('accepted', 0):,} "
                f"input={ttl_stats.get('points_input', 0):,} "
                f"rej_views={ttl_stats.get('rej_min_views', 0)} "
                f"rej_reproj={ttl_stats.get('rej_reprojection', 0)} "
                f"rej_mask={ttl_stats.get('rej_mask', 0)}"
            )

            stats['temporal_tracking']['per_pair'].append(
                {'frame_t': int(frame_idx), 'frame_tnext': int(next_frame), **ttl_stats}
            )

            _log_mem(args.log_memory, f'after temporal {frame_idx}->{next_frame}')

        else:
            tracked_pair_arrays = _empty_tracked_pair()
            tracked_accel = np.zeros((0, 3), dtype=np.float32)
            tracked_has_accel = np.zeros((0,), dtype=bool)

        if (not getattr(args, 'include_untracked_fresh', False)) and len(tracked_pair_arrays[0]) == 0:
            print(f'[warn] keyframe {frame_idx}: no tracked points; skipping.')
        else:
            _log_mem(args.log_memory, f'before assemble keyframe {frame_idx}')

            velocity_units = _append_assembled_frame(
                args=args,
                frames=frames,
                cameras=cameras,
                cloud=cloud_t,
                idx=idx,
                next_cloud=cloud_n,
                tracked_pair=tracked_pair_arrays,
                denom=denom,
                one_frame_duration=one_frame_duration,
                all_positions=frame_lists["positions"],
                all_colors=frame_lists["colors"],
                all_velocities=frame_lists["velocities"],
                all_times=frame_lists["times"],
                all_durations=frame_lists["durations"],
                all_has_velocity=frame_lists["has_velocity"],
                all_source_frames=frame_lists["source_frames"],
                all_is_static=frame_lists["is_static"],
                all_dynamic_protect=frame_lists["dynamic_protect"],
                all_dynamic_allowed=frame_lists["dynamic_allowed"],
                all_stage_keep=frame_lists["stage_keep"],
                stats=stats,
                sync_offsets_frames=sync_offsets_frames_arr,
            )
            row_positions = (
                frame_lists["positions"][0]
                if frame_lists["positions"]
                else np.zeros((0, 3), dtype=np.float32)
            )
            row_accel, row_has_accel = accelerations_for_rows(
                row_positions,
                tracked_pair_arrays[0],
                tracked_accel,
                tracked_has_accel,
            )
            if bool(getattr(args, 'trigs_velocity_per_frame', True)) is False:
                # Velocities are then per keyframe interval; keep the
                # acceleration in the same unit.
                row_accel = row_accel * float(frame_delta) ** 2
            frame_lists["accelerations"] = [row_accel]
            frame_lists["has_acceleration"] = [row_has_accel]
            frame_lists["track_flags"] = [track_flags_for_rows(
                row_positions, tracked_pair_arrays[0], tracked_has_accel,
            )]

            frame_arrays: Dict[str, np.ndarray] = {}
            for name, (dtype, tail_shape) in CACHE_ARRAY_SPECS.items():
                if frame_lists[name]:
                    frame_arrays[name] = frame_lists[name][0]
                else:
                    frame_arrays[name] = np.zeros((0, *tail_shape), dtype=dtype)
            delta = _stats_delta(stats, stats_before)
            write_frame_checkpoint(
                checkpoint_path,
                frame_idx=frame_idx,
                run_fingerprint=run_fingerprint,
                input_fingerprint=input_fingerprint,
                velocity_units=velocity_units,
                arrays=frame_arrays,
                stats_delta=delta,
            )
            checkpoint_bytes = checkpoint_path.stat().st_size
            stats["cache"]["recomputed"] += 1
            stats["cache"]["checkpoint_bytes"] += int(checkpoint_bytes)
            (assembled_subframes if is_subframe else assembled_keyframes).append(int(frame_idx))
            print(
                f"[cache] committed frame {frame_idx}: "
                f"{len(frame_arrays['positions']):,} rows, "
                f"{checkpoint_bytes / (1024 ** 2):.1f} MiB"
            )
            _log_mem(args.log_memory, f'after assemble keyframe {frame_idx}')

        del paths_t, images_t, masks_t, cloud_t
        del paths_n, images_n, masks_n, cloud_n
        del tracked_pair_arrays
        del frame_lists
        if "frame_arrays" in locals():
            del frame_arrays

        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

        _log_mem(args.log_memory, f'after free keyframe {frame_idx}')

    pbar.close()
    if not assembled_keyframes:
        raise RuntimeError('No keyframes were assembled; refusing to save an empty NPZ.')

    if args.solve_offsets_out:
        assert records is not None
        if not records:
            print("[solve_offsets] WARNING: no tracked records "
                "(temporal tracking disabled?); skipping offset solve.")
        else:
            solved = solve_camera_offsets(records, cameras, fps=args.fps)
            _atomic_write_json(Path(args.solve_offsets_out), solved)

    # Hard guard: this new pipeline must not have old prev_* streaming variables.
    for stale_name in ('prev_frame', 'prev_cloud', 'prev_paths', 'prev_images', 'prev_masks'):
        if stale_name in locals():
            raise RuntimeError(
                f'Stale old streaming variable {stale_name!r} exists. '
                'Delete the old prev_frame/prev_cloud final-assembly block.'
            )

    _log_mem(args.log_memory, 'before memory-mapped final assembly')

    logical_frame_start = int(args.trigs_logical_frame_start)
    logical_frame_end = logical_frame_start + int(args.frame_end - args.frame_start)
    keyframe_frames = np.asarray(assembled_keyframes, dtype=np.int32)

    checkpoint_metadata: List[Tuple[int, Path, str, Dict[str, object]]] = []
    total_rows = 0
    for frame_idx in assembled_keyframes:
        next_frame = (
            int(frame_idx)
            if args.disable_temporal_tracking
            else int(frame_idx + temporal_frame_delta)
        )
        input_fingerprint = frame_input_fingerprint(
            frame_idx=frame_idx,
            next_frame_idx=next_frame,
            images_root=images_root,
            masks_root=masks_root,
            cameras=cameras,
            image_pattern=args.image_pattern,
            mask_pattern=args.mask_pattern,
            fps=args.fps,
            offsets_sec=sync_offsets_sec,
            sync_enabled=sync_enabled,
            frame_start=input_frame_start,
            frame_end=input_frame_end,
        )
        path = active_cache / f"frame-{int(frame_idx):010d}.npz"
        info = load_frame_checkpoint(
            path,
            frame_idx=frame_idx,
            run_fingerprint=run_fingerprint,
            input_fingerprint=input_fingerprint,
            load_arrays=False,
            trust=bool(args.trust_cache),
        )
        if info is None:
            raise RuntimeError(
                f"Checkpoint became invalid before finalization: {path}"
            )
        checkpoint_metadata.append((int(frame_idx), path, input_fingerprint, info))
        total_rows += int(info["n_rows"])

    work_dir = active_cache / "finalize"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    raw_arrays = _create_memmap_arrays(work_dir, "raw", total_rows)
    offset = 0
    for frame_idx, path, input_fingerprint, info in checkpoint_metadata:
        loaded = load_frame_checkpoint(
            path,
            frame_idx=frame_idx,
            run_fingerprint=run_fingerprint,
            input_fingerprint=input_fingerprint,
            load_arrays=True,
            trust=bool(args.trust_cache),
        )
        if loaded is None:
            raise RuntimeError(f"Failed to load validated checkpoint: {path}")
        count = int(loaded["n_rows"])
        target = slice(offset, offset + count)
        for name in CACHE_ARRAY_SPECS:
            raw_arrays[name][target] = loaded["arrays"][name]
        offset += count
        del loaded
        gc.collect()
    _flush_memmaps(raw_arrays)
    _log_mem(args.log_memory, 'after memory-mapped final assembly')

    if args.rigid_cluster_motion:
        stats['rigid_cluster_motion'] = apply_rigid_cluster_motion(
            raw_arrays,
            frame_delta=temporal_frame_delta,
            velocity_per_frame=bool(getattr(args, 'trigs_velocity_per_frame', True)),
            speed_adaptive_duration=bool(args.speed_adaptive_duration),
            one_frame_duration=one_frame_duration,
            duration_min_scale=float(args.duration_min_scale),
            duration_max_scale=float(args.duration_max_scale),
            params=rigid_params_from_args(args),
        )
        _flush_memmaps(raw_arrays)
        _log_mem(args.log_memory, 'after rigid cluster motion')

    budget_arrays: Mapping[str, np.ndarray] = raw_arrays
    if args.motion_budget:
        # After the rigid pass, so classification uses the repaired motion.
        budget_arrays, stats['motion_budget'] = apply_motion_budget(
            raw_arrays,
            frame_delta=temporal_frame_delta,
            velocity_per_frame=bool(getattr(args, 'trigs_velocity_per_frame', True)),
            dedup_radius=float(args.dedup_radius),
            dedup_radius_max=float(args.dedup_radius_max),
            params=motion_budget_params_from_args(args),
        )
        _log_mem(args.log_memory, 'after motion budget')

    static_before = int(np.count_nonzero(budget_arrays["is_static"]))
    arrays: Mapping[str, np.ndarray] = budget_arrays
    if static_before > 0 and float(args.static_spatial_dedup_radius) > 0.0:
        arrays = _disk_backed_static_merge(
            budget_arrays,
            work_dir=work_dir,
            radius=float(args.static_spatial_dedup_radius),
            max_points=int(args.static_max_points),
            seed=int(args.seed),
            source_frame_start=int(args.frame_start),
        )
        static_after = int(np.count_nonzero(arrays["is_static"]))
        print(
            f"[static merge] static {static_before:,} -> {static_after:,}; "
            f"dynamic kept {len(arrays['positions']) - static_after:,}; "
            f"final {len(arrays['positions']):,}"
        )

    extra_npz: Dict[str, np.ndarray] = {}
    if sub_enabled:
        # Fast rows of each sub-keyframe frame (only those; the keyframe rows above went
        # through rigid repair, motion budget and static merge as usual), plus a per-row
        # local_stride: 1 for fast rows at keyframes and every sub-keyframe row,
        # --frame-step for everything else.
        arrays, local_stride, n_kf_fast, per_sub = append_subkeyframe_rows(
            arrays, assembled_keyframes,
            {int(f): _checkpoint_arrays(int(f), SUBKEYFRAME_TRACK_DELTA) for f in sorted(assembled_subframes)},
            frame_step=int(args.frame_step), fast_rows_fn=_fast_rows,
        )
        n_sub = int(sum(per_sub.values()))
        extra_npz = {
            'local_stride': local_stride,
            'subkeyframe_frames': np.asarray(sorted(int(f) for f in assembled_subframes), dtype=np.int32),
            'subkeyframe_min_speed_spacings': np.array(float(args.subkeyframe_min_speed), dtype=np.float32),
        }
        stats.setdefault('subkeyframes', {}).update({
            'keyframe_rows_local_stride_1': n_kf_fast,
            'subkeyframe_rows_added': per_sub,
            'subkeyframe_rows_total': n_sub,
        })
        print(
            f"[subkeyframes] keyframe rows marked fast (local_stride 1): {n_kf_fast:,}; "
            f"sub-keyframe rows added: {n_sub:,} over {len(assembled_subframes)} frames {per_sub}"
        )

    _log_mem(args.log_memory, 'before savez_compressed')

    n = len(arrays["positions"])
    for name, arr in arrays.items():
        if len(arr) != n:
            raise RuntimeError(
                f"Row alignment error before save: positions={n}, {name}={len(arr)}"
            )

    _atomic_savez(
        output_path,
        compressed=True,
        positions=arrays["positions"],
        colors=arrays["colors"],
        velocities=arrays["velocities"],
        accelerations=arrays["accelerations"],
        has_acceleration=arrays["has_acceleration"],
        track_flags=arrays["track_flags"],
        times=arrays["times"],
        durations=arrays["durations"],
        has_velocity=arrays["has_velocity"],
        is_static=arrays["is_static"],
        dynamic_protect=arrays["dynamic_protect"],
        dynamic_allowed=arrays["dynamic_allowed"],
        stage_keep=arrays["stage_keep"],
        frame_start=np.array(logical_frame_start, dtype=np.int32),
        frame_end=np.array(logical_frame_end, dtype=np.int32),
        n_keyframes=np.array(len(keyframe_frames), dtype=np.int32),
        keyframe_step=np.array(args.frame_step, dtype=np.int32),
        source_frame_start=np.array(args.frame_start, dtype=np.int32),
        source_frame_end=np.array(args.frame_end, dtype=np.int32),
        source_frames=arrays["source_frames"],
        keyframe_frames=keyframe_frames,
        fps=np.array(args.fps, dtype=np.float32),
        velocity_units=np.array(velocity_units),
        time_units=np.array('normalized_0_1_over_source_frame_range'),
        mode=np.array('trigs_roma_hybrid_static_dynamic'),
        **extra_npz,
    )

    _log_mem(args.log_memory, 'after savez_compressed')

    if args.stats_path:
        stats_path = Path(args.stats_path)
        _atomic_write_json(stats_path, stats)

    tracked_total = int(np.count_nonzero(arrays["has_velocity"]))
    static_total = int(np.count_nonzero(arrays["is_static"]))
    protected_total = int(np.count_nonzero(arrays["dynamic_protect"]))
    allowed_total = int(np.count_nonzero(arrays["dynamic_allowed"]))
    stage_keep_total = int(np.count_nonzero(arrays["stage_keep"]))

    print(f'Saved {n:,} TriGS NPZ points across {len(assembled_keyframes)} keyframes to {output_path}')
    print(
        f'Logical TriGS frame range: {logical_frame_start}..{logical_frame_end} '
        f'(source images {args.frame_start}..{args.frame_end})'
    )
    print(f'Velocity units: {velocity_units}')
    print(
        f'Has velocity: {tracked_total:,} / {n:,} '
        f'({100.0 * tracked_total / max(n, 1):.1f}%)'
    )
    print(
        f"Static rows: {static_total:,} / {n:,} "
        f"({100.0 * static_total / max(n, 1):.1f}%)"
    )
    print(
        f"Dynamic-protected rows: {protected_total:,} / {n:,} "
        f"({100.0 * protected_total / max(n, 1):.1f}%)"
    )
    print(
        f"Dynamic-allowed rows: {allowed_total:,} / {n:,} "
        f"({100.0 * allowed_total / max(n, 1):.1f}%)"
    )
    print(
        f"Stage-keep rows: {stage_keep_total:,} / {n:,} "
        f"({100.0 * stage_keep_total / max(n, 1):.1f}%)"
    )

    # Every requested artifact is durable at this point. Close all mappings
    # before either retaining the cache for reuse or removing it.
    _close_memmaps(arrays, raw_arrays)
    try:
        del arr
    except NameError:
        pass
    del arrays
    del raw_arrays
    del checkpoint_metadata
    gc.collect()
    # Notably on NFS, open files can briefly become .nfs* and block rmdir.
    _finish_completed_cache(
        active_cache, cache_root, keep_cache=bool(args.keep_cache)
    )
# ---------------------------------------------------------------------------
# Priority-voxel dedup that also returns kept source indices.
# ---------------------------------------------------------------------------


def _priority_voxel_to_target_with_indices(
    positions: np.ndarray,
    colors: np.ndarray,
    priorities: np.ndarray,
    target_points: int,
    min_radius: float,
    max_radius: float = 0.0,
    max_iters: int = 20,
    tolerance_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like deduplicate_to_target_with_priority but also returns kept source indices.

    Important:
      target_points <= 0 means NO deduplication and NO target cap.
    """
    n = len(positions)
    if n == 0:
        return positions, colors, priorities, np.zeros(0, dtype=np.int64)

    if target_points <= 0:
        return positions, colors, priorities, np.arange(n, dtype=np.int64)

    if n <= target_points:
        return positions, colors, priorities, np.arange(n, dtype=np.int64)

    def vox_with_idx(
        pos: np.ndarray,
        col: np.ndarray,
        pri: np.ndarray,
        v: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if v <= 0:
            return pos, col, pri, np.arange(len(pos), dtype=np.int64)

        coords = np.floor(pos / v).astype(np.int64)
        mins = coords.min(axis=0)
        coords -= mins
        maxs = coords.max(axis=0) + 1

        keys = (
            coords[:, 0] * (maxs[1] * maxs[2])
            + coords[:, 1] * maxs[2]
            + coords[:, 2]
        )

        order = np.lexsort((-pri, keys))
        keys_sorted = keys[order]
        _, first_idx = np.unique(keys_sorted, return_index=True)
        kept = order[first_idx]

        return pos[kept], col[kept], pri[kept], kept.astype(np.int64, copy=False)

    if max_radius <= 0.0:
        max_radius = auto_max_voxel_size(positions, target_points)
        print(f'  [dedup] auto max_radius = {max_radius:.5f} (bbox-derived)')

    pos0, col0, pri0, idx0 = vox_with_idx(positions, colors, priorities, min_radius)
    if len(pos0) <= target_points:
        return pos0, col0, pri0, idx0

    lo = min_radius
    hi = max(min_radius * 2.0, 1e-6)

    pos_hi, col_hi, pri_hi, idx_hi = vox_with_idx(positions, colors, priorities, hi)

    while len(pos_hi) > target_points and hi < max_radius:
        lo = hi
        hi = min(hi * 2.0, max_radius)
        pos_hi, col_hi, pri_hi, idx_hi = vox_with_idx(positions, colors, priorities, hi)

    if len(pos_hi) > target_points:
        print(
            f'  [dedup] WARNING: could not reach target {target_points} pts '
            f'with max voxel {max_radius:.5f}; returning {len(pos_hi)} pts.'
        )
        return pos_hi, col_hi, pri_hi, idx_hi

    best_pos, best_col, best_pri, best_idx = pos_hi, col_hi, pri_hi, idx_hi
    best_err = abs(len(pos_hi) - target_points)
    tol = max(1, int(target_points * tolerance_ratio))

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        pos_m, col_m, pri_m, idx_m = vox_with_idx(positions, colors, priorities, mid)
        err = abs(len(pos_m) - target_points)

        if err < best_err:
            best_pos, best_col, best_pri, best_idx = pos_m, col_m, pri_m, idx_m
            best_err = err

        if err <= tol:
            return pos_m, col_m, pri_m, idx_m

        if len(pos_m) > target_points:
            lo = mid
        else:
            hi = mid

    return best_pos, best_col, best_pri, best_idx


def _priority_voxel_to_target_with_detail_sphere(
    positions: np.ndarray,
    colors: np.ndarray,
    priorities: np.ndarray,
    *,
    target_points: int,
    min_radius: float,
    max_radius: float,
    center: np.ndarray,
    radius: float,
    target_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Downsample while reserving part of the budget for a detail sphere.

    The two regions are voxelized independently so a large floor/background
    cannot consume the detail quota. If either region has fewer candidates
    than its requested share, the unused quota is reassigned to the other.
    """
    n = len(positions)
    if n == 0 or target_points <= 0 or n <= target_points:
        idx = np.arange(n, dtype=np.int64)
        detail = np.linalg.norm(positions - center.reshape(1, 3), axis=1) <= radius
        info = {
            'input_total': int(n),
            'input_detail': int(detail.sum()),
            'detail_quota': int(detail.sum()),
            'kept_total': int(n),
            'kept_detail': int(detail.sum()),
        }
        return positions, colors, priorities, idx, info

    detail_mask = (
        np.linalg.norm(positions - center.reshape(1, 3), axis=1)
        <= float(radius)
    )
    detail_idx = np.flatnonzero(detail_mask).astype(np.int64)
    rest_idx = np.flatnonzero(~detail_mask).astype(np.int64)

    detail_quota = min(
        len(detail_idx),
        max(1, int(round(int(target_points) * float(target_fraction)))),
    )
    rest_quota = min(len(rest_idx), int(target_points) - detail_quota)
    remaining = int(target_points) - detail_quota - rest_quota
    if remaining > 0:
        add_detail = min(len(detail_idx) - detail_quota, remaining)
        detail_quota += add_detail
        remaining -= add_detail
    if remaining > 0:
        rest_quota += min(len(rest_idx) - rest_quota, remaining)

    def _select(source_idx: np.ndarray, quota: int) -> np.ndarray:
        if quota <= 0 or len(source_idx) == 0:
            return np.zeros((0,), dtype=np.int64)
        _, _, _, local_idx = _priority_voxel_to_target_with_indices(
            positions[source_idx],
            colors[source_idx],
            priorities[source_idx],
            target_points=int(quota),
            min_radius=float(min_radius),
            max_radius=float(max_radius),
        )
        return source_idx[local_idx]

    kept_detail_idx = _select(detail_idx, detail_quota)
    kept_rest_idx = _select(rest_idx, rest_quota)
    kept_idx = np.sort(
        np.concatenate([kept_detail_idx, kept_rest_idx]).astype(np.int64)
    )
    kept_detail_count = int(detail_mask[kept_idx].sum())
    info = {
        'input_total': int(n),
        'input_detail': int(len(detail_idx)),
        'detail_quota': int(detail_quota),
        'kept_total': int(len(kept_idx)),
        'kept_detail': kept_detail_count,
    }
    return (
        positions[kept_idx],
        colors[kept_idx],
        priorities[kept_idx],
        kept_idx,
        info,
    )

if __name__ == '__main__':
    main()
