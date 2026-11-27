"""The batched geometry must match the per-point functions it replaces."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_batch_geometry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RTOL = 1e-12
ATOL = 1e-12


def _look_at(center, target):
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)  # world -> camera
    return R, -R @ center


def _rig(model, n=12, seed=0):
    rng = np.random.default_rng(seed)
    cameras = []
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        center = np.array([4.0 * np.cos(angle), 0.3 * (i % 3), 4.0 * np.sin(angle)])
        R, t = _look_at(center, rng.normal(0.0, 0.1, 3))
        if model == "PINHOLE":
            params = np.array([900.0, 905.0, 960.0, 540.0])
        elif model == "OPENCV":
            params = np.array([
                900.0, 905.0, 960.0, 540.0,
                -0.08 + 0.01 * i, 0.02, 1e-3, -5e-4,
            ])
        elif model == "OPENCV_FISHEYE":
            params = np.array([600.0, 600.0, 960.0, 540.0, 0.05, -0.01, 0.002, 0.0])
        else:
            raise ValueError(model)
        cameras.append(MODULE.Camera(
            name=f"cam{i}", folder=f"{i:04d}", R=R, t=t,
            camera_model_name=model, width=1920, height=1080, params=params,
        ))
    return cameras


def _observation_sets(cameras, n_points=600, seed=1):
    """Noisy projections of random points into random camera subsets, plus
    hand-made edge cases (single view, duplicate rays, points behind or
    through a camera centre)."""
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n_points):
        X = rng.normal(0.0, 1.0, 3)
        n_views = int(rng.integers(1, len(cameras) + 1))
        cams = rng.choice(len(cameras), size=n_views, replace=False)
        obs = []
        for c in cams:
            uv = cameras[c].project_world_point(X)
            if uv is None:
                uv = rng.uniform([0, 0], [1920, 1080])
            obs.append(MODULE.Observation(
                cam_idx=int(c), uv=uv + rng.normal(0.0, 1.5, 2),
            ))
        sets.append(obs)
    # Two observations of the same ray: degenerate DLT system.
    uv = cameras[0].project_world_point(np.zeros(3))
    sets.append([
        MODULE.Observation(cam_idx=0, uv=uv.copy()),
        MODULE.Observation(cam_idx=0, uv=uv.copy()),
    ])
    # Nearly parallel rays: a far point seen by two neighbouring cameras.
    far = cameras[0].center_world + 1e4 * cameras[0].R[2]
    sets.append([
        MODULE.Observation(cam_idx=c, uv=cameras[c].project_world_point(far))
        for c in (0, 1)
        if cameras[c].project_world_point(far) is not None
    ])
    # Arbitrary pixels: triangulates somewhere, often behind some cameras.
    for _ in range(50):
        cams = rng.choice(len(cameras), size=int(rng.integers(2, 5)), replace=False)
        sets.append([
            MODULE.Observation(cam_idx=int(c), uv=rng.uniform([-500, -500], [2400, 1600]))
            for c in cams
        ])
    return sets


def _flatten(sets):
    counts = np.array([len(s) for s in sets], dtype=np.int64)
    cam = np.array([o.cam_idx for s in sets for o in s], dtype=np.int64)
    uv = np.array([o.uv for s in sets for o in s], dtype=np.float64).reshape(-1, 2)
    return cam, uv, counts


def _offsets(counts):
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


@pytest.mark.parametrize("model", ["PINHOLE", "OPENCV", "OPENCV_FISHEYE"])
def test_batch_geometry_matches_per_point_functions(model):
    cameras = _rig(model)
    sets = _observation_sets(cameras)
    cam, uv, counts = _flatten(sets)

    X_batch, ok = MODULE.triangulate_multiview_dlt_batch(cam, uv, counts, cameras)
    X_ref = [MODULE.triangulate_multiview_dlt(s, cameras) for s in sets]
    assert ok.tolist() == [x is not None for x in X_ref]
    assert not ok[np.asarray(counts) < 2].any()
    for i, x in enumerate(X_ref):
        if x is not None:
            np.testing.assert_allclose(X_batch[i], x, rtol=RTOL, atol=ATOL)
        else:
            assert np.isnan(X_batch[i]).all()

    good = np.flatnonzero(ok)
    rows = MODULE._ragged_take(counts, good)
    errors, depths = MODULE.compute_reprojection_errors_batch(
        X_batch[good], cam[rows], uv[rows], counts[good], cameras
    )
    angles = MODULE.compute_triangulation_angle_deg_batch(
        X_batch[good], cam[rows], counts[good], cameras
    )
    offsets = _offsets(counts[good])
    n_behind = n_inf = 0
    for j, i in enumerate(good):
        err_ref, depth_ref = MODULE.compute_reprojection_errors(X_ref[i], sets[i], cameras)
        sl = slice(offsets[j], offsets[j] + counts[i])
        np.testing.assert_allclose(depths[sl], depth_ref, rtol=RTOL, atol=ATOL)
        assert (np.isinf(errors[sl]) == np.isinf(err_ref)).all()
        finite = np.isfinite(err_ref)
        np.testing.assert_allclose(errors[sl][finite], err_ref[finite], rtol=RTOL, atol=ATOL)
        ang_ref = MODULE.compute_triangulation_angle_deg(X_ref[i], sets[i], cameras)
        assert angles[j] == pytest.approx(ang_ref, rel=RTOL, abs=1e-10)
        n_behind += int(np.any(depth_ref <= 0))
        n_inf += int(np.any(np.isinf(err_ref)))

    # The fixture must exercise the rejection paths, not only clean points.
    assert n_behind > 0 and n_inf > 0
    assert (~ok).sum() > 0


def test_ragged_reductions_match_numpy_on_each_row():
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 20, size=300)
    values = rng.normal(size=int(counts.sum())) * 10.0 ** rng.integers(-3, 4, size=int(counts.sum()))
    values[::17] = np.inf
    offsets = _offsets(counts)
    mean = MODULE._ragged_mean(values, counts)
    vmax = MODULE._ragged_max(values, counts)
    anyneg = MODULE._ragged_any(values < 0, counts)
    for i, (o, n) in enumerate(zip(offsets, counts)):
        row = values[o:o + n]
        # Bit-identical: the reproduction gate compares the mean to a threshold.
        assert mean[i] == np.mean(row) or (np.isnan(mean[i]) and np.isnan(np.mean(row)))
        assert vmax[i] == np.max(row)
        assert anyneg[i] == bool(np.any(row < 0))

    take = np.array([5, 0, 299, 17, 17], dtype=np.int64)
    rows = MODULE._ragged_take(counts, take)
    expected = np.concatenate([np.arange(offsets[i], offsets[i] + counts[i]) for i in take])
    assert rows.tolist() == expected.tolist()
    assert MODULE._ragged_take(counts, np.zeros(0, dtype=np.int64)).size == 0


def test_batch_handles_empty_input():
    cameras = _rig("PINHOLE", n=2)
    empty_i = np.zeros((0,), dtype=np.int64)
    X, ok = MODULE.triangulate_multiview_dlt_batch(empty_i, np.zeros((0, 2)), empty_i, cameras)
    assert X.shape == (0, 3) and ok.shape == (0,)
    errors, depths = MODULE.compute_reprojection_errors_batch(
        np.zeros((0, 3)), empty_i, np.zeros((0, 2)), empty_i, cameras
    )
    assert errors.shape == (0,) and depths.shape == (0,)
    assert MODULE.compute_triangulation_angle_deg_batch(
        np.zeros((0, 3)), empty_i, empty_i, cameras
    ).shape == (0,)
    assert MODULE.within_scene_bounds_batch(np.zeros((0, 3)), None, 1.0).shape == (0,)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_observation_fails_like_the_scalar_path(bad):
    cameras = _rig("OPENCV", n=3)
    good = [
        MODULE.Observation(cam_idx=c, uv=cameras[c].project_world_point(np.zeros(3)))
        for c in range(3)
    ]
    broken = [
        MODULE.Observation(cam_idx=0, uv=np.array([bad, 10.0])),
        good[1],
        good[2],
    ]

    def outcome(fn):
        try:
            return ("ok", fn())
        except Exception as exc:  # the scalar path lets LAPACK errors escape
            return ("raise", type(exc))

    scalar = outcome(lambda: MODULE.triangulate_multiview_dlt(broken, cameras))
    cam, uv, counts = _flatten([good, broken])
    batch = outcome(lambda: MODULE.triangulate_multiview_dlt_batch(cam, uv, counts, cameras))
    assert scalar[0] == batch[0]
    if scalar[0] == "raise":
        assert scalar[1] == batch[1]
    else:
        _, batch_ok = batch[1]
        assert (scalar[1] is not None) == bool(batch_ok[1])


def test_scene_bounds_batch_matches_scalar():
    rng = np.random.default_rng(5)
    X = rng.normal(0.0, 3.0, size=(500, 3))
    for meta in (
        None,
        {"scene_center": [0.5, -0.2, 0.1], "scene_radius_p95": 2.0},
        {"scene_center": [0.0, 0.0, 0.0], "scene_radius_max": 4.0},
        {"scene_radius_p95": 0.0},
    ):
        expected = [MODULE.within_scene_bounds(x, meta, 1.5) for x in X]
        assert MODULE.within_scene_bounds_batch(X, meta, 1.5).tolist() == expected


@pytest.mark.skipif(
    not os.environ.get("OPEN4DGS_REAL_OBS"),
    reason="set OPEN4DGS_REAL_OBS to an observation dump from a real prep run",
)
def test_batch_geometry_matches_on_recorded_observations():
    """Observation sets recorded from a real prep run (see the PR)."""
    import json

    data = np.load(os.environ["OPEN4DGS_REAL_OBS"], allow_pickle=False)
    reference = Path(str(data["reference_json"]))
    cameras, _ = MODULE.load_reference_export(reference)
    cam, uv, counts = data["obs_cam"], data["obs_uv"], data["counts"]
    offsets = _offsets(counts)
    sets = [
        [MODULE.Observation(cam_idx=int(cam[k]), uv=uv[k]) for k in range(o, o + n)]
        for o, n in zip(offsets, counts)
    ]
    X_batch, ok = MODULE.triangulate_multiview_dlt_batch(cam, uv, counts, cameras)
    good = np.flatnonzero(ok)
    rows = MODULE._ragged_take(counts, good)
    errors, depths = MODULE.compute_reprojection_errors_batch(
        X_batch[good], cam[rows], uv[rows], counts[good], cameras
    )
    angles = MODULE.compute_triangulation_angle_deg_batch(
        X_batch[good], cam[rows], counts[good], cameras
    )
    sub_offsets = _offsets(counts[good])
    j = 0
    for i, s in enumerate(sets):
        X_ref = MODULE.triangulate_multiview_dlt(s, cameras)
        assert ok[i] == (X_ref is not None)
        if X_ref is None:
            continue
        np.testing.assert_allclose(X_batch[i], X_ref, rtol=RTOL, atol=ATOL)
        err_ref, depth_ref = MODULE.compute_reprojection_errors(X_ref, s, cameras)
        sl = slice(sub_offsets[j], sub_offsets[j] + counts[i])
        np.testing.assert_allclose(errors[sl], err_ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(depths[sl], depth_ref, rtol=RTOL, atol=ATOL)
        assert angles[j] == pytest.approx(
            MODULE.compute_triangulation_angle_deg(X_ref, s, cameras), rel=RTOL, abs=1e-10
        )
        j += 1
    print(json.dumps({"points": int(len(counts)), "triangulated": int(ok.sum())}))


def test_average_observations_single_view_shortcut_matches_mean():
    rng = np.random.default_rng(9)

    def reference(component):
        grouped = {}
        for obs in component:
            grouped.setdefault(obs.cam_idx, []).append(obs)
        out = []
        for cam_idx, obs_list in grouped.items():
            uv = np.mean(np.stack([o.uv for o in obs_list], axis=0), axis=0)
            canon = None
            if any(o.uv_canon is not None for o in obs_list):
                canon = np.mean(np.stack([MODULE.observation_geom_uv(o) for o in obs_list]), axis=0)
            out.append((cam_idx, uv, canon, float(np.mean([o.flow_px for o in obs_list]))))
        return out

    for _ in range(300):
        component = []
        for _ in range(int(rng.integers(1, 8))):
            uv = rng.uniform(0, 2000, 2)
            component.append(MODULE.Observation(
                cam_idx=int(rng.integers(0, 5)),
                uv=uv if rng.random() < 0.8 else uv.astype(np.float32),
                uv_canon=None if rng.random() < 0.4 else uv + rng.normal(0, 2, 2),
                flow_px=float(rng.uniform(0, 20)),
            ))
        got = MODULE.average_observations_per_camera(component)
        want = reference(component)
        assert len(got) == len(want)
        for g, (cam_idx, uv, canon, flow) in zip(got, want):
            assert g.cam_idx == cam_idx
            assert g.uv.dtype == uv.dtype and np.array_equal(g.uv, uv)
            assert (g.uv_canon is None) == (canon is None)
            if canon is not None:
                assert g.uv_canon.dtype == canon.dtype and np.array_equal(g.uv_canon, canon)
            assert g.flow_px == flow
            assert all(g.uv is not o.uv for o in component)
