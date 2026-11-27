import gc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_warp_memory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_temporal_tracking_holds_at_most_one_dense_warp(monkeypatch):
    class Payload:
        live = 0
        peak = 0

        def __init__(self):
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

        def __del__(self):
            type(self).live -= 1

    def build(*args, **kwargs):
        return MODULE.DenseMatch(
            backend="fake",
            payload=Payload(),
            h_a=100,
            w_a=100,
            h_b=100,
            w_b=100,
        )

    def sample(_model, _dense, uvs, reverse=False):
        delta = -1.0 if reverse else 1.0
        return (
            np.asarray(uvs, dtype=np.float32) + delta,
            np.ones(len(uvs), dtype=np.float32),
        )

    monkeypatch.setattr(MODULE, "build_roma_warp", build)
    monkeypatch.setattr(MODULE, "sample_warp_at_uvs", sample)
    monkeypatch.setattr(MODULE, "validate_warp_layout", lambda *_: True)
    monkeypatch.setattr(
        MODULE,
        "triangulate_multiview_dlt_batch",
        lambda _cam, _uv, counts, _cams: (
            np.tile(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), (len(counts), 1)),
            np.ones(len(counts), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "within_scene_bounds_batch",
        lambda x, *_: np.ones(len(x), dtype=bool),
    )
    monkeypatch.setattr(
        MODULE,
        "compute_reprojection_errors_batch",
        lambda _x, cam, _uv, _counts, _cams: (
            np.zeros(len(cam), dtype=np.float64),
            np.ones(len(cam), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "compute_triangulation_angle_deg_batch",
        lambda _x, _cam, counts, _cams: np.full(len(counts), 90.0),
    )

    cameras = [
        SimpleNamespace(width=100, height=100),
        SimpleNamespace(width=100, height=100),
        SimpleNamespace(width=100, height=100),
    ]
    observations = [
        [
            (cam_idx, np.asarray([10.0 + point_idx, 20.0], dtype=np.float32))
            for cam_idx in range(3)
        ]
        for point_idx in range(4)
    ]
    cloud = MODULE.FramePointCloud(
        frame_idx=1,
        positions=np.ones((4, 3), dtype=np.float32),
        colors=np.ones((4, 3), dtype=np.float32),
        observations=observations,
    )
    model = SimpleNamespace(supports_reverse=True)

    result = MODULE.track_points_to_next_keyframe(
        cloud_t=cloud,
        cameras=cameras,
        model=model,
        device="cpu",
        paths_t=[Path("a")] * 3,
        images_t=[np.zeros((100, 100, 3), dtype=np.uint8)] * 3,
        paths_tnext=[Path("b")] * 3,
        images_tnext=[np.zeros((100, 100, 3), dtype=np.uint8)] * 3,
        temporal_min_certainty=0.0,
        temporal_reproj_thresh_px=1.0,
        min_tracked_views=2,
        min_triangulation_angle_deg=0.0,
        filter_two_view_by_core=False,
        two_view_support_radius=0.0,
        two_view_support_radius_factor=2.5,
        two_view_support_knn=2,
        scene_meta=None,
        scene_radius_mult=1.0,
        masks_t=None,
        masks_tnext=None,
        mask_min_fg_alpha=0.5,
        mask_min_fg_fraction=0.5,
        temporal_cycle_thresh_px=1.0,
    )

    gc.collect()
    assert len(result[0]) == 4
    assert Payload.peak == 1
    assert Payload.live == 0


def test_fresh_compensation_holds_at_most_one_dense_warp(monkeypatch):
    class Payload:
        live = 0
        peak = 0

        def __init__(self):
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

        def __del__(self):
            type(self).live -= 1

    def build(*args, **kwargs):
        return MODULE.DenseMatch(
            backend="fake",
            payload=Payload(),
            h_a=100,
            w_a=100,
            h_b=100,
            w_b=100,
        )

    def pair(*args, **kwargs):
        a = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        b = a + np.asarray([1.0, 0.0], dtype=np.float32)
        return a, b, np.ones(2, dtype=np.float32)

    monkeypatch.setattr(MODULE, "build_roma_warp", build)
    monkeypatch.setattr(MODULE, "run_roma_pair", pair)
    monkeypatch.setattr(MODULE, "validate_warp_layout", lambda *_: True)
    monkeypatch.setattr(
        MODULE,
        "sample_flow_per_frame",
        lambda _model, _dense, uvs, _delta: np.ones_like(uvs),
    )
    monkeypatch.setattr(
        MODULE,
        "pair_ransac_with_motion_slack",
        lambda a, *_args, **_kwargs: (np.ones(len(a), dtype=bool), 0),
    )
    monkeypatch.setattr(
        MODULE,
        "triangulate_multiview_dlt_batch",
        lambda _cam, _uv, counts, _cams: (
            np.tile(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), (len(counts), 1)),
            np.ones(len(counts), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "within_scene_bounds_batch",
        lambda x, *_: np.ones(len(x), dtype=bool),
    )
    monkeypatch.setattr(
        MODULE,
        "compute_reprojection_errors_batch",
        lambda _x, cam, _uv, _counts, _cams: (
            np.zeros(len(cam), dtype=np.float64),
            np.ones(len(cam), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "compute_triangulation_angle_deg_batch",
        lambda _x, _cam, counts, _cams: np.full(len(counts), 90.0),
    )

    cameras = [
        SimpleNamespace(width=100, height=100),
        SimpleNamespace(width=100, height=100),
        SimpleNamespace(width=100, height=100),
    ]
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in cameras]
    result = MODULE.reconstruct_frame_cloud(
        frame_idx=1,
        images_root=Path("."),
        cameras=cameras,
        image_pattern="{frame}.png",
        model=SimpleNamespace(),
        device="cpu",
        pair_list=[(0, 1), (1, 2), (0, 2)],
        roma_num_matches=10,
        roma_min_certainty=0.0,
        pair_ransac_px=1.0,
        pair_calibrated_max_error_px=0.0,
        min_pair_inliers=1,
        same_camera_merge_px=2.0,
        min_track_views=2,
        reproj_thresh_px=1.0,
        min_triangulation_angle_deg=0.0,
        scene_meta=None,
        scene_radius_mult=1.0,
        filter_two_view_by_core=False,
        two_view_support_radius=0.0,
        two_view_support_radius_factor=2.5,
        two_view_support_knn=2,
        masks_root=None,
        mask_pattern="{frame}.png",
        mask_min_fg_alpha=0.5,
        mask_min_fg_fraction=0.5,
        images_override=images,
        paths_override=[Path("a"), Path("b"), Path("c")],
        images_next_override=images,
        paths_next_override=[Path("d"), Path("e"), Path("f")],
        temporal_frame_delta=1,
        motion_slack=0.3,
    )

    gc.collect()
    assert len(result.positions) > 0
    assert Payload.peak == 1
    assert Payload.live == 0
