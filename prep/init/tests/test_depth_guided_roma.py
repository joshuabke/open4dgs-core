from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "prep/init/build-depth-guided-roma-plan.py"
ROMA_PATH = ROOT / "prep/init/prep-roma-trigs-memory.py"


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLAN = _load("build_depth_guided_roma_plan", PLAN_PATH)
ROMA = _load("prep_roma_depth_guide", ROMA_PATH)


def _plane_depth(K: np.ndarray, R: np.ndarray, t: np.ndarray, height: int, width: int):
    center = -R.T @ t
    y, x = np.mgrid[:height, :width]
    pixel = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    rays = pixel @ np.linalg.inv(K).T @ R
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    distance = (5.0 - center[2]) / rays[:, 2]
    return distance.reshape(height, width).astype(np.float32)


def _fixture(tmp_path: Path):
    width, height = 64, 48
    K = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
    reference_rows = []
    manifest_rows = []
    depth_root = tmp_path / "depth"
    for index, center_x in enumerate((-0.5, 0.5)):
        folder = f"{index:04d}"
        R = np.eye(3)
        t = np.array([-center_x, 0.0, 0.0])
        reference_rows.append(
            {
                "name": folder,
                "folder": folder,
                "R": R.tolist(),
                "t": t.tolist(),
                "camera_model_name": "PINHOLE",
                "width": width,
                "height": height,
                "params": [80.0, 80.0, 32.0, 24.0],
            }
        )
        relative = Path(folder) / "frame-000000.npz"
        (depth_root / folder).mkdir(parents=True, exist_ok=True)
        np.savez(
            depth_root / relative,
            depth=_plane_depth(K, R, t, height, width),
            opacity=np.ones((height, width), dtype=np.float32),
            K=K,
            R=R,
            T=t,
        )
        manifest_rows.append(
            {
                "camera": folder,
                "camera_index": index,
                "frame_index": 0,
                "path": relative.as_posix(),
                "height": height,
                "width": width,
            }
        )
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"cameras": reference_rows}))
    (depth_root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "l3mhet-raw-depth-v1",
                "depth_semantics": "distance_along_normalized_world_ray",
                "use_z_depth": False,
                "entries": manifest_rows,
            }
        )
    )
    # Seed only the left image half, leaving a known depth-valid hole on the right.
    source_uv = np.array(
        [[x, y] for y in range(4, height, 8) for x in range(4, width // 2, 8)],
        dtype=np.float64,
    )
    camera = PLAN.GuideCamera(
        0,
        "0000",
        "0000",
        width,
        height,
        K,
        np.eye(3),
        np.array([0.5, 0.0, 0.0]),
        _plane_depth(K, np.eye(3), np.array([0.5, 0.0, 0.0]), height, width),
        np.ones((height, width), dtype=np.float32),
    )
    depth, _ = PLAN._sample(camera.depth, source_uv)
    positions = camera.backproject(source_uv, depth).astype(np.float32)
    npz = tmp_path / "points.npz"
    np.savez(npz, positions=positions)
    return reference, depth_root, npz


def test_depth_plan_selects_crops_and_pairs_but_contains_no_depth_points(
    tmp_path: Path,
):
    reference, depth_root, npz = _fixture(tmp_path)
    args = PLAN.parse_args(
        [
            "--reference-json",
            str(reference),
            "--depth-root",
            str(depth_root),
            "--npz-path",
            str(npz),
            "--target-frame",
            "2",
            "--output-plan",
            str(tmp_path / "plan.json"),
            "--coverage-cell-px",
            "8",
            "--tile-size-px",
            "32",
            "--tile-stride-px",
            "24",
            "--min-hole-cells",
            "2",
            "--max-hole-distance-px",
            "128",
            "--min-pair-depth-support",
            "2",
            "--min-pair-depth-support-ratio",
            "0.1",
            "--min-crop-size-px",
            "16",
            "--max-crop-size-px",
            "48",
        ]
    )
    plan = PLAN.build_plan(args)

    assert plan["purpose"] == "crop_and_pair_selection_only"
    assert plan["target_frame"] == 2
    assert plan["entries"]
    assert plan["summary"]["total_hole_cells"] > 0
    assert all("points" not in row and "depth" not in row for row in plan["entries"])
    assert {row["cam_a"] for row in plan["entries"]} <= {0, 1}
    assert {row["cam_b"] for row in plan["entries"]} <= {0, 1}


def test_plan_loader_rejects_reference_hash_mismatch(tmp_path: Path):
    reference, _depth_root, _npz = _fixture(tmp_path)
    cameras, _ = ROMA.load_reference_export(reference)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "trigs-depth-guided-roma-plan-v1",
                "reference_sha256": "wrong",
                "target_frame": 2,
                "entries": [],
            }
        )
    )

    import pytest

    with pytest.raises(ValueError, match="different reference"):
        ROMA.load_depth_guide_plan(plan, cameras, reference)


def test_depth_guide_crop_matches_are_offset_then_use_normal_geometry_gates(
    tmp_path: Path, monkeypatch
):
    calls = []

    def fake_pair(_model, _pa, _pb, image_a, image_b, *_args, **_kwargs):
        calls.append((image_a.shape, image_b.shape))
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
        return a, b, np.ones(2, dtype=np.float32)

    monkeypatch.setattr(ROMA, "run_roma_pair", fake_pair)
    monkeypatch.setattr(
        ROMA,
        "pair_ransac_with_motion_slack",
        lambda a, *_args, **_kwargs: (np.ones(len(a), dtype=bool), 0),
    )
    monkeypatch.setattr(
        ROMA,
        "triangulate_multiview_dlt_batch",
        lambda _cam, _uv, counts, _cams: (
            np.tile(np.asarray([0.0, 0.0, 5.0], dtype=np.float64), (len(counts), 1)),
            np.ones(len(counts), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        ROMA,
        "within_scene_bounds_batch",
        lambda x, *_: np.ones(len(x), dtype=bool),
    )
    monkeypatch.setattr(
        ROMA,
        "compute_reprojection_errors_batch",
        lambda _x, cam, _uv, _counts, _cams: (
            np.zeros(len(cam), dtype=np.float64),
            np.ones(len(cam), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        ROMA,
        "compute_triangulation_angle_deg_batch",
        lambda _x, _cam, counts, _cams: np.full(len(counts), 20.0),
    )

    cameras = [
        ROMA.Camera(
            name=f"camera {i}",
            folder=f"{i:04d}",
            R=np.eye(3),
            t=np.array([-float(i), 0.0, 0.0]),
            camera_model_name="PINHOLE",
            width=64,
            height=48,
            params=np.array([80.0, 80.0, 32.0, 24.0]),
        )
        for i in range(2)
    ]
    images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in cameras]
    result = ROMA.reconstruct_frame_cloud(
        frame_idx=2,
        images_root=tmp_path,
        cameras=cameras,
        image_pattern="{frame}.png",
        model=object(),
        device="cpu",
        pair_list=[],
        roma_num_matches=10,
        roma_min_certainty=0.0,
        pair_ransac_px=1.0,
        pair_calibrated_max_error_px=0.0,
        min_pair_inliers=1,
        same_camera_merge_px=2.0,
        min_track_views=2,
        reproj_thresh_px=1.0,
        min_triangulation_angle_deg=1.0,
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
        paths_override=[Path("a"), Path("b")],
        depth_guide_entries=[
            {
                "cam_a": 0,
                "cam_b": 1,
                "crop_a": [10, 20, 30, 40],
                "crop_b": [12, 18, 32, 38],
            }
        ],
        depth_guide_num_matches=100,
        match_debug_dir=tmp_path / "debug",
        spatial_audit_camera=0,
        spatial_audit_roi=(10.0, 20.0, 20.0, 30.0),
    )

    assert calls == [((20, 20, 3), (20, 20, 3))]
    assert result.debug["depth_guide_pairs_succeeded"] == 1
    assert result.debug["depth_guide_matches_before_geom"] == 2
    assert result.debug["spatial_audit"]["pair_observations_in_roi"]["raw"] == 2
    assert len(result.positions) == 2
