#!/usr/bin/env python3
"""Backend adapters for RoMa v1 and RoMa v2, plus an optional MEMFOF temporal backend.

The adapter exposes one API to the point generator:
  * sampled cross-view correspondences
  * dense temporal matching
  * arbitrary forward/backward dense-warp queries

RoMa v1 uses the ``romatch`` package. RoMa v2 uses ``romav2``. MEMFOF (``memfof``)
only serves the temporal tracks (``--temporal-backend memfof``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Hashable, Optional, Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DenseMatch:
    backend: str
    payload: Any
    h_a: int
    w_a: int
    h_b: int
    w_b: int


class RomaBackend(Protocol):
    name: str
    supports_reverse: bool

    def match_points(
        self,
        path_a: Path,
        path_b: Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        num: int,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
        mask_min_fg_alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def build_dense(
        self,
        path_a: Path,
        path_b: Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
        mask_min_fg_alpha: float = 0.5,
    ) -> DenseMatch: ...

    def sample_at_uvs(
        self,
        dense: DenseMatch,
        uvs: np.ndarray,
        *,
        reverse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def sample_points_from_dense(
        self, dense: DenseMatch, *, num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def _mask_values(mask: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    if len(uvs) == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = mask.shape[:2]
    map_x = np.clip(uvs[:, 0], 0.0, w - 1.0).astype(np.float32).reshape(-1, 1)
    map_y = np.clip(uvs[:, 1], 0.0, h - 1.0).astype(np.float32).reshape(-1, 1)
    vals = cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return vals.reshape(-1).astype(np.float32)


def _filter_points_by_masks(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    cert: np.ndarray,
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones((len(pts_a),), dtype=bool)
    if mask_a is not None:
        keep &= _mask_values(mask_a, pts_a) >= float(threshold)
    if mask_b is not None:
        keep &= _mask_values(mask_b, pts_b) >= float(threshold)
    return pts_a[keep], pts_b[keep], cert[keep]


def _strip_batch(t: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while t.dim() > target_ndim and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def _v1_warp_channels_last(warp: torch.Tensor) -> torch.Tensor:
    w = _strip_batch(warp, 3)
    if w.dim() != 3:
        raise RuntimeError(f"Unexpected RoMa v1 warp rank: {tuple(warp.shape)}")
    axes = [i for i, size in enumerate(w.shape) if size == 4]
    if not axes:
        raise RuntimeError(f"RoMa v1 warp has no size-4 channel axis: {tuple(w.shape)}")
    axis = axes[0]
    if axis == 2:
        return w.contiguous()
    if axis == 0:
        return w.permute(1, 2, 0).contiguous()
    return w.permute(0, 2, 1).contiguous()


def _v1_certainty_2d(certainty: torch.Tensor) -> torch.Tensor:
    c = _strip_batch(certainty, 2)
    while c.dim() > 2:
        singleton = next((i for i, size in enumerate(c.shape) if size == 1), None)
        if singleton is None:
            break
        c = c.squeeze(singleton)
    if c.dim() != 2:
        raise RuntimeError(f"Unexpected RoMa v1 certainty rank: {tuple(certainty.shape)}")
    return c.contiguous()


def _v1_apply_masks(
    certainty: torch.Tensor,
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    threshold: float,
) -> torch.Tensor:
    c = _v1_certainty_2d(certainty)
    h, width2 = c.shape
    if width2 % 2:
        raise RuntimeError(f"RoMa v1 certainty width must be even, got {width2}")
    w = width2 // 2
    floor = 1e-4

    def factor(mask: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).to(device=c.device, dtype=c.dtype)
        return torch.where(t >= threshold, t, torch.full_like(t, floor))

    left = c[:, :w]
    right = c[:, w:]
    if mask_a is not None:
        left = left * factor(mask_a)
    if mask_b is not None:
        right = right * factor(mask_b)
    weighted = torch.cat([left, right], dim=1)
    if certainty.dim() == 3 and certainty.shape[0] == 1:
        weighted = weighted.unsqueeze(0)
    return weighted


class RoMaV1Backend:
    def __init__(
        self,
        *,
        device: str,
        model: str = "outdoor",
        symmetric: bool = True,
        upsample_preds: bool = True,
        sample_mode: str = "threshold",
    ) -> None:
        import romatch

        factories = {
            "indoor": romatch.roma_indoor,
            "outdoor": romatch.roma_outdoor,
        }
        if hasattr(romatch, "tiny_roma_v1_outdoor"):
            factories["tiny-outdoor"] = romatch.tiny_roma_v1_outdoor
        if model not in factories:
            raise ValueError(
                f"Unsupported RoMa v1 model {model!r}; choose one of {sorted(factories)}"
            )
        self.device = device
        self.model_kind = model
        self._model = factories[model](device=device)
        if hasattr(self._model, "symmetric"):
            self._model.symmetric = bool(symmetric)
        if hasattr(self._model, "upsample_preds"):
            self._model.upsample_preds = bool(upsample_preds)
        if hasattr(self._model, "sample_mode"):
            self._model.sample_mode = sample_mode
        self.name = f"roma-v1-{model}"
        self.supports_reverse = bool(getattr(self._model, "symmetric", symmetric))

    def build_dense(
        self,
        path_a: Path,
        path_b: Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
        mask_min_fg_alpha: float = 0.5,
    ) -> DenseMatch:
        with torch.inference_mode():
            warp, certainty = self._model.match(
                str(path_a), str(path_b), device=self.device
            )
            if mask_a is not None or mask_b is not None:
                certainty = _v1_apply_masks(
                    certainty, mask_a, mask_b, float(mask_min_fg_alpha)
                )
        return DenseMatch(
            backend=self.name,
            payload=(warp, certainty),
            h_a=int(image_a.shape[0]),
            w_a=int(image_a.shape[1]),
            h_b=int(image_b.shape[0]),
            w_b=int(image_b.shape[1]),
        )

    def match_points(
        self,
        path_a: Path,
        path_b: Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        num: int,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
        mask_min_fg_alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dense = self.build_dense(
            path_a,
            path_b,
            image_a,
            image_b,
            mask_a=mask_a,
            mask_b=mask_b,
            mask_min_fg_alpha=mask_min_fg_alpha,
        )
        return self.sample_points_from_dense(dense, num=num)

    def sample_points_from_dense(
        self, dense: DenseMatch, *, num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        warp, certainty = dense.payload
        with torch.inference_mode():
            matches, cert = self._model.sample(warp, certainty, num=num)
            pts_a, pts_b = self._model.to_pixel_coordinates(
                matches, dense.h_a, dense.w_a, dense.h_b, dense.w_b
            )
        return (
            pts_a.detach().cpu().numpy().astype(np.float32),
            pts_b.detach().cpu().numpy().astype(np.float32),
            cert.detach().cpu().numpy().reshape(-1).astype(np.float32),
        )

    def sample_at_uvs(
        self,
        dense: DenseMatch,
        uvs: np.ndarray,
        *,
        reverse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(uvs) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        if reverse and not self.supports_reverse:
            raise RuntimeError("RoMa v1 reverse sampling requires symmetric=True")

        warp, certainty = dense.payload
        w_cl = _v1_warp_channels_last(warp).float()
        c_2d = _v1_certainty_2d(certainty).float()
        h_w, width2, _ = w_cl.shape
        if width2 % 2:
            raise RuntimeError(f"RoMa v1 dense warp width must be even, got {width2}")
        w_w = width2 // 2

        if reverse:
            field = w_cl[:, w_w:, :2]
            cert_map = c_2d[:, w_w:]
            h_src, w_src = dense.h_b, dense.w_b
            h_dst, w_dst = dense.h_a, dense.w_a
        else:
            field = w_cl[:, :w_w, -2:]
            cert_map = c_2d[:, :w_w]
            h_src, w_src = dense.h_a, dense.w_a
            h_dst, w_dst = dense.h_b, dense.w_b

        field = field.permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        cert_map = cert_map.unsqueeze(0).unsqueeze(0).contiguous().to(self.device)
        uv_t = torch.from_numpy(np.asarray(uvs, np.float32)).to(self.device)
        grid = torch.stack(
            [uv_t[:, 0] * (2.0 / w_src) - 1.0, uv_t[:, 1] * (2.0 / h_src) - 1.0],
            dim=-1,
        ).view(1, len(uv_t), 1, 2)
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0, :, :, 0].T
        sampled_cert = F.grid_sample(
            cert_map, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(-1)
        out = torch.stack(
            [
                sampled[:, 0] * (w_dst / 2.0) + (w_dst / 2.0),
                sampled[:, 1] * (h_dst / 2.0) + (h_dst / 2.0),
            ],
            dim=-1,
        )
        return (
            out.detach().cpu().numpy().astype(np.float32),
            sampled_cert.detach().cpu().numpy().astype(np.float32),
        )


class RoMaV2Backend:
    def __init__(
        self,
        *,
        device: str,
        setting: str = "precise",
        mask_oversample: float = 2.0,
        spatial_grid: int = 0,
        spatial_fraction: float = 0.5,
    ) -> None:
        try:
            from romav2 import RoMaV2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoMa v2 requires the `romav2` package. Install it separately "
                "and review the DINOv3 license before use."
            ) from exc

        torch.set_float32_matmul_precision("highest")
        self.device = device
        self._model = RoMaV2()
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        self._model.apply_setting(setting)
        self.setting = setting
        self.mask_oversample = max(float(mask_oversample), 1.0)
        self.spatial_grid = max(int(spatial_grid), 0)
        self.spatial_fraction = float(np.clip(spatial_fraction, 0.0, 1.0))
        self.name = f"roma-v2-{setting}"
        self.supports_reverse = bool(getattr(self._model, "bidirectional", False))

    @staticmethod
    def _directional_spatial_samples(
        warp: torch.Tensor,
        overlap: torch.Tensor,
        *,
        num: int,
        grid: int,
        h_src: int,
        w_src: int,
        h_dst: int,
        w_dst: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample deterministic, evenly spaced correspondences per grid cell.

        RoMa v2's native sampler is deliberately confidence-weighted over the
        whole image. That is ideal for a global inlier set, but it can starve a
        small subject part near an image edge when a large textured floor or
        background is present. The spatial anchors here depend only on source
        image coordinates, not pair confidence. Consequently the same source
        pixel is queried in every pair containing that camera and pairwise
        matches can merge into 3+-view tracks. Choosing each pair's local
        confidence maxima instead produces different pixels per pair and a
        large collection of disconnected two-view components.
        """
        if num <= 0 or grid <= 0:
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_pts, empty_pts, np.zeros((0,), dtype=np.float32)

        field = warp
        while field.dim() > 3 and field.shape[0] == 1:
            field = field.squeeze(0)
        cert_map = overlap
        while cert_map.dim() > 2 and cert_map.shape[0] == 1:
            cert_map = cert_map.squeeze(0)
        if cert_map.dim() == 3 and cert_map.shape[-1] == 1:
            cert_map = cert_map[..., 0]
        if field.dim() != 3 or field.shape[-1] != 2:
            raise RuntimeError(
                f"Unexpected RoMa v2 warp shape for spatial sampling: {tuple(warp.shape)}"
            )
        if cert_map.dim() != 2 or tuple(cert_map.shape) != tuple(field.shape[:2]):
            raise RuntimeError(
                "Unexpected RoMa v2 overlap shape for spatial sampling: "
                f"warp={tuple(warp.shape)} overlap={tuple(overlap.shape)}"
            )

        h_map, w_map = int(field.shape[0]), int(field.shape[1])
        row_edges = np.linspace(0, h_map, grid + 1, dtype=np.int64)
        col_edges = np.linspace(0, w_map, grid + 1, dtype=np.int64)
        cells = [
            (int(row_edges[r]), int(row_edges[r + 1]),
             int(col_edges[c]), int(col_edges[c + 1]))
            for r in range(grid)
            for c in range(grid)
            if row_edges[r + 1] > row_edges[r] and col_edges[c + 1] > col_edges[c]
        ]
        if not cells:
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_pts, empty_pts, np.zeros((0,), dtype=np.float32)

        base, remainder = divmod(int(num), len(cells))
        # Spread remainder slots across the image deterministically. This must
        # not depend on pair confidence or source anchors would differ by pair.
        bonus = np.zeros((len(cells),), dtype=np.int64)
        if remainder > 0:
            bonus_idx = np.floor(
                (np.arange(remainder, dtype=np.float64) + 0.5)
                * (len(cells) / float(remainder))
            ).astype(np.int64)
            bonus[np.clip(bonus_idx, 0, len(cells) - 1)] = 1

        src_parts: list[torch.Tensor] = []
        dst_parts: list[torch.Tensor] = []
        cert_parts: list[torch.Tensor] = []
        for cell_idx, (r0, r1, c0, c1) in enumerate(cells):
            quota = int(base + bonus[cell_idx])
            if quota <= 0:
                continue
            cell_h, cell_w = r1 - r0, c1 - c0
            take = min(quota, int(cell_h * cell_w))
            if take <= 0:
                continue
            # A near-square subgrid avoids the stripes produced by uniform
            # sampling of row-major flattened indices. Dropping the final few
            # Cartesian points when the quota is not factorable remains
            # deterministic across every camera pair.
            n_rows = max(
                1,
                int(np.ceil(np.sqrt(take * (cell_h / max(cell_w, 1))))),
            )
            n_cols = max(1, int(np.ceil(take / float(n_rows))))
            ys_1d = torch.floor(
                (torch.arange(n_rows, device=field.device, dtype=torch.float32) + 0.5)
                * (float(cell_h) / float(n_rows))
            ).to(torch.int64) + r0
            xs_1d = torch.floor(
                (torch.arange(n_cols, device=field.device, dtype=torch.float32) + 0.5)
                * (float(cell_w) / float(n_cols))
            ).to(torch.int64) + c0
            ys, xs = torch.meshgrid(ys_1d, xs_1d, indexing="ij")
            ys = ys.reshape(-1)[:take]
            xs = xs.reshape(-1)[:take]
            dst_norm = field[ys, xs]
            values = cert_map[ys, xs]
            src_uv = torch.stack(
                [
                    (xs.to(torch.float32) + 0.5) * (float(w_src) / float(w_map)),
                    (ys.to(torch.float32) + 0.5) * (float(h_src) / float(h_map)),
                ],
                dim=-1,
            )
            dst_uv = torch.stack(
                [
                    dst_norm[:, 0].to(torch.float32) * (float(w_dst) / 2.0)
                    + (float(w_dst) / 2.0),
                    dst_norm[:, 1].to(torch.float32) * (float(h_dst) / 2.0)
                    + (float(h_dst) / 2.0),
                ],
                dim=-1,
            )
            src_parts.append(src_uv)
            dst_parts.append(dst_uv)
            cert_parts.append(values.to(torch.float32))

        if not src_parts:
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_pts, empty_pts, np.zeros((0,), dtype=np.float32)
        src = torch.cat(src_parts, dim=0)
        dst = torch.cat(dst_parts, dim=0)
        cert = torch.cat(cert_parts, dim=0)
        finite = torch.isfinite(src).all(dim=1) & torch.isfinite(dst).all(dim=1)
        finite &= torch.isfinite(cert)
        return (
            src[finite].detach().cpu().numpy().astype(np.float32),
            dst[finite].detach().cpu().numpy().astype(np.float32),
            cert[finite].detach().cpu().numpy().astype(np.float32),
        )

    def _sample_points_spatially(
        self, dense: DenseMatch, *, num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = dense.payload
        directions: list[tuple[str, int]] = []
        if preds.get("warp_AB") is not None and preds.get("overlap_AB") is not None:
            directions.append(("AB", (int(num) + 1) // 2))
        if preds.get("warp_BA") is not None and preds.get("overlap_BA") is not None:
            directions.append(("BA", int(num) // 2 if directions else int(num)))
        if not directions:
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_pts, empty_pts, np.zeros((0,), dtype=np.float32)

        a_parts: list[np.ndarray] = []
        b_parts: list[np.ndarray] = []
        cert_parts: list[np.ndarray] = []
        for direction, quota in directions:
            if direction == "AB":
                src, dst, cert = self._directional_spatial_samples(
                    preds["warp_AB"], preds["overlap_AB"],
                    num=quota, grid=self.spatial_grid,
                    h_src=dense.h_a, w_src=dense.w_a,
                    h_dst=dense.h_b, w_dst=dense.w_b,
                )
                pts_a, pts_b = src, dst
            else:
                src, dst, cert = self._directional_spatial_samples(
                    preds["warp_BA"], preds["overlap_BA"],
                    num=quota, grid=self.spatial_grid,
                    h_src=dense.h_b, w_src=dense.w_b,
                    h_dst=dense.h_a, w_dst=dense.w_a,
                )
                pts_a, pts_b = dst, src
            a_parts.append(pts_a)
            b_parts.append(pts_b)
            cert_parts.append(cert)
        return (
            np.concatenate(a_parts, axis=0),
            np.concatenate(b_parts, axis=0),
            np.concatenate(cert_parts, axis=0),
        )

    def _apply_source_masks_to_preds(
        self,
        preds: dict[str, Any],
        mask_a: Optional[np.ndarray],
        mask_b: Optional[np.ndarray],
        threshold: float,
    ) -> dict[str, Any]:
        # Clone only the confidence maps we alter. Endpoint masks are also
        # applied after sampling, so both sides must pass the requested mask.
        out = dict(preds)
        floor = 1e-4

        def weighted(overlap: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
            h, w = int(overlap.shape[1]), int(overlap.shape[2])
            resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            factor = torch.from_numpy(resized).to(overlap.device, overlap.dtype)
            factor = torch.where(
                factor >= threshold, factor, torch.full_like(factor, floor)
            )
            return overlap * factor[None, :, :, None]

        if mask_a is not None and out.get("overlap_AB") is not None:
            out["overlap_AB"] = weighted(out["overlap_AB"], mask_a)
        if mask_b is not None and out.get("overlap_BA") is not None:
            out["overlap_BA"] = weighted(out["overlap_BA"], mask_b)
        return out

    def build_dense(
        self,
        path_a: Path,
        path_b: Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
        mask_min_fg_alpha: float = 0.5,
    ) -> DenseMatch:
        with torch.inference_mode():
            preds = self._model.match(str(path_a), str(path_b))
        if mask_a is not None or mask_b is not None:
            preds = self._apply_source_masks_to_preds(
                preds, mask_a, mask_b, float(mask_min_fg_alpha)
            )
        return DenseMatch(
            backend=self.name,
            payload=preds,
            h_a=int(image_a.shape[0]),
            w_a=int(image_a.shape[1]),
            h_b=int(image_b.shape[0]),
            w_b=int(image_b.shape[1]),
        )

    def match_points(
        self,
        path_a: Path,
        path_b: Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        num: int,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None,
        mask_min_fg_alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dense = self.build_dense(
            path_a,
            path_b,
            image_a,
            image_b,
            mask_a=mask_a,
            mask_b=mask_b,
            mask_min_fg_alpha=mask_min_fg_alpha,
        )
        spatial_num = (
            int(round(int(num) * self.spatial_fraction))
            if self.spatial_grid > 0
            else 0
        )
        spatial_num = min(max(spatial_num, 0), int(num))
        global_num = int(num) - spatial_num

        pools: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if global_num > 0:
            global_request = (
                int(np.ceil(global_num * self.mask_oversample))
                if (mask_a is not None or mask_b is not None)
                else global_num
            )
            pools.append(self.sample_points_from_dense(dense, num=global_request))
        if spatial_num > 0:
            pools.append(self._sample_points_spatially(dense, num=spatial_num))

        if not pools:
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_pts, empty_pts, np.zeros((0,), dtype=np.float32)
        filtered = [
            _filter_points_by_masks(
                pts_a, pts_b, cert, mask_a, mask_b, float(mask_min_fg_alpha)
            )
            for pts_a, pts_b, cert in pools
        ]
        pts_a = np.concatenate([item[0] for item in filtered], axis=0)
        pts_b = np.concatenate([item[1] for item in filtered], axis=0)
        cert = np.concatenate([item[2] for item in filtered], axis=0)
        if len(cert) > num:
            # Oversampling is only used by the global pool when masks are
            # active. Trim its excess without deleting the reserved spatial
            # rows merely because their certainty is lower.
            spatial_kept = len(filtered[-1][2]) if spatial_num > 0 else 0
            global_limit = max(int(num) - spatial_kept, 0)
            global_len = len(filtered[0][2]) if global_num > 0 else 0
            global_order = np.argsort(-cert[:global_len], kind="stable")[:global_limit]
            spatial_idx = np.arange(global_len, len(cert), dtype=np.int64)
            keep_idx = np.concatenate([global_order, spatial_idx], axis=0)[: int(num)]
            pts_a, pts_b, cert = pts_a[keep_idx], pts_b[keep_idx], cert[keep_idx]
        return pts_a, pts_b, cert

    def sample_points_from_dense(
        self, dense: DenseMatch, *, num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            sampled = self._model.sample(dense.payload, int(num))
        if not isinstance(sampled, (tuple, list)) or len(sampled) < 2:
            raise RuntimeError("Unexpected RoMa v2 sample() output")
        matches, cert = sampled[0], sampled[1]
        pts_a, pts_b = self._model.to_pixel_coordinates(
            matches, dense.h_a, dense.w_a, dense.h_b, dense.w_b
        )
        return (
            pts_a.detach().cpu().numpy().astype(np.float32),
            pts_b.detach().cpu().numpy().astype(np.float32),
            cert.detach().cpu().numpy().reshape(-1).astype(np.float32),
        )

    def sample_at_uvs(
        self,
        dense: DenseMatch,
        uvs: np.ndarray,
        *,
        reverse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(uvs) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        preds = dense.payload
        if reverse:
            warp = preds.get("warp_BA")
            overlap = preds.get("overlap_BA")
            h_src, w_src = dense.h_b, dense.w_b
            h_dst, w_dst = dense.h_a, dense.w_a
        else:
            warp = preds.get("warp_AB")
            overlap = preds.get("overlap_AB")
            h_src, w_src = dense.h_a, dense.w_a
            h_dst, w_dst = dense.h_b, dense.w_b
        if warp is None or overlap is None:
            raise RuntimeError(
                f"{self.name} setting does not provide {'reverse' if reverse else 'forward'} dense predictions"
            )

        # RoMa v2 warp is [1,H,W,2] in normalized destination coordinates.
        field = warp.permute(0, 3, 1, 2).float()
        cert_map = overlap.permute(0, 3, 1, 2).float()
        uv_t = torch.from_numpy(np.asarray(uvs, np.float32)).to(field.device)
        grid = torch.stack(
            [uv_t[:, 0] * (2.0 / w_src) - 1.0, uv_t[:, 1] * (2.0 / h_src) - 1.0],
            dim=-1,
        ).view(1, len(uv_t), 1, 2)
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0, :, :, 0].T
        sampled_cert = F.grid_sample(
            cert_map, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(-1)
        out = torch.stack(
            [
                sampled[:, 0] * (w_dst / 2.0) + (w_dst / 2.0),
                sampled[:, 1] * (h_dst / 2.0) + (h_dst / 2.0),
            ],
            dim=-1,
        )
        return (
            out.detach().cpu().numpy().astype(np.float32),
            sampled_cert.detach().cpu().numpy().astype(np.float32),
        )


MEMFOF_DEFAULT_CHECKPOINT = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
MEMFOF_DEFAULT_BATCH_SIZE = 4   # triplets per inference; ~?? GB each at 1080p


COMPOSED_HOP_INVALID = -1.0e6   # destination for a composed track whose hop left the image


def sample_flow_field(
    flow: np.ndarray, scale: float, uvs: np.ndarray
) -> np.ndarray:
    """Bilinear sample of a reduced-resolution flow field at full-resolution pixels.

    ``flow`` is [2, h, w] (x, y) in reduced-resolution pixels, computed on the
    image resized by ``scale`` (full -> reduced). Returns the full-resolution
    displacement for each uv. Pixel centres map as (u + 0.5) * scale - 0.5.
    """
    if len(uvs) == 0:
        return np.zeros((0, 2), np.float32)
    fl = np.asarray(flow, np.float32)
    h, w = fl.shape[1:]
    u = (np.asarray(uvs, np.float64)[:, 0] + 0.5) * scale - 0.5
    v = (np.asarray(uvs, np.float64)[:, 1] + 0.5) * scale - 0.5
    u = np.clip(u, 0.0, w - 1.0 - 1e-6)
    v = np.clip(v, 0.0, h - 1.0 - 1e-6)
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    du, dv = (u - u0), (v - v0)
    out = np.empty((len(u), 2), np.float32)
    for c in range(2):
        f = fl[c]
        out[:, c] = (
            f[v0, u0] * (1 - du) * (1 - dv) + f[v0, u0 + 1] * du * (1 - dv)
            + f[v0 + 1, u0] * (1 - du) * dv + f[v0 + 1, u0 + 1] * du * dv
        )
    return out / float(scale)


class MemfofTemporalBackend:
    """Temporal tracks from MEMFOF multi-frame optical flow (BSD-3 code and weights).

    MEMFOF takes three consecutive frames (t-d, t, t+d) and returns the backward
    (t -> t-d) and forward (t -> t+d) flow of the middle frame, so one inference
    per camera serves both directions of the centred-velocity tracker. It only
    replaces the dense temporal warp: the tracker's triangulation, reprojection,
    mask and two-view-core gates are unchanged. MEMFOF reports no confidence, so
    every in-image sample gets certainty 1 and the dense-warp cycle check is not
    available (``supports_reverse`` is False).

    Flows are computed on images resized so the long side is ``max_side`` (MEMFOF
    is trained for 1080p) and kept at that resolution in float16; sampling
    rescales displacements back to full-resolution pixels.
    """

    supports_reverse = False

    def __init__(
        self,
        *,
        device: str,
        checkpoint: str = MEMFOF_DEFAULT_CHECKPOINT,
        max_side: int = 1920,
        batch_size: int = MEMFOF_DEFAULT_BATCH_SIZE,
    ):
        try:
            from memfof import MEMFOF
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "--temporal-backend memfof requires the `memfof` package "
                "(pip install git+https://github.com/msu-video-group/memfof)."
            ) from exc
        self.device = device
        self.checkpoint = str(checkpoint)
        self.max_side = int(max_side)
        self.batch_size = max(int(batch_size), 1)
        self._model = MEMFOF.from_pretrained(self.checkpoint).eval().to(device)
        self.name = f"memfof[{self.checkpoint.rsplit('/', 1)[-1]}]"

    def _triplet_input(
        self, image_prev: np.ndarray, image_t: np.ndarray, image_next: np.ndarray
    ) -> tuple[np.ndarray, int, int, float]:
        """One MEMFOF input [3, hp, wp, 3] float32 (resized, padded to a multiple of 32 by
        edge replication), the reduced size (h, w) and the full->reduced scale."""
        H, W = image_t.shape[:2]
        scale = min(1.0, float(self.max_side) / float(max(H, W)))
        w, h = int(round(W * scale)), int(round(H * scale))
        hp, wp = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        stack = np.zeros((3, hp, wp, 3), np.float32)
        for i, im in enumerate((image_prev, image_t, image_next)):
            small = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if scale < 1.0 else im
            stack[i, :h, :w] = small
            stack[i, h:, :w] = small[-1:]
            stack[i, :, w:] = stack[i, :, w - 1:w]
        return stack, h, w, scale

    def _infer(self, stacks: np.ndarray, h: int, w: int) -> np.ndarray:
        """Flows [B, 2 (bwd, fwd), 2, h, w] float16 for B stacked inputs [B, 3, hp, wp, 3].

        Runs in sub-batches of ``self.batch_size``. On CUDA out-of-memory the batch size is
        halved (and stays halved for later calls) and the sub-batch is retried; only an OOM
        at batch size 1 is raised. The float32 flow is rounded to float16 on the device,
        which is the same round-to-nearest-even as numpy's astype on the host."""
        out = np.empty((len(stacks), 2, 2, h, w), np.float16)
        start = 0
        while start < len(stacks):
            bs = min(self.batch_size, len(stacks) - start)
            oom = False
            try:
                inp = torch.from_numpy(stacks[start:start + bs]).permute(0, 1, 4, 2, 3).to(self.device)
                with torch.inference_mode():
                    flow = self._model(inp)["flow"][-1]
                    out[start:start + bs] = flow[:, :, :, :h, :w].half().cpu().numpy()
                del inp, flow
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                oom = True
            if oom:
                # Outside the except block, so the failed forward's tensors are released.
                self.batch_size = max(bs // 2, 1)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [memfof] CUDA out of memory at batch {bs}; continuing with batch {self.batch_size}")
                continue
            start += bs
        return out

    def triplet_flows(
        self, image_prev: np.ndarray, image_t: np.ndarray, image_next: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Backward and forward flow of the middle frame ([2,h,w] float16) and the
        full->reduced scale."""
        images = (image_prev, image_t, image_next)
        return self.triplet_flows_batch([(0, 1, 2)], images.__getitem__)[0]

    def triplet_flows_batch(
        self,
        triplets: Sequence[tuple[Hashable, Hashable, Hashable]],
        fetch: Callable[[Hashable], np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """``triplet_flows`` for many triplets, in input order, batched on the GPU.

        Each triplet names its (t-d, t, t+d) images by key; ``fetch(key)`` returns the image.
        A worker thread fetches the images and builds the inputs of the next batch while the
        GPU runs the current one; each image is fetched once and dropped after its last use.
        Consecutive triplets of the same image size share a batch, so the flows of a triplet
        do not depend on its neighbours beyond floating-point rounding of the batched kernels.
        """
        triplets = [tuple(t) for t in triplets]
        n = len(triplets)
        if n == 0:
            return []
        last_use: dict = {}
        for i, keys in enumerate(triplets):
            for key in keys:
                last_use[key] = i
        held: dict = {}

        def build(start: int):
            # A chunk is the run of consecutive same-size triplets from ``start``, at most
            # the current batch size long.
            limit = max(self.batch_size, 1)
            stacks, scales, size = [], [], None
            i = start
            while i < n and len(stacks) < limit:
                keys = triplets[i]
                for k in keys:
                    if k not in held:
                        held[k] = fetch(k)
                ims = [held[k] for k in keys]
                stack, h, w, scale = self._triplet_input(*ims)
                if size is not None and (h, w, stack.shape) != size:
                    break
                size = (h, w, stack.shape)
                # Release an image once the last triplet using it has joined a chunk.
                for k in keys:
                    if last_use[k] <= i:
                        held.pop(k, None)
                stacks.append(stack)
                scales.append(scale)
                i += 1
            return start, i, np.stack(stacks, axis=0), size[0], size[1], scales

        results: list = [None] * n
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(build, 0)
            while pending is not None:
                start, stop, stacks, h, w, scales = pending.result()
                pending = pool.submit(build, stop) if stop < n else None
                flows = self._infer(stacks, h, w)
                for j in range(stop - start):
                    # Own arrays, so dropping one direction frees it.
                    results[start + j] = (flows[j, 0].copy(), flows[j, 1].copy(), scales[j])
                del flows, stacks
        return results

    @staticmethod
    def make_dense(flow: np.ndarray, scale: float, h: int, w: int) -> DenseMatch:
        return DenseMatch(
            backend="memfof", payload={"flow": flow, "scale": float(scale)},
            h_a=int(h), w_a=int(w), h_b=int(h), w_b=int(w),
        )

    @staticmethod
    def make_composed_dense(flow: Optional[np.ndarray], scale: float, h: int, w: int, hops) -> DenseMatch:
        """A t -> t+-d warp made of d one-frame hop flows [(flow, scale), ...] in order.
        Sampling it composes the hops (see sample_at_uvs), and so does layout validation;
        ``flow`` (the direct t -> t+-d flow) is optional and never sampled."""
        dense = MemfofTemporalBackend.make_dense(flow, scale, h, w)
        dense.payload["hops"] = [(f, float(s)) for f, s in hops]
        return dense

    def build_dense(self, *args, **kwargs) -> DenseMatch:
        raise NotImplementedError(
            "MEMFOF needs three frames; build warps with triplet_flows() and pass "
            "them to the tracker as warps_override."
        )

    def sample_at_uvs(
        self, dense: DenseMatch, uvs: np.ndarray, *, reverse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if reverse:
            raise RuntimeError("MEMFOF warps are one-directional; no reverse sampling")
        uvs = np.asarray(uvs, np.float32)
        hops = dense.payload.get("hops")
        if hops is None:
            disp = sample_flow_field(dense.payload["flow"], dense.payload["scale"], uvs)
            return (uvs + disp).astype(np.float32), np.ones((len(uvs),), np.float32)
        # Composed one-frame hops: each hop's flow is sampled at the point's current (moved)
        # pixel. A hop that starts outside the image is never clamped: the point gets a
        # destination far outside the image, which the tracker rejects like any other.
        cur = uvs.copy()
        alive = np.ones((len(cur),), bool)
        w, h = int(dense.w_b), int(dense.h_b)
        for flow, scale in hops:
            alive &= (cur[:, 0] >= 0) & (cur[:, 0] <= w - 1) & (cur[:, 1] >= 0) & (cur[:, 1] <= h - 1)
            if alive.any():
                c = cur[alive]
                cur[alive] = (c + sample_flow_field(flow, scale, c)).astype(np.float32)
        cur[~alive] = COMPOSED_HOP_INVALID
        return cur, np.ones((len(uvs),), np.float32)

    def sample_points_from_dense(
        self, dense: DenseMatch, *, num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        pts_a = np.stack(
            [rng.uniform(0, dense.w_a - 1, int(num)), rng.uniform(0, dense.h_a - 1, int(num))], 1
        ).astype(np.float32)
        pts_b, cert = self.sample_at_uvs(dense, pts_a)
        return pts_a, pts_b, cert


def create_roma_backend(
    *,
    backend: str,
    device: str,
    model: str = "outdoor",
    v2_setting: str = "precise",
    symmetric: bool = True,
    upsample_preds: bool = True,
    sample_mode: str = "threshold",
    v2_mask_oversample: float = 2.0,
    v2_spatial_grid: int = 0,
    v2_spatial_fraction: float = 0.5,
) -> RomaBackend:
    backend = backend.lower()
    if backend == "v1":
        return RoMaV1Backend(
            device=device,
            model=model,
            symmetric=symmetric,
            upsample_preds=upsample_preds,
            sample_mode=sample_mode,
        )
    if backend == "v2":
        return RoMaV2Backend(
            device=device,
            setting=v2_setting,
            mask_oversample=v2_mask_oversample,
            spatial_grid=v2_spatial_grid,
            spatial_fraction=v2_spatial_fraction,
        )
    raise ValueError(f"Unsupported RoMa backend {backend!r}; choose 'v1' or 'v2'")
