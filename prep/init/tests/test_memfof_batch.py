import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
import roma_backends as rb  # noqa: E402

SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_memfof_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeMemfof(torch.nn.Module):
    """Per-sample stand-in for MEMFOF: bwd = (t-d) - t, fwd = (t+d) - t on two channels,
    so any mix-up of samples, frames or order changes the output."""

    def __init__(self, oom_above=None):
        super().__init__()
        self.oom_above = oom_above
        self.batches = []

    def forward(self, images):
        B, _, _, H, W = images.shape
        self.batches.append((B, H, W))
        if self.oom_above is not None and B > self.oom_above:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        prev, cur, nxt = images[:, 0, :2], images[:, 1, :2], images[:, 2, :2]
        flow = torch.stack([(prev - cur) / 7.0, (nxt - cur) / 7.0], dim=1)   # [B, 2, 2, H, W]
        return {"flow": [flow]}


def _backend(batch_size, max_side=1920, oom_above=None):
    be = object.__new__(rb.MemfofTemporalBackend)
    be.device = "cpu"
    be.max_side = max_side
    be.batch_size = batch_size
    be._model = FakeMemfof(oom_above)
    be.name = "memfof[fake]"
    return be


def _images(n_cams, n_frames, h=40, w=70, seed=0):
    rng = np.random.default_rng(seed)
    return {
        (c, f): rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        for c in range(n_cams) for f in range(n_frames)
    }


def _triplets(n_cams, centres):
    return [((c, f - 1), (c, f), (c, f + 1)) for c in range(n_cams) for f in centres]


def _assert_same(a, b):
    assert len(a) == len(b)
    for (bwd_a, fwd_a, s_a), (bwd_b, fwd_b, s_b) in zip(a, b):
        assert bwd_a.dtype == np.float16 and fwd_a.dtype == np.float16
        np.testing.assert_array_equal(bwd_a, bwd_b)
        np.testing.assert_array_equal(fwd_a, fwd_b)
        assert s_a == s_b


def _one_by_one(be, images, triplets):
    return [be.triplet_flows(images[p], images[t], images[n]) for p, t, n in triplets]


@pytest.mark.parametrize("batch_size", [1, 3, 4, 64])
def test_batched_flows_equal_single_triplets_in_input_order(batch_size):
    images = _images(3, 7)
    triplets = _triplets(3, range(1, 6))
    ref = _one_by_one(_backend(1), images, triplets)
    be = _backend(batch_size)
    out = be.triplet_flows_batch(triplets, images.__getitem__)
    _assert_same(out, ref)
    assert max(b for b, _, _ in be._model.batches) == min(batch_size, len(triplets))


def test_single_camera_single_triplet():
    images = _images(1, 3)
    be = _backend(4)
    out = be.triplet_flows_batch([((0, 0), (0, 1), (0, 2))], images.__getitem__)
    _assert_same(out, _one_by_one(_backend(1), images, [((0, 0), (0, 1), (0, 2))]))
    assert be.triplet_flows_batch([], images.__getitem__) == []


def test_resized_flows_equal_single_triplets():
    images = _images(2, 5, h=50, w=90)
    triplets = _triplets(2, range(1, 4))
    ref = _one_by_one(_backend(1, max_side=64), images, triplets)
    out = _backend(4, max_side=64).triplet_flows_batch(triplets, images.__getitem__)
    _assert_same(out, ref)
    assert ref[0][0].shape == (2, 36, 64) and ref[0][2] == pytest.approx(64 / 90)


def test_unequal_image_sizes_never_share_a_batch():
    small = _images(1, 4, h=40, w=70, seed=1)
    large = {(1, f): im for (_, f), im in _images(1, 4, h=64, w=96, seed=2).items()}
    images = {**small, **large}
    triplets = [((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1), (1, 2)), ((0, 1), (0, 2), (0, 3)),
                ((1, 1), (1, 2), (1, 3))]
    be = _backend(8)
    out = be.triplet_flows_batch(triplets, images.__getitem__)
    _assert_same(out, _one_by_one(_backend(1), images, triplets))
    # 40x70 pads to 64x96 as well, so size runs are split by the unpadded size.
    assert [b for b, _, _ in be._model.batches] == [1, 1, 1, 1]
    assert out[0][0].shape == (2, 40, 70) and out[1][0].shape == (2, 64, 96)


def test_out_of_memory_halves_the_batch_and_keeps_the_result():
    images = _images(2, 7)
    triplets = _triplets(2, range(1, 6))
    be = _backend(8, oom_above=2)
    out = be.triplet_flows_batch(triplets, images.__getitem__)
    _assert_same(out, _one_by_one(_backend(1), images, triplets))
    assert be.batch_size == 2
    assert all(b <= 2 for b, _, _ in be._model.batches[2:])   # tried 8, then 4, then 2s


def test_out_of_memory_at_batch_one_is_raised():
    images = _images(1, 3)
    be = _backend(4, oom_above=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        be.triplet_flows_batch([((0, 0), (0, 1), (0, 2))], images.__getitem__)


def test_each_image_is_fetched_once():
    images = _images(2, 8)
    calls = []

    def fetch(key):
        calls.append(key)
        return images[key]

    # The first triplet repeats an image, as at the sequence start where t stands in for t-1.
    triplets = [((0, 0), (0, 0), (0, 1))] + _triplets(2, range(1, 7))
    _backend(3).triplet_flows_batch(triplets, fetch)
    assert sorted(calls) == sorted(set(calls)) and len(calls) == 16


def test_float16_rounding_on_torch_matches_numpy():
    x = np.random.default_rng(3).normal(scale=40.0, size=100_000).astype(np.float32)
    x[:5] = [1e-6, -3e-8, 65519.0, 0.1, 1 / 3]
    np.testing.assert_array_equal(torch.from_numpy(x).half().numpy(), x.astype(np.float16))


# --- composed hop warps in the prep script -------------------------------------------------

def _camera_frames(lo, hi, offsets):
    def frames(g):
        return np.clip(g + np.asarray(offsets), lo, hi).astype(np.int64)
    return frames


def _old_composed(be, images, frame_idx, d, backward, camera_frames, n_cams):
    """The per-camera loop the prep script ran before batching."""
    centres_f = list(range(frame_idx, frame_idx + d))
    centres_b = list(range(frame_idx, frame_idx - d, -1)) if backward else []
    fwd, bwd = [], []
    for cam in range(n_cams):
        def img(g):
            return images[(cam, int(camera_frames(g)[cam]))]
        hop = {c: be.triplet_flows(img(c - 1), img(c), img(c + 1)) for c in sorted(set(centres_f) | set(centres_b))}
        fwd.append([(hop[c][1], hop[c][2]) for c in centres_f])
        bwd.append([(hop[c][0], hop[c][2]) for c in centres_b] if backward else None)
    return fwd, bwd


def test_hop_keys_clip_at_the_sequence_start_and_overlap_the_next_keyframe():
    frames = _camera_frames(0, 20, [0, 1])
    cf, cb, keys = MODULE.memfof_hop_triplets(0, 5, backward=False, camera_frames=frames, n_cams=2)
    assert cf == [0, 1, 2, 3, 4] and cb == []
    assert keys[(0, 0)] == (0, 0, 0, 1) and keys[(1, 0)] == (1, 0, 1, 2)
    cf5, cb5, keys5 = MODULE.memfof_hop_triplets(5, 5, backward=True, camera_frames=frames, n_cams=2)
    assert cb5 == [5, 4, 3, 2, 1]
    shared = set(keys.values()) & set(keys5.values())
    assert shared == {keys[(cam, c)] for cam in range(2) for c in (1, 2, 3, 4)}


def test_composed_warps_match_the_per_camera_loop_and_reuse_shared_triplets():
    n_cams, d = 3, 4
    offsets = [0, 1, -1]
    frames = _camera_frames(0, 13, offsets)
    images = _images(n_cams, 14, seed=5)
    sizes = [(40, 70)] * n_cams
    rng = np.random.default_rng(9)
    uvs = np.stack([rng.uniform(0, 69, 200), rng.uniform(0, 39, 200)], 1).astype(np.float32)

    be = _backend(5)
    cache = {}
    computed = []
    batch = be.triplet_flows_batch

    def counting_batch(triplets, fetch):
        computed.append(len(triplets))
        return batch(triplets, fetch)

    be.triplet_flows_batch = counting_batch
    for frame_idx, backward in ((0, False), (4, True), (8, True)):
        fwd, bwd = MODULE.memfof_composed_warps(
            be, frame_idx=frame_idx, frame_delta=d, backward=backward, camera_frames=frames,
            image_sizes=sizes, fetch=images.__getitem__, cache=cache,
        )
        old_f, old_b = _old_composed(_backend(1), images, frame_idx, d, backward, frames, n_cams)
        for cam in range(n_cams):
            ref_f = rb.MemfofTemporalBackend.make_composed_dense(None, 1.0, 40, 70, old_f[cam])
            np.testing.assert_array_equal(be.sample_at_uvs(fwd[cam], uvs)[0], be.sample_at_uvs(ref_f, uvs)[0])
            if backward:
                ref_b = rb.MemfofTemporalBackend.make_composed_dense(None, 1.0, 40, 70, old_b[cam])
                np.testing.assert_array_equal(be.sample_at_uvs(bwd[cam], uvs)[0], be.sample_at_uvs(ref_b, uvs)[0])
            else:
                assert bwd[cam] is None
            assert MODULE.validate_warp_layout(be, fwd[cam])
    # Keyframe 0: centres 0..3; keyframe 4: 1..7 of which 1..3 are cached; keyframe 8: 5..11, 5..7 cached.
    assert computed == [4 * n_cams, 4 * n_cams, 4 * n_cams]
    assert len(cache) == 7 * n_cams   # only keyframe 8's triplets remain
