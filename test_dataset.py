"""Unit tests for dataset.py"""
import torch
import pytest
from dataset import make_image_and_points, CalibDataset, build_loaders


def test_shapes():
    img, true_uv, dist_uv = make_image_and_points(n_points=256, img_size=128, seed=0)
    assert img.shape == (1, 128, 128), f"image shape {img.shape}"
    assert true_uv.shape == (256, 2), f"true_uv shape {true_uv.shape}"
    assert dist_uv.shape == (256, 2), f"dist_uv shape {dist_uv.shape}"


def test_image_range():
    img, _, _ = make_image_and_points(seed=1)
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert img.dtype == torch.float32


def test_image_has_shape():
    """Image must not be all black (a shape should be rendered)."""
    img, _, _ = make_image_and_points(seed=2)
    assert img.sum() > 0, "Image is blank – no shape rendered"


def test_uv_in_bounds():
    for seed in range(10):
        _, true_uv, dist_uv = make_image_and_points(img_size=128, seed=seed)
        assert true_uv.min() >= 0.0
        assert true_uv.max() < 128.0
        assert dist_uv.min() >= 0.0
        assert dist_uv.max() <= 127.0


def test_offset_magnitude():
    _, true_uv, dist_uv = make_image_and_points(max_offset=15.0, seed=5)
    diff = (dist_uv - true_uv).norm(dim=1)
    assert diff.max() <= 15.0 + 1e-3, f"max offset {diff.max():.2f} exceeds 15"


def test_deterministic():
    img1, t1, d1 = make_image_and_points(seed=99)
    img2, t2, d2 = make_image_and_points(seed=99)
    assert torch.allclose(img1, img2)
    assert torch.allclose(t1, t2)
    assert torch.allclose(d1, d2)


def test_different_seeds():
    _, t1, _ = make_image_and_points(seed=0)
    _, t2, _ = make_image_and_points(seed=1)
    assert not torch.allclose(t1, t2)


def test_dataset_len():
    ds = CalibDataset(length=50)
    assert len(ds) == 50


def test_dataset_item_shapes():
    ds = CalibDataset(length=10, n_points=256, img_size=128)
    img, true_uv, dist_uv = ds[0]
    assert img.shape == (1, 128, 128)
    assert true_uv.shape == (256, 2)
    assert dist_uv.shape == (256, 2)


def test_circle_shape():
    """Force circle branch: seed=0 should produce a circle (shape_type depends on seed)."""
    # Just test it doesn't crash and produces correct shapes
    for seed in range(20):
        img, t, d = make_image_and_points(seed=seed)
        assert img.shape == (1, 128, 128)
        assert t.shape == (256, 2)
        assert d.shape == (256, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
