"""Unit tests for model.py – tensor shape checks."""
import torch
import pytest
from model import CalibNet, CNNBackbone, PointMLP, CrossAttentionBlock, PosEnc2D

B, N, D_DIM = 4, 256, 128


def test_posenc2d_shape():
    pe = PosEnc2D(D_DIM)
    x = torch.zeros(B, D_DIM, 32, 32)
    out = pe(x)
    assert out.shape == (B, D_DIM, 32, 32)


def test_cnn_output_shapes():
    cnn = CNNBackbone(D_DIM)
    x = torch.zeros(B, 1, 128, 128)
    coarse, fine = cnn(x)
    assert coarse.shape == (B, D_DIM, 16, 16), f"coarse {coarse.shape}"
    assert fine.shape   == (B, D_DIM, 32, 32), f"fine {fine.shape}"


def test_point_mlp_shape():
    mlp = PointMLP(D_DIM)
    uv = torch.rand(B, N, 2)
    out = mlp(uv)
    assert out.shape == (B, N, D_DIM)


def test_cross_attention_coarse():
    ca = CrossAttentionBlock(D_DIM, n_heads=4)
    q     = torch.rand(B, N, D_DIM)
    feat  = torch.rand(B, D_DIM, 16, 16)
    uv_01 = torch.rand(B, N, 2)
    q_out, offset = ca(q, feat, uv_01)
    assert q_out.shape  == (B, N, D_DIM)
    assert offset.shape == (B, N, 2)


def test_cross_attention_fine():
    ca = CrossAttentionBlock(D_DIM, n_heads=4)
    q     = torch.rand(B, N, D_DIM)
    feat  = torch.rand(B, D_DIM, 32, 32)
    uv_01 = torch.rand(B, N, 2)
    q_out, offset = ca(q, feat, uv_01)
    assert q_out.shape  == (B, N, D_DIM)
    assert offset.shape == (B, N, 2)


def test_calibnet_output_shape():
    model = CalibNet(D_DIM)
    image = torch.zeros(B, 1, 128, 128)
    uv = torch.rand(B, N, 2) * 127.0
    offset = model(image, uv)
    assert offset.shape == (B, N, 2), f"output shape {offset.shape}"


def test_calibnet_per_point_output():
    """Each point regresses its own offset (no mean pooling)."""
    model = CalibNet(D_DIM)
    image = torch.rand(B, 1, 128, 128)
    uv = torch.rand(B, N, 2) * 127.0
    offset = model(image, uv)
    assert offset.shape == (B, N, 2)
    # Points should NOT all be identical (per-point regression)
    assert not torch.allclose(offset[:, 0:1, :].expand_as(offset), offset, atol=1e-4), \
        "Per-point regression should produce different offsets per point"


def test_calibnet_forward_no_nan():
    model = CalibNet(D_DIM)
    image = torch.rand(B, 1, 128, 128)
    uv = torch.rand(B, N, 2) * 127.0
    offset = model(image, uv)
    assert not torch.isnan(offset).any()
    assert not torch.isinf(offset).any()


def test_calibnet_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    model = CalibNet(D_DIM).to(device)
    image = torch.rand(B, 1, 128, 128, device=device)
    uv    = torch.rand(B, N, 2, device=device) * 127.0
    offset = model(image, uv)
    assert offset.shape == (B, N, 2)
    assert not torch.isnan(offset).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
