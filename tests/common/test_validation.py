"""Test validation utilities."""

from __future__ import annotations

import pytest
import torch

from torml.utils._validation import check_array


@pytest.fixture(params=["float32", "float64", "int32", "int64"])
def dtype(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64, torch.int32, torch.int64])
def torch_dtype(request):
    return request.param


@pytest.fixture
def X1d():
    return torch.randn(100)


@pytest.fixture
def X2d():
    return torch.randn(100, 10)


@pytest.fixture
def X_nd():
    return torch.randn(100, 20)


@pytest.fixture
def y1d():
    return torch.randn(100)


@pytest.fixture
def y2d():
    return torch.randn(100, 5)


@pytest.fixture
def tensor_2d():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def tensor_none():
    return None


@pytest.fixture
def invalid_input():
    return "not an array"


@pytest.fixture(params=[True, False])
def force_finite(request):
    return request.param


@pytest.fixture(params=[True, False])
def ensure_min_samples_match(request):
    """When True, samples=10, ensure_min_samples=10."""
    return request.param


class TestCheckArray:
    """Tests for check_array."""

    def test_2d_tensor(self, X2d, dtype):
        """Test that 2D tensor is returned with correct dtype."""
        X_checked = check_array(X2d, dtype=dtype)
        assert X_checked.shape == X2d.shape
        assert X_checked.dtype == getattr(torch, dtype)

    def test_1d_tensor(self, X1d, dtype):
        """Test that 1D tensor is converted to 2D."""
        X_checked = check_array(X1d, dtype=dtype, ensure_2d=True)
        assert X_checked.shape == (X1d.shape[0], 1)
        assert X_checked.dtype == getattr(torch, dtype)

    def test_2d_2d_tensor(self, dtype, tensor_2d):
        """Test that 2D tensor is passed through."""
        X_checked = check_array(tensor_2d, dtype=dtype, ensure_2d=True, allow_nd=False)
        assert torch.equal(X_checked.to(torch.float32), tensor_2d.to(torch.float32))
        assert X_checked.dtype == getattr(torch, dtype)

    def test_nd_tensor_fails_ensure_2d(self, dtype):
        """Test that nd tensor fails with ensure_2d=True."""
        X_nd_3d = torch.randn(4, 3, 2)
        with pytest.raises(ValueError, match="2D|array|3D"):
            check_array(X_nd_3d, dtype=dtype, ensure_2d=True, allow_nd=False)

    def test_nd_tensor_allowed(self, dtype):
        """Test that nd tensor is allowed with allow_nd=True."""
        X_nd_3d = torch.randn(4, 3, 2)
        X_checked = check_array(X_nd_3d, dtype=dtype, ensure_2d=False, allow_nd=True)
        assert tuple(X_checked.shape) == tuple(X_nd_3d.shape)
        assert X_checked.dtype == getattr(torch, dtype)

    def test_copy_param(self, tensor_2d, dtype):
        """Test copy param works."""
        X_checked = check_array(tensor_2d, dtype=dtype, copy=True)
        assert torch.equal(X_checked.to(torch.float32), tensor_2d.to(torch.float32))
        assert X_checked.data_ptr() != tensor_2d.data_ptr()

    def test_min_samples(self, tensor_2d, dtype, ensure_min_samples_match):
        """Test min samples check."""
        if ensure_min_samples_match:
            n_samples = 10
        else:
            n_samples = 1
        with pytest.raises(ValueError, match="samples"):
            check_array(
                tensor_2d[:n_samples],
                dtype=dtype,
                force_all_finite=False,
                ensure_min_samples=10,
            )

    def test_min_features(self, tensor_2d, dtype):
        """Test min features check."""
        X = tensor_2d[:, :2]
        with pytest.raises(ValueError, match="features"):
            check_array(
                X,
                dtype=dtype,
                force_all_finite=False,
                ensure_min_samples=3,
                ensure_min_features=5,
            )

    def test_non_finite_values(self, tensor_2d, dtype, force_finite):
        """Test non-finite values check."""
        # NaN/Inf only representable in float dtypes; use float32 for int cases.
        eff_dtype = dtype if dtype in ("float32", "float64") else "float32"
        if force_finite:
            nan_X = tensor_2d.clone()
            nan_X[0, 0] = float("nan")
            with pytest.raises(ValueError, match="NaN"):
                check_array(nan_X, dtype=eff_dtype, force_all_finite=True)
        else:
            nan_X = tensor_2d.clone()
            nan_X[0, 0] = float("nan")
            result = check_array(nan_X, dtype=eff_dtype, force_all_finite=False)
            assert torch.isnan(result[0, 0])

    def test_input_name(self, dtype):
        """Test custom input name in errors."""
        bad_X = torch.randn(4, 3, 2)
        with pytest.raises(ValueError, match="my_input"):
            check_array(
                bad_X,
                dtype=dtype,
                force_all_finite=False,
                ensure_2d=True,
                input_name="my_input",
            )

    def test_device_preserved(self, tensor_2d, dtype):
        """Test that device is preserved."""
        device = torch.device("cpu")
        X = tensor_2d.to(device)
        X_checked = check_array(X, dtype=dtype, device=device)
        assert X_checked.device == device
