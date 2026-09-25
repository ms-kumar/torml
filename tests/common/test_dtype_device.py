"""Tests for dtype preservation and device propagation."""

from __future__ import annotations

import pytest
import torch

from torml.cluster import KMeans
from torml.decomposition import PCA
from torml.linear_model import LinearRegression
from torml.neighbors import KNeighborsClassifier
from torml.preprocessing import StandardScaler
from torml.tree import DecisionTreeClassifier
from torml.utils import check_array

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


class TestCheckArrayDtype:
    """Tests for check_array dtype=None preservation."""

    def test_preserves_float64(self):
        """Test float64 tensors keep their dtype."""
        out = check_array(torch.randn(4, 2, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_preserves_float32(self):
        """Test float32 tensors keep their dtype."""
        out = check_array(torch.randn(4, 2, dtype=torch.float32))
        assert out.dtype == torch.float32

    def test_promotes_int(self):
        """Test integer tensors promote to float32."""
        out = check_array(torch.ones(4, 2, dtype=torch.int64))
        assert out.dtype == torch.float32

    def test_list_defaults_float32(self):
        """Test lists default to float32."""
        out = check_array([[1.0, 2.0], [3.0, 4.0]])
        assert out.dtype == torch.float32

    def test_explicit_dtype_still_forces(self):
        """Test explicit dtype overrides preservation."""
        out = check_array(torch.randn(4, 2, dtype=torch.float64), dtype=torch.float32)
        assert out.dtype == torch.float32


class TestDtypePreservation:
    """Tests that estimators compute in the input dtype."""

    def test_scaler_float64(self):
        """Test StandardScaler learns float64 statistics."""
        X = torch.randn(20, 3, dtype=torch.float64)
        sc = StandardScaler().fit(X)
        assert sc.mean_.dtype == torch.float64
        assert sc.scale_.dtype == torch.float64
        assert sc.transform(X).dtype == torch.float64
        assert sc.inverse_transform(sc.transform(X)).dtype == torch.float64

    def test_linear_regression_float64(self):
        """Test LinearRegression learns float64 coefficients."""
        torch.manual_seed(0)
        X = torch.randn(30, 2, dtype=torch.float64)
        y = X[:, 0] * 2 - X[:, 1]
        model = LinearRegression().fit(X, y)
        assert model.coef_.dtype == torch.float64
        assert model.predict(X).dtype == torch.float64

    def test_numeric_agreement(self):
        """Test float32 and float64 runs agree closely."""
        torch.manual_seed(1)
        X32 = torch.randn(30, 2)
        y32 = X32[:, 0] * 2 - X32[:, 1]
        X64, y64 = X32.double(), y32.double()
        pred32 = LinearRegression().fit(X32, y32).predict(X32).double()
        pred64 = LinearRegression().fit(X64, y64).predict(X64)
        torch.testing.assert_close(pred32, pred64, rtol=1e-4, atol=1e-4)

    def test_int_features_accepted(self):
        """Test integer features are promoted, not rejected."""
        X = torch.randint(0, 5, (20, 2))
        y = (X[:, 0] > 2).long()
        pred = KNeighborsClassifier(3).fit(X, y).predict(X)
        assert pred.shape == (20,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_predict_accepts_both(self, dtype):
        """Test classifiers predict on both float dtypes."""
        torch.manual_seed(2)
        X = (torch.randn(20, 2)).to(dtype)
        y = (X[:, 0] > 0).long()
        for clf in (KNeighborsClassifier(3), DecisionTreeClassifier()):
            assert clf.fit(X, y).predict(X).shape == (20,)

    def test_pca_float64(self):
        """Test PCA keeps float64 through transform."""
        X = torch.randn(20, 4, dtype=torch.float64)
        pca = PCA(n_components=2).fit(X)
        assert pca.components_.dtype == torch.float64
        assert pca.transform(X).dtype == torch.float64

    def test_kmeans_float64(self):
        """Test KMeans centers match input dtype."""
        X = torch.randn(20, 2, dtype=torch.float64)
        km = KMeans(n_clusters=2, random_state=0, n_init=2).fit(X)
        assert km.cluster_centers_.dtype == torch.float64


class TestDevice:
    """Tests for device propagation (CPU always, CUDA when present)."""

    def test_cpu_round_trip(self):
        """Test CPU tensors stay on CPU."""
        X = torch.randn(20, 2)
        y = (X[:, 0] > 0).long()
        assert StandardScaler().fit_transform(X).device.type == "cpu"
        assert LinearRegression().fit(X, y.float()).predict(X).device.type == "cpu"
        assert (
            KMeans(n_clusters=2, random_state=0, n_init=2).fit(X).labels_.device.type
            == "cpu"
        )

    @cuda_only
    def test_cuda_round_trip(self):
        """Test CUDA tensors stay on CUDA end to end."""
        torch.manual_seed(0)
        X = torch.randn(40, 3, device="cuda")
        y = (X[:, 0] > 0).long()
        yr = X[:, 0] * 2 - X[:, 1]
        assert StandardScaler().fit_transform(X).device.type == "cuda"
        assert LinearRegression().fit(X, yr).predict(X).device.type == "cuda"
        assert KNeighborsClassifier(3).fit(X, y).predict(X).device.type == "cuda"
        assert (
            KMeans(n_clusters=2, random_state=0, n_init=2).fit(X).labels_.device.type
            == "cuda"
        )


class TestMPS:
    """Tests for Apple Silicon propagation (run where MPS exists)."""

    @mps_only
    def test_estimators_stay_on_mps(self):
        """Test core estimators stay on MPS end to end."""
        from torml.ensemble import RandomForestClassifier
        from torml.mixture import GaussianMixture
        from torml.svm import LinearSVC

        torch.manual_seed(0)
        X = torch.randn(40, 3, device="mps")
        y = (X[:, 0] > 0).long()
        yr = X[:, 0] * 2 - X[:, 1]
        assert StandardScaler().fit_transform(X).device.type == "mps"
        assert LinearRegression().fit(X, yr).predict(X).device.type == "mps"
        assert KNeighborsClassifier(3).fit(X, y).predict(X).device.type == "mps"
        assert DecisionTreeClassifier().fit(X, y).predict(X).device.type == "mps"
        assert (
            KMeans(n_clusters=2, random_state=0, n_init=2).fit(X).labels_.device.type
            == "mps"
        )
        assert (
            GaussianMixture(n_components=2, random_state=0)
            .fit(X)
            .predict(X)
            .device.type
            == "mps"
        )
        assert LinearSVC(random_state=0).fit(X, y).predict(X).device.type == "mps"
        assert (
            RandomForestClassifier(n_estimators=2, random_state=0)
            .fit(X, y)
            .predict(X)
            .device.type
            == "mps"
        )

    @mps_only
    def test_mps_label_tables(self):
        """Test class tables work with on-device indices."""
        from torml.preprocessing import LabelEncoder

        y = torch.tensor([0, 1, 0], device="mps")
        assert LabelEncoder().fit_transform(y).device.type == "mps"
