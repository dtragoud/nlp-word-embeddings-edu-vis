"""Tests for viz_utils module."""
import numpy as np
import plotly.graph_objects as go
import pytest

from utils.viz_utils import reduce_dimensions, plot_3d_scatter, plot_vector_arithmetic


@pytest.fixture
def sample_vectors() -> np.ndarray:
    """Create sample high-dimensional vectors for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 50)).astype(np.float32)


@pytest.fixture
def sample_words() -> list[str]:
    """Word labels matching sample_vectors fixture."""
    return ["king", "queen", "man", "woman", "prince",
            "dog", "cat", "red", "blue", "table"]


class TestReduceDimensions:
    """Tests for the reduce_dimensions function."""

    def test_pca_reduces_to_3d(self, sample_vectors: np.ndarray) -> None:
        result = reduce_dimensions(sample_vectors, n_components=3, method="pca")
        assert result.shape == (10, 3)

    def test_pca_reduces_to_2d(self, sample_vectors: np.ndarray) -> None:
        result = reduce_dimensions(sample_vectors, n_components=2, method="pca")
        assert result.shape == (10, 2)

    def test_tsne_reduces_to_3d(self, sample_vectors: np.ndarray) -> None:
        result = reduce_dimensions(sample_vectors, n_components=3, method="tsne")
        assert result.shape == (10, 3)

    def test_tsne_reduces_to_2d(self, sample_vectors: np.ndarray) -> None:
        result = reduce_dimensions(sample_vectors, n_components=2, method="tsne")
        assert result.shape == (10, 2)

    def test_invalid_method_raises(self, sample_vectors: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Method must be"):
            reduce_dimensions(sample_vectors, method="invalid")

    def test_output_is_float(self, sample_vectors: np.ndarray) -> None:
        result = reduce_dimensions(sample_vectors, n_components=3, method="pca")
        assert np.issubdtype(result.dtype, np.floating)


class TestPlot3dScatter:
    """Tests for the plot_3d_scatter function."""

    def test_returns_plotly_figure(self, sample_words: list[str]) -> None:
        coords = np.random.default_rng(0).standard_normal((10, 3))
        fig = plot_3d_scatter(sample_words, coords)
        assert isinstance(fig, go.Figure)

    def test_with_categories(self, sample_words: list[str]) -> None:
        coords = np.random.default_rng(0).standard_normal((10, 3))
        categories = ["royalty"] * 4 + ["royalty"] + ["animals"] * 2 + ["colors"] * 2 + ["household"]
        fig = plot_3d_scatter(sample_words, coords, categories=categories)
        assert isinstance(fig, go.Figure)
        # Should have one trace per unique category
        assert len(fig.data) == 4

    def test_custom_title(self, sample_words: list[str]) -> None:
        coords = np.random.default_rng(0).standard_normal((10, 3))
        fig = plot_3d_scatter(sample_words, coords, title="Test Title")
        assert fig.layout.title.text == "Test Title"


class TestPlotVectorArithmetic:
    """Tests for the plot_vector_arithmetic function."""

    def test_returns_plotly_figure(self) -> None:
        coords = np.random.default_rng(0).standard_normal((5, 3))
        words = ["king", "man", "woman", "result", "queen"]
        fig = plot_vector_arithmetic(
            words=words,
            coordinates=coords,
            word_a="king",
            word_b="man",
            word_c="woman",
            result_word="queen",
        )
        assert isinstance(fig, go.Figure)
