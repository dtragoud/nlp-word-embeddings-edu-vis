"""Tests for embedding_utils module."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from utils.embedding_utils import get_vector, vector_arithmetic


@pytest.fixture
def mock_model():
    """Create a mock word embedding model with predictable vectors."""
    model = MagicMock()

    vectors = {
        "king": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "queen": np.array([0.8, 0.2, 0.0], dtype=np.float32),
        "man": np.array([0.5, -0.5, 0.0], dtype=np.float32),
        "woman": np.array([0.3, -0.3, 0.0], dtype=np.float32),
    }

    def getitem(word: str) -> np.ndarray:
        if word in vectors:
            return vectors[word]
        raise KeyError(word)

    model.__getitem__ = MagicMock(side_effect=getitem)
    model.__contains__ = MagicMock(side_effect=lambda w: w in vectors)
    model.similar_by_vector = MagicMock(
        return_value=[("king", 0.96), ("queen", 0.95), ("woman", 0.85),
                      ("man", 0.80), ("princess", 0.75), ("prince", 0.70),
                      ("daughter", 0.65), ("throne", 0.60)]
    )

    return model


class TestGetVector:
    """Tests for the get_vector function."""

    def test_returns_vector_for_known_word(self, mock_model: MagicMock) -> None:
        result = get_vector(mock_model, "king")
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returns_none_for_unknown_word(self, mock_model: MagicMock) -> None:
        result = get_vector(mock_model, "zzzznotaword")
        assert result is None


class TestVectorArithmetic:
    """Tests for the vector_arithmetic function."""

    def test_returns_result_with_valid_words(self, mock_model: MagicMock) -> None:
        result = vector_arithmetic(mock_model, "king", "man", "woman")
        assert "result_vector" in result
        assert "nearest_words" in result
        assert "input_vectors" in result

    def test_result_vector_is_correct(self, mock_model: MagicMock) -> None:
        # king - man + woman = [1,0,0] - [0.5,-0.5,0] + [0.3,-0.3,0] = [0.8,0.2,0]
        result = vector_arithmetic(mock_model, "king", "man", "woman")
        expected = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["result_vector"], expected)

    def test_input_vectors_stored(self, mock_model: MagicMock) -> None:
        result = vector_arithmetic(mock_model, "king", "man", "woman")
        assert "king" in result["input_vectors"]
        assert "man" in result["input_vectors"]
        assert "woman" in result["input_vectors"]

    def test_returns_error_for_unknown_word(self, mock_model: MagicMock) -> None:
        result = vector_arithmetic(mock_model, "king", "man", "zzzznotaword")
        assert "error" in result

    def test_nearest_words_is_list(self, mock_model: MagicMock) -> None:
        result = vector_arithmetic(mock_model, "king", "man", "woman")
        assert isinstance(result["nearest_words"], list)
        assert len(result["nearest_words"]) > 0

    def test_input_words_excluded_from_results(self, mock_model: MagicMock) -> None:
        """Input words (king, man, woman) must not appear in nearest_words."""
        result = vector_arithmetic(mock_model, "king", "man", "woman")
        result_words = [w for w, _ in result["nearest_words"]]
        assert "king" not in result_words
        assert "man" not in result_words
        assert "woman" not in result_words
        # "queen" should now be the top result
        assert result_words[0] == "queen"
