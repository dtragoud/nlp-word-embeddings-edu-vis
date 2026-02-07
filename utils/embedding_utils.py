"""Core embedding utilities for loading models and performing vector operations."""
from typing import Any, Dict, List, Optional, Tuple, Union

import gensim.downloader as api
import numpy as np
import streamlit as st


@st.cache_resource
def load_model(model_name: str = "glove-wiki-gigaword-50") -> Optional[Any]:
    """
    Load a pre-trained word embedding model via gensim-data.

    Args:
        model_name: Name of the gensim-data model to load.

    Returns:
        The loaded KeyedVectors model, or None if loading fails.
    """
    try:
        model = api.load(model_name)
        return model
    except ValueError as e:
        st.error(f"Unknown model name '{model_name}': {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_vector(model: Any, word: str) -> Optional[np.ndarray]:
    """
    Retrieve the embedding vector for a given word.

    Args:
        model: A gensim KeyedVectors model.
        word: The word to look up.

    Returns:
        The word's vector as a numpy array, or None if not in vocabulary.
    """
    try:
        return model[word]
    except KeyError:
        return None


def vector_arithmetic(
    model: Any, word_a: str, word_b: str, word_c: str, topn: int = 5
) -> Dict[str, Any]:
    """
    Perform word vector arithmetic: A - B + C = ?

    Computes the result vector and finds the nearest real words
    in the vocabulary.

    Args:
        model: A gensim KeyedVectors model.
        word_a: The base word (e.g., "king").
        word_b: The word to subtract (e.g., "man").
        word_c: The word to add (e.g., "woman").
        topn: Number of nearest neighbors to return.

    Returns:
        Dict with 'result_vector', 'nearest_words', and 'input_vectors',
        or a dict with 'error' key if a word is not in vocabulary.
    """
    try:
        vec_a = model[word_a]
        vec_b = model[word_b]
        vec_c = model[word_c]
    except KeyError as e:
        return {"error": f"Word not in vocabulary: {e}"}

    # Principle 2: Explicit is better than implicit
    result_vector = vec_a - vec_b + vec_c

    # Fetch extra results so we still have topn after filtering input words
    input_words = {word_a, word_b, word_c}
    raw_nearest = model.similar_by_vector(result_vector, topn=topn + len(input_words))
    nearest = [(w, s) for w, s in raw_nearest if w not in input_words][:topn]

    return {
        "result_vector": result_vector,
        "nearest_words": nearest,
        "input_vectors": {
            word_a: vec_a,
            word_b: vec_b,
            word_c: vec_c,
        },
    }
