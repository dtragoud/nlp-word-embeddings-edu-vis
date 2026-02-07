"""Visualization utilities for word embeddings."""
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(
    vectors: np.ndarray,
    n_components: int = 3,
    method: str = "pca",
) -> np.ndarray:
    """
    Reduce high-dimensional vectors to 2D or 3D for visualization.

    Args:
        vectors: Array of shape (n_words, embedding_dim).
        n_components: Target dimensions (2 or 3).
        method: 'pca' or 'tsne'.

    Returns:
        Reduced array of shape (n_words, n_components).

    Raises:
        ValueError: If method is not 'pca' or 'tsne'.
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        # Perplexity must be less than n_samples
        perplexity = min(30, vectors.shape[0] - 1)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
        )
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    return reducer.fit_transform(vectors)


def plot_3d_scatter(
    words: List[str],
    coordinates: np.ndarray,
    title: str = "Word Embeddings",
    categories: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create an interactive 3D scatter plot of word embeddings.

    Args:
        words: List of word labels.
        coordinates: Array of shape (n_words, 3).
        title: Plot title.
        categories: Optional list of category labels for color-coding.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()

    if categories is not None:
        # Principle 7: Readability counts — one trace per category for a clean legend
        unique_categories = list(dict.fromkeys(categories))
        for category in unique_categories:
            mask = [c == category for c in categories]
            indices = [i for i, m in enumerate(mask) if m]
            cat_coords = coordinates[indices]
            cat_words = [words[i] for i in indices]

            fig.add_trace(go.Scatter3d(
                x=cat_coords[:, 0],
                y=cat_coords[:, 1],
                z=cat_coords[:, 2],
                mode="markers+text",
                marker=dict(size=8),
                text=cat_words,
                textposition="top center",
                name=category,
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="markers+text",
            marker=dict(size=8, color="steelblue"),
            text=words,
            textposition="top center",
            name="words",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        ),
        height=700,
    )
    return fig


def plot_vector_arithmetic(
    words: List[str],
    coordinates: np.ndarray,
    word_a: str,
    word_b: str,
    word_c: str,
    result_word: str,
) -> go.Figure:
    """
    Visualize a vector arithmetic operation (A - B + C = ?) in 3D.

    Shows input words, the result word, and connecting lines
    to illustrate the arithmetic relationship.

    Args:
        words: All word labels (including inputs and result).
        coordinates: Array of shape (n_words, 3) matching words.
        word_a: Base word (e.g., "king").
        word_b: Word to subtract (e.g., "man").
        word_c: Word to add (e.g., "woman").
        result_word: Nearest vocabulary word to the result vector.

    Returns:
        A Plotly Figure object.
    """
    word_to_idx = {w: i for i, w in enumerate(words)}

    fig = go.Figure()

    # Plot all context words as small gray markers
    fig.add_trace(go.Scatter3d(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        z=coordinates[:, 2],
        mode="markers+text",
        marker=dict(size=5, color="lightgray"),
        text=words,
        textposition="top center",
        name="words",
        showlegend=False,
    ))

    # Highlight the key words with distinct colors
    key_words = {
        word_a: ("#1F4788", "A (base)"),
        word_b: ("#E74C3C", "B (subtract)"),
        word_c: ("#28A745", "C (add)"),
        result_word: ("#FFC107", "Result"),
    }

    for word, (color, label) in key_words.items():
        if word not in word_to_idx:
            continue
        idx = word_to_idx[word]
        fig.add_trace(go.Scatter3d(
            x=[coordinates[idx, 0]],
            y=[coordinates[idx, 1]],
            z=[coordinates[idx, 2]],
            mode="markers+text",
            marker=dict(size=12, color=color),
            text=[word],
            textposition="top center",
            name=f"{label}: {word}",
        ))

    # Draw lines showing the arithmetic: A -> B (subtract), C -> Result (add)
    for start_word, end_word in [(word_a, word_b), (word_c, result_word)]:
        if start_word not in word_to_idx or end_word not in word_to_idx:
            continue
        si = word_to_idx[start_word]
        ei = word_to_idx[end_word]
        fig.add_trace(go.Scatter3d(
            x=[coordinates[si, 0], coordinates[ei, 0]],
            y=[coordinates[si, 1], coordinates[ei, 1]],
            z=[coordinates[si, 2], coordinates[ei, 2]],
            mode="lines",
            line=dict(color="gray", width=3, dash="dash"),
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Vector Arithmetic: {word_a} - {word_b} + {word_c} = {result_word}",
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        ),
        height=700,
    )
    return fig
