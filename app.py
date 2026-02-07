"""Main Streamlit application for the Word Embeddings Explorer."""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

from utils.embedding_utils import load_model, get_vector, vector_arithmetic
from utils.viz_utils import reduce_dimensions, plot_3d_scatter, plot_vector_arithmetic

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Word Embeddings Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load demo words
# ---------------------------------------------------------------------------
DEMO_WORDS_PATH = Path(__file__).parent / "data" / "demo_words.json"


@st.cache_data
def load_demo_words() -> Dict[str, List[str]]:
    """Load curated demo words from JSON."""
    with open(DEMO_WORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


demo_words = load_demo_words()

# Flatten all demo words into a single list with category mapping
all_demo_words: List[str] = []
word_to_category: Dict[str, str] = {}
for category, words in demo_words.items():
    for word in words:
        all_demo_words.append(word)
        word_to_category[word] = category

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

model_name = st.sidebar.selectbox(
    "Embedding Model",
    options=["glove-wiki-gigaword-50", "glove-wiki-gigaword-100", "word2vec-google-news-300"],
    index=0,
    help="Larger models are more accurate but slower to download.",
)

reduction_method = st.sidebar.selectbox(
    "Dimensionality Reduction",
    options=["pca", "tsne"],
    format_func=lambda x: "PCA" if x == "pca" else "t-SNE",
    help="PCA is faster; t-SNE better preserves local structure.",
)

n_dims = st.sidebar.radio(
    "Visualization Dimensions",
    options=[3, 2],
    format_func=lambda x: f"{x}D",
    horizontal=True,
)

# ---------------------------------------------------------------------------
# Load model (cached — only slow on first run)
# ---------------------------------------------------------------------------
with st.spinner(f"Loading model **{model_name}** (first load downloads ~50-300 MB)..."):
    model = load_model(model_name)

if model is None:
    st.error("Failed to load embedding model. Please check your internet connection and try again.")
    st.stop()

# ---------------------------------------------------------------------------
# Word Selection (needs model for vocabulary validation)
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Word Selection")

# Persist custom words across reruns via session state
if "custom_words" not in st.session_state:
    st.session_state.custom_words = []
if "add_feedback" not in st.session_state:
    st.session_state.add_feedback = None


def _handle_add_words() -> None:
    """Callback: validate and add custom words, then clear the input."""
    raw = st.session_state.custom_words_input
    new_words = [w.strip().lower() for w in raw.split(",") if w.strip()]
    added = []
    not_found = []
    for w in new_words:
        if w in model and w not in st.session_state.custom_words and w not in all_demo_words:
            st.session_state.custom_words.append(w)
            word_to_category[w] = "custom"
            added.append(w)
        elif w not in model:
            not_found.append(w)
    # Store feedback to display after rerun
    st.session_state.add_feedback = {"added": added, "not_found": not_found}
    # Clear input — safe inside a callback (runs before widget renders)
    st.session_state.custom_words_input = ""


# Add custom words UI: text input + Add button
st.sidebar.text_input(
    "Add custom words (comma-separated)",
    placeholder="e.g. car, house, river",
    key="custom_words_input",
)

st.sidebar.button("Add words", type="primary", on_click=_handle_add_words)

# Show feedback from the previous add action
if st.session_state.add_feedback:
    fb = st.session_state.add_feedback
    if fb["added"]:
        st.sidebar.success(f"Added: {', '.join(fb['added'])}")
    if fb["not_found"]:
        st.sidebar.warning(f"Not in vocabulary: {', '.join(fb['not_found'])}")
    st.session_state.add_feedback = None

# Build multiselect options: demo words + persisted custom words
all_available_words = all_demo_words + st.session_state.custom_words

selected_words = st.sidebar.multiselect(
    "Choose words to visualize",
    options=all_available_words,
    default=all_available_words,
    help="Remove words by clicking X. Add new ones above.",
)

# The multiselect is the single source of truth for which words to plot
valid_words: List[str] = [w for w in selected_words if w in model]

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Word Embeddings & Vector Arithmetic")
st.markdown("**Explore how computers understand words as numbers**")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_explore, tab_arithmetic, tab_learn = st.tabs([
    "Explore Embeddings",
    "Vector Arithmetic",
    "Learn",
])

# ========================== TAB 1: Explore =================================
with tab_explore:
    if len(valid_words) < 2:
        st.info("Select at least 2 valid words from the sidebar to visualize.")
    else:
        # Build vectors matrix
        vectors = np.array([model[w] for w in valid_words])
        categories = [word_to_category.get(w, "custom") for w in valid_words]

        # Reduce dimensions
        coords = reduce_dimensions(vectors, n_components=n_dims, method=reduction_method)

        if n_dims == 3:
            fig = plot_3d_scatter(valid_words, coords, categories=categories,
                                 title="Word Embedding Clusters")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 2D scatter using Plotly for consistency
            import plotly.graph_objects as go

            fig = go.Figure()
            unique_cats = list(dict.fromkeys(categories))
            for cat in unique_cats:
                indices = [i for i, c in enumerate(categories) if c == cat]
                cat_coords = coords[indices]
                cat_words = [valid_words[i] for i in indices]
                fig.add_trace(go.Scatter(
                    x=cat_coords[:, 0],
                    y=cat_coords[:, 1],
                    mode="markers+text",
                    marker=dict(size=10),
                    text=cat_words,
                    textposition="top center",
                    name=cat,
                ))
            fig.update_layout(
                title="Word Embedding Clusters (2D)",
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("What am I looking at?"):
            st.markdown(
                """
                Each point represents a **word** positioned according to its
                meaning in the embedding model. Words that are **semantically
                similar** (e.g., *king* and *queen*) appear **close together**.

                - **PCA** projects along the directions of maximum variance.
                - **t-SNE** preserves local neighborhood structure, often
                  revealing tighter clusters.

                Try switching between PCA and t-SNE in the sidebar to see how
                the layout changes!
                """
            )

# ========================== TAB 2: Arithmetic ==============================
with tab_arithmetic:
    st.subheader("Word Vector Arithmetic")
    st.markdown(
        "Enter three words to compute: **A** - **B** + **C** = ?"
    )
    st.info(
        'Classic example: **king** - **man** + **woman** = **queen**\n\n'
        "This works because the vector from *man* to *king* encodes "
        '"royalty", and adding that to *woman* lands near *queen*.'
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        word_a = st.text_input("Word A (base)", value="king").strip().lower()
    with col2:
        word_b = st.text_input("Word B (subtract)", value="man").strip().lower()
    with col3:
        word_c = st.text_input("Word C (add)", value="woman").strip().lower()

    # Pre-validate words and give immediate feedback
    input_words_map = {"A": word_a, "B": word_b, "C": word_c}
    invalid = {label: w for label, w in input_words_map.items() if w and w not in model}
    for label, w in invalid.items():
        st.warning(f"Word {label} **\"{w}\"** is not in the model vocabulary. Try another word.")

    if st.button("Compute", type="primary"):
        if not all([word_a, word_b, word_c]):
            st.warning("Please fill in all three words.")
        elif invalid:
            st.error("Fix the invalid words above before computing.")
        else:
            result = vector_arithmetic(model, word_a, word_b, word_c)

            if "error" in result:
                st.error(result["error"])
            else:
                # Display nearest words
                st.subheader("Results")
                st.markdown(f"**{word_a}** - **{word_b}** + **{word_c}** = ?")

                for rank, (word, score) in enumerate(result["nearest_words"], start=1):
                    st.markdown(
                        f"{rank}. **{word}** (similarity: {score:.4f})"
                    )

                # Visualize the arithmetic in 3D
                st.subheader("Visualization")
                top_result_word = result["nearest_words"][0][0]
                arith_words = [word_a, word_b, word_c, top_result_word]
                arith_vectors = np.array([
                    result["input_vectors"][word_a],
                    result["input_vectors"][word_b],
                    result["input_vectors"][word_c],
                    model[top_result_word],
                ])
                arith_coords = reduce_dimensions(
                    arith_vectors, n_components=3, method="pca"
                )
                fig = plot_vector_arithmetic(
                    words=arith_words,
                    coordinates=arith_coords,
                    word_a=word_a,
                    word_b=word_b,
                    word_c=word_c,
                    result_word=top_result_word,
                )
                st.plotly_chart(fig, use_container_width=True)

# ========================== TAB 3: Learn ===================================
with tab_learn:
    st.subheader("What Are Word Embeddings?")
    st.markdown(
        """
        Word embeddings are a way to represent words as **numerical vectors**
        (lists of numbers). Instead of treating words as arbitrary symbols,
        embeddings capture **meaning** by placing similar words close together
        in a high-dimensional space.
        """
    )

    with st.expander("How are they created?"):
        st.markdown(
            """
            Models like **Word2Vec** and **GloVe** learn embeddings by
            analyzing large amounts of text. The core idea:

            > *"You shall know a word by the company it keeps."*
            > — J.R. Firth (1957)

            Words that frequently appear in similar contexts end up with
            similar vectors. For example, *dog* and *cat* often appear near
            words like *pet*, *fur*, and *vet*, so their vectors are close.
            """
        )

    with st.expander("Why does vector arithmetic work?"):
        st.markdown(
            """
            Embedding spaces encode **relationships** as directions. The
            vector offset from *man* to *woman* encodes the concept of
            gender. The same offset exists between *king* and *queen*:

            ```
            king - man + woman ≈ queen
            ```

            This means the model has learned that the relationship between
            *king* and *man* is analogous to the relationship between *queen*
            and *woman*.
            """
        )

    with st.expander("What are PCA and t-SNE?"):
        st.markdown(
            """
            Word vectors typically have 50-300 dimensions — far too many to
            visualize directly. **Dimensionality reduction** techniques
            compress these into 2D or 3D:

            - **PCA** (Principal Component Analysis) finds the axes of
              greatest variation. It's fast and preserves global structure.
            - **t-SNE** (t-distributed Stochastic Neighbor Embedding)
              focuses on preserving local neighborhoods, often producing
              tighter, more separated clusters.
            """
        )

    st.markdown("---")
    st.markdown(
        "**Try it yourself!** Head to the *Explore Embeddings* tab to "
        "visualize word clusters, or the *Vector Arithmetic* tab to "
        "experiment with word equations."
    )
