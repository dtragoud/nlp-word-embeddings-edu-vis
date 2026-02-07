# Word Embeddings & Vector Arithmetic

An interactive educational web application that teaches Natural Language Processing (NLP) concepts through hands-on exploration of word embeddings and vector arithmetic.

## Features

- **Explore Embeddings** — Visualize word vectors in interactive 2D/3D scatter plots, color-coded by semantic category (animals, colors, royalty, etc.)
- **Vector Arithmetic** — Compute word equations like `king - man + woman = queen` and see the top-5 nearest results with similarity scores
- **Learn** — Educational content explaining what embeddings are, how they're created, and why vector arithmetic works
- **Custom Words** — Add your own words to the visualization and see where they land relative to the demo set
- **Multiple Models** — Choose between GloVe 50d, GloVe 100d, and Word2Vec Google News 300d
- **PCA & t-SNE** — Switch between dimensionality reduction methods to see different perspectives on the data

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd nlp_word_embeddings

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser. On first launch, it downloads the GloVe model (~66 MB), which is cached for subsequent runs.

### Optional: Pre-download a Model

To avoid the download wait on first launch:

```bash
python data/download_model.py                          # default: glove-wiki-gigaword-50
python data/download_model.py glove-wiki-gigaword-100  # larger model
```

## Project Structure

```
nlp_word_embeddings/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .streamlit/
│   └── config.toml           # UI theme and settings
├── utils/
│   ├── __init__.py
│   ├── embedding_utils.py    # Model loading & vector operations
│   └── viz_utils.py          # Visualization functions
├── data/
│   ├── demo_words.json       # Curated word list (5 semantic clusters)
│   └── download_model.py     # Script to pre-download models
├── models/
│   └── .gitkeep              # Models downloaded at runtime
└── tests/
    ├── __init__.py
    ├── test_embeddings.py    # Tests for embedding utilities
    └── test_viz.py           # Tests for visualization utilities
```

## Available Models

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| `glove-wiki-gigaword-50` | 50 | ~66 MB | Default. Fast, good for demos |
| `glove-wiki-gigaword-100` | 100 | ~128 MB | Better accuracy |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Technology Stack

- **Streamlit** — Web UI framework
- **Gensim** — Word embedding model loading
- **Plotly** — Interactive 2D/3D visualizations
- **Scikit-learn** — PCA and t-SNE dimensionality reduction
- **NumPy** — Vector operations

## License

This project is for educational purposes.
