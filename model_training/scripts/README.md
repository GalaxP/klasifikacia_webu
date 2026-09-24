# Website Category Classifier Scripts

Language-agnostic website classification from HTML source code using fastText.

## Files

- [`website_classifier.py`](website_classifier.py) - Main CLI script for training and prediction
- [`website_classifier_demo.ipynb`](website_classifier_demo.ipynb) - Interactive notebook with examples

## Quick Start

### Training

```bash
# Using default categories (Adult, Automotive, Computers, etc.)
python scripts/website_classifier.py train \
    --data dataset.jsonl \
    --output models/my_classifier.bin

# With custom categories
python scripts/website_classifier.py train \
    --data dataset.csv \
    --output models/my_classifier.bin \
    --categories "Tech,Business,Sports,Food,Travel"

# Adjust training parameters
python scripts/website_classifier.py train \
    --data dataset.jsonl \
    --output models/my_classifier.bin \
    --epochs 30 \
    --lr 0.8 \
    --dim 128
```

### Prediction

```bash
# Single HTML prediction
python scripts/website_classifier.py predict \
    --html "<html><title>Example</title>...</html>" \
    --model models/my_classifier.bin

# Batch prediction from file (one HTML per line)
python scripts/website_classifier.py predict-file \
    --input html_samples.txt \
    --model models/my_classifier.bin \
    --top-k 5
```

## Input Data Format

### JSONL Format

Each line is a JSON object with `html` and `label` fields:

```json
{"html": "<html>...</html>", "label": "Sports"}
{"html": "<html>...</html>", "label": "Finance"}
```

Or with compressed HTML:

```json
{"compressed_html": "H4sIA...", "categories": "Computers"}
```

### CSV Format

CSV with columns `html` and `label`:

```csv
html,label
"<html>...</html>",Sports
"<html>...</html>",Finance
```

### Parquet Format

Same column structure as CSV, stored in Parquet format.

## Default Categories

- Adult
- Automotive
- Computers
- Entertainment
- Finance
- Food
- Health
- News
- Shopping
- Sports
- Travel

## Python API

```python
from scripts.website_classifier import WebsiteClassifier, extract_text_from_html

# Load trained model
classifier = WebsiteClassifier("models/my_classifier.bin")

# Single prediction
preds = classifier.predict("<html>...</html>", k=3)
print(preds[0]["label"], preds[0]["score"])

# Batch prediction
results = classifier.predict_batch(["<html>...</html>", "<html>...</html>"])
```

## Features

- **Language agnostic**: Works with any language (English, Czech, German, etc.)
- **Fast inference**: Sub-millisecond predictions with quantized models
- **Multiclass support**: Configurable category set
- **Production-ready**: Includes quantized `.ftz` models for deployment
- **HTML parsing**: Uses selectolax for fast, robust HTML extraction
- **Noise filtering**: Automatically removes scripts, styles, and other noise

## Requirements

Dependencies are listed in `pyproject.toml`. Install with:

```bash
poetry install
```

Key dependencies:
- `fasttext-wheel` - FastText classifier
- `selectolax` - Fast HTML parser
- `pandas` - Data handling
- `scikit-learn` - Train/test split
- `tqdm` - Progress bars
