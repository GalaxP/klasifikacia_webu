#!/usr/bin/env python3
"""
Website Category Classifier

A language-agnostic classifier that categorizes websites based on their HTML source code.
Uses fastText for fast, accurate multiclass classification.

Supported categories:
    - Adult, Automotive, Computers, Entertainment, Finance, Food
    - Health, News, Shopping, Sports, Travel

Usage:
    # Train a new model:
    python scripts/website_classifier.py train --data dataset.jsonl --output models/my_classifier.bin

    # Classify a single HTML page:
    python scripts/website_classifier.py predict --html "<html>...</html>"

    # Classify from a file:
    python scripts/website_classifier.py predict-file --input html_list.txt --model models/my_classifier.bin

Categories are configurable via --categories flag (comma-separated).
"""

from __future__ import annotations

import os
import re
import sys
import json
import gzip
import base64
import argparse
import html as ihtml
from pathlib import Path
from typing import Optional

import numpy as np

# Fix for fastText compatibility with NumPy 2.x
# fastText calls np.array(..., copy=False) which fails on NumPy 2.x
try:
    import fasttext.FastText as _fasttext_module
    if hasattr(_fasttext_module.np, 'array'):
        _original_np_array = _fasttext_module.np.array

        def _np_array_compat(*args, **kwargs):
            kwargs.pop("copy", None)
            return _original_np_array(*args, **kwargs)

        _fasttext_module.np.array = _np_array_compat
except Exception:
    pass

import fasttext
import pandas as pd
from selectolax.parser import HTMLParser
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CATEGORIES = [
    "Adult",
    "Automotive",
    "Computers",
    "Entertainment",
    "Finance",
    "Food",
    "Health",
    "News",
    "Shopping",
    "Sports",
    "Travel",
]

# Regex patterns compiled once for performance
RE_WS = re.compile(r"\s+")
RE_BAD_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
RE_LABEL_SAFE = re.compile(r"[^A-Za-z0-9]+")


# =============================================================================
# Text Processing
# =============================================================================

def normalize_text(text: str | None) -> str:
    """Clean and normalize text for classification."""
    if not text:
        return ""
    text = ihtml.unescape(str(text))
    text = RE_BAD_CTRL.sub(" ", text)
    text = RE_WS.sub(" ", text).strip()
    return text


def safe_label(label: str, valid_categories: set[str]) -> str:
    """Convert a category label to fastText-safe format."""
    label = label.strip()
    if label not in valid_categories:
        raise ValueError(f"Unknown label '{label}'. Valid categories: {valid_categories}")
    return f"__label__{RE_LABEL_SAFE.sub('_', label)}"


def unsafify_label(ft_label: str) -> str:
    """Convert fastText label back to human-readable format."""
    return ft_label.replace("__label__", "").replace("_", " ")


def decompress_html(value: str) -> str:
    """Decompress base64+gzip encoded HTML if needed."""
    if not value or not isinstance(value, str):
        return ""

    value = value.strip()
    if not value:
        return ""

    # Already raw HTML
    if "<html" in value.lower() or "<body" in value.lower() or "<div" in value.lower():
        return value

    # Try base64 decode first
    try:
        decoded = base64.b64decode(value)
    except Exception:
        return value

    # Try gzip decompression
    try:
        return gzip.decompress(decoded).decode("utf-8", errors="ignore")
    except Exception:
        return decoded.decode("utf-8", errors="ignore")


# =============================================================================
# HTML Parsing & Feature Extraction
# =============================================================================

def extract_text_from_html(raw_html: str, max_body_chars: int = 4000) -> str:
    """
    Extract meaningful text from HTML for classification.

    Strategy:
    - Remove noisy tags (script, style, nav, footer, etc.)
    - Extract title, meta descriptions, Open Graph tags
    - Extract headings (h1, h2, h3) with weighting
    - Extract main body text
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""

    tree = HTMLParser(raw_html)
    if tree is None:
        return ""

    # Remove noisy elements
    for tag in ["script", "style", "noscript", "svg", "canvas", "iframe", "footer", "nav", "form", "aside"]:
        for node in tree.css(tag):
            node.decompose()

    parts = []

    # Title (high weight - repeat 3x)
    title = tree.css_first("title")
    if title:
        t = normalize_text(title.text())
        if t:
            parts.extend([t, t, t])

    # Meta description
    meta_desc = tree.css_first('meta[name="description"]')
    if meta_desc:
        v = normalize_text(meta_desc.attributes.get("content", ""))
        if v:
            parts.extend([v, v])

    # Open Graph tags
    og_title = tree.css_first('meta[property="og:title"]')
    if og_title:
        v = normalize_text(og_title.attributes.get("content", ""))
        if v:
            parts.extend([v, v])

    og_desc = tree.css_first('meta[property="og:description"]')
    if og_desc:
        v = normalize_text(og_desc.attributes.get("content", ""))
        if v:
            parts.append(v)

    # Headings
    for sel in ["h1", "h2", "h3"]:
        texts = []
        for node in tree.css(sel):
            txt = normalize_text(node.text())
            if txt:
                texts.append(txt)
        if texts:
            joined = " ".join(texts[:20])
            parts.extend([joined, joined])

    # Main body text
    body = tree.body
    if body:
        body_text = normalize_text(body.text(separator=" "))
        if body_text:
            parts.append(body_text[:max_body_chars])

    return normalize_text(" ".join(parts))


# =============================================================================
# Dataset Loading
# =============================================================================

def parse_label(value: object, category_lookup: dict[str, str]) -> Optional[str]:
    """Parse label from various formats (string, list, JSON)."""
    if value is None:
        return None

    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                candidates = parsed if isinstance(parsed, list) else [raw]
            except json.JSONDecodeError:
                candidates = re.split(r"[,;|]", raw)
        else:
            candidates = re.split(r"[,;|]", raw)
    else:
        candidates = [str(value)]

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if not key:
            continue
        mapped = category_lookup.get(key)
        if mapped:
            return mapped
    return None


def load_dataset(path: Path, valid_categories: set[str]) -> pd.DataFrame:
    """Load dataset from CSV or JSONL format."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_no}")
        df = pd.DataFrame(records)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .jsonl")

    # Handle different column schemas
    if {"html", "label"}.issubset(df.columns):
        pass
    elif {"compressed_html", "categories"}.issubset(df.columns):
        df = df.copy()
        df["html"] = df["compressed_html"].apply(decompress_html)
        df["label"] = df["categories"].apply(lambda x: parse_label(x, {c.lower(): c for c in valid_categories}))
    else:
        raise ValueError(f"Input must contain either (html,label) or (compressed_html,categories) columns. Found: {df.columns.tolist()}")

    # Clean data
    df = df.dropna(subset=["html", "label"]).copy()
    df["html"] = df["html"].astype(str)
    df["label"] = df["label"].apply(lambda x: parse_label(x, {c.lower(): c for c in valid_categories}))
    df = df.dropna(subset=["label"]).copy()

    # Validate labels
    bad_labels = sorted(set(df["label"]) - valid_categories)
    if bad_labels:
        raise ValueError(f"Found unexpected labels in dataset: {bad_labels}. Valid: {valid_categories}")

    return df


# =============================================================================
# Training Pipeline
# =============================================================================

def build_ft_line(html_text: str, label: str, valid_categories: set[str]) -> str:
    """Convert HTML + label into fastText training format."""
    text = extract_text_from_html(html_text)
    text = normalize_text(text.lower())
    return f"{safe_label(label, valid_categories)} {text}"


def prepare_training_data(df: pd.DataFrame, output_dir: Path, valid_categories: set[str], test_size: float = 0.15) -> tuple[Path, Path]:
    """Split data and prepare fastText training files."""
    from sklearn.model_selection import train_test_split

    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"],
    )

    print(f"Train: {len(train_df)}, Validation: {len(valid_df)}")

    # Write training files
    train_txt = output_dir / "train.ft.txt"
    valid_txt = output_dir / "valid.ft.txt"

    lines = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
        lines.append(build_ft_line(row["html"], row["label"], valid_categories))
    train_txt.write_text("\n".join(lines), encoding="utf-8")

    lines = []
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Processing validation data"):
        lines.append(build_ft_line(row["html"], row["label"], valid_categories))
    valid_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {train_txt} ({train_txt.stat().st_size:,} bytes)")
    print(f"Saved: {valid_txt} ({valid_txt.stat().st_size:,} bytes)")

    return train_txt, valid_txt


def train_model(
    data_path: Path,
    output_path: Path,
    valid_categories: list[str],
    quantize: bool = True,
    epochs: int = 20,
    lr: float = 0.5,
    dim: int = 64,
    word_ngrams: int = 2,
    minn: int = 2,
    maxn: int = 5,
) -> fasttext.FastText._FastText:
    """Train a fastText website classifier."""

    valid_categories_set = set(valid_categories)
    output_dir = output_path.parent

    print(f"Loading dataset from {data_path}...")
    df = load_dataset(data_path, valid_categories_set)
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    print("\nPreparing training data...")
    train_txt, valid_txt = prepare_training_data(df, output_dir, valid_categories_set)

    print(f"\nTraining model (epochs={epochs}, dim={dim}, lr={lr})...")
    model = fasttext.train_supervised(
        input=str(train_txt),
        lr=lr,
        epoch=epochs,
        wordNgrams=word_ngrams,
        dim=dim,
        minn=minn,
        maxn=maxn,
        bucket=2_000_000,
        loss="softmax",
        thread=os.cpu_count() or 4,
    )

    # Save base model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"Saved model: {output_path}")

    # Evaluate
    result = model.test(str(valid_txt))
    print(f"\nValidation Results:")
    print(f"  Samples: {result[0]}")
    print(f"  Precision@1: {result[1]:.4f}")
    print(f"  Recall@1: {result[2]:.4f}")

    # Quantize for deployment
    if quantize:
        print("\nQuantizing model for faster inference...")
        quant_path = output_path.with_suffix(".ftz")
        model.quantize(
            input=str(train_txt),
            retrain=True,
            cutoff=50_000,
            thread=os.cpu_count() or 4,
        )
        model.save_model(str(quant_path))
        print(f"Saved quantized model: {quant_path}")

        # Re-load quantized model for final evaluation
        model = fasttext.load_model(str(quant_path))
        result = model.test(str(valid_txt))
        print(f"\nQuantized Model Validation:")
        print(f"  Precision@1: {result[1]:.4f}")
        print(f"  Recall@1: {result[2]:.4f}")

    return model


# =============================================================================
# Inference
# =============================================================================

class WebsiteClassifier:
    """Wrapper class for easy model deployment."""

    def __init__(self, model_path: Path):
        """Load a trained fastText model."""
        self.model = fasttext.load_model(str(model_path))
        self.categories = DEFAULT_CATEGORIES

    def predict(self, raw_html: str, k: int = 3) -> list[dict]:
        """
        Predict category for a single HTML page.

        Args:
            raw_html: Raw HTML source code
            k: Number of top predictions to return

        Returns:
            List of dicts with 'label' and 'score' keys
        """
        text = extract_text_from_html(raw_html)
        text = normalize_text(text.lower())

        labels, probs = self.model.predict(text, k=k)

        return [
            {"label": unsafify_label(lbl), "score": float(prob)}
            for lbl, prob in zip(labels, probs)
        ]

    def predict_batch(self, html_list: list[str], k: int = 3) -> list[list[dict]]:
        """Predict categories for multiple HTML pages."""
        texts = [normalize_text(extract_text_from_html(h).lower()) for h in html_list]
        labels, probs = self.model.predict(texts, k=k)

        results = []
        for lbl_group, prob_group in zip(labels, probs):
            results.append([
                {"label": unsafify_label(lbl), "score": float(prob)}
                for lbl, prob in zip(lbl_group, prob_group)
            ])
        return results


def classify_html(raw_html: str, model_path: Path, k: int = 3) -> dict:
    """Classify a single HTML page and return structured results."""
    classifier = WebsiteClassifier(model_path)
    top_k = classifier.predict(raw_html, k=k)

    return {
        "top_prediction": top_k[0]["label"],
        "confidence": top_k[0]["score"],
        "all_predictions": top_k,
    }


# =============================================================================
# CLI Interface
# =============================================================================

def cmd_train(args):
    """Handle train command."""
    valid_categories = args.categories.split(",") if args.categories else DEFAULT_CATEGORIES

    train_model(
        data_path=Path(args.data),
        output_path=Path(args.output),
        valid_categories=valid_categories,
        quantize=not args.no_quantize,
        epochs=args.epochs,
        lr=args.lr,
        dim=args.dim,
    )


def cmd_predict(args):
    """Handle predict command."""
    # Read HTML from file if --input is provided, otherwise use --html
    if args.input:
        raw_html = Path(args.input).read_text(encoding="utf-8")
    else:
        raw_html = args.html

    result = classify_html(raw_html, Path(args.model), k=args.top_k)
    print(json.dumps(result, indent=2))


def cmd_predict_file(args):
    """Handle predict-file command."""
    classifier = WebsiteClassifier(Path(args.model))

    html_list = [line.strip() for line in open(args.input, "r", encoding="utf-8") if line.strip()]
    results = classifier.predict_batch(html_list, k=args.top_k)

    output = []
    for html, preds in zip(html_list, results):
        output.append({
            "html_preview": html[:200] + "..." if len(html) > 200 else html,
            "predictions": preds,
        })

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Website Category Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model:
  python scripts/website_classifier.py train --data dataset.jsonl --output model.bin

  # Classify a single page (inline HTML):
  python scripts/website_classifier.py predict --html "<html>...</html>" --model model.bin

  # Classify from file:
  python scripts/website_classifier.py predict --input page.html --model model.bin

  # Custom categories:
  python scripts/website_classifier.py train --data data.csv --output model.bin --categories "Tech,Business,Sports"
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new classifier")
    train_parser.add_argument("--data", required=True, help="Path to training data (CSV or JSONL)")
    train_parser.add_argument("--output", required=True, help="Output path for trained model")
    train_parser.add_argument("--categories", default=None, help="Comma-separated list of categories")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    train_parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    train_parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    train_parser.set_defaults(func=cmd_train)

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Classify a single HTML page")
    input_group = pred_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--html", help="Raw HTML content (inline)")
    input_group.add_argument("--input", help="Path to file containing HTML content")
    pred_parser.add_argument("--model", required=True, help="Path to trained model")
    pred_parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    pred_parser.set_defaults(func=cmd_predict)

    # Predict-file command
    pred_file_parser = subparsers.add_parser("predict-file", help="Classify HTML pages from file")
    pred_file_parser.add_argument("--input", required=True, help="Path to file with HTML content (one per line)")
    pred_file_parser.add_argument("--model", required=True, help="Path to trained model")
    pred_file_parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    pred_file_parser.set_defaults(func=cmd_predict_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
