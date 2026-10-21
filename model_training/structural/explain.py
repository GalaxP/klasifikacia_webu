"""Turn a trained model's weights back into readable reasons.

Tokens are stored as 63-bit hashes (so the shipped model needs no text
vocabulary), which makes weights unreadable on their own.  `build_token_strings`
re-featurises a sample of pages from the source jsonl just to recover
hash -> token for the ids that made it into the vocabulary; the high-document-
frequency tokens reappear within a few hundred pages, so this stays out of the
training path.
"""

from __future__ import annotations

import json
import random
from multiprocessing import Pool

import numpy as np

from . import data as D
from . import features as F


def _init(groups):
    D._init(groups)


def _work(item):
    i, url, raw = item
    toks, _ = F.extract(raw, url, groups=D._GROUPS)
    return i, {g: list(dict.fromkeys(l)) for g, l in toks.items()}


def build_token_strings(clf, source: str, urls: list[str], sample: int = 2000,
                        workers: int = 8) -> dict[int, str]:
    """hash -> token string, for as many of the model's vocabulary ids as possible."""
    wanted = set(clf.vectorizer.vocab)
    rng = random.Random(0)
    take = set(rng.sample(range(len(urls)), min(sample, len(urls))))
    need = {urls[i] for i in take}

    items = []
    with open(source, encoding="utf-8") as f:
        for line in f:
            if not need:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("url") not in need:
                continue
            raw = D.decompress(rec.get("compressed_html") or "")
            if raw is None:
                continue
            need.discard(rec.get("url"))
            items.append((len(items), rec.get("url") or "", raw))

    out: dict[int, str] = {}
    with Pool(workers, initializer=_init, initargs=(tuple(clf.groups),)) as pool:
        for _, per_group in pool.imap(_work, items, chunksize=2):
            for lst in per_group.values():
                for tok in lst:
                    h = D.hash_token(tok)
                    if h in wanted and h not in out:
                        out[h] = tok
    return out


def top_features(clf, label: str, k: int = 20,
                 strings: dict[int, str] | None = None) -> list[tuple[str, float]]:
    """Highest-weight features of one label's one-vs-rest head."""
    j = clf.labels.index(label)
    if not hasattr(clf.model, "estimators_"):
        raise TypeError("linear model required to read coefficients")
    w = np.asarray(clf.model.estimators_[j].coef_[0]).ravel()
    n_tok = len(clf.vectorizer.vocab)
    inv = {v: h for h, v in clf.vectorizer.vocab.items()}
    out = []
    for col in np.argsort(-w)[:k]:
        col = int(col)
        if col < n_tok:
            h = inv[col]
            name = strings.get(h, f"#{h:x}") if strings else f"#{h:x}"
        else:
            name = "@" + clf.vectorizer.num_names[col - n_tok]
        out.append((name, float(w[col])))
    return out


def label_signature(clf, source: str, urls: list[str], labels=None,
                    k: int = 15, workers: int = 8) -> dict[str, list[tuple[str, float]]]:
    strings = build_token_strings(clf, source, urls, workers=workers)
    return {lab: top_features(clf, lab, k=k, strings=strings)
            for lab in (labels or clf.labels)}
