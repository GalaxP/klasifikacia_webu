"""Dataset loading, feature caching, splits.

The jsonl records look like `{"url": str, "categories": [str], "compressed_html":
base64(gzip(html))}`.  Decompressing + featurising 8k pages takes a couple of
minutes, so results are cached to `cache/<stem>_<hash>.joblib` and reused.

Tokens are stored as 63-bit hashes rather than strings: the feature space is then
vocabulary-free, which means a saved model needs no text corpus to score a new
page -- just its HTML.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from . import features as F

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# characters that essentially only occur in Czech/Slovak text -- used to label
# rows for the cross-lingual evaluation split, never fed to the model
_CS_CHARS = "ěščřžůďťňľĺŕÄ"
_CS_RE = re.compile("[" + _CS_CHARS + _CS_CHARS.upper() + "]")
_CS_HINT = 0.004


def hash_token(tok: str) -> int:
    """Stable 63-bit token id -- no vocabulary needed at inference time."""
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(),
                          "big") & 0x7FFF_FFFF_FFFF_FFFF


def decompress(b64: str) -> bytes | None:
    try:
        return gzip.decompress(base64.b64decode(b64, validate=False))
    except Exception:
        try:
            return zlib.decompress(base64.b64decode(b64, validate=False))
        except Exception:
            return None


def registrable_domain(url: str) -> str:
    """Last two host labels -- good enough to keep one site in one split."""
    try:
        host = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower()
    except ValueError:
        host = ""
    parts = [p for p in host.split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def detect_lang(raw: bytes, url: str) -> str:
    """Coarse cs/en label, used only for the cross-lingual *evaluation* split."""
    if (urlsplit(url if "//" in url else "//" + url).hostname or "").lower().endswith(".cz"):
        return "cs"
    try:
        text = F.decode_html(raw[:200_000])
    except Exception:
        return "en"
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 200:
        return "cs" if ".cz" in url else "en"
    return "cs" if len(_CS_RE.findall(text)) / letters > _CS_HINT else "en"


# --------------------------------------------------------------------------- #
# parallel worker
# --------------------------------------------------------------------------- #

_GROUPS: tuple = F.STRUCT_GROUPS


def _init(groups):
    global _GROUPS
    _GROUPS = tuple(groups)


def _work(item):
    i, url, cats, raw = item
    if raw is None:
        return None
    toks, stats = F.extract(raw, url, groups=_GROUPS)
    if not stats:
        return None
    per_group = {}
    for g in _GROUPS:
        lst = toks.get(g) or []
        if not lst:
            per_group[g] = (np.zeros(0, np.uint64), np.zeros(0, np.int32))
            continue
        c = Counter(lst)
        keys = np.fromiter((hash_token(k) for k in c), dtype=np.uint64, count=len(c))
        vals = np.fromiter((min(c[k], 32767) for k in c), dtype=np.int32, count=len(c))
        per_group[g] = (keys, vals)
    row = np.zeros(len(F._NUM_ORDER), dtype=np.float32)
    for j, name in enumerate(F._NUM_ORDER):
        v = stats.get(name, 0.0)
        row[j] = v if v == v else 0.0
    return (i, url, cats, detect_lang(raw, url), registrable_domain(url), per_group, row)


# --------------------------------------------------------------------------- #
# dataset container
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    urls: list[str]
    labels: list[list[str]]
    langs: list[str]
    domains: list[str]
    groups: list[str]
    tokens: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # keys, counts, indptr
    stats: np.ndarray                       # (n_docs, len(F._NUM_ORDER))
    num_names: list[str] = field(default_factory=lambda: list(F._NUM_ORDER))

    def __len__(self):
        return len(self.urls)

    def doc_token_counts(self, i: int, group: str) -> dict[int, int]:
        k, v, p = self.tokens[group]
        return dict(zip(k[p[i]:p[i + 1]].tolist(), v[p[i]:p[i + 1]].tolist()))

# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(path: str | Path, groups=F.STRUCT_GROUPS, n_docs: int | None = None,
         workers: int | None = None, cache: bool = True, verbose: bool = True) -> Dataset:
    path = Path(path)
    groups = tuple(groups)
    key = hashlib.md5(("|".join(sorted(groups)) + f"|{n_docs}|" + F.SCHEMA).encode()).hexdigest()[:8]
    cache_file = CACHE_DIR / f"{path.stem}_{key}.joblib"
    if cache and cache_file.exists():
        import joblib
        if verbose:
            print(f"[data] loading cached features {cache_file.name}")
        return joblib.load(cache_file)

    import joblib
    workers = workers or max(1, (os.cpu_count() or 4) - 1)

    rows = []
    with Pool(workers, initializer=_init, initargs=(groups,)) as pool:
        for n, out in enumerate(pool.imap(_work, _iter_raw(path, n_docs), chunksize=2)):
            if out is not None:
                rows.append(out)
            if verbose and (n + 1) % 1000 == 0:
                print(f"       {n + 1} docs featurised", flush=True)

    rows.sort(key=lambda r: r[0])
    print(f"[data] kept {len(rows)} pages")

    tokens: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for g in groups:
        ks, vs, ptr = [], [], [0]
        off = 0
        for r in rows:
            k, v = r[5][g]
            ks.append(k)
            vs.append(v)
            off += len(k)
            ptr.append(off)
        tokens[g] = (np.concatenate(ks) if ks else np.zeros(0, np.uint64),
                     np.concatenate(vs) if vs else np.zeros(0, np.int32),
                     np.asarray(ptr, np.int64))

    ds = Dataset([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                 [r[4] for r in rows], list(groups), tokens,
                 np.stack([r[6] for r in rows]) if rows else np.zeros((0, len(F._NUM_ORDER))))
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        joblib.dump(ds, cache_file, compress=1)
        if verbose:
            print(f"[data] wrote cache {cache_file.name} ({cache_file.stat().st_size/1e6:.0f} MB)")
    return ds


def _iter_raw(path: Path, n_docs: int | None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_docs and i >= n_docs:
                return
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield (i, d.get("url") or "", d.get("categories") or [],
                   decompress(d.get("compressed_html") or ""))
