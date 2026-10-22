"""Language-agnostic *content* signal: multilingual sentence embeddings.

The structural features throw the words away on purpose, which is why they cannot see
what a page is about (`Food` 0.30, `Travel` 0.25).  A multilingual sentence encoder
supplies that signal without tying the model to a language: it maps text from 100+
languages into one vector space trained so that translations land on top of each other
(measured on this corpus: Czech/English sentence pairs 0.84-0.95 cosine, an unrelated
pair 0.11).  Nothing here holds a per-language vocabulary -- the only language
knowledge lives in the encoder's weights, and it is the same weights for every page.

Run once, then everything else treats `embed` as just another feature group:

    python -m structural.embed --data ../27_08_01_en_cz.jsonl
    python -m structural.train train --data ../27_08_01_en_cz.jsonl \\
        --groups tag,attr,href,meta,plat,css,res,embed
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import features as F

CACHE_DIR = Path(__file__).resolve().parent / "cache"

EMBED_GROUP = "embed"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#: windows sampled per page and averaged -- the top of a page is navigation, so one
#: window from the beginning describes a portal, not its subject
CHUNKS = 4
CHUNK_CHARS = 1000


# --------------------------------------------------------------------------- #
# text for the encoder
# --------------------------------------------------------------------------- #

def page_text(doc) -> str:
    """Title + meta description + visible body text, in reading order."""
    head = []
    title = doc.find(".//title")
    if title is not None and (title.text or "").strip():
        head.append(title.text.strip())
    for m in doc.iter("meta"):
        if (m.get("name") or "").lower() in ("description", "og:description", "keywords"):
            c = (m.get("content") or "").strip()
            if c:
                head.append(c[:300])
                break
    body = F.visible_text(doc)
    txt = " ".join(head + [body])
    return " ".join(txt.split())[:CHUNKS * CHUNK_CHARS * 2]


def chunks_of(text: str, n: int = CHUNKS) -> list[str]:
    """Evenly spaced windows over the text; short pages just repeat their start."""
    if not text:
        return [""]
    if len(text) <= CHUNK_CHARS:
        return [text]
    span = len(text) - CHUNK_CHARS
    step = span / (n - 1) if n > 1 else 0.0
    return [text[int(i * step):int(i * step) + CHUNK_CHARS] for i in range(n)]


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #

_ENCODERS: dict[str, object] = {}


def get_encoder(model_name: str = DEFAULT_MODEL):
    """Lazy, cached: keeps the structural-only bundle loadable without torch.

    The download/load progress bars and weight-loading chatter go to stderr, which
    would otherwise mix into `predict`'s output, so they are muted before the import.
    """
    if model_name not in _ENCODERS:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer
        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]


def encode(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 32,
           verbose: bool = True) -> np.ndarray:
    """One L2-normalised vector per *page*: the averaged windows of its text."""
    per_page = [chunks_of(t) for t in texts]
    flat = [c for cs in per_page for c in cs]
    t0 = time.time()
    enc = get_encoder(model_name)
    out = enc.encode(flat, batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    out = np.asarray(out, dtype=np.float32)
    if verbose:
        print(f"[embed] {len(flat)} chunks in {time.time()-t0:.0f}s "
              f"({len(texts) / max(time.time()-t0, 1e-6) * 60:.0f} pages/min)", flush=True)
    # mean over the windows of each page, then renormalise
    dims = [len(cs) for cs in per_page]
    vecs = np.zeros((len(texts), out.shape[1]), dtype=np.float32)
    off = 0
    for i, d in enumerate(dims):
        vecs[i] = out[off:off + d].mean(axis=0)
        off += d
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms == 0] = 1.0
    return vecs / norms[:, None]


def embed_page(raw_html: bytes | str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Vector for a single live page -- the serving path of `clf.predict`."""
    raw = raw_html.encode("utf-8", "replace") if isinstance(raw_html, str) else raw_html
    doc = F.parse_tree(raw)
    text = "" if doc is None else page_text(doc)
    return encode([text], model_name=model_name, verbose=False)[0]


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #

def cache_file(data_path: Path, n_docs: int | None,
               model_name: str = DEFAULT_MODEL) -> Path:
    key = hashlib.md5(f"{model_name}|{n_docs}".encode()).hexdigest()[:8]
    return CACHE_DIR / f"embed_{Path(data_path).stem}_{key}.npz"


def build(data_path: str | Path, n_docs: int | None = None,
          model_name: str = DEFAULT_MODEL, batch_size: int = 32,
          force: bool = False) -> Path:
    """Embed every page the featuriser keeps, in the same order (see `D._work`:
    decodable bytes and a parseable tree)."""
    out = cache_file(data_path, n_docs, model_name)
    out.parent.mkdir(exist_ok=True)
    if out.exists() and not force:
        print(f"[embed] already built {out.name}")
        return out

    # identical iteration *and* identical keep-rule to data._work, so row i here is
    # row i of the Dataset
    items = []
    for i, url, _cats, raw in D._iter_raw(Path(data_path), n_docs):
        if raw is None:
            continue
        doc = F.parse_tree(raw)
        if doc is None:
            continue
        items.append((i, url, page_text(doc)))
    print(f"[embed] {len(items)} pages to embed from {Path(data_path).name}", flush=True)

    vecs = encode([t for _, _, t in items], model_name=model_name, batch_size=batch_size)
    np.savez_compressed(out, vec=vecs, urls=np.asarray([u for _, u, _ in items]),
                        model=np.asarray(model_name))
    print(f"[embed] wrote {out.name} {vecs.shape} "
          f"({out.stat().st_size / 1e6:.0f} MB)")
    return out


def attach(ds, data_path: str | Path, n_docs: int | None = None,
           model_name: str = DEFAULT_MODEL, verbose: bool = True) -> bool:
    """Load the cached matrix into `ds.embeds`, aligned to `ds.urls`.  False if absent."""
    f = cache_file(data_path, n_docs, model_name)
    if not f.exists():
        return False
    with np.load(f, allow_pickle=False) as z:
        vec, urls, stored = z["vec"], z["urls"].tolist(), str(z["model"])
    if stored != model_name:
        raise ValueError(f"embedding cache was built with {stored}, not {model_name}")
    if urls == list(ds.urls):
        ds.embeds = vec
    else:                                   # fall back to matching by url
        by_url = dict(zip(urls, range(len(urls))))
        take = [by_url.get(u, -1) for u in ds.urls]
        miss = sum(1 for j in take if j < 0)
        if miss > 0.02 * len(take):
            raise ValueError(f"embedding cache is stale: {miss}/{len(take)} pages missing")
        keep = np.arange(len(take))[np.asarray(take) >= 0]
        ds.embeds = np.zeros((len(take), vec.shape[1]), dtype=np.float32)
        ds.embeds[keep] = vec[[take[j] for j in keep]]
        if verbose:
            print(f"[embed] aligned by url ({miss} pages left without a vector)")
    if verbose:
        print(f"[embed] attached {ds.embeds.shape} from {f.name}")
    return True


# --------------------------------------------------------------------------- #

def main(argv=None):
    # Czech class names and sample sentences must survive a cp1252 Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../27_08_01_en_cz.jsonl")
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="just report the cross-language alignment of the encoder")
    a = ap.parse_args(argv)

    if a.check:
        # is the encoder actually cross-lingual?  two translations should score far
        # above an unrelated pair -- the whole premise of this feature group
        pairs = [("Recepty na rychlé večerní jídlo z těstovin",
                  "Pasta recipes for a quick dinner"),
                 ("Cena vozu a spotřeba paliva", "Car price and fuel consumption"),
                 ("Recepty na rychlé večerní jídlo z těstovin",
                  "Football match results and the league table")]
        emb = get_encoder(a.model).encode([s for p in pairs for s in p],
                                         normalize_embeddings=True)
        for i, (x, y) in enumerate(pairs):
            print(f"  {float(emb[2 * i] @ emb[2 * i + 1]):+.3f}  {x[:34]!r} ~ {y[:34]!r}")
        return 0

    build(a.data, n_docs=a.n_docs, model_name=a.model, batch_size=a.batch_size,
          force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
