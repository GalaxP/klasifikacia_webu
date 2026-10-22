"""Vectorizer + multilabel classifier + metrics.

The feature space is hash-keyed, so `Vectorizer.transform` needs no stored text
vocabulary: it maps 63-bit token ids to columns.  Only the *selected* ids are kept
in the fitted bundle, which keeps the saved model small and lets it score a page
it has never seen without any corpus around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, hamming_loss, precision_score,
                             recall_score, roc_auc_score)

from . import features as F
from .embed import EMBED_GROUP


# --------------------------------------------------------------------------- #
# vectorizer
# --------------------------------------------------------------------------- #

@dataclass
class Vectorizer:
    """TF-IDF over hashed structural tokens + a standardised dense block."""

    groups: list[str] = field(default_factory=list)
    min_df: int = 4
    max_df: float = 0.75
    max_features: int | None = 200_000
    sublinear_tf: bool = True
    binary_tf: bool = False
    num_scale: float = 0.5
    use_numeric: bool = True
    #: `embed` in `groups` adds the multilingual text block; the model name is stored so
    #: a saved bundle can rebuild the vector for a page it has never seen
    embed_model: str | None = None
    embed_svd: int = 128
    embed_scale: float = 1.0

    vocab: dict[int, int] = field(default_factory=dict)   # token hash -> column
    idf: np.ndarray | None = None
    num_names: list[str] = field(default_factory=list)
    num_mu: np.ndarray | None = None
    num_sd: np.ndarray | None = None
    emb_proj: object | None = None
    emb_mu: np.ndarray | None = None
    emb_sd: np.ndarray | None = None
    emb_dim_: int = 0
    n_cols_: int = 0

    # --------------------------------------------------------------- blocks --
    @property
    def uses_embed(self) -> bool:
        return EMBED_GROUP in self.groups

    def token_groups(self) -> list[str]:
        """Everything except the dense embedding block."""
        return [g for g in self.groups if g != EMBED_GROUP]

    @staticmethod
    def embed_matrix(ds) -> np.ndarray:
        if getattr(ds, "embeds", None) is None:      # getattr: older feature caches
            raise ValueError(
                "dataset carries no text embeddings -- build them with "
                "`python -m structural.embed --data <jsonl>` and load the data with "
                "`embed_model=...`")
        return ds.embeds

    def _fit_embed(self, E: np.ndarray) -> None:
        """Compress + equalise the embedding block.

        The raw 384 dims are far denser in information than any structural token, and
        their variance is wildly uneven across dimensions, so: truncated SVD to a
        manageable width, then standardise per component -- exactly what the skewed
        numeric block already gets.  `embed_scale` sets how loud the block is against
        thousands of tf-idf columns, since rows are L2-normalised after concatenation.
        """
        self.emb_proj = None
        P = E
        if self.embed_svd and 0 < self.embed_svd < E.shape[1]:
            from sklearn.decomposition import TruncatedSVD
            self.emb_proj = TruncatedSVD(n_components=self.embed_svd,
                                         random_state=0).fit(E)
            P = self.emb_proj.transform(E)
        self.emb_dim_ = int(P.shape[1])
        self.emb_mu = P.mean(0)
        self.emb_sd = P.std(0)
        self.emb_sd[self.emb_sd < 1e-6] = 1.0

    def _embed_block(self, E: np.ndarray) -> np.ndarray:
        P = self.emb_proj.transform(E) if self.emb_proj is not None else np.asarray(E)
        return np.clip((P - self.emb_mu) / self.emb_sd, -8, 8) * self.embed_scale

    # ---------------------------------------------------------------- fit ----
    def fit(self, ds, idx) -> "Vectorizer":
        idx = np.asarray(idx)
        n = len(idx)
        df: dict[int, int] = {}
        for g in self.token_groups():
            k, v, p = ds.tokens[g]
            for i in idx:
                for h in k[p[i]:p[i + 1]]:
                    df[int(h)] = df.get(int(h), 0) + 1
        hi = max(self.min_df, int(n * self.max_df)) if self.max_df < 1 else n
        kept = [(h, c) for h, c in df.items() if self.min_df <= c <= hi]
        kept.sort(key=lambda kv: -kv[1])
        if self.max_features:
            kept = kept[: self.max_features]
        self.vocab = {h: j for j, (h, c) in enumerate(kept)}
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for h, c in kept:
            idf[self.vocab[h]] = np.log((1 + n) / (1 + c)) + 1.0
        self.idf = idf

        if self.use_numeric:
            self.num_names = list(ds.num_names)
            X = ds.stats[idx]
            with np.errstate(all="ignore"):
                Xs = np.sign(X) * np.log1p(np.abs(X))   # features are heavily skewed
            self.num_mu = Xs.mean(0)
            self.num_sd = Xs.std(0)
            self.num_sd[self.num_sd < 1e-6] = 1.0
        if self.uses_embed:
            self._fit_embed(self.embed_matrix(ds)[idx])
        self.n_cols_ = (len(self.vocab)
                        + (len(self.num_names) if self.use_numeric else 0)
                        + (self.emb_dim_ if self.uses_embed else 0))
        return self

    # ---------------------------------------------------------- transform ----
    def transform(self, ds, idx) -> sp.csr_matrix:
        idx = np.asarray(idx)
        k_parts, c_parts, rows = [], [], []
        row = 0
        for i in idx:
            cols_local = []
            vals = []
            for g in self.token_groups():
                k, v, p = ds.tokens[g]
                hs = k[p[i]:p[i + 1]]
                cnts = v[p[i]:p[i + 1]]
                for h, c in zip(hs.tolist(), cnts.tolist()):
                    col = self.vocab.get(h)
                    if col is None:
                        continue
                    cols_local.append(col)
                    vals.append(c)
            k_parts.append(cols_local)
            c_parts.append(vals)
            rows.append([row] * len(cols_local))
            row += 1
        rows_np = np.concatenate([np.asarray(r) for r in rows]) if row else np.zeros(0)
        col_np = np.concatenate([np.asarray(x, np.int32) for x in k_parts]) if k_parts else np.zeros(0, np.int32)
        val_np = np.concatenate([np.asarray(x, np.float32) for x in c_parts]) if c_parts else np.zeros(0, np.float32)
        X = sp.coo_matrix((val_np, (rows_np, col_np)), shape=(len(idx), len(self.vocab)),
                          dtype=np.float32).tocsr()
        if self.binary_tf:
            X.data = np.ones_like(X.data)
        elif self.sublinear_tf:
            X.data = 1.0 + np.log(np.maximum(X.data, 1.0))
        X = X.multiply(self.idf[None, :]).tocsr()

        if self.use_numeric and self.num_names:
            Z = ds.stats[idx]
            Z = np.sign(Z) * np.log1p(np.abs(Z))
            Z = (Z - self.num_mu) / self.num_sd
            Z = np.clip(Z, -8, 8)
            # scale the dense block so it is competitive with, not swamped by,
            # thousands of tf-idf columns
            Zs = sp.csr_matrix(Z.astype(np.float32) * self.num_scale)
            X = sp.hstack([X, Zs]).tocsr()
        if self.uses_embed:
            E = sp.csr_matrix(np.asarray(self._embed_block(self.embed_matrix(ds)[idx]),
                                         dtype=np.float32))
            X = sp.hstack([X, E]).tocsr()
        return _l2(X)

    # ------------------------------------------------------------- infer ----
    def transform_tokens(self, toks: dict[str, list[str]], stats: dict[str, float],
                         embed: np.ndarray | None = None) -> sp.csr_matrix:
        """Score a single unseen page from freshly extracted features.

        `embed` is required when the model was fitted with the `embed` group; get it
        from `embed.embed_page(html, self.embed_model)`.
        """
        from .data import hash_token
        from collections import Counter
        cols, vals = [], []
        for g in self.token_groups():
            for tok, cnt in Counter(toks.get(g, [])).items():
                col = self.vocab.get(hash_token(tok))
                if col is not None:
                    cols.append(col)
                    vals.append(float(cnt))
        if not cols:  # keep the row valid even for an empty page
            cols = [0]
            vals = [0.0]
        X = sp.coo_matrix((vals, (np.zeros(len(cols), int), np.asarray(cols))),
                          shape=(1, len(self.vocab)), dtype=np.float32).tocsr()
        if self.binary_tf:
            X.data = np.ones_like(X.data)
        else:
            X.data = 1.0 + np.log(np.maximum(X.data, 1.0))
        X = X.multiply(self.idf[None, :]).tocsr()
        if self.use_numeric and self.num_names:
            Z = np.array([[stats.get(nm, 0.0) for nm in self.num_names]], dtype=np.float32)
            Z = np.sign(Z) * np.log1p(np.abs(Z))
            Z = (Z - self.num_mu) / self.num_sd
            Z = sp.csr_matrix(np.clip(Z, -8, 8) * self.num_scale)
            X = sp.hstack([X, Z]).tocsr()
        if self.uses_embed:
            if embed is None:
                raise ValueError(
                    f"this model was fitted with the `embed` group, so scoring a page "
                    f"needs its text embedding; build it with "
                    f"`structural.embed.embed_page(html, {self.embed_model!r})`")
            v = np.asarray(embed, dtype=np.float32).ravel()
            want = (int(self.emb_proj.n_features_in_) if self.emb_proj is not None
                    else self.emb_dim_)
            if v.size != want:
                raise ValueError(f"embedding has {v.size} dims, model expects {want}")
            X = sp.hstack([X, sp.csr_matrix(self._embed_block(v[None, :]))]).tocsr()
        return _l2(X)


def _l2(X: sp.csr_matrix) -> sp.csr_matrix:
    """Unit rows, so page length does not become a feature of its own."""
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).dot(X).tocsr()


def binarize(label_lists, classes) -> "np.ndarray":
    """Multi-label indicator matrix. Unlike MultiLabelBinarizer it never forgets
    a class that happens to be absent from the slice being encoded."""
    classes = list(classes)
    ix = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((len(label_lists), len(classes)), dtype=np.int8)
    for i, labs in enumerate(label_lists):
        for l in labs:
            j = ix.get(l)
            if j is not None:
                Y[i, j] = 1
    return Y


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #

def _fit_binary(est, X, y):
    """Fit one label head; guards the case where the label never occurs."""
    if y.sum() == 0:
        return _NullHead()
    est.fit(X, y)
    return est


def _proba(est, X):
    return np.asarray(est.predict_proba(X))[:, 1]


class _NullHead:
    """A head for a label that was absent from the training slice."""

    n_features_in_ = 0

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.ones(n), np.zeros(n)])

    @property
    def coef_(self):
        return [np.zeros(0)]


class OvR:
    """One-vs-rest that avoids sklearn's memmap path.

    `OneVsRestClassifier(n_jobs=-1)` memmaps the design matrix read-only into the
    workers, which liblinear rejects with `WRITEBACKIFCOPY base is read-only`.
    `max_nbytes=None` keeps the array in the pickled payload instead.
    """

    def __init__(self, template, n_jobs: int = -1):
        self.template = template
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        from joblib import Parallel, delayed
        from sklearn.base import clone
        self.estimators_ = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(_fit_binary)(clone(self.template), X, np.asarray(Y[:, j]).ravel())
            for j in range(Y.shape[1]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        from joblib import Parallel, delayed
        from sklearn.base import is_classifier
        cols = Parallel(n_jobs=self.n_jobs if X.shape[0] > 200 else 1, max_nbytes=None)(
            delayed(_proba)(e, X) for e in self.estimators_)
        return np.column_stack(cols)

    def decision_function(self, X) -> np.ndarray:
        return self.predict_proba(X)


def fit_model(X, Y, kind="logreg", C=4.0, class_weight="balanced", seed=0, n_jobs=-1):
    if kind == "logreg":
        template = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000,
                                      solver="liblinear", random_state=seed)
        return OvR(template, n_jobs=n_jobs).fit(X, Y)
    if kind == "lgbm":
        import lightgbm as lgb
        from joblib import Parallel, delayed
        template = lgb.LGBMClassifier(objective="binary", n_estimators=500,
                                      learning_rate=0.08, num_leaves=63, subsample=0.8,
                                      colsample_bytree=0.4, class_weight="balanced",
                                      n_jobs=1, verbose=-1, random_state=seed)
        clfs = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_fit_binary)(clone(template), X, np.asarray(Y[:, j]).ravel())
            for j in range(Y.shape[1]))
        return _Fitted(clfs)
    raise ValueError(kind)


class _Fitted:
    """Container for pre-fitted LightGBM heads (same interface as OvR)."""

    def __init__(self, estimators_):
        self.estimators_ = estimators_

    def predict_proba(self, X) -> np.ndarray:
        return np.column_stack([np.asarray(e.predict_proba(X))[:, 1]
                                for e in self.estimators_])

    decision_function = predict_proba


# --------------------------------------------------------------------------- #
# thresholds + metrics
# --------------------------------------------------------------------------- #

def tune_thresholds(scores, Y, objective="macro", grid=None, score_norm="none"):
    """Per-label threshold maximising the chosen F1 variant on the tuning set."""
    if grid is None:
        grid = (np.arange(-1.0, 3.001, 0.05) if score_norm == "doc_z"
                else np.arange(0.02, 0.981, 0.01))
    k = Y.shape[1]
    best = np.full(k, 0.5)
    if objective == "micro":
        # micro F1 is maximised by a shared threshold shift; sweep globally then
        # polish per label
        g0 = max(grid, key=lambda t: f1_score(Y, scores >= t, average="micro", zero_division=0))
        best[:] = g0
        return best
    for j in range(k):
        yj = Y[:, j]
        if yj.sum() == 0:
            continue
        f1s = [f1_score(yj, scores[:, j] >= t, zero_division=0) for t in grid]
        best[j] = grid[int(np.argmax(f1s))]
    return best


def prune_labels(Y, min_count=20):
    """Drop labels that are too rare to fit a usable one-vs-rest head."""
    counts = np.asarray(Y.sum(0)).ravel()
    return np.flatnonzero(counts >= min_count), counts


def report(Y_true, Y_pred, labels) -> dict:
    out = {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(Y_true, Y_pred, average="samples", zero_division=0),
        "exact_match": float((Y_pred == Y_true).all(1).mean()),
        "hamming": hamming_loss(Y_true, Y_pred),
        "empty_pred": float((Y_pred.sum(1) == 0).mean()),
    }
    prec = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    rec = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    out["per_class"] = {
        labels[j]: {"p": round(float(prec[j]), 3), "r": round(float(rec[j]), 3),
                    "f1": round(float(f1[j]), 3), "n": int(Y_true[:, j].sum())}
        for j in range(len(labels))
    }
    return out


def report_scores(Y_true, scores, labels) -> dict:
    out = {}
    try:
        out["labeled_mean_auc"] = float(roc_auc_score(Y_true, scores, average=None).mean())
    except ValueError:
        pass
    return out
