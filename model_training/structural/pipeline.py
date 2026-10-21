"""End-to-end classifier: raw HTML in, topics out.

    clf = StructuralWebsiteClassifier.load("models/structural_v1.joblib")
    clf.predict(html_bytes, url="https://example.cz/")   # -> [('Shopping', 0.93), ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as F
from .model import Vectorizer, fit_model, report, tune_thresholds


@dataclass
class StructuralWebsiteClassifier:
    labels: list[str]
    vectorizer: Vectorizer
    model: object
    thresholds: np.ndarray
    groups: list[str]
    score_norm: str = "none"
    force_one: bool = True
    metrics: dict = field(default_factory=dict)
    version: str = "1"

    # ------------------------------------------------------------------ fit --
    @classmethod
    def fit(cls, ds, train_idx, val_idx, labels, groups=F.STRUCT_GROUPS,
            kind="logreg", C=4.0, min_df=4, max_df=0.75, max_features=200_000,
            use_numeric=True, objective="macro", verbose=True, binary_tf=False,
            num_scale=0.5, score_norm="none",
            force_one=True) -> "StructuralWebsiteClassifier":
        from .model import binarize

        vec = Vectorizer(groups=list(groups), min_df=min_df, max_df=max_df,
                         max_features=max_features, use_numeric=use_numeric,
                         binary_tf=binary_tf, num_scale=num_scale)
        vec.fit(ds, train_idx)
        Xtr = vec.transform(ds, train_idx)
        Xva = vec.transform(ds, val_idx)
        Ytr = binarize([ds.labels[i] for i in train_idx], labels)
        Yva = binarize([ds.labels[i] for i in val_idx], labels)
        if verbose:
            print(f"[fit] X={Xtr.shape} labels={len(labels)} "
                  f"density={Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1]):.4f}")
        clf = fit_model(Xtr, Ytr, kind=kind, C=C)
        thr = tune_thresholds(norm_scores(_scores(clf, Xva), score_norm), Yva,
                              objective=objective, score_norm=score_norm)
        return cls(labels=list(labels), vectorizer=vec, model=clf,
                   thresholds=thr, groups=list(groups), score_norm=score_norm,
                   force_one=force_one)

    # -------------------------------------------------------------- predict --
    def predict(self, raw_html: bytes | str, url: str = "", top_k: int | None = None,
                min_prob: float = 0.0) -> list[tuple[str, float]]:
        raw = raw_html.encode("utf-8", "replace") if isinstance(raw_html, str) else raw_html
        doc_groups = list(self.groups)
        # `url` / `words` groups can only be used if they were fitted; extract accordingly
        toks, stats = F.extract(raw, url or None, groups=tuple(doc_groups))
        return self.predict_features(toks, stats, top_k=top_k, min_prob=min_prob)

    def predict_features(self, toks, stats, top_k=None, min_prob=0.0):
        X = self.vectorizer.transform_tokens(toks, stats)
        s = norm_scores(_scores(self.model, X), self.score_norm)[0]
        order = np.argsort(-s)
        out = []
        for j in order:
            p = float(s[j])
            if p < self.thresholds[j] or p < min_prob:
                continue
            out.append((self.labels[j], round(p, 4)))
            if top_k and len(out) >= top_k:
                break
        if not out and self.force_one:  # never return nothing
            out = [(self.labels[int(order[0])], round(float(s[int(order[0])]), 4))]
        return out

    def score_matrix(self, ds, idx) -> np.ndarray:
        return norm_scores(_scores(self.model, self.vectorizer.transform(ds, idx)),
                           self.score_norm)

    def decode(self, scores: np.ndarray) -> np.ndarray:
        """Scores -> label matrix.  Every page gets at least one topic.

        A page always belongs to some topic, so an all-below-threshold row is a
        threshold artefact rather than a real "none of the above"; forcing the top
        label recovers those.  `force_one=False` keeps the strict behaviour.
        """
        pred = (scores >= self.thresholds[None, :]).astype(np.int8)
        if self.force_one and pred.shape[1]:
            empty = pred.sum(axis=1) == 0
            if empty.any():
                pred[empty, np.asarray(scores[empty]).argmax(axis=1)] = 1
        return pred

    def evaluate(self, ds, idx, objective_note="") -> dict:
        from .model import binarize
        Y = binarize([ds.labels[i] for i in idx], self.labels)
        S = self.score_matrix(ds, idx)
        pred = self.decode(S)
        m = report(Y, pred, self.labels)
        m.update({"n": int(len(idx)), "objective_note": objective_note})
        try:
            from sklearn.metrics import roc_auc_score
            m["labeled_mean_auc"] = float(np.mean(roc_auc_score(Y, S, average=None)))
        except ValueError:
            pass
        return m

    # ------------------------------------------------------------- persist --
    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "StructuralWebsiteClassifier":
        import joblib
        return joblib.load(path)


def norm_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    """Put per-label scores on a comparable scale.

    `doc_z` standardises each document's score vector over its own labels. Raw
    probabilities are not comparable across corpora -- a model moved from Czech to
    English pages sees the whole score level shift, so thresholds tuned on the
    source stop making sense; centring per document removes that shift.
    """
    s = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    if mode == "none":
        return s
    if mode == "doc_z":
        mu = s.mean(axis=1, keepdims=True)
        sd = s.std(axis=1, keepdims=True)
        sd[sd < 1e-9] = 1.0
        return (s - mu) / sd
    raise ValueError(f"unknown score_norm {mode!r}")


def _scores(clf, X) -> np.ndarray:
    """Per-label scores in [0, 1] for both the logreg and the LightGBM backend."""
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X), dtype=np.float64)
    d = np.asarray(clf.decision_function(X), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-d))
