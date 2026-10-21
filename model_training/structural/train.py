#!/usr/bin/env python
"""Train / evaluate / serve the language-agnostic HTML topic classifier.

    python -m structural.train prep    --data ../27_08_01_en_cz.jsonl
    python -m structural.train train   --data ../27_08_01_en_cz.jsonl --ablation
    python -m structural.train train   --data ../27_08_01_en_cz.jsonl --sweep --save models/struct.joblib
    python -m structural.train predict --model models/struct.joblib --url https://example.cz/
    python -m structural.train explain --model models/struct.joblib --labels Shopping,Food

Splits
------
`domain`      pages of one registrable domain never appear in two splits, so the
              numbers are not inflated by near-duplicate templates.
`cs2en`       trained on every Czech page, tested on every English page.
`en2cs`       the other way round.  These two are the actual test of "does this
              depend on the language" -- note they also cross a vintage gap (the
              English half is a 2004-era archive), so they are a worst case.
Thresholds are tuned on validation rows only; test rows are untouched until the
final number.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from . import data as D
from . import features as F
from .pipeline import StructuralWebsiteClassifier

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "reports"

CORE = ("tag", "attr", "href", "meta", "plat")

# Ablation ladder: each rung adds one family of features to the one above it.
ABLATION = {
    "stats_only":     (),                              # dense statistics only
    "tag":            ("tag",),
    "tag+attr":       ("tag", "attr"),
    "no_href":        ("tag", "attr", "meta", "plat"),  # drop link-syntax tokens
    "core":           CORE,                            # no naming at all
    "core_no_href+css": ("tag", "attr", "meta", "plat", "css"),  # core+css minus link syntax
    "core+css":       CORE + ("css",),                 # + class/id name parts
    "core+css+shape": CORE + ("css", "shape"),
    "core+css+res":   CORE + ("css", "res"),           # + resource file stems
    "core+css+marker": CORE + ("css", "marker"),       # + keyword hints
    "core+cssraw":    CORE + ("cssraw",),              # + whole class strings
    "all_struct":     F.STRUCT_GROUPS,                 # incl. the keyword markers
    "text_only":      ("words",),                      # language-dependent ceiling
    "text+struct":    F.STRUCT_GROUPS + ("words",),
    "url_tokens":     F.STRUCT_GROUPS + ("url",),
}


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

def domain_splits(ds, seed=0, train=0.7, val=0.15):
    from sklearn.model_selection import GroupShuffleSplit
    idx = np.arange(len(ds))
    g = np.asarray(ds.domains, dtype=object)
    tr, valtest = next(GroupShuffleSplit(n_splits=1, test_size=1 - train,
                                         random_state=seed).split(idx, groups=g))
    frac = (1 - train - val) / (1 - train)
    v, t = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed + 1)
                .split(np.arange(len(valtest)), groups=g[valtest]))
    return tr, valtest[v], valtest[t]


def lang_splits(ds, seed=0, val_frac=0.15):
    """(train, val, test) index triples for cs->en and en->cs."""
    langs = np.asarray(ds.langs)
    cs, en = np.flatnonzero(langs == "cs"), np.flatnonzero(langs == "en")
    rng = np.random.default_rng(seed)
    out = {}
    for name, (src, dst) in {"cs2en": (cs, en), "en2cs": (en, cs)}.items():
        p = rng.permutation(src)
        n_val = max(200, int(len(p) * val_frac))
        out[name] = (p[n_val:], p[:n_val], dst)
    return out


def label_universe(ds, min_count=15):
    c = Counter()
    for labs in ds.labels:
        c.update(set(labs))
    return [l for l, n in sorted(c.items()) if n >= min_count], c


# --------------------------------------------------------------------------- #
# one configuration, all splits
# --------------------------------------------------------------------------- #

def run_config(ds, name, cfg, labels, seed=0, verbose=True, fit_final=False):
    groups = tuple(g for g in cfg["groups"] if g in ds.groups)
    missing = set(cfg["groups"]) - set(groups)
    if missing:
        print(f"  {name}: not cached -> {','.join(sorted(missing))}")
        return None
    kw = dict(groups=groups, kind=cfg.get("model", "logreg"), C=cfg.get("C", 4.0),
              min_df=cfg.get("min_df", 4), max_df=cfg.get("max_df", 0.75),
              max_features=cfg.get("max_features", 200_000),
              binary_tf=cfg.get("binary_tf", False),
              num_scale=cfg.get("num_scale", 0.5),
              score_norm=cfg.get("score_norm", "none"),
              force_one=cfg.get("force_one", True),
              objective=cfg.get("threshold_objective", "macro"))
    res = {"name": name, "cfg": {**cfg, "groups": list(groups)}, "splits": {}}
    tr, va, te = domain_splits(ds, seed=seed)
    splits = {"domain": (tr, va, te), **lang_splits(ds, seed=seed)}
    for split, (a, b, c) in splits.items():
        if len(a) < 50:
            continue
        if len(b) < 20:                      # degenerate source slice
            b = a[: max(20, len(a) // 5)]
        t0 = time.time()
        clf = StructuralWebsiteClassifier.fit(ds, a, b, labels, verbose=False, **kw)
        if split == "domain":
            res["X"] = str(clf.vectorizer.n_cols_)
            if fit_final:
                # deployment bundle: refit on train+val, keep val-tuned thresholds
                res["clf"] = StructuralWebsiteClassifier.fit(
                    ds, np.concatenate([tr, va]), va, labels, verbose=False, **kw)
        m = clf.evaluate(ds, c)
        m["secs"] = round(time.time() - t0, 1)
        res["splits"][split] = m
        if verbose:
            print(f"  {name:<22} {split:<8} n={m['n']:<5} micro {m['micro_f1']:.3f}  "
                  f"macro {m['macro_f1']:.3f}  exact {m['exact_match']:.3f}  "
                  f"auc {m.get('labeled_mean_auc', float('nan')):.3f}  "
                  f"empty {m['empty_pred']:.2f}  [{m['secs']}s]", flush=True)
    res["score"] = combine(res["splits"])
    return res


def combine(splits) -> float:
    """Headline number: unseen sites first, unseen language as the tie-breaker."""
    s = 0.0
    if "domain" in splits:
        s += splits["domain"]["macro_f1"]
    for k in ("cs2en", "en2cs"):
        if k in splits:
            s += 0.5 * splits[k]["macro_f1"]
    return round(s, 4)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_prep(a):
    t0 = time.time()
    ds = D.load(a.data, groups=F.ALL_GROUPS, n_docs=a.n_docs, workers=a.workers)
    labels, counts = label_universe(ds)
    print(f"[prep] {len(ds)} docs in {time.time()-t0:.0f}s")
    print(f"[prep] languages: {dict((l, ds.langs.count(l)) for l in set(ds.langs))}")
    print(f"[prep] domains:   {len(set(ds.domains))}")
    for l in sorted(counts, key=lambda k: -counts[k]):
        print(f"         {counts[l]:5d}  {l}")


def _configs(a):
    """What to evaluate: the ablation ladder, or a hyper-parameter grid."""
    groups = tuple(g for g in a.groups.split(",") if g)
    if a.sweep:
        grid = {"min_df": [int(x) for x in a.min_df.split(",")],
                "max_df": [float(x) for x in a.max_df.split(",")],
                "C": [float(x) for x in a.C.split(",")],
                "score_norm": a.score_norm.split(","),
                "binary_tf": [x == "1" for x in a.binary.split(",")]}
        for combo in itertools.product(*grid.values()):
            cfg = dict(zip(list(grid), combo))
            cfg["groups"] = groups
            cfg["model"] = a.model
            name = (f"mdf{cfg['min_df']} xdf{cfg['max_df']} C{cfg['C']} "
                    f"{cfg['score_norm']} b{int(cfg['binary_tf'])}")
            yield name, cfg
    else:
        run = ABLATION.items() if (a.ablation or a.only) else [("selected", groups)]
        for name, gs in run:
            if a.only and name not in a.only.split(","):
                continue
            yield name, {"groups": gs, "model": a.model,
                         "C": float(a.C.split(",")[0]),
                         "min_df": int(a.min_df.split(",")[0]),
                         "max_df": float(a.max_df.split(",")[0]),
                         "score_norm": a.score_norm.split(",")[0],
                         "binary_tf": a.binary.split(",")[0] == "1",
                         "threshold_objective": a.threshold_objective}


def cmd_train(a):
    ds = D.load(a.data, groups=F.ALL_GROUPS, n_docs=a.n_docs, workers=a.workers)
    labels, counts = label_universe(ds, min_count=10)
    print(f"[train] {len(ds)} docs, {len(labels)} labels: {labels}", flush=True)
    results = []
    for name, cfg in _configs(a):
        r = run_config(ds, name, cfg, labels, seed=a.seed,
                       verbose=not a.quiet, fit_final=bool(a.save))
        if r:
            results.append(r)
    if not results:
        print("nothing ran")
        return

    results.sort(key=lambda r: -r["score"])
    print("\n" + "=" * 100)
    print(f"{'config':<24}{'cols':>8}{'score':>7}   "
          + "   ".join(f"{s:>18}" for s in ("domain", "cs2en", "en2cs")))
    print(f"{'':<39}   " + "   ".join(f"{'micro / auc':>18}" for _ in range(3)))
    print("=" * 100)
    for r in results:
        cells = []
        for sp in ("domain", "cs2en", "en2cs"):
            m = r["splits"].get(sp)
            cells.append((f"{m['micro_f1']:.3f}/{m.get('labeled_mean_auc', 0):.3f}".rjust(18)
                          if m else "".rjust(18)))
        print(f"{r['name']:<24}{r.get('X', ''):>8}{r['score']:>7.3f}   " + "   ".join(cells))

    tag = a.tag or ("sweep" if a.sweep else "ablation" if a.ablation else "selected")
    REPORT_DIR.mkdir(exist_ok=True)
    out = {"n_docs": len(ds), "labels": labels,
           "configs": [{k: v for k, v in r.items() if k != "clf"} for r in results]}
    (REPORT_DIR / f"{tag}.json").write_text(json.dumps(out, indent=2, default=float),
                                            encoding="utf-8")
    _write_md(REPORT_DIR / f"{tag}.md", out)
    print(f"\n[report] {REPORT_DIR / (tag + '.md')}")

    if a.save and "clf" in results[0]:
        best = results[0]
        best["clf"].metrics = {"config": best["name"], "cfg": best["cfg"],
                               **best["splits"]}
        best["clf"].save(a.save)
        print(f"[save] {a.save}  <- {best['name']}  (score {best['score']})")


def _write_md(path, out):
    L = ["# Language-agnostic HTML topic classifier — evaluation", "",
         f"- pages: {out['n_docs']}",
         f"- labels: {', '.join(out['labels'])}", "",
         "**domain** — held-out websites (70/15/15 by registrable domain).  "
         "**cs2en / en2cs** — trained on all Czech pages, tested on all English "
         "ones, and the reverse: the language-independence test.  The two halves "
         "differ in vintage as well as language, so read those two as a worst case.  "
         "Per-label thresholds are tuned on validation rows only.", "",
         "| config | split | n | micro F1 | macro F1 | exact | mean AUC | empty |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out["configs"]:
        for sp, m in r["splits"].items():
            L.append(f"| `{r['name']}` | {sp} | {m['n']} | {m['micro_f1']:.3f} "
                     f"| {m['macro_f1']:.3f} | {m['exact_match']:.3f} "
                     f"| {m.get('labeled_mean_auc', float('nan')):.3f} "
                     f"| {m['empty_pred']:.3f} |")
    L += ["", "## Per-class detail", ""]
    for r in out["configs"]:
        for sp, m in r["splits"].items():
            L += [f"### `{r['name']}` / {sp}", "",
                  "| label | n | P | R | F1 |", "|---|---|---|---|---|"]
            for lab, v in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["n"]):
                L.append(f"| {lab} | {v['n']} | {v['p']:.3f} | {v['r']:.3f} | {v['f1']:.3f} |")
            L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def _load_file(p: Path):
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        return (D.decompress(rec["compressed_html"]), rec.get("url") or "",
                rec.get("categories") or [])
    raw = p.read_bytes()
    if p.suffix in (".gz", ".gzip"):
        raw = gzip.decompress(raw)
    return raw, "", []


def cmd_predict(a):
    clf = StructuralWebsiteClassifier.load(a.model)
    if a.url:
        req = urllib.request.Request(a.url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                raw = r.read()
        except (urllib.error.URLError, OSError) as e:
            # bots get 403/402 a lot; a one-line message beats a traceback here
            sys.exit(f"could not fetch {a.url}: {e}")
        raw, url, truth = raw, a.url, []
    elif a.file:
        raw, url, truth = _load_file(Path(a.file))
    else:
        raw, url, truth = sys.stdin.buffer.read(), "", []
    if raw is None:
        sys.exit("could not decompress that page")
    if truth:
        print(f"# {url}\n# truth: {truth}")
    for lab, s in clf.predict(raw, url=url, top_k=a.top_k):
        print(f"{s:6.3f}  {lab}")


def cmd_explain(a):
    from .explain import build_token_strings, top_features
    clf = StructuralWebsiteClassifier.load(a.model)
    urls = _urls_from(a.data, a.sample)
    strings = build_token_strings(clf, a.data, urls, sample=a.sample, workers=a.workers)
    print(f"[explain] recovered {len(strings)}/{len(clf.vectorizer.vocab)} "
          f"vocabulary strings from {a.sample} pages")
    for lab in (a.labels.split(",") if a.labels else clf.labels):
        print(f"\n### {lab}")
        for name, w in top_features(clf, lab, k=a.k, strings=strings):
            print(f"  {w:+.3f}  {name}")


def _urls_from(data, n):
    urls = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            if len(urls) >= n:
                break
            try:
                urls.append(json.loads(line).get("url") or "")
            except json.JSONDecodeError:
                continue
    return urls


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default="../27_08_01_en_cz.jsonl")
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--n-docs", type=int, default=None)

    p = sub.add_parser("prep")
    common(p)
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("train")
    common(p)
    p.add_argument("--groups", default=",".join(F.STRUCT_GROUPS))
    p.add_argument("--model", choices=["logreg", "lgbm"], default="logreg")
    p.add_argument("--C", default="4.0", help="comma list when sweeping")
    p.add_argument("--min-df", default="4", help="comma list when sweeping")
    p.add_argument("--max-df", default="0.75", help="comma list when sweeping")
    p.add_argument("--score-norm", default="none", help="none|doc_z, comma list to sweep")
    p.add_argument("--binary", default="0", help="1 = presence-only tf; comma list to sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold-objective", choices=["macro", "micro"], default="macro")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--only", default="", help="comma list of ablation rungs")
    p.add_argument("--save", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--url")
    p.add_argument("--file")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("explain")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="../27_08_01_en_cz.jsonl")
    p.add_argument("--labels", default="")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--sample", type=int, default=2500)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_explain)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
