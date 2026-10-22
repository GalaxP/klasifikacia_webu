# structural — topic classification from HTML structure, not from words

Predicts the topic of a web page (11 Yahoo/KDD-Cup-2004 style categories) from its
HTML source, using features that survive translation: the document tree, tag and
attribute usage, template repetition, class/id naming, metadata and resource
fingerprints, plus a *shape* profile of the text (word lengths, digit and punctuation
density) — never the words themselves.

The dataset `27_08_01_en_cz.jsonl` is 8291 pages, 5476 Czech and 2815 English, over
6259 registrable domains, which makes it possible to *measure* language independence
instead of asserting it: `--ablation` trains on Czech and tests on English, and the
other way round.

Two feature families live here and are measured against each other:

* **structural** — tags, attributes, link syntax, metadata, platform and resource
  fingerprints, class/id naming, page statistics. Contains no words by construction.
* **content, word-free** — a multilingual sentence embedding of the visible text
  ([`embed.py`](embed.py)). This is *not* a contradiction: the encoder maps
  100+ languages into one space trained so translations land on top of each other
  (measured here: Czech/English sentence pairs 0.84–0.95 cosine, unrelated pair 0.11),
  so the model learns no per-language vocabulary. It is by far the better model — see
  [Adding content](#adding-content-without-adding-a-language) — while remaining
  usable on a language it was never trained on.

## Layout

| file | role |
|---|---|
| `features.py` | HTML → (token groups, ~130 dense statistics). No word content. |
| `data.py` | jsonl + gzip decoding, language tagging, parallel featurisation, hash-token cache in `cache/` |
| `model.py` | hash-keyed TF-IDF vectoriser, one-vs-rest heads, threshold tuning, metrics |
| `pipeline.py` | `StructuralWebsiteClassifier` — fit / evaluate / save / load / predict |
| `explain.py` | recovers `hash → token` strings so weights can be read as reasons |
| `embed.py` | optional multilingual text vectors: build, cache, and embed one page |
| `train.py` | CLI: `prep`, `train` (ablation, sweep), `predict`, `explain` |
| `demo.ipynb` | executed walkthrough: ablation, per-class table, live scoring, top features |
| `models/structural_v1.joblib` | structure-only bundle, 4.5 MB, no text model needed at scoring time |
| `models/structural_v2_embed.joblib` | structure + multilingual text, the better model; needs `sentence-transformers` to score a page |
| `reports/*.md` | generated evaluation tables (`ablation`, `ablation_mdf10`, `sweep`, `embed_*`) |

## Usage

```bash
# 1. featurise once (~1 min for 8291 pages on 14 cores), cached under structural/cache/
python -m structural.train prep --data ../27_08_01_en_cz.jsonl --workers 14

# 2. feature ablation on all three evaluation splits
python -m structural.train train --data ../27_08_01_en_cz.jsonl --ablation

# 3. optional, once: multilingual text vectors (~6 min CPU, cached as a 12 MB .npz)
python -m structural.embed --data ../27_08_01_en_cz.jsonl
python -m structural.embed --check          # is the encoder really cross-lingual?

# 4. hyper-parameter sweep; the best config by `combine()` is refit on train+val and saved
python -m structural.train train --data ../27_08_01_en_cz.jsonl --sweep \
    --min-df 4,10,25 --C 1,4,16 --score-norm none,doc_z --tag sweep \
    --save models/structural_v1.joblib
#    same, with the text block added: `--embed` (and `--embed-scale 16`, see below)
python -m structural.train train --data ../27_08_01_en_cz.jsonl --embed \
    --only struct+embed --min-df 10 --embed-scale 16 \
    --save models/structural_v2_embed.joblib

# 5. read what the model keys on
python -m structural.train explain --model models/structural_v1.joblib --labels Shopping,News,Food

# 6. score a page
python -m structural.train predict --model models/structural_v2_embed.joblib --url https://example.cz/
python -m structural.train predict --model models/structural_v1.joblib --file page.html
```

```python
from structural.pipeline import StructuralWebsiteClassifier
clf = StructuralWebsiteClassifier.load("models/structural_v1.joblib")
clf.predict(open("page.html", "rb").read(), url="https://example.cz/")
# [('Shopping', 0.91), ('Computers', 0.33)]
```

## Feature groups

| group | what it is | language dependent? | buys what (see `reports/ablation.md`) |
|---|---|---|---|
| `tag` | tag counts, parent→child paths, tag@depth, repeated-sibling block shapes | no | 0.332 alone, no better than the raw statistics |
| `attr` | which attributes appear on which tag, input types, `rel`, image aspect | no | the big jump: 0.332 → 0.413 |
| `meta` | meta/OG/Twitter names, `link rel`, schema.org & JSON-LD `@type`, form actions, RSS | no (identifiers) | part of `core` |
| `href` | link *syntax* shape: extensions, query keys, numeric path segments, depth — never path words | mostly no, **but the least neutral family** | in-domain ≈0, hurts `en2cs` |
| `plat` | platform fingerprints (wp, joomla, magento, vbulletin, heureka…) | no (proper names) | part of `core` |
| `res` | script/img/link **file** stems + extensions | mostly no (CMS file names are English everywhere) | +0.02, best small addition |
| `css` | class/id names split into sub-tokens (`product-card` → `product`,`card`,`product-card`) | mostly no — CMS naming is English in every language | **the single biggest addition**, +0.05 |
| `cssraw` | whole class strings unsplit | no | **useless**, −0.02 vs `css` |
| `marker` | small bilingual hint list (`cart`, `košík`, `price`, `cena`…) | partly, by construction | +0.002 — not load-bearing |
| `shape` | word-length histogram, digit/punct/diacritic/thousands-separator ratios | weakly | +0.00 |
| `embed` | multilingual sentence embedding of the visible text (384 dims → SVD 128) | **no** — one shared space for all languages, no per-language vocabulary | +0.21 in-domain, **+0.36 cross-lingual** |
| `url` | url path + host tokens | **yes** | +0.02 in-domain, nothing cross-lingual |
| `words` | visible text words | **yes** — a per-language vocabulary | +0.08 over `embed` alone in-domain, **0.14 worse** cross-lingual |

`url` and `words` exist so the ablation can show what language independence *costs*.
`words` is a vocabulary of the languages in the training data, so it is the only group
that does not extend to a new language — `embed` carries the same content signal without
that cost.

## Evaluation protocol

- **`domain`** — one registrable domain never appears in two splits (70/15/15).
  Without this, template twins of the same site leak between splits and every number
  is inflated.
- **`cs2en` / `en2cs`** — train on all Czech pages, test on all English ones, and the
  reverse. This is the only honest test of "does not depend on language".
- Per-label decision thresholds are tuned on the validation split only; test rows are
  never touched before the final number.
- Metrics: micro / macro / samples F1, exact match, Hamming loss, per-label mean AUC.
  **Read AUC for the cross-lingual columns** — see "thresholds, not features".

## Results

Deployed model (`models/structural_v1.joblib`, refit on train+val) = all structural
groups, `min_df=10`, `max_df=0.75`, `C=4`, 58 833 hashed tokens + 129 statistics,
logistic one-vs-rest with balanced classes:

| split | micro F1 | macro F1 | exact match | mean AUC |
|---|---|---|---|---|
| `domain` (unseen websites) | 0.495 | 0.465 | 0.348 | 0.807 |
| `cs2en` (unseen language) | 0.323 | 0.213 | 0.272 | 0.682 |
| `en2cs` (unseen language) | 0.281 | 0.295 | 0.073 | 0.680 |
| `text_only` baseline, `domain` | **0.758** | 0.746 | 0.606 | **0.957** |
| `text_only` baseline, `cs2en` | 0.497 | 0.446 | 0.436 | 0.879 |
| `text_only` baseline, `en2cs` | 0.605 | 0.565 | 0.515 | 0.905 |

The ablation ladder at `min_df=4` (129 statistics alone already reach 0.344):

| rung | cols | domain micro / AUC | cs2en micro / AUC |
|---|---|---|---|
| `stats_only` | 129 | 0.344 / 0.727 | 0.269 / 0.626 |
| `tag` | 3 247 | 0.332 / 0.726 | 0.237 / 0.604 |
| `tag+attr` | 11 139 | 0.413 / 0.752 | 0.262 / 0.617 |
| `core` (+meta, `href`, `plat`) | 13 671 | 0.424 / 0.770 | 0.269 / 0.630 |
| `core+css` | 66 480 | 0.473 / 0.795 | 0.314 / 0.669 |
| `core+css+shape` | 66 528 | 0.470 / 0.795 | 0.313 / 0.669 |
| `core+css+marker` | 66 489 | 0.475 / 0.795 | 0.315 / 0.670 |
| `core+cssraw` (unsplit classes) | 58 908 | 0.453 / 0.777 | 0.281 / 0.636 |
| `core+css+res` | 84 883 | 0.492 / 0.811 | 0.319 / 0.683 |
| `all_struct` | 130 177 | 0.485 / 0.809 | 0.318 / 0.683 |
| `url_tokens` (+ path/host tokens) | 143 890 | 0.508 / 0.835 | 0.331 / 0.696 |
| `text+struct` | 200 129 | 0.669 / 0.922 | 0.470 / 0.796 |
| `text_only` | 71 807 | 0.758 / 0.957 | 0.497 / 0.879 |

`reports/ablation_mdf10.md` re-runs the interesting rungs at the deployed `min_df=10`,
which is where the `href` verdict comes from: `core+css` 0.467 in-domain vs `core+css`
without `href` 0.471 (no gain), but `en2cs` 0.234 **with** vs 0.318 **without** it. Link
syntax is the least neutral family in the set — URL conventions belong to a corpus
generation, not to a page function, so `H:ext=cz` and `H:numseg` end up encoding "which
2004 archive this page came from" rather than "what this page is for". Frequency
filtering also helps: `min_df=10` (50 k columns) beats `min_df=4` (130 k) in-domain.

### What the model actually keys on

From `train explain` (weights of each one-vs-rest head, hashes resolved by
re-featurising a sample of pages):

```
Shopping   c:product +7.8  c:price +6.6  c:cart +5.5  M:cart +4.9  c:category +3.6
           r:ext:jpg +3.6  r:shop +3.5  c:basket +3.5  c:sale +3.0
News       c:date +5.8  c:news +5.0  c:article +4.7  H:numseg +4.1  p:h3>a +3.4
           a:rel=noopener +3.1  a:td[bgcolor] +2.8  p:a>font +2.8
Food       c:recipe +6.2  r:pizza +4.4  c:recipes +4.3  c:recepty +3.5  r:wine +2.8
```

Nothing here is a word of prose: `c:`/`r:` are CSS and file-name fragments that CMS
software writes in English regardless of the site's language — `c:recepty` (Czech
"recipes") sits right next to `c:recipe` because a Czech developer named it that way,
and `News` is partly detected by *legacy* markup (`td[bgcolor]`, `font` inside a
paragraph) that dates the 2004-era half of the corpus.

### Structure names page *function*, not subject

Per-class F1 of the deployed model — held-out websites, then Czech→English:

| label | n (domain) | F1 domain | F1 cs2en |
|---|---|---|---|
| Shopping | 288 | 0.683 | **0.571** |
| News | 208 | 0.583 | 0.238 |
| Entertainment | 216 | 0.567 | 0.308 |
| Computers | 194 | 0.536 | 0.406 |
| Sports | 110 | 0.437 | 0.202 |
| Automotive | 55 | 0.368 | 0.058 |
| Finance | 111 | 0.319 | 0.170 |
| Food | 101 | 0.302 | **0.031** |
| Health | 139 | 0.277 | 0.165 |
| Travel | 96 | 0.253 | 0.134 |
| Adult | 16 | 0.786 | 0.066 |

`Shopping` is the one class that transfers well, because a product grid, a price
element and a cart form *are* a layout. `Food`, `Automotive` and `Travel` are defined by
what the page is about, not by how it is built, and a word-free model runs out of road
there — no amount of tag engineering names a soufflé. (`Adult` in the `domain` column is
16 pages; ignore it.)

### Thresholds, not features, carry part of the cross-lingual loss

`en2cs` exact match is 0.073 while mean AUC stays at 0.680: the *ranking* survives the
language switch far better than the per-label cut-offs, which were tuned on
source-language rows and stop fitting once the whole score level shifts. Freezing the
trained model and changing only the decoding — Czech pages, weights untouched, thresholds
re-tuned on a held-out half of the target rows (`demo.ipynb`):

| decoding of the same scores | micro F1 | macro F1 | exact |
|---|---|---|---|
| thresholds tuned on English rows | 0.276 | 0.284 | 0.067 |
| thresholds re-tuned on Czech rows | 0.334 | 0.319 | 0.115 |
| **top-1, no thresholds at all** | **0.356** | 0.274 | 0.317 |
| top-2, no thresholds | 0.348 | 0.298 | 0.016 |

So a fifth of the apparent cross-lingual F1 loss is pure calibration, and the simplest
possible decoder — take the best label — beats the tuned thresholds by 0.08 micro. Pages
in this corpus carry 1.2 topics on average, which is why top-1 is competitive. Deployed
on a new language: use top-k, or recalibrate on a small target-language sample; do not
ship the thresholds in the bundle.

## Adding content without adding a language

Structure names page *function*; it cannot see what a page is *about*, which is why
`Food`, `Health`, `Automotive` and `Travel` sit at 0.25–0.37. The fix is content in a
form that is not tied to a language: [`embed.py`](embed.py) encodes each page's visible
text with `paraphrase-multilingual-MiniLM-L12-v2` (4 windows per page, averaged,
L2-normalised), reduces 384 → 128 dims with SVD, standardises, and concatenates that
block onto the structural design matrix. Nothing per-language is learned — the alignment
between Czech and English lives in the encoder's weights, which were trained on 100+
languages for exactly that purpose.

Micro F1 / mean AUC, everything else identical (`min_df=10`, `C=4`, same splits):

| model | cols | `domain` | `cs2en` | `en2cs` |
|---|---|---|---|---|
| `all_struct` — no words | 50 019 | 0.495 / 0.807 | 0.323 / 0.682 | 0.281 / 0.680 |
| `text_only` — word TF-IDF | 35 452 | **0.757** / 0.956 | 0.506 / 0.875 | 0.591 / 0.896 |
| `embed_only` | 257 | 0.681 / 0.927 | 0.649 / 0.909 | 0.641 / 0.907 |
| **`struct+embed`** — shipped as v2 | 50 147 | 0.704 / 0.941 | **0.678** / 0.919 | 0.654 / 0.910 |
| `struct+text+embed` | 85 470 | 0.734 / 0.950 | 0.678 / 0.921 | **0.674** / 0.914 |

Three things worth reading off that table:

- **The content categories were the ones that moved.** Per-class F1 on held-out
  websites, structure → structure+text: `Health` 0.277 → 0.704, `Food` 0.302 → 0.714,
  `Automotive` 0.368 → 0.739, `Travel` 0.253 → 0.547. Cross-lingually `Food` goes
  0.031 → 0.567 — from "no signal at all" to "usable".
- **The embedding transfers better than the words do**, which is the whole point: word
  TF-IDF loses 0.25 micro F1 going from seen to unseen language, the embedding loses
  0.03. Words do not align across languages; these vectors are trained to.
- **Being language-agnostic is nearly free.** `struct+text+embed` beats
  `struct+embed` by 0.03 in-domain and 0.02 on `en2cs`, and the price is a vocabulary
  that only covers the languages it was trained on.

**The one knob that matters is `--embed-scale`.** Naively concatenated, the 128 dense
dims are invisible next to 50 000 tf-idf columns — after the row L2 normalisation the
text block contributes ~0.4% of each row's norm, and `struct+embed` scores 0.369 on
`cs2en`, *worse than `embed_only`'s 0.649* (`reports/embed_cmp.md` vs
`reports/embed_final.md`). Scaling the block fixes it monotonically until it saturates —
one run per column, `--only struct+embed --embed --embed-scale N`:

| `embed_scale` | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| `cs2en` micro F1 | 0.369 | 0.468 | 0.598 | 0.661 | **0.678** | 0.675 |

**What it costs.** ~6 min CPU to embed 8291 pages once (12 MB cache); the encoder
downloads 470 MB on first use; scoring a page now needs `sentence-transformers` + torch
installed (a few hundred ms on CPU, plus ~60 s once to load the encoder); the weights are
no longer human-readable — `explain` reports those columns as `~embed17` because there is
no word behind them, which is the same fact as the language independence, seen from the
other side. In-domain, plain words still edge the word-free model out (0.757 vs 0.704),
so the word-free choice costs about 0.05 there and buys 0.17 on unseen languages.

## Live spot check (out of distribution, honest version)

| URL | structure-only (v1) | + multilingual text (v2) |
|---|---|---|
| `recepty.cz` | Food 0.98 ✓ | Food 1.00 ✓ |
| `idnes.cz` | News 0.96 ✓ | News 0.95 ✓ |
| `auto.cz` | — | Automotive 1.00 ✓ |
| `svethardware.cz` | Entertainment 0.82, Computers 0.75 (half right) | Computers 0.96 ✓ |
| `bbc.com/news` | Shopping 0.34 ✗ | News 0.96 ✓ |
| `novinky.cz` | Computers 0.57 ✗ | Computers 0.64 ✗ |

Czech sites scored well even for v1, recent English sites did not — the training corpus
is old, so modern JS-heavy layouts are off-distribution for a purely structural model,
while the text of a news site is recognisable in any era. Several hosts refuse a plain
`urllib` fetch (403/402), which `predict` reports as one line instead of a traceback.

## Limitations

- **Only two languages, one of them a 2004 archive.** `cs2en` / `en2cs` cross a vintage
  and CMS-generation gap as well as a language one, so they are a pessimistic bound on
  language independence, not a clean ablation of it. The claim the embedding supports is
  the strong one (0.03 loss instead of 0.25 when the language changes); a claim about a
  *third* language would need data this corpus does not have.
- Structural features describe **how a page is built**; they separate shops, forums,
  portals and article templates well and separate cuisines, car brands or holiday
  destinations poorly. That is a ceiling on structure alone, not a tuning defect — which
  is what the `embed` block is for.
- **The encoder is a black box with its own training distribution.** It brings its own
  language coverage (100+ languages, unevenly), its own biases, and a fixed 384-dim view
  of the page; unlike every other group here, its weights cannot be inspected for
  "what made this decision". A page whose text is entirely an image, a video or a JS
  bundle that never renders loses the content signal entirely and falls back on
  structure.
- `marker` is a deliberate exception to the rule: a small bilingual keyword list. It is
  worth +0.002 micro, i.e. nothing, so the model does not lean on it — remove the group
  if you want a purer claim.
- Thresholds are tuned per-split; a model applied to a new corpus should recalibrate.
- **The text model is a live dependency, and it is not in `pyproject.toml` yet.** It was
  installed with `pip install sentence-transformers` into the existing poetry venv
  (torch was already there); run `poetry add sentence-transformers` to record it. Only
  the `embed` group needs it — `models/structural_v1.joblib` scores pages with no text
  model at all.
- **Still open:** a third language as a genuine unseen-language split; recalibration of
  thresholds on target-language rows (worth 0.06 micro, see above); a larger encoder
  (`multilingual-e5-base`, LaBSE) if the 6-minute embedding pass is affordable —
  `--embed <model>` switches without retraining anything else.
