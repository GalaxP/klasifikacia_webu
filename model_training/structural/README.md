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

## Layout

| file | role |
|---|---|
| `features.py` | HTML → (token groups, ~130 dense statistics). No word content. |
| `data.py` | jsonl + gzip decoding, language tagging, parallel featurisation, hash-token cache in `cache/` |
| `model.py` | hash-keyed TF-IDF vectoriser, one-vs-rest heads, threshold tuning, metrics |
| `pipeline.py` | `StructuralWebsiteClassifier` — fit / evaluate / save / load / predict |
| `explain.py` | recovers `hash → token` strings so weights can be read as reasons |
| `train.py` | CLI: `prep`, `train` (ablation, sweep), `predict`, `explain` |
| `demo.ipynb` | executed walkthrough: ablation, per-class table, live scoring, top features |
| `models/structural_v1.joblib` | trained bundle (4.5 MB, no vocabulary needed at load time) |
| `reports/*.md` | generated evaluation tables (`ablation`, `ablation_mdf10`, `sweep`) |

## Usage

```bash
# 1. featurise once (~1 min for 8291 pages on 14 cores), cached under structural/cache/
python -m structural.train prep --data ../27_08_01_en_cz.jsonl --workers 14

# 2. feature ablation on all three evaluation splits
python -m structural.train train --data ../27_08_01_en_cz.jsonl --ablation

# 3. hyper-parameter sweep; the best config by `combine()` is refit on train+val and saved
python -m structural.train train --data ../27_08_01_en_cz.jsonl --sweep \
    --min-df 4,10,25 --C 1,4,16 --score-norm none,doc_z --tag sweep \
    --save models/structural_v1.joblib

# 4. read what the model keys on
python -m structural.train explain --model models/structural_v1.joblib --labels Shopping,News,Food

# 5. score a page
python -m structural.train predict --model models/structural_v1.joblib --url https://example.cz/
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
| `url` | url path + host tokens | **yes** | +0.02 in-domain, nothing cross-lingual |
| `words` | visible text words | **yes** — baseline only | the ceiling, see below |

`url` and `words` exist so the ablation can show what language independence *costs*;
they are excluded from the shipped model.

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

## Live spot check (out of distribution, honest version)

| URL | prediction |
|---|---|
| `recepty.cz` | Food 0.98 ✓ |
| `idnes.cz` | News 0.96 ✓ |
| `svethardware.cz` | Entertainment 0.82, Computers 0.75 (half right) |
| `novinky.cz` | Computers 0.57 ✗ (News) |
| `bbc.com/news` | Shopping 0.34 ✗ |

Czech sites score well, recent English sites do not: the training corpus is old, so
modern JS-heavy layouts are off-distribution. Several hosts also refuse a plain
`urllib` fetch (403/402), which `predict` now reports as one line instead of a traceback.

## Limitations

- **Text beats structure in this corpus**, in-domain (0.758 vs 0.495 micro) *and*
  cross-lingually (0.497/0.605 vs 0.323/0.281). Language independence is a property of
  the features, but it is not the same as transfer quality: the Czech and English halves
  differ in vintage, CMS generation and topic mix as well as in language, so `cs2en` /
  `en2cs` are a pessimistic bound on language independence, not a clean ablation of it.
- Structural features describe **how a page is built**; they separate shops, forums,
  portals and article templates well and separate cuisines, car brands or holiday
  destinations poorly. That is a ceiling on the approach, not a tuning defect.
- `marker` is a deliberate exception to the rule: a small bilingual keyword list. It is
  worth +0.002 micro, i.e. nothing, so the model does not lean on it — remove the group
  if you want a purer claim.
- Thresholds are tuned per-split; a model applied to a new corpus should recalibrate.
- **Upgrade path** if you want the content signal *and* language independence: concatenate
  these features with a multilingual sentence embedding of the visible text
  (`laense_positional`, `multilingual-e5`, …). `Vectorizer.transform` is the only place
  that would change; the rest of the pipeline is representation-agnostic.
