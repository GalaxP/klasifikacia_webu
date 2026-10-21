"""Language-agnostic feature extraction from raw HTML.

The whole point of this module is that it never looks at *words*.  A page about
car engines in Czech and a page about car engines in English should land on the
same point in feature space, so everything is derived from the document tree
(tags, attributes, template repetition, resource names, numeric layout) plus a
script-independent shape profile of the text (how long are the words, how dense
are digits and punctuation) rather than their content.

Each extractor returns tokens for a named *group* so that the training script
can ablate whole groups at once (see GROUPS):

    tag   DOM structure            t:div, p:div>ul>li, d:depth=4
    attr  attribute usage          a:rel=nofollow, i:input[type=submit]
    href  link syntax, no words    H:ext=php, H:key=id, H:numseg, H:depth=3
    meta  head/metadata fingerprints m:og:og:image, s:schema_product
    plat  platform/tech fingerprints P:wp, P:magento, r:ext:swf
    res   resource file naming     r:cloud-zoom, r:produkt_list
    css   class / id name parts    c:product, c:add-to-cart
    cssraw whole class strings     C:product-item_price, C:col-3 v2
    marker raw-substring content hints  M:cart, M:price, M:recipe  <-- keyword lists,
                                    only partly language-free (see RAW_MARKERS)
    shape script-independent text shape  n:wordlen=7, n:cc=cur_sign
    url   url path tokens (language dependent, off by default)
    words visible text words (language dependent, baseline only)
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from lxml import html as lhtml
from lxml import etree

# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #

# Ordered roughly from most generic (survives any site/language) to most
# site-specific (`res` file stems and raw class strings are the ones that can
# overfit to a single crawler snapshot or CMS vintage).
STRUCT_GROUPS = ("tag", "attr", "href", "meta", "marker", "plat", "res", "css",
                 "cssraw", "shape")
ALL_GROUPS = STRUCT_GROUPS + ("url", "words")

# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

_CHARSET_RE = re.compile(rb"""charset[^a-zA-Z0-9]{0,3}\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.I)
_COMMENT_RE = re.compile(rb"<!--.*?-->", re.S)

# Script bodies are ~a third of this corpus and are pure noise for structure, but
# their *opening tags* carry signal (`src`, `type="application/ld+json"`), so the
# attributes are always kept and only JSON-LD / CSS bodies survive.
_NOISE_TEXT = {"script", "style"}

_MAX_HTML_BYTES = 1_500_000  # cap on the tree size we are willing to parse


def decode_html(raw: bytes) -> str:
    """Best-effort decode. 2004-era corpora mix utf-8, cp1250 and iso-8859-2."""
    for enc in ("utf-8",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    m = _CHARSET_RE.search(raw[:4096])
    if m:
        enc = m.group(1).decode("ascii", "ignore").lower()
        enc = {"windows-1250": "cp1250", "cp1250": "cp1250", "iso-8859-2": "iso8859-2",
               "latin2": "iso8859-2", "windows-1252": "cp1252"}.get(enc, enc)
        try:
            return raw.decode(enc, "replace")
        except LookupError:
            pass
    try:
        return raw.decode("cp1250", "replace")
    except LookupError:  # pragma: no cover
        return raw.decode("latin-1", "replace")


_BODY_CAP = 40_000


def _empty_bodies(raw: bytes, tag: bytes, keep) -> bytes:
    """Replace the body of every <tag> with nothing, unless `keep(opening_tag)`.

    The opening tag is preserved verbatim — dropping it along with the body also
    drops `src` and `type`, which is where the external-script and JSON-LD
    features live.
    """
    pat = re.compile(b"(<" + tag + rb"\b[^>]*>)(.*?)(?:</" + tag + rb">|\Z)",
                     re.I | re.S)

    def rep(m):
        body = m.group(2) if keep(m.group(1).lower()) else b""
        return m.group(1) + (body or b"")[:_BODY_CAP] + b"</" + tag + b">"

    return pat.sub(rep, raw)


def strip_noise(raw: bytes) -> bytes:
    """Drop JS bodies and comments; keep tag attributes, JSON-LD and CSS bodies."""
    raw = _empty_bodies(raw, b"script", lambda head: b"ld+json" in head)
    raw = _empty_bodies(raw, b"style", lambda head: True)
    raw = _COMMENT_RE.sub(b"", raw)
    return raw


def parse_tree(raw: bytes):
    """Return an lxml element for the document body, or None if unrecoverable."""
    if len(raw) > _MAX_HTML_BYTES:
        raw = raw[:_MAX_HTML_BYTES]
    text = strip_noise(raw)
    if not text.strip():
        return None
    try:
        doc = lhtml.document_fromstring(text)
    except Exception:
        try:
            doc = lhtml.fromstring(text)
        except Exception:
            return None
    if doc is None:
        return None
    # garbage pages (empty redirect stubs, feeds) still parse but give nothing
    if len(doc) == 0 and not (doc.text or "").strip():
        return None
    return doc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def name_tokens(name: str, max_len: int = 40) -> list[str]:
    """`product-Item_price v2` -> ['product','item','price','product-item','item-price'].

    Kept whole as well as split: `add-to-cart` carries more meaning than its
    parts, and CMS naming is English-flavoured in every language.
    """
    if not name:
        return []
    name = name.strip()
    if len(name) > max_len:
        name = name[:max_len]
    whole = _SPLIT_RE.sub(" ", _CAMEL_RE.sub(" ", name.lower())).split()
    whole = [w for w in whole if len(w) > 2 and not w.isdigit()]
    if not whole:
        return []
    out = list(whole)
    joined = "-".join(whole)
    if "-" in joined and len(joined) <= max_len:
        out.append(joined)
    return out


def url_tokens(url: str) -> dict[str, list[str]]:
    """Split an href/src into {host_tokens, path_tokens, file_tokens}."""
    out = {"host": [], "path": [], "file": []}
    if not url:
        return out
    try:
        parts = urlsplit(url if "://" in url or url.startswith("//") else "//" + url)
    except ValueError:
        return out
    host = (parts.hostname or "").lower()
    if host:
        labels = [l for l in host.split(".") if l and l not in
                  ("www", "com", "cz", "org", "net", "uk", "de", "eu", "co", "gov", "edu", "sk")]
        for l in labels:
            out["host"].extend(name_tokens(l))
    path = parts.path or ""
    segs = [s for s in path.split("/") if s]
    for s in segs[:-1]:
        stem, _, ext = s.rpartition(".")
        out["path"].extend(name_tokens(stem or s))
    if segs:
        last = segs[-1]
        stem, _, ext = last.rpartition(".")
        if stem:
            out["file"].extend(name_tokens(stem))
            if ext and len(ext) <= 5:
                out["file"].append("ext:" + ext.lower())
    return out


def _entropy(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


# --------------------------------------------------------------------------- #
# known fingerprints (language independent identifiers)
# --------------------------------------------------------------------------- #

# substring -> tag, matched against raw lowercased html; every one of these is
# a proper noun or a technical identifier, so it survives translation.
PLATFORMS = {
    "wp-content": "wp", "wp-includes": "wp", "wp-json": "wp",
    "joomla": "joomla", "typo3": "typo3", "drupal": "drupal",
    "vbulletin": "vbulletin", "phpbb": "phpbb", "smf论坛": "smf",
    "oscommerce": "oscommerce", "magento": "magento", "prestashop": "prestashop",
    "woocommerce": "woocommerce", "shopify": "shopify", "zen-cart": "zencart",
    "heureka": "heureka", "googleapis": "googleapis", "google-analytics": "gapi",
    "googlesyndication": "gads", "doubleclick": "doubleclick", "adsense": "gads",
    "facebook": "facebook", "twitter": "twitter", "youtube": "youtube",
    "vimeo": "vimeo", "dailymotion": "dailymotion", "flash": "flash",
    "jquery": "jquery", "bootstrap": "bootstrap", "extjs": "extjs",
    "paypal": "paypal", "google.com/maps": "gmaps", "openstreetmap": "osm",
    "disqus": "disqus", "hotjar": "hotjar", "scorecardresearch": "comscore",
    "sessiontracker": "sessiontracker", "gemius": "gemius", "piwik": "piwik",
    "matomo": "matomo", "gravatar": "gravatar", "cloudflare": "cloudflare",
    "fonts.googleapis": "gfonts", "typekit": "typekit", "recaptcha": "recaptcha",
    "stripe.com": "stripe", "braintree": "braintree", "mailchimp": "mailchimp",
    "feedburner": "feedburner", "rss": "rss", "atom.xml": "atom",
    "twitter.com/intent": "tw_intent", "plusone": "gplus",
    "sharethis": "sharethis", "addthis": "addthis", "printfriendly": "printfriendly",
    "addtoany": "addtoany", "akismet": "wp", "cdn.": "cdn",
}

# Substrings that betray what a page is *for*.  Matched against the raw text, so
# unlike everything else here this is a keyword list and only partly language-free:
# it mixes the English web (add to cart, ingredients, specification) with the Czech
# one (do košíku, cena:, technické parametry) because that is what this corpus is.
# Extend it per deployment language -- or drop the `marker` group to run without it.
RAW_MARKERS = {
    "age_gate": ("age verification", "age-gate", "agegate", "18+ only", "vetujem"),
    "cart": ("add to cart", "add-to-cart", "addtocart", "do kosiku", "kosik",
             "shopping cart", "cart_item", "basket"),
    "price": ("price:", "cena:", "class=\"price", "itemprop=\"price", "product-price"),
    "recipe": ("ingredients", "prísady", "prisady", "postup", "instructions:",
               "nutritional", "serving size"),
    "reviews": ("review", "hodnoceni", "hodnocení", "rating", "stars", "zaznamek"),
    "specs": ("specification", "technicke parametry", "technické parametry",
              "parameters", "kapacita", "rozmery"),
    "score_table": ("plused", "minused", "vs.", "zapas", "zápas", "result"),
    "forum": ("reply", "odpoved", "odpověď", "thread", "topic", "diskuse",
              "diskuse", "príspevok"),
    "job": ("salary", "plat:", "nabidka prace", "nabídka práce", "curriculum"),
    "events": ("event", "udalost", "událost", "calendar", "kalendar", "vstupenky",
               "ticket"),
}


_EXT_HREF_RE = re.compile(r"\.([A-Za-z0-9]{1,5})(?:[?#]|$)")
_NUMSEG_RE = re.compile(r"/\d{1,8}(?:\.|$|/)")


def href_shape_tokens(href: str) -> list[str]:
    """Tokens describing *how* a link is written, never what the words mean."""
    out = []
    if not href:
        return out
    m = _EXT_HREF_RE.search(href)
    if m:
        out.append(f"H:ext={m.group(1).lower()}")
    q = href.partition("?")[2]
    if q:
        out.append(f"H:q={min(len(q.split('&')), 6)}")
        for pair in q.split("&")[:6]:
            key = pair.split("=")[0].lower()
            if len(key) <= 16:
                out.append("H:key=" + key)
    if "#" in href:
        out.append("H:frag")
    if _NUMSEG_RE.search(href):
        out.append("H:numseg")
    if href.endswith("/"):
        out.append("H:slash")
    if href.startswith("mailto:"):
        out.append("H:mailto")
    elif href.startswith("javascript:"):
        out.append("H:jscript")
    if any(c in href for c in "{}") or "%d" in href:
        out.append("H:template")
    segs = [x for x in href.split("?")[0].split("/") if x]
    out.append(f"H:depth={min(max(len(segs) - 2, 0), 8)}")
    return out


# --------------------------------------------------------------------------- #
# main extractor
# --------------------------------------------------------------------------- #

_TAG_SET = set("""a abbr address area article aside audio b base bdi bdo big blockquote body br
button canvas caption center cite code col colgroup data datalist dd del details dfn dialog div dl
dt em embed fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hgroup hr html i
iframe img input ins kbd label legend li link main map mark meta meter nav noscript object ol
optgroup option output p param pre progress q rp rt ruby s samp script section select small source
span strike strong style sub summary sup table tbody td textarea tfoot th thead time title tr tt
u ul var video frame frameset noframes applet plaintext basefont dir isindex blink marquee
font center nobr layer ilayer comment""".split())


def _qname(el) -> str:
    t = el.tag
    return t if isinstance(t, str) else "_comment"


def _walk(doc):
    """Iterate (element, depth, parent_tag, position) in document order."""
    stack = [(doc, 0, "#doc", 0)]
    while stack:
        el, depth, ptag, pos = stack.pop()
        yield el, depth, ptag, pos
        children = [c for c in el.iterchildren() if isinstance(c.tag, str)]
        for i, c in enumerate(reversed(children)):
            stack.append((c, depth + 1, _qname(el), len(children) - i))


def _repeat_stats(doc) -> tuple[float, list[str], float, float]:
    """(fraction of elements inside a repeated run, shape tokens, biggest run, #runs)

    A run of >=3 siblings with the same shape is a list: product grid, article
    list, forum rows, calendar.  Prose pages barely have any.  The shape token
    records *what* repeats, which is language-free (`li:a>img+p` is a thumbnail
    grid in any language).
    """
    total = 0
    repeated = 0
    runs = 0
    biggest = 0
    shapes: Counter = Counter()
    for parent in doc.iter():
        if not isinstance(parent.tag, str):
            continue
        kids = [c for c in parent if isinstance(c.tag, str)]
        total += len(kids)
        if len(kids) < 3:
            continue
        sigs = Counter()
        for k in kids:
            kids2 = "+".join(_qname(c) for c in k if isinstance(c.tag, str))[:40]
            sigs[(_qname(k), kids2)] += 1
        for (tag, kids2), n in sigs.items():
            if n >= 3:
                repeated += n
                runs += 1
                biggest = max(biggest, n)
                shapes[f"R:{tag}>{kids2 or 'text'}"] += 1
    toks = [tok for tok, _ in shapes.most_common(4)]
    ratio = repeated / total if total else 0.0
    return ratio, toks, float(biggest), float(runs)


def _css_rule_count(css_text: str) -> int:
    return css_text.count("{")





def visible_text(doc) -> str:
    """Text a reader would see: skips <script>/<style> bodies, keeps their tails."""
    parts = []
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag not in _NOISE_TEXT and el.text:
            parts.append(el.text)
        if el.tail:
            parts.append(el.tail)
    return " ".join(parts)


def _text_shape(doc) -> tuple[dict, list[str]]:
    """Shape of the visible text, with the letters themselves thrown away."""
    text = visible_text(doc)
    low = text.lower()
    n_chars = len(low)
    if n_chars == 0:
        return {}, []
    words = low.split()
    n_words = len(words) or 1
    letters = sum(1 for ch in low if ch.isalpha())
    digits = sum(1 for ch in low if ch.isdigit())
    spaces = low.count(" ")
    punct = sum(1 for ch in low if unicodedata.category(ch).startswith("P"))
    nonlatin = sum(1 for ch in low if ord(ch) > 0x2100 and not ch.isspace())
    # word length histogram -- the single most transferable "text" statistic
    wl = Counter(min(len(w), 15) for w in words)
    # which character classes the writing system uses, language-agnostic
    classes = {
        "letter": letters, "digit": digits, "punct": punct, "space": spaces,
        "upper": sum(1 for ch in text if ch.isupper()),
        "nonlatin": nonlatin,
        "curly_quote": low.count("“") + low.count("”"),
        "guillemet": low.count("«") + low.count("»"),
        "dashes": low.count("–") + low.count("—"),
        "diacritic": sum(1 for ch in low if unicodedata.normalize("NFD", ch)[:1] != ch
                         and ch.isalpha()),
    }
    nums = len(re.findall(r"\d+\s*[.,]\s*\d+", low))
    big_nums = len(re.findall(r"\d{1,3}(?:[\s .]\d{3})+", low))
    percents = low.count("%")
    toks = []
    for ln, c in wl.items():
        if c >= 2:
            toks.append(f"n:wordlen={ln}")
    for k, v in classes.items():
        r = v / n_chars
        if r > 0.002:
            toks.append(f"n:cc={k}={int(min(9, r * 40))}")
    if nums:
        toks.append("n:numberish")
    if big_nums:
        toks.append("n:thousands_sep")
    stats = {
        "sh_text_chars": n_chars,
        "sh_n_words": n_words,
        "sh_avg_word_len": letters / n_words,
        "sh_digit_ratio": digits / n_chars,
        "sh_punct_ratio": punct / n_chars,
        "sh_space_ratio": spaces / n_chars,
        "sh_upper_ratio": classes["upper"] / max(1, letters),
        "sh_diacritic_ratio": classes["diacritic"] / max(1, n_chars),
        "sh_nonlatin_ratio": nonlatin / n_chars,
        "sh_decimal_number_density": nums / n_words,
        "sh_thousands_sep_density": big_nums / n_words,
        "sh_percent_density": percents / n_words,
        "sh_letters_per_element": letters,
    }
    return stats, toks


# Bump whenever the meaning of any feature or token changes, so a stale feature
# cache is never silently reused (`data.load` hashes this into the cache key).
SCHEMA = "2026-10-21.b"

# canonical order of dense features -- built once, saved with the model
_NUM_RE = re.compile(r"\d")

_NUM_ORDER = [
    "html_bytes", "n_elements", "n_nodes", "depth_max", "depth_mean", "depth_p90",
    "children_mean", "children_max", "n_unique_tags", "tag_entropy", "tag_top1_share",
    "leaf_ratio", "n_text_chars", "n_words", "text_per_element", "n_links",
    "link_per_element", "link_text_ratio", "n_img", "img_per_link", "img_alt_share",
    "img_with_dims_share", "n_table", "table_per_element", "n_tr", "tr_per_table",
    "max_table_cols", "n_form", "n_input", "n_select", "n_textarea", "n_button",
    "n_script", "n_ext_script", "n_style_el", "n_css_link", "css_rules",
    "style_attr_share", "n_meta", "n_head_link", "n_external_domains",
    "external_link_share", "avg_path_depth", "query_link_share", "n_iframes",
    "n_embed", "n_video", "n_audio", "n_object", "n_br", "n_hr", "n_p",
    "p_per_element", "n_li", "li_per_list", "n_dl", "n_heading", "heading_per_element",
    "anchor_words_p50", "anchor_numeric_share", "anchor_imgonly_share",
    "anchor_empty_share", "anchor_title_share", "href_keys", "repeat_biggest",
    "repeat_runs", "content_start_ratio", "title_len", "meta_kw_n",
    "n_h1", "n_div", "div_share", "span_share", "a_share", "n_bold", "n_small",
    "n_pre", "n_blockquote", "n_comment_el", "repeat_ratio", "meta_desc_len", "n_meta_desc",
    "n_classes_unique",
    "n_class_per_element", "n_ids", "unique_id_ratio", "n_inline_color",
    "n_bg_image", "n_font_size_small", "has_frameset", "has_lang", "has_charset",
    "has_viewport", "has_canonical", "has_og", "has_twitter", "has_microdata",
    "has_jsonld", "has_rdf", "n_rss_link", "n_schema_types", "n_input_hidden",
    "n_input_submit", "n_input_text", "n_input_search", "n_input_checkbox",
    "n_forms_get", "n_forms_post", "has_login_form", "has_newsletter",
    "has_price_marker", "has_cart_marker", "n_platforms", "n_raw_markers",
    "text_per_node", "avg_words_per_p",
    # text shape (script-independent, no words)
    "sh_text_chars",
    "sh_n_words",
    "sh_avg_word_len",
    "sh_digit_ratio",
    "sh_punct_ratio",
    "sh_space_ratio",
    "sh_upper_ratio",
    "sh_diacritic_ratio",
    "sh_nonlatin_ratio",
    "sh_decimal_number_density",
    "sh_thousands_sep_density",
    "sh_percent_density",
    "sh_letters_per_element",
]


def extract(raw_html: bytes, url: str | None = None,
            groups=STRUCT_GROUPS) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Return ({group: [tokens]}, {numeric feature: value})."""
    doc = parse_tree(raw_html)
    if doc is None:
        return {g: [] for g in groups}, {}

    want_url = "url" in groups
    want_words = "words" in groups
    toks: dict[str, list[str]] = {g: [] for g in groups}
    t = toks.get("tag", [])
    at = toks.get("attr", [])
    hs = toks.get("href", [])
    me = toks.get("meta", [])
    mk = toks.get("marker", [])
    plat = toks.get("plat", [])
    res = toks.get("res", [])
    cs = toks.get("css", [])
    csraw = toks.get("cssraw", [])
    sh = toks.get("shape", [])

    raw_low = None  # lazily decoded raw html, only if a fingerprint lookup happens

    tag_counts: Counter = Counter()
    attr_counts: Counter = Counter()
    classes: Counter = Counter()
    ids: Counter = Counter()
    depths: list[int] = []
    child_counts: list[int] = []
    leaves = 0
    n_nodes = 0
    n_elements = 0
    text_chars = 0
    link_text_chars = 0

    n_links = n_img = n_img_alt = n_img_dims = 0
    n_table = n_tr = n_form = n_input = n_select = n_textarea = n_button = 0
    n_script = n_ext_script = n_style_el = n_css_link = 0
    css_rule_total = 0
    style_attr = 0
    n_meta = n_head_link = 0
    n_iframes = n_embed = n_video = n_audio = n_object = 0
    n_br = n_hr = n_p = n_li = n_dl = n_heading = n_h1 = 0
    n_div = n_span = n_a = n_bold = n_small = n_pre = n_bq = 0
    n_comment_el = 0
    n_inline_color = n_bg_image = n_font_small = 0
    max_table_cols = 0
    n_class_per_element = 0
    input_types: Counter = Counter()
    form_methods: Counter = Counter()
    has_login = has_newsletter = False
    li_per_list_num = li_per_list_den = 0
    words_in_p: list[int] = []
    external_links = internal_links = 0
    href_shape: list[str] = []
    anchor_words: list[int] = []
    n_anchor_numeric = n_anchor_empty = n_anchor_imgonly = n_anchor_title = 0
    links_before_content = None
    path_depths: list[int] = []
    query_links = 0
    total_links_href = 0
    domains: set[str] = set()

    base_host = ""
    if url:
        try:
            base_host = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower()
        except ValueError:
            base_host = ""

    for el, depth, ptag, pos in _walk(doc):
        n_nodes += 1
        tag = _qname(el)
        if tag == "_comment":
            n_comment_el += 1
            continue
        n_elements += 1
        depths.append(depth)
        tag_counts[tag] += 1
        t.append(f"t:{tag}")
        t.append(f"p:{ptag}>{tag}")
        t.append(f"d:{tag}@{min(depth, 8)}")
        kids = [c for c in el if isinstance(c.tag, str)]
        child_counts.append(len(kids))
        if not kids and not (el.text or "").strip():
            leaves += 1
        if tag not in _NOISE_TEXT:
            if el.text:
                text_chars += len(el.text)
            if el.tail:
                text_chars += len(el.tail)

        attrs = el.attrib
        if attrs:
            attr_names = sorted(attrs.keys())
            for a in attr_names:
                at.append(f"a:{tag}[{a}]")
                attr_counts[a] += 1
            if "style" in attrs:
                style_attr += 1
                st = attrs["style"] or ""
                if "background-image" in st or "url(" in st:
                    n_bg_image += 1
                if "color" in st:
                    n_inline_color += 1
                if "font-size" in st and ("px" in st or "pt" in st):
                    n_font_small += 1
            if "class" in attrs:
                raw_cls = attrs["class"] or ""
                cls_tokens = []
                for c in raw_cls.split():
                    if len(c) > 30:  # generated hashes
                        continue
                    cls_tokens.append(c)
                    cs.extend(f"c:{x}" for x in name_tokens(c))
                if len(cls_tokens) > 1:
                    csraw.append("C:" + "-|-".join(cls_tokens[:3]))
                n_class_per_element += len(cls_tokens)
                classes.update(cls_tokens)
            if "id" in attrs:
                for x in name_tokens(attrs["id"]):
                    cs.append(f"i:{x}")
                ids[attrs["id"]] += 1
        else:
            attr_counts["_none"] += 1

        # ---- per-tag specifics --------------------------------------------
        if tag in ("a", "link"):
            href = attrs.get("href") or ""
            if tag == "a":
                n_a += 1
                if href:
                    total_links_href += 1
                    path_depths.append(href.count("/") - (2 if href.startswith("http") else 0))
                    if "?" in href:
                        query_links += 1
                    try:
                        h = (urlsplit(href if "//" in href else "//" + href).hostname or "").lower()
                    except ValueError:
                        h = ""
                    if h and base_host and not h.endswith(base_host.rsplit(".", 2)[-2] if base_host.count(".") > 1 else base_host):
                        external_links += 1
                        if h:
                            domains.add(h)
                    else:
                        internal_links += 1
                    ut = url_tokens(href)
                    if want_url:
                        toks["url"].extend(f"u:{x}" for x in ut["path"])
                    if want_url:
                        toks["url"].extend(f"uh:{x}" for x in ut["host"])
                    for x in ut["file"]:
                        (plat if x.startswith("ext:") else res).append(f"r:{x}")
                href_shape.extend(href_shape_tokens(href))
                link_text_chars += len((el.text or "").split())
                aw = len((el.text_content() or "").split())
                anchor_words.append(aw)
                if el.text_content() and _NUM_RE.search(el.text_content()):
                    n_anchor_numeric += 1
                if not aw:
                    n_anchor_empty += 1
                elif len([c for c in el if isinstance(c.tag, str) and _qname(c) == "img"]):
                    n_anchor_imgonly += 1
                if attrs.get("title"):
                    n_anchor_title += 1
                rel = (attrs.get("rel") or "").lower()
                if rel:
                    at.append(f"a:rel={rel[:30]}")
                if attrs.get("target"):
                    at.append("a:target")
            else:  # <link>
                n_head_link += 1
                rel = (attrs.get("rel") or "").lower()
                for r in rel.split():
                    me.append(f"m:link_rel={r[:24]}")
                    if r in ("stylesheet",):
                        n_css_link += 1
                    if "alternate" in r:
                        ty = (attrs.get("type") or "").lower()
                        me.append(f"m:alt_type={ty[-16:] if ty else 'none'}")
                        href = attrs.get("href") or ""
                ut = url_tokens(href)
                res.extend(f"r:{x}" for x in ut["file"])

        elif tag == "img":
            n_img += 1
            if attrs.get("alt"):
                n_img_alt += 1
            if attrs.get("width") and attrs.get("height"):
                n_img_dims += 1
            src = attrs.get("src") or ""
            ut = url_tokens(src)
            res.extend(f"r:{x}" for x in ut["file"])
            if want_url:
                toks["url"].extend(f"u:{x}" for x in ut["path"])
            if src.startswith("data:"):
                plat.append("P:datauri")
            w, h = attrs.get("width") or "", attrs.get("height") or ""
            if w and h:
                try:
                    iw, ih = float(re.sub(r"[^0-9.]", "", w) or 0), float(re.sub(r"[^0-9.]", "", h) or 0)
                    if iw and ih:
                        r = iw / ih
                        if r > 2.2:
                            at.append("img:wide")
                        elif r < 0.45:
                            at.append("img:tall")
                        if iw <= 32 and ih <= 32:
                            at.append("img:icon")
                except ValueError:
                    pass

        elif tag == "script":
            n_script += 1
            src = attrs.get("src") or ""
            if src:
                n_ext_script += 1
                ut = url_tokens(src)
                res.extend(f"r:{x}" for x in ut["file"])
                plat.extend(f"P:script_{x}" for x in ut["host"])
                if want_url:
                    toks["url"].extend(f"uh:{x}" for x in ut["host"])
                body = ""
            else:
                body = (el.text or "")
                if "JSON.parse" in body or "dataLayer" in body:
                    me.append("m:js_datalayer")
                if re.search(r"\{\s*[\"']?@type", body):
                    me.append("m:jsonld")

        elif tag == "style":
            n_style_el += 1
            css_rule_total += _css_rule_count(el.text or "")

        elif tag == "meta":
            n_meta += 1
            nm = (attrs.get("name") or "").lower()
            prop = (attrs.get("property") or "").lower()
            if nm:
                me.append(f"m:meta={nm[:24]}")
            if prop:
                me.append(f"m:og={prop[:28]}")
            if attrs.get("http-equiv"):
                me.append(f"m:http_equiv={(attrs.get('http-equiv') or '').lower()[:18]}")
            if attrs.get("itemprop"):
                for x in (attrs.get("itemprop") or "").split():
                    me.append(f"m:itemprop={x.lower()[:24]}")

        elif tag == "table":
            n_table += 1
        elif tag == "tr":
            n_tr += 1
            cols = len([c for c in el if isinstance(c.tag, str) and _qname(c) in ("td", "th")])
            if cols > max_table_cols:
                max_table_cols = cols
            at.append(f"x:td={min(cols, 12)}")
        elif tag == "form":
            n_form += 1
            act = (attrs.get("action") or "").lower()
            method = (attrs.get("method") or "get").lower()
            form_methods[method] += 1
            me.append(f"m:form_action={':'.join(name_tokens(act)[:3]) or 'none'}")
            for x in name_tokens(act)[:3]:
                cs.append(f"fa:{x}")
            cls = (attrs.get("class") or "").lower() + " " + (attrs.get("id") or "").lower()
            if "search" in cls or "search" in act or "hledav" in act or "hledaj" in act:
                cs.append("fa:search")
            if any(k in act or k in cls for k in ("login", "prihl", "signin", "auth")):
                has_login = True
            if any(k in act or k in cls for k in ("newsletter", " Zasylk", "zasylka", "subscribe", "news")):
                has_newsletter = True
        elif tag == "input":
            n_input += 1
            ty = (attrs.get("type") or "text").lower()
            input_types[ty] += 1
            at.append(f"x:input_type={ty}")
            nm = (attrs.get("name") or "").lower()
            for x in name_tokens(nm)[:3]:
                cs.append(f"in:{x}")
            if "search" in nm:
                cs.append("in:search")
        elif tag == "select":
            n_select += 1
        elif tag == "textarea":
            n_textarea += 1
        elif tag == "button":
            n_button += 1
        elif tag == "iframe":
            n_iframes += 1
            src = attrs.get("src") or ""
            for k in ("youtube", "google", "facebook", "twitter", "vimeo", "map"):
                if k in src:
                    plat.append(f"P:iframe_{k}")
        elif tag == "embed":
            n_embed += 1
        elif tag == "object":
            n_object += 1
        elif tag in ("video",):
            n_video += 1
        elif tag in ("audio",):
            n_audio += 1
        elif tag == "br":
            n_br += 1
        elif tag == "hr":
            n_hr += 1
        elif tag == "p":
            n_p += 1
            w = len((el.text_content() or "").split())
            words_in_p.append(w)
            # share of the page's links that sit before the first real paragraph:
            # high on template-heavy portals, low on articles
            if w >= 40 and links_before_content is None:
                links_before_content = n_a
        elif tag == "li":
            n_li += 1
        elif tag in ("ul", "ol", "dl"):
            li_per_list_num += len([c for c in el if isinstance(c.tag, str)])
            li_per_list_den += 1
            if tag == "dl":
                n_dl += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            n_heading += 1
            if tag == "h1":
                n_h1 += 1
            at.append(f"h:{tag}")
        elif tag == "div":
            n_div += 1
        elif tag == "span":
            n_span += 1
        elif tag in ("b", "strong", "em", "i"):
            n_bold += 1
        elif tag == "small":
            n_small += 1
        elif tag == "pre":
            n_pre += 1
        elif tag == "blockquote":
            n_bq += 1
        elif tag in ("frame", "frameset"):
            at.append("x:frames")
        elif tag == "font":
            at.append("x:font_tag")

    # ---- head-level metadata ------------------------------------------------
    head = doc.find(".//head")
    has_lang = doc.get("lang") is not None or doc.get("{http://www.w3.org/XML/1998/namespace}lang") is not None
    has_viewport = False
    has_canonical = False
    has_og = has_microdata = has_jsonld = has_charset = has_rdf = False
    n_rss = 0
    schema_types: Counter = Counter()
    title_len = meta_desc_len = meta_kw_n = 0
    title_el = doc.find(".//title")
    if title_el is not None and title_el.text:
        title_len = len(title_el.text.strip())
    first_big_text = -1
    for m in doc.iter("meta"):
        nm = (m.get("name") or "").lower()
        prop = (m.get("property") or "").lower()
        if prop.startswith(("og:", "twitter:", "fb:")):
            has_og = True
        if nm == "viewport":
            has_viewport = True
        if m.get("charset") or "charset" in (m.get("content") or "").lower():
            has_charset = True
        if m.get("itemtype"):
            has_microdata = True
        if "generator" in nm:
            me.append(f"m:generator={'_'.join(name_tokens(m.get('content') or '')[:3])}")
    for it in doc.iter():
        if not isinstance(it.tag, str):
            continue
        ty = it.get("itemtype") or ""
        for piece in ty.split():
            if "schema.org/" in piece:
                st = piece.rsplit("/", 1)[-1]
                schema_types[st] += 1
                me.append(f"s:schema_{st.lower()[:26]}")
        if it.get("itemprop"):
            has_microdata = True
    for l in doc.iter("link"):
        rel = (l.get("rel") or "").lower()
        if "canonical" in rel:
            has_canonical = True
        if "alternate" in rel:
            ty = (l.get("type") or "").lower()
            if "rss" in ty or "atom" in ty or "xml" in ty:
                n_rss += 1
        if "alternate" in rel and l.get("hreflang"):
            me.append("m:hreflang")
    for s in doc.iter("script"):
        if (s.get("type") or "").lower() in ("application/ld+json", "application/json+ld"):
            has_jsonld = True
            body = s.text or ""
            for mt in re.finditer(r'"@type"\s*:\s*"([A-Za-z]{2,30})"', body):
                schema_types[mt.group(1)] += 1
                me.append(f"s:ld_{mt.group(1).lower()[:26]}")

    # ---- raw fingerprints ---------------------------------------------------
    try:
        raw_low = decode_html(raw_html).lower()
    except Exception:
        raw_low = ""
    n_platform = 0
    for needle, tag in PLATFORMS.items():
        if needle in raw_low:
            n_platform += 1
            plat.append(f"P:{tag}")
            if needle in ("flash", ".swf"):
                at.append("x:flash")
    n_markers = 0
    for marker, needles in RAW_MARKERS.items():
        if any(n in raw_low for n in needles):
            n_markers += 1
            mk.append(f"M:{marker}")
    has_rdf = "xmlns:rdf" in raw_low or "rdf:RDF" in raw_low
    use_markers = "marker" in groups
    has_price = "M:price" in mk and use_markers
    has_cart = "M:cart" in me
    if "<frameset" in raw_low:
        at.append("x:frameset_raw")

    # ---- token groups that must be assembled in bulk -----------------------
    if "tag" in groups:
        top = [tg for tg, _ in tag_counts.most_common(12)]
        for tg in top:
            t.append(f"T:{tg}")
        t.append(f"X:uniq_tags={min(len(tag_counts), 40)}")
        t.append(f"X:depth={min(max(depths, default=0), 18)}")
    if "css" in groups and len(classes) > 1:
        # structural id of the template: which named sections exist
        for c in list(classes)[:200]:
            csraw.append(f"C:{c.lower()[:24]}")
        if any(n > 1 for n in ids.values()):
            csraw.append("X:duplicate_ids")
    if "href" in groups:
        hs.extend(href_shape)
    if "shape" in groups:
        shape_stats, shape_toks = _text_shape(doc)
        sh.extend(shape_toks)
    else:
        shape_stats = {}
    rep_ratio, rep_toks, rep_big, rep_runs = _repeat_stats(doc)
    if "tag" in groups:
        t.extend(rep_toks)
    if want_words:
        # the language-dependent baseline only -- never used by the shipped model
        toks["words"] = re.findall(r"[^\W\d_]{3,}", visible_text(doc).lower(),
                                   re.UNICODE)[:6000]

    # ---- dense features -----------------------------------------------------
    n_el = max(1, n_elements)
    uniq_tag_share = (tag_counts.most_common(1)[0][1] / n_el) if tag_counts else 0.0
    stats: dict[str, float] = {
        "html_bytes": math.log1p(len(raw_html)),
        "n_elements": math.log1p(n_elements),
        "n_nodes": math.log1p(n_nodes),
        "depth_max": max(depths, default=0),
        "depth_mean": sum(depths) / max(1, len(depths)),
        "depth_p90": sorted(depths)[int(0.9 * max(0, len(depths) - 1))] if depths else 0,
        "children_mean": sum(child_counts) / max(1, len(child_counts)),
        "children_max": max(child_counts, default=0),
        "n_unique_tags": len(tag_counts),
        "tag_entropy": _entropy(tag_counts, n_el),
        "tag_top1_share": uniq_tag_share,
        "leaf_ratio": leaves / n_el,
        "n_text_chars": math.log1p(text_chars),
        "n_words": math.log1p(sum(words_in_p)) if words_in_p else 0.0,
        "text_per_element": text_chars / n_el,
        "n_links": math.log1p(n_a),
        "link_per_element": n_a / n_el,
        "link_text_ratio": link_text_chars / max(1, sum(words_in_p) or 1),
        "n_img": math.log1p(n_img),
        "img_per_link": n_img / max(1, n_a),
        "img_alt_share": n_img_alt / max(1, n_img),
        "img_with_dims_share": n_img_dims / max(1, n_img),
        "n_table": math.log1p(n_table),
        "table_per_element": n_table / n_el,
        "n_tr": math.log1p(n_tr),
        "tr_per_table": n_tr / max(1, n_table),
        "max_table_cols": max_table_cols,
        "n_form": math.log1p(n_form),
        "n_input": math.log1p(n_input),
        "n_select": math.log1p(n_select),
        "n_textarea": math.log1p(n_textarea),
        "n_button": math.log1p(n_button),
        "n_script": math.log1p(n_script),
        "n_ext_script": math.log1p(n_ext_script),
        "n_style_el": math.log1p(n_style_el),
        "n_css_link": math.log1p(n_css_link),
        "css_rules": math.log1p(css_rule_total),
        "style_attr_share": style_attr / n_el,
        "n_meta": math.log1p(n_meta),
        "n_head_link": math.log1p(n_head_link),
        "n_external_domains": math.log1p(len(domains)),
        "external_link_share": external_links / max(1, total_links_href),
        "avg_path_depth": (sum(path_depths) / len(path_depths)) if path_depths else 0.0,
        "query_link_share": query_links / max(1, total_links_href),
        "n_iframes": math.log1p(n_iframes),
        "n_embed": math.log1p(n_embed),
        "n_video": math.log1p(n_video),
        "n_audio": math.log1p(n_audio),
        "n_object": math.log1p(n_object),
        "n_br": math.log1p(n_br),
        "n_hr": math.log1p(n_hr),
        "n_p": math.log1p(n_p),
        "p_per_element": n_p / n_el,
        "n_li": math.log1p(n_li),
        "li_per_list": li_per_list_num / max(1, li_per_list_den),
        "n_dl": math.log1p(n_dl),
        "n_heading": math.log1p(n_heading),
        "heading_per_element": n_heading / n_el,
        "n_h1": n_h1,
        "n_div": math.log1p(n_div),
        "div_share": n_div / n_el,
        "span_share": n_span / n_el,
        "a_share": n_a / n_el,
        "n_bold": math.log1p(n_bold),
        "n_small": math.log1p(n_small),
        "n_pre": math.log1p(n_pre),
        "n_blockquote": math.log1p(n_bq),
        "n_comment_el": math.log1p(n_comment_el),
        "anchor_words_p50": (sorted(anchor_words)[len(anchor_words) // 2]
                             if anchor_words else 0.0),
        "anchor_numeric_share": n_anchor_numeric / max(1, n_a),
        "anchor_imgonly_share": n_anchor_imgonly / max(1, n_a),
        "anchor_empty_share": n_anchor_empty / max(1, n_a),
        "anchor_title_share": n_anchor_title / max(1, n_a),
        "href_keys": len({x for x in href_shape if x.startswith("H:key=")}),
        "repeat_runs": rep_runs,
        "repeat_biggest": rep_big,
        "repeat_ratio": rep_ratio,
        "content_start_ratio": (links_before_content if links_before_content is not None
                                else n_a) / max(1, n_a),
        "title_len": title_len,
        "meta_desc_len": meta_desc_len,
        "meta_kw_n": meta_kw_n,
        "n_meta_desc": 1.0 if meta_desc_len else 0.0,
        "n_classes_unique": math.log1p(len(classes)),
        "n_class_per_element": n_class_per_element / n_el,
        "n_ids": math.log1p(len(ids)),
        "unique_id_ratio": len(ids) / max(1, sum(ids.values())),
        "n_inline_color": math.log1p(n_inline_color),
        "n_bg_image": math.log1p(n_bg_image),
        "n_font_size_small": math.log1p(n_font_small),
        "has_frameset": 1.0 if "<frameset" in raw_low else 0.0,
        "has_lang": float(has_lang),
        "has_charset": float(has_charset),
        "has_viewport": float(has_viewport),
        "has_canonical": float(has_canonical),
        "has_og": float(has_og),
        "has_twitter": float("twitter:" in raw_low or "twitter.com" in raw_low),
        "has_microdata": float(has_microdata),
        "has_jsonld": float(has_jsonld),
        "has_rdf": float(has_rdf),
        "n_rss_link": n_rss,
        "n_schema_types": len(schema_types),
        "n_input_hidden": input_types.get("hidden", 0),
        "n_input_submit": input_types.get("submit", 0),
        "n_input_text": input_types.get("text", 0),
        "n_input_search": 1 if any(k for k in input_types if "search" in k) else 0,
        "n_input_checkbox": input_types.get("checkbox", 0),
        "n_forms_get": form_methods.get("get", 0),
        "n_forms_post": form_methods.get("post", 0),
        "has_login_form": float(has_login),
        "has_newsletter": float(has_newsletter),
        "has_price_marker": float(has_price),
        "has_cart_marker": float(has_cart and use_markers),
        "n_platforms": n_platform,
        "n_raw_markers": float(n_markers if use_markers else 0),
        "text_per_node": text_chars / max(1, n_nodes),
        "avg_words_per_p": sum(words_in_p) / max(1, len(words_in_p)),
    }
    stats.update(shape_stats)

    for g in toks:
        if g not in groups:
            toks[g] = []
    return toks, stats
