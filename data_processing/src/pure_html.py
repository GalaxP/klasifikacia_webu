#!/usr/bin/env python3
"""
Extrakcia čistého textu z HTML v NDJSON súbore (robustná verzia).
Používa tolerantný parser lxml (alebo html5lib), ošetruje chyby parsovania.
"""

import argparse
import json
import re
import sys

from bs4 import BeautifulSoup
import html2text

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Vyberieme najlepší dostupný parser
def get_parser():
    for parser in ['lxml', 'html5lib', 'html.parser']:
        try:
            BeautifulSoup("<html></html>", parser)
            return parser
        except Exception:
            continue
    return 'html.parser'  # fallback, aj keď môže padať


PARSER = get_parser()


def clean_html(html: str) -> str:
    """
    Odstráni neplatné HTML znakové referencie (napr. &#... bez ;),
    ktoré by mohli rozbiť parser.
    """
    # Odstránenie neplatných číselných/html referencií, ktoré nie sú ukončené ;
    # Vzor: &# nasledované znakmi, ktoré nie sú číslice alebo #x a bez ;
    html = re.sub(r'&#\w*[^\d;]', '', html)
    # Odstránenie osamotených &
    # html = re.sub(r'&(?![a-zA-Z]+;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', html)
    return html


def extract_text(html_content: str, max_chars: int = 8000) -> str:
    """
    Extrahuje čistý text z HTML. Ak parsovanie zlyhá, vráti prázdny reťazec.
    """
    # Voliteľné predčistenie
    html_content = clean_html(html_content)

    try:
        soup = BeautifulSoup(html_content, PARSER)
    except Exception as e:
        # V prípade fatálnej chyby parsovania (málo pravdepodobné s lxml)
        print(f"Chyba parsovania HTML: {e}")
        return ""

    # Odstránenie nežiaducich tagov
    for tag in soup(['script', 'style', 'head', 'noscript', 'meta', 'link']):
        tag.decompose()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0
    h.unicode_snob = True

    text = h.handle(str(soup))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)[:max_chars]


def process_file(input_path: str, output_path: str, min_text_len: int = 0, max_chars: int = 4000) -> None:
    processed = 0
    skipped_no_html = 0
    skipped_short = 0
    skipped_parse_error = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        iterator = fin
        if tqdm is not None:
            iterator = tqdm(fin, desc="Extracting text", unit=" lines")

        for line_num, line in enumerate(iterator, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if tqdm is not None:
                    iterator.write(f"JSON chyba riadok {line_num}: {e}")
                continue

            html = obj.get('html')
            if not html or not isinstance(html, str):
                skipped_no_html += 1
                continue

            text = extract_text(html, max_chars)
            if not text:
                skipped_parse_error += 1
                continue
            if len(text) < min_text_len:
                skipped_short += 1
                continue

            out = {
                'url': obj.get('url'),
                'category': obj.get('category'),
                'text': text
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            processed += 1

    print(f"""
Spracovanie dokončené.
Spracovaných záznamov:         {processed}
Preskočených (bez html):       {skipped_no_html}
Preskočených (chyba parsovania): {skipped_parse_error}
Preskočených (krátky text):    {skipped_short}
""")


def main():
    parser = argparse.ArgumentParser(description='Extrakcia textu z HTML (robustná)')
    parser.add_argument('input', help='Vstupný NDJSON súbor')
    parser.add_argument('output', help='Výstupný NDJSON súbor')
    parser.add_argument('--min-text-len', type=int, default=0,
                        help='Minimálna dĺžka textu pre zahrnutie')
    parser.add_argument('--max-chars', type=int, default=8000,
                        help='Maximálny počet znakov textu')
    args = parser.parse_args()

    process_file(args.input, args.output, args.min_text_len, args.max_chars)


if __name__ == '__main__':
    main()