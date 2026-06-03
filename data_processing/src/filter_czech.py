#!/usr/bin/env python3
"""
Filtruje NDJSON podľa jazyka textu. Ponechá len záznamy, ktorých text je detegovaný ako čeština.
Používa langdetect (predvolene) alebo fasttext (s prepínačom --fasttext).
"""

import argparse
import json
import sys

from langdetect import detect, LangDetectException

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Voliteľne: fasttext model (ak je nainštalovaný)
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


def load_fasttext_model(model_path: str = "lid.176.bin"):
    """
    Stiahne a načíta fasttext model na detekciu jazyka (ak nie je prítomný, stiahne ho).
    """
    import os
    if not os.path.exists(model_path):
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        print(f"Sťahujem fasttext model z {url} ...")
        urllib.request.urlretrieve(url, model_path)
    return fasttext.load_model(model_path)


def detect_lang_fasttext(model, text: str) -> str:
    """Deteguje jazyk pomocou fasttext, vráti kód (napr. '__label__cs')."""
    # fasttext očakáva text bez nových riadkov
    text = text.replace('\n', ' ').strip()
    if not text:
        return None
    labels, scores = model.predict(text, k=1)
    if labels and scores[0] > 0.5:  # minimálna istota
        lang = labels[0].replace('__label__', '')
        return lang
    return None


def detect_lang_langdetect(text: str) -> str:
    """Deteguje jazyk pomocou langdetect, vráti kód (napr. 'cs') alebo None pri chybe."""
    try:
        # Pre veľmi krátke texty môže byť detekcia nespoľahlivá → ošetríme minimálnou dĺžkou
        if len(text.strip()) < 10:
            return None
        return detect(text)
    except LangDetectException:
        return None


def main():
    parser = argparse.ArgumentParser(description='Filtruj NDJSON podľa českého jazyka.')
    parser.add_argument('input', help='Vstupný NDJSON súbor (s položkou "text")')
    parser.add_argument('output', help='Výstupný NDJSON súbor (iba české záznamy)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimálna istota detekcie (0.0-1.0), len pre langdetect sa ignoruje')
    parser.add_argument('--fasttext', action='store_true',
                        help='Použiť fasttext namiesto langdetect (presnejšie)')
    args = parser.parse_args()

    # Inicializácia modelu
    if args.fasttext:
        if not FASTTEXT_AVAILABLE:
            print("Chyba: fasttext nie je nainštalovaný. Spustite: pip install fasttext")
            sys.exit(1)
        model = load_fasttext_model()
        detect_func = lambda text: detect_lang_fasttext(model, text)
    else:
        # langdetect nepotrebuje model
        detect_func = detect_lang_langdetect
        # langdetect nevracia istotu, takže min-confidence sa ignoruje

    processed = 0
    kept = 0
    skipped_not_czech = 0
    skipped_error = 0

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:

        iterator = fin
        if tqdm is not None:
            iterator = tqdm(fin, desc="Detecting language", unit=" lines")

        for line_num, line in enumerate(iterator, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if tqdm is not None:
                    iterator.write(f"JSON chyba riadok {line_num}: {e}")
                skipped_error += 1
                continue

            text = obj.get('text')
            if not text or not isinstance(text, str):
                skipped_error += 1
                continue

            lang = detect_func(text)

            if lang == 'cs' or (args.fasttext and lang == '__label__cs'):
                # Ponecháme
                out = {
                    'url': obj.get('url'),
                    'category': obj.get('category'),
                    'text': text
                }
                fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                kept += 1
            else:
                skipped_not_czech += 1

            processed += 1

    print(f"""
Hotovo.
Spracovaných záznamov celkom: {processed}
Ponechaných (čeština):        {kept}
Vynechaných (iný jazyk):      {skipped_not_czech}
Chýb pri spracovaní:           {skipped_error}
""")


if __name__ == '__main__':
    main()