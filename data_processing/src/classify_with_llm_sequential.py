#!/usr/bin/env python3
"""
Klasifikace webových stránek pomocí LLM přes OpenAI kompatibilní API.
Vstup: JSONL soubor s poli 'url', 'category', 'text'.
Výstup: JSONL s přidanými poli 'categories', 'note', 'needs_human_review', 'czech'.
Přidán parametr --max-rows pro limitování počtu zpracovaných záznamů (debugging).
"""

import argparse
import html
import json
import os
import re
import sys
import time
from typing import Optional

from openai import OpenAI
from openai._exceptions import APIError, APITimeoutError, RateLimitError

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ------------------------------------------------------------
# Výchozí prompty – upravte podle svého agenta
# ------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """Jsi expert na klasifikaci webových stránek. 
Pečlivě analyzuj text a přiřaď všechny vhodné kategorie z definovaného seznamu. 
Výstup musí být validní JSON."""


#DEFAULT_USER_TEMPLATE = """Task: Classify the following web page content into applicable categories from the given list. Output a JSON object with key "categories" containing an array of relevant category codes. Try to pick as few categories as possible. Also include a field for needs_human_review, and if that is true, also write the reason, in the note field. If no human review is necessary leave note as empty. If the page is not in Czech, set the czech field to false, and leave all the other fields empty.
#Category list: ["Adult", "Computers", "Games", "Health", "News", "Recreation", "Reference", "Science", "Shopping", "Society", "Sports", "Others"]
#The Adult category is only for pornographic/inappropriate content. Set needs_human_review to true if the web page doesn't contain enough actual content or the classification requires human review.
#Content:
#{text}
#
#Output exactly in this format:
#{{"categories": ["CATEGORY1", "CATEGORY2"], "note": "", "needs_human_review": false, "czech": true}}"""
#
DEFAULT_USER_TEMPLATE = """Task: Classify the following web page content into applicable categories from the given list.
Allowed categories:
["Adult", "Computers", "Games", "Health", "News", "Recreation", "Reference", "Science", "Shopping", "Society", "Sports", "Others"]

Rules:
1. First determine whether the page contains Czech text.
   - If the page is not in Czech, return:
     {{"categories": [], "note": "", "needs_human_review": false, "czech": false}}

2. Before assigning any category, decide whether the page contains enough substantive content to classify.
   Substantive content means text that actually describes the topic, service, product, article, organization, or subject of the page.

3. The following do NOT count as substantive content by themselves:
   - geo-block / availability notices
   - login / register / account access pages
   - payment / withdrawal / deposit instructions
   - customer support or contact-only pages
   - legal disclaimers, privacy policy, terms
   - cookie banners
   - maintenance / error / holding pages
   - pages with only navigation, brand names, or short notices

4. Do NOT classify based on the domain name, brand name, or product names alone.
   Mentions such as "casino", "poker", or "sport" inside brand names are not enough evidence.

5. If there is not enough substantive content, return:
   - "categories": []
   - "needs_human_review": true
   - "note": a short reason, e.g. "Insufficient substantive content; only an availability/withdrawal notice."

6. If there is enough substantive content, choose as few categories as possible.

7. The Adult category is only for pornographic/inappropriate content.

8. Output must be valid JSON with exactly these keys and no others:
   "categories", "note", "needs_human_review", "czech"

Content:
{text}

Output exactly one JSON object in this format:
{{"categories": ["CATEGORY1"], "note": "", "needs_human_review": false, "czech": true}}"""

HTML_BLOCK_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
DEFAULT_MAX_INPUT_CHARS = 12000
ABSOLUTE_MAX_INPUT_CHARS = 20000




def parse_json_from_response(text: str) -> Optional[dict]:
    """Pokusí se extrahovat a parsovat JSON z odpovědi modelu."""
    text = text.strip()
    # Někdy model zabalí JSON do ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Zkus najít první složené závorky
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
    return None


def prepare_text_for_prompt(text: str, max_chars: int) -> str:
    """Zmenší HTML na čitelný text a ořízne ho na bezpečnú dĺžku."""
    if not text:
        return ""

    cleaned = HTML_BLOCK_RE.sub(" ", text)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if max_chars > 0 and len(cleaned) > max_chars:
        head = max_chars * 4 // 5
        tail = max_chars - head
        cleaned = cleaned[:head].rstrip() + "\n...[TRUNCATED]...\n" + cleaned[-tail:].lstrip()

    return cleaned


def clamp_input_limit(max_input_chars: int) -> int:
    """Zaručí, že sa do promptu nikdy neposiela príliš dlhý text."""
    if max_input_chars <= 0:
        return ABSOLUTE_MAX_INPUT_CHARS
    return min(max_input_chars, ABSOLUTE_MAX_INPUT_CHARS)


def classify_text(client: OpenAI, model: str, system_prompt: str, user_template: str,
                  text: str, max_input_chars: int, max_tokens: int, temperature: float, timeout: int) -> Optional[dict]:
    """
    Zavolá LLM a vrátí parsovaný JSON výsledek, nebo None při chybě.
    """
    prompt_text = prepare_text_for_prompt(text, clamp_input_limit(max_input_chars))
    user_message = user_template.format(text=prompt_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            # response_format={"type": "json_object"}  # Odkomentujte, pokud to váš server podporuje
        )
        raw = response.choices[0].message.content
        return parse_json_from_response(raw)
    except RateLimitError as e:
        print("Rate limit hit, čekám...", file=sys.stderr)
        time.sleep(5)
        raise e
    except (APIError, APITimeoutError) as e:
        print(f"API chyba: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Klasifikace webových stránek pomocí LLM")
    parser.add_argument("--input", required=True, help="Vstupní JSONL soubor (s poli url, category, text)")
    parser.add_argument("--output", required=True, help="Výstupní JSONL soubor")
    parser.add_argument("--base-url", default=None, help="Base URL API serveru (např. http://localhost:8080/v1)")
    parser.add_argument("--api-key", default=None, help="API klíč (pokud je potřeba)")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Název modelu")
    parser.add_argument("--system-prompt", default=None, help="Soubor se systémovým promptem (přepíše výchozí)")
    parser.add_argument("--user-template", default=None, help="Soubor s uživatelskou šablonou (obsahuje {text})")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximální počet tokenů pro odpověď")
    parser.add_argument("--temperature", type=float, default=0.0, help="Teplota pro sampling")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout pro API volání (s)")
    parser.add_argument("--retries", type=int, default=3, help="Počet opakování při chybě")
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS,
                        help="Maximální délka vstupu do promptu po normalizaci (má tvrdý bezpečný strop)")
    parser.add_argument("--keep-text", action="store_true", help="Ponechat ve výstupu původní text")
    parser.add_argument("--max-rows", type=int, default=-1,
                        help="Maximální počet řádků ke zpracování (-1 znamená všechny, užitečné pro debugging)")
    args = parser.parse_args()

    # Nastavení API
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "not-needed")
    if not base_url:
        print("Chyba: base_url musí být nastaveno přes --base-url nebo proměnnou OPENAI_BASE_URL", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Načtení promptů
    system_prompt = DEFAULT_SYSTEM_PROMPT
    user_template = DEFAULT_USER_TEMPLATE
    if args.system_prompt:
        with open(args.system_prompt, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    if args.user_template:
        with open(args.user_template, 'r', encoding='utf-8') as f:
            user_template = f.read().strip()
    # Ověření, že user_template obsahuje {text}
    if "{text}" not in user_template:
        print("Varování: user_template neobsahuje {text}, text stránky nebude vložen.", file=sys.stderr)

    # Zpracování souboru proudově
    processed = 0
    errors = 0
    skipped = 0
    max_rows = args.max_rows

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:

        # Proudová iterace bez načtení celého souboru
        iterator = enumerate(fin, start=1)
        if tqdm is not None:
            # Pokud je nastaven max_rows, můžeme nastavit total pro progress bar
            total = max_rows if max_rows > 0 else None
            iterator = tqdm(iterator, desc="Klasifikace", unit="řádků", total=total)

        for line_no, raw_line in iterator:
            if max_rows > 0 and processed >= max_rows:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            text = item.get('text')
            if not text:
                skipped += 1
                continue

            # Opakování při chybě
            result = None
            for attempt in range(args.retries):
                try:
                    result = classify_text(
                        client, args.model, system_prompt, user_template, text,
                        args.max_input_chars,
                        args.max_tokens, args.temperature, args.timeout
                    )
                    if result is not None:
                        break
                except RateLimitError:
                    time.sleep(5 * (attempt + 1))
                except (APIError, APITimeoutError):
                    time.sleep(2 * (attempt + 1))

            # Sestavení výstupního záznamu
            out = {
                'url': item.get('url'),
                'original_category': item.get('category'),
            }
            if args.keep_text:
                out['text'] = text

            if result:
                out['categories'] = result.get('categories', [])
                out['note'] = result.get('note', '')
                out['needs_human_review'] = result.get('needs_human_review', False)
                out['czech'] = result.get('czech', False)
            else:
                # Selhání – uložíme chybový stav
                out['categories'] = []
                out['note'] = 'CLASSIFICATION_FAILED'
                out['needs_human_review'] = True
                out['czech'] = False

            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            processed += 1

    print(f"\nHotovo. Zpracováno: {processed}, chyb: {errors}, přeskočeno (bez textu): {skipped}")


if __name__ == '__main__':
    main()