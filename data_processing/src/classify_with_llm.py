#!/usr/bin/env python3
"""
Asynchronná klasifikácia webových stránok s opravou API Key a Batchingu.
"""

import argparse
import json
import os
import sys
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ------------------------------------------------------------
# Prompty
# ------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """Jsi expert na klasifikáciu webových stránok. 
Pečlivo analyzuj text a prirad všetky vhodné kategórie z definovaného zoznamu. 
Výstup musí byť validný JSON."""

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


def parse_json_from_response(text: str) -> Optional[dict]:
    """Pokusí sa extrahovať a parsovať JSON z odpovede modelu."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


async def classify_single_task(
        session: aiohttp.ClientSession,
        model: str,
        system_prompt: str,
        user_template: str,
        text: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        retries: int,
        api_key: str,
        base_url: str,
        delay_base: float = 1.0
) -> Optional[dict]:
    """
    Asynchrónne zavolá LLM s retry logikou a API key.
    """
    user_message = user_template.format(text=text)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # --- OPRAVA: Správne nastavenie hlavičiek ---
    headers = {
        "Content-Type": "application/json"
    }

    # Pridanie API key, ak existuje
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    # -------------------------------------------

    for attempt in range(retries):
        try:
            async with session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return parse_json_from_response(raw)
                elif response.status == 429:
                    print(f"Rate limit hit (attempt {attempt + 1}), čakám...", file=sys.stderr)
                    await asyncio.sleep(delay_base * (2 ** attempt))
                    continue
                else:
                    error_text = await response.text()
                    # Logovanie konkrétnej chyby pre debug
                    print(f"API Error {response.status} pre text (krátky): {text[:50]}... Chyba: {error_text}",
                          file=sys.stderr)

                    if attempt == retries - 1:
                        return None
                    await asyncio.sleep(delay_base)

        except asyncio.TimeoutError:
            print(f"Timeout (attempt {attempt + 1})", file=sys.stderr)
            if attempt == retries - 1:
                return None
            await asyncio.sleep(delay_base)
        except Exception as e:
            print(f"Nepredvídaná chyba: {e}", file=sys.stderr)
            if attempt == retries - 1:
                return None
            await asyncio.sleep(delay_base)

    return None


async def main_async(args):
    global base_url, api_key

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "not-needed")

    if not base_url:
        print("Chyba: base_url musí byť nastavené cez --base-url alebo OPENAI_BASE_URL.", file=sys.stderr)
        sys.exit(1)

    # Načítanie promptov
    system_prompt = DEFAULT_SYSTEM_PROMPT
    user_template = DEFAULT_USER_TEMPLATE
    if args.system_prompt:
        with open(args.system_prompt, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    if args.user_template:
        with open(args.user_template, 'r', encoding='utf-8') as f:
            user_template = f.read().strip()

    # Načítanie dát
    items = []
    try:
        with open(args.input, 'r', encoding='utf-8') as fin:
            for line_no, line in enumerate(fin, 1):
                if args.max_rows > 0 and line_no > args.max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Súbor {args.input} nenájdený.")
        return

    total_lines = len(items)
    print(
        f"Pripravených {total_lines} záznamov. API URL: {base_url}, API Key nastavená: {'Áno' if api_key and api_key != 'not-needed' else 'Nie'}")

    results_map = {}

    async with aiohttp.ClientSession() as session:
        batch_size = args.batch_size
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

        pbar = tqdm(total=len(batches), desc="Batch processing", unit="batch")

        for batch_idx, batch in enumerate(batches):
            local_results = {}
            tasks_with_indices = []

            for idx, item in enumerate(batch):
                text = item.get('text', '')
                url = item.get('url', 'unknown')

                if not text:
                    out = {
                        'url': url,
                        'original_category': item.get('category'),
                        'categories': [],
                        'note': 'NO_TEXT',
                        'needs_human_review': True,
                        'czech': False
                    }
                    if args.keep_text:
                        out['text'] = text
                    local_results[idx] = out
                    continue

                task = classify_single_task(
                    session=session,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    text=text,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    retries=args.retries,
                    api_key=api_key,  # <--- PRENOS API KEY
                    base_url=base_url  # <--- PRENOS BASE URL
                )
                tasks_with_indices.append((idx, task))

            # Čakanie na dokončenie všetkých úloh v bati
            gathered = await asyncio.gather(*(t[1] for t in tasks_with_indices), return_exceptions=True)

            for (idx, _), result in zip(tasks_with_indices, gathered):
                item = batch[idx]
                url = item.get('url', 'unknown')

                if isinstance(result, Exception):
                    final_result = None
                else:
                    final_result = result

                out = {
                    'url': url,
                    'original_category': item.get('category'),
                }
                if args.keep_text:
                    out['text'] = item.get('text')

                if final_result:
                    out['categories'] = final_result.get('categories', [])
                    out['note'] = final_result.get('note', '')
                    out['needs_human_review'] = final_result.get('needs_human_review', False)
                    out['czech'] = final_result.get('czech', False)
                else:
                    # Ak sa podarilo zistiť, že API vrátilo chybu (napr. 401), môžeme to napísať do note
                    note_msg = 'CLASSIFICATION_FAILED'
                    # Ak sme vedeli, že bola chyba autorizácie (môžeme to rozšíriť), môžeme písať viac info
                    out['categories'] = []
                    out['note'] = note_msg
                    out['needs_human_review'] = True
                    out['czech'] = False

                local_results[idx] = out

            # Uloženie výsledkov
            start_global_idx = batch_idx * batch_size
            for local_idx, res in local_results.items():
                results_map[start_global_idx + local_idx] = res

            pbar.update(1)

        pbar.close()

    # Zápis výsledkov
    with open(args.output, 'w', encoding='utf-8') as fout:
        for i in range(len(items)):
            if i in results_map:
                fout.write(json.dumps(results_map[i], ensure_ascii=False) + '\n')
            else:
                fout.write(json.dumps({'error': 'missing'}, ensure_ascii=False) + '\n')

    print(f"\nHotovo. Zpracovaných: {len(results_map)}")


def main():
    parser = argparse.ArgumentParser(description="Asynchrónna klasifikácia webových stránok")
    parser.add_argument("--input", required=True, help="Vstupný JSONL súbor")
    parser.add_argument("--output", required=True, help="Výstupný JSONL súbor")
    parser.add_argument("--base-url", default=None, help="Base URL API serveru (napr. http://localhost:11434/v1)")
    parser.add_argument("--api-key", default=None, help="API kľúč (voliteľné, ak nie je potrebné)")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Názov modelu")
    parser.add_argument("--system-prompt", default=None, help="Súbor so systémovým promptom")
    parser.add_argument("--user-template", default=None, help="Súbor s uživatelskou šablónou")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokeny")
    parser.add_argument("--temperature", type=float, default=0.0, help="Teplota")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout (s)")
    parser.add_argument("--retries", type=int, default=3, help="Počet opakovania")
    parser.add_argument("--keep-text", action="store_true", help="Ponechať text vo výstupe")
    parser.add_argument("--max-rows", type=int, default=-1, help="Limit riadkov (debug)")
    parser.add_argument("--batch-size", type=int, default=10, help="Veľkosť batchu (počet paralelných požiadavkov)")

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()