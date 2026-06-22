import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Načíta všetky riadky zo súboru .jsonl."""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Varovanie: Chyba pri čítaní {file_path} na riadku {line_num}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Súbor nenájdený: {file_path}")
    return records


def build_url_index(records: List[Dict], key_name: str) -> Dict[str, Dict]:
    """Vytvorí index pre rýchle vyhľadávanie."""
    index = {}
    for record in records:
        url = record.get(key_name)
        if url:
            normalized_url = url.rstrip('/')
            # Uchovávame prvý nájdený záznam pre danú URL
            if normalized_url not in index:
                index[normalized_url] = record
    return index


def extract_base64_content(compressed_field: Any) -> Optional[str]:
    """
    Extrahuje čistý base64 reťazec zo štruktúry:
    {"$binary": {"base64": "..."}}
    Ak má iný tvar alebo chýba, vráti None.
    """
    if not isinstance(compressed_field, dict):
        return None

    binary_data = compressed_field.get("$binary")
    if not isinstance(binary_data, dict):
        return None

    base64_str = binary_data.get("base64")

    if isinstance(base64_str, str):
        return base64_str
    return None


def main():
    parser = argparse.ArgumentParser(description="Filtruje anotácie a extrahuje formátovaný výstup.")

    # Povinné argumenty
    parser.add_argument('annotation_file', type=str, help='Cesta k hlavnému JSONL súboru s anotáciami.')
    parser.add_argument('html_files', nargs='+', type=str, help='Cesty k JSONL súborom s compressed_html.')

    # Voliteľné argumenty
    parser.add_argument('--output', '-o', type=str, default='filtered_output.jsonl', help='Názov výstupného súboru.')
    parser.add_argument('--key', '-k', type=str, default='url',
                        help='Kľúč pre hľadanie (napr. "url" alebo "domain_name").')
    parser.add_argument('--search-dir', type=str, default=None, help='Priečinok s ďalšími *.jsonl súbormi.')
    parser.add_argument('--raw-html', action='store_true', help='Použi raw html namiesto compressed_html.')

    args = parser.parse_args()

    print(f"--- Spúšťam spracovanie ---")
    print(f"Anotácie: {args.annotation_file}")
    print(f"HTML súbory: {len(args.html_files)}")
    if args.search_dir:
        print(f"Hľadanie v priečinku: {args.search_dir}")

    # 1. Načítanie a filtrovanie anotácií
    print("\n1. Načítanie a filtrovanie anotácií...")
    annotations = load_jsonl(args.annotation_file)

    # Kritériá: czech == True AND needs_human_review == False
    filtered_records = [
        r for r in annotations
        if r.get("needs_human_review") is False and r.get("czech") is True
    ]

    print(f"Nájdených {len(filtered_records)} záznamov na kontrolu.")

    if not filtered_records:
        print("Žiadne záznamy nevyhovujú kritériám. Koniec.")
        return

    target_urls = set()
    for rec in filtered_records:
        val = rec.get(args.key)
        if val:
            target_urls.add(val.rstrip('/'))

    # 2. Zozbieranie všetkých HTML súborov
    all_html_paths = list(args.html_files)

    if args.search_dir and os.path.isdir(args.search_dir):
        found_in_dir = list(Path(args.search_dir).glob("*.jsonl"))
        if found_in_dir:
            print(f"Nájdené {len(found_in_dir)} ďalších súborov v priečinku.")
            all_html_paths.extend([str(p) for p in found_in_dir])

    # 3. Indexovanie HTML súborov
    print("\n2. Indexovanie HTML súborov...")
    html_index = {}

    for file_path in all_html_paths:
        if not os.path.exists(file_path):
            continue
        records = load_jsonl(file_path)
        current_index = build_url_index(records, args.key)

        for url, data in current_index.items():
            if url not in html_index:
                html_index[url] = data

    # 4. Generovanie výstupu
    print("\n3. Generovanie výstupu...")
    output_records = []

    for target_url in target_urls:
        if target_url in html_index:
            html_record = html_index[target_url]

            # Získanie pôvodných kategórií z anotačného záznamu (aby sme mali správny kontext)
            # Hľadáme pôvodný zápis v filtered_records pre túto URL
            original_annotation = next(
                (a for a in filtered_records if a.get(args.key, '').rstrip('/') == target_url),
                None
            )

            categories = []
            if original_annotation:
                categories = original_annotation.get("categories", [])

            # Extrakcia obsahu (compressed_html alebo raw html)
            if args.raw_html:
                raw_content = html_record.get("html")
                field_name = "html"
                if isinstance(raw_content, str):
                    content_value = raw_content
                else:
                    content_value = None
            else:
                raw_compressed = html_record.get("compressed_html")
                content_value = extract_base64_content(raw_compressed)
                field_name = "compressed_html"

            if content_value:
                output_record = {
                    "url": target_url,
                    field_name: content_value,
                    "categories": categories
                }
                output_records.append(output_record)
            else:
                print(f"Upozornenie: Pre URL '{target_url}' nebol nájdený platný {field_name}.")
        else:
            # Voliteľné: vypisovať nenájdené URL
            pass

    # 5. Uloženie výsledkov
    print(f"\nUkladanie {len(output_records)} záznamov do {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for record in output_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Dokončené! Výsledky uložené do: {args.output}")


if __name__ == "__main__":
    main()