import json
import gzip
import base64
import glob
import argparse
from pathlib import Path


def compress_and_encode(html_content):
    """
    skomprimuje HTML reťazec pomocou gzip a zakóduje ho do Base64.
    """
    if not html_content:
        return ""

    # Prevedenie textu na bajty
    data_bytes = html_content.encode('utf-8')

    # Gzip kompresia
    compressed_bytes = gzip.compress(data_bytes)

    # Base64 kódovanie
    b64_encoded = base64.b64encode(compressed_bytes).decode('ascii')

    return b64_encoded


def merge_and_process_datasets(input_patterns, output_file):
    """
    Spája JSONL súbory, odstraňuje duplicity a konvertuje HTML na compressed_html.
    """

    seen_urls = set()
    processed_count = 0
    duplicate_count = 0
    error_count = 0

    # Nájde všetky súbory podľa zadaných vzorov
    all_files = []
    for pattern in input_patterns:
        found_files = glob.glob(pattern)
        if not found_files:
            print(f"Upozornenie: Žiadne súbory nenašli pre vzor '{pattern}'")
        else:
            all_files.extend(found_files)

    if not all_files:
        print("Chyba: Nenašli sa žiadne vstupné súbory podľa zadaných vzorov.")
        return

    print(f"Nájdené vstupné súbory: {len(all_files)}")

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for file_path in all_files:
                # Uistíme sa, že ide o súbor
                if not Path(file_path).is_file():
                    continue

                print(f"Spracúvam: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        for line_num, line in enumerate(f_in, 1):
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                record = json.loads(line)

                                url = record.get("url")
                                if not url:
                                    # Ignorujeme záznamy bez URL, alebo ich môžeme logovať
                                    continue

                                # Kontrola duplicity
                                if url in seen_urls:
                                    duplicate_count += 1
                                    continue

                                seen_urls.add(url)

                                categories = record.get("categories", [])

                                # Logika pre HTML vs Compressed HTML
                                raw_html = None
                                is_compressed_already = False

                                if "html" in record:
                                    raw_html = record["html"]
                                    is_compressed_already = False
                                elif "compressed_html" in record:
                                    raw_html = record["compressed_html"]
                                    is_compressed_already = True
                                else:
                                    raw_html = ""
                                    is_compressed_already = True  # Prázdny string je technicky už "spracovaný"

                                final_compressed_html = ""

                                if not is_compressed_already and raw_html:
                                    final_compressed_html = compress_and_encode(raw_html)
                                else:
                                    # Ak už bolo komprimované alebo prázdne
                                    final_compressed_html = raw_html or ""

                                final_record = {
                                    "url": url,
                                    "categories": categories,
                                    "compressed_html": final_compressed_html
                                }

                                out_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                                processed_count += 1

                            except json.JSONDecodeError:
                                error_count += 1
                            except Exception as e:
                                print(f"  Chyba v riadku {line_num}: {e}")
                                error_count += 1

                except Exception as e:
                    print(f"Chyba pri čítaní súboru {file_path}: {e}")
                    error_count += 1

    except IOError as e:
        print(f"Chyba pri zápise do výstupného súboru {output_file}: {e}")
        return

    print("\n--- Štatistiky ---")
    print(f"Úspešne spracovaných unikátnych záznamov: {processed_count}")
    print(f"Preskočených duplicit: {duplicate_count}")
    print(f"Chýb počas spracovania: {error_count}")
    print(f"Výsledný súbor uložený ako: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Spojí viacero JSONL datasetov, odstráni duplicity a skompresuje HTML do Base64.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Príklady použitia:
  python merge_datasets_cli.py --input data/*.jsonl output.jsonl
  python merge_datasets_cli.py -i file1.jsonl file2.jsonl -o merged.jsonl
  python merge_datasets_cli.py -i "dir1/*.jsonl" "dir2/*.jsonl" -o final.jsonl
        """
    )

    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Jeden alebo viacero vstupných súborov alebo priečinkov (podporuje * wildcard).'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Cesta k výstupnému JSONL súboru.'
    )

    args = parser.parse_args()

    merge_and_process_datasets(args.input, args.output)


if __name__ == "__main__":
    main()