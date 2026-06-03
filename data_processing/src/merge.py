import json
import sys
import os


def clean_text(text):
    """
    Nahradí krútené úvodzovky za bežné, aby bol text platným JSON.
    """
    if not isinstance(text, str):
        return text
    # Zamení "a" za " a ' za '
    return text.replace('"', '"').replace("'", "'")


def merge_specific_files(input_files, output_file):
    """
    Spojí zadané JSONL súbory a odstráni duplikáty.

    :param input_files: Zoznam ciest k vstupným súborom
    :param output_file: Cesta k výstupnému súboru
    """
    seen_records = set()
    unique_records = []
    duplicates_count = 0
    total_lines_processed = 0

    if not input_files:
        print("Chyba: Nezadaný žiadny vstupný súbor.")
        print("Použitie: python script.py subor1.jsonl subor2.jsonl ... [vystup.jsonl]")
        return

    print(f"Spracovávanie {len(input_files)} súborov...")

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Upozornenie: Súbor neexistuje, preskakujem: {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    total_lines_processed += 1

                    try:
                        data = json.loads(line)

                        # Oprava potenciálnych zlých úvodzoviek v texte
                        if 'text' in data and isinstance(data['text'], str):
                            data['text'] = clean_text(data['text'])

                        # Vytvorenie unikátneho kľúča
                        record_key = (
                            data.get('url', ''),
                            data.get('category', ''),
                            data.get('text', '')
                        )

                        if record_key not in seen_records:
                            seen_records.add(record_key)
                            unique_records.append(data)
                        else:
                            duplicates_count += 1

                    except json.JSONDecodeError as e:
                        print(f"Chyba JSON vo fajle {file_path}, riadok {line_num}: {e}")
                        continue

        except Exception as e:
            print(f"Kritická chyba pri čítaní {file_path}: {e}")
            continue

    # Zápis výsledkov
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in unique_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("-" * 40)
    print(f"Spracovaných súborov: {len([f for f in input_files if os.path.exists(f)])}")
    print(f"Spolu spracovaných riadkov: {total_lines_processed}")
    print(f"Odstránených duplikátov: {duplicates_count}")
    print(f"Unikátnych záznamov uložených do: {output_file}")


if __name__ == "__main__":
    # Očakávame minimálne jeden vstupný súbor
    if len(sys.argv) < 2:
        print("Nesprávne použitie.")
        print("Príklad: python merge_jsonl.py subor1.jsonl subor2.jsonl vystup.jsonl")
        sys.exit(1)

    # Posledný argument je výstupný súbor, ostatné sú vstupné
    all_args = sys.argv[1:]

    # Ak je zadáno viac ako 1 argument, posledný je výstup
    if len(all_args) > 1:
        input_files = all_args[:-1]
        output_file = all_args[-1]
    else:
        # Ak je len jeden argument, predpokladáme, že je to vstupný súbor a použijeme defaultný výstup
        input_files = all_args
        output_file = "merged_output.jsonl"

    merge_specific_files(input_files, output_file)