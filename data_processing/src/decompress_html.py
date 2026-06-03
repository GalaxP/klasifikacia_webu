import json
import base64
import gzip
import sys

def decompress_html(compressed):
    """Vrátí dekomprimovaný HTML řetězec, jinak None."""
    try:
        bin_data = compressed.get('$binary', {})
        b64 = bin_data.get('base64')
        if not b64:
            return None
        raw = base64.b64decode(b64)
        return gzip.decompress(raw).decode('utf-8')
    except Exception:
        return None

def main(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filtr: chceme jen záznamy s platným compressed_html
            compressed = obj.get('compressed_html')
            if not compressed:
                continue

            html_text = decompress_html(compressed)
            if html_text is None:
                continue

            # Sestavíme výstupní záznam
            out = {
                'url': obj.get('url'),
                'category': obj.get('category'),
                'html': html_text
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Použití: python decompress_html.py <vstupní_soubor.jsonl> <výstupní_soubor.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])