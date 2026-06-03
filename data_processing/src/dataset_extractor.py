import ijson
import json

# Nastav cesty k souborům
INPUT_FILE = "D:\\umbrella_benign_FINISHED_HTML.json"
OUTPUT_FILE = "D:\\umbrella_extracted.jsonl"  # výstupní NDJSON

with open(INPUT_FILE, 'rb') as f_in, open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    # Procházení prvků pole (kořenová úroveň)
    for obj in ijson.items(f_in, 'item'):
        url = obj.get('domain_name')
        #if url and '.cz' in url:           # zde filtruješ .cz
        category = obj.get('category')
        compressed_html = None
        if 'html' in obj and obj['html']:
            compressed_html = obj['html'].get('compressed_html')
            # compressed_html je objekt obsahující $binary s base64
        record = {
            'url': url,
            'category': category,
            'compressed_html': compressed_html
        }
        f_out.write(json.dumps(record) + '\n')

print("Hotovo. Výstup je ve formátu řádek = jeden JSON objekt.")