import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def process_jsonl(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Chyba: Súbor {input_file} neexistuje.")
        return

    # Otvoríme výstupný súbor pre zápis (prepisujeme ak existuje)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        with sync_playwright() as p:
            # Spustíme prehliadač (headless=True znamená beh bez grafického rozhrania)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
                locale="cs-CZ",
                timezone_id="Europe/Prague",
            )
            page = context.new_page()

            print(f"Spracúvam súbor: {input_file}")

            try:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            # Získame URL - predpokladáme kľúč "url", uprav podľa potreby
                            url = data.get("url")

                            if not url:
                                print(f"[Riadok {line_num}] Chýba kľúč 'url', preskakujem.")
                                continue

                            print(f"[{line_num}] Načítavam: {url}")

                            # Nastavenie timeoutu (voliteľné, napr. 30 sekúnd)
                            page.set_default_timeout(30000)

                            # Návšteva stránky
                            response = page.goto(url, wait_until="networkidle")

                            # Získanie HTML obsahu
                            html_content = page.content()

                            # Vytvorenie nového záznamu
                            result_data = {
                                "url": url,
                                "html": html_content
                            }

                            # Zápis do výstupného súboru
                            out_f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                            print(f"[{line_num}] Úspešne uložené.")

                        except Exception as e:
                            print(f"[Riadok {line_num}] Chyba pri spracovaní URL '{data.get('url', 'Neznáme')}': {e}")
                            # Voliteľne: Môžeme uložiť chybový stav namiesto HTML
                            error_data = {"url": data.get("url"), "error": str(e)}
                            out_f.write(json.dumps(error_data, ensure_ascii=False) + "\n")

            finally:
                browser.close()

    print(f"Dokončené! Výsledky uložené do: {output_file}")


if __name__ == "__main__":
    # Zadaj názvy svojich súborov tu
    INPUT_FILE = "D:\\adult_annotated_v2.jsonl"  # Vstupný súbor s URL
    OUTPUT_FILE = "D:\\czech_homepages_adult.jsonl"  # Výstupný súbor s HTML

    process_jsonl(INPUT_FILE, OUTPUT_FILE)