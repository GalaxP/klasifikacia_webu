import argparse
import asyncio
import csv
import json
import random
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import tldextract
from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect
from playwright.async_api import Error as PlaywrightError, async_playwright


OUTPUT_FILE = r"D:\thor_dataset\crawler_en.jsonl"
DEFAULT_DELAY_SECONDS = 0.5
DEFAULT_TIMEOUT_MS = 15000
DEFAULT_CONCURRENCY = 10
OUTPUT_FLUSH_EVERY = 25

BLACKLISTED_DOMAINS = [
]


def normalize_url(url):
    if not url:
        return ""
    value = str(url).strip()
    if not value:
        return ""

    try:
        parsed = urlparse(value)
        if not parsed.scheme:
            if value.startswith("localhost"):
                return f"http://{value}"
            return f"https://{value}"

        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}".lower()
    except Exception:
        return ""



def get_registered_domain(url):
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return ""
    except Exception:
        return ""


def is_homepage(url):
    try:
        parsed = urlparse(url)
        return (parsed.path == "" or parsed.path == "/") and not parsed.query
    except Exception:
        return False


async def scroll_page(page):
    try:
        height = await page.evaluate("document.body.scrollHeight")
        await page.evaluate(f"window.scrollTo(0, {height / 2})")
        await page.wait_for_timeout(1000)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1500)
    except PlaywrightError as e:
        if "Execution context was destroyed" not in str(e):
            raise


async def get_links_from_html(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()

    for tag in soup.find_all("a", href=True):
        absolute_link = urljoin(base_url, tag["href"])
        parsed = urlparse(absolute_link)

        if parsed.scheme not in ["http", "https"]:
            continue

        if any(
            absolute_link.lower().endswith(ext)
            for ext in [".jpg", ".png", ".pdf", ".js", ".css", ".gif"]
        ):
            continue

        clean_link = absolute_link.split("#")[0]
        normalized = normalize_url(clean_link)
        if normalized:
            links.add(normalized)

    return list(links)


def load_urls_from_csv(csv_path, column_name="url", limit=None, randomize=False):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV soubor neobsahuje hlavičku nebo názvy sloupců.")

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Sloupec '{column_name}' nebyl nalezen. Dostupné sloupce: {reader.fieldnames}"
            )

        for row in reader:
            value = row.get(column_name, "")
            if not value:
                continue
            normalized = normalize_url(value)
            if not normalized:
                continue
            rows.append(normalized)

    # Avoid duplicates while preserving order
    unique_rows = []
    seen = set()
    for url in rows:
        if url not in seen:
            seen.add(url)
            unique_rows.append(url)

    if randomize:
        random.shuffle(unique_rows)

    if limit is not None:
        unique_rows = unique_rows[:limit]

    return unique_rows


class ParallelScraper:
    def __init__(self, context, urls, output_file, concurrency, delay):
        self.context = context
        self.urls = list(urls)
        self.output_file = output_file
        self.concurrency = max(1, concurrency)
        self.delay = delay

        self.queue = asyncio.Queue()
        self.visited = set()
        self.processed_count = 0
        self.attempted_count = 0
        self.state_lock = asyncio.Lock()
        self.output_lock = asyncio.Lock()
        self.output_handle = None
        self.pending_output_flushes = 0

    async def init(self):
        for url in self.urls:
            self.queue.put_nowait(url)

        self.output_handle = open(self.output_file, "a", encoding="utf-8")
        self.pending_output_flushes = 0
        print(f"Spúšťam paralelný scraper na {len(self.urls)} URL.")
        print(f"Konkurencia: {self.concurrency}, oneskorenie: {self.delay}s")
        print("-" * 40)

    async def process_url(self, current_url):
        normalized_url = normalize_url(current_url)
        if not normalized_url:
            return False

        domain = get_registered_domain(normalized_url)
        if domain in BLACKLISTED_DOMAINS:
            print(f"  -> Skipping {normalized_url}: blacklisted domain")
            return False

        async with self.state_lock:
            if normalized_url in self.visited:
                return False
            self.visited.add(normalized_url)
            self.attempted_count += 1
            current_index = self.attempted_count

        print(f"\n[{current_index}/{len(self.urls)}] Spracovávam: {normalized_url}")

        page = await self.context.new_page()
        try:
            await page.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                """
            )

            await page.goto(
                normalized_url,
                wait_until="domcontentloaded",
                timeout=DEFAULT_TIMEOUT_MS,
            )

            await scroll_page(page)
            html_content = await page.content()

            record = {"url": normalized_url, "html": html_content}
            async with self.output_lock:
                self.output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.pending_output_flushes += 1
                if self.pending_output_flushes >= OUTPUT_FLUSH_EVERY:
                    self.output_handle.flush()
                    self.pending_output_flushes = 0
                self.processed_count += 1

            print("  -> homepage. Ukladám...")
            return True

        except Exception as exc:
            print(f"  -> Chyba pri {normalized_url}: {exc}")
            return False
        finally:
            await page.close()

            if self.delay > 0:
                await asyncio.sleep(self.delay / max(1, self.concurrency))

    async def run(self):
        while not self.queue.empty() and self.attempted_count < len(self.urls):
            batch = []
            while not self.queue.empty() and len(batch) < self.concurrency:
                batch.append(self.queue.get_nowait())

            if not batch:
                break

            tasks = [self.process_url(url) for url in batch]
            await asyncio.gather(*tasks)

        print("-" * 40)
        print(f"Dokončené! Celkovo uložených homepage: {self.processed_count}")
        print(f"Výstup: {self.output_file}")

    async def close(self):
        if self.output_handle is not None:
            async with self.output_lock:
                self.output_handle.flush()
                self.output_handle.close()
                self.output_handle = None
                self.pending_output_flushes = 0


async def scrape_csv_urls(csv_path, output_file, column_name="url", limit=None, randomize=False, delay=DEFAULT_DELAY_SECONDS, concurrency=DEFAULT_CONCURRENCY):
    urls = load_urls_from_csv(csv_path, column_name=column_name, limit=limit, randomize=randomize)

    if not urls:
        print("Žiadne URL na spracovanie.")
        return 0

    print(f"Načítané {len(urls)} URL z CSV súboru.")
    if randomize:
        print("Režim: náhodné poradie.")
    if limit is not None:
        print(f"Limit: {limit} URL.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="cs-CZ",
            timezone_id="Europe/Prague",
        )

        scraper = ParallelScraper(
            context=context,
            urls=urls,
            output_file=output_file,
            concurrency=concurrency,
            delay=delay,
        )

        try:
            await scraper.init()
            await scraper.run()
            return scraper.processed_count
        finally:
            await scraper.close()
            await context.close()
            await browser.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape homepage URLs from a CSV file and save the same JSONL format as cz_crawler.py.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file containing URLs.")
    parser.add_argument("--column", default="url", help="Name of the column containing URLs in the CSV file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process from the CSV file.")
    parser.add_argument("--random", action="store_true", help="Process URLs in random order. When combined with --limit, it randomly selects that many rows.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output JSONL file path.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between requests in seconds.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="How many pages to scrape in parallel.")
    return parser.parse_args()


async def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"CSV: {args.csv}")
    print(f"Output: {output_path}")

    try:
        await scrape_csv_urls(
            csv_path=args.csv,
            output_file=str(output_path),
            column_name=args.column,
            limit=args.limit,
            randomize=args.random,
            delay=args.delay,
            concurrency=args.concurrency,
        )
    except FileNotFoundError as exc:
        print(f"Chyba: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"Chyba: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
