import asyncio
import json
import os
import sys
from collections import deque
from urllib.parse import urljoin, urlparse

import tldextract
from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect


START_URL = "https://www.kadaza.cz/cestovani"
OUTPUT_FILE = "D:\\thor_dataset\\crawler_recreation.jsonl"
MAX_PAGES = 1000
DELAY_SECONDS = 1
MAX_CONCURRENCY = 10  # Maximální počet paralelních stránek
EXTRACT_LINKS_ONLY_FROM_CZ = True


def normalize_url(url):
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}".lower()
    except Exception:
        return ""


def is_czech(text):
    if not text or len(text) < 50:
        return False
    try:
        return detect(text) == "cs"
    except LangDetectException:
        return False


def get_registered_domain(url):
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            if ext.subdomain:
                return f"{ext.subdomain}.{ext.domain}.{ext.suffix}".lower()
            return f"{ext.domain}.{ext.suffix}".lower()
        return ""
    except Exception:
        return ""


def build_homepage_url(url):
    try:
        parsed = urlparse(url)
        domain = get_registered_domain(url)
        if not domain:
            return ""

        scheme = parsed.scheme if parsed.scheme in ("http", "https") else "https"
        return f"{scheme}://{domain}/"
    except Exception:
        return ""


def is_homepage(url):
    try:
        parsed = urlparse(url)
        return (parsed.path == "" or parsed.path == "/") and not parsed.query
    except Exception:
        return False


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


async def scroll_page(page):
    height = await page.evaluate("document.body.scrollHeight")
    await page.evaluate(f"window.scrollTo(0, {height / 2})")
    await page.wait_for_timeout(1000)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(1500)


class ParallelCrawler:
    """Paralelní crawler s omezením počtu současných požadavků."""

    def __init__(self, context, output_file, max_pages, concurrency):
        self.first = None
        self.context = context
        self.output_file = output_file
        self.max_pages = max_pages
        self.concurrency = concurrency

        self.start_homepage = ""
        self.visited = set()
        self.queue = deque()
        self.seen_urls = set()
        self.seen_domains = set()

        self.processed_count = 0
        self.semaphore = None
        self.state_lock = None
        self.output_lock = None

    async def init(self, start_url):
        """Inicializace crawleru (musí být v async kontextu)."""
        # Use the full input URL as the starting page
        self.start_homepage = start_url
        start_domain = get_registered_domain(self.start_homepage)

        self.queue = deque([self.start_homepage])
        self.seen_urls = {self.start_homepage}
        self.seen_domains = {start_domain} if start_domain else set()
        self.visited = set()

        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.state_lock = asyncio.Lock()
        self.output_lock = asyncio.Lock()
        self.first = True

        print(f"Spúšťam paralelný crawler na: {self.start_homepage}")
        print(f"Cieľ: {self.max_pages} homepage stránok, konvergentnosť: {self.concurrency}")
        print("-" * 40)

        # Vytvoříme prázdný výstupní soubor
        with open(self.output_file, "w", encoding="utf-8") as f:
            pass

    async def process_url(self, current_url):
        """Zpracování jedné URL s vlastním page objektem."""
        async with self.semaphore:
            # Zkontrolujeme, zda už není zpracováno
            async with self.state_lock:
                if current_url in self.visited:
                    return
                self.visited.add(current_url)

            async with self.state_lock:
                count = self.processed_count

            print(f"[{count + 1}/{self.max_pages}] Navštevujem: {current_url}")

            # Vytvoříme nový page pro tuto URL
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
                    current_url,
                    wait_until="domcontentloaded",
                    timeout=15000,
                )

                await scroll_page(page)
                html_content = await page.content()

                soup = BeautifulSoup(html_content, "html.parser")
                body_text = soup.get_text(separator=" ", strip=True)[:2000]

                is_czech_page = is_czech(body_text) and is_homepage(current_url)
                if self.first:
                    is_czech_page = is_czech(body_text)
                    self.first = False

                if is_czech_page:
                    print("  -> Česká homepage. Ukladám...")

                    record = {"url": current_url, "html": html_content}
                    async with self.output_lock:
                        with open(self.output_file, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f_out.flush()
                            os.fsync(f_out.fileno())

                    async with self.state_lock:
                        self.processed_count += 1

                else:
                    print("  -> Nie je česká homepage. Preskakujem.")

                # Extrahuje odkazy len ak je stránka česká (ak je zapnutý flag)
                if EXTRACT_LINKS_ONLY_FROM_CZ and not is_czech_page:
                    new_links = []
                else:
                    new_links = await get_links_from_html(html_content, current_url)

                added_urls = []
                async with self.state_lock:
                    for link in new_links:
                        domain = get_registered_domain(link)
                        homepage = build_homepage_url(link)

                        if not domain or not homepage:
                            continue

                        if domain in self.seen_domains:
                            continue

                        if homepage not in self.seen_urls:
                            self.queue.append(homepage)
                            self.seen_urls.add(homepage)
                            self.seen_domains.add(domain)
                            added_urls.append(homepage)

                if added_urls:
                    print(f"  -> Pridané {len(added_urls)} nových homepage domén.")
                async with self.state_lock:
                    print(f"  -> Queue po spracovaní: {len(self.queue)}")

            except Exception as e:
                print(f"  -> Chyba pri {current_url}: {e}")

            finally:
                await page.close()

            await asyncio.sleep(DELAY_SECONDS / self.concurrency)

    async def run(self):
        """Hlavní smyčka crawleru."""
        while self.queue and self.processed_count < self.max_pages:
            # Získáme batch z fronty
            async with self.state_lock:
                batch = []
                while self.queue and len(batch) < self.concurrency:
                    url = self.queue.popleft()
                    if url not in self.visited:
                        batch.append(url)

            if not batch:
                break

            # Spustíme batch paralelně
            tasks = [self.process_url(url) for url in batch]
            await asyncio.gather(*tasks)

        print("-" * 40)
        print(f"Dokončené! Celkovo uložených homepage: {self.processed_count}")
        print(f"Výstup: {self.output_file}")


async def crawl_parallel(context, start_url, output_file, max_pages, concurrency):
    """Wrapper pro spuštění paralelního crawleru."""
    crawler = ParallelCrawler(context, output_file, max_pages, concurrency)
    await crawler.init(start_url)
    await crawler.run()


async def main():
    from playwright.async_api import async_playwright

    start_url = START_URL
    if len(sys.argv) > 1:
        start_url = sys.argv[1]

    start_url = normalize_url(start_url)
    print(start_url)

    if not start_url:
        print("Chyba: Neplatná počiatočná URL.")
        return

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

        try:
            await crawl_parallel(
                context=context,
                start_url=start_url,
                output_file=OUTPUT_FILE,
                max_pages=MAX_PAGES,
                concurrency=MAX_CONCURRENCY,
            )
        finally:
            await context.close()
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())