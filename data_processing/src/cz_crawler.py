import asyncio
import json
import sys
from collections import deque
from urllib.parse import urljoin, urlparse

import tldextract
from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect


START_URL = "https://www.xnxx.com/"
OUTPUT_FILE = "D:\czech_homepages_adult.jsonl"
MAX_PAGES = 300
DELAY_SECONDS = 2


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


async def crawl_page(page, start_url, output_file, max_pages):
    start_homepage = build_homepage_url(start_url) or start_url
    start_domain = get_registered_domain(start_homepage)

    visited = set()
    queue = deque([start_homepage])
    seen_urls = {start_homepage}
    seen_domains = {start_domain} if start_domain else set()

    processed_count = 0

    print(f"Spúšťam crawler na: {start_homepage}")
    print(f"Cieľ: {max_pages} homepage stránok.")
    print("-" * 40)

    with open(output_file, "w", encoding="utf-8") as f_out:
        while queue and processed_count < max_pages:
            current_url = queue.popleft()

            if current_url in visited:
                continue

            print(f"[{processed_count + 1}/{max_pages}] Navštevujem: {current_url}")

            try:
                await page.goto(
                    current_url,
                    wait_until="domcontentloaded",
                    timeout=15000,
                )

                await scroll_page(page)
                html_content = await page.content()

                soup = BeautifulSoup(html_content, "html.parser")
                body_text = soup.get_text(separator=" ", strip=True)[:2000]

                if is_czech(body_text) and is_homepage(current_url):
                    print("  -> Česká homepage. Ukladám...")

                    record = {"url": current_url, "html": html_content}
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

                    processed_count += 1
                else:
                    print("  -> Nie je česká homepage. Preskakujem.")

                visited.add(current_url)

                new_links = await get_links_from_html(html_content, current_url)
                added_count = 0

                for link in new_links:
                    domain = get_registered_domain(link)
                    homepage = build_homepage_url(link)

                    if not domain or not homepage:
                        continue

                    if domain in seen_domains:
                        continue

                    if homepage not in seen_urls:
                        queue.append(homepage)
                        seen_urls.add(homepage)
                        seen_domains.add(domain)
                        added_count += 1

                print(f"  -> Pridané {added_count} nových homepage domén.")
                print(f"  -> Queue po spracovaní: {len(queue)}")

            except Exception as e:
                print(f"  -> Chyba: {e}")

            await asyncio.sleep(DELAY_SECONDS)

    print("-" * 40)
    print(f"Dokončené! Celkovo uložených homepage: {processed_count}")
    print(f"Výstup: {output_file}")


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

        page = await context.new_page()

        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """
        )

        try:
            await crawl_page(
                page=page,
                start_url=start_url,
                output_file=OUTPUT_FILE,
                max_pages=MAX_PAGES,
            )
        finally:
            await context.close()
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())