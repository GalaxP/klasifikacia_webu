import sys
import time
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


def normalize_url(url: str) -> str:
    """
    Ensures the URL has a protocol (http/https).
    If missing, defaults to https://.
    """
    url = str(url).strip()

    # Check if it already has a scheme
    parsed = urlparse(url)

    if not parsed.scheme:
        # No scheme found (e.g., 'google.com' or 'www.google.com')
        # Default to https for security, unless it looks like localhost
        if url.startswith('localhost'):
            return f"http://{url}"
        return f"https://{url}"

    return url


def get_random_user_agent() -> str:
    chrome_versions = [
        "119.0.6045.105", "120.0.6099.109", "121.0.6167.85", "122.0.6261.57"
    ]
    version = random.choice(chrome_versions)
    return f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"


def scrape_url(url: str, timeout_ms: int = 10000) -> Optional[Dict[str, Any]]:
    # Normalize the URL first!
    clean_url = normalize_url(url)

    user_agent = get_random_user_agent()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )

            context = browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York"
            )

            page = context.new_page()

            try:
                response = page.goto(clean_url, wait_until="domcontentloaded", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                print(f"[TIMEOUT] Skipped {clean_url}: Request timed out.")
                return None

            if response is None:
                print(f"[ERROR] Skipped {clean_url}: No response object received.")
                return None

            status_code = response.status

            if not (200 <= status_code < 300):
                print(f"[FAILED] Skipped {clean_url}: Received status code {status_code}")
                return None

            html_content = page.content()

            if not html_content.strip():
                print(f"[EMPTY] Skipped {clean_url}: Page returned empty HTML.")
                return None

            return {
                "url": clean_url,  # Save the normalized URL
                "status_code": status_code,
                "content": html_content,
                "success": True
            }

    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to process {clean_url}: {str(e)}")
        return None
    finally:
        try:
            if 'browser' in locals():
                browser.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Scrape URLs from a Parquet file using Playwright.")
    parser.add_argument("input_file", type=str, help="Path to the input Parquet file")
    parser.add_argument("--output", type=str, default="scraped_results.parquet",
                        help="Path for the output Parquet file")
    parser.add_argument("--column", type=str, default="url", help="Name of the column containing URLs")
    parser.add_argument("--timeout", type=int, default=30000, help="Timeout in milliseconds per request")
    parser.add_argument("--delay", type=float, default=1.0, help="Random delay between requests (seconds)")

    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)

    print(f"Loading data from {args.input_file}...")
    df = pd.read_parquet(args.input_file)

    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found.")
        sys.exit(1)

    # Get raw URLs
    raw_urls = df[args.column].dropna().unique().tolist()

    # Normalize ALL URLs before processing
    urls_to_process = [normalize_url(u) for u in raw_urls]

    print(f"Found {len(urls_to_process)} unique URLs to process.")
    # Debug: Print first few to verify normalization
    print(f"Sample normalized URLs: {urls_to_process[:3]}")

    results = []
    failed_count = 0

    for i, url in enumerate(urls_to_process):
        print(f"\n[{i + 1}/{len(urls_to_process)}] Processing: {url}")

        result = scrape_url(url, timeout_ms=args.timeout)

        if result:
            results.append(result)
        else:
            failed_count += 1

        if i < len(urls_to_process) - 1:
            delay = random.uniform(args.delay * 0.5, args.delay * 1.5)
            time.sleep(delay)

    if results:
        output_df = pd.DataFrame(results)
        output_df.to_parquet(args.output, index=False)
        print(f"\nSuccess! Saved {len(output_df)} successful scrapes to '{args.output}'.")
    else:
        print("\nNo data was successfully scraped.")
        output_df = pd.DataFrame(columns=["url", "status_code", "content", "success"])
        output_df.to_parquet(args.output, index=False)

    print(f"Total failures/skips: {failed_count}")


if __name__ == "__main__":
    main()