"""
Fetch the per-student test_0 dataset from the CS540 A2 assignment page using a
real headless browser (the page generates data client-side via seeded JS RNG
tied to the entered Wisc NetID, so it can't be reproduced without running
that JS).
"""
import argparse
from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--out", default="test.csv")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")

        page.fill("#id", args.netid)
        page.click("#button_0")

        # Wait for the client-side generator to populate the test set.
        page.wait_for_function(
            "document.getElementById('test_0').value.trim().length > 100",
            timeout=20000,
        )
        page.wait_for_timeout(500)  # let it settle fully

        test_data = page.eval_on_selector("#test_0", "el => el.value")
        train_hidden = page.eval_on_selector("#train_0", "el => el.textContent") if page.query_selector("#train_0") else ""

        with open(args.out, "w", newline="\n") as f:
            f.write(test_data.strip() + "\n")

        n_lines = len([l for l in test_data.strip().splitlines() if l.strip()])
        print(f"Saved {n_lines} lines to {args.out}")
        if n_lines != 300:
            print(f"WARNING: expected 300 lines, got {n_lines}")

        browser.close()


if __name__ == "__main__":
    main()
