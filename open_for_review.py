"""
Open a *visible* browser window on the assignment page, log in with the
NetID, and load the already-submitted answers so the user can inspect
everything themselves. Stays open until the user closes the window (or up
to the timeout).
"""
import argparse
import pathlib
from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--answers", default="output/answers.txt")
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--keep-open-seconds", type=int, default=3600)
    args = ap.parse_args()

    answers = pathlib.Path(args.answers).read_text()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")

        page.fill("#id", args.netid)
        page.click("#button_0")
        page.wait_for_function(
            "document.getElementById('test_0').value.trim().length > 100", timeout=20000
        )
        page.wait_for_timeout(500)

        page.fill("#loading", answers)
        page.click("#button_19")  # Load
        page.wait_for_timeout(4000)

        page.click("#button_14")  # Grade
        page.wait_for_timeout(9000)

        print("Page is open and loaded for review. Close the browser window when done.")

        # Keep the process (and window) alive until the user closes it or timeout hits.
        try:
            page.wait_for_event("close", timeout=args.keep_open_seconds * 1000)
        except Exception:
            pass

        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
