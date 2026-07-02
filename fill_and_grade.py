"""
Open the CS540 A2 page, enter the NetID, load the generated answers, click
Grade, and screenshot the result. Does NOT click Submit -- that's left as a
separate explicit step.
"""
import argparse
import pathlib
from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--answers", default="output/answers.txt")
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--screenshot", default="output/grade_result.png")
    ap.add_argument("--state-out", default="output/browser_state.json")
    args = ap.parse_args()

    answers = pathlib.Path(args.answers).read_text()

    with sync_playwright() as p:
        browser = p.chromium.launch()
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

        # sanity: check a couple of fields actually got populated
        net1_val = page.eval_on_selector("#net_1", "el => el.value")
        label_val = page.eval_on_selector("#label", "el => el.value")
        print("net_1 populated chars:", len(net1_val))
        print("label populated chars:", len(label_val))

        page.click("#button_14")  # Grade
        page.wait_for_timeout(9000)

        comment = page.eval_on_selector("#comment", "el => el.textContent")
        output_val = page.eval_on_selector("#output", "el => el.value")

        print("---- comment/score area ----")
        print(comment)
        print("---- output/submission box (first 2000 chars) ----")
        print(output_val[:2000])

        pathlib.Path("output/grade_comment.txt").write_text(comment or "")
        pathlib.Path("output/submission_box.txt").write_text(output_val or "")

        page.screenshot(path=args.screenshot, full_page=False)

        # Save storage state so a later script can resume this "session" quickly
        # (note: this page keeps everything in DOM/JS state, not cookies, so this
        # mainly just preserves cookies/local storage if any is used)
        browser.close()


if __name__ == "__main__":
    main()
