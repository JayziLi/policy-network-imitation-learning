"""
Full flow: enter NetID, load generated answers, Grade, check the confirm box,
click Submit. Screenshots the final state and saves the submit response
message for verification.
"""
import argparse
import pathlib
from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--answers", default="output/answers.txt")
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--screenshot", default="output/submit_result.png")
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

        # Confirm the "I submitted the competition file to Canvas A2C" box
        # for Question 9, now that the competition file has actually been
        # submitted.
        page.check("#answer1_9")

        page.click("#button_14")  # Grade
        page.wait_for_timeout(9000)

        comment = page.eval_on_selector("#comment", "el => el.textContent")
        print("---- grade result ----")
        print(comment)

        page.check("#submit_check")
        page.click("#button_15")  # Submit
        page.wait_for_timeout(6000)

        submit_message = page.eval_on_selector("#submit_message", "el => el.textContent")
        print("---- submit message ----")
        print(submit_message)

        pathlib.Path("output/submit_message.txt").write_text(submit_message or "")
        page.screenshot(path=args.screenshot, full_page=False)

        browser.close()


if __name__ == "__main__":
    main()
