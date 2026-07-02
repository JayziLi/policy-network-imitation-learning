"""
Fill in the Q9 competition submission form on the live page (using the
page's own "Generate" button, so the output format is guaranteed correct)
and download the resulting text file content.
"""
import argparse
import pathlib
from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--group", default="0")
    ap.add_argument("--icon", default="car")
    ap.add_argument("--player-id", default="1234")
    ap.add_argument("--net", default="output/competition_net_final.txt")
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--out", default="output/q9_submission.txt")
    args = ap.parse_args()

    net_text = pathlib.Path(args.net).read_text()

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

        page.fill("#wisc_cp", args.netid)
        page.fill("#group_cp", args.group)
        page.fill("#icon_cp", args.icon)
        page.fill("#id_cp", args.player_id)
        page.fill("#net1_cp", net_text)
        page.fill("#net2_cp", net_text)  # same network for both slots

        page.click("#button_12")  # Generate
        page.wait_for_timeout(1500)

        note = page.eval_on_selector("#note_cp", "el => el.value")
        out_cp = page.eval_on_selector("#out_cp", "el => el.value")

        print("---- generator note/warnings ----")
        print(note)
        print("---- generated file (first 500 chars) ----")
        print(out_cp[:500])

        pathlib.Path(args.out).write_text(out_cp, encoding="utf-8")
        print(f"Saved to {args.out}")

        browser.close()


if __name__ == "__main__":
    main()
