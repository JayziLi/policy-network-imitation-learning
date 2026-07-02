"""
Record a short demo clip of the final competition network actually driving
in the assignment's real simulator (with a few opponent cars and an
auto-generated track), then convert it to a GIF for the README.
"""
import argparse
import pathlib
import shutil
import subprocess
from playwright.sync_api import sync_playwright

FFMPEG = r"C:\Users\Ji's Destop\AppData\Local\ms-playwright\ffmpeg-1011\ffmpeg-win64.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--net", default="output/competition_net_final.txt")
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--video-dir", default="output/video")
    ap.add_argument("--gif-out", default="demo/demo.gif")
    args = ap.parse_args()

    net_text = pathlib.Path(args.net).read_text()
    video_dir = pathlib.Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.gif_out).parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1040, "height": 900},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1040, "height": 900},
        )
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")

        page.fill("#id", args.netid)
        page.click("#button_0")
        page.wait_for_function(
            "document.getElementById('test_0').value.trim().length > 100", timeout=20000
        )
        page.wait_for_timeout(500)

        # Full simulator (Question 8 section): my final network + 3 opponent cars.
        page.fill("#sensor", net_text)
        page.fill("#sensor_icon", "car")
        page.fill("#other_too", "3")
        page.click("#button_9")  # Restart
        page.wait_for_timeout(500)

        page.locator("#env_too").scroll_into_view_if_needed()
        page.wait_for_timeout(int(args.seconds * 1000))

        context.close()
        browser.close()

        video_files = sorted(video_dir.glob("*.webm"), key=lambda f: f.stat().st_mtime)
        webm_path = video_files[-1]
        print(f"Recorded: {webm_path}")

    # This Playwright-bundled ffmpeg is a stripped-down build (only pad/crop/scale
    # filters compiled in -- no fps/palettegen/paletteuse), so just scale down and
    # let ffmpeg's native GIF encoder handle the rest.
    subprocess.run([FFMPEG, "-y", "-i", str(webm_path), "-vf", "scale=560:-1",
                     "-r", "12", args.gif_out], check=True)

    print(f"GIF saved to {args.gif_out}")
    size_kb = pathlib.Path(args.gif_out).stat().st_size / 1024
    print(f"Size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
