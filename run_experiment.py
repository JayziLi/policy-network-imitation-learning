"""
A/B test candidate competition networks against each other using the
assignment's REAL Competition Simulator (env_c / run_race_page in comp mode),
via a headless browser. This avoids re-implementing their physics engine --
we just feed it a synthetic multi-entry "file_c" submission blob (same format
their own "Generate" button produces) and read the real leaderboard back.
"""
import argparse
import itertools
import json
import numpy as np
from playwright.sync_api import sync_playwright

from competition_net import crafted_net_general, fmt_net


def make_block(wisc, group, icon, player_id, W1, W2, W3, W1b=None, W2b=None, W3b=None):
    net1 = fmt_net(W1, W2, W3)
    if W1b is None:
        W1b, W2b, W3b = W1, W2, W3
    net2 = fmt_net(W1b, W2b, W3b)
    pid = str(player_id).zfill(4)
    return (f"** wisc : {wisc}\n** group : {group}\n** icon : {icon}\n** id : {pid}\n"
            f"** weights : \n{net1}\n** second : \n{net2}")


def build_candidates():
    """Grid of (k, speed_bias) candidates, each its own team so they all
    collide with each other like a real race, plus a unique 4-digit id so we
    can read off which is which from the leaderboard."""
    cands = []
    pid = 1
    for k in (5, 7, 9):
        for speed_bias in (-0.1, 0.0, 0.1, 0.2):
            name = f"k{k}_sb{speed_bias:+.2f}"
            W1, W2, W3 = crafted_net_general(k, max(13, k * 2), 4, speed_bias=speed_bias)
            W1, W2, W3 = np.round(W1, 4), np.round(W2, 4), np.round(W3, 4)
            cands.append({"name": name, "k": k, "speed_bias": speed_bias,
                          "pid": pid, "W": (W1, W2, W3)})
            pid += 1
    return cands


def build_candidates_grid(grid):
    """grid: list of dicts with keys k, speed_bias, peripheral_weight."""
    cands = []
    for pid, g in enumerate(grid, start=1):
        k, sb, pw = g["k"], g["speed_bias"], g.get("peripheral_weight", 0.3)
        name = f"k{k}_sb{sb:+.2f}_pw{pw:.2f}"
        W1, W2, W3 = crafted_net_general(k, max(13, k * 2), 4, speed_bias=sb, peripheral_weight=pw)
        W1, W2, W3 = np.round(W1, 4), np.round(W2, 4), np.round(W3, 4)
        cands.append({"name": name, "k": k, "speed_bias": sb, "peripheral_weight": pw,
                      "pid": pid, "W": (W1, W2, W3)})
    return cands


def run_race(candidates, time_frames, url, track_text="", same_team=True):
    blocks = []
    for i, c in enumerate(candidates):
        W1, W2, W3 = c["W"]
        # Same team by default -> no inter-candidate collisions, isolating
        # pure track-navigation quality (wall avoidance / speed) without
        # opponent-pairing luck confounding the comparison. Set
        # same_team=False to test crash-robustness against a mixed field.
        group = 0 if same_team else (i % 3)
        block = make_block("test", group, "car", c["pid"], W1, W2, W3)
        blocks.append(block)
    file_c_text = "\n----- ----- ----- -----\n".join(blocks)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        page.fill("#file_c", file_c_text)
        if track_text:
            page.fill("#track_c", track_text)
        page.fill("#time_c", str(time_frames))
        page.click("#button_5")  # Start

        # Poll the leaderboard until it stops changing (race finished) rather
        # than guessing a fixed wait -- more robust to variable frame rate
        # when many cars are in the scene.
        min_wait = time_frames / 70.0
        page.wait_for_timeout(int(min_wait * 1000))
        prev = None
        stable_count = 0
        max_polls = 60
        for _ in range(max_polls):
            cur = page.eval_on_selector("#leader_c", "el => el.textContent")
            if cur == prev and cur and cur.strip():
                stable_count += 1
                if stable_count >= 2:
                    break
            else:
                stable_count = 0
            prev = cur
            page.wait_for_timeout(1500)

        leader_text = page.eval_on_selector("#leader_c", "el => el.textContent")
        browser.close()
        return leader_text


def parse_leaderboard(text, candidates):
    """Lines look like: '<score> ... [<k>] ... <emoji> ... "<name>" ... '
    where <name> is '<group>--<zfilled id>'."""
    pid_to_cand = {str(c["pid"]).zfill(4): c for c in candidates}
    results = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split("...")]
        if len(parts) < 4:
            continue
        try:
            score = float(parts[0])
        except ValueError:
            continue
        name_field = parts[3].strip('"').strip()
        pid = name_field.split("--")[-1] if "--" in name_field else name_field
        cand = pid_to_cand.get(pid)
        results.append({"score": score, "pid": pid, "name": cand["name"] if cand else "?"})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--out", default="output/experiment_results.json")
    args = ap.parse_args()

    candidates = build_candidates()
    print(f"Testing {len(candidates)} candidates for {args.frames} frames...")
    leader_text = run_race(candidates, args.frames, args.url)
    print("---- raw leaderboard ----")
    print(leader_text)

    results = parse_leaderboard(leader_text, candidates)
    results.sort(key=lambda r: -r["score"])
    print("---- parsed & ranked ----")
    for r in results:
        print(f"{r['score']:>8.1f}  {r['name']}  (pid {r['pid']})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
