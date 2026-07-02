"""
Round 2: refine around what round 1 found -- k=7 with negative speed_bias
(more cautious/turn-eager) was the clear winner, k=9 was uniformly bad
(possibly fixable with more peripheral awareness), k=5 was roughly flat.
"""
import json
from run_experiment import build_candidates_grid, run_race, parse_leaderboard

grid = []
for k in (5, 7):
    for sb in (-0.3, -0.2, -0.1):
        grid.append({"k": k, "speed_bias": sb, "peripheral_weight": 0.3})
for pw in (0.3, 0.7, 1.2):
    for sb in (-0.3, -0.2, -0.1):
        grid.append({"k": 9, "speed_bias": sb, "peripheral_weight": pw})
# keep a k=7 baseline and best-from-round-1 for reference
grid.append({"k": 7, "speed_bias": -0.1, "peripheral_weight": 0.7})
grid.append({"k": 7, "speed_bias": -0.2, "peripheral_weight": 0.7})

candidates = build_candidates_grid(grid)
print(f"Testing {len(candidates)} candidates...")
leader_text = run_race(candidates, 1200, "https://pages.cs.wisc.edu/~yw/CS540S26A2.html")
print("---- raw ----")
print(leader_text)

results = parse_leaderboard(leader_text, candidates)
results.sort(key=lambda r: -r["score"])
print("---- ranked ----")
for r in results:
    print(f"{r['score']:>8.1f}  {r['name']}  (pid {r['pid']})")

with open("output/experiment_round2.json", "w") as f:
    json.dump(results, f, indent=2)
