"""
Round 3: fixed methodology -- all candidates on the SAME team (group=0) so
they never collide with each other, isolating pure track-navigation quality
(wall avoidance + speed) from opponent-pairing luck. Re-run the original
round-1 grid plus round-2's k values under this fair comparison.
"""
import json
from run_experiment import build_candidates_grid, run_race, parse_leaderboard

grid = []
for k in (5, 7, 9):
    for sb in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
        grid.append({"k": k, "speed_bias": sb, "peripheral_weight": 0.3})

candidates = build_candidates_grid(grid)
print(f"Testing {len(candidates)} candidates (same team, isolated)...")
leader_text = run_race(candidates, 1500, "https://pages.cs.wisc.edu/~yw/CS540S26A2.html", same_team=True)
print("---- raw ----")
print(leader_text)

results = parse_leaderboard(leader_text, candidates)
results.sort(key=lambda r: -r["score"])
print("---- ranked ----")
for r in results:
    print(f"{r['score']:>8.1f}  {r['name']}  (pid {r['pid']})")

with open("output/experiment_round3.json", "w") as f:
    json.dump(results, f, indent=2)
