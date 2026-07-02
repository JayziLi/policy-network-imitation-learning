"""
Round 5: fine sweep around the round-4 winning region (k=5/k=7, small
positive speed_bias) at full competition length with mixed-team collisions.
"""
import json
from run_experiment import build_candidates_grid, run_race, parse_leaderboard

grid = [
    {"k": 5, "speed_bias": 0.03, "peripheral_weight": 0.3},
    {"k": 5, "speed_bias": 0.05, "peripheral_weight": 0.3},
    {"k": 5, "speed_bias": 0.07, "peripheral_weight": 0.3},
    {"k": 7, "speed_bias": 0.00, "peripheral_weight": 0.3},
    {"k": 7, "speed_bias": 0.05, "peripheral_weight": 0.3},
]
full_grid = []
for g in grid:
    for _ in range(3):
        full_grid.append(dict(g))

candidates = build_candidates_grid(full_grid)
print(f"Testing {len(candidates)} cars (mixed teams, 5000 frames)...")
leader_text = run_race(candidates, 5000, "https://pages.cs.wisc.edu/~yw/CS540S26A2.html", same_team=False)
print("---- raw ----")
print(leader_text)

results = parse_leaderboard(leader_text, candidates)
results.sort(key=lambda r: -r["score"])

from collections import defaultdict
agg = defaultdict(list)
for r in results:
    agg[r["name"]].append(r["score"])
print("---- averaged by config ----")
for name, scores in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"{sum(scores)/len(scores):>8.1f}  (n={len(scores)})  {name}  scores={sorted(scores)}")

with open("output/experiment_round5.json", "w") as f:
    json.dump(results, f, indent=2)
