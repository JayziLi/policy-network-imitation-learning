"""
Round 4: final validation. Take the top candidates from round 3's isolated
(same-team) test and race them head-to-head with REAL cross-team collisions
(mixed teams) at competition-length duration (5000 frames) to make sure the
"prefer speeding up a bit" winner doesn't fall apart once it has to dodge
other cars, not just walls.
"""
import json
from run_experiment import build_candidates_grid, run_race, parse_leaderboard

grid = [
    {"k": 5, "speed_bias": 0.10, "peripheral_weight": 0.3},
    {"k": 5, "speed_bias": 0.00, "peripheral_weight": 0.3},
    {"k": 5, "speed_bias": -0.10, "peripheral_weight": 0.3},
    {"k": 7, "speed_bias": 0.00, "peripheral_weight": 0.3},
    {"k": 5, "speed_bias": 0.05, "peripheral_weight": 0.3},
]
# duplicate each config 3x under different pids so each config gets 3 cars
# scattered across the 3 teams -- averages out any single-pairing luck.
full_grid = []
for g in grid:
    for _ in range(3):
        full_grid.append(dict(g))

candidates = build_candidates_grid(full_grid)
print(f"Testing {len(candidates)} cars (mixed teams, 5000 frames, competition length)...")
leader_text = run_race(candidates, 5000, "https://pages.cs.wisc.edu/~yw/CS540S26A2.html", same_team=False)
print("---- raw ----")
print(leader_text)

results = parse_leaderboard(leader_text, candidates)
results.sort(key=lambda r: -r["score"])
print("---- ranked ----")
for r in results:
    print(f"{r['score']:>8.1f}  {r['name']}  (pid {r['pid']})")

# aggregate by config name (mean score across its 3 cars / 2 nets each = 6 entries)
from collections import defaultdict
agg = defaultdict(list)
for r in results:
    agg[r["name"]].append(r["score"])
print("---- averaged by config ----")
for name, scores in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"{sum(scores)/len(scores):>8.1f}  (n={len(scores)})  {name}")

with open("output/experiment_round4.json", "w") as f:
    json.dump(results, f, indent=2)
