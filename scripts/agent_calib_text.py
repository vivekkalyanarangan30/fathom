"""Calibration text from OpenHands trajectories of repositories disjoint from the benchmark sessions: writes DIR/train.txt and DIR/test.txt
(rendered with the same ChatML convention as R_agent.py), taken from the far end of R_agent's seeded repo order.
Usage: python scripts/agent_calib_text.py --exclude results/agent_prev.json --out-dir results/agentcal"""
import os, json, random, argparse
from datasets import load_dataset
from common import render_msg
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nebius/SWE-rebench-openhands-trajectories"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude", required=True, help="agent result JSON whose session repos must not be used"); ap.add_argument("--out-dir", required=True)
ap.add_argument("--chars", type=int, default=2_500_000)
a = ap.parse_args(); assert 100_000 <= a.chars <= 50_000_000
excl = {s["meta"]["repo"] for s in json.load(open(a.exclude))["sessions"]}
ds = load_dataset(a.dataset, split="train"); by_repo = {}
for i, r in enumerate(ds): by_repo.setdefault(r["repo"], []).append(i)
repos = sorted(r for r, ix in by_repo.items() if len(ix) >= 2); random.Random(a.seed).shuffle(repos)      # same order as R_agent.load_sessions
pool = [r for r in reversed(repos) if r not in excl]
os.makedirs(a.out_dir, exist_ok=True); used = []
for split in ("train", "test"):
    buf = []; n = 0
    while n < a.chars:
        repo = pool.pop(0); used.append(repo)
        for i in sorted(by_repo[repo], key=lambda i: ds[i]["trajectory_id"]):
            tr = ds[i]["trajectory"]; tr = json.loads(tr) if isinstance(tr, str) else tr
            txt = "".join(render_msg(m) for m in tr); buf.append(txt); n += len(txt)
    with open(os.path.join(a.out_dir, f"{split}.txt"), "w", encoding="utf-8") as f: f.write("".join(buf))
    print(f"{split}: {n} chars from {len(used)} repos so far", flush=True)
json.dump({"repos": used, "excluded": sorted(excl)}, open(os.path.join(a.out_dir, "repos.json"), "w"), indent=1)
