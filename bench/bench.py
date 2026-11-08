"""Local bench: runs the explorer K on the public ARC-AGI-3 games, OFFLINE, scored by the toolkit's RHAE.

Run from inside a clone of the official ARC-AGI-3-Kaggle-Starter (see ../README.md, "Reproducing the results"):

    cd ARC-AGI-3-Kaggle-Starter
    .venv/bin/python /path/to/repo/bench/bench.py --agents k --max-steps 1500 --seeds 4

Each (agent, seed) runs with a fresh Arcade and a fresh scorecard. Writes a JSON to bench/results/ (or --out) and
prints the per-game table and the aggregate. The H5 attribution arms of Table 4 are selected by the environment
variables read in `load_agent` (see README.md in this folder).
"""
import argparse
import importlib.util
import json
import logging
import os
import random
import zlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VENDOR))
from arc_agi import Arcade, OperationMode  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
AGENTS = {"k": REPO / "agent" / "graph_explorer.py"}
OUT = REPO / "bench" / "results"


def load_agent(name: str):
    path = AGENTS[name]
    spec = importlib.util.spec_from_file_location(f"agent_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if name == "k":
        if os.environ.get("BENCH_H5_OFF") == "1":
            mod.H5_ON = False             # K0 control: same file and same fixes, only H5 switched off
        for fam in filter(None, os.environ.get("BENCH_H5_ABLATE", "").split(",")):
            if fam not in mod.H5_FAMILY_ON:
                raise ValueError(f"unknown H5 family for ablation: {fam}")
            mod.H5_FAMILY_ON[fam] = False
        for fam in filter(None, os.environ.get("BENCH_H5_FAMILIES", "").split(",")):
            if fam not in mod.H5_FAMILY_ON:
                raise ValueError(f"unknown H5 family: {fam}")
            mod.H5_FAMILY_ON[fam] = True  # re-enables a family that is off by default (O1)
        if os.environ.get("BENCH_H5_O1_COUPLED") == "1":
            original_guide = mod.H5.guide

            def coupled_o1_guide(h5, grid, node, avail):
                top = h5.top()
                if top is not None and top.fam == "O1" and not top.coupled:
                    return None
                return original_guide(h5, grid, node, avail)

            mod.H5.guide = coupled_o1_guide
        if os.environ.get("BENCH_H5_RANDOM_ORDER") == "1":
            def random_top(h5):
                # The priority belongs to the instance, not to the call to top().
                # The generator is independent of the RNG that chooses K's actions.
                if not hasattr(h5, "_bench_order"):
                    seed = int(os.environ["BENCH_SEED"])
                    gid = h5.ag.game_id
                    h5._bench_rng = random.Random(seed * 100003 + zlib.crc32(gid.encode()) % 100003)
                    h5._bench_order = {}
                h5.cross_changed = False
                active = [h for h in h5.hyps.values() if h.state in ("active", "corroborated")]
                for h in h5.hyps.values():
                    key = (h.fam, h.bind)
                    if key not in h5._bench_order:
                        h5._bench_order[key] = h5._bench_rng.random()
                if not active:
                    h5.top_key = None
                    return None
                chosen = min(active, key=lambda h: (h5._bench_order[(h.fam, h.bind)], h.fam, repr(h.bind)))
                h5.top_key = (chosen.fam, chosen.bind)
                return chosen

            mod.H5.top = random_top
    return mod.MyAgent


def per_game_scores(sc) -> dict[str, dict]:
    d = sc.model_dump() if hasattr(sc, "model_dump") else {}
    out = {}
    for e in d.get("environments") or []:
        gid = str(e.get("id", "?")).split("-")[0]
        runs = e.get("runs") or []
        r = runs[-1] if runs else {}
        out[gid] = {"score": float(r.get("score", 0.0)),
                    "level_actions": r.get("level_actions") or [],
                    "level_scores": r.get("level_scores") or []}
    return out


def run(agent_name: str, seed: int, games: list[str], max_steps: int) -> dict:
    Cls = load_agent(agent_name)
    Cls.MAX_ACTIONS = max_steps
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    rows = {}
    t0 = time.time()
    for gid in games:
        env = arc.make(gid)
        if env is None:
            continue
        os.environ["BENCH_SEED"] = str(seed)
        agent = Cls(card_id=f"bench-{agent_name}-{seed}", game_id=gid, agent_name=f"{agent_name}.{gid}",
                    ROOT_URL="http://localhost", record=False, arc_env=env, tags=["bench"])
        random.seed(seed * 100003 + zlib.crc32(gid.encode()) % 100003)
        if hasattr(agent, "rng"):
            agent.rng = random.Random(seed * 100003 + zlib.crc32(gid.encode()) % 100003)
        agent.main()
        f = agent.frames[-1]
        rows[gid] = {"levels": f.levels_completed, "win_levels": f.win_levels,
                     "actions": agent.action_counter, "state": str(f.state).split(".")[-1],
                     "stats": {k: int(v) for k, v in dict(getattr(agent, "stats", {}) or {}).items()}}
    sc = arc.get_scorecard()
    pg = per_game_scores(sc)
    for gid in rows:
        info = pg.get(gid, {})
        rows[gid]["score"] = info.get("score", 0.0)
        rows[gid]["level_actions"] = info.get("level_actions", [])
    return {"agent": agent_name, "seed": seed, "total": float(getattr(sc, "score", 0.0)),
            "secs": round(time.time() - t0, 1), "games": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", default="k")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--games", default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR if not args.quiet else logging.CRITICAL)
    logging.getLogger().setLevel(logging.ERROR)

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
    if args.games:
        wanted = {g.strip() for g in args.games.split(",")}
        games = [g for g in games if g in wanted]
    agents = [a.strip() for a in args.agents.split(",")]

    results = []
    for a in agents:
        for s in range(args.seed_offset, args.seed_offset + args.seeds):
            r = run(a, s, games, args.max_steps)
            results.append(r)
            print(f"{a} seed{s}: total={r['total']:.4f} "
                  f"levels={sum(g['levels'] for g in r['games'].values())} "
                  f"actions={sum(g['actions'] for g in r['games'].values())} ({r['secs']}s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    arm = os.environ.get("BENCH_ARM", "")
    if arm and not all(c.isalnum() or c in "-_" for c in arm):
        raise ValueError("BENCH_ARM may contain only letters, digits, hyphen or underscore")
    suffix = f"_{arm}" if arm else ""
    out = args.out / f"{stamp}{suffix}_steps{args.max_steps}_seeds{args.seeds}_off{args.seed_offset}.json"
    out.write_text(json.dumps({"max_steps": args.max_steps, "games": games,
                               "h5_off": os.environ.get("BENCH_H5_OFF") == "1",
                               "h5_ablate": sorted(filter(None, os.environ.get("BENCH_H5_ABLATE", "").split(","))),
                               "h5_families": sorted(filter(None, os.environ.get("BENCH_H5_FAMILIES", "").split(","))),
                               "h5_o1_coupled": os.environ.get("BENCH_H5_O1_COUPLED") == "1",
                               "h5_random_order": os.environ.get("BENCH_H5_RANDOM_ORDER") == "1",
                               "runs": results}, indent=1))

    # per-game table: mean levels and mean score per agent
    print("\ngame     " + "".join(f"| {a:>14} " for a in agents) + "| win_levels")
    print("---------" + "|----------------" * len(agents) + "|-----")
    for gid in games:
        cells = []
        for a in agents:
            rs = [r["games"][gid] for r in results if r["agent"] == a and gid in r["games"]]
            lv = statistics.mean(x["levels"] for x in rs) if rs else 0
            scr = statistics.mean(x["score"] for x in rs) if rs else 0
            cells.append(f"| {lv:4.1f}n {scr:6.3f}  ")
        wl = next((r["games"][gid]["win_levels"] for r in results if gid in r["games"]), "?")
        print(f"{gid:8} " + "".join(cells) + f"| {wl}")
    print("\naggregate (mean ± population SD of the total score per agent):")
    for a in agents:
        ts = [r["total"] for r in results if r["agent"] == a]
        dp = statistics.pstdev(ts) if len(ts) > 1 else 0.0
        lv = statistics.mean(sum(g["levels"] for g in r["games"].values()) for r in results if r["agent"] == a)
        print(f"  {a:>3}: {statistics.mean(ts):.4f} ± {dp:.4f}   levels/run {lv:.1f}   (n={len(ts)})")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
