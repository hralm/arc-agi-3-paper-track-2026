#!/usr/bin/env python3
"""phase_a.py: from the output of a Kaggle run of one of our notebooks to the rows of Tables 1 and 3 (paper, Section 5).

A Kaggle run of any notebook in `notebooks/` leaves, in its output, the kernel log, `summary.txt`, `benchmark.json` and the
two modules the notebook wrote. `kaggle kernels output <user>/<slug> -p <folder>` downloads them. This script reads such a
folder (ours are shipped in `results/phase-a/`, one per round, with the files it needs and nothing else), checks the
validity conditions fixed before the runs, and prints the per-game scores and the explorer's counters.

    python3 bench/phase_a.py                       # every shipped round: validity, Table 3, Table 1, batch summaries
    python3 bench/phase_a.py <folder> [<folder>...]  # your own Kaggle outputs (one folder per run)
    python3 bench/phase_a.py --csv games.csv       # also write one row per round and game

Validity (the letter of the pre-registration, Section 5): the run finished its three games with no traceback and at least one
level completed in at least one game; in the arms with the explorer, `layer_exc` and `k_exc` are zero in every game and at least
one `cert_*`, `h8_masked_keys` or `k_*` counter is positive in some game; the modules the kernel wrote match the digests in
`notebooks/<slug>/SHA256SUMS` when both are present. The reading rule compares arms with three valid rounds against the
control's rounds of the same batch: an effect is read only if the difference of means exceeds twice the larger sample SD.
`results/phase-a/rounds.tsv` records which rounds the paper read and why the others were not read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results/phase-a"
NOTEBOOKS = REPO / "notebooks"

COUNTER_ROW = re.compile(r"^(\S+)\s+((?:\d+\s+)+?)(\d+/\d+)\s+(\[.*?\])\s+(\d+)\s*$")
SCORE_ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+([\d.]+)\s*$")


def log_lines(path: Path) -> tuple[list[str], list[str]]:
    """(stdout, stderr) lines of a Kaggle kernel log (JSON list of events) or of a plain text log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        events = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines(), []
    out = [l for e in events if e.get("stream_name") == "stdout" for l in e["data"].splitlines()]
    err = [l for e in events if e.get("stream_name") == "stderr" for l in e["data"].splitlines()]
    return out, err


def parse_counters(lines: list[str]) -> dict[str, dict]:
    """The last counters table printed by the notebook: {game: {counter: value, turns, hidden, hint_max}}."""
    heads = [i for i, l in enumerate(lines) if l.startswith("game   model_act_")]
    if not heads:
        return {}
    header = lines[heads[-1]].split()
    keys = header[1:header.index("turns(model/K)")]
    # the header truncates names to ten characters; restore the published names in order
    full = ["model_act_total", "model_act_certified", "cert_emitted", "cert_falsified", "cert_revoked_reindex",
            "cert_revoked_reopen", "h8_masked_keys", "h8_refused", "k_failure", "k_stall", "k_repeat", "k_time", "k_yield",
            "model_batches", "model_deaths", "layer_exc", "k_exc", "model_exc", "fallback_act", "cert_hidden_death", "k_endgame"]
    names = full[:len(keys)]
    table = {}
    for l in lines[heads[-1] + 1:heads[-1] + 4]:
        m = COUNTER_ROW.match(l)
        if not m:
            break
        values = [int(v) for v in m.group(2).split()]
        row = dict(zip(names, values))
        row["turns_model_k"], row["hidden_levels"], row["hint_max"] = m.group(3), m.group(4), int(m.group(5))
        table[m.group(1)] = row
    return table


def parse_scores(lines: list[str]) -> dict[str, dict]:
    heads = [i for i, l in enumerate(lines) if l.startswith("game   levels actions")]
    if not heads:
        return {}
    table = {}
    for l in lines[heads[-1] + 1:heads[-1] + 4]:
        m = SCORE_ROW.match(l)
        if m:
            table[m.group(1)] = {"levels": int(m.group(2)), "actions": int(m.group(3)), "tokens": int(m.group(4)),
                                 "score": float(m.group(7))}
    return table


def parse_summary(path: Path) -> dict:
    out = {"games": {}}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.startswith("mean score:"):
            out["mean"] = float(l.split()[-1])
        elif l.startswith("total actions:"):
            out["actions"] = int(l.split()[-1])
        elif l.startswith("runs:"):
            out["runs"] = l.split(":", 1)[1].strip()
        elif l.startswith("games:"):
            out["n_games"] = int(l.split()[-1])
        elif l.startswith("benchmark:"):
            out["label"] = l.split(":", 1)[1].strip()
        m = re.match(r"\s+(\S+): score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)", l)
        if m:
            out["games"][m.group(1)[:4]] = {"score": float(m.group(2)), "levels": float(m.group(3)), "of": int(m.group(4)),
                                            "actions": int(m.group(5)), "tokens": int(m.group(6))}
    return out


def module_check(folder: Path, kernel: str) -> str:
    sums = NOTEBOOKS / kernel / "SHA256SUMS"
    mods = [p for p in folder.glob("*.py")]
    if not mods or not sums.exists():
        return "n/a"
    expected = {l.split()[1]: l.split()[0] for l in sums.read_text().splitlines()}
    bad = [p.name for p in mods if hashlib.sha256(p.read_bytes()).hexdigest() != expected.get(p.name)]
    return "ok" if not bad else "MISMATCH " + ",".join(bad)


def read_round(folder: Path, kernel: str = "", with_explorer: bool | None = None) -> dict:
    log = next(iter(sorted(folder.glob("*.log"))), None)
    out, err = log_lines(log) if log else ([], [])
    counters, scores = parse_counters(out), parse_scores(out)
    summary = parse_summary(folder / "summary.txt") if (folder / "summary.txt").exists() else {"games": {}}
    bench = json.loads((folder / "benchmark.json").read_text()) if (folder / "benchmark.json").exists() else {}
    runs = bench.get("game_runs", [])
    games = {g["game_id"][:4]: g for g in runs}
    if with_explorer is None:
        with_explorer = bool(counters) or "k-" in kernel or "duck" in kernel
    traceback = any("Traceback" in l for l in out + err)
    complete = len(runs) == 3 and summary.get("n_games") == 3
    levels = sum(int(g.get("levels_completed", 0)) for g in runs)
    exc_ok = (not with_explorer) or (bool(counters) and all(r["layer_exc"] == 0 and r["k_exc"] == 0 for r in counters.values()))
    active = (not with_explorer) or any(
        any(v > 0 for k, v in r.items() if isinstance(v, int) and (k.startswith("cert_") or k == "h8_masked_keys" or k.startswith("k_")))
        for r in counters.values())
    modules = module_check(folder, kernel) if kernel else "n/a"
    valid = complete and not traceback and levels > 0 and exc_ok and active and not modules.startswith("MISMATCH")
    if runs:   # raw scores with every decimal place (the summary rounds to two); the paper rounds only when it prints
        summary["mean"] = statistics.mean(float(g["final_score"]) for g in runs)
        for g in runs:
            if g["game_id"][:4] in summary["games"]:
                summary["games"][g["game_id"][:4]]["score"] = float(g["final_score"])
    return {"folder": folder, "complete": complete, "traceback": traceback, "levels": levels, "exc_ok": exc_ok,
            "active": active, "modules": modules, "valid": valid, "mean": summary.get("mean"),
            "actions": summary.get("actions"), "games": games, "scores": scores, "summary_games": summary["games"],
            "counters": counters, "label": summary.get("label", "")}


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main(argv: list[str]) -> int:
    csv_path = None
    if "--csv" in argv:
        i = argv.index("--csv")
        csv_path = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    rows_csv = []
    if argv:   # the reader's own outputs
        for a in argv:
            r = read_round(Path(a), "")
            print(f"{a}: valid={r['valid']} complete={r['complete']} traceback={r['traceback']} levels={r['levels']} "
                  f"exceptions_zero={r['exc_ok']} explorer_active={r['active']} modules={r['modules']} mean={fmt(r['mean'])}")
            for g, s in sorted(r["summary_games"].items()):
                c = r["counters"].get(g, {})
                print(f"   {g}: score={s['score']:.2f} levels={s['levels']:.0f}/{s['of']} actions={s['actions']} "
                      + " ".join(f"{k}={v}" for k, v in c.items() if k in ("cert_emitted", "cert_falsified", "h8_masked_keys",
                                                                       "layer_exc", "k_exc", "turns_model_k", "hidden_levels")))
        return 0
    declared = list(csv.DictReader((RESULTS / "rounds.tsv").open(encoding="utf-8"), delimiter="\t"))
    results = []
    print("=== rounds (validity computed from the files; `read` as declared in rounds.tsv) ===")
    for d in declared:
        folder = RESULTS / d["batch"] / f"{d['arm']}-{d['round']}"
        kernel = d["kernel"] if d["kernel"].startswith("what-") else ""
        with_explorer = d["arm"] not in ("control", "base")
        r = read_round(folder, kernel, with_explorer)
        r.update(d)
        results.append(r)
        flag = "" if (d["read"] != "yes" or r["valid"]) else "   <-- declared read but not valid"
        print(f"{d['batch']:13s} {d['arm']:22s} {d['round']:14s} valid={str(r['valid']):5s} read={d['read']:11s} "
              f"mean={fmt(r['mean']):>6s} actions={r['actions'] or '—':>5} modules={r['modules']}{flag}")
        for g, s in sorted(r["summary_games"].items()):
            c = r["counters"].get(g, {})
            rows_csv.append({"batch": d["batch"], "arm": d["arm"], "round": d["round"], "kaggle_version": d["kaggle_version"],
                             "read": d["read"], "valid": r["valid"], "game": g, "score": s["score"], "levels": s["levels"],
                             "levels_total": s["of"], "actions": s["actions"], "tokens": s["tokens"],
                             **{k: c.get(k, "") for k in ("model_act_total", "model_act_certified", "cert_emitted", "cert_falsified",
                                                          "cert_revoked_reindex", "cert_revoked_reopen", "h8_masked_keys", "h8_refused",
                                                          "k_failure", "k_stall", "k_repeat", "k_time", "k_yield", "model_batches",
                                                          "model_deaths", "layer_exc", "k_exc", "model_exc", "fallback_act",
                                                          "cert_hidden_death", "k_endgame", "turns_model_k", "hidden_levels")}})
    print("\n=== Table 3: second batch, one row per round (mean RHAE; per game; actions summed) ===")
    for r in results:
        if r["batch"] == "second-batch" and r["arm"] != "duck-fork-k-playing" and r["complete"] and \
                (r["read"] == "yes" or not r["valid"]):   # as in the paper: rounds read, plus the invalid one it reports
            g = r["summary_games"]
            print(f"| {r['arm']} | {r['round']} | {fmt(r['mean'])} | {g['ls20']['score']:.2f} | {g['tu93']['score']:.2f} | "
                  f"{g['vc33']['score']:.2f} | {r['actions']:,} | {r['note']} |")
    print("\n=== Table 1: certificate counters per game, Duck line phase A ===")
    for r in results:
        if r["batch"] == "duck-line":
            for g, c in sorted(r["counters"].items()):
                lv = int(r["games"][g]["levels_completed"]) if g in r["games"] else "?"
                print(f"| {r['arm']} | {g} | {lv} | {c['model_act_certified']} | {c['cert_emitted']} | {c['cert_falsified']} | "
                      f"{c['cert_revoked_reindex'] + c['cert_revoked_reopen']} | {c['h8_masked_keys']} | {c['hidden_levels']} |")
    print("\n=== batch summaries (rounds read; arm with three rounds against the control of its batch) ===")
    for batch, control in (("first-batch", "base"), ("second-batch", "control")):
        arms: dict[str, list[float]] = {}
        for r in results:
            if r["batch"] == batch and r["read"] == "yes" and r["valid"]:
                arms.setdefault(r["arm"], []).append(r["mean"])
        base = arms.get(control, [])
        for arm, v in arms.items():
            sd = statistics.stdev(v) if len(v) > 1 else float("nan")
            line = f"{batch:13s} {arm:18s} n={len(v)} mean={statistics.mean(v):.2f} sd={sd:.2f}"
            if arm != control and len(v) >= 3 and len(base) >= 3:
                diff, bar = abs(statistics.mean(base) - statistics.mean(v)), 2 * max(statistics.stdev(base), sd)
                line += f"  |diff|={diff:.2f} bar={bar:.2f} -> {'effect' if diff > bar else 'no measurable effect'}"
            print(line)
    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\nwritten {csv_path} ({len(rows_csv)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
