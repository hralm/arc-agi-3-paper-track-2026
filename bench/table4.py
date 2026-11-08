"""Recomputes Table 4 of the paper (H5 attribution bench) from the bench JSON files.

    python3 bench/table4.py                       # the eight runs shipped in bench/results/table4/
    python3 bench/table4.py path/to/*.json        # your own runs (one file per arm)

RHAE is the mean of the total score over the seeds, with the population SD between seeds. Wins are levels completed,
summed over seeds and games. "No true instance" and "ambiguous" are the H5 counters `h5_win_no_true_instance` and
`h5_win_ambiguous`, summed the same way; the K0 arm runs without H5 and has none.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = sorted((HERE / "results" / "table4").glob("*.json"))
ORDER = ["k0", "default", "without-o3", "without-g1", "without-g2", "without-g4", "default-plus-o1", "random-order"]
LABELS = {"k0": "K0 (no H5; H1/H2 guides)", "default": "default (O3, G1, G2, G4)", "without-o3": "without O3",
          "without-g1": "without G1", "without-g2": "without G2", "without-g4": "without G4",
          "default-plus-o1": "default + O1", "random-order": "randomized order"}


def arm_of(path: Path, data: dict) -> str:
    for arm in ORDER:
        if f"_{arm}_" in path.name:
            return arm
    return path.stem


def summarize(data: dict) -> dict:
    totals = [r["total"] for r in data["runs"]]
    games = [g for r in data["runs"] for g in r["games"].values()]
    count = lambda key: sum(g["stats"].get(key, 0) for g in games)  # noqa: E731
    return {"rhae": statistics.mean(totals), "sd": statistics.pstdev(totals), "seeds": len(totals),
            "wins": sum(g["levels"] for g in games), "no_true": count("h5_win_no_true_instance"),
            "ambiguous": count("h5_win_ambiguous"), "h5": not data.get("h5_off", False)}


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv] or DEFAULT
    if not files:
        print("no JSON files found")
        return 1
    rows = []
    for path in files:
        data = json.loads(path.read_text())
        rows.append((arm_of(path, data), summarize(data)))
    rows.sort(key=lambda row: ORDER.index(row[0]) if row[0] in ORDER else len(ORDER))
    print("| arm | RHAE ± SD | wins | no true instance | ambiguous |")
    print("|---|---|---|---|---|")
    for arm, s in rows:
        no_true = s["no_true"] if s["h5"] else "—"
        ambiguous = s["ambiguous"] if s["h5"] else "—"
        print(f"| {LABELS.get(arm, arm)} | {s['rhae']:.4f} ± {s['sd']:.4f} | {s['wins']} | {no_true} | {ambiguous} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
