# bench

The offline local bench: K alone (no language model) on the 25 public ARC-AGI-3 games, scored by the toolkit's RHAE.

| file | contents |
|---|---|
| `bench.py` | runs `agent/graph_explorer.py` on every public game for N seeds and writes one JSON per run |
| `table3.py` | recomputes Table 3 from the JSON files (by default, the eight shipped in `results/table3/`) |
| `results/table3/` | the eight runs of Table 3 (25 games × 4 seeds × 1,500 steps, 3 November 2026) |

## The eight arms of Table 3

Each arm is the same file with one environment variable; `BENCH_ARM` only names the output file.

| arm | environment |
|---|---|
| K0 (no H5; H1/H2 guides) | `BENCH_H5_OFF=1 BENCH_ARM=k0` |
| default (O3, G1, G2, G4) | `BENCH_ARM=default` |
| without O3 | `BENCH_H5_ABLATE=O3 BENCH_ARM=without-o3` |
| without G1 | `BENCH_H5_ABLATE=G1 BENCH_ARM=without-g1` |
| without G2 | `BENCH_H5_ABLATE=G2 BENCH_ARM=without-g2` |
| without G4 | `BENCH_H5_ABLATE=G4 BENCH_ARM=without-g4` |
| default + O1 | `BENCH_H5_FAMILIES=O1 BENCH_ARM=default-plus-o1` |
| randomized order | `BENCH_H5_RANDOM_ORDER=1 BENCH_ARM=random-order` |

`BENCH_H5_O1_COUPLED=1` (O1 guides only when coupled) is also available; it is not part of Table 3.

Example, from inside the official starter clone (see the main README for the setup):

```bash
BENCH_H5_ABLATE=G2 BENCH_ARM=without-g2 .venv/bin/python /path/to/repo/bench/bench.py \
    --agents k --max-steps 1500 --seeds 4 --quiet
python3 /path/to/repo/bench/table3.py /path/to/repo/bench/results/*_without-g2_*.json
```

In the paper's runs, one arm (4 seeds × 25 games × 1,500 steps) took 13 minutes for K0 and 35–37 minutes for each H5
arm, with seven arms running in parallel on one desktop CPU (the `secs` field of each run). The runs are deterministic
for a given seed: the same arm and seed reproduce the same JSON, apart from `secs`.

## About the shipped results

The JSON files in `results/table3/` are the runs reported in the paper. They were produced by the same bench under the
development names of the counters; the shipped copies use the public names of `graph_explorer.py` (for example,
`h5_win_no_true_instance` and `h5_win_ambiguous`), which is the only change. A run of this bench on a short
configuration (5 games × 2 seeds × 400 steps, four arms) reproduces the development bench byte for byte under that
renaming.
