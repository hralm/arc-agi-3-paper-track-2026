# results/phase-a

The Kaggle runs behind Section 5 of the paper (Tables 1 and 3, Figure 3), one folder per round, holding only the files that
`bench/phase_a.py` reads: `kernel.log` (the Kaggle kernel log, a JSON list of stdout/stderr events), `summary.txt` and
`benchmark.json` written by the TAAF harness, and, for the runs of the public notebooks, the two modules the notebook wrote.
Everything else that a run produces (model transcripts, request logs, intermediate states, videos, HTML diagnostics) is not
released. `rounds.tsv` lists every round with the batch, the arm, the Kaggle version, whether the paper read it, and why not.

| batch | folders | in the paper |
|---|---|---|
| `second-batch/` | the four public notebooks run on 7 November 2026: `control-r1`…`r4`, `k-playing-r1`…`r4` (and the three attempts that did not start under the old input layout), `k-advising-r1`…`r3`, `last-resort-r1`…`r4`; and the Duck fork run the same day, `duck-fork-k-playing-r1`…`r3` | Table 3, Figure 3B, Section 10 |
| `first-batch/` | the ablation of 1–3 November on the same base: `base`, `advisor` (notes and pruning), `advice-text-only`, `k-playing` (two rounds) | Section 5, Figure 3A |
| `duck-line/` | phase A of the Duck line, v4 and v5 (v5 is the scored entry, ref. 57852519) | Table 1 |
| `v14-1/` | phase A of v14.1 (last resort with the 600 s window plus the cross-game memory) | Section 5 |

```bash
python3 bench/phase_a.py                 # validity of every round, Table 3 rows, Table 1 rows, batch summaries
python3 bench/phase_a.py --csv games.csv # the same, plus one row per round and game with every counter
python3 bench/phase_a.py /path/to/your/kaggle/output   # a run of your own
```

## Validity and reading

The conditions were fixed before the runs (Section 5): three games finished, no traceback, at least one level completed, and,
in the arms with the explorer, `layer_exc = k_exc = 0` in every game with at least one positive `cert_*`, `h8_masked_keys` or
`k_*` counter; the modules the kernel wrote must match the frozen digests. The script recomputes these from the files and
prints them next to the declared reading. An arm is read with three valid rounds of its batch against the control of the same
batch, and an effect is registered only if the difference of means exceeds twice the larger sample SD of the pair. Rounds
pushed after a batch closed (text-only re-pushes) are shipped and marked `read = no`.

## Sanitization of the logs

The kernel logs of the public notebooks are the downloaded files. The logs of the development kernels (first batch, Duck line,
v14.1) printed their tables and a few status lines with development names; those lines were translated to the names used in
this repository (table headers, `benchmark:` labels, module file names, the `[explorer]` and `[cross-game memory]` lines, and
the counters `hidden_death`, `k_cross_decisions`, `h5_cross_records`, `h5_cross_leader_changed`). No number changed, and every
distinct line of every log was reviewed before release.
