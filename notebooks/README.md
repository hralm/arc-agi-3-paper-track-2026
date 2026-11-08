# notebooks

The five Kaggle notebooks of the paper, one folder each, so that a clone of this repository holds everything the paper
cites: the notebook (`.ipynb`, code and text, no outputs), its `kernel-metadata.json` (inputs, accelerator, settings) and the
two modules the notebook writes to `/kaggle/working/` before the run.

| folder | role in the paper | Kaggle |
|---|---|---|
| `what-exploration-can-ignore-duck-fork-k-playing/` | the clean copy of the code of the eligible entry (ref. 57852519): Duck fork, K playing (mode 1); Section 10 | [notebook](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-duck-fork-k-playing) |
| `what-exploration-can-ignore-dfranzen-control/` | the unmodified dfranzen base, the control of Table 3 | [notebook](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-control) |
| `what-exploration-can-ignore-dfranzen-k-playing/` | dfranzen base, K playing (mode 1) | [notebook](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-playing) |
| `what-exploration-can-ignore-dfranzen-k-advising/` | dfranzen base, K advising (mode 2) | [notebook](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-advising) |
| `what-exploration-can-ignore-dfranzen-last-resort/` | dfranzen base, K as last resort (mode 3) | [notebook](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-last-resort) |

## Names, digests and what ran

The copies here carry the identifier names of this repository (the ones in `agent/`). The versions that ran on Kaggle for the
second batch of Section 5 (Table 3) kept a few development names for internal counters, for example `morte_oculta`, published
here as `hidden_death`; the code is otherwise the same, statement for statement (abstract syntax tree identical under the
rename map, checked by the repository's verifier). The SHA-256 digests of the bytes that ran are in
[`../manifests/frozen-artifacts.tsv`](../manifests/frozen-artifacts.tsv); each folder's `SHA256SUMS` covers the copies here.
Versions with the names of this repository are planned for Kaggle after the Paper Track closes; they will be declared, like
every later version, outside the batch the paper read.

## Running one on Kaggle

Create a notebook from the `.ipynb`, attach the inputs listed in `kernel-metadata.json` (the dfranzen model and serving
datasets, the competition data; the Duck fork also needs the TAAF source bundle and the vLLM wheelhouse named in its
metadata), select the accelerator named there, and run with internet off. The run writes `summary.txt`, `benchmark.json`
and the kernel log; `python3 bench/phase_a.py <output folder>` reads them and prints the validity checks and the counters
(see [`../bench/results/phase-a/README.md`](../bench/results/phase-a/README.md)).
