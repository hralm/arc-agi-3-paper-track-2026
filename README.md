English · [Português (Brasil)](README.pt-BR.md)

# Certifying No Effect and Identifying the Win Condition Under Per-Action Scoring in ARC-AGI-3

Code and materials for our submission to the [ARC Prize 2026 Paper Track](https://arcprize.org/competitions/arc-prize-2026) (ARC-AGI-3).

| | |
|---|---|
| Paper (PDF) | [`paper/`](paper/) (English, version of record; Brazilian Portuguese translation alongside) |
| Kaggle Writeup | public when the Paper Track closes (9 November 2026) |
| Public notebook (eligible entry, ref. 57852519) | [What Exploration Can Ignore: Duck fork, K playing](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-duck-fork-k-playing) |
| The three modes on the dfranzen base | [control](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-control), [K playing](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-playing), [K advising](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-advising), [last resort](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-last-resort) |
| Project page (how the work was done) | [docs/](docs/index.md) |

Everything the paper cites is in this repository: the paper, the five notebooks with their metadata and modules, the agent
code, the bench, the tests, the figures, and the files behind every phase-A round. A clone replaces the round of Kaggle pages.


## What the paper argues

ARC-AGI-3 asks agents to set goals without instructions, yet every game already fixes its win condition, and the
score measures how economically it is reached. What an agent can do, then, is identify the goal the game has set and
not choose one, without spending the actions the score charges. We built our agent to measure that step, and the
measure came back mostly negative: in 50 of 66 local wins, the explorer won without the identifier naming the goal.

The agent couples a graph-based explorer (K) to the harness of an open-weight large language model (LLM) agent. The
explorer issues revocable *no-effect certificates* for actions observed to change nothing at a state, detects hidden
state when a certificate fails or a safe action kills, and ranks goal hypotheses (H5) that only order candidate
actions. Because the LLM is the agent's default policy, we treat this layer as a measurement instrument and
vary how it is present: playing, advising, or acting only as a last resort. No mode produced a measurable
score gain in two phase-A batches, and the paper reports each comparison with what it can and cannot establish.

## Repository layout

| folder | contents | paper |
|---|---|---|
| `paper/` | the PDF (English, version of record) and the Brazilian Portuguese translation | — |
| `notebooks/` | the five Kaggle notebooks (code and text, no outputs), their kernel metadata and the modules they write, with the identifier names of this repository | §§5, 10, Table 3 |
| `agent/` | the explorer K, the no-effect certificates and hidden-state triggers (H8), the goal-hypothesis component (H5) and the cross-game memory | §§3.1–3.4 |
| `bench/` | the offline local bench (25 public games, 4 seeds, 1,500 steps) with the flags of the eight attribution arms, and the eight runs of Table 4; `results/phase-a/` holds the files behind every phase-A round (kernel log, summary, benchmark record) and `phase_a.py` recomputes the validity checks and Tables 1 and 3 from them, or from a new Kaggle run | §5, Tables 1, 3 and 4, Figures 1 and 2 |
| `tests/` | the test suite and the mutants of the adversarial verification protocol | §4 |
| `manifests/` | SHA-256 manifests of the frozen artifacts cited in the paper | §§5, 10 |
| `figures/` | the scripts that generate the four figures (Figures 1 and 2 are diagrams; Figure 3 has two panels, one per phase-A batch), and the figures | §§1, 3.2, 5 |
| `docs/` | a project page (GitHub Pages) with an account of how the work was done | — |

## Reproducing the results

Everything here runs on a CPU, offline, with no language model. We tested it on Linux with Python 3.14.7. The local
timings below come from a mini PC with an AMD Ryzen 5 5500U (six cores), 32 GB of RAM, and no discrete GPU. Run the
commands from the root of this repository.

### 1. Environment

Clone the official starter and the agents repository at the commits we used, and create the environment:

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
git -C ARC-AGI-3-Kaggle-Starter checkout eeb1535404f321d280a8f9194bbc1d7aca5f05fc
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents
git -C ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents checkout 4743e7d0aaae0ded0d98a89a7e282e63564cd58b
python3.14 -m venv ARC-AGI-3-Kaggle-Starter/.venv
ARC-AGI-3-Kaggle-Starter/.venv/bin/pip install --no-deps -r requirements.txt
```

### 2. Games

Download the 25 public games once into `ARC-AGI-3-Kaggle-Starter/environment_files/` with the toolkit: create
`arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir="environment_files")` and call `make()` on each game
listed by `get_environments()`. A free [ARC API key](https://arcprize.org/) gives the full public set. Our runs used the
versions downloaded on 30 September 2026 (for example `ls20-9607627b` and `vc33-5430563c`).

### 3. Table 4 and Figure 4

Recompute them from the shipped runs. It takes seconds:

```bash
python3 bench/table4.py
ARC-AGI-3-Kaggle-Starter/.venv/bin/python figures/make_figures.py   # writes the two PDFs here
```

To rerun an arm (13 to 37 minutes each), pick one of the eight arms in [`bench/README.md`](bench/README.md):

```bash
cd ARC-AGI-3-Kaggle-Starter
BENCH_ARM=default .venv/bin/python ../bench/bench.py --agents k --max-steps 1500 --seeds 4 --quiet
```

### 4. Tests and mutants

The tests also need the TAAF source bundle that the Duck harness and the dfranzen base run on. Download the public Kaggle
dataset [`jeroencottaar/taaf-kaggle-source-share`](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share)
(bundle created 12 June 2026, with `tufa-arc-agi-framework` at `fe9f7c4` and `ARC3-Inference` at `aa69123`), unpack it so
that `src/` is inside it, and run:

```bash
export ARC_STARTER=$PWD/ARC-AGI-3-Kaggle-Starter TAAF_SOURCE=/path/to/taaf-kaggle-source-share
for f in tests/test_*.py; do ARC-AGI-3-Kaggle-Starter/.venv/bin/python "$f" || echo "FAILED: $f"; done   # ~5 min, 92 tests
ARC-AGI-3-Kaggle-Starter/.venv/bin/python tests/mutants.py                                              # ~40 min, 85 mutants
```

### 5. Integrity

Run `sha256sum -c manifests/SHA256SUMS` to check this repository. The digests of the submitted notebooks and modules are in
[`manifests/frozen-artifacts.tsv`](manifests/frozen-artifacts.tsv); each folder of [`notebooks/`](notebooks/) has its own
`SHA256SUMS` for the copies kept here.

### 6. Phase A on Kaggle

The rounds of Table 3 (second batch, 7 November 2026) were produced by the four public notebooks
listed above, run on Kaggle with the same frozen code, three rounds per mode and four for the control. The files from which
every number was read are in [`bench/results/phase-a/`](bench/results/phase-a/), one folder per round, with the first batch, the
Duck-line rounds of Table 1 and the v14.1 run alongside; `rounds.tsv` says which rounds the paper read. Kaggle mounted the
notebooks' inputs under two different layouts for different kernels of this set. The notebooks resolve both and record which one
they found; the three attempts that did not start under the old layout are shipped too.

```bash
python3 bench/phase_a.py                 # validity of every round, Table 3, Table 1, batch summaries (seconds)
```

To run a notebook yourself, copy it from [`notebooks/`](notebooks/) to Kaggle with the inputs and accelerator named in its
`kernel-metadata.json`, download the output (`kaggle kernels output <user>/<slug> -p <folder>`) and run
`python3 bench/phase_a.py <folder>`: it prints the same validity checks and counters for your run.

## Built on

- Tufa Labs' Duck harness (Bessis et al., 2026), MIT license, from the [TAAF source bundle](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share).
- The dfranzen ARC-AGI-3 Milestone 2 notebook (Franzen, 2026), Apache-2.0 license.
- The ARC-AGI-3 toolkit and competition materials (ARC Prize Foundation, 2026): the [starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) and the [agents repository](https://github.com/arcprize/ARC-AGI-3-Agents).

Full references are in the paper.

## How this work was done

A longer account, with the time it took, what went wrong, and what we learned, is on the [project page](docs/index.md).
This work was carried out with large language model agents from different providers, used deliberately under a fixed
protocol: pre-registration, a blue team before every red team, red teams by other models, regression tests and
mutants for every defect, and full reading of a source before citing it. Every output was adjudicated by the author,
who ran every Kaggle experiment and takes responsibility for the content. No language model is an author. The full
statement is in the paper.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 (see [`LICENSE`](LICENSE)). Third-party components keep their licenses, listed in [`NOTICE`](NOTICE).
