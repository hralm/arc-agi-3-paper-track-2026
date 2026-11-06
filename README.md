# Identifying, Not Choosing, Goals: Win-Condition Identification, No-Effect Certificates and the Limits of Exploration in an ARC-AGI-3 Agent

*Measuring what exploration can ignore: no-effect certificates, a goal identifier that mostly missed, and three modes for an explorer.*

Code and materials for our submission to the **ARC Prize 2026 Paper Track** (ARC-AGI-3).

| | |
|---|---|
| Paper (PDF) | *link added at submission* |
| Kaggle Writeup | *public when the Paper Track closes (9 November 2026)* |
| Public notebook (eligible entry, ref. 57852519) | *link added at submission* |
| Reproducibility package (Kaggle Dataset) | *link added at submission* |


## What the paper argues

ARC-AGI-3 asks agents to set goals without instructions, yet every game already fixes its own win condition, and the
score measures how economically it is reached. What an agent can do, then, is not choose a goal but identify the one
the game has set, without spending the actions the score charges. We built our agent to measure that step, and the
measure came back mostly negative: in 50 of 66 local wins, the explorer won without the identifier naming the goal.

The agent couples a graph-based explorer (K) to the harness of an open-weight large language model (LLM) agent. The
explorer issues revocable **no-effect certificates** for actions observed to change nothing at a state, detects
**hidden state** when a certificate fails or a safe action kills, and ranks **goal hypotheses** (H5) that only order
candidate actions. Because the LLM is the agent's default policy, we treat this layer as a measurement instrument and
vary how it is present: **playing**, **advising**, or acting only as a **last resort**. No mode produced a measurable
score gain; the paper reports each comparison with what it can and cannot establish.

## Repository layout

| folder | contents | paper |
|---|---|---|
| `paper/` | the PDF | — |
| `agent/` | the explorer K, the no-effect certificates and hidden-state triggers (H8), the goal-hypothesis component (H5) and the cross-game memory | §§3.1–3.4 |
| `bench/` | the offline local bench (25 public games, 4 seeds, 1,500 steps) with the flags of the eight attribution arms, and the eight runs of Table 3 | §5, Table 3, Figure 2 |
| `tests/` | the test suite and the mutants of the adversarial verification protocol | §4 |
| `manifests/` | SHA-256 manifests of the frozen artifacts cited in the paper | §§5, 10 |
| `figures/` | the script that generates Figures 1 and 2, and the figures | §5 |
| `docs/` | a project page (GitHub Pages) with an account of how the work was done | — |

## Reproducing the results

Everything here runs on a CPU, offline, with no language model. Tested on Linux with Python 3.14.7. The commands run
from the root of this repository.

**1. Environment.** The games and the agent base class come from the official starter, pinned to the commits we used:

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
git -C ARC-AGI-3-Kaggle-Starter checkout eeb1535404f321d280a8f9194bbc1d7aca5f05fc
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents
git -C ARC-AGI-3-Kaggle-Starter/vendor/ARC-AGI-3-Agents checkout 4743e7d0aaae0ded0d98a89a7e282e63564cd58b
python3.14 -m venv ARC-AGI-3-Kaggle-Starter/.venv
ARC-AGI-3-Kaggle-Starter/.venv/bin/pip install --no-deps -r requirements.txt
```

**2. Games.** Download the 25 public games once into `ARC-AGI-3-Kaggle-Starter/environment_files/` with the toolkit
(`arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir="environment_files")`, then `make()` each game
listed by `get_environments()`; a free ARC API key gives the full public set). Our runs used the versions downloaded
on 30 September 2026 (for example `ls20-9607627b`, `vc33-5430563c`).

**3. Table 3 and Figure 2.** From the shipped runs, in seconds:

```bash
python3 bench/table3.py
ARC-AGI-3-Kaggle-Starter/.venv/bin/python figures/make_figures.py   # writes the two PDFs here
```

To re-run an arm (13–37 minutes each; see [`bench/README.md`](bench/README.md) for the eight arms):

```bash
cd ARC-AGI-3-Kaggle-Starter
BENCH_ARM=default .venv/bin/python ../bench/bench.py --agents k --max-steps 1500 --seeds 4 --quiet
```

**4. Tests and mutants.** The tests also need the TAAF source bundle that the Duck harness and the dfranzen base run
on: the public Kaggle dataset [`jeroencottaar/taaf-kaggle-source-share`](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share)
(bundle created 12 June 2026; `tufa-arc-agi-framework` at `fe9f7c4`, `ARC3-Inference` at `aa69123`). Unpack it so that
`src/` is inside it, then:

```bash
export ARC_STARTER=$PWD/ARC-AGI-3-Kaggle-Starter TAAF_SOURCE=/path/to/taaf-kaggle-source-share
for f in tests/test_*.py; do ARC-AGI-3-Kaggle-Starter/.venv/bin/python "$f" || echo "FAILED: $f"; done   # ~5 min, 92 tests
ARC-AGI-3-Kaggle-Starter/.venv/bin/python tests/mutants.py                                              # ~40 min, 85 mutants
```

**5. Integrity.** `sha256sum -c manifests/SHA256SUMS` checks this repository; `manifests/frozen-artifacts.tsv` gives the
digests of the submitted notebooks and modules.

## Built on

- Tufa Labs' Duck harness (Bessis et al., 2026), MIT license.
- The dfranzen ARC-AGI-3 Milestone 2 notebook (Franzen, 2026), Apache-2.0 license.
- The ARC-AGI-3 toolkit and competition materials (ARC Prize Foundation, 2026).

Full references are in the paper.

## How this work was done

This work was carried out with large language model agents from different providers, used deliberately under a fixed
protocol: pre-registration, a blue team before every red team, red teams by other models, regression tests and
mutants for every defect, and full reading of a source before citing it. Every output was adjudicated by the author,
who ran every Kaggle experiment and takes responsibility for the content; no language model is an author. The full
statement is in the paper.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 (see [`LICENSE`](LICENSE)). Third-party components keep their own licenses; see [`NOTICE`](NOTICE).
