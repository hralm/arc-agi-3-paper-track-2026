# Identifying, Not Choosing, Goals: Win-Condition Identification, No-Effect Certificates and the Limits of Exploration in an ARC-AGI-3 Agent

*Measuring what exploration can ignore: no-effect certificates, a goal identifier that mostly missed, and three modes for an explorer.*

Code and materials for our submission to the **ARC Prize 2026 Paper Track** (ARC-AGI-3).

| | |
|---|---|
| Paper (PDF) | *link added at submission* |
| Kaggle Writeup | *public when the Paper Track closes (9 November 2026)* |
| Public notebook (eligible entry, ref. 57852519) | *link added at submission* |
| Reproducibility package (Kaggle Dataset) | *link added at submission* |

> **Status.** This repository is under construction. The structure and the paper's summary are in place; the code,
> the bench, the tests and the manifests are on their way. Folders marked *pending* below are ready to receive them.

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
| `agent/` | *pending:* the explorer K, the no-effect certificates and hidden-state triggers (H8), the goal-hypothesis component (H5) and the cross-game memory | §§3.1–3.5 |
| `bench/` | *pending:* the offline local bench (25 public games, 4 seeds, 1,500 steps) with the flags of the eight attribution arms | §5, Table 3, Figure 2 |
| `tests/` | *pending:* the test suite and the mutants of the adversarial verification protocol | §4 |
| `manifests/` | *pending:* SHA-256 manifests of the frozen artifacts cited in the paper | §§5, 10 |
| `figures/` | *pending:* the scripts that generate Figures 1 and 2 | §5 |
| `docs/` | a project page (GitHub Pages) with an account of how the work was done | — |

## Reproducing the results

*Pending, with the code.* This section will give the exact commands, pinned versions and expected run times to
reproduce Table 3 and Figure 2 on the local bench, and to run the tests and the mutants.

The games are not redistributed here; they come from the official ARC-AGI-3 toolkit. Model weights are not
redistributed either: the notebooks load them from public Kaggle resources (Qwen3.6-27B-FP8 for the Duck line; the
dfranzen base's own model for the later versions; see the paper, §3.1).

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
