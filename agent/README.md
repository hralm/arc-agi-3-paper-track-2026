# agent

The agent code, as submitted in v14.1 (dfranzen line; submission ref. 57923098, scored after the deadline).
The two files are the frozen modules of that submission with comments and docstrings in English and internal
identifiers renamed to the names used in the public notebooks; the code is otherwise unchanged (abstract syntax tree
identical under the rename map).
SHA-256 digests of the frozen originals are in [`../manifests/frozen-artifacts.tsv`](../manifests/frozen-artifacts.tsv).
The Kaggle versions of the notebooks that ran the second batch kept a few development names for internal counters (for
example `morte_oculta`, published here and in [`../notebooks`](../notebooks) as `hidden_death`); the copies in this
repository use the names below throughout.

| file | contents | paper |
|---|---|---|
| `graph_explorer.py` | the explorer K: masked-hash states, the graph of tested actions, BFS to the nearest untested frontier, life budget; the goal hypotheses H1/H2 and the win-condition identification component H5 (families O1, O3, G1, G2, G4; credit in the winning frame; the cross-game store's tie-breakers) | §§3.1, 3.3, 3.4 |
| `hybrid_analyzer.py` | the coupling to the harness of the LLM agent: no-effect certificates and hidden-state detection (H8), the action-list pruning and prompt notes, the three modes (playing, advising, last resort), the cross-game store, the optional `k_map` and pace hooks, and the robustness layer | §§3.1, 3.2, 3.4 |

`graph_explorer.MyAgent` subclasses `agents.agent.Agent` from the official
[ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) repository and runs on its own in the local bench
([`../bench`](../bench)). `hybrid_analyzer.hybrid_factory(make_tool_agent, **options)` returns the
`analyzer_factory` that the TAAF `HarnessSolver` (Duck harness and dfranzen base) calls once per game; the options of
each submitted variant are in the kernel metadata and notebooks identified in the manifests.

Bench switches read from the module: `H5_ON` (K0 control) and `H5_FAMILY_ON` (per-family ablation; O1 is off by
default).
