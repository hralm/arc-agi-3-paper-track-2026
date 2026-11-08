"""Shared setup of the test suite (started with v1 of the Duck line).

The analyzer under test is `agent/hybrid_analyzer.py`, or the file named by KA_PATH (mutants; see mutants.py).
The harness is Tufa's HarnessSolver, OFFLINE, with no network and no vLLM; the model is a stub with the ToolAgent
interface.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
AGENT = REPO / "agent"
# The official starter clone (games in environment_files/, ARC-AGI-3-Agents in vendor/) and the TAAF source bundle
# (Kaggle dataset jeroencottaar/taaf-kaggle-source-share); see the README.
STARTER = Path(os.environ.get("ARC_STARTER") or (REPO.parent / "ARC-AGI-3-Kaggle-Starter")).resolve()
TAAF = Path(os.environ.get("TAAF_SOURCE") or (REPO.parent / "taaf-kaggle-source-share")).resolve()
BUNDLE = TAAF / "src"
for p in (BUNDLE / "tufa-arc-agi-framework/src", BUNDLE / "ARC3-Inference", STARTER, STARTER / "vendor/ARC-AGI-3-Agents"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
ENV_DIR = str(STARTER / "environment_files")
logging.basicConfig(level=logging.CRITICAL)

import arc_agi  # noqa: E402
import taaf.benchmark  # noqa: E402
import taaf.game_api  # noqa: E402
import inference.framework.solver as solver_mod  # noqa: E402
from inference.framework.solver import HarnessSolver  # noqa: E402
from inference.agent.tool_agent import AnalyzerTurnResult  # noqa: E402

# The harness rewrites the whole state file after every action (quadratic cost); the stub never reads it.
solver_mod.write_runtime_state = lambda *a, **k: None

KA_FILE = Path(os.environ.get("KA_PATH") or (AGENT / "hybrid_analyzer.py")).resolve()
HOOK_OPTS = dict(inject_hint=True, stall_steps=3, repeat_noop_limit=2, time_floor_s=600.0, level_share=0.6,
             selector="triggers", probe_actions=20, yield_window=40, cross_level=False, cross_level_turns=5,
             h8="list", refuse_limit=2)                      # = the hybrid hook of the Duck line (mode 1)


def load_ka():
    """Loads the analyzer under test as module `hybrid_analyzer` (once per process)."""
    if "hybrid_analyzer" in sys.modules and getattr(sys.modules["hybrid_analyzer"], "__file__", None) == str(KA_FILE):
        return sys.modules["hybrid_analyzer"]
    spec = importlib.util.spec_from_file_location("hybrid_analyzer", KA_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_analyzer"] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "_import_graph_explorer") and not hasattr(mod, "_import_graph_explorer"):
        mod._import_graph_explorer = mod._import_graph_explorer
    return mod


NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE", "ACTION6": "MOUSE"}


class Stub:
    """ToolAgent interface: `_build_user_prompt`, `analyze`, `total_tokens`, and `_timeout`. Acts like the real ToolAgent:
    step_env({"actions": [...]}) (tool_agent.py:1529). mode: random | stall | fail | mix | raise."""

    def __init__(self, seed, mode: str = "mix", timeout: float | None = 900.0) -> None:
        self.rng = random.Random(seed)
        self.mode = mode
        self.prompts: list[str] = []
        self.valid_seen: list[list[str]] = []
        self.total_tokens = 0
        self.generated_tokens = 0
        self._timeout = timeout
        self.n = 0

    def _build_user_prompt(self, action_num, **kw):
        return f"Step {action_num}\nline2\nrest of the prompt"

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        self.n += 1
        self.total_tokens += 10
        self.generated_tokens += 3
        self.prompts.append(self._build_user_prompt(action_num, valid_actions=valid_actions))
        self.valid_seen.append(list(valid_actions or []))
        mode = self.mode if self.mode != "mix" else self.rng.choice(["stall", "random", "random", "fail"])
        if mode == "raise":
            raise RuntimeError("stub: model raised")
        if mode == "fail":
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True)
        if mode == "stall":
            return AnalyzerTurnResult(step_executed=False, reasoning="thinking...")
        acts = [NAMES.get(a, a) for a in (valid_actions or []) if a not in ("RESET",)] or ["UP"]
        a = self.rng.choice(acts)
        item = {"action": a, "row": self.rng.randrange(64), "col": self.rng.randrange(64)} if a == "MOUSE" else {"action": a}
        p = step_env({"actions": [item]})
        return AnalyzerTurnResult(step_executed=bool(p and p.get("executed_count")), reasoning="stub")


def run_games(games, factory, max_actions=60, conc=2, max_runtime=600.0, passes=1):
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)
    solver = HarnessSolver(label="b1", model="none", analyzer_timeout=30.0, max_actions_per_game=max_actions,
                           max_runtime_s_per_game=max_runtime, concurrency=conc, start_local_server=False,
                           kaggle_enable_vllm=False)
    solver.analyzer_factory = factory
    bm = taaf.benchmark.Benchmark(label="b1", games=[taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in games],
                                  solver=solver, n_passes=passes)
    bm.job_dir = Path(tempfile.mkdtemp(prefix="b1_"))
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        asyncio.run(bm.run())
    return bm


def make_factory(ka, analyzers: dict, mode="mix", **opts):
    o = dict(HOOK_OPTS)
    o.update(opts)

    def make(game, index):
        gid = str(game.env_name).split("-")[0]
        a = ka.HybridAnalyzer(game, Stub(gid, mode), **o)
        analyzers.setdefault(gid, []).append(a)
        return a
    return make


def gid_of(run) -> str:
    return str(run.game_id).split("-")[0]


def main(tests) -> int:
    sel = sys.argv[1:]
    failures = 0
    for t in tests:
        if sel and t.__name__ not in sel:
            continue
        try:
            msg = t()
            print(f"PASS  {t.__name__}: {msg or ''}", flush=True)
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"[{KA_FILE}] failures={failures}")
    return 1 if failures else 0
