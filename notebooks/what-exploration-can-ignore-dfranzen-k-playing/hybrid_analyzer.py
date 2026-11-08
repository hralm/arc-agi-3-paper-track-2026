"""Hybrid Analyzer: integrates graph explorer K into the dfranzen Milestone 2 / TAAF harness.

Provides the analyzer adapter for HarnessSolver via `analyzer_factory(game, index)`:
- KPolicy: adapts graph_explorer.py (Agent subclass) to the TAAF harness interface:
  builds FrameData from raw observations, queries graph explorer K, and translates actions
  to harness step_env dictionaries. Also observes third-party model actions to maintain the state graph.
- HybridAnalyzer: manages policy arbitration between the base model and K:
  evaluates certificates, masks keys certified to have no effect from this state,
  injects informative guidance into prompts, and invokes K when stall, failure,
  repetition, or time pressure triggers fire.
- Robustness layer: wraps step_env and exceptions to prevent analyzer failures from terminating
  evaluation runs, falling back gracefully to the original model.
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

import arcengine
from arcengine import FrameData, GameAction, GameState

# ---------------------------------------------------------------------------------------------
# Locate graph_explorer.py and ARC-AGI-3-Agents shim
# ---------------------------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent


def _candidates() -> list[Path]:
    """Resolved per KPolicy instance, not on import."""
    return ([Path(os.environ["EXPLORER_MODULE_PATH"])] if os.environ.get("EXPLORER_MODULE_PATH") else []) + [
        _HERE.parent / "agents" / "graph_explorer.py",
        _HERE / "graph_explorer.py",
        Path("/kaggle/working/graph_explorer.py"),
    ]


def _ensure_agents_shim() -> None:
    """Shim providing the minimal Agent base interface when ARC-AGI-3-Agents is not in sys.path."""
    try:
        import agents.agent  # noqa: F401
        return
    except Exception:
        pass
    import types

    pkg = types.ModuleType("agents")
    mod = types.ModuleType("agents.agent")

    class Agent:  # minimal Agent interface compatible with vendor agents/agent.py
        MAX_ACTIONS: int = 80
        action_counter: int = 0

        def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None) -> None:
            self.card_id = card_id
            self.game_id = game_id
            self.agent_name = agent_name
            self.ROOT_URL = ROOT_URL
            self.arc_env = arc_env
            self.tags = tags or []
            self.guid = ""
            self.frames = [FrameData(levels_completed=0)]

        @property
        def name(self) -> str:
            return f"{self.game_id}.{self.__class__.__name__.lower()}"

    mod.Agent = Agent
    pkg.agent = mod
    sys.modules["agents"] = pkg
    sys.modules["agents.agent"] = mod


def _import_graph_explorer():
    _ensure_agents_shim()
    cands = _candidates()
    for p in cands:
        if p.exists():
            spec = importlib.util.spec_from_file_location("graph_explorer", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("graph_explorer.py not found: " + ", ".join(str(p) for p in cands))


try:
    from inference.agent.tool_agent import AnalyzerTurnResult  # type: ignore
except Exception:  # standalone testing fallback replicating FrameView dataclass
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class AnalyzerTurnResult:  # type: ignore[no-redef]
        step_executed: bool
        retryable_failure: bool = False
        reasoning: str = ""
        yielded_control: bool = False


_MODEL_NAME = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE", 7: "UNDO", 0: "RESET"}
_ENGINE_ID = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7, "RESET": 0,
              "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "MOUSE": 6, "UNDO": 7}
# Unsupported actions returning None without spending turn are marked invalid (mark_invalid) to prevent repeated proposals.
_NOOP, _CHANGE, _PROGRESS, _DEATH = 0, 1, 2, 3      # mirrors graph_explorer transition outcomes
HINT_MAX_CHARS = 700                                 # maximum prompt hint character budget


def _fmt_act(c: tuple) -> str:
    return _MODEL_NAME.get(c[0], str(c[0])) + (f"(row={c[2]},col={c[1]})" if c[0] == 6 else "")


def _first_action(arguments: dict[str, Any]) -> tuple | None:
    """Tuple representing the first action from a step_env request dictionary."""
    d = arguments if arguments.get("action") else ((arguments.get("actions") or [None])[0] or {})
    aid = _ENGINE_ID.get(str(d.get("action", "")).strip().upper())
    if aid is None or aid == 0:
        return None
    if aid == 6:
        return (6, int(d.get("col", 0)), int(d.get("row", 0)))
    return (aid,)


def _cap_lines(lines: list[str], max_chars: int = HINT_MAX_CHARS) -> list[str]:
    out, used = [], 0
    for ln in lines:
        if used + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        used += len(ln) + 1
    return out


def _payload_ok(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload) and not payload.get("error") and "executed_count" in payload


def _legal_fallback(game: Any, rng: random.Random) -> dict[str, Any] | None:
    """Fallback action sampled from current legal actions when an exception occurs."""
    try:
        ids = [int(a) for a in game.current_state.available_actions if int(a) in (1, 2, 3, 4, 5, 6)]
    except Exception:
        ids = []
    if not ids:
        return None
    aid = rng.choice(ids)
    if aid == 6:
        return {"action": "MOUSE", "row": rng.randrange(64), "col": rng.randrange(64)}
    return {"action": _MODEL_NAME[aid]}


def _untried_legal(node: Any) -> list:
    """Untried candidate actions that were legal on most recent visit."""
    legal = getattr(node, "legal", frozenset())
    return [c for c in node.cands if c not in node.tried and (not legal or c[0] in legal)]


def _game_key(game: Any, default: Any = "game") -> str:
    return str(getattr(game, "env_name", None) or getattr(game, "game_id", None) or default).split("-")[0]


class _DummyEnv:
    """Minimal environment placeholder for Agent base class initialization."""
    observation_space = None


class KPolicy:
    """Policy adapter integrating graph explorer into the harness solver interface."""

    def __init__(self, game: Any, max_actions: int = 10 ** 9) -> None:
        mod = _import_graph_explorer()
        self.cls = mod.MyAgent
        game_id = str(getattr(game, "env_name", None) or getattr(game, "game_id", "game")).split("-")[0]
        self.agent = self.cls(card_id="hybrid", game_id=game_id, agent_name=f"k.{game_id}",
                              ROOT_URL="http://localhost", record=False, arc_env=_DummyEnv(), tags=["hybrid"])
        self.agent.MAX_ACTIONS = max_actions
        self.frames: list[FrameData] = [FrameData(levels_completed=0)]
        self.game = game
        self.last_proposal: tuple | None = None
        self.hidden_levels: set[int] = set()           # shared set of levels detected to have hidden state

    # ---- visual observation ----
    def frame_from_state(self, state: Any) -> FrameData:
        raw = state.raw
        frames = [np.asarray(f).tolist() for f in raw.frame] if raw.frame is not None and len(raw.frame) else []
        return FrameData(game_id=str(getattr(raw, "game_id", "") or ""), frame=frames, state=raw.state,
                         levels_completed=int(raw.levels_completed), win_levels=int(getattr(raw, "win_levels", 0) or 0),
                         available_actions=[int(a) for a in raw.available_actions])

    def current_frame(self) -> FrameData:
        return self.frame_from_state(self.game.current_state)

    # ---- observation of external actions ----
    def observe(self, engine_action: str, data: dict | None, frame_after: FrameData) -> None:
        """Informs explorer that an engine action was executed and observes resulting frame."""
        aid = _ENGINE_ID.get(str(engine_action).upper())
        if aid is None:
            return
        if aid == 0:
            # RESET (auto or from model): explorer clears step memory
            self.agent.prev_key = self.agent.prev_act = None
            self.agent.prev_grid = None
            self.agent.plan.clear()
            self.agent.hud_reset = True
            self.agent.life = 0
            return
        if self.agent.prev_key is None:
            return
        if aid == 6 and data:
            self.agent.prev_act = (6, int(data.get("x", 0)), int(data.get("y", 0)))
        else:
            self.agent.prev_act = (aid,)

    # ---- policy proposal ----
    def propose(self, frame: FrameData | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Returns step_env arguments or None for reset, accompanied by diagnostic reason."""
        frame = frame or self.current_frame()
        self.frames.append(frame)
        if len(self.frames) > 4:
            del self.frames[:-4]
        act = self.agent.choose_action(self.frames, frame)
        reasoning = getattr(act, "reasoning", None)
        if act is GameAction.RESET or act.name == "RESET":
            self.last_proposal = (0,)
            return {"action": "RESET"}, {"k": "reset", "why": reasoning}
        aid = int(act.value[0]) if isinstance(act.value, tuple) else int(act.value)
        if aid == 6:
            # Click coordinates are retrieved from instance prev_act to prevent cross-thread interference.
            pa = self.agent.prev_act
            if isinstance(pa, tuple) and len(pa) == 3 and pa[0] == 6:
                x, y = int(pa[1]), int(pa[2])
            else:
                d = getattr(self.agent, "_click", None) or (act.action_data.model_dump() if hasattr(act, "action_data") else {})
                x, y = int(d.get("x", 0)), int(d.get("y", 0))
            x, y = max(0, min(63, x)), max(0, min(63, y))
            self.last_proposal = (6, x, y)
            return {"action": "MOUSE", "row": y, "col": x}, {"k": reasoning}
        self.last_proposal = (aid,)
        return {"action": _MODEL_NAME.get(aid, f"ACTION{aid}")}, {"k": reasoning}

    def absorb(self, frame: FrameData) -> None:
        """Processes observed frame into state graph without executing explorer proposal. Level transition resets life to zero."""
        ag = self.agent
        life, max_life, hud, steps, lv0 = ag.life, ag.max_life, ag.hud_reset, ag.steps, ag.prev_levels
        args, _ = self.propose(frame)
        new_level = int(frame.levels_completed) > lv0
        if args.get("action") != "RESET" or frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            if new_level:
                ag.life = 0
            return
        grid = frame.frame[-1] if frame.frame else []
        if not grid:
            return
        ag.prev_grid = grid
        ag.prev_key = ag._key_of(grid, int(frame.levels_completed))
        ag.prev_act = None
        ag.life = 0 if new_level else life + 1
        ag.max_life = max(max_life, ag.life)
        ag.hud_reset = hud if ag.steps == steps else False
        self.last_proposal = None

    def mark_invalid(self) -> None:
        """Marks last proposed action as invalid/tested when rejected by environment."""
        ag = self.agent
        node = ag.nodes.get(ag.prev_key) if ag.prev_key is not None else None
        if node is not None and self.last_proposal and self.last_proposal[0] != 0:
            node.tried[self.last_proposal] = 0
            getattr(node, "unproven", set()).discard(self.last_proposal)
        ag.prev_key = ag.prev_act = None

    # ---- cross-level persistence ----
    def cross_level_lines(self, max_items: int = 6) -> list[str]:
        """Returns persistent findings across level transitions."""
        ag = self.agent
        lines = ["Cross-level notes from the explorer (persist across levels; informational only):"]
        react = sorted(((ag.cls_tries[k], k) for k in ag.reactive), reverse=True)[:max_items]
        if react:
            lines.append("- component classes that REACTED to a click earlier in this game (color,size,w,h; times tried): "
                         + ", ".join(f"{k}×{n}" for n, k in react))
        dead = sorted(((n, k) for k, n in ag.cls_tries.items() if k not in ag.reactive and n >= 3), reverse=True)[:max_items]
        if dead:
            lines.append("- component classes clicked ≥3 times with NO effect so far: " + ", ".join(f"{k}×{n}" for n, k in dead))
        dirs = []
        for a in (1, 2, 3, 4):
            d = self.agent._dir_of(a) if hasattr(self.agent, "_dir_of") else None
            if d:
                dirs.append(f"{_MODEL_NAME[a]} moves the mover by {d}")
        if dirs:
            lines.append("- learned key effects (dx,dy sign): " + "; ".join(dirs))
        if ag.life_budget:
            lines.append(f"- a life/energy budget of about {ag.life_budget} actions per attempt was detected (game over when exhausted)")
        per_level = {}
        for n in ag.nodes.values():
            per_level[n.level] = per_level.get(n.level, 0) + 1
        if per_level:
            lines.append("- distinct states explored per level so far: " + ", ".join(f"L{l + 1}:{c}" for l, c in sorted(per_level.items())))
        return lines if len(lines) > 1 else []

    # ---- Top win-condition hypothesis guidance (advisory hint) ----
    def h5_lines(self) -> list[str]:
        h5 = getattr(self.agent, "h5", None)
        if h5 is not None:
            h5.top()
        key = getattr(h5, "top_key", None) if h5 is not None else None
        h = h5.hyps.get(key) if key is not None else None
        if h is None or h.state not in ("active", "corroborated"):
            return []
        if h.fam == "O1":
            if isinstance(h.bind, tuple):
                txt = f"move an object with value {h.bind[0]} onto an object with value {h.bind[1]}"
            else:
                txt = f"move the object you control onto the object drawn with value {h.bind}"
        elif h.fam == "O3":
            txt = f"make every cell with value {h.bind} disappear"
        elif h.fam == "G1":
            x0, y0, bw, bh, rx, ry = h.bind
            txt = (f"make the {bw}x{bh} region at column {x0}, row {y0} identical to the reference region at column {rx}, "
                   f"row {ry}")
        elif h.fam == "G2":
            txt = f"place every object with value {h.bind[0]} inside a closed object with value {h.bind[1]}"
        else:
            txt = "make the picture mirror-symmetric " + ("left-right" if h.bind == "v" else "top-bottom")
        prog = f"; progress measure now {h.best} (lower is better)" if h.best is not None else ""
        return ["Win-condition hypothesis from the explorer (an unverified guess that may be wrong; it never forbids any "
                "action):", f"- {txt}{prog}"]

    # ---- prompt guidance ----
    def hint_lines(self, max_items: int = 8) -> list[str]:
        ag = self.agent
        key = ag.prev_key
        node = ag.nodes.get(key) if key is not None else None
        lines = ["Explorer notes (graph-based, deterministic; informational only):"]
        if node is None:
            lines.append("- no graph knowledge for this exact state yet")
            return lines
        noop = [c for c, r in node.tried.items() if r == 0]
        untried = _untried_legal(node)                   # exclude actions outside current availability
        fmt = lambda c: _MODEL_NAME[c[0]] + (f"(row={c[2]},col={c[1]})" if c[0] == 6 else "")
        # unproven noops from reindexing merge or hidden state do not constitute proof
        inherited_noops = getattr(node, "unproven", set())
        if node.level in self.hidden_levels:
            downgraded_noops, downgrade_reason = noop, "this level has hidden state"
        else:
            downgraded_noops, downgrade_reason = [c for c in noop if c in inherited_noops], "this state was merged from two different screens"
        proven_noops = [c for c in noop if c not in downgraded_noops]
        if downgraded_noops:
            lines.append(f"- actions that changed nothing here before, but {downgrade_reason}, so that is NOT a proof: "
                         f"{', '.join(fmt(c) for c in downgraded_noops[:max_items])}")
        if proven_noops:
            lines.append(f"- actions already proven to change NOTHING from this exact state: {', '.join(fmt(c) for c in proven_noops[:max_items])}")
        if untried:
            lines.append(f"- actions not yet tried from this exact state: {', '.join(fmt(c) for c in untried[:max_items])}" + (" ..." if len(untried) > max_items else ""))
        lvl = ag.prev_levels
        n_states = sum(1 for n in ag.nodes.values() if n.level == lvl)
        lines.append(f"- distinct states seen on this level: {n_states}; actions this life: {ag.life}" + (f" of a budget of {ag.life_budget}" if ag.life_budget else ""))
        return lines


# ---------------------------------------------------------------------------------------------
# Autonomous Graph Explorer Mode
# ---------------------------------------------------------------------------------------------
class KAnalyzer:
    label = "k-analyzer"

    def __init__(self, game: Any, max_actions: int = 10 ** 9) -> None:
        self.policy = KPolicy(game, max_actions=max_actions)
        self.game = game
        self.total_tokens = 0          # harness tracks model tokens; explorer consumes zero LLM tokens
        self.steps = 0
        self.stats: dict[str, int] = {"layer_exc": 0, "k_exc": 0}
        self.last_exc = ""
        self._rng = random.Random(zlib.crc32(_game_key(game).encode()))

    def analyze(self, state_path: Path, action_num: int, valid_actions=None, step_env: Callable | None = None,
                transcript_path: Path | None = None, analysis_step: int | None = None, transcript_updated=None,
                request_timeout_seconds=None, should_stop: Callable[[], bool] | None = None, **_: Any):
        if step_env is None:
            return AnalyzerTurnResult(step_executed=False, reasoning="no step_env")
        if should_stop is not None and should_stop():
            return AnalyzerTurnResult(step_executed=False, reasoning="stopped")
        try:                                             # errors trigger legal fallback action, logged and handled gracefully
            args, diag = self.policy.propose()
        except Exception as e:
            self.stats["k_exc"] += 1
            self.last_exc = f"{type(e).__name__}: {e}"[:200]
            try:
                self.policy.agent.prev_key = self.policy.agent.prev_act = None
            except Exception:
                pass
            self.policy.last_proposal = None
            args, diag = _legal_fallback(self.game, self._rng), {"k": "exc"}
            if args is None:
                return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning="k_exc without legal action")
        payload = step_env(args)
        self.steps += 1
        ok = _payload_ok(payload)
        if transcript_path is not None:
            try:
                with open(transcript_path, "a", encoding="utf-8") as f:
                    f.write(f"[k-analyzer step {analysis_step}] {args} -> ok={ok} {str(diag)[:120]}\n")
            except Exception:
                pass
        if not ok:
            # invalid action rejected by environment: mark tested and yield without consuming step
            try:
                self.policy.mark_invalid()
            except Exception as e:
                self.stats["layer_exc"] += 1
                self.last_exc = f"{type(e).__name__}: {e}"[:200]
        return AnalyzerTurnResult(step_executed=ok, reasoning=str(diag)[:200])


# ---------------------------------------------------------------------------------------------
# Hybrid Analyzer: model by default, explorer takes over on failure, stall, repetition or timeout
# ---------------------------------------------------------------------------------------------
class HybridAnalyzer:
    label = "hybrid-duck-k"

    def __init__(self, game: Any, tool_agent: Any, *, stall_steps: int = 3, repeat_noop_limit: int = 2,
                 time_floor_s: float = 600.0, level_share: float = 0.6, max_runtime_s: float | None = None,
                 inject_hint: bool = True, k_burst: int = 1, selector: str = "triggers", probe_actions: int = 20,
                 yield_window: int = 40, cross_level: bool = False, cross_level_turns: int = 5,
                 h8: str | None = None, refuse_limit: int = 2, h5_hint: bool = False) -> None:
        # selector: "triggers" (fixed stall/fail triggers) | "yield" (yield-based arbitration per level)
        # h8: None (accounting only) | "list" (action masking + predicted outcome) | "refusal" (masking + limited refusal)
        assert h8 in (None, "list", "refusal"), h8
        self.h8 = h8
        self.h5_hint = h5_hint                       # inject win-condition hypothesis into prompt
        self.refuse_limit = refuse_limit
        self.certs: dict[tuple, int] = {}            # (node_key, action) -> count of observed self-loop transitions
        self.refusals: dict[tuple, int] = {}
        self.hidden_levels: set[int] = set()         # levels where a certified non-effect was falsified
        self._reindex_seen = 0
        self._reopened_seen: dict[int, int] = {}
        self._last_valid: list[str] = []
        self.max_hint_chars = 0
        self.selector = selector
        self.probe_actions = probe_actions
        self.yield_window = yield_window
        self.cross_level = cross_level
        self.cross_level_turns = cross_level_turns
        self.outcomes: dict[str, list[tuple[int, int]]] = {"model": [], "k": []}   # (effective?, progress?) per action
        self.turns_in_level = 0
        self.level_of_turns = 0
        self.game = game
        self.model = tool_agent
        self.policy = KPolicy(game)
        self.policy.hidden_levels = self.hidden_levels   # shared reference to hidden state levels
        self._hidden_death_seen = 0                   # hidden deaths already processed
        self.stall_steps = stall_steps
        self.repeat_noop_limit = repeat_noop_limit
        self.time_floor_s = time_floor_s
        self.level_share = level_share
        self.max_runtime_s = max_runtime_s
        self.inject_hint = inject_hint
        self.k_burst = k_burst
        self.t0 = time.monotonic()
        self.level_t0 = time.monotonic()
        self.level_seen = 0
        self.no_step = 0
        self.noop_repeats = 0
        self.k_turns = 0
        self.model_turns = 0
        self.stats: dict[str, int] = {"k_failure": 0, "k_stall": 0, "k_repeat": 0, "k_time": 0, "k_yield": 0, "model_yield": 0,
                                      # non-effect certification tracker (always active)
                                      "cert_emitted": 0, "cert_falsified": 0, "cert_hidden_death": 0, "cert_revoked_reindex": 0, "cert_revoked_reopen": 0,
                                      "model_act_total": 0, "model_act_certified": 0, "model_batches": 0, "model_deaths": 0,
                                      "h8_masked_keys": 0, "h8_refused": 0,
                                      # Layer and model exceptions counted and tracked
                                      "layer_exc": 0, "k_exc": 0, "model_exc": 0, "fallback_act": 0}
        self.last_exc = ""
        self._rng = random.Random(zlib.crc32(_game_key(game).encode()))
        self._executed = 0                               # actions executed via step_env on this instance
        self._patch_prompt()

    def _note_exc(self, key: str, e: BaseException) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1
        self.last_exc = f"{key}: {type(e).__name__}: {e}"[:200]

    # Forward solver timeout and token count attributes
    @property
    def _timeout(self):
        return getattr(self.model, "_timeout", None)

    @property
    def generated_tokens(self) -> int:
        try:
            m = self.model
            v = getattr(m, "generated_tokens", None) if hasattr(m, "generated_tokens") else getattr(m, "total_tokens", 0)
            return max(0, int(v or 0))
        except Exception:
            return 0

    # ---- Non-effect certificates ----
    def _drop_level(self, level: int) -> int:
        nodes = self.policy.agent.nodes
        gone = [c for c in self.certs if (nodes.get(c[0]) is None or nodes[c[0]].level == level)]
        for c in gone:
            self.certs.pop(c, None)
            self.refusals.pop(c, None)
        return len(gone)

    def _cert_sync(self, before_key, before_act, batch_n: int = 1, died: bool = False) -> None:
        """Updates non-effect certificates after frame processing: emits, confirms, falsifies, or revokes."""
        ag = self.policy.agent
        ri = int(ag.stats.get("reindex", 0))
        if ri != self._reindex_seen:                     # HUD mask changed: node keys reindexed, revoke obsolete certificates
            self._reindex_seen = ri
            self.stats["cert_revoked_reindex"] += len(self.certs)
            self.certs.clear()
            self.refusals.clear()
        for lvl, n in list(ag.reopened.items()):         # explorer reopened tested actions: potential hidden state, revoke certificates
            if n != self._reopened_seen.get(lvl):
                self._reopened_seen[lvl] = n
                self.stats["cert_revoked_reopen"] += self._drop_level(lvl)
        mo = int(ag.stats.get("hidden_death", 0))
        if mo != self._hidden_death_seen:
            # Death on previously safe edge indicates hidden state: revoke level certificates and mark hidden
            self._hidden_death_seen = mo
            nb = ag.nodes.get(before_key) if before_key is not None else None
            lvl = nb.level if nb is not None else int(ag.prev_levels)
            self.stats["cert_hidden_death"] += 1
            self.hidden_levels.add(lvl)
            self._drop_level(lvl)
        if before_key is None or before_act is None or batch_n != 1:
            return                                       # batch execution: edge does not correspond to single action, skip certification
        node = ag.nodes.get(before_key)
        if node is None:
            return
        r = node.tried.get(before_act)
        c = (before_key, before_act)
        if c in self.certs or (died and r == _NOOP):
            # Lethal outcomes invalidate certificates; unexpected death indicates hidden state.
            if r in (_CHANGE, _PROGRESS, _DEATH) or died:  # falsified certificate implies hidden state exists in current level
                self.stats["cert_falsified"] += 1
                self.hidden_levels.add(node.level)
                self._drop_level(node.level)
            elif r == _NOOP:
                self.certs[c] += 1
        elif r == _NOOP and node.level not in self.hidden_levels:  # lethal outcome handled above: never certify non-effect on GAME_OVER
            self.certs[c] = 1
            self.stats["cert_emitted"] += 1

    def _masked_here(self) -> tuple[set[str], list[tuple]]:
        key = self.policy.agent.prev_key
        names: set[str] = set()
        clicks: list[tuple] = []
        if key is None:
            return names, clicks
        for (k, a) in self.certs:
            if k != key:
                continue
            if a[0] == 6:
                clicks.append(a)
            else:
                names.add(_MODEL_NAME[a[0]])
        return names, clicks

    def _filter_valid(self, valid_actions) -> list[str] | None:
        """Filters certified non-effect keys from advisory action list (preserves MOUSE and UNDO)."""
        if self.h8 is None or not isinstance(valid_actions, list):
            return valid_actions
        names, _ = self._masked_here()
        if not names:
            return valid_actions
        keep = self._mask(valid_actions, names)
        if keep is None:
            return valid_actions
        self.stats["h8_masked_keys"] += len(valid_actions) - len(keep)
        return keep

    @staticmethod
    def _mask(valid_actions: list, names: set[str]) -> list | None:
        """Returns masked action list; returns None if filtering would yield an empty list."""
        keep = [v for v in valid_actions if _MODEL_NAME.get(_ENGINE_ID.get(str(v).strip().upper(), -1), None) not in names]
        return keep if keep else None

    def h8_lines(self, max_items: int = 6) -> list[str]:
        """Formats prompt section detailing masked non-effect actions and predicted outcomes."""
        ag = self.policy.agent
        key = ag.prev_key
        node = ag.nodes.get(key) if key is not None else None
        if node is None:
            return []
        lines: list[str] = []
        names, clicks = self._masked_here()
        if names:
            # Report only actions actually removed; if all actions were masked and restored, do not announce removal
            valid = list(self._last_valid)
            keep = self._mask(valid, names) if valid else []
            if valid and keep is None:
                lines.append("- every key in your valid list was verified to change nothing from this exact state (list kept "
                             "whole): " + ", ".join(sorted(names)) + "; prefer MOUSE, RESET or a different approach")
            elif valid:
                shown = {_MODEL_NAME.get(_ENGINE_ID.get(str(v).strip().upper(), -1)) for v in valid}
                removed = sorted(n for n in names if n in shown)
                if removed:
                    lines.append("- removed from your valid list (verified to change nothing from this exact state): " + ", ".join(removed))
            else:
                # empty valid action list (harness returns [] when only RESET exists)
                lines.append("- verified to change nothing from this exact state (advisory; no action list to filter): " + ", ".join(sorted(names)))
        if clicks:
            lines.append("- clicks verified to change nothing here: " + ", ".join(_fmt_act(c) for c in clicks[:max_items]) + (" ..." if len(clicks) > max_items else ""))
        preds = []
        for a, t in list(node.edges.items())[:max_items * 2]:
            r = node.tried.get(a)
            if r == _NOOP or t == node.key:
                continue
            tgt = ag.nodes.get(t)
            if r == _PROGRESS:
                preds.append(f"{_fmt_act(a)} → COMPLETED THE LEVEL before")
            elif tgt is not None:
                n_un = len(_untried_legal(tgt))
                preds.append(f"{_fmt_act(a)} → state seen before ({n_un} untried actions there)")
        deaths = [a for a, r in node.tried.items() if r == _DEATH]
        for a in deaths[:2]:
            preds.append(f"{_fmt_act(a)} → GAME OVER before" + (f" (life budget ≈ {ag.life_budget})" if ag.life_budget else ""))
        if preds:
            lines.append("- predicted outcomes of known actions here: " + "; ".join(preds[:max_items]))
        if node.level in self.hidden_levels:
            lines.append("- caution: on this level the same screen evolved differently twice (hidden state); graph notes are weaker")
        return lines

    # ---- yield-based arbitration ----
    def _record(self, who: str, payload: dict[str, Any]) -> None:
        if not payload or payload.get("error") or "executed_count" not in payload:
            return
        n = int(payload.get("executed_count") or 1)
        eff = 1 if payload.get("board_changed") else 0
        prog = 1 if payload.get("level_completed") else 0
        lst = self.outcomes[who]
        lst.extend([(eff, prog)] * max(1, n))
        if len(lst) > 4 * self.yield_window:
            del lst[:-2 * self.yield_window]
        lvl = int(self.game.current_state.levels_completed)
        if lvl != self.level_of_turns:
            self.level_of_turns = lvl
            self.turns_in_level = 0
            self.outcomes = {"model": [], "k": []}            # yield measured per level

    def _yield(self, who: str) -> float | None:
        w = self.outcomes[who][-self.yield_window:]
        if len(w) < self.probe_actions:
            return None
        return sum(10 * p + e for e, p in w) / len(w)

    def _yield_says_k(self) -> bool:
        ym, yk = self._yield("model"), self._yield("k")
        if ym is None:
            return False                                      # model not yet probed on this level
        if yk is None:
            return True                                       # probe explorer on this level
        return yk > ym

    # inject prompt guidance
    def _patch_prompt(self) -> None:
        if not self.inject_hint or not hasattr(self.model, "_build_user_prompt"):
            return
        orig = self.model._build_user_prompt
        policy = self.policy

        hyb = self

        def patched(action_num, **kw):
            text = orig(action_num, **kw)
            try:
                lines = policy.hint_lines()
                if hyb.h8 is not None:
                    lines = lines + hyb.h8_lines()
                if hyb.h5_hint:
                    lines = lines + policy.h5_lines()
                if hyb.cross_level and hyb.turns_in_level < hyb.cross_level_turns:
                    lines = policy.cross_level_lines() + lines
                lines = _cap_lines(lines)
                hyb.max_hint_chars = max(hyb.max_hint_chars, sum(len(l) + 1 for l in lines))
                parts = text.split("\n", 2)
                if len(parts) == 3:
                    return parts[0] + "\n" + parts[1] + "\n" + "\n".join(lines) + "\n" + parts[2]
                return text + "\n" + "\n".join(lines)
            except Exception as e:
                hyb._note_exc("layer_exc", e)
                return text
        self.model._build_user_prompt = patched

    # observe all executed actions
    def _wrap_step_env(self, step_env: Callable) -> Callable:
        policy = self.policy
        hyb = self

        def wrapped(arguments: dict[str, Any]) -> dict[str, Any]:
            # Guarded execution ensuring model request reaches step_env exactly once
            before_key, first, cert, had_cert = None, None, None, False
            try:
                before_key = policy.agent.prev_key
                first = _first_action(arguments)
                cert = (before_key, first) if (before_key is not None and first is not None) else None
                had_cert = cert in hyb.certs if cert else False
                # limited refusal without spending action; executes after refuse_limit attempts
                if hyb.h8 == "refusal" and had_cert and hyb.refusals.get(cert, 0) < hyb.refuse_limit:
                    hyb.refusals[cert] = hyb.refusals.get(cert, 0) + 1
                    hyb.stats["h8_refused"] += 1
                    return {"executed": False, "h8_refused": True,
                            "error": (f"{_fmt_act(first)} was NOT executed: the explorer verified {hyb.certs[cert]}x that it changes "
                                      "nothing from this exact state. Choose a different action."),
                            "valid_actions": hyb._filter_valid(list(hyb._last_valid))}
            except Exception as e:
                hyb._note_exc("layer_exc", e)
                before_key, first, cert, had_cert = None, None, None, False
            payload = step_env(arguments)
            if _payload_ok(payload):
                hyb._executed += 1                       # record execution before layer processing to prevent re-execution
            try:
                hyb._record("model", payload)
                executed = payload.get("executed_actions") or []
                if executed and not payload.get("error"):
                    n_exec = len(executed)
                    hyb.stats["model_act_total"] += 1
                    if had_cert:
                        hyb.stats["model_act_certified"] += 1
                    if n_exec > 1:
                        hyb.stats["model_batches"] += 1
                    # only the final action of a batch has visible resulting frame; record it
                    name = str(executed[-1]).split("(")[0].strip().upper()
                    eng_id = _ENGINE_ID.get(name)
                    data = None
                    if eng_id is not None:
                        if eng_id == 6:
                            # In batch execution, only the n_exec-th requested action corresponds to the final visible frame
                            _lst = [arguments] if arguments.get("action") else list(arguments.get("actions") or [{}])
                            d = _lst[min(n_exec, len(_lst)) - 1]
                            # Harness clamps coordinates to 0..63 before execution; explorer observes identical clamped click
                            data = {"x": max(0, min(63, int(d.get("col", 0)))), "y": max(0, min(63, int(d.get("row", 0))))}
                        policy.observe("RESET" if eng_id == 0 else f"ACTION{eng_id}", data, None)
                    if n_exec > 1:
                        # batch: edge (previous node, last action) is unobserved; update node without edge learning
                        policy.agent.prev_act = None
                    act_t = policy.agent.prev_act
                    frame = policy.current_frame()
                    was_noop = False
                    node_before = policy.agent.nodes.get(before_key) if before_key is not None else None
                    if frame.state is GameState.GAME_OVER:
                        hyb.stats["model_deaths"] += 1
                    elif node_before is not None and act_t is not None and node_before.tried.get(act_t) == 0:
                        was_noop = True
                    # Explorer processes new frame (edge, graph; on GAME_OVER records death). Explorer proposal is absorbed without mutating decision state.
                    policy.absorb(frame)
                    hyb._cert_sync(before_key, act_t, n_exec, died=frame.state is GameState.GAME_OVER)
                    if frame.state is not GameState.GAME_OVER:
                        # discarded proposal must not create spurious self-loop edges on subsequent takeover
                        policy.agent.prev_act = None
                        hyb.noop_repeats = hyb.noop_repeats + 1 if was_noop else 0
                    if isinstance(payload.get("valid_actions"), list):
                        payload["valid_actions"] = hyb._filter_valid(payload["valid_actions"])
            except Exception as e:
                hyb._note_exc("layer_exc", e)
            return payload
        return wrapped

    def _time_pressure(self) -> bool:
        if self.max_runtime_s is None:
            return False
        elapsed = time.monotonic() - self.t0
        remaining = self.max_runtime_s - elapsed
        if remaining < self.time_floor_s:
            return True
        lvl = int(self.game.current_state.levels_completed)
        if lvl != self.level_seen:
            self.level_seen = lvl
            self.level_t0 = time.monotonic()
        return (time.monotonic() - self.level_t0) > self.level_share * self.max_runtime_s

    def _fallback_turn(self, step_env: Callable, why: str) -> Any:
        """Fallback action when an unexpected error occurs; ensures run continuity."""
        try:
            ag = self.policy.agent
            ag.prev_key = ag.prev_act = None             # fallback action is not an explorer decision: do not record graph edge
        except Exception:
            pass
        self.policy.last_proposal = None
        args = _legal_fallback(self.game, self._rng)
        if args is None:
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=f"{why}: without legal action")
        payload = step_env(args)
        ok = _payload_ok(payload)
        if ok:
            self._executed += 1
            self.stats["fallback_act"] += 1
        self.no_step = 0
        self.noop_repeats = 0
        return AnalyzerTurnResult(step_executed=ok, reasoning=f"fallback({why}) {args}")

    def _k_turn(self, step_env: Callable, why: str, transcript_path: Path | None, analysis_step):
        self.stats[why] += 1
        self.k_turns += 1
        try:
            args, diag = self.policy.propose()
        except Exception as e:                           # catch error to prevent evaluation crash
            self._note_exc("k_exc", e)
            return self._fallback_turn(step_env, why)
        ag = self.policy.agent
        k_key, k_act = ag.prev_key, ag.prev_act          # explorer action to be learned after execution
        payload = step_env(args)
        ok = _payload_ok(payload)
        if ok:
            self._executed += 1
        try:
            self._record("k", payload)
            if ok and k_act is not None:
                # Explorer absorbs frame from its own executed action to ensure correct edge and state attribution before subsequent model steps.
                life, max_life, level = ag.life, ag.max_life, int(ag.prev_levels)
                k_frame = self.policy.current_frame()
                self.policy.absorb(k_frame)
                self._cert_sync(k_key, k_act, 1, died=k_frame.state is GameState.GAME_OVER)
                if ag.prev_key is not None:               # non-terminal frame: extra proposal is not an action
                    ag.life = 0 if int(ag.prev_levels) > level else life  # level transition resets life budget in new level
                    ag.max_life = max_life
                    ag.prev_act = None
            if not ok:
                self.policy.mark_invalid()
        except Exception as e:
            self._note_exc("layer_exc", e)
        if transcript_path is not None:
            try:
                with open(transcript_path, "a", encoding="utf-8") as f:
                    f.write(f"[hybrid: K takes over ({why}) step {analysis_step}] {args} -> ok={ok}\n")
            except Exception:
                pass
        self.no_step = 0
        self.noop_repeats = 0
        return AnalyzerTurnResult(step_executed=ok, reasoning=f"K({why}) {str(diag)[:160]}")

    def analyze(self, state_path: Path, action_num: int, valid_actions=None, step_env: Callable | None = None,
                transcript_path: Path | None = None, analysis_step: int | None = None, transcript_updated=None,
                request_timeout_seconds=None, should_stop: Callable[[], bool] | None = None, **kw: Any):
        """Entry point invoked by harness solver. Handles errors gracefully to avoid run termination."""
        if step_env is None:
            return AnalyzerTurnResult(step_executed=False, reasoning="no step_env")
        n0 = self._executed
        try:
            return self._analyze(state_path, action_num, valid_actions, step_env, transcript_path, analysis_step,
                                 transcript_updated, request_timeout_seconds, should_stop, kw)
        except Exception as e:
            self._note_exc("layer_exc", e)
            if self._executed > n0:
                return AnalyzerTurnResult(step_executed=True, reasoning="layer_exc after action")
            try:
                return self._fallback_turn(step_env, "layer_exc")
            except Exception as e2:
                self._note_exc("layer_exc", e2)
                return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning="layer_exc")

    def _analyze(self, state_path, action_num, valid_actions, step_env, transcript_path, analysis_step,
                 transcript_updated, request_timeout_seconds, should_stop, kw):
        # Evaluate takeover triggers
        if self.no_step >= self.stall_steps:
            return self._k_turn(step_env, "k_stall", transcript_path, analysis_step)
        if self.noop_repeats >= self.repeat_noop_limit:
            return self._k_turn(step_env, "k_repeat", transcript_path, analysis_step)
        if self._time_pressure():
            return self._k_turn(step_env, "k_time", transcript_path, analysis_step)
        if self.selector == "yield" and self._yield_says_k():
            return self._k_turn(step_env, "k_yield", transcript_path, analysis_step)
        # Model turn, observed by explorer
        self.model_turns += 1
        self.turns_in_level += 1
        if self.selector == "yield":
            self.stats["model_yield"] += 1
        if isinstance(valid_actions, list):
            self._last_valid = list(valid_actions)
        valid_actions = self._filter_valid(valid_actions)
        n0 = self._executed
        try:
            result = self.model.analyze(state_path, action_num, valid_actions=valid_actions, step_env=self._wrap_step_env(step_env),
                                        transcript_path=transcript_path, analysis_step=analysis_step,
                                        transcript_updated=transcript_updated, request_timeout_seconds=request_timeout_seconds,
                                        should_stop=should_stop, **kw)
        except Exception as e:                           # Model ToolAgent failed: explorer assumes control instead of crashing
            self._note_exc("model_exc", e)
            if self._executed > n0:
                return AnalyzerTurnResult(step_executed=True, reasoning="model_exc after action")
            result = None
        if result is None or result.retryable_failure:
            # Model serving unavailable: explorer takes over to maintain progress
            return self._k_turn(step_env, "k_failure", transcript_path, analysis_step)
        if not result.step_executed and not getattr(result, "yielded_control", False):
            self.no_step += 1
        elif result.step_executed:
            self.no_step = 0
        return result

    @property
    def total_tokens(self) -> int:
        try:
            return int(getattr(self.model, "total_tokens", 0) or 0)
        except Exception:
            return 0


# ---------------------------------------------------------------------------------------------
# Factory for `bm.solver.analyzer_factory`
# ---------------------------------------------------------------------------------------------
def k_factory(max_actions: int = 10 ** 9):
    def make(game, index):
        a = KAnalyzer(game, max_actions=max_actions)
        REGISTRY[_game_key(game, index)] = a               # registry for final metrics inspection
        return a
    return make


REGISTRY: dict[str, Any] = {}      # game -> last analyzer created (for final diagnostic logging)
FACTORY_ERRORS: dict[str, str] = {}  # game -> error during construction (fallback to pure model)


def hybrid_factory(make_tool_agent: Callable[[Any, int], Any], **opts):
    """Factory constructing HybridAnalyzer instance with specified arbitration options."""
    def make(game, index):
        model = make_tool_agent(game, index)            # from harness: unexpected exception propagated as in original
        try:
            a = HybridAnalyzer(game, model, **opts)
        except Exception as e:
            # If explorer construction fails, proceed with base model
            FACTORY_ERRORS[_game_key(game, index)] = f"{type(e).__name__}: {e}"[:200]
            return model
        REGISTRY[_game_key(game, index)] = a
        return a
    return make
