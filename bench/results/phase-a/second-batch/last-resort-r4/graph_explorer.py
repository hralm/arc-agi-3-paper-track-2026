"""Graph Explorer K: frontier-driven BFS graph exploration with goal hypotheses, win-condition induction, and life budgeting for ARC-AGI-3.

Architecture and exploration strategy:
- State abstraction: node key = frame pixels (with HUD masked) + completed levels.
  Each graph node stores raw frame bytes, candidate actions categorized by priority tier,
  tried actions, and directed action->node transition edges.
- HUD mask discovery: static/timer regions detected across transitions are masked; when the mask
  updates, the node index is reindexed from raw frames, merging equivalent states.
- Priority tiers: tier 0 = available directional keys; tier 1 = non-background connected component
  centers (sorted by size); tier 2 = secondary component centers; tier 3 = ACTION7; tier 4 = random grid clicks.
- Planning & Decision: if current node has an untried candidate at priority <= p, sample it;
  otherwise, BFS along known edges to the nearest node with untried actions; if none, increment p.
- Life budget & hidden-state handling: repeated deaths at consistent step counts infer life timers,
  triggering voluntary reset before lethal steps. Deaths on previously safe edges are treated as
  hidden-state events rather than deterministic lethal edges.
- Goal hypotheses (H1 & H2):
  * Directional motion (H1): tracks mobile color components and static target components to bias
    action selection towards goal alignment.
  * Work area vs reference window (H2): identifies dynamic work regions and static reference windows,
    biasing BFS and action selection towards minimizing cell discrepancy.
- Win-condition induction (H5):
  * Maintains candidate win hypotheses in five predicate families: mover-onto-target (O1, off by
    default), class removal (O3), region-equals-reference (G1), objects inside a closed contour (G2),
    and mirror symmetry (G4). The leading hypothesis only orders candidates; it never excludes an action.
  * Evaluates instance progress on every frame to bias action ordering and BFS path selection.
"""
from __future__ import annotations

import random
import time
import zlib
from collections import Counter, deque
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

NOOP, CHANGE, PROGRESS, DEATH = 0, 1, 2, 3
TIER_KEYS, TIER_CLICK_A, TIER_CLICK_B, TIER_UNDO, TIER_RANDOM = 0, 1, 2, 3, 4
MAX_TIER = TIER_RANDOM
MAX_COMPS = 128            # candidate components per state
RANDOM_CLICKS = 12
REOPEN_MAX = 8             # max times tested actions may be reopened per level
BFS_LIMIT = 20000
EXHAUST_RESET = 2          # exhausted: alternates between reopening tested actions and level reset
HUD_BORDER = 4
HUD_MIN_EVENTS = 3
HUD_MAX_STEP = 3
HUD_CELL_MIN = 5
HUD_RATE = 0.9
TIME_LIMIT_S = 20 * 60
TARGET_PATIENCE = 15       # actions near target without progress before switching target
FRONTIER_SLACK = 3         # extra steps beyond nearest frontier accepted by BFS for higher-priority target
DIR_MIN = 2                # observations required to trust directional displacement of a key
H2_MIN_OVERLAP = 0.5       # fraction of work area structure that reference area must cover
H2_MIN_CHANGED = 4         # changed cells threshold before searching for work/reference areas
H2_PATIENCE = 150          # guided actions without reducing discrepancy falsifies hypothesis and disables guidance
TIMER_HITS = 3             # deaths at consistent step counts required to infer life timer budget


# ---------- H5: win-condition hypothesis identification ----------
# Hypotheses order candidate actions within priority tiers and break BFS ties; never exclude actions.
# Hypotheses expire when progress stalls; elimination requires observed refutation: exact predicate true
# on stable frame without level advance after trying each available action type at state. O1 has no exact
# predicate and only expires. Module failure transitions to error state, resuming baseline exploration.
H5_ON = True
H5_PATIENCE = 40           # actions without progress improvement threshold before instance expires
H5_SYM_MAX = 0.2           # symmetry discrepancy on initial frame <= 20% of non-background cells
H5_O3_MAX = 0.25           # component class <= 25% of grid in initial frame (background floor excluded)
H5_COMPOSE = 2             # prior level corroborated family bonus
H5_FAMS = ("O1", "O3", "G1", "G2", "G4")
H5_EXACT = ("O3", "G1", "G2", "G4")   # families with exact predicate evaluation across full grid
H5_FAMILY_ON = {fam: fam != "O1" for fam in H5_FAMS}  # family ablation; O1 disabled by default due to quadratic latency on large grids


class Hyp:
    __slots__ = ("fam", "bind", "state", "base", "best", "stale", "coupled")

    def __init__(self, fam: str, bind, base: int) -> None:
        self.fam, self.bind, self.base = fam, bind, base
        self.state = "active"                       # lifecycle: active, expired, eliminated, corroborated, error
        self.best: int | None = None
        self.stale = 0
        self.coupled = False                      # progress improved under action (coupled)

    def order(self, compose: str | None) -> tuple:
        ev = self.base + (1 if self.coupled else 0) + (H5_COMPOSE if self.fam == compose else 0)
        return (-ev, H5_FAMS.index(self.fam), repr(self.bind))


class H5:
    """Level instance registry: tracks active hypotheses, evaluates win-condition progress, and supplies guidance."""

    def __init__(self, ag: "MyAgent") -> None:
        self.ag = ag
        self.level: int | None = None
        self.hyps: dict[tuple, Hyp] = {}
        self.elim: set[tuple] = set()             # monotonic per level: reopening does not restore eliminated hypotheses
        self.compose: str | None = None           # corroborated family from previously solved level
        self.counts0: Counter = Counter()
        self.removed: Counter = Counter()         # color -> cell count converted to background (removal)
        self.ncells = 1
        self.top_key: tuple | None = None
        self.mobile_colors: set[int] = set()  # classes displaced under action, including click-driven movement
        self._cache_grid = None
        self._cache_ui = None
        self._cache_counts = None
        self._cache_positions = None
        self._cache_comps: dict[int, list] = {}
        self._cache_holes: dict[int, set[tuple[int, int]]] = {}

    def _cache_for(self, grid) -> None:
        if grid is not self._cache_grid or self.ag.ui is not self._cache_ui:
            self._cache_grid = grid
            self._cache_ui = self.ag.ui
            self._cache_counts = None
            self._cache_positions = None
            self._cache_comps.clear()
            self._cache_holes.clear()

    def _positions(self, grid) -> dict[int, set[tuple[int, int]]]:
        self._cache_for(grid)
        if self._cache_positions is None:
            positions = {}
            ui = self.ag.ui
            for y, row in enumerate(grid):
                for x, color in enumerate(row):
                    if (x, y) not in ui:
                        positions.setdefault(color, set()).add((x, y))
            self._cache_positions = positions
        return self._cache_positions

    # ---- perception (feature extraction per frame, outside HUD) ----
    def _counts(self, grid) -> Counter:
        self._cache_for(grid)
        if self._cache_counts is not None:
            return self._cache_counts
        self._cache_counts = Counter({color: len(cells) for color, cells in self._positions(grid).items()})
        return self._cache_counts

    @staticmethod
    def _bg(grid) -> int:
        return Counter(v for r in grid for v in r).most_common(1)[0][0]

    def _comps(self, grid, color: int) -> list[tuple[float, float]]:
        self._cache_for(grid)
        if color in self._cache_comps:
            return self._cache_comps[color]
        remaining = set(self._positions(grid).get(color, set()))
        out = []
        while remaining:
            seed = next(iter(remaining))
            st, cells = [seed], []
            remaining.remove(seed)
            while st:
                cx, cy = st.pop()
                cells.append((cx, cy))
                for p in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if p in remaining:
                        remaining.remove(p)
                        st.append(p)
            out.append(MyAgent._centroid(cells))
        self._cache_comps[color] = out
        return out

    def _hole_cells(self, grid, color: int) -> set[tuple[int, int]]:
        """Interior cells enclosed by 4-connected component boundaries; exterior excluded."""
        self._cache_for(grid)
        if color in self._cache_holes:
            return self._cache_holes[color]
        bs = set(self._positions(grid).get(color, set()))
        if not bs:
            self._cache_holes[color] = set()
            return set()
        components = []
        while bs:
            seed = next(iter(bs))
            comp, stack = {seed}, [seed]
            bs.remove(seed)
            while stack:
                x, y = stack.pop()
                for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if p in bs:
                        bs.remove(p)
                        comp.add(p)
                        stack.append(p)
            components.append(comp)
        holes = set()
        for comp in components:
            xs, ys = [p[0] for p in comp], [p[1] for p in comp]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            box = {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)} - comp
            exterior = {p for p in box if p[0] in (x0, x1) or p[1] in (y0, y1)}
            stack = list(exterior)
            while stack:
                x, y = stack.pop()
                for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if p in box and p not in exterior:
                        exterior.add(p)
                        stack.append(p)
            holes |= box - exterior
        self._cache_holes[color] = holes
        return holes

    def _sym_cells(self, grid, axis: str) -> set[tuple[int, int]]:
        if MyAgent._ragged(grid):                 # Guard ragged frames in every caller
            return set()
        ui = self.ag.ui
        h, w = len(grid), len(grid[0])
        bad = set()
        for y in range(h):
            for x in range(w):
                mx, my = (w - 1 - x, y) if axis == "v" else (x, h - 1 - y)
                if (x, y) in ui or (mx, my) in ui:
                    continue
                if grid[y][x] != grid[my][mx]:
                    bad.add((x, y))
        return bad

    def progress(self, h: Hyp, grid, mover) -> int | None:
        """Scalar metric to minimize; None if inapplicable on current frame."""
        if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
            return None                        # irregular grid dimensions are inapplicable across all hypothesis families
        if h.fam == "O1":
            if isinstance(h.bind, tuple):
                a, b = h.bind
                aa, bb = self._comps(grid, a), self._comps(grid, b)
                if not aa or not bb:
                    return None
                return int(round(min(MyAgent._l1(x, y) for x in aa for y in bb)))
            cs = self._comps(grid, h.bind)
            if mover is None or not cs:
                return None
            return int(round(min(MyAgent._l1(mover, c) for c in cs)))
        if h.fam == "O3":
            return self._counts(grid)[h.bind]
        if h.fam == "G1":
            x0, y0, bw, bh, rx, ry = h.bind
            height, width = len(grid), len(grid[0])
            if (min(x0, y0, rx, ry) < 0 or bw <= 0 or bh <= 0 or
                    max(x0 + bw, rx + bw) > width or max(y0 + bh, ry + bh) > height or
                    any(len(row) < width for row in grid)):
                return None
            return self.ag._h2_mismatch(grid, h.bind)
        if h.fam == "G2":
            a, b = h.bind
            holes = self._hole_cells(grid, b)
            if not holes or not self._positions(grid).get(a):
                return None
            return sum(v == a and (x, y) not in holes and (x, y) not in self.ag.ui
                       for y, row in enumerate(grid) for x, v in enumerate(row))
        if h.fam == "G4":
            return len(self._sym_cells(grid, h.bind))
        return None

    # ---- frame update cycle ----
    def observe(self, levels: int, prev_levels: int, frames: list, grid, node: "Node", pn: "Node | None",
                avail: list[int]) -> None:
        st = self.ag.stats
        st["h5_calls"] += 1
        self._cache_grid = None                # invalidate cached features when grid or UI mask mutates
        if self.level is not None and levels > self.level:
            win = frames[-2] if len(frames) >= 2 else self.ag.prev_grid
            self.credit(win, pn.mover if pn is not None else None)
        if self.level is None or levels != self.level:
            self._new_level(levels, grid)
        elif self.ag.prev_grid is not None and len(self.ag.prev_grid) == len(grid) and grid:
            pg, ui, bg = self.ag.prev_grid, self.ag.ui, self._bg(grid)
            if len(pg[0]) == len(grid[0]):
                prev_counts = self._counts(pg)
                now_counts = self._counts(grid)
                bg = self._bg(grid)
                gone, arrived = set(), set()
                for y, (before, after) in enumerate(zip(pg, grid)):
                    for x, (old, new) in enumerate(zip(before, after)):
                        if old != new and (x, y) not in ui:
                            gone.add(old)
                            arrived.add(new)
                self.mobile_colors.update(c for c in gone & arrived
                                          if c != bg and now_counts[c] and now_counts[c] == prev_counts[c])
                for y, row in enumerate(grid):
                    prow = pg[y]
                    for x, v in enumerate(row):
                        if v == bg and prow[x] != bg and (x, y) not in ui:
                            self.removed[prow[x]] += 1
        self._instantiate(levels, grid, node, avail)
        self._update(grid, node)

    def _new_level(self, levels: int, grid) -> None:
        self.level = levels
        self.hyps.clear()
        self.elim.clear()
        self.top_key = None
        self.counts0 = self._counts(grid)
        self.removed.clear()
        self.mobile_colors.clear()
        self.ncells = max(1, sum(len(r) for r in grid))
        bg = self._bg(grid)
        nonbg = sum(c for v, c in self.counts0.items() if v != bg) or 1
        for axis in ("v", "h"):
            d = len(self._sym_cells(grid, axis))
            if 0 < d <= H5_SYM_MAX * nonbg:
                self._add("G4", axis, 1)

    def _add(self, fam: str, bind, base: int) -> None:
        if not H5_FAMILY_ON[fam]:
            return
        k = (fam, bind)
        if k in self.hyps or k in self.elim:
            return
        self.hyps[k] = Hyp(fam, bind, base)
        self.ag.stats["h5_instance_" + fam] += 1

    def _instantiate(self, levels: int, grid, node: "Node", avail: list[int]) -> None:
        ag = self.ag
        bg = self._bg(grid)
        if node.mover is not None and any(a in avail for a in (1, 2, 3, 4)):
            seen: list[int] = []
            for col in ag._static_colors(levels, grid):
                if col != bg and col not in seen:
                    seen.append(col)
            for i, col in enumerate(seen):
                self._add("O1", col, 3 if i == 0 else 2)            # mobile + static target pairing (rarest first)
        # correlated displacement across positions identifies target under click
        static = [c for c in ag._static_colors(levels, grid) if c != bg]
        for a in sorted(self.mobile_colors):
            for b in static:
                if a != b:
                    self._add("O1", (a, b), 2)
            for b in sorted(self._counts(grid)):
                if a != b and b != bg and self._hole_cells(grid, b):
                    self._add("G2", (a, b), 2)
        now = self._counts(grid)
        for col, n0 in self.counts0.items():
            if col != bg and 0 < n0 <= H5_O3_MAX * self.ncells and now.get(col, 0) < n0 and self.removed[col]:
                self._add("O3", col, 2)                             # present in initial frame and subsequently removed to background
        if 6 in avail:
            reg = ag._h2_regions(levels, grid)
            if reg is not None:
                self._add("G1", reg, 3)                             # work area W + reference region R with overlap >= 50%

    def _update(self, grid, node: "Node") -> None:
        st = self.ag.stats
        for k, h in list(self.hyps.items()):
            if h.state not in ("active", "corroborated"):
                continue
            try:
                pr = self.progress(h, grid, node.mover)
            except Exception:
                st["h5_update_exc"] += 1       # isolate instance update exceptions
                continue
            if pr is None:
                continue
            if h.best is None or pr < h.best:
                if h.best is not None:
                    h.coupled = True
                h.best, h.stale = pr, 0
            else:
                h.stale += 1
                if h.stale > H5_PATIENCE:
                    h.state = "expired"
                    st["h5_expired"] += 1
            if pr == 0 and h.fam in H5_EXACT and self._triggers_done(node):
                h.state = "eliminated"
                self.elim.add(k)
                del self.hyps[k]
                st["h5_eliminated_" + h.fam] += 1

    @staticmethod
    def _triggers_done(node: "Node") -> bool:
        """Verifies whether all available legal action types have been tested at this state without level advance."""
        if not node.legal or node.legal == frozenset({0}):
            return False
        action_types = {c[0] for c, r in node.tried.items()}
        if any(r == PROGRESS for r in node.tried.values()):
            return False
        return all(a in action_types for a in node.legal if a != 0)

    def top(self) -> Hyp | None:
        active_hypotheses = [h for h in self.hyps.values() if h.state in ("active", "corroborated")]
        if not active_hypotheses:
            self.top_key = None
            return None
        h = min(active_hypotheses, key=lambda x: x.order(self.compose))
        self.top_key = (h.fam, h.bind)
        return h

    def on_exhaust(self) -> None:
        """Level graph exhausted: expire current top hypothesis and rotate to next."""
        if self.top_key in self.hyps:
            self.hyps[self.top_key].state = "expired"
            self.ag.stats["h5_expired_on_exhaust"] += 1
        self.top()                             # advance to next ranked hypothesis

    def credit(self, win, mover) -> None:
        """Winning frame credit: reward top-ranked hypothesis matching winning state transition."""
        st = self.ag.stats
        st["h5_win_levels"] += 1
        if not win:
            return
        true_instances = []
        cw = self._counts(win)
        for k, h in self.hyps.items():
            try:
                pr = self.progress(h, win, mover)
            except Exception:
                st["h5_credit_exc"] += 1       # isolate credit evaluation exception
                continue
            if h.fam == "O1":
                b = h.bind[1] if isinstance(h.bind, tuple) else h.bind
                # target component partially covered on win frame or mobile adjacent to target
                if cw[b] < self.counts0.get(b, 0) or (pr is not None and pr <= 1):
                    true_instances.append(k)
            elif pr == 0:
                true_instances.append(k)
        if not true_instances:
            st["h5_win_no_true_instance"] += 1                    # uncredited win frame anomaly
            self.compose = None
            return
        st["h5_win_true_instances"] += len(true_instances)
        if self.top_key in true_instances:
            st["h5_win_top"] += 1
            self.compose = self.top_key[0]
        else:
            self.compose = min(true_instances, key=lambda k: self.hyps[k].order(None))[0]
        if len(true_instances) > 1:
            st["h5_win_ambiguous"] += 1

    # ---- decision guidance ----
    def guide(self, grid, node: "Node", avail: list[int]):
        """Returns (directional target, reference region, preferred clicks, distance heuristic)."""
        h = self.top()
        if h is None:
            return None
        self.ag.stats["h5_guidance_" + h.fam] += 1
        keys = any(a in avail for a in (1, 2, 3, 4))
        target = reg = None
        clicks: set[tuple[int, int]] = set()
        if h.fam in ("O1", "O3") and keys and node.mover is not None:
            b = h.bind[1] if h.fam == "O1" and isinstance(h.bind, tuple) else h.bind
            cs = self._comps(grid, b)
            if cs:
                target = min(cs, key=lambda c: (MyAgent._l1(node.mover, c), c))
        if h.fam == "O3" and 6 in avail:
            ui = self.ag.ui
            clicks = {(x, y) for y, r in enumerate(grid) for x, v in enumerate(r) if v == h.bind and (x, y) not in ui}
        if h.fam == "O1" and isinstance(h.bind, tuple) and 6 in avail:
            a, _ = h.bind
            clicks = {(x, y) for y, row in enumerate(grid) for x, v in enumerate(row)
                      if v == a and (x, y) not in self.ag.ui}
        if h.fam == "G2" and 6 in avail:
            a, _ = h.bind
            clicks = {(x, y) for y, row in enumerate(grid) for x, v in enumerate(row)
                      if v == a and (x, y) not in self.ag.ui}
        if h.fam == "G1":
            reg = h.bind
        if h.fam == "G4" and 6 in avail:
            clicks = self._sym_cells(grid, h.bind)

        def hs(n: "Node") -> int | None:
            g = MyAgent._unpack(n.raw, n.shape)
            return self.progress(h, g, n.mover)
        return target, reg, clicks, hs


class Node:
    __slots__ = ("key", "level", "raw", "shape", "cands", "tier", "cls", "tried", "edges", "visits", "mover", "legal", "fused", "core", "unproven")

    def __init__(self, key: int, level: int, raw: bytes, shape: tuple[int, int]) -> None:
        self.key = key
        self.level = level
        self.raw = raw
        self.shape = shape
        self.cands: list[tuple] = []
        self.tier: dict[tuple, int] = {}
        self.cls: dict[tuple, tuple] = {}            # click candidate -> component class (color, size, width, height)
        self.tried: dict[tuple, int] = {}
        self.edges: dict[tuple, int] = {}
        self.visits = 0
        self.mover: tuple[float, float] | None = None
        self.legal: frozenset[int] = frozenset()      # available action ids on last visit (can vary without changing pixels)
        self.fused = False                             # merged node flag: inherited tried actions require re-verification
        self.core: frozenset[int] | None = None        # legal action ids intersection across all merged visits
        self.unproven: set[tuple] = set()              # tried actions inherited during merge, not yet re-observed

    def eff_tier(self, c: tuple, reactive: set[tuple]) -> int:
        k = self.cls.get(c)
        if k is not None and k in reactive:
            return TIER_CLICK_A
        return self.tier[c]

    def untried(self, p: int, reactive: set[tuple]) -> list[tuple]:
        return [c for c in self.cands if c not in self.tried and (not self.legal or c[0] in self.legal)
                and self.eff_tier(c, reactive) <= p]

    def merge(self, other: "Node") -> None:
        for c in other.cands:
            if c not in self.tier:
                self.cands.append(c)
                self.tier[c] = other.tier[c]
                if c in other.cls:
                    self.cls[c] = other.cls[c]
        self.tried.update(other.tried)
        self.edges.update(other.edges)
        self.visits += other.visits
        a = self.core if self.core is not None else (self.legal or None)
        b = other.core if other.core is not None else (other.legal or None)
        self.core = b if a is None else (a if b is None else a & b)   # intersect core availability across merged visits
        self.legal = self.legal | other.legal          # reindexing merges visits across differing availability sets
        self.fused = True                              # inherited noop from raw frame merge is not proof on new visit
        self.unproven |= set(self.tried) | other.unproven   # mark inherited tried actions as unproven until re-observed

    def plan_legal(self) -> frozenset[int] | None:
        """Action ids a plan may use through this node without knowing which merged frame it will reach."""
        if self.fused and self.core is not None:
            return self.core
        return self.legal or None


class MyAgent(Agent):
    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rng = random.Random(zlib.crc32(self.game_id.encode()))
        self.t0 = time.time()
        self.nodes: dict[int, Node] = {}
        self.p_level: dict[int, int] = {}
        self.exhaust: dict[int, int] = {}
        self.reopened: dict[int, int] = {}
        self.reactive: set[tuple] = set()           # component classes that already responded to click actions
        self.cls_tries: Counter[tuple] = Counter()
        self.plan: deque[tuple[tuple, int]] = deque()
        self.prev_grid: list[list[int]] | None = None
        self.prev_key: int | None = None
        self.prev_act: tuple | None = None
        self.prev_levels = 0
        self._click: dict[str, int] | None = None   # instance click coordinates avoiding shared mutable state on Enum
        self.steps = 0
        self.ui: set[tuple[int, int]] = set()
        self.hud: dict[int, dict] = {}
        self.hud_reset = False
        self.last_effective: set[tuple[int, int]] = set()
        self.stats = Counter()
        self.life = 0                                  # action count since last reset
        self.death_at: Counter[int] = Counter()
        self.max_life = 0
        self.life_budget: int | None = None
        self.start_key: dict[int, int] = {}            # level -> root node (life 0)
        self.dir_map: dict[int, Counter] = {a: Counter() for a in (1, 2, 3, 4)}
        self.changed_ever: dict[int, set[tuple[int, int]]] = {}
        self.target_idx: dict[int, int] = {}
        self.target_wait: dict[int, int] = {}
        self.tcache: dict[tuple, list] = {}
        self.guide_off: set[int] = set()               # levels where graph was exhausted; hypothesis failed, resume exploration
        self.h2cache: dict[tuple, tuple | None] = {}    # (level, changed_count) -> (x0, y0, bw, bh, rx, ry) or None
        self.h2_best: dict[int, int] = {}               # level -> minimum observed cell discrepancy
        self.h2_stale: dict[int, int] = {}              # level -> count of guided actions without improvement
        self.tcolors: dict[tuple, list[int]] = {}       # static component colors sorted by rarity
        self.h5: H5 | None = H5(self) if H5_ON else None

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return (time.time() - self.t0) > TIME_LIMIT_S

    # ---------- frame processing ----------
    @staticmethod
    def _grid(frame: FrameData) -> list[list[int]]:
        return frame.frame[-1] if frame.frame else []

    # ---------- HUD mask discovery ----------
    @staticmethod
    def _line(x: int, y: int, h: int, w: int):
        if y < HUD_BORDER or y >= h - HUD_BORDER:
            return ("r", y)
        if x < HUD_BORDER or x >= w - HUD_BORDER:
            return ("c", x)
        return None

    def _hud_learn(self, level: int, changed_vals: list[tuple[int, int, int, int]], h: int, w: int) -> None:
        hud = self.hud.setdefault(level, {"_n": 0, "_cells": {}, "bars": {}})
        hud["_n"] += 1
        n = hud["_n"]
        if self.hud_reset:
            for e in hud["bars"].values():
                e["cells"] = set()
            self.hud_reset = False
        groups: dict[tuple, set] = {}
        for x, y, f, t in changed_vals:
            line = self._line(x, y, h, w)
            if line is not None:
                groups.setdefault((line, f, t), set()).add((x, y))
        bars = hud["bars"]
        mirror = {k for k in groups if (k[0], k[2], k[1]) in groups}
        gained: dict[int, int] = {}
        for (line, f, t), cells in groups.items():
            gained[t] = gained.get(t, 0) + len(cells)
        for (line, f, t), cells in groups.items():
            if (line, f, t) in mirror or gained.get(f, 0) > 0:
                continue
            for c in cells:
                rec = hud["_cells"].setdefault(c, [0, n])
                rec[0] += 1
        for (line, f, t), cells in groups.items():
            if (line, f, t) in mirror:
                bars.setdefault((line, f, t), {"cells": set(), "seen": set(), "events": 0, "bad": False})["bad"] = True
                continue
            rev = bars.get((line, t, f))
            if rev is not None and rev["events"] > 0 and (cells & rev["cells"] or len(cells) > HUD_MAX_STEP):
                if len(cells) > HUD_MAX_STEP:
                    rev["cells"] = set()
                else:
                    rev["bad"] = True
                continue
            e = bars.setdefault((line, f, t), {"cells": set(), "seen": set(), "events": 0, "bad": False})
            if e["bad"]:
                continue
            if cells & e["cells"]:
                e["bad"] = True
                continue
            if e["cells"]:
                pos = (lambda c: c[0]) if line[0] == "r" else (lambda c: c[1])
                if min(abs(pos(p) - pos(q)) for p in cells for q in e["cells"]) > HUD_MAX_STEP:
                    e["bad"] = True
                    continue
            if len(cells) > HUD_MAX_STEP and e["events"] < HUD_MIN_EVENTS:
                e["bad"] = True
                continue
            e["events"] += 1
            e["cells"] |= cells
            e["seen"] |= cells

    def _hud_keys(self, level: int):
        hud = self.hud.get(level)
        if not hud:
            return [], set()
        keys = [k for k, e in hud["bars"].items()
                if not e["bad"] and e["events"] >= HUD_MIN_EVENTS and len(e["seen"]) >= HUD_MIN_EVENTS]
        n = hud["_n"]
        counters = {c for c, (k, first) in hud["_cells"].items()
                    if k >= HUD_CELL_MIN and k / max(1, n - first + 1) >= HUD_RATE}
        return keys, counters

    def _hud_mask(self, level: int, grid: list[list[int]]) -> set[tuple[int, int]]:
        keys, counters = self._hud_keys(level)
        m = set(counters)
        h = len(grid)
        w = len(grid[0]) if h else 0
        bars = self.hud.get(level, {}).get("bars", {})
        for (line, f, t) in keys:
            axis, idx = line
            cells = [(x, idx) for x in range(w)] if axis == "r" else [(idx, y) for y in range(h)]
            ok = [grid[y][x] in (f, t) for x, y in cells]
            e = bars[(line, f, t)]
            seed = e["seen"] | e["cells"]
            i = 0
            while i < len(cells):
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j < len(cells) and ok[j]:
                    j += 1
                if any(cells[k] in seed for k in range(i, j)):
                    m.update(cells[i:j])
                i = j
        return m

    @staticmethod
    def _is_hud_change(keys, counters, x: int, y: int, vb: int, va: int) -> bool:
        if (x, y) in counters:
            return True
        for (axis, idx), f, t in keys:
            if (y if axis == "r" else x) == idx and vb in (f, t) and va in (f, t):
                return True
        return False

    def _update_mask(self, grid: list[list[int]], level: int) -> set[tuple[int, int]]:
        """Compares with previous grid to update HUD mask; returns coordinates of cells changed outside HUD."""
        vals: list[tuple[int, int, int, int]] = []
        pg = self.prev_grid
        h, w = len(grid), len(grid[0]) if grid else 0
        if pg is not None and len(pg) == len(grid):
            for y, (r0, r1) in enumerate(zip(pg, grid)):
                if r0 != r1:
                    for x, (a, b) in enumerate(zip(r0, r1)):
                        if a != b:
                            vals.append((x, y, a, b))
            self.steps += 1
            if level == self.prev_levels:
                self._hud_learn(level, vals, h, w)
        keys, counters = self._hud_keys(level)
        eff = {(x, y) for x, y, a, b in vals if not self._is_hud_change(keys, counters, x, y, a, b)}
        new_ui = (self._hud_mask(level, grid) if grid else set()) | self.ui
        if new_ui != self.ui:
            self.ui = new_ui
            self._reindex()
        self.last_effective = eff
        return eff

    # ---------- graph representation ----------
    def _key_of(self, grid: list[list[int]], levels: int) -> int:
        if not self.ui:
            return hash((levels, tuple(tuple(r) for r in grid)))
        rows = []
        for y, r in enumerate(grid):
            rows.append(tuple(-1 if (x, y) in self.ui else v for x, v in enumerate(r)))
        return hash((levels, tuple(rows)))

    @staticmethod
    def _pack(grid: list[list[int]]) -> bytes:
        return bytes(v & 0xFF for r in grid for v in r)

    @staticmethod
    def _unpack(raw: bytes, shape: tuple[int, int]) -> list[list[int]]:
        h, w = shape
        return [list(raw[y * w:(y + 1) * w]) for y in range(h)]

    def _reindex(self) -> None:
        """Recomputes node keys using raw frames when HUD mask changes, merging newly equivalent nodes."""
        self.stats["reindex"] += 1
        old = list(self.nodes.values())
        remap: dict[int, int] = {}
        fresh: dict[int, Node] = {}
        for n in old:
            k = self._key_of(self._unpack(n.raw, n.shape), n.level)
            remap[n.key] = k
            if k in fresh:
                fresh[k].merge(n)
            else:
                n.key = k
                fresh[k] = n
        for n in fresh.values():
            n.edges = {a: remap.get(t, t) for a, t in n.edges.items()}
        self.nodes = fresh
        self.start_key = {lv: remap.get(k, k) for lv, k in self.start_key.items()}   # update root keys after node reindexing
        if self.prev_key is not None:
            self.prev_key = remap.get(self.prev_key, self.prev_key)
        self.plan.clear()

    @staticmethod
    def _ragged(grid) -> bool:
        """Empty or ragged grid guard for methods indexing by the first row."""
        return not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid)

    def _click_targets(self, grid: list[list[int]]) -> list[tuple[int, int]]:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if not h or not w or self._ragged(grid):    # Reject ragged grid
            return []
        bg = Counter(c for row in grid for c in row).most_common(1)[0][0]
        seen = [[False] * w for _ in range(h)]
        comps: list[tuple[int, int, int]] = []
        for y in range(h):
            for x in range(w):
                if seen[y][x] or grid[y][x] == bg or (x, y) in self.ui:
                    continue
                col = grid[y][x]
                stack = [(x, y)]
                seen[y][x] = True
                cells = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == col:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
                mx = sum(c[0] for c in cells) // len(cells)
                my = sum(c[1] for c in cells) // len(cells)
                cx, cy = min(cells, key=lambda c: abs(c[0] - mx) + abs(c[1] - my))
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                comps.append((len(cells), cx, cy, (int(col), len(cells), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)))
        comps.sort(key=lambda t: (t[0], t[3][0], t[1], t[2]))
        out = []
        for _, cx, cy, k in comps[:MAX_COMPS]:
            if w != 64 or h != 64:
                cx, cy = cx * 64 // w, cy * 64 // h
            out.append((cx, cy, k))
        return out

    def _node(self, key: int, grid: list[list[int]], levels: int, avail: list[int]) -> Node:
        n = self.nodes.get(key)
        if n is None:
            n = Node(key, levels, self._pack(grid), (len(grid), len(grid[0]) if grid else 0))
            self.nodes[key] = n
        # Node key is pixels + level; available_actions may change across visits without altering frame pixels. Each
        # visit reconciles: newly available keys become untried candidates, and legal set updates to current availability.
        self._seed_cands(n, grid, avail)
        n.legal = frozenset(avail) if avail else frozenset({0})   # reset-only fallback when no action keys available
        return n

    def _seed_cands(self, n: Node, grid: list[list[int]], avail: list[int]) -> None:
        for a in avail:
            if a in (1, 2, 3, 4, 5) and (a,) not in n.tier:
                n.cands.append((a,))
                n.tier[(a,)] = TIER_KEYS
        if 6 in avail and not any(c[0] == 6 for c in n.cands):
            seen_cls: set[tuple] = set()
            for x, y, k in self._click_targets(grid):
                c = (6, x, y)
                if c in n.tier:
                    continue
                n.cands.append(c)
                n.cls[c] = k
                n.tier[c] = TIER_CLICK_A if k not in seen_cls else TIER_CLICK_B
                seen_cls.add(k)
        if 7 in avail and (7,) not in n.tier:
            n.cands.append((7,))
            n.tier[(7,)] = TIER_UNDO

    def _add_random_clicks(self, n: Node) -> int:
        added = 0
        for _ in range(RANDOM_CLICKS):
            c = (6, self.rng.randrange(64), self.rng.randrange(64))
            if c not in n.tier:
                n.cands.append(c)
                n.tier[c] = TIER_RANDOM
                added += 1
        return added

    # ---------- directional goal hypothesis: mobile component, target, heading ----------
    @staticmethod
    def _centroid(cells) -> tuple[float, float]:
        n = len(cells)
        return (sum(c[0] for c in cells) / n, sum(c[1] for c in cells) / n)

    def _learn_mover(self, node: Node, pn: Node, act: tuple, pg: list[list[int]], grid: list[list[int]],
                     effective: set[tuple[int, int]], res: int) -> None:
        if res == PROGRESS:
            return
        if act[0] not in (1, 2, 3, 4) or res != CHANGE or not effective:
            if node.mover is None and pn.mover is not None and act[0] not in (1, 2, 3, 4):
                node.mover = pn.mover
            return
        # color pair (C->F, F->C) with matching counts indicates color C moved over background F
        trans: Counter = Counter((pg[y][x], grid[y][x]) for (x, y) in effective)
        pairs = [(c, f) for (c, f), n in trans.items() if c < f and trans.get((f, c), 0) == n]
        if len(pairs) != 1:                      # ambiguous (multiple moving entities) or no exact count match: skip displacement update
            return
        c, f = pairs[0]
        appeared = [(x, y) for (x, y) in effective if pg[y][x] == f and grid[y][x] == c]
        vanished = [(x, y) for (x, y) in effective if pg[y][x] == c and grid[y][x] == f]
        if not self._connected(appeared) and not self._connected(vanished):
            return
        if not self._connected(appeared):        # swap if c was the background
            c, f = f, c
            appeared, vanished = vanished, appeared
        cv = self._centroid(vanished)
        ca = self._centroid(appeared)
        dx, dy = ca[0] - cv[0], ca[1] - cv[1]
        sx = 0 if abs(dx) < 0.5 else (1 if dx > 0 else -1)
        sy = 0 if abs(dy) < 0.5 else (1 if dy > 0 else -1)
        if (sx, sy) != (0, 0):
            self.dir_map[act[0]][(sx, sy)] += 1
        node.mover = ca
        self.stats["mover_moves"] += 1

    @staticmethod
    def _connected(cells) -> bool:
        if not cells:
            return False
        s = set(cells)
        q = [cells[0]]
        seen = {cells[0]}
        while q:
            x, y = q.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in s and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return len(seen) == len(s)

    def _dir_of(self, a: int) -> tuple[int, int] | None:
        c = self.dir_map.get(a)
        if not c:
            return None
        (d, n), = c.most_common(1)
        if n < DIR_MIN or n / sum(c.values()) < 0.6:
            return None
        return d

    def _targets(self, level: int, grid: list[list[int]]) -> list[tuple[float, float]]:
        ch = self.changed_ever.get(level, set())
        ck = (level, len(ch), len(self.ui))
        if ck in self.tcache:
            return self.tcache[ck]
        h = len(grid)
        w = len(grid[0]) if h else 0
        bg = Counter(v for r in grid for v in r).most_common(1)[0][0]
        seen = [[False] * w for _ in range(h)]
        comps = []
        for y in range(h):
            for x in range(w):
                if seen[y][x] or grid[y][x] == bg or (x, y) in self.ui:
                    continue
                col = grid[y][x]
                stack = [(x, y)]
                seen[y][x] = True
                cells = []
                static = True
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    if (cx, cy) in ch:
                        static = False
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == col:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
                if static and len(cells) <= 0.25 * h * w:
                    comps.append((int(col), len(cells), self._centroid(cells)))
        rar = Counter(c[0] for c in comps)
        comps.sort(key=lambda t: (rar[t[0]], t[1], t[2]))
        out = [c[2] for c in comps]
        self.tcache[ck] = out
        self.tcolors[ck] = [c[0] for c in comps]
        return out

    def _static_colors(self, level: int, grid: list[list[int]]) -> list[int]:
        self._targets(level, grid)
        return self.tcolors[(level, len(self.changed_ever.get(level, set())), len(self.ui))]

    def _target(self, level: int, grid: list[list[int]]) -> tuple[float, float] | None:
        ts = self._targets(level, grid)
        if not ts:
            return None
        return ts[self.target_idx.get(level, 0) % len(ts)]

    @staticmethod
    def _l1(a: tuple[float, float], b: tuple[float, float]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _goal_score(self, c: tuple, mover, target) -> int:
        """Returns +1 if action reduces distance to target on axis, -1 if distance increases, 0 if unknown."""
        if mover is None or target is None or c[0] not in (1, 2, 3, 4):
            return 0
        d = self._dir_of(c[0])
        if d is None:
            return 0
        tx = 0 if abs(target[0] - mover[0]) < 0.5 else (1 if target[0] > mover[0] else -1)
        ty = 0 if abs(target[1] - mover[1]) < 0.5 else (1 if target[1] > mover[1] else -1)
        return d[0] * tx + d[1] * ty

    # ---------- discrepancy hypothesis: work region and reference window ----------
    def _h2_regions(self, level: int, grid: list[list[int]]):
        if self._ragged(grid):                      # Reject ragged grid
            return None
        ch = self.changed_ever.get(level, set())
        ck = (level, len(ch))
        if ck in self.h2cache:
            return self.h2cache[ck]
        res = None
        h = len(grid)
        w = len(grid[0]) if h else 0
        cells = [c for c in ch if c not in self.ui]
        if len(cells) >= H2_MIN_CHANGED and h and w:
            bg = Counter(v for r in grid for v in r).most_common(1)[0][0]
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            if 2 <= bw and 2 <= bh and bw * bh <= 0.25 * h * w:
                wmask = [[grid[y0 + j][x0 + i] != bg for i in range(bw)] for j in range(bh)]
                wstruct = sum(sum(r) for r in wmask)
                blocked = ch | self.ui
                best = None
                for ry in range(0, h - bh + 1):
                    for rx in range(0, w - bw + 1):
                        if not (rx + bw <= x0 or rx >= x0 + bw or ry + bh <= y0 or ry >= y0 + bh):
                            continue
                        overlap = 0
                        cols: set[int] = set()
                        ok = True
                        for j in range(bh):
                            row = grid[ry + j]
                            for i in range(bw):
                                if (rx + i, ry + j) in blocked:
                                    ok = False
                                    break
                                v = row[rx + i]
                                if v != bg:
                                    cols.add(v)
                                    if wmask[j][i]:
                                        overlap += 1
                            if not ok:
                                break
                        if ok and len(cols) >= 2 and (best is None or overlap > best[0]):
                            best = (overlap, rx, ry)
                if best is not None and wstruct and best[0] >= H2_MIN_OVERLAP * wstruct:
                    res = (x0, y0, bw, bh, best[1], best[2])
                    if ck[1] and self.stats.get("h2_ref_level_%d" % level) is None:
                        self.stats["h2_ref_level_%d" % level] = 1
        self.h2cache[ck] = res
        return res

    @staticmethod
    def _h2_mismatch(grid: list[list[int]], reg) -> int:
        x0, y0, bw, bh, rx, ry = reg
        return sum(1 for j in range(bh) for i in range(bw) if grid[y0 + j][x0 + i] != grid[ry + j][rx + i])

    def _h2_mismatch_node(self, n: Node, reg) -> int | None:
        grid = self._unpack(n.raw, n.shape)
        x0, y0, bw, bh, rx, ry = reg
        h, w = len(grid), len(grid[0]) if grid else 0
        if (min(x0, y0, rx, ry) < 0 or max(x0 + bw, rx + bw) > w or max(y0 + bh, ry + bh) > h
                or any(len(row) < w for row in grid)):
            return None                        # skip discrepancy check if dimensions mismatch
        return self._h2_mismatch(grid, reg)

    @staticmethod
    def _h2_cell_mismatch(grid: list[list[int]], reg, x: int, y: int) -> bool:
        x0, y0, bw, bh, rx, ry = reg
        if not (x0 <= x < x0 + bw and y0 <= y < y0 + bh):
            return False
        return grid[y][x] != grid[ry + (y - y0)][rx + (x - x0)]

    def _bfs(self, start: Node, p: int, target=None, reg=None, hs=None) -> list[tuple[tuple, int]] | None:
        """Finds shortest path along known transitions to nearest node containing untried candidates at priority <= max_p."""
        parent: dict[int, tuple[int, tuple] | None] = {start.key: None}
        depth: dict[int, int] = {start.key: 0}
        q = deque([start.key])
        found: int | None = None
        best_d = None
        d_min = None
        def safe_hs(n):
            try:
                return hs(n)
            except Exception:
                self.stats["h5_hs_exc"] += 1
                return None
        while q and len(parent) < BFS_LIMIT:
            k = q.popleft()
            n = self.nodes.get(k)
            if n is None:
                continue
            if d_min is not None and depth[k] > d_min + FRONTIER_SLACK:
                break
            if k != start.key and n.untried(p, self.reactive):
                if d_min is None:
                    d_min = depth[k]
                    found = k
                    if hs is not None:                 # evaluate progress heuristic from top hypothesis
                        best_d = safe_hs(n)
                        if best_d is None:
                            continue
                    elif target is not None and n.mover is not None:
                        best_d = self._l1(n.mover, target)
                    elif reg is not None:
                        best_d = self._h2_mismatch_node(n, reg)
                        if best_d is None:
                            continue
                    else:
                        break
                else:
                    if hs is not None:
                        dd = safe_hs(n)
                        if dd is None:
                            continue
                    elif target is not None and n.mover is not None:
                        dd = self._l1(n.mover, target)
                    elif target is None and reg is not None:
                        dd = self._h2_mismatch_node(n, reg)
                        if dd is None:
                            continue
                    else:
                        continue
                    if best_d is None or dd < best_d:
                        best_d, found = dd, k
                        self.stats["guided_frontier"] += 1
                continue
            for a, t in n.edges.items():
                if t == k or t in parent:
                    continue
                ok = (n.legal or None) if k == start.key else n.plan_legal()   # validate against core legal set for intermediate nodes
                if ok is not None and a[0] not in ok:  # prune edges incompatible with node availability
                    continue
                tn = self.nodes.get(t)
                if tn is None or tn.level != start.level:
                    continue
                parent[t] = (k, a)
                depth[t] = depth[k] + 1
                q.append(t)
        if found is None:
            return None
        path: list[tuple[tuple, int]] = []
        k = found
        while parent[k] is not None:
            pk, a = parent[k]
            path.append((a, k))
            k = pk
        path.reverse()
        return path

    # ---------- action selection & planning ----------
    def _reset(self, why: str) -> GameAction:
        self.prev_key = self.prev_act = None
        self.prev_grid = None
        self.plan.clear()
        self.hud_reset = True
        self.life = 0
        self.stats["reset_" + why] += 1
        act = GameAction.RESET
        act.reasoning = {"why": why}
        return act

    def _emit(self, n: Node, c: tuple, how: str) -> GameAction:
        if n.legal and c[0] not in n.legal:
            # Ensure proposed action is currently legal before mutating state; discard stale candidates or plans
            # inherited from visits with differing availability. Single legality check before state mutation.
            self.stats["illegal_action_avoided"] += 1
            if how == "plan":
                self.plan.clear()
            pool = n.untried(MAX_TIER, self.reactive) or [k for k in n.cands if k[0] in n.legal] \
                or [(a,) for a in sorted(n.legal)]
            c, how = self.rng.choice(pool), "legal"
        self.prev_key, self.prev_act = n.key, c
        self.life += 1
        self.max_life = max(self.max_life, self.life)
        self.stats[how] += 1
        act = GameAction.from_id(c[0])
        act.reasoning = {"how": how, "nodes": len(self.nodes), "p": self.p_level.get(n.level, 0)}
        # Action enum member is shared across threads; click coordinates are kept on instance
        # (_click, prev_act) to avoid cross-instance collision.
        self._click = {"x": int(c[1]), "y": int(c[2])} if act.is_complex() and len(c) == 3 else None
        return act

    def do_action_request(self, action: GameAction) -> FrameData:
        """Standalone harness entry point: submits action with instance-specific click coordinates."""
        if action.is_complex() and self._click:
            data = dict(self._click)
        else:
            data = action.action_type().model_dump()     # fresh instance: isolated state
        reasoning = getattr(action, "reasoning", None)
        if reasoning is not None and not isinstance(reasoning, dict):
            reasoning = {"text": str(reasoning)}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        levels = latest_frame.levels_completed
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self.death_at[self.life] += 1
                if self.life_budget is None and self.death_at[self.life] >= TIMER_HITS and self.life >= self.max_life:
                    self.life_budget = self.life
                    self.stats["timer"] = self.life
            if self.prev_key is not None and self.prev_act is not None and self.prev_key in self.nodes:
                pn = self.nodes[self.prev_key]
                safe_before = pn.tried.get(self.prev_act) in (NOOP, CHANGE, PROGRESS)
                timed = self.life_budget is not None and self.life >= self.life_budget
                if safe_before or timed:
                    self.stats["hidden_death"] += 1
                    if not safe_before:
                        pn.tried.pop(self.prev_act, None)
                else:
                    pn.tried[self.prev_act] = DEATH
                    pn.edges.pop(self.prev_act, None)
            return self._reset("game_over" if latest_frame.state is GameState.GAME_OVER else "start")

        grid = self._grid(latest_frame)
        if not grid:
            return self._reset("empty_frame")
        effective = self._update_mask(grid, levels)
        key = self._key_of(grid, levels)
        avail = [a for a in latest_frame.available_actions if a != 0]
        node = self._node(key, grid, levels, avail)    # empty availability: legal={0}, candidate-free node
        node.visits += 1
        if levels > self.prev_levels:
            self.life = 0                              # new level: reset life budget tracker
            if self.plan:                              # clear pending plan on level transition to avoid cross-level action bleed
                self.stats["plan_new_level"] += 1
                self.plan.clear()
        if self.life == 0:
            self.start_key[levels] = key

        # 1. record directed edge from previous action
        if self.prev_key is not None and self.prev_act is not None and self.prev_key in self.nodes:
            pn = self.nodes[self.prev_key]
            if levels > self.prev_levels:
                res = PROGRESS
            elif effective or key != pn.key or not avail:  # losing all available actions represents state change
                res = CHANGE
            else:
                res = NOOP
            pn.tried[self.prev_act] = res
            pn.unproven.discard(self.prev_act)
            pn.edges[self.prev_act] = key
            if self.prev_grid is not None and levels == self.prev_levels:
                self.changed_ever.setdefault(levels, set()).update(effective)
                self._learn_mover(node, pn, self.prev_act, self.prev_grid, grid, effective, res)
            k = pn.cls.get(self.prev_act)
            if k is not None:
                self.cls_tries[k] += 1
                if res != NOOP:
                    self.reactive.add(k)
            if self.plan:
                exp_act, exp_key = self.plan[0]
                if exp_act == self.prev_act and exp_key == key:
                    self.plan.popleft()
                else:
                    self.stats["plan_diverged"] += 1
                    self.plan.clear()
        if self.h5 is not None:                        # update hypothesis progress on real frames
            try:
                pn5 = self.nodes.get(self.prev_key) if self.prev_key is not None else None
                self.h5.observe(levels, self.prev_levels, latest_frame.frame, grid, node, pn5, avail)
            except Exception:                          # module failure transitions to error state, resuming baseline
                self.stats["h5_exc"] += 1
                self.h5 = None
        self.prev_levels = levels
        self.prev_grid = grid
        if not avail:
            # only reset available: learn transition edge before triggering reset
            # to prevent looping on untried candidates.
            return self._reset("only_reset_available")

        # 2. life budget: reset is free, lethal action consumes step; preempt death with voluntary reset
        if self.life_budget is not None and self.life >= self.life_budget - 1 and not self.plan:
            return self._reset("life_budget_exhausted")

        # 2b. continue executing active BFS plan
        if self.plan:
            a, _ = self.plan[0]
            return self._emit(node, a, "plan")

        # 2c. directional goal guidance: track target and evaluate patience
        target = None
        if levels not in self.guide_off and any(a in avail for a in (1, 2, 3, 4)):
            target = self._target(levels, grid)
        if target is not None and node.mover is not None:
            if self._l1(node.mover, target) <= 2.0:
                self.target_wait[levels] = self.target_wait.get(levels, 0) + 1
                if self.target_wait[levels] > TARGET_PATIENCE:
                    self.target_idx[levels] = self.target_idx.get(levels, 0) + 1
                    self.target_wait[levels] = 0
                    self.stats["target_switched"] += 1
                    target = self._target(levels, grid)

        reg = None
        if levels not in self.guide_off and 6 in avail:
            reg = self._h2_regions(levels, grid)
            if reg is not None:
                mis = self._h2_mismatch(grid, reg)
                if mis < self.h2_best.get(levels, 1 << 30):
                    self.h2_best[levels] = mis
                    self.h2_stale[levels] = 0
                else:
                    self.h2_stale[levels] = self.h2_stale.get(levels, 0) + 1
                    if self.h2_stale[levels] > H2_PATIENCE:
                        self.guide_off.add(levels)
                        self.stats["h2_falsified"] += 1
                        reg = None

        clicks: set[tuple[int, int]] = set()
        hs = None
        if self.h5 is not None:                        # hypothesis guidance overrides default heuristic
            try:
                g = self.h5.guide(grid, node, avail)
                if g is not None:
                    target, reg, clicks, hs = g
            except Exception:
                self.stats["h5_exc"] += 1
                self.h5 = None

        # 3. select untried candidate at current priority; else BFS to frontier; else increment priority
        p = self.p_level.get(levels, 0)
        while True:
            un = node.untried(p, self.reactive)
            if un:
                lo = min(node.eff_tier(c, self.reactive) for c in un)
                pool = [c for c in un if node.eff_tier(c, self.reactive) == lo]
                if target is not None and node.mover is not None:
                    sc = {c: self._goal_score(c, node.mover, target) for c in pool}
                    hi = max(sc.values())
                    if hi > 0:
                        pool = [c for c in pool if sc[c] == hi]
                        self.stats["guided"] += 1
                elif reg is not None:
                    h0, w0 = node.shape
                    def _mis(c):
                        if c[0] != 6:
                            return False
                        gx, gy = (c[1] * w0 // 64, c[2] * h0 // 64) if (w0, h0) != (64, 64) else (c[1], c[2])
                        return self._h2_cell_mismatch(grid, reg, gx, gy)
                    pref = [c for c in pool if _mis(c)]
                    if pref:
                        pool = pref
                        self.stats["guided_h2"] += 1
                elif clicks:
                    h0, w0 = node.shape
                    pref = [c for c in pool if c[0] == 6 and ((c[1] * w0 // 64, c[2] * h0 // 64) if (w0, h0) != (64, 64)
                                                               else (c[1], c[2])) in clicks]
                    if pref:
                        pool = pref
                        self.stats["guided_h5_click"] += 1
                c = self.rng.choice(pool)
                self.p_level[levels] = p
                return self._emit(node, c, "local")
            path = self._bfs(node, p, target, reg, hs)
            if path and self.life_budget is not None and len(path) > self.life_budget - self.life - 1:
                sk = self.start_key.get(levels)
                sn = self.nodes.get(sk) if sk is not None else None
                alternate_path = self._bfs(sn, p) if sn is not None and sn.key != node.key else None
                if alternate_path is not None and len(alternate_path) < self.life_budget - 1 and len(alternate_path) <= len(path):
                    return self._reset("plan_over_budget")
            if path:
                self.plan = deque(path)
                self.p_level[levels] = p
                a, _ = self.plan[0]
                self.stats["plan_new"] += 1
                return self._emit(node, a, "plan")
            if p < MAX_TIER:
                p += 1
                if p == TIER_RANDOM and 6 in avail:
                    self._add_random_clicks(node)
                continue
            break

        # 4. exhausted graph on this level: retry tested actions (possible hidden state) or trigger reset
        self.exhaust[levels] = self.exhaust.get(levels, 0) + 1
        self.stats["exhausted"] += 1
        if levels not in self.guide_off:
            self.guide_off.add(levels)
            self.stats["guidance_disabled"] += 1
        if self.h5 is not None:
            self.h5.on_exhaust()
        if self.exhaust[levels] % EXHAUST_RESET == 1 and self.reopened.get(levels, 0) < REOPEN_MAX:
            self.reopened[levels] = self.reopened.get(levels, 0) + 1
            for n in self.nodes.values():
                if n.level == levels:
                    n.tried = {c: r for c, r in n.tried.items() if r == DEATH}
            self.p_level[levels] = 0
            self.stats["reopened"] += 1
            un = node.untried(MAX_TIER, self.reactive)
            if un:
                return self._emit(node, self.rng.choice(un), "reopened")
        if self.exhaust[levels] % EXHAUST_RESET == 0 or self.reopened.get(levels, 0) >= REOPEN_MAX:
            if self.life == 0 and node.cands:
                self.stats["reset_avoided"] += 1
                return self._emit(node, self.rng.choice(node.cands), "reset_avoided")
            return self._reset("exhausted")
        if 6 in avail and self._add_random_clicks(node):
            un = node.untried(MAX_TIER, self.reactive)
            return self._emit(node, self.rng.choice(un), "random")
        pool = [k for k in node.cands if k[0] in node.legal] or [(a,) for a in avail]
        return self._emit(node, self.rng.choice(pool), "retry")
