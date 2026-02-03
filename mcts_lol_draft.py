#!/usr/bin/env python3
"""
MCTS (UCT/UCB1) draft recommender for pro League of Legends drafts.

- Uses the *same* encoding & model artifact produced by train_draft_model.py
- Supports bans + picks with standard pro draft order
- Evaluates terminal drafts with model.predict_proba (P(BLUE wins))
- Candidate pruning via:
    (a) team pickrate CSV (file2) OR
    (b) global pickrate from draft CSV (file1) OR
    (c) all champions (slow)

Example:
  python mcts_lol_draft.py \
    --model lck_2025.joblib \
    --champion-json champion.json \
    --patch 15.10 \
    --blue-team "NONGSHIM RED FORCE" \
    --red-team "Dplus KIA" \
    --blue-picks "Azir|Sejuani" \
    --red-picks "Ahri" \
    --blue-bans "Sion|Pantheon" \
    --red-bans "Vi|Gwen" \
    --team-stats file2_lck.csv \
    --split-key LCK_Split2_2025 \
    --iters 1200 \
    --topk 25

Notes:
- team1 = BLUE side; winner in training is 1 if BLUE won.
- This script assumes standard pro draft order (6 bans, 6 picks, 4 bans, 4 picks).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Set

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack  # type: ignore

# Import your exact encoding helpers from the training script.
# Keep this script in the same directory as train_draft_model.py.
from train_draft_model import (  # type: ignore
    load_champion_master,
    _split_pipe_list,
    _patch_bucket,
    _encode_drafts,
    encode_pair_features,
    encode_counter_features,
)

Team = Literal["BLUE", "RED"]
ActType = Literal["PICK", "BAN"]

# Standard pro draft order: 20 actions total
DRAFT_ORDER: List[Tuple[Team, ActType]] = [
    ("BLUE", "BAN"), ("RED", "BAN"), ("BLUE", "BAN"), ("RED", "BAN"), ("BLUE", "BAN"), ("RED", "BAN"),
    ("BLUE", "PICK"), ("RED", "PICK"), ("RED", "PICK"), ("BLUE", "PICK"), ("BLUE", "PICK"), ("RED", "PICK"),
    ("RED", "BAN"), ("BLUE", "BAN"), ("RED", "BAN"), ("BLUE", "BAN"),
    ("RED", "PICK"), ("BLUE", "PICK"), ("BLUE", "PICK"), ("RED", "PICK"),
]

@dataclass(frozen=True)
class Action:
    kind: ActType
    champ: str

@dataclass
class DraftState:
    blue_picks: Tuple[str, ...] = ()
    red_picks: Tuple[str, ...] = ()
    blue_bans: Tuple[str, ...] = ()
    red_bans: Tuple[str, ...] = ()
    step: int = 0  # 0..19

    def picked_set(self) -> Set[str]:
        return set(self.blue_picks) | set(self.red_picks)

    def banned_set(self) -> Set[str]:
        return set(self.blue_bans) | set(self.red_bans)

    def taken_set(self) -> Set[str]:
        return self.picked_set() | self.banned_set()

    def is_terminal(self) -> bool:
        return self.step >= len(DRAFT_ORDER)

    def apply(self, team: Team, action: Action) -> "DraftState":
        if action.champ in self.taken_set():
            raise ValueError(f"Illegal: {action.champ} already taken/banned")

        bp = list(self.blue_picks)
        rp = list(self.red_picks)
        bb = list(self.blue_bans)
        rb = list(self.red_bans)

        if action.kind == "PICK":
            if team == "BLUE":
                bp.append(action.champ)
            else:
                rp.append(action.champ)
        else:  # BAN
            if team == "BLUE":
                bb.append(action.champ)
            else:
                rb.append(action.champ)

        return DraftState(tuple(bp), tuple(rp), tuple(bb), tuple(rb), self.step + 1)

@dataclass
class Node:
    state: DraftState
    parent: Optional["Node"] = None
    action_from_parent: Optional[Action] = None
    children: Dict[Action, "Node"] = field(default_factory=dict)
    untried: List[Action] = field(default_factory=list)
    N: int = 0         # visits
    W: float = 0.0     # total value (P(BLUE wins))

    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

def ucb1(parent: Node, child: Node, c: float) -> float:
    if child.N == 0:
        return float("inf")
    return child.Q() + c * math.sqrt(math.log(parent.N) / child.N)

class DraftWinModel:
    """
    Inference wrapper that reproduces *exactly* the feature construction
    used during training: [champ_vec | pair_vec | counter_vec | meta_onehot]
    """
    def __init__(self, model_path: str, champion_json_path: str):
        payload = joblib.load(model_path)
        self.pipeline = payload["pipeline"]
        self.vocab = payload["champion_vocab"]
        self.pick_weight = float(payload["pick_weight"])
        self.ban_weight = float(payload["ban_weight"])
        self.ban_scale = payload.get("ban_scale", {}) or {}
        self.pairs = payload.get("pairs", []) or []
        self.counters = payload.get("counters", []) or []

        vocab_ids, alias_to_id = load_champion_master(champion_json_path)
        self.alias_to_id = alias_to_id

        # Your training uses champion_vocab order; make index map from it.
        self.champ_to_idx = {c: i for i, c in enumerate(self.vocab)}

    def predict_p_blue(self, patch: str, state: DraftState) -> float:
        row = {
            "match_id": "mcts_dummy",
            "patch": patch,
            "region": "MCTS",
            "tournament": "MCTS",
            "team1_name": "BLUE",
            "team2_name": "RED",
            "team1_drafted_champions": "|".join(state.blue_picks),
            "team2_drafted_champions": "|".join(state.red_picks),
            "team1_banned_champions": "|".join(state.blue_bans),
            "team2_banned_champions": "|".join(state.red_bans),
            "winner": 0,
        }
        df = pd.DataFrame([row])
        df["patch_bucket"] = df["patch"].map(_patch_bucket).astype(str)

        Xc = _encode_drafts(
            df,
            self.champ_to_idx,
            self.alias_to_id,
            pick_weight=self.pick_weight,
            ban_weight=self.ban_weight,
            ban_scale=self.ban_scale,
        )

        Xp = encode_pair_features(df, self.pairs, self.alias_to_id) if self.pairs else np.zeros((1, 0), np.float32)
        Xct = encode_counter_features(df, self.counters, self.alias_to_id) if self.counters else np.zeros((1, 0), np.float32)

        meta = df[["patch_bucket"]]
        meta_encoded = self.pipeline.named_steps["meta_preprocess"].transform(meta)

        X = hstack([csr_matrix(Xc), csr_matrix(Xp), csr_matrix(Xct), meta_encoded], format="csr")
        clf = self.pipeline.named_steps["clf"]
        p = float(clf.predict_proba(X)[0, 1])
        # Safety clamp
        return max(0.0, min(1.0, p))

class CandidatePolicy:
    """
    Provides candidate champs for PICK/BAN to control branching factor.
    Priority order:
      1) team stats CSV (file2): team-specific pick pools (and opponent pools for bans)
      2) global pickrate from draft CSV (file1)
      3) all champs
    """
    def __init__(
        self,
        all_champs: List[str],
        alias_to_id: Dict[str, str],
        team_stats_csv: Optional[str] = None,
        split_key: Optional[str] = None,
        blue_team: Optional[str] = None,
        red_team: Optional[str] = None,
        global_draft_csv: Optional[str] = None,
        topk_pick: int = 30,
        topk_ban: int = 30,
    ):
        self.all_champs = all_champs
        self.alias_to_id = alias_to_id
        self.topk_pick = topk_pick
        self.topk_ban = topk_ban

        self.blue_team = (blue_team or "").strip()
        self.red_team = (red_team or "").strip()
        self.split_key = (split_key or "").strip()

        self.team_pools: Dict[Tuple[str, str], List[str]] = {}  # (team_name, split_key)->[canon champs...]
        self.global_pool: List[str] = []

        if team_stats_csv and self.blue_team and self.red_team and self.split_key:
            self._load_team_stats(team_stats_csv)

        if global_draft_csv:
            self._load_global_from_file1(global_draft_csv)

        if not self.global_pool:
            # fallback: all champs as "global"
            self.global_pool = list(self.all_champs)

    def _canon(self, token: str) -> Optional[str]:
        # Reuse same normalization mapping as training: alias_to_id maps normalized strings
        # load_champion_master already built variants for cid + display.
        # encode uses _norm_name internally; we rely on encode to map at eval time,
        # but for pools we need canonical IDs. We'll approximate by calling training mapping keys:
        # easiest: use training _encode_drafts mapping rules by feeding display names later.
        # Here, our all_champs are already canonical IDs from champion.json.
        # For team_stats (file2), champ_name may be display; we map by scanning variants.
        # We'll do a robust approach: normalize like training does by using alias_to_id keys:
        import re
        key = re.sub(r"[^a-z0-9]+", "", str(token).lower())
        return self.alias_to_id.get(key)

    def _load_team_stats(self, path: str):
        df = pd.read_csv(path)
        # expected columns: region,split_key,team_name,champion_name,champ_pct (from your file2)
        need = {"split_key", "team_name", "champion_name", "champ_pct"}
        if not need.issubset(df.columns):
            return

        df = df[df["split_key"].astype(str) == self.split_key]
        for team in [self.blue_team, self.red_team]:
            dft = df[df["team_name"].astype(str) == team].copy()
            dft["canon"] = dft["champion_name"].map(self._canon)
            dft = dft.dropna(subset=["canon"])
            dft = dft.sort_values("champ_pct", ascending=False)
            champs = dft["canon"].astype(str).tolist()
            self.team_pools[(team, self.split_key)] = champs

    def _load_global_from_file1(self, path: str):
        df = pd.read_csv(path)
        # Count picks & bans globally
        cols = [
            "team1_drafted_champions", "team2_drafted_champions",
            "team1_banned_champions", "team2_banned_champions",
        ]
        for c in cols:
            if c not in df.columns:
                return

        counts: Dict[str, int] = {}
        for _, row in df.iterrows():
            for col in cols:
                for tok in _split_pipe_list(row[col]):
                    canon = self._canon(tok)
                    if canon:
                        counts[canon] = counts.get(canon, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        self.global_pool = [c for c, _n in ranked] or list(self.all_champs)

    def pick_candidates(self, team: Team) -> List[str]:
        tname = self.blue_team if team == "BLUE" else self.red_team
        pool = self.team_pools.get((tname, self.split_key), [])
        if pool:
            return pool[: max(self.topk_pick, 5)]
        return self.global_pool[: max(self.topk_pick, 5)]

    def ban_candidates(self, team: Team) -> List[str]:
        # Simple and effective: ban from opponent's pick pool
        opp_team = "RED" if team == "BLUE" else "BLUE"
        opp_name = self.blue_team if opp_team == "BLUE" else self.red_team
        pool = self.team_pools.get((opp_name, self.split_key), [])
        if pool:
            return pool[: max(self.topk_ban, 5)]
        return self.global_pool[: max(self.topk_ban, 5)]

def legal_actions(state: DraftState, team: Team, kind: ActType, policy: CandidatePolicy) -> List[Action]:
    taken = state.taken_set()
    pool = policy.pick_candidates(team) if kind == "PICK" else policy.ban_candidates(team)

    # Always allow fallback to global/all champs if pool gets exhausted
    base = pool if pool else policy.all_champs

    acts: List[Action] = []
    for champ in base:
        if champ not in taken:
            acts.append(Action(kind, champ))

    # If pruning too aggressive (no legal moves), expand to all champs
    if not acts:
        for champ in policy.all_champs:
            if champ not in taken:
                acts.append(Action(kind, champ))
    return acts

def rollout_complete(state: DraftState, policy: CandidatePolicy) -> DraftState:
    s = state
    while not s.is_terminal():
        team, kind = DRAFT_ORDER[s.step]
        acts = legal_actions(s, team, kind, policy)
        a = random.choice(acts)
        s = s.apply(team, a)
    return s

def select(node: Node, c: float) -> Node:
    while (not node.state.is_terminal()) and (not node.untried) and node.children:
        node = max(node.children.values(), key=lambda ch: ucb1(node, ch, c))
    return node

def expand(node: Node, policy: CandidatePolicy) -> Node:
    if node.state.is_terminal():
        return node

    team, kind = DRAFT_ORDER[node.state.step]
    if not node.untried:
        node.untried = legal_actions(node.state, team, kind, policy)
        random.shuffle(node.untried)

    if not node.untried:
        return node

    a = node.untried.pop()
    child_state = node.state.apply(team, a)
    child = Node(state=child_state, parent=node, action_from_parent=a)
    node.children[a] = child
    return child

def backprop(node: Node, value_blue: float) -> None:
    while node is not None:
        node.N += 1
        node.W += value_blue
        node = node.parent  # type: ignore[assignment]

def mcts_recommend(
    root_state: DraftState,
    patch: str,
    model: DraftWinModel,
    policy: CandidatePolicy,
    iters: int,
    c: float,
) -> Tuple[Action, List[Tuple[Action, float, int]]]:
    team, kind = DRAFT_ORDER[root_state.step]
    root = Node(state=root_state)

    root.untried = legal_actions(root_state, team, kind, policy)
    random.shuffle(root.untried)

    for _ in range(iters):
        node = select(root, c)
        node = expand(node, policy)
        terminal = rollout_complete(node.state, policy)
        value = model.predict_p_blue(patch, terminal)
        backprop(node, value)

    # Choose best by mean value
    if not root.children:
        raise RuntimeError("No children expanded; check candidate policy / legality.")

    ranked = sorted(
        [(a, ch.Q(), ch.N) for a, ch in root.children.items()],
        key=lambda t: t[1],
        reverse=True,
    )
    best_action = ranked[0][0]
    return best_action, ranked

def infer_step_from_counts(state: DraftState) -> int:
    # We trust user-provided lists and compute step.
    # Validate that counts match some prefix of DRAFT_ORDER.
    s = DraftState(
        blue_picks=state.blue_picks,
        red_picks=state.red_picks,
        blue_bans=state.blue_bans,
        red_bans=state.red_bans,
        step=0
    )

    # Reconstruct the step count only (we don't know exact sequence of champs, but counts must match).
    # We'll just count how many actions have already happened:
    done = len(state.blue_picks) + len(state.red_picks) + len(state.blue_bans) + len(state.red_bans)

    # Basic feasibility checks
    if len(state.blue_picks) > 5 or len(state.red_picks) > 5:
        raise ValueError("Too many picks; each side max 5.")
    if len(state.blue_bans) > 5 or len(state.red_bans) > 5:
        raise ValueError("Too many bans; each side max 5.")
    if done > 20:
        raise ValueError("Too many total actions (>20).")

    # Also check the done count aligns with draft order capacity by phase:
    # (This is a soft check; we allow partial inside any phase.)
    return done

def parse_state(args) -> DraftState:
    blue_picks = tuple(_split_pipe_list(args.blue_picks))
    red_picks = tuple(_split_pipe_list(args.red_picks))
    blue_bans = tuple(_split_pipe_list(args.blue_bans))
    red_bans = tuple(_split_pipe_list(args.red_bans))

    # If user provides display names, we keep them; model encoder will canonicalize.
    # But legality/candidate pools use canonical IDs. We'll canonicalize later via alias map.

    return DraftState(
        blue_picks=blue_picks,
        red_picks=red_picks,
        blue_bans=blue_bans,
        red_bans=red_bans,
        step=0
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained .joblib (from train_draft_model.py)")
    ap.add_argument("--champion-json", required=True, help="Path to champion.json used in training")
    ap.add_argument("--patch", required=True, help="Patch string, e.g. 15.10")

    ap.add_argument("--blue-team", default="", help="Blue team name (for team-stats candidate pools)")
    ap.add_argument("--red-team", default="", help="Red team name (for team-stats candidate pools)")
    ap.add_argument("--team-stats", default="", help="Optional team pickrate CSV (your file2 layout)")
    ap.add_argument("--split-key", default="", help="Split key used in team-stats CSV, e.g. LCK_Split2_2025")
    ap.add_argument("--global-draft-csv", default="", help="Optional draft CSV (file1) for global pickrate fallback")

    ap.add_argument("--blue-picks", default="", help="Pipe list, e.g. 'Azir|Sejuani'")
    ap.add_argument("--red-picks", default="", help="Pipe list")
    ap.add_argument("--blue-bans", default="", help="Pipe list")
    ap.add_argument("--red-bans", default="", help="Pipe list")

    ap.add_argument("--iters", type=int, default=1200, help="MCTS iterations")
    ap.add_argument("--c", type=float, default=0.7, help="UCB exploration constant")
    ap.add_argument("--topk-pick", type=int, default=30, help="Candidate pool size for picks")
    ap.add_argument("--topk-ban", type=int, default=30, help="Candidate pool size for bans")
    ap.add_argument("--show", type=int, default=10, help="How many top actions to print")

    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load champion master for canonical IDs + alias mapping
    vocab_ids, alias_to_id = load_champion_master(args.champion_json)

    model = DraftWinModel(args.model, args.champion_json)

    # Parse current (possibly partial) draft
    state_raw = parse_state(args)

    # Canonicalize champs in state for legality + MCTS transitions
    def canon_list(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        import re
        out = []
        for t in tokens:
            key = re.sub(r"[^a-z0-9]+", "", str(t).lower())
            cid = alias_to_id.get(key)
            if cid:
                out.append(cid)
        return tuple(out)

    state = DraftState(
        blue_picks=canon_list(state_raw.blue_picks),
        red_picks=canon_list(state_raw.red_picks),
        blue_bans=canon_list(state_raw.blue_bans),
        red_bans=canon_list(state_raw.red_bans),
        step=0,
    )

    # Compute current step from counts
    step = infer_step_from_counts(state)
    state = DraftState(state.blue_picks, state.red_picks, state.blue_bans, state.red_bans, step=step)

    if state.is_terminal():
        p = model.predict_p_blue(args.patch, state)
        print(f"Draft is terminal. Model P(BLUE wins) = {p:.4f}")
        return

    team_to_act, kind = DRAFT_ORDER[state.step]
    print(f"Current step {state.step+1}/20: {team_to_act} to {kind}")
    print(f"BLUE picks: {list(state.blue_picks)}")
    print(f"RED  picks: {list(state.red_picks)}")
    print(f"BLUE bans : {list(state.blue_bans)}")
    print(f"RED  bans : {list(state.red_bans)}")

    policy = CandidatePolicy(
        all_champs=vocab_ids,
        alias_to_id=alias_to_id,
        team_stats_csv=(args.team_stats or None),
        split_key=(args.split_key or None),
        blue_team=(args.blue_team or None),
        red_team=(args.red_team or None),
        global_draft_csv=(args.global_draft_csv or None),
        topk_pick=args.topk_pick,
        topk_ban=args.topk_ban,
    )

    best, ranked = mcts_recommend(
        root_state=state,
        patch=args.patch,
        model=model,
        policy=policy,
        iters=args.iters,
        c=args.c,
    )

    print("\nRecommendation")
    print(f"  {team_to_act} {best.kind}: {best.champ}")

    print(f"\nTop {min(args.show, len(ranked))} actions (mean P(BLUE wins), visits):")
    for a, q, n in ranked[: args.show]:
        print(f"  {a.kind:<4} {a.champ:<18}  Q={q:.4f}  N={n}")

if __name__ == "__main__":
    main()
