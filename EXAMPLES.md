# Examples

This file contains **copy‑paste commands** for training draft models and running **valid MCTS draft states**.

The examples below are rewritten so that:
- **All pick/ban orders are legally possible in pro drafts**
- **1.x (training) ↔ 2.x (usage) always match the same model**
- Each MCTS example clearly states **what stage of draft it represents**
- Examples are suitable for training **stage‑specific MCTS policies**

---

## Draft legality assumptions (important)

All examples assume the standard modern pro draft flow:

1. **Ban Phase 1**: B1 → R1 → B2 → R2
2. **Pick Phase 1**: B1 → R1,R2 → B2,B3
3. **Ban Phase 2**: R3 → B3
4. **Pick Phase 2**: R4 → B4 → R5

No example below violates this order.

---

## 1) Train models

> These commands create models tied to a **league + time window**.
> Each model is referenced later by a matching **2.x** MCTS example.

### 1.1 Train LCK model for all 2025 games (broad)

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_2025.joblib `
  --tournament-contains "2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

---

### 1.2 Train LCK Split 2 2025 model (recommended)

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_split2_2025.joblib `
  --tournament-contains "LCK - Split 2 2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

---

### 1.3 Train LTA North 2025 model (broad)

```powershell
python train_draft_model.py lta_games_with_bans.csv `
  --out models/lta_north_2025.joblib `
  --tournament-contains "LTA North" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

---

### 1.4 Train LTA North Split 2 2025 model (recommended)

```powershell
python train_draft_model.py lta_games_with_bans.csv `
  --out models/lta_north_split2_2025.joblib `
  --tournament-contains "LTA North - Split 2 2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

---

## 2) Run MCTS recommendations (LEGAL STATES ONLY)

Each section below **explicitly matches a 1.x model** and represents a **real draft stage**.

---

### 2.1 LCK – Start of draft (no bans yet)
Matches **model 1.1**

```powershell
python mcts_lol_draft.py `
  --model models/lck_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "T1" `
  --red-team "Gen.G Esports" `
  --iters 800 `
  --show 10
```

Draft state: **Before Ban Phase 1**

---

### 2.2 LCK – After Ban Phase 1
Matches **model 1.2**

```powershell
python mcts_lol_draft.py `
  --model models/lck_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "Dplus KIA" `
  --red-team "Hanwha Life Esports" `
  --blue-bans "Kalista|Renata Glasc" `
  --red-bans "Neeko|Rumble" `
  --iters 1000 `
  --show 10
```

Draft state: **Ban Phase 1 complete, Blue pick next**

---

### 2.3 LCK – After first pick rotation
Matches **model 1.2**

```powershell
python mcts_lol_draft.py `
  --model models/lck_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "T1" `
  --red-team "Gen.G Esports" `
  --blue-bans "Kalista|Renata Glasc" `
  --red-bans "Neeko|Rumble" `
  --blue-picks "Azir" `
  --red-picks "Orianna|Rakan" `
  --iters 1600 `
  --show 10
```

Draft state: **End of Pick Phase 1, Ban Phase 2 next**

---

### 2.4 LTA North – After Ban Phase 1
Matches **model 1.4**

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "FlyQuest" `
  --red-team "Team Liquid" `
  --blue-bans "Kalista|Renata Glasc" `
  --red-bans "Neeko|Rumble" `
  --iters 1200 `
  --show 12
```

Draft state: **Ban Phase 1 complete**

---

### 2.5 LTA North – Mid‑draft (after Ban Phase 2)
Matches **model 1.4**

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "100 Thieves" `
  --red-team "Shopify Rebellion" `
  --blue-bans "Kalista|Renata Glasc|Kennen" `
  --red-bans "Neeko|Rumble|Poppy" `
  --blue-picks "Azir|Sejuani" `
  --red-picks "Orianna|Rakan|Jax" `
  --iters 1800 `
  --show 10
```

Draft state: **Final pick rotation (R4 → B4 → R5)**

---

### Tagging convention for MCTS training

If you are collecting rollouts or logs, tag them as:

```
[mcts-stage]=start | post-ban1 | post-pick1 | post-ban2 | end
[mcts-model]=1.2
```

This allows **stage‑specific policy learning** while sharing a single trained model.

