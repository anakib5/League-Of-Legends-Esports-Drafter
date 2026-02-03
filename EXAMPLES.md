# Examples

This file contains copy‑paste commands for training models and running MCTS recommendations.

**Naming rule (now enforced):**
- **1.x training ↔ 2.x usage** always reference the **same model filename**
- 1.1 ↔ 2.1, 1.2 ↔ 2.2, etc

---

## 1) Train models

### 1.1 Train LCK model for all 2025 games (broad model)

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_2025.joblib `
  --tournament-contains "2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

### 1.2 Train LCK model for Split 2 2025 only (recommended)

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_split2_2025.joblib `
  --tournament-contains "LCK - Split 2 2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

### 1.3 Train LCK Split 2 model for a single patch (example: 15.10)

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_split2_15_10.joblib `
  --tournament-contains "LCK - Split 2 2025" `
  --patch-filter "15.10" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

### 1.4 Train LTA North model for all 2025 games (broad model)

```powershell
python train_draft_model.py lta_games_with_bans.csv `
  --out models/lta_north_2025.joblib `
  --tournament-contains "LTA North" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

### 1.5 Train LTA North model for Split 2 2025 only (recommended)

```powershell
python train_draft_model.py lta_games_with_bans.csv `
  --out models/lta_north_split2_2025.joblib `
  --tournament-contains "LTA North - Split 2 2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
  --pair-min-count 9999
```

---

## 2) Run MCTS recommendations

### 2.1 LCK broad 2025 draft
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

### 2.2 LCK Split 2 early draft
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
  --blue-bans "Kalista" `
  --red-bans "Renata Glasc" `
  --iters 800 `
  --show 10
```

### 2.3 LCK Split 2 mid draft
Matches **model 1.2**

```powershell
python mcts_lol_draft.py `
  --model models/lck_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "KT Rolster" `
  --red-team "T1" `
  --blue-picks "Azir" `
  --red-picks "Orianna|Rakan" `
  --iters 1600 `
  --show 10
```

### 2.4 LTA North broad 2025 draft
Matches **model 1.4**

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "Team Liquid" `
  --red-team "Cloud9 Kia" `
  --iters 800 `
  --show 10
```

### 2.5 LTA North Split 2 early draft
Matches **model 1.5**

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "FlyQuest" `
  --red-team "Team Liquid" `
  --blue-bans "Kalista" `
  --red-bans "Renata Glasc" `
  --iters 1200 `
  --show 12
```

### 2.6 LTA North Split 2 mid draft
Matches **model 1.5**

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "100 Thieves" `
  --red-team "Shopify Rebellion" `
  --blue-picks "Azir" `
  --red-picks "Orianna|Rakan" `
  --iters 1600 `
  --show 10
```

