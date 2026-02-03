# Examples

This file contains copy paste commands for training models and running MCTS recommendations.

Assumptions:
- You are running from the repository root
- You are using PowerShell on Windows
- Your data files follow these names:
  - Match drafts: lck_games_with_bans.csv, lta_games_with_bans.csv
  - Team stats: out/LCK_split_stats.csv, out/LTA_split_stats.csv
- You have installed dependencies with:
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1
  - pip install -r requirements.txt

Notes:
- Team names must match exactly the values in the split stats CSV
- For LTA, the examples below use LTA North only

---

## 1) Train models

### 1.1 Train LCK model for all 2025 games

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_2025.joblib `
  --tournament-contains "2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
```

### 1.2 Train LCK model for Split 2 2025 only

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_split2_2025.joblib `
  --tournament-contains "LCK - Split 2 2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
```

### 1.3 Train LCK model for Split 2 2025 on a specific patch (example 15.10)

Only do this if you have enough rows after filtering.

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_split2_15_10.joblib `
  --tournament-contains "LCK - Split 2 2025" `
  --patch-contains "15.10" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
```

### 1.4 Train LTA North model for all 2025 games

This filters to LTA North tournaments only.

```powershell
python train_draft_model.py lta_games_with_bans.csv `
  --out models/lta_north_2025.joblib `
  --tournament-contains "LTA North" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
```

### 1.5 Train LTA North model for Split 2 2025 only

```powershell
python train_draft_model.py lta_games_with_bans.csv `
  --out models/lta_north_split2_2025.joblib `
  --tournament-contains "LTA North - Split 2 2025" `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --C 0.3 `
```

---

## 2) Run MCTS recommendations

All commands below assume:
- Standard pro draft order
- team1 is Blue side, team2 is Red side
- champion names are canonicalized via champion.json (examples: Renata Glasc maps to Renata, Wukong maps to MonkeyKing)

You can pass champions as display names, the code will canonicalize them.

### 2.1 LCK early draft (after one ban each)

Teams (from out/LCK_split_stats.csv, LCK_Split2_2025):
- T1
- Gen.G Esports

```powershell
python mcts_lol_draft.py `
  --model models/lck_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "T1" `
  --red-team "Gen.G Esports" `
  --blue-bans "Kalista" `
  --red-bans "Renata Glasc" `
  --iters 800 `
  --topk-ban 35 `
  --show 10
```

### 2.2 LCK before first pick (all 6 bans done)

```powershell
python mcts_lol_draft.py `
  --model models/lck_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "Dplus KIA" `
  --red-team "Hanwha Life Esports" `
  --blue-bans "Kalista|Renata Glasc|Vi" `
  --red-bans "Maokai|Rumble|Sejuani" `
  --iters 1200 `
  --topk-pick 40 `
  --show 12
```

### 2.3 LCK mid draft (after B1 and R1 R2)

This example is at the point where Blue is about to pick B2 or B3.

```powershell
python mcts_lol_draft.py `
  --model models/lck_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "KT Rolster" `
  --red-team "T1" `
  --blue-bans "Kalista|Renata Glasc|Vi" `
  --red-bans "Maokai|Rumble|Sejuani" `
  --blue-picks "Azir" `
  --red-picks "Orianna|Rakan" `
  --iters 1600 `
  --topk-pick 40 `
  --show 10
```

### 2.4 LTA North early draft example

Teams (from out/LTA_split_stats.csv, LTA_North_Split2_2025):
- Team Liquid
- Cloud9 Kia

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "Team Liquid" `
  --red-team "Cloud9 Kia" `
  --blue-bans "Kalista" `
  --red-bans "Renata Glasc" `
  --iters 800 `
  --topk-ban 35 `
  --show 10
```

### 2.5 LTA North before first pick

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "FlyQuest" `
  --red-team "Team Liquid" `
  --blue-bans "Kalista|Renata Glasc|Vi" `
  --red-bans "Maokai|Rumble|Sejuani" `
  --iters 1200 `
  --topk-pick 40 `
  --show 12
```

### 2.6 LTA North mid draft (after B1 and R1 R2)

```powershell
python mcts_lol_draft.py `
  --model models/lta_north_split2_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LTA_split_stats.csv `
  --split-key LTA_North_Split2_2025 `
  --blue-team "100 Thieves" `
  --red-team "Shopify Rebellion" `
  --blue-bans "Kalista|Renata Glasc|Vi" `
  --red-bans "Maokai|Rumble|Sejuani" `
  --blue-picks "Azir" `
  --red-picks "Orianna|Rakan" `
  --iters 1600 `
  --topk-pick 40 `
  --show 10
```

---

## 3) Team name reference

These team names are read from the split stats files. You can refer to other files for other teams from different regions.

### LCK teams for LCK_Split2_2025

- BNK FearX
- DN FREECS
- DRX
- Dplus KIA
- Gen.G Esports
- Hanwha Life Esports
- KT Rolster
- NONGSHIM RED FORCE
- OKSavingsBank BRION
- T1

### LTA North teams for LTA_North_Split2_2025

- 100 Thieves
- Cloud9 Kia
- Dignitas
- Disguised
- FlyQuest
- LYON
- Shopify Rebellion
- Team Liquid
