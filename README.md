# LoL Pro Draft Winrate Modeling and MCTS Draft Recommender

> **Examples included**
>
> This repository includes an `examples` file containing ready-to-run command-line examples
> demonstrating common workflows such as model training and MCTS draft recommendation.


This repository implements a research-grade League of Legends draft analysis system inspired by the paper  
**"Winning Is Not Everything: Enhancing Draft Recommendation in MOBA Games"** (arXiv:1806.10130).

The project combines:
- A linear win probability model trained on professional drafts
- A Monte Carlo Tree Search (MCTS / UCT) engine that simulates full drafts with picks and bans
- Team-specific champion priors for realistic professional draft recommendations

Given a partial draft state, the system recommends the best next pick or ban.

---

## Repository Structure

```text
.
├── train_draft_model.py
├── mcts_lol_draft.py
├── champion.json
│
├── lck_games_with_bans.csv
├── lec_games_with_bans.csv
├── lta_games_with_bans.csv
├── lpl_games_with_bans.csv
│
├── out/
│   ├── LCK_split_stats.csv
│   ├── LTA_split_stats.csv
│   ├── LEC_split_stats.csv
│   └── LPL_split_stats.csv
│
├── models/
│   └── lck_2025.joblib
│
├── examples.txt
└── README.md

## Requirements

This project requires Python 3.9 or newer.

Required Python packages:
- numpy
- pandas
- scipy
- scikit-learn
- joblib

Optional but recommended:
- tqdm

---

## Installation

It is strongly recommended to use a virtual environment.

### PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Formats

### Match Draft Data

Files named like:
(region)_games_with_bans.csv

These files are used to train the winrate model.

Required columns:

match_id, patch, region, tournament,
team1_name, team2_name,
team1_drafted_champions, team2_drafted_champions,
team1_banned_champions, team2_banned_champions,
winner

Notes:
- team1 is Blue side
- winner = 1 means Blue wins
- winner = 0 means Red wins
- champion lists are pipe separated

---

### Team Champion Statistics

Files located in the out/ directory:
out/(REGION)_split_stats.csv

Required columns:

region, split_key, team_id, team_name, team_games,
champion_id, champion_name, champ_games, champ_pct

These files are used by MCTS to construct team specific priors.

---

## Training a Model

You can train region specific, split specific, or patch specific models using train_draft_model.py.

### Example (PowerShell)

```powershell
python train_draft_model.py lck_games_with_bans.csv `
  --out models/lck_2025.joblib `
  --tournament-contains "2025" `
  --pair-min-count 9999 `
  --C 0.3
```

Important options:
- --tournament-contains restricts training to a season or split
- --patch-contains restricts training to a patch
- --pair-min-count 9999 disables pair synergy features
- --counter-min-count 9999 disables counter features
- --C controls logistic regression regularization

The training script evaluates multiple random seeds and saves the best model.

---

## Using Monte Carlo Tree Search

The MCTS engine:
- Uses the standard professional draft order
- Simulates full drafts via rollouts
- Evaluates terminal drafts using the trained model
- Uses the UCB1 selection rule

### Example MCTS Command

```powershell
python mcts_lol_draft.py `
  --model models/lck_2025.joblib `
  --champion-json champion.json `
  --patch 15.10 `
  --team-stats out/LCK_split_stats.csv `
  --split-key LCK_Split2_2025 `
  --blue-team "T1" `
  --red-team "Gen.G Esports" `
  --blue-bans "Kalista" `
  --red-bans "Renata Glasc" `
  --iters 800 `
  --show 10
```

---

## Draft Order

The system uses the standard professional tournament draft format:

1. Ban phase 1 (6 bans)
2. Pick phase 1 (6 picks)
3. Ban phase 2 (4 bans)
4. Pick phase 2 (4 picks)

Total actions per draft: 20.
Each team has 5 picks and 5 bans.

---

## Champion Canonicalization

Champion names are normalized using champion.json.

Examples:
- Renata Glasc maps to Renata
- Kai'Sa maps to Kaisa
- Miss Fortune maps to MissFortune
- Wukong maps to MonkeyKing

All training and inference uses the same canonical identifiers.

---

## Common Pitfalls

Team names passed to MCTS must exactly match the values in out/(REGION)_split_stats.csv.
If they do not match, MCTS falls back to global champion priors.

Patch alignment also matters. Models trained on broad time ranges average across metas.

---

## Reproducibility

- Use --seed in MCTS for deterministic runs
- Increase --iters for more stable recommendations
- Reduce --topk-pick and --topk-ban to reduce randomness

---

## License and Attribution

This project is intended for research and educational use.

Inspired by academic work on MOBA draft optimization and professional esports analytics.
