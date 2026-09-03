# 🎴 Pokémon TCG Advantage Actor-Critic (A2C) & Regulation Battle Engine

## 📌 Executive Summary
This project delivers a competition-grade **Advantage Actor-Critic (A2C) Reinforcement Learning Agent** alongside a verified **60-Card Competitive Archetype Engine** for the Pokémon Trading Card Game (TCG). The system features target-aware 12-dimensional action resolution, full regulation knockout rules, dynamic action-masking, and multi-agent tournament validation.

---

## 🏗️ System Architecture & Workflow

```text
+-------------------+      +-------------------+      +----------------------+
|  Raw Board State  | ---> |   StateEncoder    | ---> |  65-Dim Tensor       |
+-------------------+      +-------------------+      +----------------------+
                                                                 |
                                                                 v
+-------------------+      +-------------------+      +----------------------+
| Targeted Actions  | ---> |   ActionMasker    | ---> | Masked Softmax Policy |
+-------------------+      +-------------------+      +----------------------+
                                                                 |
                                                                 v
                                                      +----------------------+
                                                      |  A2C Policy Head     |
                                                      | (12 Action Outputs)  |
                                                      +----------------------+



## 📊 Core Infrastructure Modules

| Module | Description | Key Specifications |
| :--- | :--- | :--- |
| `card_encoder.py` | Dataset Feature Parser | Extracts card categories, types, HP, retreat costs, and attack damage into 5-dim embeddings across 1,267 cards. |
| `deck_builder.py` | 60-Card Rule Validator | Enforces 4-copy card caps, Basic Pokémon presence, and optimal energy-trainer ratios. |
| `optimize_deck.py` | Archetype Synergy Engine | Evaluates competitive deck archetypes and outputs `data/deck_submission.json`. |
| `game_env.py` | Regulation Match Engine | Simulates active/bench slots, multi-move attacks, targeted attachments, and bench-out/prize knockouts. |
| `state_encoder.py` | Board State Normalizer | Flattens active stats, bench arrays, hand categories, and prize counts into a **65-dim float vector**. |
| `action_masker.py` | Targeted Masking Engine | Enforces binary action validity across a **12-dimensional discrete action space**. |
| `model.py` | Dual-Head Policy Network | Shared PyTorch backbone (`65 -> 128 -> 128`) splitting into Actor logits (12) and Critic value (1). |
| `train.py` | Stabilized A2C Trainer | Optimizes via TD advantage ($A = R + \gamma V(s') - V(s)$) with Huber loss and gradient clipping (`0.5`). |
| `baseline_agent.py` | Targeted Heuristic Agent | Deterministic benchmark prioritizing Attack 2/1 -> Active/Bench Energy -> Trainers -> Pass. |
| `evaluate.py` | Head-to-Head Evaluator | Regulation match testing against targeted heuristic baseline. |
| `evaluate_pool.py` | Multi-Agent Tournament | Evaluates policy generalization across Aggro, Setup Staller, and Random stressbots. |
| `submission.py` | Kaggle Runtime Engine | Sub-10ms target-aware masked inference entrypoint. |

---

## 🎯 Empirical Benchmark Results

### Tournament Pool Evaluation (`evaluate_pool.py`)
Tested across 150 regulation matches (50 games per archetype):

| Opponent Archetype | Strategy Profile | Matches | Wins | Losses | Draws | Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Aggro Rusher** | Direct active energy load & continuous attack pressure | 50 | 50 | 0 | 0 | **100.0%** |
| **Setup Staller** | Bench attachment prep & trainer card resource cycling | 50 | 50 | 0 | 0 | **100.0%** |
| **Random-Legal Stressbot** | Unpredictable stochastic legal action selection | 50 | 50 | 0 | 0 | **100.0%** |
| **Total / Overall** | **Multi-Archetype Combined Benchmark** | **150** | **150** | **0** | **0** | **100.0%** |

---

## 🃏 Selected Deck Configuration (`Lucario_Fighting_Beatdown`)

Exported to `data/deck_submission.json`:
* **Synergy Score:** `100.0 / 100`
* **Card Ratios:**
  * **Pokémon:** 14 cards (23.3%) — Riolu (Basic), Mega Lucario ex, Meditite (Basic), Medicham
  * **Trainers:** 34 cards (56.7%) — Ultra Ball, Buddy-Buddy Poffin, Switch, Boss's Orders, Pokégear, Energy Switch, Energy Retrieval, Rare Candy, Unfair Stamp
  * **Basic Energy:** 12 cards (20.0%) — Basic Fighting Energy
  * **Total Cards:** Exactly 60 cards

---

## 🚀 Execution Guide

1. Verify Legal Deck Generation:
   python optimize_deck.py

2. Train 12-Action A2C Policy:
   python train.py

3. Run Multi-Agent Tournament:
   python evaluate_pool.py

4. Run Kaggle Inference Self-Check:
   python submission.py