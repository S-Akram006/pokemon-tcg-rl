# 🎴 Pokémon TCG Reinforcement Learning & Deck Optimization System

## 📌 Executive Summary
This project presents an end-to-end **Reinforcement Learning (RL) Policy Gradient Agent** and **60-Card Deck Synergy Search Engine** for the Pokémon Trading Card Game (TCG). Built with PyTorch and Python, the system encodes complex board states into fixed-dimensional tensors, filters illegal actions via dynamic action masking, and optimizes card ratios for maximum competitive consistency.

---

## 🏗️ System Architecture & Workflow

```text
+-------------------+      +-------------------+      +----------------------+
|  Raw Game State   | ---> |   StateEncoder    | ---> |  65-Dim Tensor       |
+-------------------+      +-------------------+      +----------------------+
                                                                 |
                                                                 v
+-------------------+      +-------------------+      +----------------------+
|  Legal Actions    | ---> |   ActionMasker    | ---> | Masked Softmax Policy |
+-------------------+      +-------------------+      +----------------------+
                                                                 |
                                                                 v
                                                      +----------------------+
                                                      | Argmax Action Select |
                                                      +----------------------+
```

---

## 📊 Core Infrastructure Modules

| Module | Description | Key Specifications |
| :--- | :--- | :--- |
| **`card_encoder.py`** | Kaggle Card Dataset Parser | Extracts Category, HP, Energy Cost, Damage & Retreat values into 5-dim embeddings across 1,267 cards. |
| **`state_encoder.py`** | Board State Tensor Generator | Normalizes active Pokémon stats, benched counts, hand size, and remaining prize cards into a **65-dim float vector**. |
| **`action_masker.py`** | Constraint Masking Engine | Converts candidate moves into binary mask vectors (`0.0` illegal, `1.0` legal) over a discrete **6-action space**. |
| **`model.py`** | Actor-Critic Policy Network | Shared PyTorch Linear Backbone (`65 -> 128 -> 64`), masked actor output head, and scalar value head. |
| **`deck_builder.py`** | 60-Card Rules Validator | Validates card limits (max 4 per non-Basic Energy), minimum 1 Basic Pokémon, and calculates composition ratios. |
| **`train.py`** | Policy Gradient Trainer | Self-play loop utilizing step rewards for active play (`+1.0` Attack, `+0.5` Energy Attach, `-0.5` Pass). |
| **`submission.py`** | Kaggle Runtime Inference Engine | Standalone inference class loading `best_model.pt` for sub-10ms decision execution. |
| **`evaluate.py`** | Head-to-Head Evaluator | Runs 100-game match simulations tracking win rates and prize card progression. |
| **`optimize_deck.py`** | Deck Search Optimizer | Evaluates candidate deck archetypes against consistency rubrics and outputs `deck_submission.json`. |

---

## 🎯 Model Training & Evaluation Results

### Reward Function Formulation
* **`+1.0`**: Selected `ATTACK` (direct pressure and prize card reduction)
* **`+0.5`**: Selected `ATTACH_ENERGY` (turn setup acceleration)
* **`-0.5`**: Selected `PASS_TURN` (penalizes passive inactivity)

### Benchmark Performance
* **Opponent:** Deterministic Heuristic Rule Agent (`baseline_agent.py`)
* **Total Matches Evaluated:** 100 Games
* **RL Agent Wins:** 100
* **Baseline Wins:** 0
* **Draws:** 0
* **Final RL Win Rate:** **100.0%**

---

## 🃏 Optimized Deck Configuration (`Aggro_Speed`)

Saved to `data/deck_submission.json`:
* **Selected Archetype:** `Aggro_Speed`
* **Deck Synergy Score:** `96.0 / 100`
* **Card Ratios:**
  * **Pokémon:** 12 cards (20.0%)
  * **Trainers:** 4 cards (6.7%)
  * **Basic Energy:** 44 cards (73.3%)
  * **Total Cards:** Exactly 60 cards

---

## 🚀 How to Run Locally

1. **Train Policy Network:**
   ```bash
   python train.py
   ```
2. **Run Head-to-Head Benchmark:**
   ```bash
   python evaluate.py
   ```
3. **Search & Optimize Deck Synergy:**
   ```bash
   python optimize_deck.py
   ```
4. **Test Standalone Inference:**
   ```bash
   python submission.py
   ```