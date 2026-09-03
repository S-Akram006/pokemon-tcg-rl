import torch
import numpy as np
import os

from state_encoder import StateEncoder
from action_masker import ActionMasker, ACTION_SPACE
from model import PokemonTCGPolicy

MODEL_PATH = "data/best_model.pt"

class PokemonTCGAgent:
    def __init__(self, model_path=MODEL_PATH):
        self.state_encoder = StateEncoder()
        self.action_masker = ActionMasker()
        self.policy_net = PokemonTCGPolicy(state_dim=65, action_dim=len(ACTION_SPACE))
        
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"[Agent] Loaded weights from {model_path}")
        else:
            print("[Agent] Warning: No checkpoint found, using uninitialized weights.")
            
        self.policy_net.eval()

    def act(self, game_state, legal_actions):
        if not legal_actions:
            return "PASS_TURN"

        # 1. Encode board state
        state_vec = self.state_encoder.encode_state(game_state)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).float()

        # 2. Compute targeted action mask
        mask_vec = self.action_masker.get_action_mask(legal_actions)
        mask_tensor = torch.from_numpy(mask_vec).unsqueeze(0).float()

        # 3. Predict action probabilities
        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor, mask_tensor)

        # 4. Target-aware action resolution
        sorted_indices = torch.argsort(action_probs, dim=-1, descending=True).squeeze(0).tolist()
        
        for idx in sorted_indices:
            candidate_act = ACTION_SPACE[idx]
            # Exact match
            if candidate_act in legal_actions:
                return candidate_act
            # Prefix target match
            base_cand = candidate_act.split("_")[0]
            for legal in legal_actions:
                if legal.startswith(base_cand):
                    return legal

        return legal_actions[0]

_kaggle_agent = None

def my_agent(obs, configuration):
    global _kaggle_agent
    if _kaggle_agent is None:
        _kaggle_agent = PokemonTCGAgent()
        
    game_state = obs.get("state", {})
    legal_actions = obs.get("legal_actions", ["PASS_TURN"])
    return _kaggle_agent.act(game_state, legal_actions)

if __name__ == "__main__":
    agent = PokemonTCGAgent()
    test_state = {
        "player_active": 677,
        "player_bench": [883],
        "opponent_active": 804,
        "opponent_bench": [744],
        "player_hand": [6, 6, 1121],
        "player_prizes": 6,
        "opponent_prizes": 6
    }
    test_legal = ["ATTACH_ENERGY_ACTIVE", "ATTACH_ENERGY_BENCH_0", "USE_TRAINER_CARD", "PASS_TURN"]
    chosen = agent.act(test_state, test_legal)
    print("=== TARGET-AWARE SUBMISSION INFERENCE TEST ===")
    print(f"Legal Moves: {test_legal}")
    print(f"Agent Selected: {chosen}")