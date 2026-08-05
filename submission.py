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
        
        # Load pre-trained weights if checkpoint exists
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"[Agent] Loaded weights from {model_path}")
        else:
            print("[Agent] Warning: No checkpoint found, using uninitialized weights.")
            
        self.policy_net.eval() # Set model to evaluation mode

    def act(self, game_state, legal_actions):
        """
        Takes raw game state dict & legal actions list.
        Returns selected action string.
        """
        if not legal_actions:
            return "PASS_TURN"

        # 1. Encode 65-dim state vector
        state_vec = self.state_encoder.encode_state(game_state)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).float()

        # 2. Generate binary action mask
        mask_vec = self.action_masker.get_action_mask(legal_actions)
        mask_tensor = torch.from_numpy(mask_vec).unsqueeze(0).float()

        # 3. Model inference (no gradient computation needed)
        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor, mask_tensor)
            
        # 4. Map top probability output to legal action
        best_action_idx = torch.argmax(action_probs, dim=-1).item()
        base_selected = ACTION_SPACE[best_action_idx]

        # Match base action back to specific legal action list item
        for act in legal_actions:
            if act.startswith(base_selected):
                return act

        # Fallback to first legal action
        return legal_actions[0]

# Global agent instance for Kaggle runtime
_kaggle_agent = None

def my_agent(obs, configuration):
    """
    Standard Kaggle environment agent entrypoint interface.
    """
    global _kaggle_agent
    if _kaggle_agent is None:
        _kaggle_agent = PokemonTCGAgent()
        
    game_state = obs.get("state", {})
    legal_actions = obs.get("legal_actions", ["PASS_TURN"])
    
    return _kaggle_agent.act(game_state, legal_actions)

if __name__ == "__main__":
    agent = PokemonTCGAgent()
    
    # Mock inference test state
    test_state = {
        "player_active": 374,
        "player_bench": [744],
        "opponent_active": 804,
        "opponent_bench": [],
        "player_hand": [1, 2, 3],
        "player_prizes": 6,
        "opponent_prizes": 6
    }
    test_legal_actions = ["ATTACH_ENERGY", "ATTACK", "PASS_TURN"]
    
    selected_action = agent.act(test_state, test_legal_actions)
    
    print("=== SUBMISSION ENGINE INFERENCE TEST SUCCESSFUL ===")
    print(f"Input Legal Actions: {test_legal_actions}")
    print(f"Agent Selected Action: {selected_action}")