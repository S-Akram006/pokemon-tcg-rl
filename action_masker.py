import numpy as np

# Total master action space definition
ACTION_SPACE = [
    "ATTACH_ENERGY",
    "PLAY_BENCH_POKEMON",
    "EVOLVE_POKEMON",
    "USE_TRAINER_CARD",
    "ATTACK",
    "RETREAT",
    "PASS_TURN"
]

class ActionMasker:
    def __init__(self, action_space=ACTION_SPACE):
        self.action_space = action_space
        self.action_to_idx = {action: idx for idx, action in enumerate(action_space)}

    def get_action_mask(self, legal_actions):
        """
        Generates a binary vector (1 for legal, 0 for illegal) 
        matching the master action space size.
        """
        mask = np.zeros(len(self.action_space), dtype=np.float32)
        
        for action in legal_actions:
            # Match base action types
            base_action = action.split("_")[0] if "_" in action else action
            for full_action, idx in self.action_to_idx.items():
                if full_action.startswith(base_action):
                    mask[idx] = 1.0
                    
        return mask

    def mask_logits(self, logits, mask, invalid_penalty=-1e9):
        """
        Applies action mask to network logits (sets illegal actions to large negative numbers).
        """
        return logits + (1.0 - mask) * invalid_penalty

if __name__ == "__main__":
    masker = ActionMasker()
    
    # Example turn legal actions from game engine
    sample_legal_actions = ["ATTACH_ENERGY", "ATTACK", "PASS_TURN"]
    
    binary_mask = masker.get_action_mask(sample_legal_actions)
    
    print("=== ACTION MASKER TEST SUCCESSFUL ===")
    print(f"Master Action Space ({len(ACTION_SPACE)} actions): {ACTION_SPACE}")
    print(f"Legal Input Actions: {sample_legal_actions}")
    print(f"Generated Mask Vector: {binary_mask}")