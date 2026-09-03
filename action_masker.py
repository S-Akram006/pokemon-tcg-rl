import numpy as np

# Granular targeted master action space
ACTION_SPACE = [
    "ATTACK_1",
    "ATTACK_2",
    "ATTACH_ENERGY_ACTIVE",
    "ATTACH_ENERGY_BENCH_0",
    "ATTACH_ENERGY_BENCH_1",
    "ATTACH_ENERGY_BENCH_2",
    "ATTACH_ENERGY_BENCH_3",
    "ATTACH_ENERGY_BENCH_4",
    "PLAY_BENCH_POKEMON",
    "USE_TRAINER_CARD",
    "RETREAT",
    "PASS_TURN"
]

class ActionMasker:
    def __init__(self, action_space=ACTION_SPACE):
        self.action_space = action_space
        self.action_to_idx = {action: idx for idx, action in enumerate(action_space)}

    def get_action_mask(self, legal_actions):
        """
        Builds an exact binary mask vector for targeted legal actions.
        Matches exact action strings, with smart fallback to sub-prefixes.
        """
        mask = np.zeros(len(self.action_space), dtype=np.float32)
        
        for legal_act in legal_actions:
            # 1. Exact match
            if legal_act in self.action_to_idx:
                mask[self.action_to_idx[legal_act]] = 1.0
                continue
                
            # 2. Base prefix match fallback
            base = legal_act.split("_")[0]
            for full_act, idx in self.action_to_idx.items():
                if full_act.startswith(base):
                    mask[idx] = 1.0
                    
        # Guarantee PASS_TURN is valid if all masked
        if mask.sum() == 0.0:
            mask[self.action_to_idx["PASS_TURN"]] = 1.0
            
        return mask

    def mask_logits(self, logits, mask, invalid_penalty=-1e9):
        """Penalizes illegal action logits towards negative infinity."""
        return logits + (1.0 - mask) * invalid_penalty

if __name__ == "__main__":
    masker = ActionMasker()
    test_legal = ["ATTACK_1", "ATTACH_ENERGY_ACTIVE", "PASS_TURN"]
    mask = masker.get_action_mask(test_legal)
    print("=== TARGET-AWARE ACTION MASKER INITIALIZED ===")
    print(f"Action Space Size: {len(ACTION_SPACE)}")
    print(f"Legal Input: {test_legal}")
    print(f"Generated Binary Mask: {mask}")