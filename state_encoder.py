import json
import numpy as np

LOOKUP_PATH = "data/card_lookup.json"

class StateEncoder:
    def __init__(self, lookup_file=LOOKUP_PATH):
        # Load pre-computed card feature vectors
        with open(lookup_file, 'r', encoding='utf-8') as f:
            self.lookup = json.load(f)
            
        # Standard zero vector for empty slots (HP=0, Type=0, etc.)
        self.empty_feature = [0.0, 0.0, 0.0, 0.0, 0.0]

    def get_card_vector(self, card_id):
        """Fetches 5-element feature array for a given Card ID."""
        card_str = str(card_id)
        if card_str in self.lookup:
            return self.lookup[card_str]["features"]
        return self.empty_feature

    def encode_board_zone(self, active_id, bench_ids):
        """Encodes active card + up to 5 bench card slots."""
        zone_vector = []
        
        # 1. Active Pokémon
        zone_vector.extend(self.get_card_vector(active_id))
        
        # 2. Bench (Pad or truncate to exactly 5 slots)
        for i in range(5):
            if i < len(bench_ids):
                zone_vector.extend(self.get_card_vector(bench_ids[i]))
            else:
                zone_vector.extend(self.empty_feature)
                
        return zone_vector

    def encode_state(self, game_state):
        """
        Takes raw game state dictionary and converts it to a 65-element numpy array.
        """
        tensor = []
        
        # 1. Player Board State (30 floats)
        player_board = self.encode_board_zone(
            game_state.get('player_active'), 
            game_state.get('player_bench', [])
        )
        tensor.extend(player_board)
        
        # 2. Opponent Board State (30 floats)
        opponent_board = self.encode_board_zone(
            game_state.get('opponent_active'), 
            game_state.get('opponent_bench', [])
        )
        tensor.extend(opponent_board)
        
        # 3. Hand Category Counts (3 floats)
        hand_ids = game_state.get('player_hand', [])
        categories = [0, 0, 0] # [Pokémon, Trainer, Energy]
        for cid in hand_ids:
            vec = self.get_card_vector(cid)
            cat = int(vec[0]) # category code
            if 1 <= cat <= 3:
                categories[cat - 1] += 1
        tensor.extend([float(c) for c in categories])
        
        # 4. Prize Cards Meta (2 floats)
        p_prizes = float(game_state.get('player_prizes', 6))
        o_prizes = float(game_state.get('opponent_prizes', 6))
        tensor.extend([p_prizes, o_prizes])
        
        return np.array(tensor, dtype=np.float32)

if __name__ == "__main__":
    encoder = StateEncoder()
    
    # Mock game state for testing
    sample_state = {
        "player_active": 374,          # Electrike
        "player_bench": [744, 402],    # Jynx, Smoliv
        "opponent_active": 804,        # Piplup
        "opponent_bench": [],
        "player_hand": [1, 2, 3, 1137],# 3 Energies + 1 Trainer
        "player_prizes": 6,
        "opponent_prizes": 5
    }
    
    state_tensor = encoder.encode_state(sample_state)
    print("=== STATE ENCODER TEST SUCCESSFUL ===")
    print(f"Encoded Vector Shape: {state_tensor.shape}")
    print(f"Sample Encoded Vector (first 10 values):\n{state_tensor[:10]}")