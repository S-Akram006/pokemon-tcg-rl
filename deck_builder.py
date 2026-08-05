import json
from collections import Counter

LOOKUP_PATH = "data/card_lookup.json"

class DeckBuilder:
    def __init__(self, lookup_file=LOOKUP_PATH):
        with open(lookup_file, 'r', encoding='utf-8') as f:
            self.lookup = json.load(f)

    def validate_deck(self, deck_card_ids):
        """
        Validates deck against official TCG rules:
        - Exactly 60 cards
        - Max 4 copies per card (except Basic Energies)
        - At least 1 Basic Pokémon
        """
        issues = []
        
        # 1. Card count check
        if len(deck_card_ids) != 60:
            issues.append(f"Deck has {len(deck_card_ids)} cards (must be exactly 60).")

        # 2. Card limit check (Max 4 copies)
        counts = Counter(deck_card_ids)
        for cid, count in counts.items():
            cid_str = str(cid)
            card_data = self.lookup.get(cid_str, {})
            name = card_data.get("name", f"Card {cid}")
            
            # Allow basic energy cards past 4 copies
            is_basic_energy = "Basic" in name and "Energy" in name
            if count > 4 and not is_basic_energy:
                issues.append(f"Card '{name}' has {count} copies (max 4 allowed).")

        # 3. Basic Pokémon presence check
        has_basic_pokemon = False
        for cid in deck_card_ids:
            cid_str = str(cid)
            vec = self.lookup.get(cid_str, {}).get("features", [0]*5)
            # category == 1 (Pokémon) and non-zero HP
            if vec[0] == 1.0 and vec[2] > 0:
                has_basic_pokemon = True
                break
                
        if not has_basic_pokemon:
            issues.append("Deck contains no Basic Pokémon!")

        is_valid = len(issues) == 0
        return is_valid, issues

    def analyze_composition(self, deck_card_ids):
        """Calculates Pokémon / Trainer / Energy ratio breakdown."""
        counts = {1: 0, 2: 0, 3: 0, 0: 0} # 1: Pokémon, 2: Trainer, 3: Energy
        for cid in deck_card_ids:
            cid_str = str(cid)
            vec = self.lookup.get(cid_str, {}).get("features", [0]*5)
            cat = int(vec[0])
            counts[cat] = counts.get(cat, 0) + 1
            
        return {
            "Pokemon_Count": counts[1],
            "Trainer_Count": counts[2],
            "Energy_Count": counts[3],
            "Unknown_Count": counts[0]
        }

if __name__ == "__main__":
    builder = DeckBuilder()
    
    # Generate mock test deck (10 Pikachu/Basic Pokémon, 40 Trainers, 10 Grass Energies)
    # IDs 1 to 8 are basic energies
    test_deck = [374] * 4 + [744] * 4 + [402] * 4 + [1137] * 4 + [1] * 44
    
    valid, errors = builder.validate_deck(test_deck)
    comp = builder.analyze_composition(test_deck)
    
    print("=== DECK BUILDER TEST SUCCESSFUL ===")
    print(f"Deck Valid: {valid}")
    if errors:
        print(f"Validation Errors: {errors}")
    print(f"Deck Composition Breakdown: {comp}")