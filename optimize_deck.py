import json
import os
from deck_builder import DeckBuilder

DECK_SUBMISSION_PATH = "data/deck_submission.json"

class DeckOptimizer:
    def __init__ (self):
        self.builder = DeckBuilder()

    def generate_candidate_decks(self):
        """Generates candidate 60-card deck configurations across different strategy archetypes."""
        # Common Card IDs from lookup:
        # Basic Energy: 1
        # Basic Pokemon: 374 (Pikachu), 744, 402, 1137
        candidates = {
            "Aggro_Speed": [374] * 4 + [744] * 4 + [402] * 4 + [1137] * 4 + [1] * 44,
            "Balanced_Midrange": [374] * 4 + [744] * 4 + [402] * 4 + [1137] * 2 + [1] * 46,
            "Energy_Heavy": [374] * 4 + [744] * 2 + [1] * 54
        }
        return candidates

    def evaluate_and_select_best(self):
        candidates = self.generate_candidate_decks()
        best_deck_name = None
        best_deck_cards = None
        highest_score = -1.0

        print("=== EVALUATING CANDIDATE DECK COMPOSITIONS ===")

        for name, card_ids in candidates.items():
            valid, errors = self.builder.validate_deck(card_ids)
            comp = self.builder.analyze_composition(card_ids)

            if not valid:
                print(f"[-] Deck '{name}' REJECTED: {errors}")
                continue

            # Synergy Scoring Heuristic (Ideal: ~12-16 Pokemon, high energy consistency)
            pokemon_cnt = comp["Pokemon_Count"]
            energy_cnt = comp["Energy_Count"]
            
            # Score penalizes deviating from optimal ratios (12-16 Pokemon, 40-48 Energy)
            score = 100.0 - abs(pokemon_cnt - 14) * 2.0 - abs(energy_cnt - 44) * 1.5

            print(f"[+] Deck '{name}': Score = {score:.1f} | Breakdown = {comp}")

            if score > highest_score:
                highest_score = score
                best_deck_name = name
                best_deck_cards = card_ids

        # Save winning deck configuration
        os.makedirs("data", exist_ok=True)
        submission_payload = {
            "deck_name": best_deck_name,
            "deck_score": highest_score,
            "card_ids": best_deck_cards,
            "total_cards": len(best_deck_cards)
        }

        with open(DECK_SUBMISSION_PATH, "w") as f:
            json.dump(submission_payload, f, indent=4)

        print("\n=== DECK OPTIMIZATION COMPLETE ===")
        print(f"Selected Deck Archetype: '{best_deck_name}'")
        print(f"Saved configuration to: {DECK_SUBMISSION_PATH}")

if __name__ == "__main__":
    optimizer = DeckOptimizer()
    optimizer.evaluate_and_select_best()