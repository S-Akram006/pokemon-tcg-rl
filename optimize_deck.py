import json
import os
from deck_builder import DeckBuilder

DECK_SUBMISSION_PATH = "data/deck_submission.json"

class DeckOptimizer:
    def __init__(self):
        self.builder = DeckBuilder()

    def generate_candidate_decks(self):
        """
        Builds realistic 60-card competitive configurations using valid card IDs:
        - 677: Riolu (Basic Fighting Pokémon, 80 HP)
        - 678: Mega Lucario ex (Fighting Pokémon, 340 HP, 270 Dmg)
        - 883: Meditite (Basic Fighting Pokémon, 70 HP)
        - 884: Medicham (Fighting Pokémon, 120 HP, 150 Dmg)
        - 6:   Basic Fighting Energy
        - Meta Trainers (Search, Draw, Switching, Disruption)
        """
        trainers = (
            [1086] * 4 +  # Buddy-Buddy Poffin
            [1121] * 4 +  # Ultra Ball
            [1123] * 4 +  # Switch
            [1182] * 4 +  # Boss's Orders
            [1116] * 4 +  # Energy Switch
            [1118] * 4 +  # Energy Retrieval
            [1079] * 4 +  # Rare Candy
            [1122] * 4    # Pokégear 3.0
        )  # 32 Trainers

        candidates = {
            "Lucario_Fighting_Beatdown": (
                [677] * 4 +   # Riolu
                [678] * 4 +   # Mega Lucario ex
                [883] * 3 +   # Meditite
                [884] * 3 +   # Medicham
                trainers + [1080] * 2 +  # 34 Trainers (including 2 Unfair Stamp)
                [6] * 12      # 12 Basic Fighting Energy
            ),
            "Lucario_Hyper_Aggro": (
                [677] * 4 +
                [678] * 4 +
                [883] * 4 +
                trainers + [1080] * 4 +  # 36 Trainers
                [6] * 12
            ),
            "Lucario_Midrange": (
                [677] * 4 +
                [678] * 4 +
                [883] * 4 +
                [884] * 4 +
                trainers +
                [6] * 12
            )
        }
        return candidates

    def evaluate_and_select_best(self):
        candidates = self.generate_candidate_decks()
        best_deck_name = None
        best_deck_cards = None
        highest_score = -1.0

        print("=== EVALUATING COMPETITIVE CANDIDATE DECKS ===")

        for name, card_ids in candidates.items():
            valid, errors = self.builder.validate_deck(card_ids)
            comp = self.builder.analyze_composition(card_ids)

            if not valid:
                print(f"[-] Deck '{name}' REJECTED: {errors}")
                continue

            # Competitive Scoring: Target 12-16 Pokemon, 30-36 Trainers, 10-14 Energy
            pokemon_cnt = comp["Pokemon_Count"]
            trainer_cnt = comp["Trainer_Count"]
            energy_cnt = comp["Energy_Count"]

            score = (
                100.0
                - abs(pokemon_cnt - 14) * 2.5
                - abs(trainer_cnt - 34) * 1.5
                - abs(energy_cnt - 12) * 2.0
            )

            print(f"[+] Deck '{name}': Score = {score:.1f} | Breakdown = {comp}")

            if score > highest_score:
                highest_score = score
                best_deck_name = name
                best_deck_cards = card_ids

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