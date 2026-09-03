class BaselineRuleAgent:
    def __init__(self, name="RuleBasedBaseline"):
        self.name = name

    def select_action(self, game_state, available_actions):
        """
        Deterministic priority order for targeted actions:
        1. Attack 2 (heavy hit) -> Attack 1 (standard hit)
        2. Attach Energy to Active Pokemon -> Attach to Bench
        3. Use Trainer Card (resource acceleration)
        4. Play Pokemon to Bench
        5. Pass Turn
        """
        if not available_actions:
            return "PASS_TURN"

        # 1. Attacks (prioritize stronger attack)
        if "ATTACK_2" in available_actions:
            return "ATTACK_2"
        if "ATTACK_1" in available_actions:
            return "ATTACK_1"

        # 2. Attach Energy (prioritize Active to power attacks)
        if "ATTACH_ENERGY_ACTIVE" in available_actions:
            return "ATTACH_ENERGY_ACTIVE"
        bench_energy = [a for a in available_actions if a.startswith("ATTACH_ENERGY_BENCH")]
        if bench_energy:
            return bench_energy[0]

        # 3. Use Trainer Card
        if "USE_TRAINER_CARD" in available_actions:
            return "USE_TRAINER_CARD"

        # 4. Play Bench Pokemon
        if "PLAY_BENCH_POKEMON" in available_actions:
            return "PLAY_BENCH_POKEMON"

        # Default fallback
        return available_actions[-1] if "PASS_TURN" in available_actions else available_actions[0]

if __name__ == "__main__":
    agent = BaselineRuleAgent()
    possible = ["ATTACH_ENERGY_ACTIVE", "USE_TRAINER_CARD", "PASS_TURN"]
    print("Baseline selected:", agent.select_action({}, possible))