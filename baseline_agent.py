import numpy as np

class BaselineRuleAgent:
    def __init__(self, name="RuleBasedBaseline"):
        self.name = name

    def select_action(self, game_state, available_actions):
        """
        Selects an action based on deterministic priority rules:
        1. Attach Energy to Active Pokémon
        2. Play Basic Pokémon to Bench
        3. Attack opponent
        4. Fallback to first valid action
        """
        if not available_actions:
            return "PASS"

        # Action mapping heuristic priorities
        energy_actions = [a for a in available_actions if "ATTACH_ENERGY" in a]
        bench_actions = [a for a in available_actions if "BENCH_POKEMON" in a]
        attack_actions = [a for a in available_actions if "ATTACK" in a]

        # 1. Prioritize Energy attachment
        if energy_actions:
            return energy_actions[0]

        # 2. Prioritize Bencing Pokémon
        if bench_actions:
            return bench_actions[0]

        # 3. Prioritize Attacking
        if attack_actions:
            return attack_actions[0]

        # Default fallback
        return available_actions[0]

if __name__ == "__main__":
    agent = BaselineRuleAgent()

    # Mock game turn actions for testing
    possible_actions = [
        "BENCH_POKEMON_744",
        "ATTACH_ENERGY_3",
        "ATTACK_MOVE_1",
        "PASS_TURN"
    ]

    selected = agent.select_action({}, possible_actions)
    print(f"=== BASELINE AGENT TEST SUCCESSFUL ===")
    print(f"Available Actions: {possible_actions}")
    print(f"Selected Priority Action: {selected}")