import numpy as np
from submission import PokemonTCGAgent
from baseline_agent import BaselineRuleAgent

class SimulatedGameEnvironment:
    """Tracks persistent game state across turns for evaluation."""
    def __init__(self):
        self.p1_prizes = 6
        self.p2_prizes = 6
        
    def get_state(self, is_p1=True):
        return {
            "player_active": 374,
            "player_bench": [744],
            "opponent_active": 804,
            "opponent_bench": [],
            "player_hand": [1, 2, 3],
            "player_prizes": self.p1_prizes if is_p1 else self.p2_prizes,
            "opponent_prizes": self.p2_prizes if is_p1 else self.p1_prizes
        }

def run_evaluation_match(num_games=100):
    rl_agent = PokemonTCGAgent()
    baseline_agent = BaselineRuleAgent()

    rl_wins = 0
    baseline_wins = 0
    draws = 0

    print(f"=== LAUNCHING HEAD-TO-HEAD BENCHMARK ({num_games} GAMES) ===")
    print("RL Agent (submission.py)  VS  Baseline Rule Agent (baseline_agent.py)\n")

    for game in range(1, num_games + 1):
        env = SimulatedGameEnvironment()
        turn_limit = 30
        
        for turn in range(turn_limit):
            legal_actions = ["ATTACH_ENERGY", "ATTACK", "PASS_TURN"]

            # --- RL Agent Turn (Player 1) ---
            p1_state = env.get_state(is_p1=True)
            rl_action = rl_agent.act(p1_state, legal_actions)
            
            if "ATTACK" in rl_action:
                env.p2_prizes -= 1
                
            if env.p2_prizes <= 0:
                rl_wins += 1
                break

            # --- Baseline Agent Turn (Player 2) ---
            p2_state = env.get_state(is_p1=False)
            base_action = baseline_agent.select_action(p2_state, legal_actions)
            
            if "ATTACK" in base_action:
                env.p1_prizes -= 1
                
            if env.p1_prizes <= 0:
                baseline_wins += 1
                break
        else:
            # Tiebreaker based on remaining prize cards
            if env.p2_prizes < env.p1_prizes:
                rl_wins += 1
            elif env.p1_prizes < env.p2_prizes:
                baseline_wins += 1
            else:
                draws += 1

    win_rate = (rl_wins / num_games) * 100
    print("=== FIXED BENCHMARK EVALUATION RESULTS ===")
    print(f"Total Matches Played: {num_games}")
    print(f"RL Agent Wins:       {rl_wins}")
    print(f"Baseline Wins:       {baseline_wins}")
    print(f"Draws:               {draws}")
    print(f"RL Win Rate:         {win_rate:.1f}%")

if __name__ == "__main__":
    run_evaluation_match(100)