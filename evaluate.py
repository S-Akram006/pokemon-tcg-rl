from submission import PokemonTCGAgent
from baseline_agent import BaselineRuleAgent
from game_env import PokemonBattleEnv

def run_evaluation_match(num_games=100):
    rl_agent = PokemonTCGAgent()
    baseline_agent = BaselineRuleAgent()

    rl_wins = 0
    baseline_wins = 0
    draws = 0

    print(f"=== BENCHMARK: 12-ACTION REGULATION MATCHES ({num_games} GAMES) ===")
    print("RL Agent (P1)  VS  Targeted Baseline Agent (P2)\n")

    for game in range(1, num_games + 1):
        env = PokemonBattleEnv()
        done = False
        turn_limit = 50
        current_turn = 0

        while not done and current_turn < turn_limit:
            current_turn += 1

            # Player 1 (RL Agent)
            p1_active = True
            p1_actions = 0
            while p1_active and not done and p1_actions < 5:
                p1_actions += 1
                state = env.get_state(is_p1=True)
                legal = env.get_legal_actions(is_p1=True)
                action = rl_agent.act(state, legal)
                _, _, done, _ = env.step(action, is_p1=True)
                if action.startswith("ATTACK") or action == "PASS_TURN" or done:
                    p1_active = False

            if done:
                break

            # Player 2 (Baseline Agent)
            p2_active = True
            p2_actions = 0
            while p2_active and not done and p2_actions < 5:
                p2_actions += 1
                state = env.get_state(is_p1=False)
                legal = env.get_legal_actions(is_p1=False)
                action = baseline_agent.select_action(state, legal)
                _, _, done, _ = env.step(action, is_p1=False)
                if action.startswith("ATTACK") or action == "PASS_TURN" or done:
                    p2_active = False

            if done:
                break

        # Check official victory condition
        if env.winner == "P1" or env.p1_prizes < env.p2_prizes:
            rl_wins += 1
        elif env.winner == "P2" or env.p2_prizes < env.p1_prizes:
            baseline_wins += 1
        else:
            draws += 1

    win_rate = (rl_wins / num_games) * 100
    print("=== REGULATION BENCHMARK RESULTS ===")
    print(f"Total Games:   {num_games}")
    print(f"RL Agent Wins: {rl_wins}")
    print(f"Baseline Wins: {baseline_wins}")
    print(f"Draws:         {draws}")
    print(f"RL Win Rate:   {win_rate:.1f}%")

if __name__ == "__main__":
    run_evaluation_match(100)