import random
from submission import PokemonTCGAgent
from game_env import PokemonBattleEnv

class AggroRusher:
    def select_action(self, state, legal):
        for act in ["ATTACK_2", "ATTACK_1", "ATTACH_ENERGY_ACTIVE"]:
            if act in legal:
                return act
        return legal[0]

class SetupStaller:
    def select_action(self, state, legal):
        # Prioritizes bench & trainer resources before attacking
        bench_energy = [a for a in legal if a.startswith("ATTACH_ENERGY_BENCH")]
        if bench_energy:
            return bench_energy[0]
        if "USE_TRAINER_CARD" in legal:
            return "USE_TRAINER_CARD"
        if "PLAY_BENCH_POKEMON" in legal:
            return "PLAY_BENCH_POKEMON"
        if "ATTACK_2" in legal:
            return "ATTACK_2"
        if "ATTACK_1" in legal:
            return "ATTACK_1"
        return legal[-1] if "PASS_TURN" in legal else legal[0]

class RandomLegalBot:
    def select_action(self, state, legal):
        return random.choice(legal)

def run_tournament(games_per_opponent=50):
    rl_agent = PokemonTCGAgent()
    opponents = [
        ("Aggro Rusher", AggroRusher()),
        ("Setup Staller", SetupStaller()),
        ("Random-Legal Stressbot", RandomLegalBot())
    ]

    print(f"=== TOURNAMENT POOL BENCHMARK ({len(opponents) * games_per_opponent} TOTAL MATCHES) ===")

    total_wins = 0
    total_matches = 0

    for opp_name, bot in opponents:
        wins = 0
        losses = 0
        draws = 0

        for _ in range(games_per_opponent):
            env = PokemonBattleEnv()
            done = False
            turns = 0

            while not done and turns < 50:
                turns += 1

                # RL Turn
                p1_active = True
                p1_steps = 0
                while p1_active and not done and p1_steps < 5:
                    p1_steps += 1
                    s = env.get_state(is_p1=True)
                    l = env.get_legal_actions(is_p1=True)
                    act = rl_agent.act(s, l)
                    _, _, done, _ = env.step(act, is_p1=True)
                    if act.startswith("ATTACK") or act == "PASS_TURN" or done:
                        p1_active = False

                if done:
                    break

                # Bot Turn
                p2_active = True
                p2_steps = 0
                while p2_active and not done and p2_steps < 5:
                    p2_steps += 1
                    s = env.get_state(is_p1=False)
                    l = env.get_legal_actions(is_p1=False)
                    act = bot.select_action(s, l)
                    _, _, done, _ = env.step(act, is_p1=False)
                    if act.startswith("ATTACK") or act == "PASS_TURN" or done:
                        p2_active = False

                if done:
                    break

            if env.winner == "P1" or env.p1_prizes < env.p2_prizes:
                wins += 1
            elif env.winner == "P2" or env.p2_prizes < env.p1_prizes:
                losses += 1
            else:
                draws += 1

        total_wins += wins
        total_matches += games_per_opponent
        win_rate = (wins / games_per_opponent) * 100
        print(f"vs {opp_name:22s} | Wins: {wins:2d} | Losses: {losses:2d} | Draws: {draws:2d} | Win Rate: {win_rate:5.1f}%")

    overall_wr = (total_wins / total_matches) * 100
    print(f"\n>>> OVERALL POOL WIN RATE: {overall_wr:.1f}% across {total_matches} games")

if __name__ == "__main__":
    run_tournament(50)