import numpy as np
import json

LOOKUP_PATH = "data/card_lookup.json"

class PokemonBattleEnv:
    def __init__(self, lookup_path=LOOKUP_PATH):
        with open(lookup_path, 'r', encoding='utf-8') as f:
            self.lookup = json.load(f)
        self.reset()

    def _get_card(self, card_id):
        cid_str = str(card_id)
        if cid_str in self.lookup:
            f = self.lookup[cid_str]["features"]
            return {
                "id": card_id,
                "name": self.lookup[cid_str]["name"],
                "category": f[0],
                "type": f[1],
                "hp": f[2],
                "max_hp": f[2],
                "retreat": f[3],
                "damage_1": max(f[4], 35.0),
                "damage_2": max(f[4] * 1.5, 70.0),
                "energy": 0
            }
        return {
            "id": 0, "name": "Empty", "category": 0, "type": 0,
            "hp": 60.0, "max_hp": 60.0, "retreat": 1.0,
            "damage_1": 30.0, "damage_2": 60.0, "energy": 0
        }

    def reset(self):
        # P1: Lucario Fighting beatdown line
        self.p1_active = self._get_card(677)  # Riolu
        self.p1_bench = [self._get_card(883), self._get_card(678)]
        self.p1_hand = [6, 6, 6, 6, 1121, 1086]
        self.p1_prizes = 6

        # P2: Opponent line
        self.p2_active = self._get_card(804)  # Piplup
        self.p2_bench = [self._get_card(744), self._get_card(805)]
        self.p2_hand = [3, 3, 3, 3, 1123]
        self.p2_prizes = 6

        self.turn_count = 0
        self.winner = None  # Explicit winner marker
        self.p1_energy_attached = False
        self.p2_energy_attached = False
        return self.get_state(is_p1=True)

    def get_state(self, is_p1=True):
        active = self.p1_active if is_p1 else self.p2_active
        opp_active = self.p2_active if is_p1 else self.p1_active
        bench = self.p1_bench if is_p1 else self.p2_bench
        opp_bench = self.p2_bench if is_p1 else self.p1_bench
        hand = self.p1_hand if is_p1 else self.p2_hand
        p_prizes = self.p1_prizes if is_p1 else self.p2_prizes
        o_prizes = self.p2_prizes if is_p1 else self.p1_prizes

        return {
            "player_active": active["id"] if active else 0,
            "player_bench": [p["id"] for p in bench],
            "opponent_active": opp_active["id"] if opp_active else 0,
            "opponent_bench": [p["id"] for p in opp_bench],
            "player_hand": hand,
            "player_prizes": p_prizes,
            "opponent_prizes": o_prizes
        }

    def get_legal_actions(self, is_p1=True):
        actions = []
        active = self.p1_active if is_p1 else self.p2_active
        bench = self.p1_bench if is_p1 else self.p2_bench
        hand = self.p1_hand if is_p1 else self.p2_hand
        attached_this_turn = self.p1_energy_attached if is_p1 else self.p2_energy_attached

        if active and active["energy"] >= 1:
            actions.append("ATTACK_1")
        if active and active["energy"] >= 2:
            actions.append("ATTACK_2")

        has_energy = any(self._get_card(cid)["category"] == 3.0 for cid in hand)
        if has_energy and not attached_this_turn:
            actions.append("ATTACH_ENERGY_ACTIVE")
            for idx in range(len(bench)):
                actions.append(f"ATTACH_ENERGY_BENCH_{idx}")

        has_basic = any(self._get_card(cid)["category"] == 1.0 for cid in hand)
        if has_basic and len(bench) < 5:
            actions.append("PLAY_BENCH_POKEMON")

        has_trainer = any(self._get_card(cid)["category"] == 2.0 for cid in hand)
        if has_trainer:
            actions.append("USE_TRAINER_CARD")

        actions.append("PASS_TURN")
        return actions

    def step(self, action, is_p1=True):
        reward = 0.0
        done = False

        active = self.p1_active if is_p1 else self.p2_active
        opp_active = self.p2_active if is_p1 else self.p1_active
        bench = self.p1_bench if is_p1 else self.p2_bench
        opp_bench = self.p2_bench if is_p1 else self.p1_bench
        hand = self.p1_hand if is_p1 else self.p2_hand

        if action.startswith("ATTACH_ENERGY"):
            for cid in hand:
                if self._get_card(cid)["category"] == 3.0:
                    hand.remove(cid)
                    if action == "ATTACH_ENERGY_ACTIVE" and active:
                        active["energy"] += 1
                    elif "BENCH_" in action:
                        b_idx = int(action.split("_")[-1])
                        if b_idx < len(bench):
                            bench[b_idx]["energy"] += 1

                    if is_p1:
                        self.p1_energy_attached = True
                    else:
                        self.p2_energy_attached = True
                    reward += 0.35
                    break

        elif action == "PLAY_BENCH_POKEMON":
            for cid in hand:
                if self._get_card(cid)["category"] == 1.0 and len(bench) < 5:
                    hand.remove(cid)
                    bench.append(self._get_card(cid))
                    reward += 0.2
                    break

        elif action == "USE_TRAINER_CARD":
            for cid in hand:
                if self._get_card(cid)["category"] == 2.0:
                    hand.remove(cid)
                    hand.append(6 if is_p1 else 3)
                    reward += 0.3
                    break

        elif action.startswith("ATTACK"):
            dmg = active["damage_2"] if action == "ATTACK_2" else active["damage_1"]
            opp_active["hp"] -= dmg
            reward += 0.6

            # Reset energy lock for next turn
            if is_p1:
                self.p1_energy_attached = False
            else:
                self.p2_energy_attached = False

            if opp_active["hp"] <= 0:
                if is_p1:
                    self.p1_prizes -= 2  # EX prize mechanic
                    reward += 2.5
                    if self.p1_prizes <= 0 or not opp_bench:
                        done = True
                        self.winner = "P1"
                        reward += 5.0
                    else:
                        self.p2_active = opp_bench.pop(0)
                        self.p2_active["energy"] = 0
                else:
                    self.p2_prizes -= 2
                    if self.p2_prizes <= 0 or not self.p1_bench:
                        done = True
                        self.winner = "P2"
                    else:
                        self.p1_active = self.p1_bench.pop(0)
                        self.p1_active["energy"] = 0

        elif action == "PASS_TURN":
            if is_p1:
                self.p1_energy_attached = False
                self.p1_hand.append(6)
            else:
                self.p2_energy_attached = False
                self.p2_hand.append(3)
            self.turn_count += 1

        if self.turn_count >= 50:
            done = True

        next_state = self.get_state(is_p1=is_p1)
        return next_state, reward, done, {}