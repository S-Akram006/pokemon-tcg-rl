import torch
import torch.optim as optim
import numpy as np

from state_encoder import StateEncoder
from action_masker import ActionMasker, ACTION_SPACE
from model import PokemonTCGPolicy

def train():
    state_encoder = StateEncoder()
    action_masker = ActionMasker()
    policy_net = PokemonTCGPolicy(state_dim=65, action_dim=len(ACTION_SPACE))
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    num_episodes = 200
    print(f"=== RE-TRAINING RL POLICY ({num_episodes} EPISODES WITH INCENTIVIZED REWARDS) ===")

    for episode in range(1, num_episodes + 1):
        log_probs = []
        rewards = []
        
        for turn in range(15):
            # Mock turn state
            dummy_state = {
                "player_active": 374,
                "player_bench": [744],
                "opponent_active": 804,
                "opponent_bench": [],
                "player_hand": [1, 2, 3],
                "player_prizes": 6,
                "opponent_prizes": 6
            }
            
            state_tensor = torch.from_numpy(state_encoder.encode_state(dummy_state)).unsqueeze(0).float()
            legal_actions = ["ATTACH_ENERGY", "ATTACK", "PASS_TURN"]
            mask_tensor = torch.from_numpy(action_masker.get_action_mask(legal_actions)).unsqueeze(0).float()

            action_probs, _ = policy_net(state_tensor, mask_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            selected_action = ACTION_SPACE[action_idx.item()]
            
            log_prob = dist.log_prob(action_idx)

            # --- INCENTIVIZED REWARD STRUCTURE ---
            if selected_action == "ATTACK":
                reward = 1.0
            elif selected_action == "ATTACH_ENERGY":
                reward = 0.5
            else:  # PASS_TURN
                reward = -0.5
            
            log_probs.append(log_prob)
            rewards.append(reward)

        # Policy gradient calculation
        discounted_rewards = []
        gamma = 0.99
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        policy_loss = []
        for lp, r in zip(log_probs, discounted_rewards):
            policy_loss.append(-lp * r)

        optimizer.zero_grad()
        total_loss = torch.cat(policy_loss).sum()
        total_loss.backward()
        optimizer.step()

        if episode % 50 == 0 or episode == 1:
            print(f"Episode {episode:03d}/{num_episodes} | Total Reward: {sum(rewards):.2f} | Loss: {total_loss.item():.4f}")

    checkpoint_path = "data/best_model.pt"
    torch.save(policy_net.state_dict(), checkpoint_path)
    print(f"\n=== RETRAINING COMPLETE: Model Checkpoint Saved to {checkpoint_path} ===")

if __name__ == "__main__":
    train()