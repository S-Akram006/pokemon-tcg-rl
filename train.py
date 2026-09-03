import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from state_encoder import StateEncoder
from action_masker import ActionMasker, ACTION_SPACE
from model import PokemonTCGPolicy
from game_env import PokemonBattleEnv
from baseline_agent import BaselineRuleAgent

def train_a2c():
    state_encoder = StateEncoder()
    action_masker = ActionMasker()
    env = PokemonBattleEnv()
    baseline = BaselineRuleAgent()
    
    policy_net = PokemonTCGPolicy(state_dim=65, action_dim=len(ACTION_SPACE))
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)

    num_episodes = 300
    gamma = 0.95
    print(f"=== TRAINING 12-ACTION REGULATION A2C AGENT ({num_episodes} EPISODES) ===")

    for episode in range(1, num_episodes + 1):
        raw_state = env.reset()
        done = False
        
        episode_reward = 0.0
        policy_losses = []
        value_losses = []
        entropies = []

        turn_count = 0
        while not done and turn_count < 40:
            turn_count += 1
            
            # --- Player 1 (RL Agent) Turn ---
            legal_actions = env.get_legal_actions(is_p1=True)
            state_vec = state_encoder.encode_state(raw_state)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).float()
            
            mask_vec = action_masker.get_action_mask(legal_actions)
            mask_tensor = torch.from_numpy(mask_vec).unsqueeze(0).float()

            action_probs, state_value = policy_net(state_tensor, mask_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            selected_action = ACTION_SPACE[action_idx.item()]
            
            next_state, reward, done, _ = env.step(selected_action, is_p1=True)
            episode_reward += reward

            if done:
                next_val = 0.0
            else:
                next_vec = state_encoder.encode_state(next_state)
                next_tensor = torch.from_numpy(next_vec).unsqueeze(0).float()
                with torch.no_grad():
                    _, next_val_t = policy_net(next_tensor)
                    next_val = next_val_t.item()

            td_target = reward + gamma * next_val
            advantage = td_target - state_value.item()

            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()

            policy_losses.append(-log_prob * advantage)
            v_loss = F.smooth_l1_loss(state_value.squeeze(-1), torch.tensor([td_target], dtype=torch.float32))
            value_losses.append(v_loss)
            entropies.append(entropy)

            raw_state = next_state

            if done:
                break

            # --- Player 2 (Baseline Agent) Turn ---
            p2_legal = env.get_legal_actions(is_p1=False)
            p2_state = env.get_state(is_p1=False)
            p2_action = baseline.select_action(p2_state, p2_legal)
            raw_state, _, done, _ = env.step(p2_action, is_p1=False)

        # Backpropagation
        if policy_losses:
            optimizer.zero_grad()
            total_policy_loss = torch.stack(policy_losses).sum()
            total_value_loss = torch.stack(value_losses).sum()
            total_entropy = torch.stack(entropies).sum()

            total_loss = total_policy_loss + 0.5 * total_value_loss - 0.01 * total_entropy
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
            optimizer.step()

        if episode % 30 == 0 or episode == 1:
            print(f"Episode {episode:03d}/{num_episodes} | Ep Reward: {episode_reward:5.2f} | Loss: {total_loss.item():6.4f} | Turns: {turn_count}")

    checkpoint_path = "data/best_model.pt"
    torch.save(policy_net.state_dict(), checkpoint_path)
    print(f"\n=== TRAINING FINISHED: Saved 12-Action Weights to {checkpoint_path} ===")

if __name__ == "__main__":
    train_a2c()