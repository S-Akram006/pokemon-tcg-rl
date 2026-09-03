import torch
import torch.nn as nn
import torch.nn.functional as F

class PokemonTCGPolicy(nn.Module):
    def __init__(self, state_dim=65, action_dim=12):
        super(PokemonTCGPolicy, self).__init__()
        
        # Feature Extraction Backbone
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor Head: Outputs logits across all 12 targeted actions
        self.actor_head = nn.Linear(128, action_dim)
        
        # Critic Head: Scalar state-value estimator
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state, action_mask=None):
        features = self.shared_net(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        
        if action_mask is not None:
            invalid_penalty = -1e9
            logits = logits + (1.0 - action_mask) * invalid_penalty
            
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, value

if __name__ == "__main__":
    from action_masker import ACTION_SPACE
    model = PokemonTCGPolicy(state_dim=65, action_dim=len(ACTION_SPACE))
    dummy_state = torch.randn(1, 65)
    dummy_mask = torch.ones(1, len(ACTION_SPACE))
    probs, val = model(dummy_state, dummy_mask)
    
    print("=== MODEL POLICY (12-ACTION TARGETED) INITIALIZED ===")
    print(f"Action Dimension: {len(ACTION_SPACE)}")
    print(f"Output Probabilities Shape: {probs.shape}")
    print(f"Sample Probabilities Sum: {probs.sum().item():.2f}")