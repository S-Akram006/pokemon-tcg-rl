import torch
import torch.nn as nn
import torch.nn.functional as F

class PokemonTCGPolicy(nn.Module):
    def __init__(self, state_dim=65, action_dim=7):
        super(PokemonTCGPolicy, self).__init__()
        
        # Shared Feature Extraction Layers
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor Head (Policy output logits over 7 actions)
        self.actor_head = nn.Linear(128, action_dim)
        
        # Critic Head (Board value / Win probability estimation)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state, action_mask=None):
        """
        Forward pass through shared network, actor, and critic heads.
        """
        features = self.shared_net(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        
        # Apply action mask if provided (illegal actions get -1e9 logit)
        if action_mask is not None:
            invalid_penalty = -1e9
            logits = logits + (1.0 - action_mask) * invalid_penalty
            
        # Compute probabilities across actions using Softmax
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs, value

if __name__ == "__main__":
    # Test tensor inputs
    model = PokemonTCGPolicy(state_dim=65, action_dim=7)
    
    # Mock batch of 1 state tensor
    dummy_state = torch.randn(1, 65)
    dummy_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    
    probs, val = model(dummy_state, dummy_mask)
    
    print("=== MODEL ARCHITECTURE TEST SUCCESSFUL ===")
    print(f"Action Probabilities Output:\n{probs.detach().numpy()}")
    print(f"Critic State Value Output: {val.item():.4f}")