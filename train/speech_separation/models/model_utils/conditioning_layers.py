import torch.nn as nn

class FiLM(nn.Module):
    """
    Scale (gamma) and shift (beta) applied to mixture features, 
    conditioned on speaker embedding
    """
    def __init__(self, embed_dim, channel_dim):
        super().__init__()
        # Two linear layers to map embedding -> gamma, beta
        self.gamma_fc = nn.Linear(embed_dim, channel_dim)
        self.beta_fc = nn.Linear(embed_dim, channel_dim)
    def forward(self, x, embed):
        """
        x:     [B, channel_dim, S]
        embed: [B, embed_dim]      
        Returns: [B, channel_dim, S], 
        """
        # 1) produce gamma, beta from embedding => [B, channel_dim]
        gamma = self.gamma_fc(embed)   # [B, channel_dim]
        beta  = self.beta_fc(embed)
        # 2) broadcast along time dimension S
        gamma = gamma.unsqueeze(-1)    # => [B, channel_dim, 1]
        beta  = beta.unsqueeze(-1)
        # 3) apply FiLM: x_new = gamma * x + beta
        return gamma * x + beta

