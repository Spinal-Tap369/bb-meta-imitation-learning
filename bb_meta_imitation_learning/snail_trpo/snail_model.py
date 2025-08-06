# snail_trpo/snail_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNEncoder

# CombinedEncoder
class CombinedEncoder(nn.Module):
    """
    Combined encoder that mirrors the paper:
      - Preprocesses only the image part (first 3 channels) using the CNN architecture from Duan et al. (2016)
      - The remaining 3 channels (previous action, previous reward, termination bit) are averaged over H and W.
      - The 256-d image embedding and the 3-d scalar vector are concatenated (resulting in 259 dimensions).
      - Optionally, this combined vector can be projected (via a linear layer) back to a desired dimension.
    """
    def __init__(self, base_dim=256):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_channels=3)
        # Concatenated dimension is 256+3=259.
        if base_dim != 259:
            self.proj = nn.Linear(259, base_dim)
        else:
            self.proj = None

    def forward(self, obs):
        image_part = obs[:, :3, :, :]   # shape: (B, 3, H, W)
        scalar_part = obs[:, 3:, :, :]   # shape: (B, 3, H, W)
        img_embed = self.cnn_encoder(image_part)  # (B, 256)
        scalar_vec = scalar_part.mean(dim=[2, 3])  # (B, 3)
        combined = torch.cat([img_embed, scalar_vec], dim=1)  # (B, 259)
        if self.proj is not None:
            return F.relu(self.proj(combined))
        else:
            return combined

class DenseBlock(nn.Module):
    def __init__(self, in_dim, dilation, filters):
        super().__init__()
        self.out_dim = in_dim + filters
        self.conv_f = nn.Conv1d(in_dim, filters, kernel_size=2, dilation=dilation, padding=dilation)
        self.conv_g = nn.Conv1d(in_dim, filters, kernel_size=2, dilation=dilation, padding=dilation)

    def forward(self, x):
        B, C, T = x.shape
        xf = self.conv_f(x)[:, :, :T]
        xg = self.conv_g(x)[:, :, :T]
        act = torch.tanh(xf) * torch.sigmoid(xg)
        return torch.cat([x, act], dim=1)

class TCBlock(nn.Module):
    def __init__(self, in_dim, seq_len, filters):
        super().__init__()
        ms = math.ceil(math.log2(seq_len + 1))
        blocks = []
        cur_dim = in_dim
        for i in range(ms):
            blk = DenseBlock(cur_dim, dilation=2**i, filters=filters)
            blocks.append(blk)
            cur_dim = blk.out_dim
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = cur_dim

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SnailAttentionBlock(nn.Module):
    def __init__(self, in_dim, key_size, value_size):
        super().__init__()
        self.linear_query = nn.Linear(in_dim, key_size)
        self.linear_keys = nn.Linear(in_dim, key_size)
        self.linear_values = nn.Linear(in_dim, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.out_dim = in_dim + value_size

    def forward(self, x):
        B, C, T = x.shape
        x_bt = x.permute(0, 2, 1)
        query = self.linear_query(x_bt)
        keys = self.linear_keys(x_bt)
        values = self.linear_values(x_bt)
        logits = torch.bmm(query, keys.transpose(1, 2)) / self.sqrt_key_size
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits.masked_fill_(mask, -float('inf'))
        attn_weights = F.softmax(logits, dim=-1)
        read = torch.bmm(attn_weights, values)
        out = torch.cat([x_bt, read], dim=-1)
        return out.permute(0, 2, 1).contiguous()

class SNAILPolicyValueNet(nn.Module):
    def __init__(
        self,
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16, 
        value_filters=16,
        seq_len=800
    ):
        super().__init__()
        self.action_dim = action_dim
        self.combined_encoder = CombinedEncoder(base_dim=base_dim)
        self.base_dim = base_dim
        self.seq_len = seq_len

        # Policy branch.
        self.policy_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=policy_filters)
        self.attn1 = SnailAttentionBlock(
            in_dim=self.policy_block1.out_dim,
            key_size=policy_attn_dim,
            value_size=policy_attn_dim
        )
        self.policy_block2 = TCBlock(in_dim=self.attn1.out_dim, seq_len=seq_len, filters=policy_filters)
        self.attn2 = SnailAttentionBlock(
            in_dim=self.policy_block2.out_dim,
            key_size=policy_attn_dim,
            value_size=policy_attn_dim
        )
        self.policy_out_dim = self.attn2.out_dim
        self.policy_head = nn.Conv1d(self.policy_out_dim, action_dim, kernel_size=1)

        # Value branch.
        self.value_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=value_filters)
        self.value_block2 = TCBlock(in_dim=self.value_block1.out_dim, seq_len=seq_len, filters=value_filters)
        self.value_out_dim = self.value_block2.out_dim
        self.value_head = nn.Conv1d(self.value_out_dim, 1, kernel_size=1)

        # For TRPO_FO compatibility
        self.num_layers = 1
        self.hidden_size = base_dim

    def forward(self, x):
        """
        x: Tensor of shape (B, T, 6, H, W)
        Returns:
            policy_logits: (B, T, action_dim)
            values: (B, T)
        """
        B, T, C, H, W = x.shape
        x2 = x.view(B * T, C, H, W)
        feats = self.combined_encoder(x2)  # (B*T, base_dim)
        feats_1D = feats.view(B, T, self.base_dim).permute(0, 2, 1).contiguous()
        # Policy branch.
        p_out = self.policy_block1(feats_1D)
        p_out = self.attn1(p_out)
        p_out = self.policy_block2(p_out)
        p_out = self.attn2(p_out)
        logits_1D = self.policy_head(p_out)
        policy_logits = logits_1D.permute(0, 2, 1).contiguous()
        # Value branch.
        v_out = self.value_block1(feats_1D)
        v_out = self.value_block2(v_out)
        v_1D = self.value_head(v_out)
        values = v_1D.squeeze(1).permute(0, 1).contiguous()
        return policy_logits, values

    def forward_rollout(self, x):
        return self.forward(x)

    def act_single_step(self, x):
        logits, values = self.forward(x)
        return logits[:, -1, :], values[:, -1]

    def forward_with_state(self, x, dummy_state=None):
        logits, values = self.forward_rollout(x)
        return logits, values, None

    def policy_parameters(self):
        return list(self.parameters())

    def value_parameters(self):
        return list(self.parameters())
