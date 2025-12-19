import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from GDSS.models.layers import DenseGCNConv, MLP
from GDSS.utils.graph_utils import mask_adjs, pow_tensor
from GDSS.models.attention import  Attention
from GDSS.utils.graph_utils import mask_x, node_feature_to_matrix

class TimeEmbeddingReLu(nn.Module):
    """ Simple MLP that maps a scalar t ∈ [0,1] to a T‐dim feature of size `embed_dim`. """
    def __init__(self, embed_dim):
        super(TimeEmbeddingReLu, self).__init__()
        self.lin1 = nn.Linear(1, embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)
        # You can add BatchNorm or LayerNorm if desired.

    def forward(self, t_scalar):
        """
        t_scalar: either a float in [0,1], or a tensor of shape (B,) with values in [0,1].
        We will reshape it to (B,1), run through MLP, and return (B, embed_dim).
        """
        if not torch.is_tensor(t_scalar):
            # convert float → tensor
            t = torch.tensor([t_scalar], dtype=torch.float32, device=next(self.parameters()).device)
            t = t.unsqueeze(0)  # shape (1,1)
        else:
            # assume t_scalar is (B,) or (B,1)
            t = t_scalar.view(-1,1).to(next(self.parameters()).device)  # shape (B,1)
        h = F.relu(self.lin1(t))
        out = F.relu(self.lin2(h))  # shape (B, embed_dim)
        return out  # (B, embed_dim)
        
class TimeEmbeddingSiLu(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),                 # SiLU = swish
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
    def forward(self, t_scalar):
        if not torch.is_tensor(t_scalar):
            t = torch.tensor([t_scalar], dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        else:
            t = t_scalar.view(-1, 1).to(next(self.parameters()).device)
        return self.net(t)

import math
import torch
import torch.nn as nn



class BaselineNetworkLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

        super(BaselineNetworkLayer, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2*conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
                            use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)
        
    def forward(self, x, adj, flags):
    
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:,_,:,:])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)) , flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid  = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        
        
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

            elif _==self.num_layers-1:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        
        '''self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)   '''

    def forward(self, x, adj, flags=None):

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):

            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)

        '''self.mask = self.mask.to(score.device)
        score = score * self.mask'''

        #newly added, dynamically masking
        B, N, _ = score.shape
        mask = torch.ones(N, N, device=score.device) - torch.eye(N, device=score.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        score = score * mask
        
        score = mask_adjs(score, flags)
        assert torch.all(score[~flags.unsqueeze(1).expand(-1, N, N)] == 0), "Padded nodes not fully masked"
        return score



class AdaRMSNorm(nn.Module):
    """
    Adaptive Root Mean Square Layer Normalization.

    Args:
        dim (int): The feature dimension of the input.
        eps (float): A small value for numerical stability. Default: 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # This layer has no learnable parameters of its own.
        # The scaling parameter `gamma` is computed externally and passed in.

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor to normalize. Shape: (B, ..., dim).
            gamma (torch.Tensor): The adaptive scaling parameter. Shape: (B, dim).
        
        Returns:
            torch.Tensor: The normalized and scaled tensor.
        """
        # RMSNorm: Normalize by the root mean square of the features.
        # torch.rsqrt is reciprocal square root (1 / sqrt).
        norm_factor = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * norm_factor
        
        # Apply the adaptive scale (gamma).
        # We need to unsqueeze gamma to match the dimensions of x for broadcasting.
        # e.g., if x is (B, N, dim), gamma is (B, dim), so we make it (B, 1, dim).
        while len(gamma.shape) < len(x.shape):
            gamma = gamma.unsqueeze(1)
            
        return x_norm * gamma



class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for t in [0,1].
    dim must be even. Output: (B, dim).
    scale:
      - 2*pi  -> continuous-time, one base period over [0,1]
      - S     -> mimic S discrete steps (e.g., 1000)
    """
    def __init__(self, dim: int, scale: float = 2*math.pi):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb: dim must be even, got {dim}.")
        self.dim = dim
        self.scale = scale
        half = dim // 2
        # inv_freq[i] = 1 / (10000 ** (i / (half - 1)))
        inv_freq = torch.exp(-torch.arange(half, dtype=torch.float32) *
                             (math.log(10000.0) / (half - 1)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1), values in [0,1]
        t = t.view(-1) * self.scale                         # (B,)
        inv = self.inv_freq.to(device=t.device, dtype=t.dtype)
        angles = t[:, None] * inv[None, :]                  # (B, dim/2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, dim)


class AttentionLayer(torch.nn.Module):
    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim,
                 num_heads=4, conv='GCN', time_embed_dim=32): # <-- NEW: time_embed_dim arg

        super(AttentionLayer, self).__init__()

        # --- GNN and MLP modules (unchanged) ---
        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn.append(Attention(conv_input_dim, attn_dim, conv_output_dim,
                                       num_heads=num_heads, conv=conv))
        
        self.hidden_dim = 2*max(input_dim, output_dim)
        self.mlp = MLP(num_linears, 2*input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim,
                                 use_bn=False, activate_func=F.elu)

        # --- AdaNorm Setup ---
        # 1. Add the normalization layers
        self.norm_x = AdaRMSNorm(dim=conv_input_dim)
        self.norm_adj = AdaRMSNorm(dim=2*input_dim)
        
        # 2. Add MLPs to project the shared time_emb to the specific gammas needed
        self.time_mlp_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, conv_input_dim)
        )
        hidden_dim = time_embed_dim * 4
        self.time_mlp_adj = nn.Sequential(
            # 1. Project from the time embedding dim to a hidden dim
            nn.Linear(time_embed_dim, hidden_dim),
            # 2. Apply a non-linear activation function
            nn.SiLU(),
            # 3. Project from the hidden dim to the final output dim (for gamma)
            nn.Linear(hidden_dim, 2*input_dim)
        )

    
    def forward(self, x, adj, flags, time_emb):
        
        gamma_x = self.time_mlp_x(time_emb)
        gamma_adj = self.time_mlp_adj(time_emb)

        # Normalize node features before attention
        x_norm = self.norm_x(x, gamma_x)

        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x_norm, adj[:,_,:,:], flags) # Use normalized x
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)

        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0,2,3,1)], dim=-1)
        
        # Normalize adjacency features before the MLP
        mlp_in_norm = self.norm_adj(mlp_in, gamma_adj)

        shape = mlp_in_norm.shape
        mlp_out = self.mlp(mlp_in_norm.view(-1, shape[-1]))
        
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out

class DenoiseNetworkA(BaselineNetwork):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        # Call original __init__ to set up some base parameters
        super().__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears,
                         c_init, c_hid, c_final, adim, num_heads, conv)
        
        # --- Time Conditioning Setup ---
        self.time_embed_dim = 32 # This will be the size of the shared embedding
        self.time_embed = SinusoidalTimeEmbedding(self.time_embed_dim)

        # --- Re-initialize Layers to use the new AttentionLayer ---
        # The new AttentionLayer needs to know the time_embed_dim
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                  self.c_hid, self.num_heads, self.conv, self.time_embed_dim))
            elif _==self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                  self.c_final, self.num_heads, self.conv, self.time_embed_dim))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                  self.c_hid, self.num_heads, self.conv, self.time_embed_dim))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                         use_bn=False, activate_func=F.elu)

    # The forward pass now requires `t`
    def forward(self, x, adj, flags, t):
        
        # 1. Create the single, shared time embedding
        time_emb = self.time_embed(t)

        # 2. Process inputs (same as before)
        if adj.ndim == 4:
            adj = adj[:, 0]
        adjc = pow_tensor(adj, self.c_init)
        
        adj_list = [adjc]
        
        # 3. Pass the *same* time_emb to each layer
        for layer in self.layers:
            x, adjc = layer(x, adjc, flags, time_emb)
            adj_list.append(adjc)
            
        # 4. Final prediction (same as before)
        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        B, N, _, _ = adjs.shape
        score = self.final(adjs).view(B, N, N)
        
        mask = torch.ones(N, N, device=score.device) - torch.eye(N, device=score.device)
        score = score * mask.unsqueeze(0)
        score = mask_adjs(score, flags)
        
        return score
