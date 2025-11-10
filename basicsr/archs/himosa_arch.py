'''
An official Pytorch impl of `Transcending the Limit of Local Window: 
Advanced Super-Resolution Transformer with Adaptive Token Dictionary`.

Arxiv: 'https://arxiv.org/abs/2401.08209'
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.layers import DropPath
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper
from basicsr.archs.HAT_arch import CAB
from inspect import isfunction

from basicsr.utils.registry import ARCH_REGISTRY
from tqdm import tqdm
from thop import profile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from einops import rearrange

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features
        
    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class ConvGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = dwconv(hidden_features, kernel_size=kernel_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, x_size)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class MSConvGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = dwconv(hidden_features, kernel_size=3)
        self.dwconv2 = dwconv(hidden_features, kernel_size=5)
        self.dwconv3 = dwconv(hidden_features, kernel_size=7)
        self.act = act_layer()
        self.squeeze = nn.Linear(3*hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x1 = self.act(self.dwconv(x, x_size))
        x2 = self.act(self.dwconv2(x, x_size))
        x3 = self.act(self.dwconv3(x, x_size))
        x = self.squeeze(torch.cat([x1, x2, x3], dim=-1)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ExpertGather(nn.Module):
    def __init__(self, heads: int, input_dim: int, head_dim: int):
        super().__init__()

        self.heads, self.input_dim, self.head_dim = heads, input_dim, head_dim
        self.W = nn.Parameter(torch.empty(heads, input_dim, head_dim))

        self.reset_parameters()

    def forward(self, X: torch.Tensor, ind: torch.Tensor):
        # output has shape [B,heads,K,J]
        B, N, c = X.shape
        _, heads, K = ind.shape
        
        index=ind.reshape(B, heads*K)[...,None].expand(-1,-1,c)
        X_gathered = torch.gather(X, dim=1, index=index).reshape(B, heads, K, c)
        # 
        out = torch.einsum('beki, eij->bekj', X_gathered, self.W)
        return out
    
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.head_dim)
        nn.init.uniform_(self.W, -bound, bound)
        
class ExpertScatter(nn.Module):
    def __init__(self, heads: int, head_dim: int, out_dim: int):
        super().__init__()

        self.heads, self.head_dim, self.out_dim = heads, head_dim, out_dim
        self.W = nn.Parameter(torch.empty(heads, head_dim, out_dim))

        self.reset_parameters()

    def forward(self, Y: torch.Tensor, Ind: torch.Tensor, T: int):
        B, heads, K, head_dim = Y.shape
        # Ind shape [B, heads, K]

        X_prescatter = torch.einsum('bekj, eji->beki', Y, self.W)

        I = X_prescatter.shape[-1]
        scattered = torch.zeros(B, T, I, device=Y.device, dtype=Y.dtype)
        Ind = Ind[..., None].expand(-1,-1,-1,I)
        scattered.scatter_add_(1, Ind.reshape(B, heads*K, I), X_prescatter.reshape(B, heads*K, I))
        return scattered
    
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.out_dim*self.heads)
        nn.init.uniform_(self.W, -bound, bound)

class WMOSA(nn.Module):
    """ Spatial-Channel Correlation.
    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, base_win_size, window_size, sparsity, num_heads, head_dim=None, value_drop=0., proj_drop=0.):

        super().__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads
        self.sparsity = sparsity
        self.include_first = True
        self.num_tokens = num_heads # number of tokens in the token dictionary
        self.n_iter = 3

        # feature projection
        self.proj = nn.Linear(dim, dim)

        # dropout
        self.proj_drop = nn.Dropout(proj_drop)
        self.value_drop = nn.Dropout(value_drop)

        # base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = 0.01

        # sparse spatial attention
        self.r = torch.nn.Sequential(
            torch.nn.Linear(self.dim, self.num_heads, bias=False),
            torch.nn.Sigmoid()
        )

        self.QKV = ExpertGather(self.num_heads, self.dim, 3*head_dim)
        self.O = ExpertScatter(self.num_heads, head_dim, self.dim)

    def get_topk_includefirst(self, logits: torch.Tensor):
        # logits: [B, L, h]
        B, L, h = logits.shape
        k = int(L // self.sparsity)
        k = min(max(k, 2), L)           # at least 2, at most L
        k1 = k - 1                      
        
        # topk idx
        tail_vals, tail_idx = torch.topk(logits[:, 1:, :], k=k1, dim=1)
        # # tail_vals: [B, k1, E], tail_idx: [B, k1, E]
        tail_idx = tail_idx + 1         # now in [1 … T-1]
        
        
        # # random idx                   
        # tail_idx = torch.randint(1, L, (B, k1, h), device=logits.device)
        # tail_vals = logits.gather(dim=1, index=tail_idx)

        # # sequential idx
        # tail_idx = torch.arange(1, k, device=logits.device).view(1, k1, 1).expand(B, k1, h)
        # tail_vals = logits.gather(dim=1, index=tail_idx)
        
        first_vals = logits[:, :1, :]   # [B, 1, E]
        first_idx  = torch.zeros(B, 1, h, dtype=torch.long, device=logits.device)

        vals = torch.cat([first_vals, tail_vals], dim=1)  # [B, k, E]
        idxs = torch.cat([first_idx,  tail_idx ], dim=1)  # [B, k, E]

        return vals.transpose(1,2), idxs.transpose(1,2), k

    def get_topk(self, x: torch.Tensor):
        """
        Selects tokens for the experts
        Input:
            x - inputs shape [B, T, heads]
        Output (3-tuple):
            - scores of the tokens for given router [B, heads, k]
            - indices of selected tokens in the original sequence [B, heads, k]
            - selected number of tokens
        """
        B, L, h = x.shape

        logits = self.r(x)

        if self.include_first:
            return self.get_topk_includefirst(logits)
        
        k = int(L // self.sparsity)
        k = min(max(k, 2), L) # 2 is the minimum number of tokens to select from

        logits_topk = logits.topk(dim=1, k=k) # [b, k, heads]
        topk_I = logits_topk.indices.transpose(1,2) # [b, heads, k]
        topk_vals = logits_topk.values.transpose(1,2) # [b, heads, k]

        return topk_vals, topk_I, k
    
    def check_image_size(self, x, win_size):
        x = x.permute(0,3,1,2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = x.permute(0,2,3,1).contiguous()
        return x
    
    def channel_self_correlation(self, x):
                
        b, h, w, c = x.shape
        # compute correlation map
        x = self.channel_block(x.permute(0, 3, 1, 2))
        return x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

    def sparse_self_correlation(self, x):
        B, L, C = x.shape
        topk_vals, topk_I, k = self.get_topk(x)
        
        Q, K, V = self.QKV(x, topk_I).chunk(3, dim=-1) # [B, heads, k, self.h_prim]
        attn = F.scaled_dot_product_attention( # unsqueezes to make mask head specifc
            Q.unsqueeze(2), K.unsqueeze(2), V.unsqueeze(2)
        ).squeeze(2)
        attn = attn * topk_vals.unsqueeze(-1) # [B, heads, k, h_prim]
        # output
        x = self.O(attn, topk_I, L)
        
        return x

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        xB,xH,xW,xC = x.shape
        shortcut = x

        # window partition
        x = window_partition(x, self.window_size)
        # x_channel = self.channel_self_correlation(x)  # xB xL xC
        # x_channel = window_reverse(x_channel, (self.window_size[0], self.window_size[1]), xH, xW) # xB xH xW xC
        
        # spatial self-correlation (S-SC)
        x = x.view(-1, self.window_size[0]*self.window_size[1], xC)
        x_sparse = self.sparse_self_correlation(x)
        x_sparse = x_sparse.view(-1, self.window_size[0], self.window_size[1], xC)
        x_sparse = window_reverse(x_sparse, (self.window_size[0],self.window_size[1]), xH, xW)  # xB xH xW xC

        # x_out = shortcut + x_sparse #+ x_channel * self.scale  # residual connection
        x_out = x_sparse
        x = self.proj_drop(self.proj(x_out))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class HIMOSALayer(nn.Module):
    r"""
    ATD Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        is_last (bool): True if this layer is the last of a ATD Block. Default: False 
    """

    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 num_heads,
                 base_win_size,
                 window_size,
                 mosa_heads,
                 sparsity,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 drop=0.,
                 value_drop=0.,
                 drop_path=0.,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mosa_heads = mosa_heads
        self.sparsity = sparsity
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.conv_scale = 0.01

        # check window size
        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]):
            assert window_size[0] % base_win_size[0] == 0, "please ensure the window size is smaller than or divisible by the base window size"
            assert window_size[1] % base_win_size[1] == 0, "please ensure the window size is smaller than or divisible by the base window size"


        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.correlation = WMOSA(
            dim, base_win_size=base_win_size, window_size=self.window_size, sparsity=sparsity, num_heads=mosa_heads, head_dim = 48,
            value_drop=value_drop, proj_drop=drop)
        self.channel_block = CAB(dim, compress_ratio = 3, squeeze_factor=30)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvGLU(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=act_layer)

    def check_image_size(self, x, win_size):
        x = x.permute(0,3,1,2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = x.permute(0,2,3,1).contiguous()
        return x
    
    def forward(self, x, x_size, win_size):
        h, w = x_size
        b, n, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)  # b, h, w, c

        # channel_attention
        conv_x = self.channel_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)


        # padding
        x = self.check_image_size(x, win_size)
        _, H_pad, W_pad, _ = x.shape # shape after padding
        
        x = self.correlation(x) 
        # unpad
        x = x[:, :h, :w, :].contiguous()
        # reshape
        x = x.view(b, h * w, c)

        x = shortcut + self.drop_path(x) + conv_x*self.conv_scale
        x = self.drop_path(self.convffn(self.norm2(x), x_size)) + x
        
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicBlock(nn.Module):
    """ A basic ATD Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 base_win_size,
                 mosa_heads,
                 sparsity,
                 convffn_kernel_size,
                 reducted_dim,
                 mlp_ratio=4.,
                 drop=0., 
                 value_drop=0.,
                 drop_path=0., 
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, 
                 hier_win_ratios=[0.5,1,2,4,6,8]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx
        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                HIMOSALayer(
                    dim=dim,
                    idx=i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    base_win_size=base_win_size,
                    window_size=(self.win_hs[i], self.win_ws[i]),
                    mosa_heads=mosa_heads,
                    sparsity=sparsity[i],
                    convffn_kernel_size=convffn_kernel_size,
                    reducted_dim=reducted_dim, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop, 
                    value_drop=value_drop,       
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    is_last=i == depth-1,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        b, n, c = x.shape
        for i, layer in enumerate(self.layers):
            # adjust the value of idx_checkpoint to change the number of layers processed by checkpoint_wrapper
            # increase the value of idx_checkpoint could save more GPU memory footprint but slow down the training
            # idx_checkpoint need to be set as at least 4 for eight 24G GPU when training ATD
            idx_checkpoint = 4
            if self.use_checkpoint and self.idx < idx_checkpoint:
                layer = checkpoint_wrapper(layer, offload_to_cpu=False)
            x = layer(x, x_size, (self.win_hs[i], self.win_ws[i]))

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class HIMOSAB(nn.Module):
    """Hierarchical Mixture of Sparse Attention Block (ATDB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 base_win_size,
                 mosa_heads,
                 sparsity,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 drop=0., 
                 value_drop=0.,
                 drop_path=0.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv', 
                 hier_win_ratios=[0.5,1,2,4,6,8]):
        super(HIMOSAB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            base_win_size=base_win_size,
            mosa_heads=mosa_heads,
            sparsity=sparsity,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            drop=drop, 
            value_drop=value_drop,
            drop_path=drop_path,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            hier_win_ratios=hier_win_ratios
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class HIMOSAv1(nn.Module):
    r""".

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 base_win_size=(8, 8),
                 mosa_heads=(16, 8, 8, 8),
                 sparsity=(4, 8, 12, 16),
                 reducted_dim=4,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 drop_rate=0., 
                 value_drop_rate=0., 
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 hier_win_ratios=[0.5,1,2,4,6,8],
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.base_win_size = base_win_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # # relative position index
        # relative_position_index_SA = self.calculate_rpi_sa()
        # self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build Residual Adaptive Token Dictionary Blocks (ATDB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = HIMOSAB(
                dim=embed_dim,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                base_win_size=base_win_size,
                mosa_heads=mosa_heads[i_layer],
                sparsity=sparsity,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate, 
                value_drop=value_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hier_win_ratios=hier_win_ratios
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # padding
        h, w = x.shape[2:]

        self.mean = self.mean.type_as(x)
        global inp
        inp = x.clone().permute(0,2,3,1)
        x = (x - self.mean) * self.img_range
        
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h * self.upscale, :w * self.upscale]

        return x

if __name__ == '__main__':
    upscale = 4
    
    # model = HIMOSAv2(
    #     upscale=4,
    #     img_size=64,
    #     embed_dim=210,
    #     depths=[6, 6, 6, 6, 6, 6, ],
    #     num_heads=[6, 6, 6, 6, 6, 6, ],
    #     window_size=16,
    #     mosa_heads=[8, 8, 8, 8, 8, 8,],
    #     num_tokens=128,
    #     reducted_dim=20,
    #     convffn_kernel_size=5,
    #     img_range=1.,
    #     down_c=20,
    #     mlp_ratio=2,
    #     upsampler='pixelshuffle',
    #     hier_win_ratios=[0.5,1,2,4,6,8],
    #     ).cuda()
    
    model = HIMOSAv1(
        upscale=2,
        img_size=64,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[4, 4, 4, 4],
        base_win_size= (8, 8),
        mosa_heads=[8, 8, 8, 8],
        sparsity=[1, 1, 2, 4, 8, 8],
        reducted_dim= 8,
        convffn_kernel_size=7,
        img_range= 1.,
        mlp_ratio= 1,
        upsampler='pixelshuffledirect',
        hier_win_ratios = [0.5,1,2,4,6,8]).cuda()
    
    input = torch.rand(1, 3, 256, 256).cuda()  # B C H W
    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    flops, _ = profile(model, inputs=(input,))
    print("FLOPs: {} G".format(flops/1e9))

    def cal_time(net, cal_iter=50):
        img0 = torch.randn(size=(1, 3, 256, 256)).float().cuda()
        # warmup
        with torch.no_grad():
            for i in range(10):
                _ = net(img0)
            torch.cuda.synchronize()

        # 设置用于测量时间的 cuda Event
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 初始化一个时间容器
        timings = np.zeros((cal_iter, 1))
        pbar = tqdm(total=cal_iter)
        with torch.no_grad():
            for rep in range(cal_iter):
                starter.record()
                _ = net(img0)
                ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
                timings[rep] = curr_time
                pbar.update()
                pbar.set_description("testing: " + str(curr_time))
        pbar.close()
        avgtime = timings.sum() / cal_iter

        return avgtime


    time_cost = cal_time(model)
    print('\033[0;36m time {} \033[0m'.format(time_cost))
