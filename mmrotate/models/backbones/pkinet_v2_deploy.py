import torch
import torch.nn as nn
from functools import partial
import math

# --- Dependency handling ---
try:
    from mmcv.runner import BaseModule
    from mmcv.cnn import build_norm_layer
    try:
        from mmrotate.models.builder import ROTATED_BACKBONES
    except ImportError:
        from mmdet.models.builder import BACKBONES as ROTATED_BACKBONES
except ImportError:
    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None): super().__init__()
    def build_norm_layer(cfg, num_features, postfix=''): return 'BN', nn.BatchNorm2d(num_features)
    class Registry:
        def register_module(self):
            def decorator(cls): return cls
            return decorator
    ROTATED_BACKBONES = Registry()

from timm.models.layers import DropPath, to_2tuple

# =============================================================================
# 1. Deploy core module: PKSModuleDeploy
# =============================================================================

class PKSModuleDeploy(nn.Module):
    """
    [PKINet-v2 Deploy Mode - Pure PyTorch]
    Structure: x * Conv1( Fused_Large_Kernel( Conv0(x) ) )
    
    - Fixed Max Kernel Size = 19 (determined by PKINet-v2 branches)
    """
    def __init__(self, dim, kernel_sizes=None, attempt_use_lk_impl=False):
        super().__init__()
        self.dim = dim
        
        # [V6 Architecture Fixed Constraint]
        # Branch 1 (Axial) is 1x19 + 19x1 => RF 19
        # Branch 2 (Dilated) is 7x7 (d=3) => RF 19
        self.max_k = 19 

        # 1. Conv0 (5x5) pre-processing
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 2. Fused parallel conv (19x19)
        # This single convolution contains the fused weights of all 5 branches from training
        self.fused_parallel_conv = nn.Conv2d(
            dim, dim, 
            kernel_size=self.max_k, 
            padding=self.max_k//2, 
            groups=dim, 
            bias=True
        )

        # 3. Conv1 (1x1) channel mixing
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.fused_parallel_conv(attn)
        attn = self.conv1(attn)
        return x * attn

class PKSBlockDeploy(nn.Module):
    def __init__(self, dim, kernel_sizes=None, attempt_use_lk_impl=False):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        # Pass kernel_sizes for API compatibility, though V6 ignores them
        self.spatial_gating_unit = PKSModuleDeploy(dim, kernel_sizes, attempt_use_lk_impl=attempt_use_lk_impl)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

# =============================================================================
# 2. Standard components (Mlp, PatchEmbed) - Unchanged
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

# =============================================================================
# 3. Block & backbone (Deploy)
# =============================================================================

class PKINetV2BlockDeploy(nn.Module):
    def __init__(self, dim, mlp_ratio=4., kernel_sizes=None, 
                 drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=dict(type='BN'),
                 attempt_use_lk_impl=False):
        super().__init__()
        
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
            
        self.attn = PKSBlockDeploy(dim, kernel_sizes, attempt_use_lk_impl=attempt_use_lk_impl)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(self.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(self.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

@ROTATED_BACKBONES.register_module()
class PKINetV2Deploy(BaseModule):
    """
    [Deploy] PKINet-v2 backbone
    """
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], 
                 kernel_sizes=None, # Deprecated in V6 but kept for API
                 drop_rate=0., drop_path_rate=0., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 norm_cfg=dict(type='BN', requires_grad=True),
                 attempt_use_lk_impl=False, 
                 **kwargs): 
        super().__init__()
        
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([PKINetV2BlockDeploy(
                dim=embed_dims[i], 
                mlp_ratio=mlp_ratios[i], 
                kernel_sizes=kernel_sizes,
                drop=drop_rate, 
                drop_path=dpr[cur + j],
                norm_cfg=norm_cfg,
                attempt_use_lk_impl=attempt_use_lk_impl) 
                for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            
            for blk in block:
                x = blk(x)
            
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

@ROTATED_BACKBONES.register_module()
class PKINetV2_S_Deploy(PKINetV2Deploy):
    def __init__(self, **kwargs):
        cfg = dict(
            embed_dims=[64, 128, 320, 512],
            depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4], 
            drop_rate=0.1,
            drop_path_rate=0.15,
            norm_cfg=dict(type='BN', requires_grad=True),
            attempt_use_lk_impl=False
        )
        cfg.update(kwargs)
        super().__init__(**cfg)

@ROTATED_BACKBONES.register_module()
class PKINetV2_T_Deploy(PKINetV2Deploy):
    def __init__(self, **kwargs):
        cfg = dict(
            embed_dims=[32, 64, 160, 256],
            depths=[3, 3, 5, 2],
            mlp_ratios=[8, 8, 4, 4], 
            drop_rate=0.1,
            drop_path_rate=0.1,
            norm_cfg=dict(type='BN', requires_grad=True),
            attempt_use_lk_impl=False
        )
        cfg.update(kwargs)
        super().__init__(**cfg)
