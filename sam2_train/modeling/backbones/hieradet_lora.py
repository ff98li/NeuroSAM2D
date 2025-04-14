import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple, Union
from sam2_train.modeling.backbones.hieradet import MultiScaleAttention, MultiScaleBlock, Hiera, do_pool

class LoRAMultiScaleAttention(MultiScaleAttention):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: Optional[nn.Module] = None,
        # LoRA specific parameters
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        use_output_lora: bool = True,
        use_k_lora: bool = False,
        k_lora_scaling_decay: float = 1.0,
        k_lora_init_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(dim, dim_out, num_heads, q_pool, **kwargs)
        
        # Freeze original QKV parameters
        self.qkv.weight.requires_grad_(False)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(False)

        # LoRA parameters for Q and V projections
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        head_dim = dim_out // num_heads
        assert head_dim * num_heads == dim_out, f"dim_out({dim_out}) must be divisible by num_heads({num_heads})"
        
        # Q LoRA adapters
        self.lora_A_q = nn.Parameter(torch.empty(dim, lora_r))
        self.lora_B_q = nn.Parameter(torch.zeros(lora_r, head_dim * num_heads))
        
        # V LoRA adapters
        self.lora_A_v = nn.Parameter(torch.empty(dim, lora_r))
        self.lora_B_v = nn.Parameter(torch.zeros(lora_r, head_dim * num_heads))
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        
        # Scaling factor
        self.scaling = lora_alpha / lora_r
        
        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Theoretically using K LoRA doesn't benefit the performance and would add computational cost
        if use_k_lora:
            self.lora_A_k = nn.Parameter(torch.empty(dim, lora_r))
            self.lora_B_k = nn.Parameter(torch.zeros(lora_r, head_dim * num_heads))
            nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5) * k_lora_init_scale)
            assert self.lora_A_k.shape == (dim, lora_r), f"K LoRA shape error: {self.lora_A_k.shape} != ({dim}, {lora_r})"
            print(f"K LoRA parameters trainable: {self.lora_A_k.requires_grad}, {self.lora_B_k.requires_grad}")
            self.k_scaling = self.scaling * k_lora_scaling_decay
        self.use_k_lora = use_k_lora

        # Output projection LoRA
        if use_output_lora:
            self.lora_A_proj = nn.Parameter(torch.empty(dim_out, lora_r))
            self.lora_B_proj = nn.Parameter(torch.zeros(lora_r, dim_out))
            nn.init.kaiming_uniform_(self.lora_A_proj, a=math.sqrt(5))
        self.use_output_lora = use_output_lora

    def _apply_lora(
        self,
        x: Tensor,
        lora_A: Tensor,
        lora_B: Tensor,
        num_heads: int,
        scaling: float = 1.0
    ) -> Tensor:
        """Applies LoRA transformation for a single projection"""
        # x shape: [B, H, W, dim]
        B, H, W, _ = x.shape
        assert lora_A.shape[-1] == lora_B.shape[0], f"lora_A.shape[-1]: {lora_A.shape[-1]}, lora_B.shape[0]: {lora_B.shape[0]}"
        
        # LoRA computation
        lora = (x @ self.lora_dropout(lora_A)) @ lora_B  # [B, H, W, head_dim]
        lora = lora * scaling
        
        # Reshape to match q, k, v shape: [B, H*W, num_heads, head_dim]
        head_dim = lora.shape[-1] // num_heads
        lora = lora.reshape(B, H*W, 1, num_heads, head_dim)
        lora = lora.squeeze(2)  # Now [B, H*W, num_heads, head_dim]
    
        return lora

    def forward(self, x: Tensor) -> Tensor:
        # Original QKV projection
        B, H, W, _ = x.shape
        qkv = self.qkv(x)  # [B, H*W, 3*dim_out]
        
        # Split into Q/K/V components
        qkv = qkv.reshape(B, H*W, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(2)  # Each [B, H*W, num_heads, head_dim]
        
        # Apply LoRA to Q and V
        x_flat = x.reshape(B, H*W, -1)
        lora_q = self._apply_lora(x, self.lora_A_q, self.lora_B_q, self.num_heads, self.scaling)
        lora_v = self._apply_lora(x, self.lora_A_v, self.lora_B_v, self.num_heads, self.scaling)

        assert q.shape == lora_q.shape, f"Shape mismatch: q {q.shape} vs lora_q {lora_q.shape}"
        assert v.shape == lora_v.shape, f"Shape mismatch: v {v.shape} vs lora_v {lora_v.shape}"
        
        # Add LoRA components
        q = q + lora_q
        v = v + lora_v

        if self.use_k_lora:
            lora_k = self._apply_lora(x, self.lora_A_k, self.lora_B_k, self.num_heads, self.k_scaling)
            assert k.shape == lora_k.shape, f"Shape mismatch: k {k.shape} vs lora_k {lora_k.shape}"
            k = k + lora_k

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)
        
        # Attention computation (preserve original SDPA)
        q = q.transpose(1, 2)  # [B, H*W, num_heads, pooled_head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        
        # Projection
        x = self.proj(x)

        # Apply LoRA to output projection
        if self.use_output_lora:
            B, H, W, _ = x.shape
            lora_proj = (x @ self.lora_dropout(self.lora_A_proj)) @ self.lora_B_proj
            x = x + lora_proj * self.scaling  # [B, H, W, dim_out]

        return x


class LoRAMultiScaleBlock(MultiScaleBlock):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        # LoRA specific parameters
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        use_output_lora: bool = True,
        use_k_lora: bool = False,
        k_lora_scaling_decay: float = 1.0,
        k_lora_init_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(
            dim=dim,
            dim_out=dim_out,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            q_stride=q_stride,
            act_layer=act_layer,
            window_size=window_size,
            **kwargs
        )
        
        # Replace original attention module with LoRA version
        self.attn = LoRAMultiScaleAttention(
            dim=dim,
            dim_out=dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_output_lora=use_output_lora,
            use_k_lora=use_k_lora,
            k_lora_scaling_decay=k_lora_scaling_decay,
            k_lora_init_scale=k_lora_init_scale,
        )
        
        # Freeze original MLP and projection layer parameters
        if hasattr(self, 'proj'):
            self.proj.weight.requires_grad_(False)
            if self.proj.bias is not None:
                self.proj.bias.requires_grad_(False)
        self.mlp.apply(lambda m: m.weight.requires_grad_(False) if isinstance(m, nn.Linear) else None)


class HieraLoRA(Hiera):
    def __init__(
        self,
        # Inherit all parameters from original Hiera
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        return_interm_layers=True,
        # LoRA specific parameters
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        use_output_lora: bool = True,
        use_k_lora: bool = False,
        k_lora_scaling_decay: float = 1.0,
        k_lora_init_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            q_pool=q_pool,
            q_stride=q_stride,
            stages=stages,
            dim_mul=dim_mul,
            head_mul=head_mul,
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
            window_spec=window_spec,
            global_att_blocks=global_att_blocks,
            return_interm_layers=return_interm_layers,
            **kwargs
        )
        
        # Rewrite the block construction process
        depth = sum(stages)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        cur_stage = 1
        for i in range(sum(stages)):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = LoRAMultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                # Pass LoRA parameters
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_output_lora=use_output_lora,
                use_k_lora=use_k_lora,
                k_lora_scaling_decay=k_lora_scaling_decay,
                k_lora_init_scale=k_lora_init_scale,
            )
            embed_dim = dim_out
            self.blocks.append(block)
        
        # Freeze all non-LoRA parameters
        self._freeze_non_lora_params()

    def _freeze_non_lora_params(self):
        """Freeze all non-LoRA parameters"""
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)
    
    def count_parameters(self, only_trainable=True):
        """Count parameters in the model"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def print_model_info(self):
        """Print model information"""
        print(f"Model: {self.__class__.__name__}")
        print(f"Total parameters: {self.count_parameters(only_trainable=False)}")
        print(f"Trainable parameters: {self.count_parameters(only_trainable=True)}")
        print(f"LoRA parameter fraction: {(self.count_parameters(only_trainable=True) / self.count_parameters(only_trainable=False)) * 100}%")