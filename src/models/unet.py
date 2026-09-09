"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        image_size: int = 64,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # A default setting: time_embed_dim = 4 times of base_channels
        self.time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(self.time_embed_dim)

        # The first initial conv
        self.in_conv = nn.Conv2d(self.in_channels,self.base_channels,kernel_size=3,padding=1)

        # Each down blocks with the current layer's channel dim
        self.down_blocks = nn.ModuleList() 
        self.downsamples = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.up_attentions = nn.ModuleList()


        in_ch = base_channels
        resolution = image_size


        # Down Samples
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult

            level_blocks = nn.ModuleList()
            level_attentions = nn.ModuleList()

            for _ in range(num_res_blocks):
                block = ResBlock(
                    in_ch,
                    out_ch,
                    self.time_embed_dim,
                    self.dropout,
                    self.use_scale_shift_norm
                )

            if resolution in attention_resolutions:
                level_attentions.append(AttentionBlock(out_ch, num_heads))
            else:
                level_attentions.append(nn.Identity())

            level_blocks.append(block)

            in_ch = out_ch

            self.down_blocks.append(level_blocks)
            self.down_attentions.append(level_attentions)


            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(in_ch))
                resolution = resolution // 2

        # Midcoder
        self.middle_block = ResBlock(
        in_channels=in_ch,
        out_channels=in_ch,
        time_embed_dim=self.time_embed_dim,
        dropout=dropout,
        use_scale_shift_norm=use_scale_shift_norm,
    )

        # Upsamples
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult

            level_blocks = nn.ModuleList()
            level_attentions = nn.ModuleList()

            for _ in range(num_res_blocks):
                block = ResBlock(
                    in_ch+out_ch,
                    out_ch,
                    self.time_embed_dim,
                    self.dropout,
                    self.use_scale_shift_norm
                )

            if resolution in attention_resolutions:
                level_attentions.append(AttentionBlock(out_ch, num_heads))
            else:
                level_attentions.append(nn.Identity())

                level_blocks.append(block)

                in_ch = out_ch

            self.up_blocks.append(level_blocks)
            self.up_attentions.append(level_attentions)


            if level < len(channel_mult) - 1:
                self.upsamples.append(Upsample(in_ch))
                resolution = resolution * 2

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        time_emb = self.time_embed(t)
        skips = []

        # Down Sample Half
        h = self.in_conv(x)

        for level, level_blocks in enumerate(self.down_blocks):
            level_attentions = self.down_attentions[level]
            for block, attention in zip(level_blocks, level_attentions):
                h = block(h, time_emb)
                h = attention(h)
                skips.append(h)

            if level < len(self.channel_mult) - 1:
                h = self.downsamples[level](h)

                

        


        # Mid Coder
        h = self.middle_block(h, time_emb)

        # Upsample Half
        for level, level_blocks in enumerate(self.up_blocks):
            level_attentions = self.up_attentions[level]
            for block, attention in zip(level_blocks, level_attentions):
                skip = skips.pop()
                h = torch.cat([h, skip], dim = 1)
                h = block(h, time_emb)
                        
            if level < len(self.channel_mult) - 1:
                h = self.upsamples(h)

        self.out_norm = GroupNorm32(32, self.base_channels)
        self.out_conv = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=3, padding=1)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
            
        


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
        image_size=data_config['image_size'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
