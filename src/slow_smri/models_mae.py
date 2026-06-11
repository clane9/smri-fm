import inspect
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int

from .modules import Block, LayerNorm, SeparablePosEmbed3D


Layer = Type[nn.Module]


class VoxelMAE(nn.Module):
    def __init__(
        self,
        num_bins: int = 256,
        grid_size: int | tuple[int, int, int] = (1024, 1024, 1024),
        grid_unit: float = 1.0,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        decoder_depth: int = 4,
        decoder_embed_dim: int | None = 512,
        decoder_num_heads: int | None = 16,  # default from mae, head dim = 32
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_tokens: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.grid_size = grid_size
        self.grid_unit = grid_unit
        self.class_tokens = class_tokens

        # encoder
        self.voxel_embed = nn.Embedding(num_bins, embed_dim)
        self.pos_embed = SeparablePosEmbed3D(grid_size, embed_dim, unit=grid_unit)
        if class_tokens:
            self.cls_token = nn.Parameter(torch.empty(1, class_tokens, embed_dim))
        else:
            self.cls_token = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                )
                for ii in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = SeparablePosEmbed3D(grid_size, decoder_embed_dim, unit=grid_unit)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                )
                for ii in range(decoder_depth)
            ]
        )

        self.decoder_norm = LayerNorm(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, num_bins)
        self.init_weights()

    def extra_repr(self):
        return (
            f"{self.num_bins}, {self.grid_size}, "
            f"grid_unit={self.grid_unit}, class_tokens={self.class_tokens}"
        )

    def init_weights(self):
        self.apply(_init_weights)
        if self.class_tokens:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # init from nanochat
        nn.init.normal_(self.voxel_embed.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.decoder_head.weight, mean=0.0, std=0.001)

    def forward_encoder(
        self,
        x: Int[Tensor, "B N"],
        coord: Float[Tensor, "B N 3"],
    ) -> tuple[
        Float[Tensor, "B R D"] | None,
        Float[Tensor, "B N D"],
    ]:
        B, N = x.shape
        assert coord.shape == (B, N, 3)

        x = self.voxel_embed(x)
        x = self.pos_embed(x, coord)

        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.class_tokens:
            cls_embeds = x[:, : self.class_tokens]
            x = x[:, self.class_tokens :]
        else:
            cls_embeds = None
        return cls_embeds, x

    def forward_decoder(
        self,
        embeds_vis: Float[Tensor, "B N D"],
        coord_target: Float[Tensor, "B M 3"],
    ) -> Float[Tensor, "B M K"]:
        B, N, D = embeds_vis.shape
        M = coord_target.shape[1]
        assert coord_target.shape == (B, M, 3)

        embeds_vis = self.decoder_proj(embeds_vis)

        mask = self.mask_token.expand(B, M, -1)
        mask = self.decoder_pos_embed(mask, coord_target)

        x = torch.cat([embeds_vis, mask], dim=1)

        for block in self.decoder_blocks:
            x = block(x)

        x = x[:, N:]
        x = self.decoder_norm(x)
        x = self.decoder_head(x)
        return x

    def forward(
        self,
        x: Int[Tensor, "B N"],
        coord: Float[Tensor, "B N 3"],
        num_visible: int | None = None,
    ):
        B, N = x.shape
        num_visible = num_visible or N // 2

        vis = x[:, :num_visible]
        target = x[:, num_visible:]
        coord_vis = coord[:, :num_visible]
        coord_target = coord[:, num_visible:]

        _, embeds = self.forward_encoder(vis, coord_vis)
        logits = self.forward_decoder(embeds, coord_target)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        state = {
            "vis": vis,
            "target": target,
            "coord_vis": coord_vis,
            "coord_target": coord_target,
            "embeds": embeds,
            "logits": logits,
        }
        return loss, state


# JAX ViT xavier uniform init
# https://github.com/facebookresearch/capi/blob/main/model.py
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def filter_kwargs(func, kwargs):
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs


def voxel_mae_base(**kwargs):
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    kwargs = filter_kwargs(VoxelMAE, kwargs)
    return VoxelMAE(**model_args, **kwargs)


# Here we create 5 scaled models with the same proportion as vit base
# Specifically, we preserve:
# - encoder depth / decoder depth = 3
# - encoder aspect ratio embed_dim / depth = 64
# - encoder head_dim = 64
# - decoder aspect ratio embed_dim / depth = 128
# - decoder head_dim = 32
# This follows the design of the nanochat model scaling experiment :)


def voxel_mae_d15(**kwargs):
    kwargs.pop("decoder_depth", None)
    model_args = dict(
        embed_dim=960,
        depth=15,
        num_heads=15,
        decoder_depth=5,
        decoder_embed_dim=640,
        decoder_num_heads=20,
    )
    kwargs = filter_kwargs(VoxelMAE, kwargs)
    return VoxelMAE(**model_args, **kwargs)


def voxel_mae_d12(**kwargs):
    # nb same as vit base but to complete the sequence
    kwargs.pop("decoder_depth", None)
    model_args = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_depth=4,
        decoder_embed_dim=512,
        decoder_num_heads=16,
    )
    kwargs = filter_kwargs(VoxelMAE, kwargs)
    return VoxelMAE(**model_args, **kwargs)


def voxel_mae_d9(**kwargs):
    kwargs.pop("decoder_depth", None)
    model_args = dict(
        embed_dim=576,
        depth=9,
        num_heads=9,
        decoder_depth=3,
        decoder_embed_dim=384,
        decoder_num_heads=12,
    )
    kwargs = filter_kwargs(VoxelMAE, kwargs)
    return VoxelMAE(**model_args, **kwargs)


def voxel_mae_d6(**kwargs):
    kwargs.pop("decoder_depth", None)
    model_args = dict(
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_depth=2,
        decoder_embed_dim=256,
        decoder_num_heads=8,
    )
    kwargs = filter_kwargs(VoxelMAE, kwargs)
    return VoxelMAE(**model_args, **kwargs)


def voxel_mae_d3(**kwargs):
    kwargs.pop("decoder_depth", None)
    model_args = dict(
        embed_dim=192,
        depth=3,
        num_heads=3,
        decoder_depth=1,
        decoder_embed_dim=128,
        decoder_num_heads=4,
    )
    kwargs = filter_kwargs(VoxelMAE, kwargs)
    return VoxelMAE(**model_args, **kwargs)
