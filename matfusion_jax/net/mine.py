from dataclasses import field
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from .resnet import (
    ConvnextBlock,
    ConvnextTransposeBlock,
    ResBlock,
    ResNetBackbone,
    ResNetTransposeBackbone,
    ResTransposeBlock,
    SelfAttentionBlock,
)


def match_reflectance(y_est, y):
    r_est = y_est[:, :, 0:6]
    r = y[:, :, 0:6]
    factor = jnp.sum(r * r) / jnp.sum(r * r_est)
    return jnp.concatenate((factor * r_est, y_est[:, :, 6:]), 2)


def batched_match_reflectance(y_est, y):
    return jax.vmap(match_reflectance, 0, 0)(y_est, y)


BLOCKS: dict[str, tuple[Callable[..., nn.Module], Callable[..., nn.Module]]] = {
    'resnet': (ResBlock, ResTransposeBlock),
    'convnext': (ConvnextBlock, ConvnextTransposeBlock),
}

ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    'silu': nn.silu,
    'gelu': nn.gelu,
    'relu': nn.relu,
}

class MyMidBlock(nn.Module):
    training: bool
    deterministic: bool
    block: str = 'convnext'
    features: int = 1024
    activation: str = 'silu'
    cond_activation: bool = True
    dropout: Optional[float] = 0.1
    num_head_channels: int = 8
    wrong_heads: bool = False
    use_self_attn: bool = True

    @nn.remat
    @nn.compact
    def __call__(
        self,
        x: Array,
        cond: Optional[Array] = None,
    ) -> Array:
        x = BLOCKS[self.block][0](
            features=self.features,
            dropout=self.dropout,
            training=self.training,
            deterministic=self.deterministic,
            activation=ACTIVATIONS[self.activation],
            cond_activation=self.cond_activation,
            name='resnet_1',
        )(x, cond)
        if self.use_self_attn:
            x = SelfAttentionBlock(
                channels=self.features,
                num_head_channels=self.num_head_channels,
                wrong_heads=self.wrong_heads,
                name='attention_1',
            )(x)
        x = BLOCKS[self.block][0](
            features=self.features,
            dropout=self.dropout,
            training=self.training,
            deterministic=self.deterministic,
            activation=ACTIVATIONS[self.activation],
            cond_activation=self.cond_activation,
            name='resnet_2',
        )(x, cond)
        return x


class MyNoiseModel(nn.Module):
    training: bool
    inputs: int
    channels: int
    cond_mlp_inputs: int = 128
    cond_mlp_outputs: int = 512
    activation: str = 'silu'
    mid_activation: Optional[str] = None
    cond_activation: bool = True
    block: str = 'convnext'
    features: list[int] = field(default_factory=lambda: [128, 256, 512, 512, 1024, 1024])
    layers_per: int = 2
    mid_attn: bool = True
    layer_attn: bool = True
    wrong_heads: bool = False

    def use_self_attn(self, i):
        layer_inds = range(len(self.features))
        return self.layer_attn and (i in layer_inds[-3:-1])

    @nn.compact
    def __call__(
        self,
        x: Array,
        embedding: Array,
        skip_additions: Optional[dict[int, Array]] = None,
        mid_additions: Optional[Array] = None,
    ) -> Array:
        u = []
        c = nn.Dense(features=self.cond_mlp_outputs, name='linear_1')(embedding[:, :self.cond_mlp_inputs])
        c = ACTIVATIONS[self.activation](c)
        c = nn.Dense(features=self.cond_mlp_outputs, name='linear_2')(c)
        if embedding.shape[1] > self.cond_mlp_inputs:
            c = jnp.concatenate((c, embedding[:, self.cond_mlp_inputs:]), axis=1)

        # print(x.shape, c.shape, self.channels, self.inputs)
        x = ResNetBackbone(
            inputs=self.channels + self.inputs,
            features=self.features,
            layers_per=self.layers_per,
            use_self_attn=self.use_self_attn,
            wrong_heads=self.wrong_heads,
            block=BLOCKS[self.block][0],
            training=self.training,
            deterministic=not self.training,
            activation=ACTIVATIONS[self.activation],
            cond_activation=self.cond_activation,
            name='down_blocks',
        )(x, cond=c, u=u)

        if skip_additions is not None:
            u = [u + skip_additions.get(i, 0) for (i, u) in enumerate(u)]
        if mid_additions is not None:
            x = x + mid_additions

        x = MyMidBlock(
            training=self.training,
            deterministic=not self.training,
            features=self.features[-1],
            activation=self.mid_activation or self.activation,
            cond_activation=self.cond_activation,
            use_self_attn=self.mid_attn,
            wrong_heads=self.wrong_heads,
            block=self.block,
            name='mid_block',
        )(x, cond=c)
        x = ResNetTransposeBackbone(
            inputs=self.channels,
            features=self.features,
            layers_per=self.layers_per+1,
            use_self_attn=self.use_self_attn,
            wrong_heads=self.wrong_heads,
            block=BLOCKS[self.block][1],
            training=self.training,
            deterministic=not self.training,
            activation=ACTIVATIONS[self.activation],
            cond_activation=self.cond_activation,
            name='up_blocks',
        )(x, cond=c, u=u)
        return x
