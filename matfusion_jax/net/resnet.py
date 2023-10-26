import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import Array
from typing import Optional, Any, Callable
from functools import partial

from memory_efficient_attention import efficient_dot_product_attention_jax


# Some modifications by Sam Sartor. Original version at
# https://github.com/huggingface/diffusers/blob/045157a46fb16a21a5f37a5f3f3ad710895b680b/src/diffusers/models/vae_flax.py
class SelfAttentionBlock(nn.Module):
    r"""
    Flax Convolutional based multi-head attention block for diffusion-based VAE.

    Parameters:
        channels (:obj:`int`):
            Input channels
        num_head_channels (:obj:`int`, *optional*, defaults to `None`):
            Number of attention heads
        num_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for group norm
    """
    channels: int
    num_head_channels: Optional[int] = None
    wrong_heads: bool = False
    memory_efficient: bool = False
    num_groups: int = 32

    def setup(self):
        self.num_heads = self.channels // self.num_head_channels if self.num_head_channels is not None else 1

        dense = partial(nn.Dense, self.channels)

        self.group_norm = nn.GroupNorm(num_groups=self.num_groups, epsilon=1e-6)
        self.query, self.key, self.value = dense(), dense(), dense()
        self.proj_attn = dense()

    def transpose_for_scores(self, projection):
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D)
        new_projection = projection.reshape(new_projection_shape)
        if self.wrong_heads:
            # (B, T, H, D) -> (B, H, T, D)
            new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
        return new_projection

    def untranspose_after_scores(self, projection):
        if self.wrong_heads:
            # (B, H, T, D) -> (B, T, H, D)
            new_projection = jnp.transpose(projection, (0, 2, 1, 3))
        else:
            new_projection = projection
        # (B, T, H, D) -> (B, T, H * D)
        new_projection_shape = new_projection.shape[:-2] + (self.channels,)
        new_projection = new_projection.reshape(new_projection_shape)
        return new_projection

    #@nn.remat
    def __call__(self, hidden_states):
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.reshape((batch, height * width, channels))

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # transpose
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # compute attentions
        scale = (self.channels / self.num_heads) ** -0.5
        if self.wrong_heads:
            query = query * (scale ** 0.5)
            key = key * (scale ** 0.5)
        if self.memory_efficient:
            hidden_states = efficient_dot_product_attention_jax(query, key, value)
        else:
            attention_scores = jnp.einsum("b i h d, b j h d -> b h i j", query, key)
            attention_probs = nn.softmax(attention_scores*scale, axis=3)
            hidden_states = jnp.einsum("b h i j, b j h d -> b i h d", attention_probs, value)

        # untranspose
        hidden_states = self.untranspose_after_scores(hidden_states)

        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.reshape((batch, height, width, channels))
        hidden_states = hidden_states + residual
        return hidden_states


class ResBlock(nn.Module):
    training: bool
    deterministic: bool
    features: int
    size: int = 3
    activation: Callable[[Array], Array] = nn.relu
    cond_activation: bool = True
    dropout: Optional[float] = None
    conv: Any = nn.Conv

    @nn.remat
    @nn.compact
    def __call__(self, x: Array, cond: Optional[Array] = None) -> Array:
        s = x

        x = self.conv(
            features=self.features,
            kernel_size=(self.size, self.size),
            strides=(1, 1),
            use_bias=False,
            name='conv_1',
        )(x)
        if cond is not None:
            if self.cond_activation:
                e = self.activation(cond)
            else:
                e = cond
            e = jnp.expand_dims(e, (1, 2))
            e = nn.Conv(
                features=x.shape[-1],
                kernel_size=(1, 1),
                use_bias=True,
                name='conv_extra',
            )(e)
            x = x + e
        x = nn.GroupNorm(num_groups=32, name='norm_1')(x)
        x = self.activation(x)

        x = self.conv(
            features=self.features,
            kernel_size=(self.size, self.size),
            use_bias=False,
            name='conv_2',
        )(x)
        x = nn.GroupNorm(num_groups=32, name='norm_2')(x)
        if self.dropout:
            x = nn.Dropout(self.dropout, deterministic=self.deterministic, name='dropout')(x)

        if x.shape[3] != s.shape[3]:
            s = self.conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=True,
                name='conv_shortcut',
            )(s)

        x = x + s
        x = self.activation(x)

        return x


ResTransposeBlock = partial(ResBlock, conv=nn.ConvTranspose)


class UpsampleBlock(nn.Module):
    features: int

    @nn.remat
    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = jax.image.resize(
            x,
            shape=(b, h * 2, w * 2, c),
            method="nearest",
        )
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=True,
            name='conv',
        )(x)
        return x


class DownsampleBlock(nn.Module):
    features: int

    @nn.remat
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(2, 2),
            use_bias=True,
            name='conv',
        )(x)
        return x


def no_self_attn(i: int) -> bool:
    return False


class ResNetBackbone(nn.Module):
    training: bool
    deterministic: bool
    features: list[int]
    layers_per: int
    block: Callable[..., nn.Module]
    inputs: Optional[int] = None
    activation: Callable[[Array], Array] = nn.relu
    cond_activation: bool = True
    use_self_attn: Callable[[int], bool] = no_self_attn
    dropout_cond: float = 0.0
    num_head_channels: int = 8
    wrong_heads: bool = False

    @nn.compact
    def __call__(
        self,
        x: Array,
        u: Optional[list[Array]] = None,
        cond: Optional[Array] = None,
    ) -> Array:
        if self.inputs is not None:
            assert x.shape[-1] == self.inputs
        x = nn.Conv(
            features=self.features[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=True,
            name='conv_in',
        )(x)
        if u is not None:
            u.append(x)

        for i in range(len(self.features)):
            for j in range(self.layers_per):
                x = self.block(
                    features=self.features[i],
                    training=self.training,
                    deterministic=self.deterministic,
                    activation=self.activation,
                    cond_activation=self.cond_activation,
                    name=f'resnet_{i+1}_{j+1}'
                )(x, cond)
                if self.use_self_attn(i):
                    x = SelfAttentionBlock(
                        channels=self.features[i],
                        num_head_channels=self.num_head_channels,
                        wrong_heads=self.wrong_heads,
                        name=f'attention_{i+1}_{j+1}',
                    )(x)
                if u is not None:
                    u.append(x)
            if i != len(self.features) - 1:
                x = DownsampleBlock(
                    features=self.features[i],
                    name=f'downsample_{i+1}',
                )(x)
                if u is not None:
                    u.append(x)

        return x


class ResNetTransposeBackbone(nn.Module):
    training: bool
    deterministic: bool
    features: list[int]
    layers_per: int
    block: Any
    inputs: int
    activation: Callable[[Array], Array] = nn.relu
    cond_activation: bool = True
    use_self_attn: Callable[[int], bool] = no_self_attn
    num_head_channels: int = 8
    wrong_heads: bool = False

    @nn.compact
    def __call__(
        self,
        x: Array,
        u: Optional[list[Array]] = None,
        cond: Optional[Array] = None,
    ) -> Array:
        for (i, features) in enumerate(reversed(self.features)):
            for j in range(self.layers_per):
                if u is not None:
                    this_u = u.pop()
                    # print(x.shape, this_u.shape)
                    x = jnp.concatenate((x, this_u), 3)
                x = self.block(
                    features=features,
                    training=self.training,
                    deterministic=self.deterministic,
                    activation=self.activation,
                    cond_activation=self.cond_activation,
                    name=f'resnet_{i+1}_{j+1}'
                )(x, cond)
                if self.use_self_attn(len(self.features)-1-i):
                    x = SelfAttentionBlock(
                        channels=features,
                        num_head_channels=self.num_head_channels,
                        wrong_heads=self.wrong_heads,
                        name=f'attention_{i+1}_{j+1}',
                    )(x)
            if i != len(self.features) - 1:
                x = UpsampleBlock(
                    features=features,
                    name=f'upsample_{i+1}',
                )(x)

        x = nn.GroupNorm(num_groups=32, name='conv_norm_out')(x)
        x = self.activation(x)
        x = nn.ConvTranspose(
            features=self.inputs,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=True,
            name='conv_out',
        )(x)
        return x


class ConvnextBlock(nn.Module):
    training: bool
    deterministic: bool
    features: int
    size: int = 7
    feature_mult: int = 4
    activation: Any = nn.relu
    cond_activation: bool = True
    dropout: Optional[float] = None
    conv: Any = nn.Conv

    @nn.remat
    @nn.compact
    def __call__(self, x: Array, cond: Optional[Array] = None) -> Array:
        s = x

        x = nn.Conv(
            features=x.shape[-1],
            feature_group_count=x.shape[-1],
            kernel_size=(self.size, self.size),
            use_bias=True,
            name='conv_1',
        )(x)
        if cond is not None:
            if self.cond_activation:
                e = self.activation(cond)
            else:
                e = cond
            e = jnp.expand_dims(e, (1, 2))
            e = nn.Conv(
                features=x.shape[-1],
                kernel_size=(1, 1),
                use_bias=True,
                name='conv_extra',
            )(e)
            x = x + e
        x = nn.GroupNorm(num_groups=32, name='norm_1')(x)
        if self.dropout:
            x = nn.Dropout(self.dropout, deterministic=self.deterministic, name='dropout')(x)

        x = self.conv(
            self.features * self.feature_mult,
            kernel_size=(1, 1),
            use_bias=True,
            name='conv_2',
        )(x)
        x = self.activation(x)
        x = self.conv(
            self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            name='conv_3',
        )(x)

        if x.shape[3] != s.shape[3]:
            s = self.conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=True,
                name='conv_shortcut',
            )(s)

        x = x + s

        return x


ConvnextTransposeBlock = partial(ConvnextBlock, conv=nn.ConvTranspose)
