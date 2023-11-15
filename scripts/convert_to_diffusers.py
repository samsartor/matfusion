import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse

import numpy as np
import jax

if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from matfusion_jax.model import Model
from matfusion_jax.pipeline import Diffusion
from matfusion_jax.net.mine import MyNoiseModel
from matfusion_jax.net.resnet import ResNetBackbone, ConvnextBlock
from jax.tree_util import tree_map, tree_flatten_with_path
from flax import linen as nn
from copy import deepcopy


import diffusers
import torch


def convert_tensor(pt_tensor, jax_tensor):
    assert pt_tensor.shape == jax_tensor.shape, f'{jax_tensor.shape} does not fit into {pt_tensor.shape}'
    pt_tensor.copy_(torch.from_numpy(np.array(jax_tensor)))


def convert_auto(t, j, all=True):
    j_kernel = None
    if 'kernel' in j:
        j_kernel = j.pop('kernel')
    elif 'scale' in j:
        j_kernel = j.pop('scale')

    if j_kernel is not None:
        # TODO: transpose kernel
        if len(t.weight.shape) == 1:
            convert_tensor(t.weight, j_kernel)
        elif len(t.weight.shape) == 2:
            convert_tensor(t.weight, jnp.transpose(jnp.squeeze(j_kernel), (1, 0)))
        elif len(t.weight.shape) == 4:
            # (7, 7, 1, 128) -> (128, 1, 7, 7)
            convert_tensor(t.weight, jnp.transpose(j_kernel, (3, 2, 0, 1)))
        else:
            assert False, f'{t.weight.shape} is not understood'
        if t.bias is not None:
            convert_tensor(t.bias, j.pop('bias'))
        else:
            assert 'bias' not in j
    elif isinstance(t, diffusers.models.unet_2d_blocks.Attention):
        convert_auto(t.to_q, j.pop('query'))
        convert_auto(t.to_k, j.pop('key'))
        convert_auto(t.to_v, j.pop('value'))
        convert_auto(t.to_out[0], j.pop('proj_attn'))

    for k in list(j.keys()):
        try:
            if hasattr(t, k):
                convert_auto(getattr(t, k), j.pop(k))
            elif hasattr(t, k.replace('_', '')):
                convert_auto(getattr(t, k.replace('_', '')), j.pop(k))
            elif k == 'conv_extra':
                convert_auto(t.time_emb_proj, j.pop(k))
        except Exception as e:
            print(f'error converting {k}')
            raise e

    if all:
        assert len(j) == 0, f'failed to convert {j.keys()}'


def convert_backbone(pt_model, pt_blocks, jax_blocks):
    for (i, pt_block) in enumerate(pt_blocks):
        for (j, pt_resnet) in enumerate(pt_block.resnets):
            try:
                convert_auto(pt_resnet, jax_blocks.pop(f'resnet_{i+1}_{j+1}'))
            except Exception as e:
                print(f'error converting resnet_{i+1}_{j+1}')
                raise e
        if hasattr(pt_block, 'attentions'):
            for (j, pt_attn) in enumerate(pt_block.attentions):
                convert_auto(pt_attn, jax_blocks.pop(f'attention_{i+1}_{j+1}'))
        if hasattr(pt_block, 'downsamplers') and pt_block.downsamplers is not None:
            convert_auto(pt_block.downsamplers[0], jax_blocks.pop(f'downsample_{i+1}'))
        if hasattr(pt_block, 'upsamplers') and pt_block.upsamplers is not None:
            convert_auto(pt_block.upsamplers[0], jax_blocks.pop(f'upsample_{i+1}'))
    convert_auto(pt_model, jax_blocks)


def convert_model(pt_model, jax_model):
    convert_backbone(pt_model, pt_model.down_blocks, jax_model.pop('down_blocks'))
    convert_backbone(pt_model, pt_model.up_blocks, jax_model.pop('up_blocks'))
    j_mid = jax_model.pop('mid_block')
    convert_auto(pt_model.mid_block.resnets[0], j_mid.pop('resnet_1'))
    convert_auto(pt_model.mid_block.attentions[0], j_mid.pop('attention_1'))
    convert_auto(pt_model.mid_block.resnets[1], j_mid.pop('resnet_2'))
    assert len(j_mid) == 0
    convert_auto(pt_model.time_embedding, jax_model, all=False)
    convert_auto(pt_model, jax_model)


def block_type(name, attn, dir):
    return {
        ('resnet', False): f'{dir}Block2D',
        ('resnet', True): f'Attn{dir}Block2D',
        ('convnext', False): f'Convnext{dir}Block2D',
        ('convnext', True): f'ConvnextAttn{dir}Block2D',
    }[(name, attn)]


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

    parser = argparse.ArgumentParser(description='Convert a MatFusion model for use by huggingface diffusers.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--print_layers', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--verify-count', type=int, default=1)
    args = parser.parse_args()

    jax_model = Model.from_checkpoint(args.input)
    assert jax_model.unreplicated_gen_state is not None
    jax_params = jax_model.unreplicated_gen_state.eval_params
    if hasattr(jax_params, 'unfreeze'):
        jax_params_mut = jax_params.unfreeze()
    else:
        jax_params_mut = deepcopy(jax_params)
    jax_net = MyNoiseModel(**jax_model.mode['noise_model'], training=False)

    total_params = 0
    for (name, params) in tree_flatten_with_path(jax_params)[0]:
        total_params += np.product(params.shape)
        if args.print_layers:
            print('/'.join(n.key for n in name), params.shape)
    print(f'Converting {total_params} total parameters')

    pt_model = diffusers.UNet2DModel(
            sample_size=jax_model.diffusion_shape[-2],
            in_channels=jax_net.channels + jax_net.inputs,
            out_channels=jax_net.channels,
            center_input_sample=False,
            time_embedding_type="positional",
            flip_sin_to_cos=False,
            down_block_types=tuple(
                block_type(jax_net.block, jax_net.use_self_attn(i), 'Down')
                for i in range(len(jax_net.features))
            ),
            mid_block_type='ConvnextMidBlock2D',
            up_block_types=tuple(
                block_type(jax_net.block, jax_net.use_self_attn(i), 'Up')
                for i in reversed(range(len(jax_net.features)))
            ),
            block_out_channels=jax_net.features,
            layers_per_block=jax_net.layers_per,
            act_fn=jax_net.activation,
            mid_act_fn=None if jax_net.mid_activation is None else jax_net.mid_activation,
            attention_head_dim=ResNetBackbone.num_head_channels,
            convnext_channels_mult=ConvnextBlock.feature_mult,
            convnext_time_embedding_activation=jax_net.cond_activation,
            downsample_padding=0,
            wrong_heads=jax_net.wrong_heads,
    ).cpu()
    pt_model.requires_grad = False

    for name, params in pt_model.named_parameters():
        params.requires_grad = False
        params.fill_(np.nan)
        if args.print_layers:
            print(name, params.shape)

    convert_model(pt_model, jax_params_mut)

    for name, params in pt_model.named_parameters():
        assert torch.all(torch.isfinite(params)), f'{name} is not initilized'
    pt_model.save_pretrained(args.output)
    print(f'Saved pytorch model to {args.output}')

    if not args.verify:
        exit(0)

    dif = Diffusion.from_mode(jax_model.mode)
    jax_net_apply = nn.jit(lambda n, *args: n.apply(*args))
    pt_model = torch.compile(pt_model)
    for _ in range(args.verify_count):
        print('Running JAX diffusion')
        test_input = np.float32(np.random.randn(*jax_model.diffusion_shape))
        test_t = np.random.randint(0, 1000, (1,))
        test_output_j = np.array(jax_net_apply(
            jax_net,
            {'params': jax_params},
            jnp.array(test_input),
            dif.batched_sincos_encode(jnp.array(test_t * jax_model.mode['timestep_mult'])),
        ))
        assert np.all(np.isfinite(test_output_j))
        print('Running PyTorch diffusion')
        # [b, c, w, h] <-> [b, w, h, c]
        test_output_p = np.transpose(pt_model(
            torch.from_numpy(np.transpose(test_input, (0, 3, 1, 2))),
            torch.from_numpy(test_t * jax_model.mode['timestep_mult']),
        ).sample.numpy(), (0, 2, 3, 1))
        assert np.all(np.isfinite(test_output_p))

        mse = np.mean((test_output_j - test_output_p) ** 2)
        print(f'MSE={mse} t={test_t.item()}')
        if mse > 1e-10:
            print('WARNING results are not close')
