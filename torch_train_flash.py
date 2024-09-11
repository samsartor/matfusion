import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Train an svbrdf model.')
parser.add_argument('--epocs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation', type=int, default=4)
parser.add_argument('--eval_every', type=int, default=200)
parser.add_argument('--finetune_checkpoint', type=Path, required=True)
parser.add_argument('--output_checkpoint', type=Path, required=True)
parser.add_argument('--base_lr', type=float, default=2e-5)
parser.add_argument('--cosine_lr', type=bool, default=False)
parser.add_argument('--compile', type=bool, default=False)
parser.add_argument('--tensorboard', type=bool, default=True)
args = parser.parse_args()

import json
from typing import Any
import diffusers
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from matfusion_jax.data import Generator
import numpy as np
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter


# Rust object to create training batches
gen = Generator(
    'datasets/train_rasterized.yml',
    seed=0,
    batch_size=args.batch_size,
    replicates=1,
    worker_count=16,
)

# Rust object to create eval batches
eval_gen = Generator(
    'datasets/test_rasterized.yml',
    seed=0,
    batch_size=1,
    replicates=1,
    worker_count=1,
)

device = torch.device('cuda')
dtype = torch.bfloat16

# The model to finetune
model = diffusers.UNet2DModel.from_pretrained(args.finetune_checkpoint)
assert isinstance(model, diffusers.UNet2DModel)
model = model.to(device=device, dtype=dtype)
timestep_mult = model.config.get('timestep_mult', 1/1000)

# An average of the model parameters through training
ema_model = diffusers.training_utils.EMAModel(model.parameters())

# Direct conditioning
in_channels = 10 + 3 + 3
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(in_channels, model.conv_in.out_channels, kernel_size=3, padding=(1, 1)).to(device=device, dtype=dtype)
    new_conv_in.weight.copy_(F.pad(model.conv_in.weight, (0, 0, 0, in_channels - model.conv_in.in_channels, 0, 0, 0, 0)))
    assert model.conv_in.bias is not None and new_conv_in.bias is not None
    new_conv_in.bias.copy_(F.pad(model.conv_in.bias, (0, in_channels - model.conv_in.in_channels)))
model.conv_in = new_conv_in

# Make the model faster to run, given we wait for a long time up-front
if args.compile:
    model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

# The diffusion schedule to use when evaluating
schedule = diffusers.schedulers.EulerAncestralDiscreteScheduler(
    beta_schedule='linear',
    prediction_type='v_prediction',
    timestep_spacing="linspace",
)
# The schedule to use for training (used to correctly mix noise and signal)
ddim_schedule = diffusers.schedulers.DDIMScheduler(
    beta_schedule='linear',
    prediction_type='v_prediction',
    rescale_betas_zero_snr=True,
    clip_sample=False,
    timestep_spacing="linspace",
)

@torch.no_grad()
def eval_model(test_batch):
    model.eval()
    if ema_model.model_cls is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    
    schedule.set_timesteps(20)
    y = torch.randn(test_batch['svbrdf'].shape[0], 10, 256, 256, device=device, dtype=dtype)
    y = y * schedule.init_noise_sigma

    image = rearrange(torch.tensor(test_batch['rast_flash'], device=device, dtype=dtype), 'b h w c -> b c h w')
    halfway = rearrange(torch.tensor(test_batch['rast_halfway'], device=device, dtype=dtype), 'b h w c -> b c h w')

    for t in schedule.timesteps:
        noisy_svbrdf = schedule.scale_model_input(y, t)
        model_output = model(
            torch.cat((noisy_svbrdf, image * 2 - 1, halfway * 2 - 1), 1),
            t.to(device=device, dtype=dtype)*timestep_mult,
        ).sample

        step_output = schedule.step(model_output, t, y)
        assert isinstance(step_output, diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput)
        assert step_output.pred_original_sample is not None

        y = step_output.prev_sample

        svbrdf_est = (step_output.pred_original_sample * 0.5 + 0.5).clamp(0, 1).cpu()

    if ema_model.model_cls is not None:
        ema_model.restore(model.parameters())

    return svbrdf_est, image, halfway

def train_loss(batch) -> torch.Tensor:
    model.train()
    
    svbrdf = rearrange(torch.tensor(batch['svbrdf'], device=device, dtype=dtype), 'b h w c -> b c h w') * 2 - 1
    noise = torch.randn(svbrdf.shape[0], 10, 256, 256, device=device, dtype=dtype)
    image = rearrange(torch.tensor(batch['rast_flash'], device=device, dtype=dtype), 'b h w c -> b c h w') * 2 - 1
    halfway = rearrange(torch.tensor(batch['rast_halfway'], device=device, dtype=dtype), 'b h w c -> b c h w') * 2 - 1
    t = torch.randint(0, 1000, [svbrdf.shape[0]], device=device)

    noisy_svbrdf = ddim_schedule.add_noise(svbrdf, noise, t)
    velocity = ddim_schedule.get_velocity(svbrdf, noise, t)

    model_output = model(
        torch.cat((noisy_svbrdf, image, halfway), 1),
        t*timestep_mult,
    ).sample

    return F.mse_loss(model_output, velocity)

# Start the dataloader
gen.begin()
print(f'Found {gen.total_samples} training SVBRDFs')

# Setup logging and a progress bar
if args.tensorboard:
    tb_writer = SummaryWriter()
progress = tqdm(total=args.epocs*gen.total_steps)

# Main training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=1e-2)
train_step = 0
for epoc_num in range(1, args.epocs+1):
    while True:
        # Evaluate the model every args.eval_every steps
        if train_step % args.eval_every == 0:
            eval_batch = eval_gen.take()
            if eval_batch is None:
                eval_gen.begin()
                eval_batch = eval_gen.take()
                assert eval_batch is not None
            generated, image, halfway = eval_model(eval_batch)
            generated = torch.cat([
                generated[:, 0:3],
                generated[:, 3:6],
                generated[:, 6:7].repeat((1, 3, 1, 1)),
                generated[:, 7:10],
            ], 3)
            if args.tensorboard:
                tb_writer.add_images('eval/image', image, train_step)
                tb_writer.add_images('eval/halfway', halfway, train_step)
                tb_writer.add_images('eval/output', generated, train_step)

        # Get a batch of training data
        batch = gen.take()
        if batch is None:
            break

        # Backpropagate
        loss = train_loss(batch)
        (loss / args.accumulation).backward()

        # Log the loss
        if args.tensorboard:
            tb_writer.add_scalar("loss", loss.item(), train_step)

        # Take an optimizer step every args.accumulation steps
        if train_step % args.accumulation == 0:
            lr = args.base_lr
            if args.cosine_lr:
                lr *= math.cos(train_step / (gen.total_steps * args.epocs) * math.pi) * 0.5 + 0.5
            optimizer.defaults['lr'] = lr
            if args.tensorboard:
                tb_writer.add_scalar("lr", lr, train_step)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema_model.step(model.parameters())

        train_step += 1
        progress.update(1)

    # End of epoc, save the model
    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
    model.save_pretrained(args.output_checkpoint)
    ema_model.restore(model.parameters())

    # Start loading a new epoc
    gen.begin()

if args.tensorboard:
    tb_writer.close()
