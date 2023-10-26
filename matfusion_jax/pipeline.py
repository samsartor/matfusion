import math
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import struct
from jax import Array

from .net.mine import batched_match_reflectance


distributed = os.environ.get('JAX_DISTRIBUTED') == '1'
if distributed:
    func_map = (lambda fun, static_argnums: jax.pmap(fun, axis_name='devices', static_broadcasted_argnums=static_argnums))
    func_mean = (lambda x: lax.pmean(x, 'devices'))
    func_replicate = flax.jax_utils.replicate
    func_unreplicate = flax.jax_utils.unreplicate
else:
    func_map = jax.jit
    func_mean = (lambda x: x)
    func_replicate = (lambda x: x)
    func_unreplicate = (lambda x: x)


loaded_clip = None

def load_clip():
    global loaded_clip
    if loaded_clip is None:
        import clip_jax
        loaded_clip = clip_jax.load('ViT-B/32', 'cpu')
    return loaded_clip

@jax.jit
def clip_image_fn(*args, **kwargs):
    image_fn, text_fn, params, _ = load_clip()
    return image_fn(params, *args, **kwargs)

@jax.jit
def clip_text_fn(*args, **kwargs):
    image_fn, text_fn, params, _ = load_clip()
    return text_fn(params, *args, **kwargs)

def clip_tokenize(*args, **kwargs):
    import clip_jax
    return clip_jax.tokenize(*args, **kwargs)
    

def clip_preprocess(img):
    img = img[:, 16:-16, 16:-16, 0:3]
    img = img - jnp.array([
        0.48145466,
        0.4578275,
        0.40821073
      ])
    img = img / jnp.array([
        0.26862954,
        0.26130258,
        0.27577711
    ])
    img = jnp.transpose(img, (0, 3, 1, 2))
    assert img.shape[1:] == (3, 224, 224)
    return img


class DiffusionModel(struct.PyTreeNode):
    main_state: Any
    main_vars: dict[str, Any]
    ctrl_state: Optional[Any] = None
    ctrl_vars: Optional[dict[str, Any]] = None
    null_state: Optional[Any] = None
    null_vars: Optional[dict[str, Any]] = None


@dataclass
class DiffusionSettings:
    sampler: str = 'euler_a'
    steps: int = 20
    text: Optional[str] = None
    true_render: Optional[Array] = None
    true_svbrdf: Optional[Array] = None
    starting_timestep: Optional[int] = None
    guidance_scale: Optional[float] = None
    derivative_mult: float = 1.0
    sigma_up_mult: float = 1.0
    eta: float = 1.0
    animation: bool = False
    match_color: bool = False
    match_reflectance: bool = False
    late_match: bool = True


@jax.jit
def diffusion_step_impl(
    model: DiffusionModel,
    noisy, sample, input, encoded, key, guidance_scale,
):
    if model.ctrl_state is not None:
        key, dropout_key = jax.random.split(key)
        model_args, _ = model.ctrl_state.apply_fn(
            model.ctrl_vars,
            jnp.concatenate((noisy, input), 3),
            encoded,
            rngs={'dropout': dropout_key},
        )
    else:
        model_args = dict()

    key, dropout_key = jax.random.split(key)
    model_output = model.main_state.apply_fn(
        model.main_vars,
        noisy,
        encoded,
        **model_args,
        rngs={'dropout': dropout_key},
    )
    if guidance_scale is not None:
        assert model.null_state is not None,\
            'guidance scale requires an unconditional backbone model to be loaded'
        null_output = model.null_state.apply_fn(
            model.null_vars,
            sample,
            encoded,
            **model_args,
            rngs={'dropout': dropout_key},
        )
        model_output = null_output + guidance_scale * (model_output - null_output)
    return model_output


@dataclass(frozen=True)
class Diffusion:
    predict: str = 'velocity'
    condition: str = 'none'
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_train_steps: int = 1000
    max_period: int = 10000
    timestep_channels: int = 32
    timestep_mult: float = 1.0
    zero_snr: bool = False

    @classmethod
    def from_mode(cls, mode: Any) -> 'Diffusion':
        return cls(            
            timestep_mult=mode['timestep_mult'],
            timestep_channels=mode['timestep_channels'],
            condition=mode['condition'],
            zero_snr=mode['zero_snr'],
        )

    @cached_property
    def betas(self) -> Array:
        betas = jnp.linspace(self.beta_start, self.beta_end, num=self.num_train_steps) 
        if self.zero_snr:
            # Zero SNNR fix, based on rescale_zero_terminal_snr in huggingface diffusers
            alphas = 1.0 - betas
            alphas_cumprod = jnp.cumprod(alphas)
            alphas_bar_sqrt = alphas_cumprod**0.5

            # Store old values.
            alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
            alphas_bar_sqrt_T = alphas_bar_sqrt[-1]

            # Shift so the last timestep is zero.
            alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T

            # Scale so the first timestep is back to the old value.
            alphas_bar_sqrt = alphas_bar_sqrt * alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

            # Convert alphas_bar_sqrt to betas
            alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
            alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
            alphas = jnp.concatenate([alphas_bar[0:1], alphas])
            betas = 1 - alphas
        return betas

    @cached_property
    def alphas(self) -> Array:
        return 1 - self.betas

    @cached_property
    def alphas_cumprod(self) -> Array:
        return jnp.cumprod(self.alphas)

    @cached_property
    def betas_cumprod(self) -> Array:
        return 1 - self.alphas_cumprod

    @cached_property
    def sigmas(self) -> Array:
        return (self.betas_cumprod / self.alphas_cumprod) ** 0.5

    @cached_property
    def signal_weight(self):
        return self.alphas_cumprod ** 0.5

    @cached_property
    def noise_weight(self):
        return self.betas_cumprod ** 0.5

    def sincos_encode(self, t):
        half_dim = self.timestep_channels // 2
        exponent = -math.log(self.max_period) * jnp.arange(0, half_dim) / half_dim
        thetas = t * jnp.exp(exponent)
        return jnp.concatenate((jnp.sin(thetas), jnp.cos(thetas)), 0)

    def batched_sincos_encode(self, t):
        return jax.vmap(lambda t: self.sincos_encode(t), 0, 0)(t)

    def training_pair(self, signal, noise, timestep):
        if self.predict == 'velocity':
            return (
                self.signal_weight[timestep] * signal + self.noise_weight[timestep] * noise,
                -self.noise_weight[timestep] * signal + self.signal_weight[timestep] * noise,
            )
        elif self.predict == 'signal':
            return (
                self.signal_weight[timestep] * signal + self.noise_weight[timestep] * noise,
                signal,
            )
        else:
            raise Exception(f'unknown predict {self.predict}')

    def predict_signal(self, sample, estimate, timestep, settings: DiffusionSettings, last: bool = False):
        if self.predict == 'velocity':
            signal = self.signal_weight[timestep] * sample - self.noise_weight[timestep] * estimate
        elif self.predict == 'signal':
            signal = estimate
        elif self.predict == 'noise':
            signal = (sample - estimate) / self.signal_weight[timestep]
        else:
            raise Exception(f'unknown predict {self.predict}')
        if settings.match_color and (last or not settings.late_match):
            assert settings.true_render is not None
            signal = signal * 0.5 + 0.5
            true_color = jnp.mean(settings.true_render, axis=(1, 2), keepdims=True)
            albedo = signal[..., 0:3] + signal[..., 3:6]
            signal_color = jnp.mean(albedo, axis=(1, 2), keepdims=True)
            factor = true_color / signal_color
            factor = factor / jnp.mean(factor, axis=3, keepdims=True)
            signal = signal.at[..., 0:3].set(signal[..., 0:3] * factor)
            signal = signal.at[..., 3:6].set(signal[..., 3:6] * factor)
            signal = signal * 2 - 1
        if settings.match_reflectance and (last or not settings.late_match):
            assert settings.true_svbrdf is not None
            signal = signal * 0.5 + 0.5
            signal = batched_match_reflectance(signal, settings.true_svbrdf)
            signal = signal * 2 - 1
        return signal

    def predict_noise(self, sample, estimate, timestep, settings: DiffusionSettings):
        if self.predict == 'velocity':
            return self.noise_weight[timestep] * sample + self.signal_weight[timestep] * estimate
        elif self.predict == 'signal':
            return (sample - estimate) / self.signal_weight[timestep]
        elif self.predict == 'noise':
            return estimate
        else:
            raise Exception(f'unknown predict {self.predict}')

    def loss(
        self,
        model_state,
        model_vars: dict[str, Array],
        ctrl_state,
        ctrl_vars: Optional[dict[str, Array]],
        input: Array,
        svbrdf: Array,
        key,
    ):
        key, tstep_key = jax.random.split(key)
        timesteps = jax.random.randint(tstep_key, (svbrdf.shape[0], 1), minval=0, maxval=self.num_train_steps)
        encoded_t = self.batched_sincos_encode(timesteps * self.timestep_mult)
        timesteps = timesteps.reshape((-1, 1, 1, 1))

        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, svbrdf.shape)
        x, y = self.training_pair(svbrdf, noise, timesteps)
        embedding = None

        if self.condition == 'direct':
            x = jnp.concatenate((x, input), 3)
        elif self.condition == 'clip':
            embedding = clip_image_fn(clip_preprocess(input))

        if embedding is None:
            embedding = encoded_t
        else:
            embedding = jnp.concatenate((encoded_t, embedding), axis=1)

        if ctrl_state is not None:
            # jlprint(ctrl_state.tabulate(ctrl_vars, input, encoded_t))
            # jlprint(ctrl_vars['params']['Dense_0'], input.shape, encoded_t.shape)
            key, dropout_key = jax.random.split(key)
            model_args = ctrl_state.apply_fn(
                ctrl_vars,
                jnp.concatenate((x, input), 3),
                encoded_t,
                rngs={'dropout': dropout_key},
            )
        else:
            model_args = dict()

        key, dropout_key = jax.random.split(key)
        y_est = model_state.apply_fn(
            model_vars,
            x,
            embedding,
            **model_args,
            rngs={'dropout': dropout_key},
        )

        l2_loss = (y_est - y)**2
        # ignore errors less than 1/255, since the ground truth is 8-bit
        l2_loss = l2_loss * (l2_loss > (1/255)**2)
        l2_loss = jnp.mean(l2_loss)

        detailed_loss = {'diffusion': func_mean(l2_loss)}
        loss = l2_loss
        detailed_loss['total'] = loss
        return loss, detailed_loss

    def model_input(self, sample, input, timestep):
        encoded_t = self.sincos_encode(timestep * self.timestep_mult)
        embedding = jnp.expand_dims(encoded_t, 0)
        if self.condition == 'direct':
            x = jnp.concatenate((sample, input), 3)
        elif self.condition == 'clip':
            x = sample
            assert input.shape == (sample.shape[0], 512)
            embedding = jnp.broadcast_to(embedding, (input.shape[0], embedding.shape[1]))
            embedding = jnp.concatenate((embedding, input), axis=1)
        elif self.condition == 'none':
            x = sample
        else:
            assert False, f'unknown condition {self.condition}'

        return embedding, x

    def sample_ddim(
        self,
        model: DiffusionModel,
        input: Array,
        initial: Array,
        key,
        settings: DiffusionSettings,
        create_noise=None,
        callback=lambda x, o, **kwargs: x,
    ):
        if settings.starting_timestep is None:
            timestep = self.num_train_steps - 1
        else:
            timestep = settings.starting_timestep

        if self.condition == 'clip':
            if settings.text is not None:
                input = clip_text_fn(clip_tokenize(settings.text))
            else:
                input = clip_image_fn(clip_preprocess(input))

        sample = initial
        processed = callback(
            initial,
            None,
            input=input,
            initial=initial,
            sigma=1.0,
            sample=sample,
            noise=initial,
        )

        while timestep > 0:
            prev_timestep = timestep - self.num_train_steps // settings.steps

            encoded_t, x = self.model_input(sample, input, timestep)
            estimate = jax.remat(diffusion_step_impl)(
                model,
                x, sample, input, encoded_t,  key,
                settings.guidance_scale,
            )[:, :, :, 0:sample.shape[-1]]
            pred_signal = self.predict_signal(sample, estimate, timestep, settings)
            pred_noise = self.predict_noise(sample, estimate, timestep, settings)

            if prev_timestep > 0:
                prev_sample = self.signal_weight[prev_timestep] * pred_signal \
                    + self.noise_weight[prev_timestep] * pred_noise
            else:
                prev_sample = pred_signal

            sample = prev_sample
            timestep = prev_timestep

            processed = callback(
                pred_signal,
                processed,
                input=input,
                sigma=self.noise_weight[timestep],
                initial=None,
                sample=sample,
                noise=pred_noise,
            )

        return processed

    def sample_euler_ancestral(
        self,
        model: DiffusionModel,
        input: Array,
        initial: Array,
        key,
        settings: DiffusionSettings,
        create_noise=None,
        callback=lambda x, o, **kwargs: x,
    ):
        if settings.starting_timestep is None:
            timestep = self.num_train_steps - 1
        else:
            timestep = settings.starting_timestep

        if self.condition == 'clip':
            if settings.text is not None:
                input = clip_text_fn(clip_tokenize(settings.text))
            else:
                input = clip_image_fn(clip_preprocess(input))

        sample = initial
        processed = callback(
            initial,
            None,
            input=input,
            initial=initial,
            sigma=1.0,
            sample=sample,
            noise=sample,
        )

        while timestep > 0:
            prev_timestep = timestep - self.num_train_steps // settings.steps

            sigma_from = self.sigmas[timestep]
            if prev_timestep > 0:
                sigma_to = self.sigmas[prev_timestep]
            else:
                sigma_to = 0.0
            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5 * settings.eta
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

            scaled_sample = sample / ((sigma_from**2 + 1) ** 0.5)
            encoded_t, x = self.model_input(scaled_sample, input, timestep)
            estimate = jax.remat(diffusion_step_impl)(
                model,
                x, scaled_sample, input, encoded_t, key,
                settings.guidance_scale,
            )[:, :, :, 0:sample.shape[-1]]
            pred_signal = self.predict_signal(scaled_sample, estimate, timestep, settings, last=prev_timestep<=0)
            pred_noise = self.predict_noise(scaled_sample, estimate, timestep, settings)

            key, noise_key = jax.random.split(key)
            if create_noise is None:
                noise = jax.random.normal(noise_key, sample.shape)
            else:
                noise = create_noise(noise_key, sample.shape)

            derivative = (sample - pred_signal) / sigma_from
            dt = sigma_down - sigma_from

            prev_sample = sample + derivative * dt * settings.derivative_mult + noise * sigma_up * settings.sigma_up_mult

            sample = prev_sample
            timestep = prev_timestep

            processed = callback(
                pred_signal,
                processed,
                input=input,
                sigma=self.noise_weight[timestep],
                initial=None,
                sample=sample,
                noise=pred_noise,
                velocity=-estimate,
            )

        return processed

    def sample_euler(
        self,
        model: DiffusionModel,
        input: Array,
        initial: Array,
        key,
        settings: DiffusionSettings,
        create_noise=None,
        callback=lambda x, o, **kwargs: x,
    ):
        if settings.starting_timestep is None:
            timestep = self.num_train_steps - 1
        else:
            timestep = settings.starting_timestep

        if self.condition == 'clip':
            if settings.text is not None:
                input = clip_text_fn(clip_tokenize(settings.text))
            else:
                input = clip_image_fn(clip_preprocess(input))

        sample = initial
        processed = callback(
            initial,
            None,
            input=input,
            initial=initial,
            sigma=1.0,
            sample=sample,
            noise=sample,
        )

        while timestep > 0:
            prev_timestep = timestep - self.num_train_steps // settings.steps

            sigma_from = self.sigmas[timestep]
            if prev_timestep > 0:
                sigma_to = self.sigmas[prev_timestep]
            else:
                sigma_to = 0.0

            scaled_sample = sample / ((sigma_from**2 + 1) ** 0.5)
            encoded_t, x = self.model_input(scaled_sample, input, timestep)
            estimate = jax.remat(diffusion_step_impl)(
                model,
                x, scaled_sample, input, encoded_t, key,
                settings.guidance_scale,
            )[:, :, :, 0:sample.shape[-1]]
            pred_signal = self.predict_signal(scaled_sample, estimate, timestep, settings)
            pred_noise = self.predict_noise(scaled_sample, estimate, timestep, settings)

            derivative = (sample - pred_signal) / sigma_from
            dt = sigma_to - sigma_from
            prev_sample = sample + derivative * dt * settings.derivative_mult

            sample = prev_sample
            timestep = prev_timestep

            processed = callback(
                pred_signal,
                processed,
                input=input,
                sigma=self.noise_weight[timestep],
                initial=None,
                sample=sample,
                noise=pred_noise,
            )

        return processed

    def sample_heun(
        self,
        model: DiffusionModel,
        input: Array,
        initial: Array,
        key,
        settings: DiffusionSettings,
        create_noise=None,
        callback=lambda x, o, **kwargs: x,
    ):
        if settings.starting_timestep is None:
            timestep = self.num_train_steps - 1
        else:
            timestep = settings.starting_timestep

        if self.condition == 'clip':
            if settings.text is not None:
                input = clip_text_fn(clip_tokenize(settings.text))
            else:
                input = clip_image_fn(clip_preprocess(input))

        sample = initial
        processed = callback(
            initial,
            None,
            input=input,
            initial=initial,
            sigma=1.0,
            sample=sample,
            noise=sample,
        )

        last_derivative = None
        while timestep > 0:
            prev_timestep = timestep - self.num_train_steps // settings.steps

            sigma_from = self.sigmas[timestep]
            if prev_timestep > 0:
                sigma_to = self.sigmas[prev_timestep]
            else:
                sigma_to = 0.0

            scaled_sample = sample / ((sigma_from**2 + 1) ** 0.5)
            encoded_t, x = self.model_input(scaled_sample, input, timestep)
            estimate = jax.remat(diffusion_step_impl)(
                model,
                x, scaled_sample, input, encoded_t, key,
                settings.guidance_scale,
            )[:, :, :, 0:sample.shape[-1]]
            pred_signal = self.predict_signal(scaled_sample, estimate, timestep, settings)
            pred_noise = self.predict_noise(scaled_sample, estimate, timestep, settings)

            derivative = (sample - pred_signal) / sigma_from
            dt = sigma_to - sigma_from

            if last_derivative is not None and prev_timestep > 0:
                prev_sample = sample + (0.5 * derivative + 0.5 * last_derivative) * dt * settings.derivative_mult
            else:
                prev_sample = sample + derivative * dt * settings.derivative_mult

            sample = prev_sample
            timestep = prev_timestep
            last_derivative = derivative

            processed = callback(
                pred_signal,
                processed,
                input=input,
                sigma=self.noise_weight[timestep],
                initial=None,
                sample=sample,
                noise=pred_noise,
            )

        return processed

    def init_noise_scale(self, sampler):
        if sampler == 'ddim':
            return 1.0
        elif sampler in ('euler', 'euler_a', 'heun'):
            return jnp.max(self.sigmas).item()
        else:
            raise ValueError(f'unknown sampler {sampler}')

    def sample(self, *args, **kwargs):
        settings = kwargs.get('settings', DiffusionSettings())
        if settings.sampler == 'ddim':
            return self.sample_ddim(*args, **kwargs)
        elif settings.sampler == 'euler':
            return self.sample_euler(*args, **kwargs)
        elif settings.sampler == 'euler_a':
            return self.sample_euler_ancestral(*args, **kwargs)
        elif settings.sampler == 'heun':
            return self.sample_heun(*args, **kwargs)
        else:
            raise ValueError(f'unknown sampler {settings.sampler}')

