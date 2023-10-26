import math

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.random as random
import numpy as np
from jax import Array


def nprast_D_GGX(roughness, NdotH, eps):
    alpha = jnp.square(roughness)
    under_d = 1 / jnp.maximum(
        eps,
        (jnp.square(NdotH) * (jnp.square(alpha) - 1.0) + 1.0))
    return jnp.square(alpha * under_d) / math.pi


def nprast_F_GGX(specular, VdotH, eps):
    sphg = jnp.power(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
    return specular + (1.0 - specular) * sphg


def nprast_G_GGX(roughness, NdotL, NdotV, eps):
    k = jnp.maximum(0.5 * jnp.square(roughness), eps)
    return nprast_G1(NdotL, k, eps) * nprast_G1(NdotV, k, eps)


def nprast_G1(NdotW, k, eps):
    return 1.0 / (NdotW * (1.0 - k) + k)


def nprast_render_pixel(
    svbrdf: Array, wi: Array, wo: Array, hw: Array,
    include_diffuse=True, tonemap=True, eps=1e-8, exposure=1.0, dot_in_denom=False,
    ambient=0.0,
):
    diffuse = jnp.clip(svbrdf[0:3], 0.0, 1.0)
    specular = jnp.clip(svbrdf[3:6], 0.0, 1.0)
    roughness = jnp.clip(svbrdf[6:7], 0.01, 1.0)
    normals = 2 * svbrdf[7:10] - 1
    normals /= jla.norm(normals)
    normals *= jnp.sign(normals[3])
    NdotH = jnp.maximum(jnp.dot(normals, hw), eps)
    NdotL = jnp.maximum(jnp.dot(normals, wi), eps)
    NdotV = jnp.maximum(jnp.dot(normals, wo), eps)
    VdotH = jnp.maximum(jnp.dot(wo, hw), eps)

    ambient_result = ambient * (diffuse + specular)

    D_rendered = nprast_D_GGX(roughness, NdotH, eps)
    G_rendered = nprast_G_GGX(roughness, NdotL, NdotV, eps)
    F_rendered = nprast_F_GGX(specular, VdotH, eps)
    if dot_in_denom:
        specular_result = D_rendered * G_rendered * F_rendered / (4 * NdotL * NdotV + eps)
    else:
        specular_result = D_rendered * G_rendered * F_rendered / (4 + eps)
    specular_result = specular_result * NdotL * math.pi

    if include_diffuse:
        diffuse_result = diffuse * NdotL
    else:
        diffuse_result = 0.0

    result = specular_result + diffuse_result + ambient_result

    if tonemap:
        return jnp.clip(result * exposure, 0.0, 1.0)
    else:
        return result * exposure


def height2normals(height: Array):
    height = jnp.pad(height, ((1, 1), (1, 1), (0, 0)), mode='edge')
    dy = (height[2:, 1:-1, :] - height[:-2, 1:-1, :]) * 0.5 * height.shape[0]
    dx = (height[1:-1, :-2, :] - height[1:-1, 2:, :]) * 0.5 * height.shape[1]
    dz = jnp.ones(dx.shape)
    normals = jnp.concatenate((dx, dy, dz), axis=2)
    normals /= jla.norm(normals, axis=2, keepdims=True)
    return normals / 2 + 0.5


def height2normals_batch(height: Array):
    return jax.vmap(
        lambda h: height2normals(h),
        0,
        0,
    )(height)


def nprast_render(svbrdf: Array, wi: Array, wo: Array, hw: Array, geo='normals', **kwargs):
    def pixel_func(s, wi, wo, hw):
        return nprast_render_pixel(s, wi, wo, hw, **kwargs)
    if geo == 'height':
        svbrdf = jnp.concatenate(
            (svbrdf[:, :, 0:7], height2normals(svbrdf[:, :, 7:8])),
            axis=2,
        )
    return jax.vmap(jax.vmap(pixel_func, 0, 0), 0, 0)(svbrdf, wi, wo, hw)


def nprast_wioh_basic(h: int, w: int, camera_pos: Array, light_pos: Array):
    surface = jnp.indices((h, w)) + 0.5
    surface = jnp.stack((
        2 * surface[1, :, :] / h - 1,
        -2 * surface[0, :, :] / w + 1,
        jnp.zeros((h, w), jnp.float32),
    ), axis=-1)

    wi = light_pos - surface
    wi /= jla.norm(wi, axis=2, keepdims=True)

    wo = camera_pos - surface
    wo /= jla.norm(wo, axis=2, keepdims=True)

    hw = wi + wo
    hw /= jla.norm(hw, axis=2, keepdims=True)

    return (wi, wo, hw)


def nprast_wioh(
    h: int,
    w: int,
    highlight_center: list[float] | Array,
    camera_pos: list[float] | Array,
    light_distance_scale: Array | float = 1.0,
):
    highlight_center = jnp.append(jnp.array(highlight_center), 0)
    camera_pos = jnp.array(camera_pos)
    light_pos = (camera_pos - highlight_center) * jnp.array([
        -light_distance_scale,
        -light_distance_scale,
        light_distance_scale,
    ]) + highlight_center

    surface = jnp.indices((h, w)) + 0.5
    surface = jnp.stack((
        2 * surface[1, :, :] / h - 1,
        -2 * surface[0, :, :] / w + 1,
        jnp.zeros((h, w), jnp.float32),
    ), axis=-1)

    wi = light_pos - surface
    wi /= jla.norm(wi, axis=2, keepdims=True)

    wo = camera_pos - surface
    wo /= jla.norm(wo, axis=2, keepdims=True)

    hw = wi + wo
    hw /= jla.norm(hw, axis=2, keepdims=True)

    return (wi, wo, hw)


def nprast_ortho_wioh(
    w: int,
    h: int,
    light_pos: list[float] | Array,
):
    light_pos = jnp.array(light_pos)
    surface = jnp.indices((h, w)) + 0.5
    surface = jnp.stack((
        2 * surface[1, :, :] / h - 1,
        -2 * surface[0, :, :] / w + 1,
        jnp.zeros((h, w), jnp.float32),
    ), axis=-1)

    wi = light_pos - surface
    wi /= jla.norm(wi, axis=2, keepdims=True)

    wo = jnp.array([0, 0, 1], dtype=wi.dtype)
    wo = jnp.expand_dims(wo, (0, 1))
    wo = jnp.tile(wo, (h, w, 1))

    hw = wi + wo
    hw /= jla.norm(hw, axis=2, keepdims=True)

    return (wi, wo, hw)


def nprast_flash_rerender(
    svbrdf: Array,
    distance=4,
    **kwargs,
):
    h, w, _ = svbrdf.shape
    wi, wo, hw = nprast_wioh(h, w, [0, 0], [0, 0, distance])
    r = nprast_render(svbrdf, wi, wo, hw, **kwargs)
    return r, hw


def nprast_flash_rerender_batch(svbrdf: Array, distance=4, **kwargs):
    if jnp.isscalar(distance):
        return jax.vmap(
            lambda s: nprast_flash_rerender(s, distance=distance, **kwargs),
            0,
            0,
        )(svbrdf)
    else:
        return jax.vmap(
            lambda s, d: nprast_flash_rerender(s, distance=d, **kwargs),
            0,
            0,
        )(svbrdf, distance)


def nprast_ortho_render(svbrdf: Array, flash_pos, **kwargs):
    h, w, _ = svbrdf.shape
    wi, wo, hw = nprast_ortho_wioh(h, w, flash_pos)
    r = nprast_render(svbrdf, wi, wo, hw, **kwargs)
    return r, hw


def nprast_ortho_render_batch(svbrdf: Array, flash_pos, **kwargs):
    return jax.vmap(
        lambda s, p: nprast_ortho_render(s, flash_pos=p, **kwargs),
        0,
        0,
    )(svbrdf, flash_pos)


def random_direction(key):
    key1, key2 = random.split(key)
    phi = random.uniform(key1, minval=0, maxval=math.pi * 2)
    costheta = random.uniform(key2, minval=0, maxval=1)

    theta = jnp.arccos(costheta)
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack((x, y, z))


def nprast_random_render(svbrdf, key, **kwargs):
    h, w, _ = svbrdf.shape
    key, subkey = random.split(key)
    highlight_center = random.uniform(subkey, minval=-1.5, maxval=1.5, shape=(2,), dtype=jnp.float32)
    key, subkey = random.split(key)
    camera_position = random_direction(subkey) * 4
    key, subkey = random.split(key)
    light_distance_scale = 1 + random.uniform(subkey, minval=-0.5, maxval=0.5, shape=(), dtype=jnp.float32)
    wi, wo, hw = nprast_wioh(h, w, highlight_center, camera_position, light_distance_scale)
    r = nprast_render(svbrdf, wi, wo, hw, **kwargs)
    return r, hw


def nprast_dome_render(svbrdf, angles, camera_distance=2.41, light_distance=2.41, **kwargs):
    h, w, _ = svbrdf.shape

    phi = angles[0] * math.pi * 2
    costheta = angles[1]

    theta = jnp.arccos(costheta)
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    light_pos = jnp.stack((x, y, z)) * light_distance

    wi, wo, hw = nprast_wioh_basic(h, w, jnp.array([0.0, 0.0, camera_distance]), light_pos)
    r = nprast_render(svbrdf, wi, wo, hw, **kwargs)
    return r, hw


def nprast_topdown_render(svbrdf, h_pos, camera_distance=2.41, light_distance=2.41, **kwargs):
    h, w, _ = svbrdf.shape

    cam_pos = jnp.array([0.0, 0.0, camera_distance])
    h_pos = jnp.append(h_pos, 0.0)
    light_pos = cam_pos
    light_pos -= h_pos
    light_pos /= jla.norm(light_pos)
    light_pos *= jnp.array([-1, -1, 1.0])
    light_pos *= light_distance
    light_pos += h_pos

    wi, wo, hw = nprast_wioh_basic(h, w, cam_pos, light_pos)
    r = nprast_render(svbrdf, wi, wo, hw, **kwargs)
    return r, hw


def nprast_basic_render(svbrdf, camera_pos, light_pos, **kwargs):
    h, w, _ = svbrdf.shape
    wi, wo, hw = nprast_wioh_basic(h, w, camera_pos, light_pos)
    r = nprast_render(svbrdf, wi, wo, hw, **kwargs)
    return r, hw


def nprast_animated_render(svbrdf, distance=4.0, frames=30, **kwargs):
    h, w, _ = svbrdf.shape
    renders = []
    for theta in np.linspace(0.0, 2 * math.pi, num=frames):
        wi, wo, hw = nprast_wioh(h, w, [math.sin(theta), math.cos(theta)], [0, 0, distance])
        renders.append(nprast_render(svbrdf, wi, wo, hw, **kwargs))
    return jnp.stack(renders)


def nprast_random_render_batch(svbrdf, key, **kwargs):
    keys = random.split(key, svbrdf.shape[0])
    return jax.vmap(
        lambda s, k: nprast_random_render(s, k, **kwargs),
        0,
        0,
    )(svbrdf, keys)


def nprast_random_render_multi(svbrdf, key, count, **kwargs):
    keys = random.split(key, count)
    return jax.vmap(
        lambda k: nprast_random_render(svbrdf, k, **kwargs),
        0,
        0,
    )(keys)


def nprast_dome_render_batch(svbrdf, angles, **kwargs):
    return jax.vmap(
        lambda s, a: nprast_dome_render(s, a, **kwargs),
        0,
        0,
    )(svbrdf, angles)


def nprast_dome_render_multi(svbrdf, angles, **kwargs):
    return jax.vmap(
        lambda a: nprast_dome_render(svbrdf, a, **kwargs),
        0,
        0,
    )(angles)


def nprast_topdown_render_batch(svbrdf, h_pos, **kwargs):
    return jax.vmap(
        lambda s, a: nprast_topdown_render(s, a, **kwargs),
        0,
        0,
    )(svbrdf, h_pos)


def nprast_topdown_render_multi(svbrdf, h_pos, **kwargs):
    return jax.vmap(
        lambda a: nprast_topdown_render(svbrdf, a, **kwargs),
        0,
        0,
    )(h_pos)


def nprast_basic_render_batch(svbrdf, camera_pos, light_pos, **kwargs):
    return jax.vmap(
        lambda s, c, l: nprast_basic_render(s, c, l, **kwargs),
        0,
        0,
    )(svbrdf, camera_pos, light_pos)


def nprast_basic_render_multi(svbrdf, camera_pos, light_pos, **kwargs):
    return jax.vmap(
        lambda c, l: nprast_basic_render(svbrdf, c, l, **kwargs),
        0,
        0,
    )(camera_pos, light_pos)
