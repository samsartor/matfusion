#![allow(non_snake_case)]

use cgmath::{dot, prelude::*, vec3, Vector2, Vector3};
use ndarray::{s, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

pub type Vec3 = Vector3<f32>;
pub type Vec2 = Vector2<f32>;

fn square(x: f32) -> f32 {
    x * x
}

fn maximum(x: f32, y: f32) -> f32 {
    x.max(y)
}

fn power(b: f32, e: f32) -> f32 {
    b.powf(e)
}

fn D_GGX(roughness: f32, NdotH: f32, eps: f32) -> f32 {
    let alpha = square(roughness);
    let under_d = 1.0 / maximum(square(NdotH) * (square(alpha) - 1.0) + 1.0, eps);
    square(alpha * under_d) / PI
}

fn F_GGX(specular: f32, VdotH: f32, _eps: f32) -> f32 {
    let sphg = power(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
    specular + (1.0 - specular) * sphg
}

fn G_GGX(roughness: f32, NdotL: f32, NdotV: f32, eps: f32) -> f32 {
    let k = maximum(0.5 * square(roughness), eps);
    G1(NdotL, k, eps) * G1(NdotV, k, eps)
}

fn G1(NdotW: f32, k: f32, _eps: f32) -> f32 {
    1.0 / (NdotW * (1.0 - k) + k)
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct Options {
    pub include_diffuse: bool,
    pub texture_gamma: f32,
    pub render_gamma: f32,
    pub render_clip: bool,
    pub eps: f32,
    pub exposure: f32,
    pub ambient: f32,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            include_diffuse: true,
            texture_gamma: 2.2,
            render_gamma: 2.2,
            render_clip: true,
            eps: 1e-8,
            exposure: 1.0,
            ambient: 0.0,
        }
    }
}

pub fn render_pixel(
    svbrdf: ArrayView1<f32>,
    mut output: ArrayViewMut1<f32>,
    wi: Vec3,
    wo: Vec3,
    hw: Vec3,
    &Options {
        include_diffuse,
        texture_gamma,
        render_gamma,
        render_clip,
        eps,
        exposure,
        ambient,
    }: &Options,
) {
    let diffuse =
        vec3(svbrdf[0], svbrdf[1], svbrdf[2]).map(|x| x.clamp(0.0, 1.0).powf(texture_gamma));
    let specular =
        vec3(svbrdf[3], svbrdf[4], svbrdf[5]).map(|x| x.clamp(0.0, 1.0).powf(texture_gamma));
    let roughness = svbrdf[6].clamp(0.01, 1.0);
    let normals = vec3(svbrdf[7], svbrdf[8], svbrdf[9])
        .map(|x| x * 2.0 - 1.0)
        .normalize();
    let normals = normals * normals.z.signum();

    let NdotH = maximum(dot(normals, hw), eps);
    let NdotL = maximum(dot(normals, wi), eps);
    let NdotV = maximum(dot(normals, wo), eps);
    let VdotH = maximum(dot(wo, hw), eps);

    let ambient_result = ambient * (diffuse + specular);

    let D_rendered = D_GGX(roughness, NdotH, eps);
    let G_rendered = G_GGX(roughness, NdotL, NdotV, eps);
    let F_rendered = specular.map(|s| F_GGX(s, VdotH, eps));
    let specular_result = D_rendered * G_rendered * F_rendered / (4.0 + eps);
    let specular_result = specular_result * NdotL * PI;

    let diffuse_result = if include_diffuse {
        diffuse * NdotL
    } else {
        vec3(0.0, 0.0, 0.0)
    };

    let result = specular_result + diffuse_result + ambient_result;
    let result = result.map(|x| {
        let x = x * exposure;
        let x = if render_clip {
            x.clamp(0.0, 1.0)
        } else {
            x.max(0.0)
        };
        let x = x.powf(render_gamma.recip());
        x
    });

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    if output.len() > 3 {
        output[3] = hw.x * 0.5 + 0.5;
        output[4] = hw.y * 0.5 + 0.5;
        output[5] = hw.z * 0.5 + 0.5;
    }
}

pub fn render(
    svbrdf: ArrayView3<f32>,
    mut output: ArrayViewMut3<f32>,
    wi: impl Fn(Vec3) -> Vec3,
    wo: impl Fn(Vec3) -> Vec3,
    opts: &Options,
) {
    let (w, h, svbrdf_dims) = svbrdf.dim();
    assert_eq!(svbrdf_dims, 10);
    let (output_w, output_h, output_dims) = output.dim();
    assert_eq!(w, output_w);
    assert_eq!(h, output_h);
    assert!(output_dims == 3 || output_dims == 6);

    let h = svbrdf.shape()[0];
    let w = svbrdf.shape()[1];
    for i in 0..h {
        for j in 0..w {
            let p = vec3(
                (j as f32 / h as f32) * 2.0 - 1.0,
                -(i as f32 / h as f32) * 2.0 + 1.0,
                0.0,
            );
            let wi = wi(p).normalize();
            let wo = wo(p).normalize();
            let hw = (wi + wo).normalize();
            render_pixel(
                svbrdf.slice(s![i, j, ..]),
                output.slice_mut(s![i, j, ..]),
                wi,
                wo,
                hw,
                opts,
            );
        }
    }
}

pub fn render_basic(
    svbrdf: ArrayView3<f32>,
    output: ArrayViewMut3<f32>,
    light_pos: Vec3,
    camera_pos: Vec3,
    opts: &Options,
) {
    render(
        svbrdf,
        output,
        |surface| light_pos - surface,
        |surface| camera_pos - surface,
        opts,
    );
}

pub fn render_highlight(
    svbrdf: ArrayView3<f32>,
    output: ArrayViewMut3<f32>,
    highlight_center: Vec2,
    light_distance: f32,
    camera_pos: Vec3,
    opts: &Options,
) {
    let light_pos = vec3(
        2.0 * highlight_center.x - camera_pos.x,
        2.0 * highlight_center.y - camera_pos.y,
        camera_pos.z,
    ) / camera_pos.z
        * light_distance;
    render_basic(svbrdf, output, light_pos, camera_pos, opts);
}

pub fn render_ortho(
    svbrdf: ArrayView3<f32>,
    output: ArrayViewMut3<f32>,
    light_pos: Vec3,
    opts: &Options,
) {
    render(
        svbrdf,
        output,
        |surface| light_pos - surface,
        |_| vec3(0.0, 0.0, 1.0),
        opts,
    );
}

pub fn render_colocated(
    svbrdf: ArrayView3<f32>,
    output: ArrayViewMut3<f32>,
    light_distance: f32,
    camera_distance: f32,
    opts: &Options,
) {
    render_basic(
        svbrdf,
        output,
        vec3(0.0, 0.0, light_distance),
        vec3(0.0, 0.0, camera_distance),
        opts,
    );
}

pub fn render_dome(
    svbrdf: ArrayView3<f32>,
    output: ArrayViewMut3<f32>,
    light_u: f32,
    light_v: f32,
    light_distance: f32,
    camera_distance: f32,
    opts: &Options,
) {
    let phi = light_u * (PI * 2.0);
    let theta = light_v.acos();

    let x = theta.sin() * phi.cos();
    let y = theta.sin() * phi.sin();
    let z = theta.cos();

    render_basic(
        svbrdf,
        output,
        vec3(x, y, z) * light_distance,
        vec3(0.0, 0.0, camera_distance),
        opts,
    );
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum RenderArgs {
    Basic {
        camera_pos: Vec3,
        light_pos: Vec3,
        #[serde(default)]
        opts: Options,
    },
    Highlight {
        camera_pos: Vec3,
        light_distance: f32,
        highlight_center: Vec2,
        #[serde(default)]
        opts: Options,
    },
    Dome {
        light_u: f32,
        light_v: f32,
        light_distance: f32,
        camera_distance: f32,
        #[serde(default)]
        opts: Options,
    },
    Colocated {
        light_distance: f32,
        camera_distance: f32,
        #[serde(default)]
        opts: Options,
    },
    Ortho {
        light_pos: Vec3,
        #[serde(default)]
        opts: Options,
    },
}

pub fn render_any(svbrdf: ArrayView3<f32>, output: ArrayViewMut3<f32>, args: RenderArgs) {
    use RenderArgs::*;
    match args {
        Basic {
            camera_pos,
            light_pos,
            opts,
        } => render_basic(svbrdf, output, light_pos, camera_pos, &opts),
        Highlight {
            camera_pos,
            light_distance,
            highlight_center,
            opts,
        } => render_highlight(
            svbrdf,
            output,
            highlight_center,
            light_distance,
            camera_pos,
            &opts,
        ),
        Dome {
            light_u,
            light_v,
            light_distance,
            camera_distance,
            opts,
        } => render_dome(
            svbrdf,
            output,
            light_u,
            light_v,
            light_distance,
            camera_distance,
            &opts,
        ),
        Colocated {
            light_distance,
            camera_distance,
            opts,
        } => render_colocated(svbrdf, output, light_distance, camera_distance, &opts),
        Ortho { light_pos, opts } => render_ortho(svbrdf, output, light_pos, &opts),
    }
}
