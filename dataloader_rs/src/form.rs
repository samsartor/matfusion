use crate::{data_path, downcale, open_texture, warp, DataId, DatasetConfig, Part};
use anyhow::{anyhow, ensure, Context, Error};
use cgmath::{point2, Vector2};
use ndarray::{s, Array3, ArrayView3, ArrayViewMut3, Axis};
use rand::{seq::IteratorRandom, Rng};
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

pub enum RenderPart {
    Env(char),
    Flash(char),
    Combined(char),
}

pub struct Render {
    pub id: DataId,
    pub which: char,
    pub meta: RenderMeta,
    pub crop: f32,
    pub uvs: Array3<f32>,
    pub image: Array3<f32>,
    pub other_image: Option<Array3<f32>>,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct BoundsMeta {
    pub x: (f32, f32),
    pub y: (f32, f32),
    pub z: (f32, f32),
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct ViewMeta {
    pub distance: Option<f32>,
    pub flash_distance: Option<f32>,
    pub flash_offset: Option<Vector2<f32>>,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct RenderMeta {
    pub flash_offset: Option<Vector2<f32>>,
    pub distance: Option<f32>,
    pub world: PathBuf,
    pub views: HashMap<char, ViewMeta>,
}

impl Render {
    pub fn matching_svbrdf(
        &self,
        root: &DatasetConfig,
        rng: &mut impl Rng,
    ) -> Result<Array3<f32>, Error> {
        let svbrdf = form_svbrdf(root, &self.id, rng, None, |p| p)?;
        let mut dest = Array3::zeros([self.image.dim().0, self.image.dim().1, svbrdf.dim().2]);
        let mut uvs = self.uvs.clone();
        uvs.slice_mut(s![.., .., 1]).mapv_inplace(|y| -y);
        uvs.mapv_inplace(|v| v / 2.0 + 0.5);
        uvs.slice_mut(s![.., .., 0])
            .mapv_inplace(|x| x * (svbrdf.dim().1 - 1) as f32);
        uvs.slice_mut(s![.., .., 1])
            .mapv_inplace(|y| y * (svbrdf.dim().0 - 1) as f32);
        let uvs = warp::UvBuffer(uvs.view());
        warp::reverse_warp(
            warp::SvbrdfInterpolation(warp::BilinearInterpolation(svbrdf.view())),
            dest.view_mut(),
            |pt| uvs.query(pt),
            |_, mut arr| arr.fill(0.0),
        );
        Ok(dest)
    }

    pub fn flash_distance(&self) -> f32 {
        let v = &self.meta.views[&self.which];
        // unless otherwise specified, the light was colocated with a 1:1 focal length camera
        v.flash_distance
            .or(v.distance)
            .or(self.meta.distance)
            .unwrap_or(2.0)
            / self.crop
    }

    pub fn flash_offset(&self) -> Vector2<f32> {
        let v = &self.meta.views[&self.which];
        v.flash_offset
            .or(self.meta.flash_offset)
            .unwrap_or(Vector2::new(0.0, 0.0))
    }

    pub fn view_distance(&self) -> Option<f32> {
        self.meta.views[&self.which].distance.map(|d| d / self.crop)
    }
}

pub fn decide_whitepoint(image: ArrayView3<f32>) -> f32 {
    let mut vec: Vec<f32> = image.iter().copied().filter(|x| x.is_finite()).collect();
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let whitepoint = vec[(vec.len() - 1) * 95 / 100] * 1.2;
    whitepoint
}

pub fn correct_exposure(mut image: ArrayViewMut3<f32>) -> f32 {
    let whitepoint = decide_whitepoint(image.view());
    image.mapv_inplace(|x| x / whitepoint);
    whitepoint
}

pub fn form_render(
    root: &DatasetConfig,
    id: &DataId,
    rng: &mut impl Rng,
    crop: Option<f32>,
    resolution: usize,
    flash: f32,
    other_flash: Option<f32>,
) -> Result<Render, Error> {
    // Load metadata.
    let meta_path = data_path(root, id, &Part::RenderMeta)?;
    let meta: RenderMeta = serde_json::from_reader(
        File::open(&meta_path).context(format!("opening metadata at {}", meta_path.display()))?,
    )?;
    let which = *meta.views.keys().choose(rng).ok_or(anyhow!("no views"))?;

    // Functions to load renders.
    let mut env_whitepoint = None;
    let mut env = |which: char| -> Result<_, Error> {
        let mut env = open_texture(root, id, &Part::RenderEnv { which })?;
        let whitepoint = *env_whitepoint.get_or_insert_with(|| decide_whitepoint(env.view()));
        env.mapv_inplace(|x| x / whitepoint);
        Ok(env)
    };
    let mut fla_whitepoint = None;
    let mut fla = |which: char| -> Result<_, Error> {
        let mut fla = open_texture(root, id, &Part::RenderFlash { which })?;
        let whitepoint = *fla_whitepoint.get_or_insert_with(|| decide_whitepoint(fla.view()));
        fla.mapv_inplace(|x| x / whitepoint);
        Ok(fla)
    };

    // Create the initial image and UV map.
    let source_image = match flash {
        _ if flash <= 0.0 => env(which)?,
        _ if flash >= 1.0 => fla(which)?,
        _ => (1.0 - flash) * env(which)? + flash * fla(which)?,
    };
    let source_uvs = open_texture(root, id, &Part::RenderPos { which })?;

    // Crop and resize as needed.
    let mut image = Array3::zeros([resolution, resolution, 3]);
    if let Some(crop) = crop {
        warp::resize_and_crop(
            source_image.view(),
            image.view_mut(),
            crop,
            point2(0.5, 0.5),
        );
    } else {
        warp::resize(source_image.view(), image.view_mut());
    }

    // Similarly form the UV map.
    let mut uvs = Array3::zeros([resolution, resolution, 3]);
    if let Some(crop) = crop {
        warp::resize_and_crop(source_uvs.view(), uvs.view_mut(), crop, point2(0.5, 0.5));
    } else {
        warp::resize(source_uvs.view(), uvs.view_mut());
    }

    // Handle the other image.
    let other_image = match other_flash {
        Some(other_flash) => {
            // Metadata stuff.
            let other_which = *meta
                .views
                .keys()
                .filter(|this_which| **this_which != which)
                .choose(rng)
                .ok_or(anyhow!("no other views"))?;

            // Create initial image and UVS.
            let other_env = env(other_which)?;
            let other_fla = fla(other_which)?;
            let other_source_image = (1.0 - flash) * other_env + (flash + other_flash) * other_fla;
            let mut other_image = Array3::zeros([resolution, resolution, 3]);
            let other_source_uvs = open_texture(root, id, &Part::RenderPos { which: other_which })?;

            // Align to original image.
            let other_uvs_resolution = resolution / 2;
            let mut other_uvs = Array3::zeros([other_uvs_resolution, other_uvs_resolution, 3]);
            downcale(other_source_uvs.view(), other_uvs.view_mut());
            let margin = 4.0;
            let nearest = warp::NearestBuffer::new(uvs.view(), other_source_uvs.view())
                .homograpy(20, 10, margin)?;
            warp::reverse_warp(
                warp::BilinearInterpolation(other_source_image.view()),
                other_image.view_mut(),
                |pt| nearest.query(pt),
                |_, mut arr| arr.fill(0.0),
            );

            // Final tonemapping stuff.
            let final_whitepoint = decide_whitepoint(other_image.view());
            image.mapv_inplace(|x| (x / final_whitepoint).max(0.0).min(1.0));
            other_image.mapv_inplace(|x| (x / final_whitepoint).max(0.0).min(1.0));

            Some(other_image)
        }
        None => {
            // Final tone mapping stuff.
            image.mapv_inplace(|x| x.max(0.0).min(1.0));

            None
        }
    };

    // Result
    Ok(Render {
        id: id.clone(),
        which,
        meta,
        crop: crop.unwrap_or(1.0),
        uvs,
        image,
        other_image,
    })
}

pub fn form_svbrdf(
    root: &DatasetConfig,
    id: &DataId,
    rng: &mut impl Rng,
    resolution: Option<usize>,
    map_part: impl Fn(Part) -> Part,
) -> Result<Array3<f32>, Error> {
    let diffuse;
    let specular;
    if root.dirs[id.whichdir.unwrap_or(0)].metalness {
        let albedo = open_texture(root, id, &map_part(Part::Albedo))?;
        let metalness = open_texture(root, id, &map_part(Part::Metalness))?;
        let dielectricness = 1.0 - &metalness;
        diffuse = albedo.clone() * &dielectricness;
        specular = albedo.clone() * &metalness + 0.04;
    } else {
        diffuse = open_texture(root, id, &map_part(Part::Diffuse))?;
        specular = open_texture(root, id, &map_part(Part::Specular))?;
    }
    let roughness = open_texture(root, id, &map_part(Part::Roughness))?;
    let normals = if root.normals {
        Some(open_texture(root, id, &map_part(Part::Normals))?)
    } else {
        None
    };
    let height = if root.height {
        let mut height = open_texture(root, id, &map_part(Part::Height))?;
        height -= *height.iter().choose(rng).unwrap();
        Some(height)
    } else {
        None
    };
    let (h, w, _) = diffuse.dim();
    let mut arr = ndarray::concatenate(
        Axis(2),
        &[
            diffuse.view(),
            specular.view(),
            roughness.slice(s![.., .., 0usize..=0]),
        ],
    )
    .with_context(|| format!("combining textures for {id:?}"))?;
    if let Some(normals) = normals {
        arr = ndarray::concatenate(Axis(2), &[arr.view(), normals.view()])
            .with_context(|| format!("combining textures for {id:?}"))?;
    }
    if let Some(height) = height {
        arr = ndarray::concatenate(Axis(2), &[arr.view(), height.view()])
            .with_context(|| format!("combining textures for {id:?}"))?;
    }
    if let Some(resolution) = resolution {
        ensure!(h >= resolution, "texture {id:?} not tall enough");
        ensure!(w >= resolution, "texture {id:?} not wide enough");
        let oi = if h == resolution {
            0
        } else {
            rng.gen_range(0..h - resolution)
        };
        let oj = if w == resolution {
            0
        } else {
            rng.gen_range(0..w - resolution)
        };
        arr = arr
            .slice_move(s![oi.., oj.., ..])
            .slice_move(s![..resolution, ..resolution, ..]);
        assert_eq!(
            arr.dim(),
            (
                resolution,
                resolution,
                7 + if root.height { 1 } else { 0 } + if root.normals { 3 } else { 0 },
            )
        );
    } else {
        assert_eq!(
            arr.dim(),
            (
                h,
                w,
                7 + if root.height { 1 } else { 0 } + if root.normals { 3 } else { 0 },
            )
        );
    }
    Ok(arr)
}

/// ==========================================
/// Based on https://github.com/Vandermode/ELD
/// ==========================================
pub const SONY_A7S2: SensorProfile = SensorProfile {
    k_min: 0.09391607086816033,
    k_max: 6.010628535562261,
    g_bias: 1.218237123334061,
    g_sigma: 0.26751211630129734,
    g_slope: 0.5407496082896145,
};

pub struct SensorProfile {
    pub k_min: f32,
    pub k_max: f32,
    pub g_bias: f32,
    pub g_sigma: f32,
    pub g_slope: f32,
}

impl Distribution<SensorFormation> for SensorProfile {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SensorFormation {
        let log_k = rng.sample(Uniform::new(self.k_min.ln(), self.k_max.ln()));
        let log_g =
            rng.sample(Normal::new(self.g_bias, self.g_sigma).unwrap()) + log_k * self.g_slope;
        SensorFormation {
            k: log_k.exp(),
            g: log_g.exp(),
            saturation_level: 2f32.powi(14) - 1.0,
        }
    }
}

pub struct SensorFormation {
    pub k: f32,
    pub g: f32,
    pub saturation_level: f32,
}

pub fn form_sensor_noise<R: Rng>(
    mut image: ArrayViewMut3<f32>,
    rng: &mut R,
    profile: &SensorProfile,
) {
    let form = rng.sample(profile);
    let shot_noise = |i: f32, rng: &mut R| {
        rng.sample(Poisson::new(i.max(f32::EPSILON) / form.k).unwrap()) * form.k
    };
    let read_noise = |i: f32, rng: &mut R| i + rng.sample(Normal::new(0.0, form.g).unwrap());
    image.mapv_inplace(|i| {
        read_noise(shot_noise(i * form.saturation_level, rng), rng).round() / form.saturation_level
    });
}
