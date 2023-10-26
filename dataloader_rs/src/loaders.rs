use crate::{
    correct_exposure, form_render, form_sensor_noise, form_svbrdf, open_image, search_for_ids,
    warp, DataId, DatasetConfig, DatasetPath, Sample, SONY_A7S2,
};
use anyhow::{anyhow, bail, Error};
use cgmath::point2;
use map_macro::hash_map;
use ndarray::{s, Array3};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Distr {
    #[serde(rename = "constant")]
    Constant(f32),
    #[serde(rename = "gamma_sampled")]
    Gamma(f32, f32),
    #[serde(rename = "uniform_log_sampled")]
    UniformLog(f32, f32),
}

impl Distribution<f32> for Distr {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        match *self {
            Distr::Constant(value) => value,
            Distr::Gamma(a, b) => Gamma::new(a, b).unwrap().sample(rng),
            Distr::UniformLog(a, b) => Uniform::new(a.ln(), b.ln()).sample(rng).exp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum LoaderConfig {
    Rasterized(Rasterized),
    Rendered(Rendered),
    Images(Images),
}

impl LoaderConfig {
    pub fn into_loader(self) -> Arc<dyn Loader> {
        match self {
            LoaderConfig::Rasterized(x) => Arc::new(x),
            LoaderConfig::Rendered(x) => Arc::new(x),
            LoaderConfig::Images(x) => Arc::new(x),
        }
    }

    pub fn as_loader(&self) -> &dyn Loader {
        match self {
            LoaderConfig::Rasterized(x) => x,
            LoaderConfig::Rendered(x) => x,
            LoaderConfig::Images(x) => x,
        }
    }
}

pub trait Loader: Sync + Send {
    fn load_epoc(&self, root: &DatasetPath) -> Result<Vec<DataId>, Error> {
        let mut out = Vec::new();
        search_for_ids(
            root.svbrdfs
                .as_deref()
                .ok_or(anyhow!("no svbrdf directory"))?,
            root.svbrdf_mode,
            &mut out,
        )?;
        Ok(out)
    }

    fn load_sample(
        &self,
        cfg: &DatasetConfig,
        id: &DataId,
        rng: &mut StdRng,
    ) -> Result<Sample, Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rasterized {
    pub distance: Distr,
}

impl Loader for Rasterized {
    fn load_sample(
        &self,
        root: &DatasetConfig,
        id: &DataId,
        rng: &mut StdRng,
    ) -> Result<Sample, Error> {
        let svbrdf = form_svbrdf(root, id, rng, Some(root.resolution), |p| p)?;
        let distance = self.distance.sample(rng);
        Ok(Sample::Dict(hash_map! {
            "svbrdf" => Sample::Image(svbrdf),
            "view_distance" => Sample::Scalar(distance),
            "flash_distance" => Sample::Scalar(distance),
            "flash_x" => Sample::Scalar(0.0),
            "flash_y" => Sample::Scalar(0.0),
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rendered {
    pub flash_weight: f32,
    pub crop: Option<f32>,
    pub other_weight: Option<Distr>,
}

impl Loader for Rendered {
    fn load_sample(
        &self,
        root: &DatasetConfig,
        id: &DataId,
        rng: &mut StdRng,
    ) -> Result<Sample, Error> {
        let other_weight = self.other_weight.as_ref().map(|dist| dist.sample(rng));
        let render = form_render(
            root,
            id,
            rng,
            self.crop,
            root.resolution,
            self.flash_weight,
            other_weight,
        )?;
        let svbrdf = render.matching_svbrdf(&root, rng)?;
        let mut dict = hash_map! {
            "flash_distance" => Sample::Scalar(render.flash_distance()),
            "view_distance" => match render.view_distance() {
                Some(d) => Sample::Scalar(d.into()),
                None => Sample::Scalar(f32::INFINITY),
            },
            "flash_x" => Sample::Scalar(render.flash_offset().x),
            "flash_y" => Sample::Scalar(render.flash_offset().y),
            "svbrdf" => Sample::Image(svbrdf),
        };
        let mut image = render.image;
        form_sensor_noise(image.view_mut(), rng, &SONY_A7S2);
        if let Some(mut other_image) = render.other_image {
            form_sensor_noise(other_image.view_mut(), rng, &SONY_A7S2);
            dict.insert("other_render", Sample::Image(other_image.clone()));
        }
        dict.insert("render", Sample::Image(image));
        Ok(Sample::Dict(dict))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Images {
    /// If the contents of the images directory are folders, the filename in each folder which is the primary image
    pub filename: Option<String>,
    /// If the contents of the images directory are folders, the filename in each folder which is the other image
    pub other_filename: Option<String>,
    /// The distance that each image was taken from (where the width of the captured surface is 2 units)
    pub distance: f32,
    /// Crop each image so that the actual distance is this
    pub desired_distance: Option<f32>,
    pub gamma: f32,
    #[serde(default)]
    pub auto_exposure: bool,
}

impl Loader for Images {
    fn load_epoc(&self, root: &DatasetPath) -> Result<Vec<DataId>, Error> {
        root.images
            .as_ref()
            .ok_or(anyhow!("no images directory"))?
            .read_dir()?
            .into_iter()
            .map(|file| {
                let file = file?;
                let source = match &self.filename {
                    None => Some(file.path().to_owned()),
                    Some(filename) => Some(file.path().join(filename)),
                };
                let mut name = file.file_name().into_string().unwrap();
                if let Some((base, _)) = name.rsplit_once('.') {
                    name = base.to_string();
                }
                Ok(DataId {
                    name,
                    replicate: None,
                    source,
                    whichdir: None,
                    seed: None,
                })
            })
            .collect()
    }

    fn load_sample(
        &self,
        root: &DatasetConfig,
        id: &DataId,
        _rng: &mut StdRng,
    ) -> Result<Sample, Error> {
        let mut whitepoint = None;
        let mut do_load = |source: &Path| -> Result<(_, _), Error> {
            let mut input = open_image(source)?;
            if self.gamma != 1.0 {
                input.mapv_inplace(|x| x.powf(self.gamma));
            }
            match (whitepoint, self.auto_exposure) {
                (None, true) => whitepoint = Some(correct_exposure(input.view_mut())),
                (Some(w), true) => input.mapv_inplace(|x| x / w),
                (_, false) => (),
            }
            let input_resolution = input.dim().0.min(input.dim().1);
            let input_offset_r = (input.dim().0 - input_resolution) / 2;
            let input_offset_c = (input.dim().1 - input_resolution) / 2;
            let input = input.slice(s![input_offset_r.., input_offset_c.., ..]);
            let input = input.slice(s![..input_resolution, ..input_resolution, ..]);
            let mut dest = Array3::zeros([root.resolution, root.resolution, 3]);
            // focal_length / 35mm = distance / 2.0
            let mut distance = self.distance;
            if let Some(desired_distance) = self.desired_distance {
                let crop = distance / desired_distance;
                warp::resize_and_crop(input, dest.view_mut(), crop, point2(0.5, 0.5));
                distance = desired_distance;
            } else {
                warp::resize(input, dest.view_mut());
            }
            Ok((dest, distance))
        };
        let DataId {
            source: Some(source),
            ..
        } = id
        else {
            bail!("id does not have source")
        };
        let other_input = match &self.other_filename {
            Some(other_filename) => {
                let other_source = source.with_file_name(other_filename);
                let (other_input, _) = do_load(&other_source)?;
                Some(other_input)
            }
            None => None,
        };
        let (input, distance) = do_load(source)?;
        let mut dict = hash_map! {
            "input" => Sample::Image(input),
            "source" => Sample::Any(source.to_string_lossy().into()),
            "view_distance" => Sample::Scalar(distance),
            "flash_distance" => Sample::Scalar(distance),
            "flash_x" => Sample::Scalar(0.0),
            "flash_y" => Sample::Scalar(0.0),
        };
        if let Some(other_input) = other_input {
            dict.insert("other_input", Sample::Image(other_input));
        }
        Ok(Sample::Dict(dict))
    }
}
