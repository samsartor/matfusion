use crate::{DatasetConfig, DatasetPathMode, DatasetPathMode::*};
use anyhow::{anyhow, bail, Context, Error};
use ndarray::{concatenate, Array3, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::create_dir;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub struct DataId {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whichdir: Option<usize>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replicate: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<<StdRng as SeedableRng>::Seed>,
}

impl DataId {
    pub fn full_name(&self) -> String {
        match self {
            DataId {
                replicate: Some(replicate),
                name,
                whichdir: Some(whichdir),
                ..
            } => format!("{whichdir}-{name}.{replicate:03}"),
            DataId {
                replicate: None,
                name,
                whichdir: Some(whichdir),
                ..
            } => format!("{whichdir}-{name}"),
            DataId {
                replicate: Some(replicate),
                name,
                whichdir: None,
                ..
            } => format!("{name}.{replicate:03}"),
            DataId {
                replicate: None,
                name,
                whichdir: None,
                ..
            } => format!("{name}"),
        }
    }

    pub fn basic_name(&self) -> String {
        match self {
            DataId {
                replicate: Some(replicate),
                name,
                ..
            } => format!("{name}.{replicate:03}"),
            DataId {
                replicate: None,
                name,
                ..
            } => format!("{name}"),
        }
    }

    pub fn for_results(&self, replicate: usize) -> Option<Self> {
        let mut this = self.clone();
        this.replicate = Some(replicate);
        Some(this)
    }

    pub fn configure_rng(&self, rng: &mut StdRng) {
        if let Some(seed) = self.seed {
            *rng = StdRng::from_seed(seed);
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub enum Part {
    Diffuse,
    Albedo,
    Specular,
    Roughness,
    Metalness,
    Normals,
    Height,
    SvbrdfMeta { file: String },
    RenderMeta,
    RenderPos { which: char },
    RenderFlash { which: char },
    RenderEnv { which: char },
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Part::*;
        match self {
            Diffuse => write!(f, "diffuse"),
            Albedo => write!(f, "albedo"),
            Specular => write!(f, "specular"),
            Roughness => write!(f, "roughness"),
            Metalness => write!(f, "metalness"),
            Normals => write!(f, "normals"),
            Height => write!(f, "height"),
            SvbrdfMeta { file } => write!(f, "{file}"),
            RenderMeta => write!(f, "metadata"),
            RenderPos { which, .. } => write!(f, "{which}_position"),
            RenderFlash { which, .. } => write!(f, "{which}_flash"),
            RenderEnv { which, .. } => write!(f, "{which}_envio"),
        }
    }
}

pub fn data_path(cfg: &DatasetConfig, id: &DataId, part: &Part) -> Result<PathBuf, Error> {
    use Part::*;

    let dir = &cfg.dirs[id.whichdir.unwrap_or(0)];
    let name = id.basic_name();
    Ok(match (dir.svbrdf_mode, part) {
        (
            PartThenNamePart,
            Diffuse | Albedo | Specular | Roughness | Metalness | Normals | Height,
        ) => {
            let Some(mut path) = dir.svbrdfs.clone() else {
                bail!("no svbrdfs dir")
            };
            path.push(part.to_string());
            if let Height = part {
                path.push(format!("{name}_{part}.exr"));
            } else {
                path.push(format!("{name}_{part}.png"));
            }
            path
        }
        (NameThenPart, Diffuse | Albedo | Specular | Roughness | Metalness | Normals | Height) => {
            let Some(mut path) = dir.svbrdfs.clone() else {
                bail!("no svbrdfs dir")
            };
            path.push(name);
            match part {
                Height => path.push(format!("{part}.exr")),
                _ => path.push(format!("{part}.png")),
            }
            path
        }
        (_, SvbrdfMeta { file }) => {
            let Some(mut path) = dir.svbrdfs.clone() else {
                bail!("no svbrdfs dir")
            };
            path.push(name);
            path.push(file);
            path
        }
        (_, &RenderMeta {} | &RenderPos { .. } | &RenderFlash { .. } | &RenderEnv { .. }) => {
            let Some(mut path) = dir.renders.clone() else {
                bail!("no renders dir")
            };
            match part {
                RenderMeta { .. } => path.push(format!("{name}.json")),
                RenderPos { .. } | RenderFlash { .. } | RenderEnv { .. } => {
                    path.push(format!("{name}_{part}_0000.exr"))
                }
                _ => unreachable!(),
            }
            path
        }
    })
}

pub fn texture_gamma(cfg: &DatasetConfig, id: &DataId, part: &Part) -> f32 {
    use Part::*;

    match part {
        Diffuse | Specular | Albedo => cfg.dirs[id.whichdir.unwrap_or(0)].texture_gamma,
        _ => 1.0,
    }
}

pub fn open_image(path: &Path) -> Result<Array3<f32>, Error> {
    let img = image::open(&path).with_context(|| format!("opening image {:?}", path.display()))?;
    let img = img.into_rgb32f();
    Ok(Array3::from_shape_vec(
        [img.height() as usize, img.width() as usize, 3],
        img.into_raw(),
    )?)
}

pub fn open_texture(cfg: &DatasetConfig, id: &DataId, part: &Part) -> Result<Array3<f32>, Error> {
    let path = data_path(cfg, id, part)?;
    let mut img = if let Part::Height = part {
        let img = exr::image::read::read_first_flat_layer_from_file(&path)
            .with_context(|| format!("opening image {:?}", path.display()))?;
        let gray = img
            .layer_data
            .channel_data
            .list
            .get(0)
            .ok_or(anyhow!("missing gray channel"))?;
        let size = img.layer_data.size;
        let gray: Vec<f32> = gray.sample_data.values_as_f32().collect();
        Array3::from_shape_vec([size.1 as usize, size.0 as usize, 1], gray)?
    } else {
        open_image(&path)?
    };
    let gamma = texture_gamma(cfg, id, part);
    if gamma != 1.0 {
        for c in img.iter_mut() {
            *c = c.powf(gamma);
        }
    }
    Ok(img)
}

pub fn save_texture(
    cfg: &DatasetConfig,
    id: &DataId,
    part: &Part,
    mut texture: Array3<f32>,
) -> Result<(), Error> {
    let gamma = texture_gamma(cfg, id, part).recip();
    if gamma != 1.0 {
        for c in texture.iter_mut() {
            *c = c.powf(gamma);
        }
    }
    let path = data_path(cfg, id, part)?;
    if let Some(parent) = path.parent() {
        let _ = create_dir(parent);
    }
    if texture.dim().2 == 1 {
        texture = concatenate![Axis(2), texture, texture, texture]
            .as_standard_layout()
            .into_owned();
    }
    if let Part::Height = part {
        let img = crate::array2rgbf(texture);
        img.save(&path)
            .context(format!("saving float map at {}", path.display()))?;
    } else {
        let img = crate::array2rgb8(texture);
        img.save(&path)
            .context(format!("saving map at {}", path.display()))?;
    };
    Ok(())
}

pub fn search_for_ids(
    root: &Path,
    mode: DatasetPathMode,
    out: &mut Vec<DataId>,
) -> Result<(), Error> {
    let root = match mode {
        PartThenNamePart => root.join("diffuse"),
        NameThenPart => root.to_owned(),
    };
    for file in root
        .read_dir()
        .with_context(|| format!("looking for files in {:?}", root.display()))?
    {
        let (name, source) = match mode {
            PartThenNamePart => {
                let Ok(file) = file else { continue };
                let Ok(name) = file.file_name().into_string() else {
                    continue;
                };
                let Some((name, "")) = name.rsplit_once("_diffuse.png") else {
                    continue;
                };
                (name.to_owned(), None)
            }
            NameThenPart => {
                let Ok(file) = file else { continue };
                match file.file_type() {
                    Ok(ty) if ty.is_dir() => (),
                    _ => continue,
                }
                let Ok(name) = file.file_name().into_string() else {
                    continue;
                };
                let source = std::fs::read_to_string(file.path().join("source.txt"))
                    .map(PathBuf::from)
                    .ok();
                (name, source)
            }
        };
        let (name, replicate) = match name.rsplit_once(".") {
            Some((name, replicate)) => {
                let Ok(replicate) = replicate.parse::<usize>() else {
                    continue;
                };
                (name.to_string(), Some(replicate))
            }
            None => (name, None),
        };
        out.push(DataId {
            replicate,
            name,
            source,
            // these are filled in later
            whichdir: None,
            seed: None,
        });
    }
    Ok(())
}
