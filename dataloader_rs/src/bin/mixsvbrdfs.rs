use anyhow::{anyhow, Error};
use crossbeam_channel::bounded;
use data::loaders::{LoaderConfig, Rasterized};
use data::{resize, save_texture, DataId, DatasetConfig, DatasetPath, Part, Sample};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{concatenate, s, Array2, Array3, ArrayView3, ArrayViewMut3, Axis};
use ndarray_rand::RandomExt;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::Normal;
use std::num::ParseFloatError;
use std::path::PathBuf;
use std::str::FromStr;
use std::thread::spawn;

struct Weights(Vec<(usize, f32)>);

impl FromStr for Weights {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        let vec = s
            .split(',')
            .enumerate()
            .map(|(i, s)| Ok((i + 1, s.parse::<f32>()?)))
            .collect::<Result<_, ParseFloatError>>()?;
        Ok(Weights(vec))
    }
}

impl Weights {
    fn choose(&self, rng: &mut impl Rng) -> usize {
        self.0.choose_weighted(rng, |&(_, w)| w).unwrap().0
    }
}

#[derive(argh::FromArgs)]
/// create random multi-material svbrdfs
struct MixSvbrdf {
    /// the root datasets directory
    #[argh(positional, default = "PathBuf::from(\"./datasets\")")]
    root: PathBuf,
    /// weights for different counts of maps
    #[argh(option, default = "Weights(vec![(1, 0.0), (2, 2.0), (3, 1.0)])")]
    weights: Weights,
    /// number of workers
    #[argh(option, default = "16")]
    workers: usize,
}

pub fn normalize(mut arr: ArrayViewMut3<f32>) {
    for k in 0..arr.dim().2 {
        let mut s = arr.slice_mut(s![.., .., k]);
        s -= s.mean().unwrap();
        let std = s.std(0.0);
        if std > f32::EPSILON {
            s /= std;
        }
    }
}

pub fn random_layer(input: ArrayView3<f32>, mut output: ArrayViewMut3<f32>) {
    let m = Array2::<f32>::random(
        [input.dim().2, output.dim().2],
        Normal::new(0.0, 1.0).unwrap(),
    );
    assert_eq!(input.shape()[..2], output.shape()[..2]);
    for i in 0..input.dim().0 {
        for j in 0..input.dim().1 {
            let input = input.slice(s![i, j, ..]);
            let mut output = output.slice_mut(s![i, j, ..]);
            general_mat_vec_mul(1.0, &m, &input, 0.0, &mut output);
        }
    }
    normalize(output.view_mut());
    for f in output.iter_mut() {
        *f = f.tanh();
    }
}

pub fn maskmax(mut arr: ArrayViewMut3<f32>) {
    for i in 0..arr.dim().0 {
        for j in 0..arr.dim().1 {
            let mut arr = arr.slice_mut(s![i, j, ..]);
            let (max_i, _) = arr
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            for (i, val) in arr.iter_mut().enumerate() {
                if i == max_i {
                    *val = 1.0;
                } else {
                    *val = 0.0;
                }
            }
        }
    }
}

pub fn randomly_mix_svbrdfs(
    cfg: &DatasetConfig,
    ids: impl IntoIterator<Item = DataId>,
    rng: &mut StdRng,
    antialias: usize,
) -> Result<Array3<f32>, Error> {
    let maps = ids
        .into_iter()
        .map(
            |id| match cfg.loader.as_loader().load_sample(cfg, &id, rng) {
                Ok(Sample::Dict(mut d)) => match d.remove("svbrdf") {
                    Some(Sample::Image(a)) => Ok(a),
                    Some(_) => Err(anyhow!("svbrdf is not an array")),
                    None => Err(anyhow!("missing svbrdf")),
                },
                Ok(_) => Err(anyhow!("sample is not a dict")),
                Err(e) => Err(e),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let svbrdfs = maps
        .iter()
        .map(|a| a.view().insert_axis(Axis(3)))
        .collect::<Vec<_>>();
    let mut svbrdfs = concatenate(Axis(3), &svbrdfs)?;
    let (h, w, _, b) = svbrdfs.dim();

    let mut filters0 = Array3::<f32>::zeros([h * antialias, w * antialias, b]);
    let mut filters1 = Array3::<f32>::zeros([h * antialias, w * antialias, b]);
    resize(svbrdfs.slice(s![.., .., 10, ..]), filters0.view_mut());
    normalize(filters0.view_mut());
    random_layer(filters0.view(), filters1.view_mut());
    random_layer(filters1.view(), filters0.view_mut());
    maskmax(filters0.view_mut());
    let filters = filters0;

    for i in 0..h {
        for j in 0..w {
            for k in 0..b {
                let f = filters
                    .slice(s![
                        i * antialias..(i + 1) * antialias,
                        j * antialias..(j + 1) * antialias,
                        k
                    ])
                    .mean()
                    .unwrap();
                for x in svbrdfs.slice_mut(s![i, j, .., k]).iter_mut() {
                    *x *= f;
                }
            }
        }
    }

    Ok(svbrdfs.sum_axis(Axis(3)))
}

pub fn do_mixing(
    input_cfg: &DatasetConfig,
    output_cfg: &DatasetConfig,
    id: &DataId,
    selected: Vec<DataId>,
    rng: &mut StdRng,
) -> Result<(), Error> {
    //println!("Mixing {selected:?}.");
    let svbrdf = randomly_mix_svbrdfs(input_cfg, selected, rng, 2)?;

    let diffuse = svbrdf.slice(s![.., .., 0usize..3]).into_owned();
    save_texture(output_cfg, id, &Part::Diffuse, diffuse)?;
    let specular = svbrdf.slice(s![.., .., 3usize..6]).into_owned();
    save_texture(output_cfg, id, &Part::Specular, specular)?;
    let roughness = svbrdf.slice(s![.., .., 6usize..=6]);
    let roughness = concatenate![Axis(2), roughness, roughness, roughness]
        .as_standard_layout()
        .into_owned();
    // TODO: save as grayscale
    save_texture(output_cfg, id, &Part::Roughness, roughness)?;
    let normals = svbrdf.slice(s![.., .., 7usize..10]).into_owned();
    save_texture(output_cfg, id, &Part::Normals, normals)?;

    Ok(())
}

pub fn main() -> Result<(), Error> {
    let MixSvbrdf {
        root,
        weights,
        workers,
    } = argh::from_env();

    let input_cfg = DatasetConfig {
        resolution: 288,
        dirs: vec![
            DatasetPath {
                svbrdfs: Some(root.join("inria_svbrdfs")),
                renders: None,
                images: None,
                svbrdf_mode: data::DatasetPathMode::PartThenNamePart,
                texture_gamma: 1.0,
                metalness: false,
                count: None,
            },
            DatasetPath {
                svbrdfs: Some(root.join("cc0_svbrdfs")),
                renders: None,
                images: None,
                svbrdf_mode: data::DatasetPathMode::PartThenNamePart,
                texture_gamma: 1.0,
                metalness: false,
                count: None,
            },
        ],
        loader: LoaderConfig::Rasterized(Rasterized {
            // doesn't matter
            distance: data::loaders::Distr::Constant(4.0),
        }),
        normals: true,
        height: true,
    };
    let output_cfg = DatasetConfig {
        resolution: 288,
        dirs: vec![
            DatasetPath {
                svbrdfs: Some(root.join("mixed_svbrdfs")),
                renders: None,
                images: None,
                svbrdf_mode: data::DatasetPathMode::PartThenNamePart,
                texture_gamma: 1.0,
                metalness: false,
                count: None,
            },
            DatasetPath {
                svbrdfs: Some(root.join("cc0_svbrdfs")),
                renders: None,
                images: None,
                svbrdf_mode: data::DatasetPathMode::PartThenNamePart,
                texture_gamma: 1.0,
                metalness: false,
                count: None,
            },
        ],
        loader: LoaderConfig::Rasterized(Rasterized {
            // doesn't matter
            distance: data::loaders::Distr::Constant(4.0),
        }),
        normals: true,
        height: true,
    };

    let (send_tasks, recv_tasks) = bounded::<(DataId, Vec<DataId>)>(16);
    let join_handles: Vec<_> = (0..workers)
        .map(|_| {
            let recv_tasks = recv_tasks.clone();
            let input_cfg = input_cfg.clone();
            let output_cfg = output_cfg.clone();
            let mut rng = StdRng::from_entropy();
            spawn(move || loop {
                let Ok((id, selected)) = recv_tasks.recv() else {
                    return;
                };
                if let Err(err) = do_mixing(&input_cfg, &output_cfg, &id, selected, &mut rng) {
                    eprintln!("ERROR mixing {id:?}: {err:?}");
                }
            })
        })
        .collect();

    let mut ids = input_cfg.load_epoc()?;
    let total_ids = ids.len() as u64;
    println!("Found {} ids.", ids.len());
    let progress = ProgressBar::new(total_ids);
    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}/{eta}] {bar:80.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap(),
    );
    let mut rng = StdRng::from_entropy();
    ids.shuffle(&mut rng);
    let mut selected = Vec::new();
    let mut num = 0;
    'main: loop {
        let id = DataId {
            whichdir: None,
            name: format!("{num:07}"),
            replicate: None,
            source: None,
            seed: None,
        };

        selected.clear();
        let count = weights.choose(&mut rng);
        for _ in 0..count {
            let Some(sel) = ids.pop() else { break 'main };
            selected.push(sel);
        }
        progress.set_position(total_ids - ids.len() as u64);
        if count < 2 {
            continue 'main;
        }
        send_tasks.send((id, selected.split_off(0))).unwrap();
        num += 1;
    }
    drop(send_tasks);

    for handle in join_handles {
        handle.join().unwrap();
    }

    Ok(())
}
