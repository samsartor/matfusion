use anyhow::{anyhow, bail, Context, Error};
use argh;
use crossbeam_channel::{bounded, Receiver};
use data::{data_path, texture_gamma, DataId, DatasetConfig, Part};
use indicatif::{ProgressIterator, ProgressStyle};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng as _};
use serde::Serialize;
use serde_yaml::Value;
use std::fs::remove_file;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::{sleep, spawn};
use std::time::Duration;

#[derive(argh::FromArgs)]
/// create random multi-material svbrdfs
pub struct RenderAll {
    /// the dataset config YAML file (contains both input and output paths)
    #[argh(positional)]
    config_path: PathBuf,
    /// number of indepedent workers
    #[argh(option, default = "16")]
    workers: usize,
    /// number of threads per worker
    #[argh(option, default = "8")]
    threads: usize,
    /// overwrite existing render outputs
    #[argh(switch)]
    overwrite: bool,
    /// run orthographic renders
    #[argh(switch)]
    ortho: bool,
    /// initially show worker outputs
    #[argh(switch)]
    show_output: bool,
    /// single environment map to use
    #[argh(option)]
    world: Option<PathBuf>,
    /// directory of environment maps to use
    #[argh(option)]
    worlds: Option<PathBuf>,
    /// override the flash/camera distance
    #[argh(option)]
    distance: Option<f32>,
    /// disable random offset/rotation
    #[argh(switch)]
    fixed: bool,
    /// disable global illumination
    #[argh(switch)]
    no_gi: bool,
    /// visible CUDA devices
    #[argh(option)]
    cuda_devices: Option<String>,
}

#[derive(Serialize, Debug)]
struct WorkerInput {
    name: String,
    output_dir: PathBuf,
    height: PathBuf,
    diffuse: PathBuf,
    specular: PathBuf,
    roughness: PathBuf,
    normals: PathBuf,
    colorspace: String,
    world: PathBuf,
    world_rotation: Option<f32>,
    flash_distance: Option<f32>,
    flash_offset: Option<[f32; 2]>,
    target_rotation: Option<[f32; 4]>,
    target_location: Option<[f32; 2]>,
    use_cuda: bool,
}

pub fn get_worlds(args: &RenderAll, project_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let worlds_dir = match &args.worlds {
        Some(w) => w.clone(),
        None => project_dir.join("../datasets/environments"),
    };
    match &args.world {
        Some(w) => Ok(vec![w.to_owned()]),
        None => std::fs::read_dir(worlds_dir)
            .context("environments dataset is missing")?
            .map(|d| Ok(d?.path()))
            .collect(),
    }
}

pub fn render_worker(
    worker: usize,
    ids: Receiver<DataId>,
    args: &RenderAll,
    cfg: &DatasetConfig,
) -> Result<(), Error> {
    let mut rng = thread_rng();
    sleep(Duration::from_secs_f32(rng.gen_range(0.0..30.0)));
    let mut show_output = args.show_output;
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let template_dir = project_dir
        .join("../scripts/render_synthetic")
        .canonicalize()?;
    let worlds_list = get_worlds(args, &project_dir)?;
    let mut proc = Command::new("blender");
    proc.args([
        template_dir
            .join(match (args.ortho, args.no_gi) {
                (false, false) => "render_template.blend",
                (true, false) => "ortho_render_template.blend",
                (false, true) => "nogi_render_template.blend",
                (true, true) => panic!(),
            })
            .as_os_str(),
        "-b".as_ref(),
        "-P".as_ref(),
        template_dir
            .join(match args.ortho {
                false => "render_control.py",
                true => "ortho_render_control.py",
            })
            .as_os_str(),
        "--python-use-system-env".as_ref(),
        "--threads".as_ref(),
        &args.threads.to_string().as_ref(),
    ]);
    proc.stdin(Stdio::piped());
    proc.stdout(Stdio::piped());
    if let Some(v) = &args.cuda_devices {
        let v: Vec<_> = v.split(',').collect();
        let v = v
            .get(worker % v.len())
            .ok_or(anyhow!("no cuda devices given"))?;
        proc.env("CUDA_VISIBLE_DEVICES", v);
    }
    println!(
        "WORKER {worker} starting blender in {}: {:?}",
        template_dir.display(),
        proc,
    );
    let mut proc = proc.spawn().context("spawning blender process")?;
    let Some(stdin) = &mut proc.stdin else {
        bail!("stdin not attached")
    };
    let mut stdin = BufWriter::new(stdin);
    let Some(stdout) = &mut proc.stdout else {
        bail!("stdout not attached")
    };
    let mut stdout = BufReader::new(stdout);
    while let Ok(id) = ids.recv() {
        let meta_path = data_path(cfg, &id, &Part::RenderMeta)?;
        let output_dir = meta_path.parent().ok_or(anyhow!("unkown parent dir"))?;
        if meta_path.exists() {
            if args.overwrite {
                let _ = remove_file(&meta_path);
            } else {
                continue;
            }
        }
        let input = WorkerInput {
            name: id.basic_name(),
            output_dir: output_dir.to_owned(),
            height: data_path(cfg, &id, &Part::Height)?,
            diffuse: data_path(cfg, &id, &Part::Diffuse)?,
            specular: data_path(cfg, &id, &Part::Specular)?,
            roughness: data_path(cfg, &id, &Part::Roughness)?,
            normals: data_path(cfg, &id, &Part::Normals)?,
            colorspace: match texture_gamma(cfg, &id, &Part::Diffuse) {
                gamma if gamma == 1.0 => "Linear",
                gamma if gamma == 2.2 => "sRGB",
                gamma => bail!("unsupported gamma {gamma}"),
            }
            .to_string(),
            world: worlds_list.choose(&mut rng).unwrap().clone(),
            world_rotation: if args.fixed { Some(0.0) } else { None },
            flash_distance: args.distance,
            flash_offset: if args.fixed { Some([0.0, 0.0]) } else { None },
            target_rotation: if args.fixed {
                Some([1.0, 0.0, 0.0, 0.0])
            } else {
                None
            },
            target_location: if args.fixed { Some([0.0, 0.0]) } else { None },
            use_cuda: args.cuda_devices.is_some(),
        };
        serde_json::to_writer(&mut stdin, &input)?;
        stdin.write_all(&[b'\n'])?;
        stdin.flush()?;
        let mut line = String::new();
        loop {
            line.clear();
            stdout.read_line(&mut line)?;
            let line = line.trim();
            if line.contains("READY") {
                break;
            }
            if show_output && !line.is_empty() {
                println!("FROM WORKER {worker}: {line}");
            }
        }
        show_output = false;
    }
    drop(stdin);
    drop(stdout);
    proc.kill()?;
    Ok(())
}

pub fn main() -> Result<(), Error> {
    let args: RenderAll = argh::from_env();
    let args: &'static RenderAll = Box::leak(Box::new(args));
    let cfg = DatasetConfig::open(&args.config_path, Value::Null)?;
    let cfg: &'static DatasetConfig = Box::leak(Box::new(cfg));

    let (send_ids, recv_ids) = bounded::<DataId>(32);
    let join_handles: Vec<_> = (0..args.workers)
        .map(|worker| {
            let recv_ids = recv_ids.clone();
            spawn(move || {
                if let Err(err) = render_worker(worker, recv_ids, args, cfg) {
                    println!("ERROR: {err:?}");
                }
            })
        })
        .collect();
    drop(recv_ids);

    let ids = cfg.load_epoc()?;
    println!("Found {} ids.", ids.len());
    for id in ids.into_iter().progress_with_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}/{eta}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap(),
    ) {
        let Ok(_) = send_ids.send(id) else { break };
    }
    drop(send_ids);
    for handle in join_handles {
        handle.join().unwrap();
    }

    Ok(())
}
