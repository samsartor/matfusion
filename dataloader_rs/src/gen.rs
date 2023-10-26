use crate::loaders::LoaderConfig;
use crate::DataId;
use anyhow::{bail, ensure, Context, Error};
use crossbeam_channel::{bounded, Receiver};
use ndarray::{concatenate, Array1, Array3, Array4, Axis};
use numpy::PyArray;
use pyo3::{IntoPy, PyAny, PyErr, PyObject, Python, ToPyObject};
use pythonize::{depythonize, pythonize};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::collections::HashMap;
use std::fs::File;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

#[derive(Clone, Debug)]
pub enum Sample {
    Dict(HashMap<&'static str, Sample>),
    Image(Array3<f32>),
    Scalar(f32),
    Any(Value),
}

impl IntoPy<PyObject> for Sample {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Sample::Dict(x) => x.into_py(py),
            Sample::Image(x) => PyArray::from_owned_array(py, x).to_object(py),
            Sample::Scalar(x) => x.into_py(py),
            Sample::Any(x) => pythonize(py, &x).unwrap(),
        }
    }
}

impl Sample {
    pub fn provide(&mut self, name: &'static str, sample: impl serde::Serialize) {
        if let Sample::Dict(x) = self {
            x.insert(name, Sample::Any(serde_yaml::to_value(sample).unwrap()));
        }
    }
}

#[derive(Debug)]
pub enum BatchedSample {
    Dict(HashMap<&'static str, BatchedSample>),
    Image(Array4<f32>),
    Scalar(Array1<f32>),
    Any(Vec<Value>),
    Unbatched(Sample),
}

impl FromIterator<Sample> for BatchedSample {
    fn from_iter<T: IntoIterator<Item = Sample>>(iter: T) -> Self {
        use Sample::*;
        let mut iter = iter.into_iter();
        let size_hint = iter.size_hint().0;
        match iter.next().expect("must contain at least one sample") {
            Dict(dict) => {
                let mut dict: HashMap<_, _> = dict
                    .into_iter()
                    .map(|(k, v)| {
                        let mut list = Vec::with_capacity(size_hint);
                        list.push(v);
                        (k, list)
                    })
                    .collect();
                for sample in iter {
                    if let Dict(sample) = sample {
                        for (k, v) in sample {
                            dict.get_mut(k)
                                .expect("mismatched samples: inconsistant key")
                                .push(v);
                        }
                    } else {
                        panic!("mismatched samples: expected dict")
                    }
                }
                BatchedSample::Dict(
                    dict.into_iter()
                        .map(|(k, v)| (k, v.into_iter().collect()))
                        .collect(),
                )
            }
            Image(v) => {
                let mut list = Vec::with_capacity(size_hint);
                list.push(v);
                for sample in iter {
                    if let Image(sample) = sample {
                        list.push(sample);
                    } else {
                        panic!("mismatched samples: expected image")
                    }
                }
                let list: Vec<_> = list.iter().map(|v| v.view().insert_axis(Axis(0))).collect();
                BatchedSample::Image(
                    concatenate(Axis(0), &list).expect("mismatched samples: wrong image shape"),
                )
            }
            Scalar(x) => {
                let mut list = Vec::with_capacity(size_hint);
                list.push(x);
                for sample in iter {
                    if let Scalar(sample) = sample {
                        list.push(sample);
                    } else {
                        panic!("mismatched samples: expected scalar")
                    }
                }
                BatchedSample::Scalar(Array1::from_vec(list))
            }
            Any(x) => {
                let mut list = Vec::with_capacity(size_hint);
                list.push(x);
                for sample in iter {
                    if let Any(sample) = sample {
                        list.push(sample);
                    } else {
                        panic!("mismatched samples: expected any")
                    }
                }
                BatchedSample::Any(list)
            }
        }
    }
}

impl IntoPy<PyObject> for BatchedSample {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            BatchedSample::Dict(x) => x.into_py(py),
            BatchedSample::Image(x) => PyArray::from_owned_array(py, x).to_object(py),
            BatchedSample::Scalar(x) => PyArray::from_owned_array(py, x).to_object(py),
            BatchedSample::Any(x) => pythonize(py, &x).unwrap(),
            BatchedSample::Unbatched(x) => x.into_py(py),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetPathMode {
    #[serde(rename = "part/name_part")]
    PartThenNamePart,
    #[serde(rename = "name/part")]
    NameThenPart,
}

fn default_gamma() -> f32 {
    2.2
}

fn default_path_mode() -> DatasetPathMode {
    DatasetPathMode::NameThenPart
}

fn default_use_metalness() -> bool {
    false
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetPath {
    pub svbrdfs: Option<PathBuf>,
    pub renders: Option<PathBuf>,
    pub images: Option<PathBuf>,
    #[serde(default = "default_path_mode")]
    pub svbrdf_mode: DatasetPathMode,
    #[serde(default = "default_gamma")]
    pub texture_gamma: f32,
    #[serde(default = "default_use_metalness")]
    pub metalness: bool,
    pub count: Option<usize>,
}

impl DatasetPath {
    pub fn paths_mut(&mut self) -> impl IntoIterator<Item = &mut PathBuf> {
        [
            self.svbrdfs.as_mut(),
            self.renders.as_mut(),
            self.images.as_mut(),
        ]
        .into_iter()
        .flatten()
    }
}

fn default_include_normals() -> bool {
    true
}

fn default_include_height() -> bool {
    false
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    pub resolution: usize,
    pub dirs: Vec<DatasetPath>,
    pub loader: LoaderConfig,
    #[serde(default = "default_include_normals")]
    pub normals: bool,
    #[serde(default = "default_include_height")]
    pub height: bool,
}

impl DatasetConfig {
    fn do_open(cfg_path: &Path, loader_options: Value) -> Result<DatasetConfig, Error> {
        let cfg_path = cfg_path.canonicalize()?;
        let cfg_path_dir = cfg_path.parent().expect("absolute paths have a parent");
        let mut dataset_cfg: Value = serde_yaml::from_reader(File::open(&cfg_path)?)?;
        if let (Some(Value::Mapping(loader)), Value::Mapping(options)) =
            (dataset_cfg.get_mut("loader"), loader_options)
        {
            for (k, v) in options {
                loader.insert(k, v);
            }
        }
        let mut dataset_cfg: DatasetConfig = serde_yaml::from_value(dataset_cfg)?;
        for dir in &mut dataset_cfg.dirs {
            for path in dir.paths_mut() {
                if !path.is_absolute() {
                    *path = cfg_path_dir.join(&path);
                }
            }
        }
        Ok(dataset_cfg)
    }

    pub fn open(cfg_path: &Path, loader_options: Value) -> Result<DatasetConfig, Error> {
        DatasetConfig::do_open(cfg_path, loader_options).context(format!(
            "when opening dataset config {:?}",
            cfg_path.display()
        ))
    }

    pub fn load_epoc(&self) -> Result<Vec<DataId>, Error> {
        let mut ids = Vec::new();
        if let [path] = self.dirs.as_slice() {
            ids = self.loader.as_loader().load_epoc(path)?;
            if let Some(count) = path.count {
                ensure!(
                    ids.len() == count,
                    "expected {} samples in dataset, found {}",
                    count,
                    ids.len(),
                );
            }
        } else {
            for (i, path) in self.dirs.iter().enumerate() {
                let mut sub_ids = self.loader.as_loader().load_epoc(path)?;
                for id in &mut sub_ids {
                    id.whichdir = Some(i);
                }
                if let Some(count) = path.count {
                    ensure!(
                        sub_ids.len() == count,
                        "expected {} samples in dataset, found {}",
                        count,
                        sub_ids.len(),
                    );
                }
                ids.append(&mut sub_ids);
            }
        }
        ids.sort();
        Ok(ids)
    }
}

#[pyo3::pyclass(name = "Loader")]
pub struct PyLoader {
    pub config: DatasetConfig,
}

fn pathbuf_to_path_inner(py: Python<'_>, path: &Path) -> Result<PyObject, PyErr> {
    let path_class = py.import("pathlib")?.getattr("Path")?;
    let path_res = path_class.call1((path.as_os_str().to_owned(),))?;
    Ok(path_res.to_object(py))
}

fn pathbuf_to_path(py: Python<'_>, path: &Path) -> Result<PyObject, PyErr> {
    match pathbuf_to_path_inner(py, path) {
        Ok(p) => Ok(p),
        Err(_) => Ok(ToPyObject::to_object(path, py)),
    }
}

#[pyo3::pymethods]
impl PyLoader {
    #[new]
    #[pyo3(signature = (config_path, **kwargs))]
    fn new(config_path: PathBuf, kwargs: Option<&PyAny>) -> Result<Self, Error> {
        let mut options = Value::Null;
        if let Some(kwargs) = kwargs {
            options = depythonize(kwargs)?;
        }
        let config = DatasetConfig::open(&config_path, options)?;
        Ok(PyLoader { config })
    }

    #[pyo3(signature = (*paths))]
    fn with_svbrdfs(&self, paths: Vec<PathBuf>) -> Result<Self, Error> {
        let mut this = PyLoader {
            config: self.config.clone(),
        };
        ensure!(this.config.dirs.len() == paths.len());
        for (dir, path) in this.config.dirs.iter_mut().zip(paths) {
            dir.svbrdfs = Some(path);
        }
        Ok(this)
    }

    fn epoc(&self, py: Python<'_>) -> Result<PyObject, Error> {
        let ids = self.config.load_epoc()?;
        Ok(pythonize(py, &ids)?)
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> pythonize::Result<PyObject> {
        pythonize(py, &self.config)
    }

    fn save_config(&self, config_path: PathBuf) -> Result<(), Error> {
        serde_yaml::to_writer(File::create(&config_path)?, &self.config)?;
        Ok(())
    }

    fn load(&self, py: Python<'_>, id: &PyAny) -> Result<PyObject, Error> {
        let mut rng = StdRng::from_entropy();
        if id.is_instance_of::<pyo3::types::PyList>()? {
            let ids: Vec<DataId> = depythonize(&id)?;
            let samples: Result<BatchedSample, Error> = ids
                .into_iter()
                .map(|id| {
                    id.configure_rng(&mut rng);
                    self.config
                        .loader
                        .as_loader()
                        .load_sample(&self.config, &id, &mut rng)
                })
                .collect();
            Ok(samples?.into_py(py))
        } else {
            let id: DataId = depythonize(&id)?;
            id.configure_rng(&mut rng);
            let sample = self
                .config
                .loader
                .as_loader()
                .load_sample(&self.config, &id, &mut rng);
            Ok(sample?.into_py(py))
        }
    }

    fn data_path(&self, py: Python<'_>, id: &PyAny, part: &PyAny) -> Result<PyObject, Error> {
        let id: DataId = depythonize(id)?;
        let part: crate::Part = depythonize(part)?;
        let path = crate::data_path(&self.config, &id, &part)?;
        Ok(pathbuf_to_path(py, &path)?)
    }

    fn metadata_path(&self, py: Python<'_>, id: &PyAny, file: String) -> Result<PyObject, Error> {
        let id: DataId = depythonize(id)?;
        let part = crate::Part::SvbrdfMeta { file };
        let path = crate::data_path(&self.config, &id, &part)?;
        Ok(pathbuf_to_path(py, &path)?)
    }

    fn data_texture(&self, py: Python<'_>, id: &PyAny, part: &PyAny) -> Result<PyObject, Error> {
        let id: DataId = depythonize(id)?;
        let part: crate::Part = depythonize(part)?;
        let tex = crate::open_texture(&self.config, &id, &part)?;
        Ok(PyArray::from_owned_array(py, tex).into_py(py))
    }
}

#[pyo3::pyclass]
pub struct Generator {
    pub config: DatasetConfig,
    pub epoc: Option<Vec<DataId>>,
    #[pyo3(get, set)]
    pub seed: u64,
    #[pyo3(get, set)]
    pub batch_size: Option<usize>,
    #[pyo3(get, set)]
    pub replicates: usize,
    #[pyo3(get, set)]
    pub worker_count: usize,
    #[pyo3(get)]
    pub total_steps: Option<usize>,
    #[pyo3(get)]
    pub total_samples: Option<usize>,
    #[pyo3(get, set)]
    pub limit_samples: Option<usize>,
    #[pyo3(get, set)]
    pub whitelist: Option<Vec<String>>,
    pub batches: Option<Receiver<BatchedSample>>,
    pub handles: Vec<JoinHandle<()>>,
}

impl Generator {
    pub fn begin(&mut self) -> Result<(), Error> {
        use std::thread::spawn;

        let mut seed_bytes = [0; 32];
        seed_bytes[..8].copy_from_slice(&self.seed.to_be_bytes());
        let mut rng = StdRng::from_seed(seed_bytes);

        let mut epoc = match &self.epoc {
            Some(epoc) => epoc.clone(),
            None => {
                let mut first_epoc = self.config.load_epoc()?;
                first_epoc.sort();
                self.epoc = Some(first_epoc.clone());
                first_epoc
            }
        };
        epoc.shuffle(&mut rng);
        if let Some(limit) = self.limit_samples {
            epoc.truncate(limit);
        }

        let replicates = self.replicates;
        let batch_size = self.batch_size.unwrap_or(1);
        let use_batching = self.batch_size.is_some();
        let total_steps = (epoc.len() * replicates) / batch_size;
        self.total_steps = Some(total_steps);
        self.total_samples = Some(epoc.len() * replicates);

        let (id_send, id_recv) = bounded::<(DataId, usize)>(128);
        let (batch_send, batch_recv) = bounded::<BatchedSample>(8);
        self.batches = Some(batch_recv);

        self.handles.push(spawn(move || {
            for (j, id) in epoc.iter().enumerate() {
                let mut id = id.clone();
                let mut seed = <StdRng as SeedableRng>::Seed::default();
                rng.fill_bytes(&mut seed);
                id.seed = Some(seed);
                if let Err(_) = id_send.send((id.clone(), j)) {
                    return;
                }
            }
        }));

        for worker_ind in 0..self.worker_count {
            let config = self.config.clone();
            let id_recv = id_recv.clone();
            let batch_send = batch_send.clone();
            let whitelist = self.whitelist.clone();
            self.handles.push(spawn(move || {
                let mut rng = StdRng::from_entropy();
                let mut batch = Vec::with_capacity(batch_size);
                while let Ok((id, epoc_index)) = id_recv.recv() {
                    id.configure_rng(&mut rng);
                    match catch_unwind(AssertUnwindSafe(|| {
                        config
                            .loader
                            .as_loader()
                            .load_sample(&config, &id, &mut rng)
                    })) {
                        Ok(Ok(mut sample)) => {
                            let max_replicate =
                                replicates.checked_sub(1).expect("replicates not >0");
                            for index in 0..max_replicate {
                                sample.provide("id", &id.for_results(index));
                                sample.provide("seed", rng.gen::<u32>());
                                sample.provide("index", epoc_index);
                                match &whitelist {
                                    Some(w) if !w.is_empty() && !w.contains(&id.basic_name()) => {
                                        continue
                                    }
                                    _ => (),
                                }
                                batch.push(sample.clone())
                            }
                            sample.provide("id", &id.for_results(max_replicate));
                            sample.provide("seed", rng.gen::<u32>());
                            sample.provide("index", epoc_index);
                            match &whitelist {
                                Some(w) if !w.is_empty() && !w.contains(&id.basic_name()) => {
                                    continue
                                }
                                _ => (),
                            }
                            batch.push(sample);
                            if !use_batching {
                                for sample in batch.drain(..) {
                                    if let Err(_) =
                                        batch_send.send(BatchedSample::Unbatched(sample))
                                    {
                                        return;
                                    }
                                }
                            }
                        }
                        Ok(Err(err)) => {
                            eprintln!("ERROR loading batch on worker #{worker_ind}: {err:?}")
                        }
                        Err(err) => {
                            eprintln!("PANIC loading batch on worker #{worker_ind}: {err:?}")
                        }
                    }
                    if batch.len() >= batch_size {
                        if let Err(_) = batch_send.send(batch.drain(..batch_size).collect()) {
                            return;
                        }
                    }
                }
                while batch.len() >= batch_size {
                    if let Err(_) = batch_send.send(batch.drain(..batch_size).collect()) {
                        return;
                    }
                }
                if batch.len() > 0 {
                    if let Err(_) = batch_send.send(batch.into_iter().collect()) {
                        return;
                    }
                }
            }));
        }

        Ok(())
    }

    pub fn end(&mut self) -> Result<(), Error> {
        self.batches = None;
        for join in self.handles.drain(..) {
            if let Err(err) = join.join() {
                bail!("PANIC exited worker with: {err:?}");
            }
        }
        Ok(())
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        if let Err(e) = self.end() {
            eprintln!("{e}")
        }
    }
}

#[pyo3::pymethods]
impl Generator {
    #[new]
    #[pyo3(signature = (config_path, seed, batch_size=16, replicates=1, worker_count=16, limit_samples=None, whitelist=None, **kwargs))]
    pub fn py_new(
        config_path: PathBuf,
        seed: u64,
        batch_size: Option<usize>,
        replicates: usize,
        worker_count: usize,
        limit_samples: Option<usize>,
        whitelist: Option<Vec<String>>,
        kwargs: Option<&PyAny>,
    ) -> Result<Self, Error> {
        let mut options = Value::Null;
        if let Some(kwargs) = kwargs {
            options = depythonize(kwargs)?;
        }
        let config = DatasetConfig::open(&config_path, options)?;
        let gen = Generator {
            config,
            seed,
            batch_size,
            replicates,
            worker_count,
            whitelist,
            total_steps: None,
            total_samples: None,
            limit_samples,
            batches: None,
            epoc: None,
            handles: Vec::new(),
        };
        Ok(gen)
    }

    #[pyo3(name = "begin")]
    pub fn py_begin(&mut self) -> Result<(), Error> {
        self.end()?;
        self.begin()?;
        Ok(())
    }

    pub fn take(&self) -> Option<BatchedSample> {
        self.batches.as_ref()?.recv().ok()
    }

    #[getter]
    fn resolution(&self) -> usize {
        self.config.resolution
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> pythonize::Result<PyObject> {
        pythonize(py, &self.config)
    }

    #[getter]
    fn loader(&self) -> PyLoader {
        PyLoader {
            config: self.config.clone(),
        }
    }
}
