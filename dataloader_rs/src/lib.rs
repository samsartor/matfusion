mod ids;
pub use ids::*;
mod gen;
pub use gen::*;
mod form;
pub use form::*;
mod warp;
use pythonize::depythonize;
use rayon::prelude::*;
pub use warp::*;
pub mod loaders;

use anyhow::Error;
use image::{DynamicImage, Rgb32FImage, RgbImage};
use ndarray::{s, Array3};
use pyo3::{pymodule, types::PyModule, PyAny, PyResult, Python};
use std::{fs::create_dir_all, path::PathBuf};

pub fn array2rgbf(arr: Array3<f32>) -> Rgb32FImage {
    let shape = arr.dim();
    assert_eq!(shape.2, 3, "must have 3 channels");
    Rgb32FImage::from_raw(shape.1 as u32, shape.0 as u32, arr.into_raw_vec())
        .expect("could not create image object")
}

pub fn array2rgb8(arr: Array3<f32>) -> RgbImage {
    DynamicImage::ImageRgb32F(array2rgbf(arr)).into_rgb8()
}

#[pyo3::pyfunction]
#[pyo3(name = "warp_svbrdf")]
pub fn warp_svbrdf_py<'py>(
    py: Python<'py>,
    svbrdf: &numpy::PyArray3<f32>,
    uvs: &numpy::PyArray3<f32>,
) -> &'py numpy::PyArray3<f32> {
    let svbrdf = svbrdf.to_owned_array();
    let mut uvs = uvs.to_owned_array();
    let mut dest = Array3::zeros([uvs.dim().0, uvs.dim().1, svbrdf.dim().2]);
    uvs.slice_mut(s![.., .., 0])
        .mapv_inplace(|x| x * (svbrdf.dim().1 - 1) as f32);
    uvs.slice_mut(s![.., .., 1])
        .mapv_inplace(|y| y * (svbrdf.dim().0 - 1) as f32);
    let uvs = warp::UvBuffer(uvs.view());
    warp::reverse_warp(
        warp::SvbrdfInterpolation(warp::BilinearInterpolation(svbrdf.view())),
        dest.view_mut(),
        |pt| uvs.query(pt),
        |_, mut arr| arr.fill(f32::NAN),
    );
    numpy::PyArray3::from_owned_array(py, dest)
}

#[pyo3::pyfunction]
#[pyo3(name = "save_exr")]
fn save_exr_py(arr: &numpy::PyArray3<f32>, path: PathBuf) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    let arr = arr.to_owned_array();
    let img = array2rgbf(arr);
    img.save(path)?;
    Ok(())
}

#[pyo3::pyfunction]
#[pyo3(name = "save_png")]
fn save_png_py(arr: &numpy::PyArray3<f32>, path: PathBuf) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    let arr = arr.to_owned_array();
    let img = array2rgb8(arr);
    img.save(path)?;
    Ok(())
}

#[pyo3::pyfunction]
#[pyo3(name = "save_png_batch")]
fn save_png_batch_py(arr: &numpy::PyArray4<f32>, paths: Vec<PathBuf>) -> Result<(), Error> {
    let arr = arr.to_owned_array();
    paths
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, path)| {
            let img = array2rgb8(arr.slice(s![i, .., .., ..]).to_owned());
            if let Some(parent) = path.parent() {
                let _ = create_dir_all(parent);
            }
            img.save(path)
        })?;
    Ok(())
}

#[pyo3::pyfunction]
#[pyo3(name = "save_svbrdf")]
fn save_svbrdf_py(
    arr: &numpy::PyArrayDyn<f32>,
    loader: &PyLoader,
    id: &PyAny,
) -> Result<(), Error> {
    if id.is_instance_of::<pyo3::types::PyList>()? {
        let arr = arr.to_owned_array();
        let ids: Vec<DataId> = depythonize(id)?;
        for (i, id) in ids.into_iter().enumerate() {
            save_texture(
                &loader.config,
                &id,
                &Part::Diffuse,
                arr.slice(s![i, .., .., 0..3]).to_owned(),
            )?;
            save_texture(
                &loader.config,
                &id,
                &Part::Specular,
                arr.slice(s![i, .., .., 3..6]).to_owned(),
            )?;
            save_texture(
                &loader.config,
                &id,
                &Part::Roughness,
                arr.slice(s![i, .., .., 6..7]).to_owned(),
            )?;
            save_texture(
                &loader.config,
                &id,
                &Part::Normals,
                arr.slice(s![i, .., .., 7..10]).to_owned(),
            )?;
        }
    } else {
        let arr = arr.to_owned_array();
        let id: DataId = depythonize(id)?;
        save_texture(
            &loader.config,
            &id,
            &Part::Diffuse,
            arr.slice(s![.., .., 0..3]).to_owned(),
        )?;
        save_texture(
            &loader.config,
            &id,
            &Part::Specular,
            arr.slice(s![.., .., 3..6]).to_owned(),
        )?;
        save_texture(
            &loader.config,
            &id,
            &Part::Roughness,
            arr.slice(s![.., .., 6..7]).to_owned(),
        )?;
        save_texture(
            &loader.config,
            &id,
            &Part::Normals,
            arr.slice(s![.., .., 7..10]).to_owned(),
        )?;
    }
    Ok(())
}

#[pyo3::pyfunction]
#[pyo3(name = "save_texture")]
fn save_texture_py(
    arr: &numpy::PyArrayDyn<f32>,
    loader: &PyLoader,
    id: &PyAny,
    part: &PyAny,
) -> Result<(), Error> {
    let part: Part = depythonize(part)?;
    if id.is_instance_of::<pyo3::types::PyList>()? {
        let arr = arr.to_owned_array();
        let ids: Vec<DataId> = depythonize(id)?;
        for (i, id) in ids.into_iter().enumerate() {
            save_texture(
                &loader.config,
                &id,
                &part,
                arr.slice(s![i, .., .., ..]).to_owned(),
            )?;
        }
    } else {
        let arr = arr.to_owned_array();
        let id: DataId = depythonize(id)?;
        save_texture(
            &loader.config,
            &id,
            &part,
            arr.slice(s![.., .., 0..3]).to_owned(),
        )?;
    }
    Ok(())
}

#[pymodule]
fn data(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    use pyo3::prelude::*;

    m.add_class::<Generator>()?;
    m.add_class::<PyLoader>()?;
    m.add_function(wrap_pyfunction!(save_exr_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_png_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_png_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_svbrdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_texture_py, m)?)?;
    m.add_function(wrap_pyfunction!(warp_svbrdf_py, m)?)?;
    Ok(())
}
