# MatFusion [[Paper](https://bin.samsartor.com/matfusion.pdf)] [[Website](https://samsartor.com/matfusion)]

![](https://samsartor.com/matfusion_teaser.png)

SVBRDF estimation from photographs for three different lighting conditions (directional, natural, and flash/no-flash illumination) is shown by refining a novel SVBRDF diffusion backbone model, named MatFusion.

## Citation

```bibtex
@conference{Sartor:2023:MFA,
    author    = {Sartor, Sam and Peers, Pieter},
    title     = {MatFusion: a Generative Diffusion Model for SVBRDF Capture},
    month     = {December},
    year      = {2023},
    booktitle = {ACM SIGGRAPH Asia Conference Proceedings},
}
```

## Installation

You will need Git, Python, and Conda on a system with at least CUDA 11 and CUDNN 8.2. This code is only tested on Linux, but Windows and MacOS may be usable with some tweaks.

```sh
git clone 'https://github.com/samsartor/matfusion' && cd matfusion
git submodule update --init --recursive
conda env create -f environment.yml
conda activate matfusion
pip install -e matfusion_diffusers
ipython kernel install --user --name=matfusion
```

Before running a notebook in Jupyter, make sure to set your kernel to "matfusion".

### Optional Dependencies

When running more complicated evaluation and training jobs you may need additional dependencies.

To render SVBRDFs under environment lighting or with global illumination, you should install Blender version 3.6 or higher.

To compile the `matfusion_jax.data` module, install Rust version 1.65 or higher and compile the dataloader_rs package.

```sh
cargo build --manifest-path dataloader_rs/Cargo.toml --release
```

In order to finetune the flash/no-flash model you will also need to pass `--features ndarray-linalg` to cargo, which will download and compile openblas so that the dataloader can simulate imperfect camera alignment between pairs of rendered images.

You may also need a Julia environment compatible with language version 1.7. Also install the required julia packages by running the `julia` command and entering:

```julia
import Pkg
Pkg.add(name="Images", version="0.23.3")
Pkg.add(name="FFTW", version="1.6.0")
Pkg.add(name="ProgressMeter", version="1.7.2")
```

These dependencies are NOT needed if you only use the demo notebooks or otherwise the write code to assemble batches yourself. They are only needed when using our dataset processing code through scripts like `train.py` and `eval.py`.

## Pretrained Models

The following pretrained models are avalible.

| Model Finetuning | Framework | Version |  Download |
| ---------------- | --------- | ------- | --------- |
| *Backbone*       | Jax       | 1       | [unconditional_v1_jax.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/unconditonal_v1_jax.tar.lz4) |
| Flash            | Jax       | 1       | [flash_v1_jax.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/flash_v1_jax.tar.lz4) |
| Environment      | Jax       | 1       | [env_v1_jax.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/env_v1_jax.tar.lz4) |
| Flash/No-flash   | Jax       | 1       | [fnf_v1_jax.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/fnf_v1_jax.tar.lz4) |
| *Backbone*       | Diffusers | 1       | [unconditional_v1_diffusers.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/unconditional_v1_diffusers.tar.lz4) |
| Flash            | Diffusers | 1       | [flash_v1_diffusers.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/flash_v1_diffusers.tar.lz4) |
| Environment      | Diffusers | 1       | [env_v1_diffusers.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/env_v1_diffusers.tar.lz4) |
| Flash/No-flash   | Diffusers | 1       | [fnf_v1_diffusers.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/fnf_v1_diffusers.tar.lz4) |

To use any of the pretrained models above, untar the downloaded archive into the `checkpoints` folder.

## Inference

The easiest way to run MatFusion on your own photographs is with the `matfusion_jax_demo.ipynb` and `matfusion_diffusers_demo.ipynb` Jupyter notebooks.

Alternatively you can create an `!Images` dataset for your photographs. For example, the [MaterialGAN](https://github.com/tflsguoyu/materialgan) paper by Yu et al. provides a selection of real flash-lit photographs which you can download from https://www.dropbox.com/s/6k3n5xntelqeypk/in.zip, unzip into the `datasets` directory, and rename to `datasets/materialgan_real_inputs`. Then you can batch-process the test images with `eval.py`.

```sh
python eval.py \
    --dataset datasets/real_test_materialgan.yml \
    --checkpoint checkpoints/flash_v1_jax \
    --output results/flash_v1_on_materialgan`
```

## Datasets

MatFusion was trained on three different datasets of SVBRDFs which each have a slightly different download process. Some fine tunings also require path-traced images of those SVBRDFs.

### INRIA

The inria dataset can be downloaded from https://team.inria.fr/graphdeco/projects/deep-materials/. Unzip it into the `datasets` directory of this repo so that `datasets/DeepMaterialsData` is populated by lots of png files and then run `python scripts/convert_inria.py`. You should see the script create a `datasets/inria_svbrdfs` folder.

### CC0

Download and untar [cc0_svbrdfs.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/cc0_svbrdfs.tar.lz4) into the `datasets` directory so that it contains a `datasets/cc0_svbrdfs` folder.

### Mixed

_Coming Soon..._

<!--Download and untar TODO into the `datasets` directory so that it contains a `datasets/mixed_svbrdfs` folder.-->

### Rendering

Rendering the SVBRDFs is very CPU intensive and also requires about 1TB of free storage (since the renders are stored in OpenEXR format). This data is impossible to distribute online because of its size, but feel free to email us if rendering it yourself proves impossible.

First you should integrate the normal maps to produce displacement maps by running `julia scripts/integrate-normals.jl datasets/*_svbrdfs` and waiting a few hours.

To open several background Blender workers to render all the maps over a few days, run `dataloader_rs/target/release/renderall datasets/train_rendered_env.yml`. You may need to adjust worker and thread counts for optimal performance, since the defaults are for a machine with 128 cores.

### Test Data

Our paper also presents a new test set, made up of diverse SVBRDFs from a variety of sources. Download and untar [test_svbrdf.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/test_svbrdfs.tar.lz4) into the datasets directory so that it contains a `test_svbrdfs` folder.

## Compute Error Statistics

You can use the `eval.py` and `score.py` scripts to run rigorous evaluations of MatFusion. For example, to evaluate the
flash model on our test set (once downloaded), use:
```bash
python eval.py \
    --dataset ./datasets/test_rasterized.yml \
    --checkpoint ./checkpoints/flash_v1_jax \
    --output ./results/flash_v1_on_test
python score.py --output  ./results/flash_v1_on_test
```

The error numbers can be viewed with the `view_eval.ipynb` Jupyter notebook.

## Training & Finetuning

All of the various training routines can be accomplished with the `train.py` script. For our training we generally used 4x NVIDIA A40 GPUs each with 45GB of memory. If you are using more or less compute you should probably adjust the batch size and learning rate by passing `-O batch_size={BATCH_SIZE}` and `-O lr={LEARNING_RATE}` or by passing a custom `--mode_json {PATH_TO_JSON_FILE}`. Alternatively, we provide gradient accumulation with the `--accumulation` option, but it is not very well tested.

### Unconditional Backbone Model
```sh
python train.py \
    --mode CONVNEXT_V1_UNCONDITIONAL_MODE \
    --epocs 50 \
    --dataset ./datasets/train_rasterized.yml
```

### Flash Finetuning
```sh
python train.py \
    --finetune_checkpoint checkpoints/unconditional_v1_jax \
    --mode CONVNEXT_V1_DIRECT_RAST_IMAGES_MODE \
    --epocs 19 \
    --dataset ./datasets/train_rasterized.yml
```

### Environment-Lit Finetuning
```sh
python train.py \
    --finetune_checkpoint checkpoints/unconditional_v1_jax \
    --mode CONVNEXT_V1_DIRECT_RENDERED_IMAGES_MODE \
    --epocs 19 \
    --dataset ./datasets/train_rendered_env.yml
```

### Flash/No-Flash Finetuning
```sh
python train.py \
    --finetune_checkpoint checkpoints/unconditional_v1_jax \
    --mode CONVNEXT_V1_DIRECT_RENDERED_OTHER_IMAGES_MODE \
    --epocs 19 \
    --dataset ./datasets/train_rendered_fnf.yml
```

## Project Structure

- `train.py` is our training script
- `eval.py` is our evaluation script
- `score.py` does error number computation for evaluations
- `matfusion_jax/` is our from-scratch diffusion implementation in Jax/Flax
    - `model.py` contains logic for saving/loading/initing/training models
    - `net/resnet.py` implements fundamental model layers
    - `net/mine.py` implements our actual diffusion backbone
    - `pipeline.py` defines diffusion schedule and samplers
    - `config.py` has all the default model configuration modes
    - `nprast.py` implements the cook-torrance SVBRDF rasterizer
    - `vis.py` has lots of utils for displaying results during training and evaluation
- `matfusion_diffusers/` is our fork of huggingface diffusers needed to run the MatFusion model in PyTorch
- `dataloader_rs/` is our Rust codebase for managing and generating datasets
    - `src/lib.rs` exposes the API
    - `src/gen.rs` actually does the dataloading in a Python-accessible way
    - `src/loaders.rs` has the logic for loading different dataset formats and modes
    - `src/ids.rs` identifies training samples and the textures that make them up
    - `src/form.rs` turns raw image files on disk into actual SVBRDFs and tonemapped renderings
    - `src/warp.rs` is used for transforming images and SVBRDFs, including to pixel-align pairs of flash/no-flash renderings
    - `src/bin/` has executable Rust scripts
        - `renderall.rs` dispatches bulk blender render jobs as defined by the render_synthetic scripts
        - `mixsvbrdfs.rs` implements our mixture augmentation as used to make the mixed dataset
- `scripts/` contains misc utilities
    - `compress_exrs.jl` is a Julia script for compressing large numbers of EXR files
    - `integrate_normals.jl` is a Julia script for integrating normal maps to produce heightmaps
    - `convert_to_diffusers.py` can convert our checkpoint files into huggingface-compatable checkpoint files
    - `prune_checkpoint.py` removes checkpointed parameters that are not needed for inference
    - `render_synthetic/` has our synthetic blender scenes and related python scripts
- `datasets/` contains the various datasets and corresponding YAML specifications
- `checkpoints/` can contain various pretrained models
