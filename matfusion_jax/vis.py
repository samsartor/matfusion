import numpy as np
from torch import Tensor
import jax.numpy as jnp
import math
import ffmpegio
import imageio
import wandb
import io
from pathlib import Path
from collections import defaultdict

def imnp(img):
    if isinstance(img, Tensor):
        return img.permute(0, 2, 3, 1).numpy()
    else:
        return np.array(img)


def imbytes(img):
    img = imnp(img)
    img = np.nan_to_num(img)
    img = np.clip(img, 0.0, 1.0)
    img *= 255.0
    return img.astype('uint8')


def imcond(img):
    img = imnp(img)
    return np.concatenate(np.dsplit(img, math.ceil(img.shape[2]/3)), axis=-2)


def imsvbrdf(img, geo='normals', vertical=False, horizontal=False, **kwargs):
    img = imnp(img)
    roughness = np.tile(img[..., 6:7], (1, 1, 3))
    if geo == 'normals':
        geo = img[..., 7:10]
    else:
        geo = img[..., 7:8]
        geo -= jnp.min(geo, axis=(-3, -2), keepdims=True)
        geo = np.tile(geo, (1, 1, 3))
    if vertical:
        return np.concatenate([
            img[..., 0:3],
            img[..., 3:6],
            roughness,
            geo,
        ], axis=-3)
    elif horizontal:
        return np.concatenate([
            img[..., 0:3],
            img[..., 3:6],
            roughness,
            geo,
        ], axis=-2)
    else:
        return np.concatenate((
            np.concatenate((img[..., 0:3], img[..., 3:6]), axis=-2),
            np.concatenate((roughness,     geo          ), axis=-2),
        ), axis=-3)


def serialize_image(img, format='webp', **kwargs):
    output = io.BytesIO()
    imageio.v3.imwrite(output, img, extension='.' + format, **kwargs)
    return output.getbuffer()


def deserialize_image(img, format=None):
    input = io.BytesIO(img)
    return imageio.v3.imread(input, format_hint=format)


def display_svbrdf(arr, gamma=1.0, **kwargs):
    from IPython import display
    arr = arr.copy()
    arr[..., 0:6] = arr[..., 0:6] ** (1/gamma)
    return display.Image(serialize_image(imsvbrdf(imbytes(arr), **kwargs), **kwargs))


def show_svbrdf(arr, **kwargs):
    from IPython.display import display
    display(display_svbrdf(arr, **kwargs))


def display_image(arr, gamma=1.0, **kwargs):
    from IPython import display
    arr = arr**(1/gamma)
    return display.Image(serialize_image(imbytes(arr), **kwargs))


def show_image(arr, **kwargs):
    from IPython.display import display
    display(display_image(arr, **kwargs))

ffmpeg_args = {
    'vcodec': 'libx264',
    'vf': 'format=yuv420p',
    'profile:v': 'main',
    'crf': 10, # nearly lossless
}

def save_image_video(path, arr, fps=30, gamma=1.0, **kwargs):
    arr = np.clip(arr, 0, 1)**(1/gamma)
    vid = imbytes(arr)
    ffmpegio.video.write(
        path,
        fps,
        vid,
        **ffmpeg_args,
        overwrite=True)


def save_svbrdf_video(path, arr, fps=30, gamma=1.0, **kwargs):
    arr = arr.copy()
    arr[..., 0:6] = jnp.clip(arr[..., 0:6], 0, 1) ** (1/gamma)
    vid = imbytes(imsvbrdf(arr, **kwargs))
    ffmpegio.video.write(
        path,
        fps,
        vid,
        **ffmpeg_args,
        overwrite=True)


class Report:
    contents = {}
    running = defaultdict(lambda: np.empty((0,)))

    def clear(self):
        self.contents.clear()
        self.running.clear()

    def add(self, path, value, mean=False, **kwargs):
        path = path.format(**kwargs)
        if mean:
            self.running[path] = np.append(self.running[path], value)
            self.contents[path] = np.mean(self.running[path])
        else:
            self.contents[path] = value

    def scalar(self, path, value, **kwargs):
        if not np.isscalar(value):
            value = value.item()
        self.add(path, value, **kwargs)

    def image(self, path, img, **kwargs):
        self.add(path, imbytes(img), **kwargs)

    def video(self, path, img, fps=30, **kwargs):
        self.add(path, imbytes(img), **kwargs)

    def cond_image(self, path, img, **kwargs):
        self.image(path, imcond(img), **kwargs)

    def svbrdf_image(self, path, img, geo='normals', **kwargs):
        self.image(path, imsvbrdf(img, geo=geo), **kwargs)

    def svbrdf_video(self, path, img, geo='normals', **kwargs):
        self.video(path, imsvbrdf(img, geo=geo), **kwargs)


class PrintReport(Report):
    def image(self, *args, **kwargs):
        pass

    def video(self, *args, **kwargs):
        pass

    def submit(self, **kwargs):
        print(self.contents)


class WandbReport(Report):
    def image(self, path, img, **kwargs):
        self.add(path, wandb.Image(imbytes(img)), **kwargs)

    def video(self, path, img, fps=30, **kwargs):
        self.add(path, wandb.Video(imbytes(img), fps=fps), **kwargs)

    def submit(self, step, **kwargs):
        wandb.log(self.contents, step=step)


class ResultsReport(Report):
    def __init__(self, dir):
        self.dir = dir

    def image(self, path, img, **kwargs):
        self.add(path, ('image', imbytes(img)), **kwargs)

    def video(self, path, img, fps=30, **kwargs):
        self.add(path, ('video', imbytes(img), fps), **kwargs)

    def submit(self, step=None, **kwargs):
        submit_path: Path = self.dir
        if step is not None:
            submit_path = submit_path / str(step)
        submit_path.mkdir(parents=True)
        for (path, item) in self.contents.items():
            path = submit_path / path.replace('/', '_')
            if isinstance(item, tuple) and item[0] in ('image', 'video'):
                imageio.v3.imwrite(
                    path.with_name(path.name + '.webp'),
                    item[1], quality=95,
                )
            else:
                path.with_name(path.name + '.txt').write_text(str(item))


class IPythonReport(Report):
    def image(self, path, img, **kwargs):
        self.add(path, display_image(np.array(img), **kwargs), **kwargs)

    def video(self, path, img, fps=30, **kwargs):
        self.add(path, display_image(np.array(img), fps=fps, **kwargs), **kwargs)

    def svbrdf_image(self, path, img, **kwargs):
        self.add(path, display_svbrdf(np.array(img), **kwargs), **kwargs)

    def svbrdf_video(self, path, img, fps=30, **kwargs):
        self.add(path, display_svbrdf(np.array(img), fps=fps, **kwargs), **kwargs)

    def submit(self, **kwargs):
        from IPython import display
        objs = []
        for (path, item) in self.contents.items():
            objs.append(display.Markdown(f'# {path}'))
            objs.append(item)
        display.display(*objs)

