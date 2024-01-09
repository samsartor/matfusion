import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm

datasets_dir = Path(__file__).absolute().parent.parent / 'datasets'
input_dir = datasets_dir / 'DeepMaterialsData' / 'trainBlended'
output_dir = datasets_dir / 'inria_svbrdfs'

if output_dir.exists():
    print('Output datasets/inrea_svbrdfs directory already exists. Exiting')
    exit(1)

(output_dir / 'diffuse').mkdir(parents=True, exist_ok=True)
(output_dir / 'specular').mkdir(parents=True, exist_ok=True)
(output_dir / 'roughness').mkdir(parents=True, exist_ok=True)
(output_dir / 'normals').mkdir(parents=True, exist_ok=True)

image_paths = list(input_dir.iterdir())
for image_path in tqdm(image_paths):
    if not image_path.name.endswith('.png'):
        continue
    name = image_path.name.split(';')[0]
    svbrdf = iio.imread(image_path)
    try:
        normals = svbrdf[:, 288*1:288*2, 0:3]
        iio.imwrite(output_dir / 'normals' / f'{name}_normals.png', normals)
        diffuse = svbrdf[:, 288*2:288*3, 0:3]
        iio.imwrite(output_dir / 'diffuse' / f'{name}_diffuse.png', diffuse)
        roughness = svbrdf[:, 288*3:288*4, 0:1]
        iio.imwrite(output_dir / 'roughness' / f'{name}_roughness.png', roughness)
        specular = svbrdf[:, 288*4:288*5, 0:3]
        iio.imwrite(output_dir / 'specular' / f'{name}_specular.png', specular)
    except Exception as e:
        print(f'Error processing {image_path}:', e)
        print(svbrdf.shape)

