from PIL import Image
import imageio
from process_data.files_utils import collect
import os
from process_data.files_utils import init_folders
from show.viewer_mpl import view_mpl as view_backend
import matplotlib.pyplot as plt
from custom_types import *


def gifed(folder: str, interval: float, preffix: str = '', multi_mode: bool = True):
    files = collect(folder, '.png')
    last_tag, root = None, None
    images = []
    duration = []
    for file in files:
        if multi_mode:
            split = file[1].split('_')
            if len(split) < 3:
                continue
            cur_tag = split[0]
        else:
            cur_tag = ''
        if last_tag is not None and (cur_tag != last_tag or root != file[0]) and len(images) > 1:
            imageio.mimsave(f'{folder}{preffix}{last_tag}.gif', images, duration=interval)
            images = []
        file_name = ''.join(file)
        images.append(imageio.imread(file_name))
        if len(images) % 14 == 1:
            duration.append(0.4)
        else:
            duration.append(interval)
        last_tag = cur_tag
        root = file[0]
    if last_tag is not None and len(images) > 1:
        imageio.mimsave(f'{folder}{preffix}{last_tag}.gif', images, duration=interval)


def sample_on_mesh(mesh: tuple, face_areas: V, num_samples: int) -> V:
    vs, faces = mesh
    chosen_faces = faces[np.random.choice(range(len(face_areas)), size=num_samples, p=face_areas / face_areas.sum())]
    u, v = np.random.rand(num_samples, 1), np.random.rand(num_samples, 1)
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    w = 1 - u - v
    samples = u * vs[chosen_faces[:, 0]] + v * vs[chosen_faces[:, 1]] + w * vs[chosen_faces[:, 2]]
    return samples.astype(np.float32)


def init_palettes(cmap='Spectral'):
    colors = {}
    color_map = plt.cm.get_cmap(cmap)

    def get_palette(num_colors: int) -> list:
        nonlocal colors, color_map
        if num_colors == 1:
            return [.45]
        if num_colors not in colors:
            colors[num_colors] = [color_map(float(idx) / (num_colors - 1)) for idx in range(num_colors)]
        return colors[num_colors]

    return get_palette


def create_palettes(splits: list, pallete = None) -> list:
    if pallete is None:
        palette = []
        picker = init_palettes()
        for split in splits:
            num_colors = len(split) - 1
            palette.append(picker(num_colors))
    return pallete

#
# def init_palette(splits: list, palette):
#
#     # def build_palette(palette_: list, size: int):
#     #     ratio = float(len(palette_)) / size
#     #     if ratio == 1:
#     #         new_palette = palette_
#     #     elif ratio == 0:
#     #         new_palette = [int((i + .5) * 200 / size) for i in range(size)]
#     #     else:
#     #         new_palette = []
#     #         for i in range(size):
#     #             c = palette_[int(i * ratio)]
#     #             c = min(max(c + int(np.random.randint(-40, 41)), 0), 255)
#     #             new_palette.append(c)
#     #     return new_palette
#
#     if palette is None:
#         palette = []
#         for split in splits:
#             if len(split) == 2:
#                 palette.append([.3])
#             else:
#                 num_colors = len(split) - 1
#                 palette.append([ .1 + (.7 * i) / (num_colors - 1) for i in range(num_colors)])
#
#     return palette


def split_by_ref(points, refs):
    grained = []
    for pts, ref in zip(points,refs):
        grains = [pts[ref[i -1]: ref[i], :] for i in range(1, len(ref))]
        grained.append(grains)
    return grained

def is_kind_of_lst(thing):
    return type(thing) is tuple or type(thing) is list


def fix_points(samples, splits, global_transform):
    if not is_kind_of_lst(samples):
        if samples.ndim == 3:
            samples = [samples[i] for i in range(samples.shape[0])]
        else:
            samples = (samples,)
    if not is_kind_of_lst(splits):
        splits = (splits,)
    if len(splits) == 0:
        splits = [V([0, pts.shape[0]]) for pts in samples]
    if global_transform is not None:
        samples = [global_transform(sample) for sample in samples]
    return samples, splits


def reposition(samples, axis):
    if len(samples) > 1:
        delta = V([samples[i - 1][:, axis].std() for i in range(len(samples))]).mean() * 5 + .2
        for i in range(1, len(samples)):
            # delta = samples[i - 1][:, axis].max() - samples[i][:, axis].min()
            # samples[i][:, axis] = samples[i][:, axis] + delta + .5
            samples[i][:, axis] = samples[i][:, axis] + delta * i
    return samples


def get_bounds(*samples):
    bounds = [[np.expand_dims(func(sample, 0), 0) for sample in samples] for func in [np.min, np.max]]
    bounds = [np.expand_dims(func(np.concatenate(bound), 0),0) for bound, func in zip(bounds,[np.min, np.max])]
    return np.concatenate(bounds, 0)


def blend_images(images: List[V], blend_height: int, blend_width: int, rows: int) -> List[V]:
    cols = len(images) // rows
    for i in range(cols - 1):
        for j in range(rows):
            image_index = i + j * cols
            blend_a = images[image_index][:, -blend_width:]
            blend_b = images[image_index + 1][:, : blend_width]
            ma = blend_b < blend_a
            blend_a[ma] = blend_b[ma]
            images[image_index][:, -blend_width:] = blend_a
            images[image_index + 1] = images[image_index + 1][:, blend_width:]
    for i in range(rows - 1):
        for j in range(cols):
            image_index = i * cols + j
            blend_a = images[image_index][-blend_width:, :]
            blend_b = images[image_index + cols][: blend_width, :]
            ma = blend_b < blend_a
            blend_a[ma] = blend_b[ma]
            images[image_index][-blend_width:, :] = blend_a
            images[image_index + cols] = images[image_index + cols][blend_width:, :]

    # for j in range(blend_width):
    #     alpha = j / (blend_width -1)
    #     for im in images:
    #         im[:, j, :] =  alpha * im[:, j, :] + (1 - alpha) * 255.
    #         im[:, -j, :] = alpha * im[:, -j, :] + (1 - alpha) * 255.
    # for j in range(blend_height):
    #     alpha = j / (blend_height -1)
    #     for im in images:
    #         im[j, :, :] =  alpha * im[j, :, :] + (1 - alpha) * 255.
    #         im[-j, :, :] = alpha * im[-j, :, :] + (1 - alpha) * 255.
    return images


def make_pretty(np_images: List[V], offset=.01, blend=0.35, rows=1):
    if type(offset) is not tuple:
        offset = [offset] * 4
    offset = [- np_images[0].shape[idx % 2] if off == 0 else int(np_images[0].shape[idx % 2] * off) for idx, off in enumerate(offset)]
    cols = len(np_images) // rows
    np_images = np_images[: cols * rows]

    # offset_height, offset_width = int(np_images[0].shape[0] * offset ), int(np_images[0].shape[1] * offset)
    blend_height, blend_width = int(np_images[0].shape[0] * blend), int(np_images[0].shape[1] * blend)
    np_images = [image[offset[3]: - offset[1], offset[0]: - offset[2]] for image in np_images]
    # np_images = [image[offset_height: - offset_height, offset_width: - offset_width] for image in np_images]
    if blend != 0:
        np_images = blend_images(np_images, blend_height, blend_width, rows)
    np_images = [np.concatenate(np_images[i * cols: (i + 1) * cols], axis=1) for i in range(rows)]
    im = np.concatenate(np_images, axis=0)
    im = Image.fromarray(im)
    return im


def images_to_numpy(*paths):
    images = [V(Image.open(path)) for path in paths]
    return images


def view(points: V or list, splits: V or list or tuple = (), palette=None, save_path='', global_transform=None, titles=(), rows=1):
    points, splits = fix_points(points, splits, global_transform)
    bounds = get_bounds(*points)
    palette = create_palettes(splits, palette)
    np_images = view_backend(points, splits, bounds, palette, titles)
    im = make_pretty(np_images, rows=rows)
    if save_path:
        init_folders(save_path)
        im.save(save_path)
    elif 'DISPLAY' in os.environ:
        im.show()
    return im, points
