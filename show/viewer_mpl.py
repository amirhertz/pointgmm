from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from custom_types import *

mpl.use('Agg')


def clear_outside_points(points, bounds):
    mask = np.ones(points.shape[0], dtype=np.bool)
    for i in range(points.shape[1]):
        axis_mask = np.logical_and(bounds[0][i] <= points[:, i], points[:, i] <= bounds[1][i])
        mask = np.logical_and(mask, axis_mask)
    return points[mask]


def add_points(container, points, bounds, split, color: int or list or tuple):
    if type(color) is int or type(color) is tuple:
        color = (split.shape[0] - 1) * [color]
    for i in range(0, split.shape[0] - 1):
        c = color[i]
        if type(c) is int or type(c) is float:
            c = (c, c, c)
        if type(c[0]) is int:
            c = [float(c[i]) / 255. for i in range(3)]
        cur_points = clear_outside_points(points[split[i]: split[i + 1], :], bounds)
        if len(cur_points) > 0:
            container.scatter(cur_points[:, 0], cur_points[:, 1], cur_points[:, 2], marker='o', s=3, c=((c[0],c[1],c[2]),), alpha=.4)


def init_fig(bounds: V, num_objects:int):
    fig = plt.figure(figsize=(max(num_objects * 2, 3.5), 3.5))
    ax = fig.gca(projection='3d')
    scale = bounds[1] - bounds[0]
    scale = np.diag([scale[0], scale[1], scale[2], 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.dist = 7
    return ax, fig


def fig2data(fig: plt.Figure):
    # taken from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    # Thanks!
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


def extrude_bounds(bounds:V) -> V:
    bounding_box = np.zeros((2 ** bounds.shape[1], bounds.shape[1]))
    f = 1
    for axis in range(bounding_box.shape[1]):
        for corner in range(bounding_box.shape[0]):
            bounding_box[corner, axis] = bounds[(corner % (f * 2)) // f, axis]
        f = f * 2
    return bounding_box


def view_mpl(points: list, splits: list, bounds: V, palette: int or list, titles: Tuple[str]):
    numpy_images = []
    bounding_box = extrude_bounds(bounds)
    titles = [titles[i] if i < len(titles) else '' for i in range(len(points))]
    for pts, split, color, title in zip(points, splits, palette, titles):
        ax, fig = init_fig(bounds, 1)
        add_points(ax, bounding_box, bounds, V([0, 8], dtype=np.int), 255)
        add_points(ax, pts, bounds, split, color)
        if title:
            ax.set_title(title)
        numpy_images.append(fig2data(fig))
        plt.close(fig)
    return numpy_images

