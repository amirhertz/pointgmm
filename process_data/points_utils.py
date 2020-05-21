import numpy as np
from typing import Tuple


V = np.array


def create_dist_mat(points, min=True):
    mat_square = np.matmul(points, points.T)
    diag = np.expand_dims(np.diag(mat_square), axis=1).repeat(points.shape[0], 1)
    dist_mat = diag + diag.T - 2 * mat_square
    if min:
        dist_mat[np.arange(points.shape[0]), np.arange(points.shape[0])] = 100
        dist = np.sqrt(dist_mat.min(1))
        return dist
    return dist_mat




def apply_affine(vs, r, t):
    vs = np.einsum('nd,rd->nr', vs, r)
    return vs + t[np.newaxis, :]


def add_noise(vs: V, std: float, in_place: bool):
    noise = np.random.normal(0, std, vs.shape)
    if in_place:
        vs += noise
    else:
        vs = vs + noise
    return vs


def sub_sample_points(base_pc: V, sample_size: int) -> V:
    assert (sample_size <= base_pc.shape[0])
    return base_pc[np.random.choice(base_pc.shape[0], sample_size)]


def get_random_partition(base_pc: V, p: float) -> V:
    rotated_pc = transform_rotation((base_pc,))[0][0][:, 1]
    num_points = int(p * base_pc.shape[0])
    idx = np.argsort(rotated_pc)
    return base_pc[idx[: num_points]]


# def get_random_rotation(one_axis=False):
#
#     def random_rot():
#         nonlocal r
#         b, c = np.random.rand(2)
#         v = np.zeros((3, 1))
#         v[0, 0] = np.cos(2 * np.pi * b) * np.sqrt(c)
#         v[1, 0] = np.sin(2 * np.pi * b) * np.sqrt(c)
#         v[2, 0] = np.sqrt(1 - c)
#         h = np.eye(3) - 2 * np.matmul(v, v.T)
#         m = - np.matmul(h, r)
#         return m
#
#     a = np.random.rand(1)
#     r = np.zeros((3, 3))
#     r[0, 0] = np.cos(2 * np.pi * a)
#     r[0, 1] = np.sin(2 * np.pi * a)
#     r[1, 0] = - np.sin(2 * np.pi * a)
#     r[1, 1] = np.cos(2 * np.pi * a)
#     r[2, 2] = 1
#     if not one_axis:
#         r = random_rot()
#     return r.astype(np.float32)


def get_random_rotation(one_axis=False, max_angle=-1):

    theta = np.random.rand(1)
    if max_angle > 0:
        if theta > .5:
            theta = 1 - (theta - .5) * max_angle / 180
        else:
            theta = theta * max_angle / 180
    r = np.zeros((3, 3))
    if one_axis:
        axis = V([0, 0, 1.])
    else:
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis, 2)
    cos_a = np.cos(np.pi * theta)
    sin_a = np.sin(np.pi * theta)
    q = list(cos_a) + list(sin_a * axis)
    q = [[q[i] * q[j] for i in range(j + 1)] for j in range(len(q))]
    r[0, 0] = .5 - q[2][2] - q[3][3]
    r[1, 1] = .5 - q[1][1] - q[3][3]
    r[2, 2] = .5 - q[1][1] - q[2][2]
    r[0, 1] = q[2][1] - q[3][0]
    r[1, 0] = q[2][1] + q[3][0]
    if not one_axis:
        r[0, 2] = q[3][1] + q[2][0]
        r[2, 0] = q[3][1] - q[2][0]
        r[1, 2] = q[3][2] - q[1][0]
        r[2, 1] = q[3][2] + q[1][0]
    r *= 2

    return r.astype(np.float32), theta.astype(np.float32), axis.astype(np.float32)


def get_center(vs: V) -> V:
    max_vals = np.max(vs, axis=0)
    min_vals = np.min(vs, axis=0)
    center = (max_vals + min_vals) / 2
    return center


def to_center(vs: V) -> V:
    return vs - get_center(vs)[None, :]


def translate_to_center(points: tuple, translate_by=-1):
    if translate_by < 0:
        transformed = [to_center(pts) for pts in points]
        transform = None
    else:
        transform = - get_center(points[translate_by])
        transformed = [pts + transform[None, :] for pts in points]
    return transformed, transform


def transform_rotation(points: tuple, one_axis=False, max_angle=-1):
    r, theta, axis = get_random_rotation(one_axis, max_angle)
    transformed = [np.einsum('nd,rd->nr', pts, r) for pts in points]
    return transformed, [r, theta, axis]


def transform_translation(points: tuple, from_to: tuple):
    delta = np.random.rand(3) * (from_to[1] - from_to[0]) + from_to[0]
    transformed = [pts + delta[np.newaxis, :] for pts in points]
    return transformed, delta


def apply_transforms(transforms, *points) -> tuple:
    transform_switch = {'rotate': transform_rotation, 'translate': transform_translation, 'translate_to_center': translate_to_center}
    applied_transforms = []
    for transform in transforms:
        points, applied = transform_switch[transform[0]](points, *transform[1])
        if type(applied) is list:
            applied_transforms += applied
        elif applied is not None:
            applied_transforms.append(applied)
    return points, applied_transforms


def z_axis_rotaion(theta):
    rot = np.eye(3)
    if type(theta) is float or theta.shape[0] == 1:
        theta = 2 * np.pi * theta
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    else:
        cos_theta, sin_theta = theta

    rot[0, 0] = cos_theta
    rot[1, 1] = cos_theta
    rot[1, 0] = sin_theta
    rot[0, 1] = - sin_theta
    return rot


def to_affine(r, t):
    affine = np.eye(4)
    if r is not None:
        if type(r) is float or r.shape[0] < 3:
            r = z_axis_rotaion(r)
        affine[:3, :3] = r
    if t is not None:
        affine[:3, 3] = t
    return affine


def from_affine(affine: V) -> Tuple[V, V]:
    rot = affine[:3, :3]
    translate = affine[:3, 3]
    return rot, translate


def combine_affines(*affines):
    if len(affines) == 0:
        return np.eye(4)
    matrices = [to_affine(*affine) for affine in affines]
    affine = matrices[0]
    for mat in matrices[1:]:
        affine = affine @ mat
    return affine