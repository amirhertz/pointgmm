import numpy as np
from collections import deque
import os
import  process_data.points_utils as pc_utils
from typing import Tuple
from constants import EPSILON


V = np.array


class MeshWrap:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def mesh(self):
        return self.data['vs'], self.data['faces']


def compute_face_areas(mesh: tuple) -> Tuple[V, V]:
    vs, faces = mesh
    face_normals = np.cross(vs[faces[:, 1]] - vs[faces[:, 0]], vs[faces[:, 2]] - vs[faces[:, 1]])
    face_normals = np.linalg.norm(face_normals, axis=1)
    face_areas = 0.5 * face_normals
    return face_areas, face_normals


# in place
def to_unit(mesh: tuple) -> V:
    vs, _ = mesh
    max_vals = np.max(vs, axis=0)
    min_vals = np.min(vs, axis=0)
    max_range = (max_vals - min_vals).max() / 2
    center = (max_vals + min_vals) / 2
    vs -= center[None, :]
    vs /= max_range
    return mesh


def load_meshes(*files) -> list:
    return [load_mesh(file_name) for file_name in files]


def export_mesh(mesh: tuple, file_name: str) -> None:
    vs, faces = mesh
    if faces is not None:
        faces = faces + 1
    with open(file_name, 'w') as f:
        for vi, v in enumerate(vs):
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        if faces is not None:
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0], faces[face_id][1], faces[face_id][2]))
            f.write("f %d %d %d" % (faces[-1][0], faces[-1][1], faces[-1][2]))


def load_mesh(file_name: str) -> tuple:

    def off_parser():
        header = None

        def parser_(clean_line: list):
            nonlocal header
            if not clean_line:
                return False
            if len(clean_line) == 3 and not header:
                header = True
            elif len(clean_line) == 3:
                return 0, 0, float
            elif len(clean_line) > 3:
                return 1, -int(clean_line[0]), int

        return parser_

    def obj_parser(clean_line: list):
        if not clean_line:
            return False
        elif clean_line[0] == 'v':
            return 0, 1, float
        elif clean_line[0] == 'f':
            return 1, 1, int

    def fetch(lst: list, idx: int, dtype: type):
        if '/' in lst[idx]:
            lst = [item.split('/')[0] for item in lst[idx:]]
            idx = 0
        face_vs_ids = [dtype(c.split('/')[0]) for c in lst[idx:]]
        assert (len(face_vs_ids) == 3)
        return face_vs_ids

    def load_from_txt(parser) -> tuple:
        mesh_ = [[], []]
        with open(file_name, 'r') as f:
            for line in f:
                clean_line = line.strip().split()
                info = parser(clean_line)
                if not info:
                    continue
                mesh_[info[0]].append(fetch(clean_line, info[1], info[2]))

        mesh_ = [V(mesh_[0], dtype=np.float32), V(mesh_[1], dtype=int)]
        if mesh_[1].min() != 0:
            mesh_[1] -= 1
        return tuple(mesh_)
    
    name, extension = os.path.splitext(file_name)
    if extension == '.obj':
        mesh = load_from_txt(obj_parser)
    elif extension == '.off':
        mesh = load_from_txt(off_parser())
    else:
        return None, None
    if not np.logical_and(mesh[1] >= 0, mesh[1] < len(mesh[0])).all():
        assert False
    # assert np.logical_and(mesh[1] >= 0, mesh[1] < len(mesh[0])).all()
    return mesh


def resume_split(mesh, face_areas, mask):
    vs, faces = mesh
    mask = np.logical_not(mask)
    partial_mesh = (vs, faces[mask])
    return partial_mesh, face_areas[mask]


def split_mesh_ne(mesh: tuple, face_ne: V, face_areas: V, total_area: float, should_stay: float):
    vs, faces = mesh
    mask = np.ones(faces.shape[0], np.bool)
    target = total_area * should_stay
    all_faces = np.arange(faces.shape[0])
    queue = deque()
    while 0 < target:
        p = face_areas[mask] / (total_area * (1 - should_stay) + target)
        queue.append(np.random.choice(all_faces[mask], size=1, p=p)[0])
        local_mask = np.ones(faces.shape[0], np.bool)
        while 0 < target and 0 < len(queue):
            cur_face = queue.popleft()
            target -= face_areas[cur_face]
            mask[cur_face] = 0
            for face_id in face_ne[cur_face]:
                if face_id != -1 and mask[face_id] and local_mask[face_id]:
                    local_mask[face_id] = 0
                    queue.append(face_id)
    return resume_split(mesh, face_areas, mask)


def split_mesh_side(mesh: tuple, face_center: V, face_areas: V, total_area: float, should_stay: float):
    vs, faces = mesh
    mask = np.ones(faces.shape[0], np.bool)
    target = total_area * should_stay
    face_center_z = pc_utils.transform_rotation((face_center,))[0][0][:, 1]
    faces_order = np.argsort(face_center_z)
    for face_idx in faces_order:
        mask[face_idx] = 0
        target -= face_areas[face_idx]
        if target <= 0:
            break
    return resume_split(mesh, face_areas, mask)


def compute_faces_centers(mesh):
    vs, faces = mesh
    centers = (vs[faces[:, 0]] + vs[faces[:, 1]] + vs[faces[:, 2]]) / 3
    return centers


def compute_face_ne(mesh):
    vs, faces = mesh
    faces_ne = np.zeros((faces.shape[0], 2), dtype=np.int) - 1
    faces_ne_counter = np.zeros(faces.shape[0], dtype=np.int)
    edges_faces_dict = dict()
    for face_id, face in enumerate(faces):
        for i in range(len(face)):
            cur_edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            if cur_edge in edges_faces_dict:
                ne_face_id = edges_faces_dict[cur_edge]
                faces_ne[face_id, faces_ne_counter[face_id]] = ne_face_id
                faces_ne[ne_face_id, faces_ne_counter[ne_face_id]] = face_id
                faces_ne_counter[ne_face_id] = (faces_ne_counter[ne_face_id] + 1) % 2
                faces_ne_counter[face_id] += (faces_ne_counter[face_id] + 1) % 2
            else:
                edges_faces_dict[cur_edge] = face_id
    return faces_ne


def sample_on_mesh(mesh: tuple, face_areas: V, num_samples: int) -> V:
    vs, faces = mesh
    chosen_faces = faces[np.random.choice(range(len(face_areas)), size=num_samples, p=face_areas / face_areas.sum())]
    u, v = np.random.rand(num_samples, 1), np.random.rand(num_samples, 1)
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    w = 1 - u - v
    samples = u * vs[chosen_faces[:, 0]] + v * vs[chosen_faces[:, 1]] + w * vs[chosen_faces[:, 2]]
    return samples.astype(np.float32)

def is_nan(a):
    return np.isnan(a).sum() > 0
