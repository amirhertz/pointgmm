from torch.utils.data import Dataset, DataLoader, Subset
from zipfile import BadZipFile
import os
import process_data.points_utils as pc_utils
import process_data.mesh_utils as mesh_utils
import process_data.files_utils as files_utils
import options
from constants import DATASET
from custom_types import *

V = np.array

dummy_pt = V([[1., 2., 3.]])


def empty_requirements(_):
    return dummy_pt


def pointnet_requirements(data: mesh_utils.MeshWrap, num_samples: int, should_stay: tuple) -> tuple:
    p = should_stay[0] + np.random.random() * (should_stay[1] - should_stay[0])
    sub_mesh, sub_areas = mesh_utils.split_mesh_side(data.mesh(), data['face_centers'], data['face_areas'],
                                                     data['total_area'], p)
    points = mesh_utils.sample_on_mesh(sub_mesh, sub_areas, num_samples)
    return points


class MeshDataset(Dataset):

    @property
    def transforms(self):
        return self.opt.transforms


    @property
    def recon(self) -> bool:
        return self.opt.recon

    def delete_cache(self):
        if self.cache_length > 0:
            for i in range(len(self)):
                post_path = os.path.join(self.data_paths[i][0], f'{self.opt.info}_{i:05d}.npy')
                if os.path.isfile(post_path):
                    os.remove(post_path)

    def __del__(self):
        self.delete_cache()

    def sample_sub_points(self, data: mesh_utils.MeshWrap):
        p = self.opt.partial_range[0] + np.random.random() * (self.opt.partial_range[1] - self.opt.partial_range[0])
        sub_mesh, sub_areas = mesh_utils.split_mesh_side(data.mesh(), data['face_centers'], data['face_areas'],
                                                         data['total_area'], p)
        sub_points = mesh_utils.sample_on_mesh(sub_mesh, sub_areas, self.opt.partial_samples[1])
        return sub_points

    def get_sub_points(self, idx: int, data: mesh_utils.MeshWrap) -> V:
        if self.cache_length < 1:
            return self.sample_sub_points(data)
        else:
            post_path = os.path.join(self.data_paths[idx][0], f'{self.opt.info}_{idx:05d}')
            if not os.path.isfile(post_path + '.npy'):
                sub_pc_data = [np.expand_dims(self.sample_sub_points(data), axis=0) for _ in range(self.cache_length)]
                sub_pc_data = np.concatenate(sub_pc_data, axis=0)
                np.save(post_path, sub_pc_data)
            else:
                sub_pc_data = np.load(post_path + '.npy')
            pc_idx = int(np.random.randint(0, self.cache_length))
            return sub_pc_data[pc_idx]

            # if self.sub_data[idx] is None:
            #     sub_pc_data = [np.expand_dims(self.sample_sub_points(data), axis=0) for _ in range(self.cache_length)]
            #     self.sub_data[idx]  = np.concatenate(sub_pc_data, axis=0)
            # pc_idx = int(np.random.randint(0, self.cache_length))
            # return self.sub_data[idx][pc_idx]


    def get_transformed_pc(self, idx: int, data: mesh_utils.MeshWrap, base_points: V) -> Tuple[VS, VS]:
        partial = self.get_sub_points(idx, data)
        partial, transforms = pc_utils.apply_transforms(self.transforms, base_points, partial)
        return partial, transforms

    def __getitem__(self, idx):
        data = self.load_mesh(idx)
        mesh, face_areas = data.mesh(), data['face_areas']
        points = mesh_utils.sample_on_mesh(mesh, face_areas, self.opt.partial_samples[0])
        if self.recon or len(self.transforms) > 0:
            pc_trans, transforms = self.get_transformed_pc(idx, data, points)
            return [points] + [pc.astype(np.float32) for pc in pc_trans] + [tr.astype(np.float32) for tr in transforms]
        else:
            return points

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def first_load(mesh_path: str, post_path: str) -> mesh_utils.MeshWrap:
        mesh = mesh_utils.load_mesh(mesh_path)
        mesh = mesh_utils.to_unit(mesh)
        face_areas, face_normals = mesh_utils.compute_face_areas(mesh)
        face_centers = mesh_utils.compute_faces_centers(mesh)
        data = {'vs': mesh[0], 'faces': mesh[1], 'face_areas': face_areas, 'face_normals': face_normals,
                'total_area': face_areas.sum(), 'face_ne': mesh_utils.compute_face_ne(mesh),
                'face_centers': face_centers}
        np.savez_compressed(post_path, **data)
        return mesh_utils.MeshWrap(data)

    def load_mesh(self, idx: int) -> mesh_utils.MeshWrap:
        if self.all_data[idx] is None:
            requested_att = ['vs', 'faces', 'face_areas', 'face_normals', 'face_ne', 'total_area', 'face_centers']
            base_path = self.data_paths[idx]
            mesh_path = os.path.join(base_path[0], f'{base_path[1]}{base_path[2]}')
            post_path = os.path.join(base_path[0], f'{self.opt.tag}_{base_path[1]}.npz')
            if os.path.isfile(post_path):
                try:
                    data = np.load(post_path)
                    if sum([int(att not in data) for att in requested_att]) == 0:
                        self.all_data[idx] = mesh_utils.MeshWrap(dict(data))
                except BadZipFile:
                    print("BadZipFile")
                    self.all_data[idx] = self.first_load(mesh_path, post_path)
            else:
                self.all_data[idx] = self.first_load(mesh_path, post_path)
        return self.all_data[idx]

    def __init__(self, opt: options.Options, cache_length:int):
        super(MeshDataset, self).__init__()
        self.opt = opt
        self.data_paths = files_utils.collect(os.path.join(DATASET, opt.tag), '.obj', '.off')
        self.cache_length = cache_length
        self.all_data: List[Union[N, mesh_utils.MeshWrap]] = [None] * len(self)
        self.delete_cache()
        # self.sub_data: List[Union[N, V]] = [None] * len(self)


class AnotherLoaderWrap:

    def __init__(self, base_loader, batch_size):
        self.base_loader = base_loader
        self.batch_size = batch_size
        self.choices = np.arange(len(self.base_loader.dataset))
        self.wrap_iter, self.counter = self.init_iter()

    def __iter__(self):
        return self.base_loader.__iter__()

    def init_iter(self):
        return self.__iter__(), len(self.base_loader.dataset)

    def __next__(self):
        if self.counter < 0:
            self.wrap_iter, self.counter = self.init_iter()
        self.counter = self.counter - self.batch_size
        return next(self.wrap_iter)

    def get_random_batch(self):
        indices = np.random.choice(self.choices, self.batch_size, replace=False)
        batch = [self.base_loader.dataset[idx] for idx in indices]
        return indices, self.base_loader.collate_fn(batch)

    def get_by_ids(self, *indices):
        batch = [self.base_loader.dataset[idx] for idx in indices]
        return self.base_loader.collate_fn(batch)

    def __getitem__(self, idx):
        return self.base_loader.dataset[idx]

    def __len__(self):
        return len(self.base_loader.dataset)


def get_loader(opt: options.Options, train=True) -> DataLoader:
    base_ds = MeshDataset(opt, 20)
    ds_length = len(base_ds)
    splits_file = f'{DATASET}/{opt.tag}/{opt.tag}_split'
    if os.path.isfile(splits_file + '.npy'):
        ds_inds = np.load(splits_file + '.npy')
    else:
        ds_inds = np.arange(ds_length)
        np.random.shuffle(ds_inds)
        np.save(splits_file, ds_inds)
    inds = {True: ds_inds[int(0.1 * ds_length):], False: ds_inds[:int(0.1 * ds_length)]}
    dataset = Subset(base_ds, inds[train])
    loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1 + (2 * train), shuffle=(train),
                         drop_last=(train))
    print(f"{opt.tag}- {'train' if train else 'test'} dataset length is: {len(dataset)}")
    return loader
