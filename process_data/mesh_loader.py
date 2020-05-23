from torch.utils.data import Dataset, DataLoader, Subset
from zipfile import BadZipFile
import os
from process_data import files_utils, mesh_utils, points_utils
import options
from constants import DATASET
from custom_types import *
import json


class MeshDataset(Dataset):

    @property
    def transforms(self):
        return self.opt.transforms

    @property
    def recon(self) -> bool:
        return self.opt.recon

    def cache_path(self, idx) -> str:
        return os.path.join(DATASET, self.opt.tag, f'{self.opt.info}_{idx:04d}.npy')

    def data_path(self, idx) -> str:
        return os.path.join(DATASET, self.opt.tag, f'{self.opt.tag}_{idx:04d}.npz')

    def delete_cache(self):
        if self.cache_length > 0:
            for idx in range(len(self)):
                cache_path = self.cache_path(idx)
                if os.path.isfile(cache_path):
                    os.remove(cache_path)

    def __del__(self):
        self.delete_cache()

    @staticmethod
    def join2root(sub_name) -> str:
        return os.path.join(DATASET, sub_name)

    def get_taxonomy_models_paths(self):
        with open(self.join2root('taxonomy.json'), 'r') as f:
            metadata = json.load(f)
        for info in metadata:
            class_name = info['name'].split(',')[0].replace(' ', '_')
            if class_name == self.opt.tag:
                taxonomy_dir = self.join2root(info['synsetId'])
                if os.path.isdir(taxonomy_dir):
                    return files_utils.collect(taxonomy_dir, '.obj', '.off')

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
            cache_path = self.cache_path(idx)
            if not os.path.isfile(cache_path):
                sub_pc_data = [np.expand_dims(self.sample_sub_points(data), axis=0) for _ in range(self.cache_length)]
                sub_pc_data = np.concatenate(sub_pc_data, axis=0)
                np.save(cache_path[:-4], sub_pc_data)
            else:
                sub_pc_data = np.load(cache_path)
            pc_idx = int(np.random.randint(0, self.cache_length))
            return sub_pc_data[pc_idx]

    def get_transformed_pc(self, idx: int, data: mesh_utils.MeshWrap, base_points: V) -> Tuple[VS, VS]:
        partial = self.get_sub_points(idx, data)
        partial, transforms = points_utils.apply_transforms(self.transforms, base_points, partial)
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
    def first_load(mesh_path: str, data_path: str) -> mesh_utils.MeshWrap:
        vs, faces = mesh_utils.load_mesh(mesh_path)
        hold = vs[:, 1].copy()
        # swapping y and z
        vs[:, 1] = vs[:, 2]
        vs[:, 2] = hold
        mesh = (vs, faces)
        mesh = mesh_utils.to_unit(mesh)
        face_areas, face_normals = mesh_utils.compute_face_areas(mesh)
        face_centers = mesh_utils.compute_faces_centers(mesh)
        data = {'vs': mesh[0], 'faces': mesh[1], 'face_areas': face_areas, 'face_normals': face_normals,
                'total_area': face_areas.sum(), 'face_ne': mesh_utils.compute_face_ne(mesh),
                'face_centers': face_centers}
        np.savez_compressed(data_path, **data)
        return mesh_utils.MeshWrap(data)

    def load_mesh(self, idx: int) -> mesh_utils.MeshWrap:
        if self.all_data[idx] is None:
            requested_att = ['vs', 'faces', 'face_areas', 'face_normals', 'face_ne', 'total_area', 'face_centers']
            base_path = self.data_paths[idx]
            mesh_path = os.path.join(base_path[0], f'{base_path[1]}{base_path[2]}')
            data_path = self.data_path(idx)
            if os.path.isfile(data_path):
                try:
                    data = np.load(data_path)
                    if sum([int(att not in data) for att in requested_att]) == 0:
                        self.all_data[idx] = mesh_utils.MeshWrap(dict(data))
                except BadZipFile:
                    print("BadZipFile")
                    self.all_data[idx] = self.first_load(mesh_path, data_path)
            else:
                self.all_data[idx] = self.first_load(mesh_path, data_path)
        return self.all_data[idx]

    def __init__(self, opt: options.Options, cache_length:int):
        super(MeshDataset, self).__init__()
        self.opt = opt
        files_utils.init_folders(self.data_path(0))
        self.data_paths = self.get_taxonomy_models_paths()
        self.cache_length = cache_length
        self.all_data: List[Union[N, mesh_utils.MeshWrap]] = [None] * len(self)
        self.delete_cache()


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
    dataset = MeshDataset(opt, 20)
    ds_length = len(dataset)
    if 'vae' not in opt.task:
        splits_file = f'{DATASET}/{opt.tag}/{opt.tag}_split'
        if os.path.isfile(splits_file + '.npy'):
            ds_inds = np.load(splits_file + '.npy')
        else:
            ds_inds = np.arange(ds_length)
            np.random.shuffle(ds_inds)
            np.save(splits_file, ds_inds)
        inds = {True: ds_inds[int(0.1 * ds_length):], False: ds_inds[:int(0.1 * ds_length)]}
        dataset = Subset(dataset, inds[train])
    loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1 + (2 * train), shuffle=(train),
                         drop_last=(train))
    print(f"{opt.tag}- {'train' if train else 'test'} dataset length is: {len(dataset)}")
    return loader
