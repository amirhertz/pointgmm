import os
import pickle
import constants as const
from typing import Union


# sorry, I don't like argparse
class Options:

    def load(self):
        if os.path.isfile(self.save_path):
            print(f'loading opitons from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                options = pickle.load(f)
            return options
        return self

    def save(self):
        if os.path.isdir(self.cp_folder) and not self.already_saved:
            self.already_saved = True
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def make_path(tag:str, encoder:str, det:str) -> tuple:
        info = f'{tag}_{encoder}_{det}'
        cp_folder = f'{const.PROJECT_ROOT}/checkpoints/{info}'
        save_path = f'{cp_folder}/options.pkl'
        return info, cp_folder, save_path

    @property
    def registration(self):
        return False

    @property
    def split_cast(self):
        return not self.only_last

    @property
    def info(self) -> str:
        return f'{self.tag}_{self.task}'

    @property
    def cp_folder(self):
        return f'{const.PROJECT_ROOT}/checkpoints/{self.info}'

    @property
    def recon(self):
        return self.partial_range[0] < 1

    @property
    def save_path(self):
        return f'{const.PROJECT_ROOT}/checkpoints/{self.info}/options.pkl'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        self.dim_z = 256
        self.dim_h = 512
        self.num_splits = 3
        self.batch_size = 82
        self.tag = 'chair'
        self.encoder = 'PointNet'
        self.decoder = 'PointGMM'
        self.task = 'vae'
        self.variational = True
        self.flatten_sigma = False
        self.k_extract = 0
        self.only_last = False
        self.transforms = ()
        self.already_saved = False
        self.to_unit = True
        self.partial_range = (1, 1)
        self.partial_samples = (2048, 1024)
        self.attentive = True
        self.fill_args(kwargs)


class TrainOptions(Options):

    def __init__(self, **kwargs):
        super(TrainOptions, self).__init__()
        self.epochs = 1200
        self.save_every_epochs = 100
        self.plot_every_epochs = 100
        self.decay_every = 100
        self.decay = 0.98
        self.lr = 0.001
        self.lr_decay = .5
        self.lr_decay_every_epochs = 200
        self.gamma = 1.
        self.penalty_gamma = 10
        self.fill_args(kwargs)



class RegOptions(TrainOptions):
    def __init__(self, **kwargs):
        super(RegOptions, self).__init__()
        self.dim_t = 128
        self.trans_gamma = 50
        self.rot_gamma = 10
        self.k_extract = 40
        self.task = 'reg'
        self.encoder = 'PointNetDual'
        self.variational = False
        self.baseline = False
        self.transforms = (('rotate', [True]), ('translate_to_center', [1]))
        self.trans_layers = 1
        self.partial_range = (.3, .8)
        self.fill_args(kwargs)

    @property
    def registration(self):
        return True


def apply_decay(gamma, decay):
    return gamma * decay


def do_when_its_time(when, do, now, *with_what, default_return=None):
    if (now + 1) % when == 0:
        return do(*with_what)
    else:
        return default_return
