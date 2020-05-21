import torch
from process_data.files_utils import init_folders
import shutil
from models.encoders.pointnet import PointNet, PointNetDual
from models.model_gm import PointGMM
from custom_types import *
from tqdm import tqdm
import os
import pickle
import options


DEBUG = False


def is_model_clean(model) -> bool:
    for wh in model.parameters():
        if torch.isnan(wh).sum() > 0:
            return False
    return True


def model_lc(model_class_name: str, opt: options.Options, device: D) -> Tuple[Module, Callable[[Module], bool]]:

    def save_model(model):
        nonlocal already_init, last_recover, model_path
        if not already_init:
            init_folders(model_path)
            already_init = True
        if is_model_clean(model):
            recover_path = f'{opt.cp_folder}/{model_class_name}_r{last_recover + 1}.pth'
            if os.path.isfile(recover_path):
                os.remove(recover_path)
            torch.save(model.state_dict(), model_path)
            shutil.copy(model_path, recover_path)
            last_recover = 1 - last_recover
            return True
        else:
           return False
    last_recover = 0
    model_path = f'{opt.cp_folder}/{model_class_name}.pth'
    model = eval(model_class_name)(opt).to(device)
    if os.path.isfile(model_path):
        print(f'loading {model_class_name} model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
        already_init = True
    else:
        print(f'init {model_class_name} model')
        already_init = False
    return model, save_model


def optimizer_lc(args: options.Options, *models, device=torch.device('cpu')):

    def save_optimizer(optimizer_):
        nonlocal already_init, optimizer_path, scheduler

        if not already_init:
            init_folders(optimizer_path)
            already_init = True
        torch.save(optimizer_.state_dict(), optimizer_path)

    def decay_lr():
        scheduler.step()

    already_init = False
    optimizer_path = f'{args.cp_folder}/optimizer.pkl'
    lr = args.lr
    if type(lr) is float:
        lr = [lr] * len(models)
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr[i], 'betas': (0.9, 0.95)} for i, model in enumerate(models)
    ])
    if os.path.isfile(optimizer_path):
        print(f'loading optimizer from {optimizer_path}')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    else:
        print(f'init optimizer')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    return optimizer, save_optimizer, decay_lr


class Logger:

    def __init__(self, opt: options.TrainOptions):
        self.opt = opt
        self.epoch_dictionary = dict()
        self.iter_dictionary = dict()
        self.progress: Union[N, tqdm] = None
        self.epoch = 0

    @staticmethod
    def is_raw_key(key: str) -> bool:
        return '_counter' not in key and '_total' not in key

    def aggregate(self) -> dict:
        aggregate_dictionary = dict()
        for key in self.iter_dictionary:
            if self.is_raw_key(key):
                value = self.iter_dictionary[f"{key}_total"] / float(max(1, self.iter_dictionary[f"{key}_counter"]))
                aggregate_dictionary[key] = value
                if key not in self.epoch_dictionary:
                    self.epoch_dictionary[key] = []
                self.epoch_dictionary[key].append(value)
        return aggregate_dictionary

    @staticmethod
    def stash(dictionary: dict, items: Tuple[Union[str, Union[T, float]], ...]) -> dict:
        for i in range(0, len(items), 2):
            key, item = items[i], items[i + 1]
            if type(item) is T:
                item = item.item()
            if not np.isnan(item):
                if key not in dictionary:
                    dictionary[key] = 0
                    dictionary[f"{key}_counter"] = 0
                    dictionary[f"{key}_total"] = 0
                dictionary[key] = item
                dictionary[f"{key}_total"] += item
                dictionary[f"{key}_counter"] += 1
        return dictionary

    def stash_iter(self, *items: Union[str, Union[T, float]]):
        self.iter_dictionary = self.stash(self.iter_dictionary, items)

    def update_iter(self):
        filtered_dict = {key: item for key, item in self.iter_dictionary.items() if self.is_raw_key(key)}
        self.progress.set_postfix(filtered_dict)
        self.progress.update()

    @property
    def path(self) -> str:
        return f'{self.opt.cp_folder}/log.pkl'

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.epoch_dictionary, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        if os.path.isfile(self.path):
            with open(self.path, 'rb') as f:
                self.epoch_dictionary = pickle.load(f)

    def stop(self):
        aggregate_dictionary = self.aggregate()
        self.progress.set_postfix(aggregate_dictionary)
        self.progress.update()
        self.progress.close()
        self.epoch += 1
        self.iter_dictionary = dict()

    def start(self, loader):
        self.progress = tqdm(total=len(loader), desc=f'{self.opt.info} Epoch: {self.epoch + 1} / {self.opt.epochs}')
