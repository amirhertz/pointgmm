import sys
import abc
import time
from process_data.mesh_loader import get_loader, AnotherLoaderWrap
import models.model_factory as factory
import models.gm_utils as gm_utils
from models import models_utils
from show.view_utils import view
import options
from custom_types import *


class Trainer(abc.ABC):

    @abc.abstractmethod
    def train_iter(self, data):
        raise NotImplemented

    def set_hgm_loss(self, loss: TS, prefix='loss'):
        for i in range(len(loss)):
            self.logger.stash_iter(f'{prefix}{8 * 4 ** i}', loss[i])

    def train_epoch(self):
        self.logger.start(self.dl_train)
        for idx, data in enumerate(self.dl_train):
            self.train_iter(data)
            self.logger.update_iter()
        self.logger.stop()


    def between_epochs(self, epoch):
        options.do_when_its_time(self.opt.save_every_epochs, self.save, epoch)
        options.do_when_its_time(self.opt.save_every_epochs, self.dl_train.dataset.dataset.delete_cache, epoch)
        options.do_when_its_time(self.opt.lr_decay_every_epochs, self.decay_optim, epoch)
        self.opt.penalty_gamma = options.do_when_its_time(self.opt.decay_every, options.apply_decay, epoch,
                                                          self.opt.penalty_gamma, self.opt.decay,
                                                          default_return=self.opt.penalty_gamma)

    def train(self):
        start_time = time.time()
        for epoch in range(self.opt.epochs):
            self.train_epoch()
            self.between_epochs(epoch)

    def save(self):
        if self.save_models:
            self.save_encoder(self.encoder)
            self.save_decoder(self.decoder)
            self.save_optim(self.optim)
        self.opt.save()
        self.logger.save()

    def __init__(self, opt: Union[options.TrainOptions, options.RegOptions], device: D):
        self.opt = opt
        self.device = device
        self.encoder, self.save_encoder = factory.model_lc(opt.encoder, opt, device=device)
        self.decoder, self.save_decoder = factory.model_lc(opt.decoder, opt, device=device)
        self.optim, self.save_optim, self.decay_optim = factory.optimizer_lc(opt, self.encoder, self.decoder, device=device)
        self.dl_train = get_loader(opt)
        self.start_time: Union[N, float] = None
        self.logger = factory.Logger(opt)
        self.save_models = True


class VaeTrainer(Trainer):

    def before_plot(self, data, eval_size=-1):
        y, encoder_inp = self.arrange_data(data, eval_size)
        z, _, _ = self.encoder(encoder_inp)
        gms = self.decoder(z)
        vs, splits = gm_utils.hierarchical_gm_sample(gms, self.opt.partial_samples[0], self.opt.flatten_sigma)
        # splits_inp = np.array([0, encoder_inp.shape[1]])
        # splits_y = np.array([0, y[1]])
        # transform back
        if self.opt.recon or len(self.opt.transforms) > 0:
            transforms = data[3:]
            for i in range(len(transforms)):
                transform = transforms[-(i + 1)][:vs.shape[0]].to(self.device)
                if transform.dim() == 2:
                    t = lambda x: x - transform[:, None, :]
                else:
                    t = lambda x: torch.einsum('bnd,brd->bnr', x, transform)
                vs, vs_in, y = list(map(t, [vs, encoder_inp, y]))
        out = list(map(lambda x: x.data.cpu().numpy(), [vs, encoder_inp, y, splits]))
        return out

    def plot(self, epoch):
        self.decoder.eval(), self.encoder.eval()
        with torch.no_grad():
            data = next(self.dl_plot)
            eval_size = min(4, data[0].shape[0])
            vs, vs_in, y, splits = self.before_plot(data, eval_size)
            base_split = V([0, self.opt.partial_samples[0]], dtype=np.int)
            for i in range(len(vs)):
                view([y[i], vs[i]], splits=[base_split, splits[i]],
                     save_path=f'{self.opt.cp_folder}/{epoch: 03d}_{i}.png')
        self.decoder.train(), self.encoder.train()

    def arrange_data(self, data, batch_size_=-1):
        if type(data) is T:
            y = data.to(self.device)
            encoder_inp = y
        else:
            data = list(map(lambda x: x.to(self.device) if type(x) is T else x, data))
            y = data[1]
            encoder_inp = data[2]
        if batch_size_ > 0:
            batch_size_ = min(batch_size_, y.shape[0])
            y = y[:batch_size_]
            encoder_inp = encoder_inp[:batch_size_]
        return y, encoder_inp

    def train_iter(self, data):
        y, encoder_inp = self.arrange_data(data)
        z, mu, log_sigma = self.encoder(encoder_inp)
        gms = self.decoder(z)
        self.optim.zero_grad()
        losses = self.criterion(gms, y)
        loss_variational = models_utils.dkl(mu, log_sigma)
        penal = gm_utils.eigen_penalty_loss(gms, self.opt.penalty_gamma)
        loss_ = sum(losses) + self.opt.penalty_gamma * penal + self.opt.gamma * loss_variational
        loss_.backward()
        self.optim.step()
        self.set_hgm_loss(losses)
        self.logger.stash_iter('penal', penal, 'dkl', loss_variational)

    def between_epochs(self, epoch):
        super(VaeTrainer, self).between_epochs(epoch)
        options.do_when_its_time(self.opt.plot_every_epochs, self.plot, epoch, epoch)
        self.opt.gamma = options.do_when_its_time(self.opt.decay_every, options.apply_decay, epoch, self.opt.gamma,
                                                  self.opt.decay, default_return=self.opt.gamma)

    def __init__(self, opt: options.TrainOptions, device: D):
        super(VaeTrainer, self).__init__(opt, device)
        if opt.only_last:
            self.criterion = gm_utils.gm_loglikelihood_loss
        else:
            self.criterion = gm_utils.hierarchical_gm_log_likelihood_loss
        self.dl_plot = AnotherLoaderWrap(get_loader(opt, False), opt.batch_size)


class RegistrationTrainer(Trainer):

    @property
    def using_pointgmm(self):
        return not self.opt.baseline and self.opt.decoder == 'PointGMM'

    def arrange_data(self, data: TS, batch_size: int =-1) -> TS:
        data = list(map(lambda x: x.to(self.device) if type(x) is T else x, data))
        if batch_size > 0:
            batch_size = min(batch_size, data[0].shape[0])
            data = list(map(lambda x: x[: batch_size] if type(x) is T else x, data))
        y_a, y_b, encoder_inp, r, theta, _, trnl = data
        if self.zeros.shape[0] != y_a.shape[0]:
            self.zeros = torch.zeros(y_a.shape[0], self.opt.dim_t, device=self.device).detach()
        theta = 2 * np.pi * theta
        theta = torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
        return y_a, y_b, encoder_inp, r, theta, trnl

    def train_iter(self, data):
        y_a, y_b, encoder_inp, r, theta_real, trnl_real = self.arrange_data(data)
        z_shape, z_trans, trnl, theta = self.encoder(encoder_inp)
        loss_trnl = self.criterion_trnl(trnl, trnl_real)
        loss_rot = self.criterion_angle(theta, theta_real)
        loss = self.opt.trans_gamma * loss_trnl + self.opt.rot_gamma * loss_rot
        if not self.opt.baseline:
            gms_a = self.decoder(z_shape, self.zeros)
            gms_b = self.decoder(z_shape, z_trans)
            losses_gms_a = self.criterion_hgm(gms_a, y_a)
            losses_gms_b = self.criterion_hgm(gms_b, y_b)
            self.logger.stash_iter(f'loss_a', losses_gms_a[-1],
                                   f'loss_b', losses_gms_b[-1])
            loss += sum(losses_gms_a) + sum(losses_gms_b)
        self.logger.stash_iter('trans', loss_trnl, 'rot', loss_rot)
        self.optim.zero_grad()
        if torch.isnan(loss):
            print('error')
            return None
        loss.backward()
        self.optim.step()

    def __init__(self, opt: options.RegOptions, device:D):
        super(RegistrationTrainer, self).__init__(opt, device)
        if self.opt.baseline:
            self.save_decoder = lambda x: True
            self.optim, self.save_optim, self.decay_optim = factory.optimizer_lc(opt, self.encoder, device=device)
        self.criterion_angle = lambda x, y: - torch.einsum('bd,bd', x, y) / x.shape[0]
        self.criterion_trnl = torch.nn.L1Loss()
        self.criterion_hgm = gm_utils.hierarchical_gm_log_likelihood_loss
        self.zeros = torch.zeros(self.opt.batch_size, self.opt.dim_t, device=device).detach()


# sorry, I don't like argparse
def get_args(args: List[str]) -> Tuple[D, str, bool]:
    train_type = '-r' in args
    device_id = int(args[args.index('-d') + 1]) if '-d' in args else -1
    cls = args[args.index('-c') + 1] if '-c' in args else 'chair'
    device = CUDA(device_id) if  device_id >=0 else (CUDA(0) if torch.cuda.is_available() else CPU)
    return device, cls, train_type


def main():
    device, cls, train_type = get_args(sys.argv[1:])
    if train_type:
        trainer = RegistrationTrainer(options.RegOptions(tag=cls).load(), device)
    else:
        trainer = VaeTrainer(options.TrainOptions(tag=cls).load(), device)
    trainer.train()


if __name__ == '__main__':
    main()
