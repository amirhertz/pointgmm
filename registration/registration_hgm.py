from registration.registration_algorithm import RegistrationAlgorithm
import options
from process_data import points_utils
import models.model_factory as factory
import abc
from custom_types import *


class RegistrationHgm(RegistrationAlgorithm):

    def __init__(self, opt: options.RegOptions):
        super(RegistrationHgm, self).__init__(opt)
        self.device = torch.device('cpu')
        self.encoder = factory.model_lc(opt.encoder, opt, device=self.device)[0].eval()
        self.input_center = np.zeros((2, 3))

    def to(self, device: torch.device):
        self.device = device
        self.encoder = self.encoder.to(device)
        return super(RegistrationHgm, self).to(device)

    def to_torch(self, *points) -> T:
        points = np.concatenate([np.expand_dims(pts, axis=0) for pts in points], axis=0)
        self.input_center = .5 * (points.max(axis=1) + points.min(axis=1))
        points = points - self.input_center[:, np.newaxis, :]
        points = torch.from_numpy(points).float().to(self.device)
        return points

    @abc.abstractmethod
    def register(self, source: V, target: V):
        raise NotImplementedError

    def register_pre(self, source: V, target: V) -> TS:
        with torch.no_grad():
            encoder_inp = self.to_torch(source, target)
            out = self.encoder(encoder_inp)
        return out


class RegistrationHgmDual(RegistrationHgm):

    def __init__(self, opt: options.RegOptions):
        super(RegistrationHgmDual, self).__init__(opt)

    @property
    def name(self) -> str:
        return "PointGMM"

    @staticmethod
    def transform_source(source: V, theta: V, trnl: V):
        rot = [points_utils.z_axis_rotaion(theta[i]) for i in range(theta.shape[0])]
        source = np.einsum('dc,nd->nc',rot[0], source) - trnl[0][np.newaxis, :]
        source = np.einsum('rd,nd->nr',rot[1], source + trnl[1][np.newaxis, :])
        return source

    def arrange_transformation(self, translate:T, theta: T) -> Tuple[T, T]:
        translate, theta = translate.cpu().numpy(), theta.cpu().numpy()
        theta[0] = -theta[0]
        if theta.shape[1] == 2:
            theta[0][0] = -theta[0][0]
        affine = points_utils.combine_affines((None, self.input_center[1]), (theta[1], None),
                                              (None, translate[1]),
                                              (theta[0], -translate[0]), (None, -self.input_center[0]))
        return points_utils.from_affine(affine)

    def register(self, source: V, target: V) -> Tuple[V, V]:
        z_shape, z_trans, trnl, theta = super(RegistrationHgmDual, self).register_pre(source, target)
        return self.arrange_transformation(trnl, theta)

