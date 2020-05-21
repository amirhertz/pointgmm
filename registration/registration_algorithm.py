import abc
import options
from custom_types import *


class RegistrationAlgorithm(abc.ABC):

    def __init__(self, opt: options.RegOptions):
        self.opt = opt

    def __call__(self, source: V, target: V) -> VS:
        return self.register(source, target)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def register(self, source: V, target: V) -> VS:
        raise NotImplementedError()

    @staticmethod
    def decompose_affine(transform: V) -> VS:
        rot = transform[:3, :3]
        trnsl = transform[:3, 3]
        return rot, trnsl

    def skip(self):
        return

    def to(self, device: D):
        return self
