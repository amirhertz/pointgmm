import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Union, Callable, Type, Iterator
import torch.optim.optimizer


N = type(None)
V = np.array
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Union[T, N]
TNS = Union[TS, N]
D = torch.device
CPU = torch.device('cpu')
CUDA = lambda device_idx: torch.device(f'cuda:{device_idx}')

Optimizer = torch.optim.Adam
Module = nn.Module