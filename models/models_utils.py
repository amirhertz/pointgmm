from custom_types import *


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Embe') != -1:
        # nn.init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        nn.init.normal_(m.weight, mean=0, std=1)


class Concatenate(Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class View(Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Transpose(Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Dummy(Module):

    def __init__(self, *args):
        super(Dummy, self).__init__()

    def forward(self, *args):
        return args[0]


class MLP(Module):

    def __init__(self,ch: tuple, norm_class=nn.LayerNorm, dropout=0, skip=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(ch[i], ch[i + 1]))
            if i < len(ch) - 2:
                layers += [
                    norm_class(ch[i + 1]),
                    nn.ReLU(True)
                ]
        self.skip = skip
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.skip:
            out = x + out
        return out


class GMAttend(Module):

    def __init__(self, hidden_dim: int):
        super(GMAttend, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.value_w = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1 / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))

    def forward(self, x):
        queries = self.query_w(x)
        keys = self.key_w(x)
        vals = self.value_w(x)
        attention = self.softmax(torch.einsum('bgqf,bgkf->bgqk', queries, keys))
        out = torch.einsum('bgvf,bgqv->bgqf', vals, attention)
        out = self.gamma * out + x
        return out


def dkl(mu, log_sigma):
    if log_sigma is None:
        return torch.zeros(1).to(mu.device)
    else:
        return 0.5 * torch.sum(torch.exp(log_sigma) - 1 - log_sigma + mu ** 2) / (mu.shape[0] * mu.shape[1])


def recursive_to(item, device):
    if type(item) is T:
        return item.to(device)
    elif type(item) is tuple or type(item) is list:
        return [recursive_to(item[i]) for i in range(len(item))]
    return item

