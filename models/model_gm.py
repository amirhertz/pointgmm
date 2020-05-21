from options import Options, RegOptions
import models.models_utils as m_utils
import constants
from custom_types import *


# class ProcessEmbSimple(Module):
#     def __init__(self, emb_dim: int, hidden_dim: int):
#         super(ProcessEmbSimple, self).__init__()
#         self.net_up = nn.Sequential(m_utils.MLP((emb_dim, 2 * hidden_dim,
#                                     4 * hidden_dim, 8 * hidden_dim)),
#                                     m_utils.View(-1, 1, 8, hidden_dim))
#
#     def forward(self, *args):
#         return self.net_up(args[0])


class ProcessEmbSimple(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(ProcessEmbSimple, self).__init__()
        upsample = [
            nn.Linear(emb_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(True),
            m_utils.View(-1, 2, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(True),
            m_utils.View(-1, 4, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            m_utils.View(-1, 1, 8, hidden_dim)
        ]
        self.net_up = nn.Sequential(*upsample)

    def forward(self, *args):
        return self.net_up(args[0])


class GMCast(Module):

    def __init__(self, hidden_dim: int):
        super(GMCast, self).__init__()
        projection_dim = constants.DIM ** 2 + 2 * constants.DIM + 1
        self.mlp = m_utils.MLP((hidden_dim, hidden_dim // 2, projection_dim), dropout=0.1)
        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1])
    
    @staticmethod
    def dot(x, y, dim=3):
        return torch.sum(x * y, dim=dim)

    def remove_projection(self, v_1, v_2):
        proj = (self.dot(v_1, v_2) / self.dot(v_2, v_2))
        return v_1 - proj[:, :, :,None] * v_2

    def forward(self, x, t: None or T):
        if t is not None:
            t = t.unsqueeze(1).unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[2], -1)
            x = torch.cat((x, t), 3)
        x = self.mlp(x)
        splitted = torch.split(x, self.split_shape, dim=3)
        # Gram–Schmidt process
        raw_base = []
        for i in range(constants.DIM):
            u = splitted[i]
            for j in range(i):
                u = self.remove_projection(u, raw_base[j])
            raw_base.append(u)
        p = torch.stack(raw_base, dim=3)
        p = p / torch.norm(p, p=2, dim=4)[:, :, :, :, None]  # + self.noise[None, None, :, :]
        # eigenvalues
        eigen = splitted[constants.DIM] ** 2 + constants.EPSILON
        sigma_det = eigen[:, :, :, 0]
        for i in range(1, constants.DIM):
            sigma_det = sigma_det * eigen[:, :, :, i]
        mu = splitted[constants.DIM + 1]
        phi = splitted[constants.DIM + 2].squeeze(3)
        return mu, p, sigma_det, phi, eigen


class PointGMM(Module):

    def __init__(self, opt: Union[Options, RegOptions]):
        super(PointGMM, self).__init__()
        self.split_cast = opt.split_cast
        self.process_layer = ProcessEmbSimple(opt.dim_z, opt.dim_h)
        t_dim = opt.dim_t if opt.registration else 0
        self.projector = GMCast(opt.dim_h + t_dim)
        if opt.split_cast:
            self.mid_projector = nn.ModuleList([GMCast(opt.dim_h + t_dim) for _ in range(opt.num_splits)])
        else:
            self.mid_projector = (lambda *x: None)if opt.only_last else self.projector
        if opt.attentive:
            self.attention = nn.ModuleList([m_utils.GMAttend(opt.dim_h) for _ in range(opt.num_splits)])
        else:
            self.attention = [m_utils.Dummy() for _ in range(opt.num_splits)]
        self.mlp_split = nn.ModuleList([m_utils.MLP((opt.dim_h, opt.dim_h * 2, opt.dim_h * 4), dropout=0.1)
                                        for _ in range(opt.num_splits)])

    def forward(self, x: T, t: TN = None) -> List[Tuple[T, ...]]:
        gms = []
        raw_gm = self.process_layer(x)
        for i in range(len(self.attention)):
            if self.split_cast:
                gms.append(self.mid_projector[i](raw_gm, t))
            else:
                gms.append(self.mid_projector(raw_gm, t))
            raw_gm = self.gm_split(raw_gm, self.attention[i], self.mlp_split[i])
        gms.append(self.projector(raw_gm, t))
        return gms

    @staticmethod
    def gm_split(x: T, attention: nn.Module, mlp: nn.Module) -> T:
        b_size, grand_parents, parents, dim = x.shape
        x = attention(x)
        out = mlp(x).view(b_size, grand_parents, parents, -1, dim) # + x[:, :, :, None, :]
        return out.view(b_size, grand_parents * parents, -1, dim)

    def interpulate(self, z, res):
        inter = []
        for a in range(res):
            alpha = float(a) / (res - 1)
            z_mid = alpha * z[0] + (1 - alpha) * z[1]
            inter.append(z_mid.unsqueeze(0))
        gms = self(torch.stack(inter, 0))
        return gms
