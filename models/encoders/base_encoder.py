import abc
from options import Options
from custom_types import *


def get_knn(vs: T,  k: int, batch_idx: T) -> T:
    mat_square = torch.matmul(vs, vs.transpose(2,1))
    diag = torch.diagonal(mat_square, dim1=1, dim2=2)
    diag = diag.unsqueeze(2).expand(mat_square.shape)
    dist_mat = (diag + diag.transpose(2, 1) - 2 * mat_square)
    _, index = dist_mat.topk(k + 1, dim=2, largest=False, sorted=True)
    index = index[:, :, 1:].view(-1, k) + batch_idx[:, None] * vs.shape[1]
    return index.flatten()


def extract_angles(vs: T, distance_k: T, vs_k: T) -> TS:
    proj = torch.einsum('nd,nkd->nk', vs, vs_k)
    cos_angles = torch.clamp(proj / distance_k, -1., 1.)
    proj = vs_k - vs[:, None, :] * proj[:, :, None]
    # moving same axis points
    ma = torch.abs(proj).sum(2) == 0
    num_points_to_replace = ma.sum().item()
    if num_points_to_replace:
        proj[ma] = torch.rand(num_points_to_replace, vs.shape[1], device=ma.device)
    proj = proj / torch.norm(proj, p=2, dim=2)[:, :, None]
    angles = torch.acos(cos_angles)
    return angles, proj


def min_angles(dirs: T, up: T) -> T:
    ref = dirs[:, 0]
    all_cos = torch.einsum('nd,nkd->nk', ref, dirs)
    all_sin = torch.cross(ref.unsqueeze(1).expand(-1, dirs.shape[1], -1), dirs, dim=2)
    all_sin = torch.einsum('nd,nkd->nk', up, all_sin)
    all_angles = torch.atan2(all_sin, all_cos)
    all_angles[:, 0] = 0
    all_angles[all_angles < 0] = all_angles[all_angles < 0] + 2 * np.pi
    all_angles, inds = all_angles.sort(dim=1)
    inds = torch.argsort(inds, dim=1)
    all_angles_0 = 2 * np.pi - all_angles[:, -1]
    all_angles[:, 1:] = all_angles[:, 1:] - all_angles[:, :-1]
    all_angles[:, 0] = all_angles_0
    all_angles = torch.gather(all_angles, 1, inds)
    return all_angles


def extract_rotation_invariant_features(k: int) -> Tuple[Callable[[Union[T, V]], T], int]:

    batch_idx  = None
    num_features = k * 3 + 1

    def get_input(xyz: T) -> TS:
        nonlocal batch_idx
        if batch_idx is None or len(batch_idx) != xyz.shape[0] * xyz.shape[1]:
            batch_idx, _ = torch.meshgrid([torch.arange(xyz.shape[0]), torch.arange(xyz.shape[1])])
            batch_idx = batch_idx.flatten().to(xyz.device)
        return xyz.view(-1, 3), batch_idx

    def extract(base_vs: Union[T, V]):
        nonlocal num_features
        with torch.no_grad():
            if type(base_vs) is np.array:
                base_vs = T(base_vs)
            batch_size, num_pts = base_vs.shape[:2]
            vs, batch_idx = get_input(base_vs)
            knn = get_knn(base_vs, k, batch_idx)
            vs_k = vs[knn].view(-1, k, vs.shape[1])
            distance = torch.norm(vs, p=2, dim=1)
            vs_unit = vs / distance[:, None]
            distance_k = distance[knn].view(-1, k)
            angles, proj_unit = extract_angles(vs_unit, distance_k, vs_k)
            proj_min_angle =  min_angles(proj_unit, vs_unit)
            fe = torch.cat([distance.unsqueeze(1), distance_k, angles, proj_min_angle], dim=1)
        return fe.view(batch_size, num_pts, num_features)

    return extract, num_features


class BaseEncoder(Module):

    def __init__(self, opt: Options):
        super(BaseEncoder, self).__init__()
        if opt.k_extract > 0:
            self.extractor, self.inc = extract_rotation_invariant_features(opt.k_extract)
        else:
            self.extractor, self.inc = lambda x: x, 3
        if opt.variational:
            self.embd_mu = nn.Linear(opt.dim_z, opt.dim_z)
            self.embd_log_sigma = nn.Linear(opt.dim_z, opt.dim_z)
        self.variational = opt.variational

    @abc.abstractmethod
    def main(self, x):
        raise NotImplementedError()

    def forward(self, x):
        x = self.extractor(x)
        z = mu = self.main(x)
        log_sigma = None
        if self.variational:
            mu = self.embd_mu(z)
            log_sigma = self.embd_log_sigma(z)
            sigma = torch.exp(log_sigma / 2.)
            # N ~ N(0,1)
            z_size = mu.size()
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).to(mu.device)
            z = mu + sigma * N
        return z, mu, log_sigma

