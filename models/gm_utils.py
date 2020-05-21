import torch
import math
from custom_types import *
import torch.distributions as dst
import constants as const

def flatten(x):
    shape = x.shape
    new_shape = [shape[0], shape[1] * shape[2]] + [s for s in shape[3:]]
    return x.view(new_shape)


def gm_loglikelihood_loss(gms: tuple, x: T) -> list:

    losses = []
    for gm in gms:
        if gm is not None:
            mu, p, sigma_det, phi, eigen = list(map(flatten, gm))
            phi = torch.softmax(phi, dim=1)
            eigen_inv = 1 / eigen
            sigma_inverse = torch.matmul(p.transpose(2, 3), p * eigen_inv[:, :, :, None])
            batch_size, num_points, dim = x.shape
            const_1 = phi / torch.sqrt((2 * math.pi) ** dim * sigma_det)
            distance = x[:, None, :, :] - mu[:, :, None, :]
            mahalanobis_distance = -.5 * (distance.matmul(sigma_inverse) * distance).sum(3)
            const_2, _ = mahalanobis_distance.max(1)  # for numeric stability
            mahalanobis_distance -= const_2[:, None, :]
            probs = const_1[:, :, None] * torch.exp(mahalanobis_distance)
            probs = torch.log(probs.sum(1)) + const_2
            logliklihood = probs.sum()
            loss = - logliklihood / (batch_size * num_points)
            losses.append(loss)
    return losses


def compute_plane_penalty(planes: T or None, mu: T, points: T, probs: T,
                          mask: T or int) -> T:
    if planes is None:
        return 0
    points = points[:, None, None, :, :] - mu[:, :, :, None, :]
    b, p, g, n, d = points.shape
    plane_penalty = -torch.einsum('bpgnd,bpgd->bpgn', points, planes) * probs
    plane_penalty = plane_penalty.sum(2) * mask
    plane_penalty = torch.relu(plane_penalty).sum() / (b * n)
    return plane_penalty


def old_hierarchical_gm_log_likelihood_loss(gms: tuple, x: T, ) -> list:
    mask = next_mask = 1
    losses = []
    for idx, gm in enumerate(gms):
        if gm is None:
            continue
        mu, p, sigma_det, phi, eigen = gm
        eigen_inv = 1 / eigen
        sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None])
        batch_size, num_points, dim = x.shape
        if gms[0] is None:
            phi = phi.view(batch_size, -1)
            phi = torch.softmax(phi, dim=1)
            phi = phi.view(batch_size, mu.shape[1], -1)
        else:
            phi = torch.softmax(phi, dim=2)
        const_1 = phi / torch.sqrt((2 * math.pi) ** dim * sigma_det)
        distance = x[:, None, None, :, :] - mu[:, :, :, None, :]
        mahalanobis_distance = -.5 * (distance.matmul(sigma_inverse) * distance).sum(4)
        const_2, _ = mahalanobis_distance.max(dim=2)  # for numeric stability
        mahalanobis_distance -= const_2[:, :, None, :]
        probs = const_1[:, :, :, None] * torch.exp(mahalanobis_distance)
        hard_split = torch.argmax(probs, dim=2)

        if idx < len(gms) - 1:
            next_mask = torch.zeros_like(probs).scatter(2, hard_split.unsqueeze(2), 1)
            if 0 < idx:
                next_mask = next_mask * mask[:, :, None, :]
            next_mask = next_mask.view(batch_size, -1, num_points)

        probs = (torch.log(probs.sum(2)) + const_2)
        probs = probs * mask
        likelihood = probs.sum()
        loss = - likelihood / (batch_size * num_points)
        losses.append(loss)
        mask = next_mask
    return losses


def hierarchical_gm_log_likelihood_loss(gms: tuple, x: T, ) -> list:

    def reshape_param(param):
        return param.view([-1] + list(param.shape[2:]))[parent_idx]

    batch_size, num_points, dim = x.shape
    x = x.view(batch_size * num_points, dim)
    parent_idx = torch.meshgrid([torch.arange(batch_size), torch.arange(num_points)])[0].flatten().to(x.device)
    losses = []

    for idx, gm in enumerate(gms):
        if gm is None:
            continue
        mu, p, sigma_det, phi, eigen = gm
        eigen_inv = 1 / eigen
        sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None])
        num_children = mu.shape[2]

        if gms[0] is None:
            phi = phi.view(batch_size, -1)
            phi = torch.softmax(phi, dim=1)
            phi = phi.view(batch_size, mu.shape[1], -1)
        else:
            phi = torch.softmax(phi, dim=2)
        const_1 = phi / torch.sqrt((2 * math.pi) ** dim * sigma_det)
        mu, sigma_inverse, const_1 = list(map(reshape_param, [mu, sigma_inverse, const_1]))
        distance = x[:, None, :] - mu
        mahalanobis_distance = - .5 * torch.einsum('ngd,ngdc,ngc->ng', distance, sigma_inverse, distance)
        const_2, _ = mahalanobis_distance.max(dim=1)  # for numeric stability
        mahalanobis_distance -= const_2[:, None]
        probs = const_1 * torch.exp(mahalanobis_distance)
        hard_split = torch.argmax(probs, dim=1)
        parent_idx = parent_idx * num_children + hard_split
        probs = torch.log(probs.sum(dim=1)) + const_2
        likelihood = probs.sum()
        loss = - likelihood / (batch_size * num_points)
        losses.append(loss)

    return losses


def gm_sample(gms:tuple, num_samples:int) -> tuple:
    gm = gms[-1]
    mu, inv_sigma, _, phi = list(map(flatten, gm))
    phi = torch.softmax(phi, dim=1)
    sigma = torch.inverse(inv_sigma) #  + (torch.eye(3) * C.EPSILON).to(mu.device)[None, None, :, :]
    b, k, d = mu.shape
    classes = torch.arange(k).to(mu.device)
    samples = []
    splits = []

    def get_model(b_id, j):
        return dst.MultivariateNormal(mu[b_id, j, :], sigma[b_id, j, :, :])

    def sample_batch(b_id):
        vs = []
        splits_ = torch.zeros(1, phi.shape[1] + 1, dtype=torch.int64)
        models = [get_model(b_id, j) for j in range(k)]
        idx = dst.Categorical(phi[b_id]).sample((num_samples,))
        children_num_samples = (idx[None, :] == classes[:, None]).sum(1)
        for j, num in enumerate(children_num_samples):
            splits_[0, j + 1] = splits_[0, j] + num
            vs.append(models[j].sample((num.item(),)))
        return torch.cat(vs, 0).unsqueeze(0), splits_

    #  yachs double for loop
    for batch_id in range(b):
        vs_, splits_ = sample_batch(batch_id)
        samples.append(vs_)
        splits.append(splits_)

    return torch.cat(samples, 0), torch.cat(splits, 0)


def hierarchical_gm_sample(gms: tuple or list, num_samples: int, flatten_sigma=True) -> tuple:
    batch_size = gms[-1][0].shape[0]
    device = gms[-1][0].device
    samples = []
    splits = []

    def bottom_phi():
        if gms[0] is None:
            phi = gms[-1][3].view(batch_size, -1)
            phi = torch.softmax(phi, dim=1)
        else:
            last_phi = torch.ones(batch_size, 1, device=device)
            for gm in gms:
                _, _, _, phi, _ = gm
                # phi = phi * last_phi[:, :, None]
                phi = torch.softmax(phi, dim=2) * last_phi[:, :, None]
                last_phi = phi.view(batch_size, -1)
        return phi.view(batch_size, -1)

    def sample_g(b, j, num_samples_):
        # L = sigma[b, j, :, :].cholesky(upper=True)
        mu_ = mu[b, j, :]
        samples_ = torch.randn(num_samples_, mu_.shape[0], device=device)
        # samples_ = samples_a.mm(L)
        samples_ = samples_.mm(L[b, j])
        return samples_ + mu_[None, :]

    def sample_batch(b_id):
        vs = []
        splits_ = torch.zeros(phi.shape[1] + 1, dtype=torch.int64)
        idx = dst.Categorical(phi[b_id]).sample((num_samples,))
        children_num_samples = (idx[None, :] == classes[:, None]).sum(1)
        for j, num in enumerate(children_num_samples):
            splits_[j + 1] = splits_[j] + num
            if num > 0:
                vs.append(sample_g(b_id, j, num.item()))
        return torch.cat(vs, 0).unsqueeze(0), splits_.unsqueeze(0)

    phi = bottom_phi()
    mu, p, _, _, eigen = gms[-1]
    if flatten_sigma:
        shape = eigen.shape
        min_eigen_indices = eigen.argmin(dim=3).flatten()
        eigen = eigen.view(-1, shape[-1])
        eigen[torch.arange(eigen.shape[0]), min_eigen_indices] = const.EPSILON
        eigen = eigen.view(*shape)

    sigma = torch.matmul(p.transpose(3, 4), p * eigen[:, :, :, :, None])
    mu, sigma = mu.view(batch_size, phi.shape[1], 3), sigma.view(batch_size, -1, 3, 3)
    L = (p * torch.sqrt(eigen[:, :, :, :, None])).view(batch_size, -1, 3, 3)
    classes = torch.arange(phi.shape[1], device=device)

    for b in range(batch_size):
        vs_, splits_ = sample_batch(b)
        samples.append(vs_)
        splits.append(splits_)

    return torch.cat(samples, 0), torch.cat(splits, 0)


def eigen_penalty_loss(gms: list, eigen_penalty: float) -> T:
    eigen = gms[-1][-1]
    if eigen_penalty > 0:
        penalty = eigen.min(3)[0]
        penalty = penalty.sum() / (eigen.shape[0] * eigen.shape[1] * eigen.shape[2])
    else:
        penalty = torch.zeros(0)
    return penalty


if __name__ == '__main__':
    import pickle
    from process_data.files_utils import measure_time
    device = torch.device('cuda:1')
    with open('../temp_ig_.pkl', 'rb') as f:
        gms_ = pickle.load(f)
    x_ = torch.rand(gms_[0][0].shape[0],2000, 3).to(device)
    gms_ = [[param.to(device) for param in gm_ ] for gm_ in gms_]
    measure_time(hierarchical_gm_log_likelihood_loss, 100, gms_, x_)
    measure_time(old_hierarchical_gm_log_likelihood_loss, 100, gms_, x_)

