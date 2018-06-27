import torch
import torch.nn as nn


def projection(vertices, P, dist_coeffs, orig_size):
    vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
    # vertices = torch.bmm(vertices, P.transpose(2,1))
    vertices = torch.bmm(vertices, P.transpose(2,1))
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)
    # vertices[:, :, :-1] = (vertices[:, :, :-1] / (vertices[:, :, None, -1] + 1e-5))
    k1, k2, p1, p2, k3 = dist_coeffs
    # we use x_ for x' and x__ for x'' etc.
    # x_, y_ = vertices[:, :, 0], vertices[:, :, 1]
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    x__ = 2 * (x__ - orig_size / 2.) / orig_size
    y__ = 2 * (y__ - orig_size / 2.) / orig_size
    vertices = torch.stack([x__,y__,z], dim=-1)
    return vertices
