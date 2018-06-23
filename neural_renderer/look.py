import numpy as np
import torch
import torch.nn.functional as F


def look(vertices, eye, direction=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndimension() == 3)
    device = vertices.device

    if direction is None:
        direction = torch.from_numpy(np.array([0, 1, 0], np.float32)).to(device)
    if direction is None:
        direction = torch.from_numpy(np.array([0, 1, 0], np.float32)).to(device)
    elif isinstance(direction, list) or isinstance(direction, tuple):
        direction = torch.from_numpy(np.array(direction, np.float32)).to(device)
    elif isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).to(device)
    else:
        direction.to(device)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.from_numpy(np.array(eye, np.float32)).to(device)
    # if numpy array convert to tensor
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    else:
        eye = eye.to(device)
    if eye.ndimension() == 1:
        eye = eye[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]
    if up.ndimension() == 1:
        up = up[None, :]

    # create new axes
    z_axis = F.normalize(direction, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1,2))

    return vertices
