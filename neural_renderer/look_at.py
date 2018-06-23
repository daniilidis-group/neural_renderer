import numpy as np
import torch
import torch.nn.functional as F


def look_at(vertices, eye, at=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndimension() == 3)
    device = vertices.device

    batch_size = vertices.shape[0]
    if at is None:
        at = torch.from_numpy(np.array([0, 0, 0], np.float32)).to(device)
    elif isinstance(at, list) or isinstance(at, tuple):
        at = torch.from_numpy(np.array(at, np.float32)).to(device)
    elif isinstance(at, np.ndarray):
        at = torch.from_numpy(at).to(device)
    else:
        at.to(device)

    if up is None:
        up = torch.from_numpy(np.array([0, 1, 0], np.float32)).to(device)
    elif isinstance(up, list) or isinstance(up, tuple):
        up = torch.from_numpy(np.array(up, np.float32)).to(device)
    elif isinstance(up, np.ndarray):
        up = torch.from_numpy(up).to(device)
    else:
        up.to(device)

    # if list or tuple convert to numpy array
    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.from_numpy(np.array(eye, np.float32)).to(device)
    # if numpy array convert to tensor
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    else:
        eye = eye.to(device)

    if eye.ndimension() == 1:
        eye = eye[None, :].repeat(batch_size, 1)
    if at.ndimension() == 1:
        at = at[None, :].repeat(batch_size, 1)
    if up.ndimension() == 1:
        up = up[None, :].repeat(batch_size, 1)

    # create new axes
    # eps is chosen as 0.5 to match the chainer version
    z_axis = F.normalize(at - eye, eps=1e-5)
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
