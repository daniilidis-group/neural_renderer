import os

import numpy as np
import torch

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def to_minibatch(data, batch_size=4, target_num=2):
    ret = []
    for d in data:
        device = d.device
        d2 = torch.unsqueeze(torch.zeros_like(d), 0)
        r = [1 for _ in d2.shape]
        r[0] = batch_size
        d2 = torch.unsqueeze(torch.zeros_like(d), 0).repeat(*r).to(device)
        d2[target_num] = d
        ret.append(d2)
    return ret

def load_teapot_batch(batch_size=4, target_num=2):
    vertices, faces = nr.load_obj(os.path.join(data_dir, 'teapot.obj'))
    textures = torch.ones((faces.shape[0], 4, 4, 4, 3), dtype=torch.float32)
    vertices, faces, textures = to_minibatch((vertices, faces, textures), batch_size, target_num)
    return vertices, faces, textures

def guess_foreground_mask(ref, bg_uv=(0, 0)):
    assert ref.ndim == 3
    bg_pixel = ref[bg_uv].reshape((1, 1, -1))
    return (ref != bg_pixel).any(axis=2)

def calc_mask_iou(a, b):
    assert a.dtype == bool
    assert b.dtype == bool
    assert a.shape == b.shape
    union = np.sum(a | b)
    if union == 0.0:
        return 0.0
    else:
        intersect = np.sum(a & b)
        return intersect / union
