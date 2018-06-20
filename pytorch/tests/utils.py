import torch

from .context import neural_renderer


def to_minibatch(data, batch_size=4, target_num=2):
    ret = []
    for d in data:
        device = d.device
        print(torch.unsqueeze(torch.zeros_like(d), 0).shape)
        d2 = torch.unsqueeze(torch.zeros_like(d), 0)
        r = [1 for _ in d2.shape]
        r[0] = batch_size
        d2 = torch.unsqueeze(torch.zeros_like(d), 0).repeat(*r).to(device)
        d2[target_num] = d
        ret.append(d2)
    return ret

def load_teapot_batch(batch_size=4, target_num=2):
    vertices, faces = neural_renderer.load_obj('../../chainer/tests/data/teapot.obj')
    textures = torch.ones((faces.shape[0], 4, 4, 4, 3), dtype=torch.float32)
    vertices, faces, textures = to_minibatch((vertices, faces, textures), batch_size, target_num)
    return vertices, faces, textures
