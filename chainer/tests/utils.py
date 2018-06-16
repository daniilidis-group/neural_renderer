import chainer
import numpy as np

import neural_renderer


def to_minibatch(data, batch_size=4, target_num=2):
    ret = []
    for d in data:
        xp = chainer.cuda.get_array_module(d)
        d2 = xp.repeat(xp.expand_dims(xp.zeros_like(d), 0), batch_size, axis=0)
        d2[target_num] = d
        ret.append(d2)
    return ret


def load_teapot_batch(batch_size=4, target_num=2):
    vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
    textures = np.ones((faces.shape[0], 4, 4, 4, 3), 'float32')
    vertices, faces, textures = to_minibatch((vertices, faces, textures), batch_size, target_num)
    vertices = chainer.cuda.to_gpu(vertices)
    faces = chainer.cuda.to_gpu(faces)
    textures = chainer.cuda.to_gpu(textures)
    return vertices, faces, textures
