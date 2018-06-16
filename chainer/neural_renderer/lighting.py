import chainer
import chainer.functions as cf
import chainer.functions.math.basic_math as cfmath

import neural_renderer


def lighting(
        faces, textures, intensity_ambient=0.5, intensity_directional=0.5, color_ambient=(1, 1, 1),
        color_directional=(1, 1, 1), direction=(0, 1, 0)):
    xp = chainer.cuda.get_array_module(faces)
    bs, nf = faces.shape[:2]

    # arguments
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = xp.array(color_ambient, 'float32')
    if isinstance(color_directional, tuple) or isinstance(color_directional, list):
        color_directional = xp.array(color_directional, 'float32')
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = xp.array(direction, 'float32')
    if color_ambient.ndim == 1:
        color_ambient = cf.broadcast_to(color_ambient[None, :], (bs, 3))
    if color_directional.ndim == 1:
        color_directional = cf.broadcast_to(color_directional[None, :], (bs, 3))
    if direction.ndim == 1:
        direction = cf.broadcast_to(direction[None, :], (bs, 3))

    # create light
    light = xp.zeros((bs, nf, 3), 'float32')

    # ambient light
    if intensity_ambient != 0:
        light = light + intensity_ambient * cf.broadcast_to(color_ambient[:, None, :], light.shape)

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        normals = cf.normalize(neural_renderer.cross(v10, v12))
        normals = normals.reshape((bs, nf, 3))

        if direction.ndim == 2:
            direction = cf.broadcast_to(direction[:, None, :], normals.shape)
        cos = cf.relu(cf.sum(normals * direction, axis=2))
        light = (
            light + intensity_directional * cfmath.mul(*cf.broadcast(color_directional[:, None, :], cos[:, :, None])))

    # apply
    light = cf.broadcast_to(light[:, :, None, None, None, :], textures.shape)
    textures = textures * light
    return textures
