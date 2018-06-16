import chainer
import chainer.functions as cf


def perspective(vertices, angle=30.):
    assert (vertices.ndim == 3)
    xp = chainer.cuda.get_array_module(vertices)
    if isinstance(angle, float) or isinstance(angle, int):
        angle = chainer.Variable(xp.array(angle, 'float32'))
    angle = angle / 180. * 3.1416
    angle = cf.broadcast_to(angle[None], (vertices.shape[0],))

    width = cf.tan(angle)
    width = cf.broadcast_to(width[:, None], vertices.shape[:2])
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = cf.concat((x[:, :, None], y[:, :, None], z[:, :, None]), axis=2)
    return vertices
