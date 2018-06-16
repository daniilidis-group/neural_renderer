import chainer
import chainer.functions as cf

import neural_renderer


def look(vertices, eye, direction=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndim == 3)

    xp = chainer.cuda.get_array_module(vertices)
    if direction is None:
        direction = xp.array([0, 0, 1], 'float32')
    if up is None:
        up = xp.array([0, 1, 0], 'float32')

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = xp.array(eye, 'float32')
    if eye.ndim == 1:
        eye = eye[None, :]
    if direction.ndim == 1:
        direction = direction[None, :]
    if up.ndim == 1:
        up = up[None, :]

    # create new axes
    z_axis = cf.normalize(direction)
    x_axis = cf.normalize(neural_renderer.cross(up, z_axis))
    y_axis = cf.normalize(neural_renderer.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    r = cf.concat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), axis=1)
    if r.shape[0] != vertices.shape[0]:
        r = cf.broadcast_to(r, vertices.shape)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = cf.broadcast_to(eye[:, None, :], vertices.shape)
    vertices = vertices - eye
    vertices = cf.matmul(vertices, r, transb=True)

    return vertices
