import chainer
import chainer.functions as cf

import neural_renderer


def look_at(vertices, eye, at=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndim == 3)

    xp = chainer.cuda.get_array_module(vertices)
    batch_size = vertices.shape[0]
    if at is None:
        at = xp.array([0, 0, 0], 'float32')
    if up is None:
        up = xp.array([0, 1, 0], 'float32')

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = xp.array(eye, 'float32')
    if eye.ndim == 1:
        eye = cf.tile(eye[None, :], (batch_size, 1))
    if at.ndim == 1:
        at = cf.tile(at[None, :], (batch_size, 1))
    if up.ndim == 1:
        up = cf.tile(up[None, :], (batch_size, 1))

    # create new axes
    z_axis = cf.normalize(at - eye)
    x_axis = cf.normalize(neural_renderer.cross(up, z_axis))
    y_axis = cf.normalize(neural_renderer.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    r = cf.concat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), axis=1)
    if r.shape[0] != vertices.shape[0]:
        r = cf.broadcast_to(r, (vertices.shape[0], 3, 3))

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = cf.broadcast_to(eye[:, None, :], vertices.shape)
    vertices = vertices - eye
    vertices = cf.matmul(vertices, r, transb=True)

    return vertices
