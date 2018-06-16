import chainer
import cupy as cp
import numpy as np


class Cross(chainer.Function):
    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 2,
            in_types[0].shape[1] == 3,
            in_types[1].dtype.kind == 'f',
            in_types[1].ndim == 2,
            in_types[1].shape[1] == 3,
            in_types[0].shape[0] == in_types[1].shape[0],
        )

    def forward_cpu(self, inputs):
        a, b = inputs
        c = np.cross(a, b)
        return c,

    def forward_gpu(self, inputs):
        a, b = inputs
        c = cp.zeros_like(a, 'float32')
        chainer.cuda.elementwise(
            'int32 j, raw T a, raw T b',
            'raw T c',
            '''
                float* ap = &a[j * 3];
                float* bp = &b[j * 3];
                float* cp = &c[j * 3];
                cp[0] = ap[1] * bp[2] - ap[2] * bp[1];
                cp[1] = ap[2] * bp[0] - ap[0] * bp[2];
                cp[2] = ap[0] * bp[1] - ap[1] * bp[0];
            ''',
            'function',
        )(
            cp.arange(a.size / 3).astype('int32'), a, b, c,
        )
        return c,

    def backward_cpu(self, inputs, gradients):
        a, b = inputs
        gc = gradients[0]
        ga = np.cross(b, gc)
        gb = np.cross(gc, a)
        return ga, gb

    def backward_gpu(self, inputs, gradients):
        a, b = inputs
        gc = gradients[0]
        ga = self.forward_gpu((b, gc))[0]
        gb = self.forward_gpu((gc, a))[0]
        return ga, gb


def cross(a, b):
    return Cross()(a, b)
