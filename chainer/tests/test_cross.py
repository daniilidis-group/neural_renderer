import unittest

import chainer
import chainer.gradient_check
import chainer.testing
import numpy as np

import neural_renderer


class TestCross(unittest.TestCase):
    def test_forward(self):
        a_cpu = np.random.normal(size=(10, 3)).astype('float32')
        b_cpu = np.random.normal(size=(10, 3)).astype('float32')
        c_ref = np.cross(a_cpu, b_cpu)
        a_gpu = chainer.cuda.to_gpu(a_cpu)
        b_gpu = chainer.cuda.to_gpu(b_cpu)

        c_cpu = neural_renderer.cross(a_cpu, b_cpu).data
        c_gpu = neural_renderer.cross(a_gpu, b_gpu).data

        chainer.testing.assert_allclose(c_ref, c_cpu)
        chainer.testing.assert_allclose(c_ref, chainer.cuda.to_cpu(c_gpu))

    def test_backward_cpu(self):
        a = np.random.normal(size=(10, 3)).astype('float32')
        b = np.random.normal(size=(10, 3)).astype('float32')
        gy = np.random.normal(size=(10, 3)).astype('float32')
        chainer.gradient_check.check_backward(neural_renderer.cross, (a, b), gy, atol=1e-3, rtol=1e-3)

    def test_backward_gpu(self):
        a = chainer.cuda.to_gpu(np.random.normal(size=(10, 3)).astype('float32'))
        b = chainer.cuda.to_gpu(np.random.normal(size=(10, 3)).astype('float32'))
        gy = chainer.cuda.to_gpu(np.random.normal(size=(10, 3)).astype('float32'))
        chainer.gradient_check.check_backward(neural_renderer.cross, (a, b), gy, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
