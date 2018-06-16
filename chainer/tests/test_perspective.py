import unittest

import chainer
import chainer.gradient_check
import chainer.testing
import numpy as np

import neural_renderer


class TestPerspective(unittest.TestCase):
    def test_case1(self):
        v_in = [1, 2, 10]
        v_out = [np.sqrt(3) / 10, 2 * np.sqrt(3) / 10, 10]
        vertices = np.array(v_in, 'float32')
        vertices = vertices[None, None, :]
        transformed = neural_renderer.perspective(vertices)
        chainer.testing.assert_allclose(transformed.data.flatten(), np.array(v_out, 'float32'))


if __name__ == '__main__':
    unittest.main()
