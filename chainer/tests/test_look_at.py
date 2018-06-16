import unittest

import chainer
import chainer.gradient_check
import chainer.testing
import numpy as np

import neural_renderer


class TestLookAt(unittest.TestCase):
    def test_case1(self):
        eyes = [
            [1, 0, 1],
            [0, 0, -10],
            [-1, 1, 0],
        ]
        answers = [
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [1, 0, 10],
            [0, np.sqrt(2) / 2, 3. / 2. * np.sqrt(2)],
        ]
        vertices = np.array([1, 0, 0], 'float32')
        vertices = vertices[None, None, :]
        for e, a in zip(eyes, answers):
            eye = np.array(e, 'float32')
            transformed = neural_renderer.look_at(vertices, eye)
            chainer.testing.assert_allclose(transformed.data.flatten(), np.array(a))


if __name__ == '__main__':
    unittest.main()
