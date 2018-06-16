import unittest

import numpy as np

import neural_renderer


class TestLighting(unittest.TestCase):
    def test_case1(self):
        """Test whether it is executable."""
        faces = np.random.normal(size=(64, 16, 3, 3)).astype('float32')
        textures = np.random.normal(size=(64, 16, 8, 8, 8, 3)).astype('float32')
        neural_renderer.lighting(faces, textures)


if __name__ == '__main__':
    unittest.main()
