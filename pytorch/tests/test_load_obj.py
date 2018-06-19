import unittest

import torch
import numpy as np
from skimage.io import imsave

import neural_renderer


class TestCore(unittest.TestCase):
    # def test_tetrahedron(self):
    #     vertices_ref = np.array(
    #         [
    #             [1., 0., 0.],
    #             [0., 1., 0.],
    #             [0., 0., 1.],
    #             [0., 0., 0.]],
    #         'float32')
    #     faces_ref = np.array(
    #         [
    #             [1, 3, 2],
    #             [3, 1, 0],
    #             [2, 0, 1],
    #             [0, 2, 3]],
    #         'int32')

    #     vertices, faces = neural_renderer.load_obj('../../chainer/tests/data/tetrahedron.obj', False)
    #     assert (np.allclose(vertices_ref, vertices))
    #     assert (np.allclose(faces_ref, faces))
    #     vertices, faces = neural_renderer.load_obj('../../chainer/tests/data/tetrahedron.obj', True)
    #     assert (np.allclose(vertices_ref * 2 - 1.0, vertices))
    #     assert (np.allclose(faces_ref, faces))

    # def test_teapot(self):
    #     vertices, faces = neural_renderer.load_obj('../../chainer/tests/data/teapot.obj')
    #     assert (faces.shape[0] == 2464)
    #     assert (vertices.shape[0] == 1292)

    def test_texture(self):
        renderer = neural_renderer.Renderer()

        # vertices, faces, textures = neural_renderer.load_obj(
        #     '../../chainer/tests/data/1cde62b063e14777c9152a706245d48/model.obj', load_texture=True)

        # renderer.eye = neural_renderer.get_points_from_angles(2, 15, 30)
        # images = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).permute(0,2,3,1).detach().cpu().numpy()
        # imsave('test_out/car.png', images[0])

        vertices, faces, textures = neural_renderer.load_obj(
            '../../chainer/tests/data/4e49873292196f02574b5684eaec43e9/model.obj', load_texture=True, texture_size=16)
        renderer.eye = neural_renderer.get_points_from_angles(2, 15, -90)
        images = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).permute(0,2,3,1).detach().cpu().numpy()
        imsave('test_out/display.png', images[0])


if __name__ == '__main__':
    unittest.main()
