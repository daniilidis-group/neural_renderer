import unittest

import torch
import numpy as np
from skimage.io import imread, imsave

from context import neural_renderer
import utils


class TestRasterizeDepth(unittest.TestCase):
    def test_forward_case1(self):
        """Whether a silhouette by neural renderer matches that by Blender."""

        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer.render_depth(vertices, faces)
        images = images.detach().cpu().numpy()
        image = images[2]
        image = image != image.max()

        # load reference image by blender
        ref = imread('../../chainer/tests/data/teapot_blender.png')
        ref = (ref.min(axis=-1) != 255).astype('float32')

        assert(np.allclose(ref, image))

    def test_forward_case2(self):
        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer.render_depth(vertices, faces)
        images = images.detach().cpu().numpy()
        image = images[2]
        image[image == image.max()] = image.min()
        image = (image - image.min()) / (image.max() - image.min())

        ref = imread('../../chainer/tests/data/test_depth.png').astype('float32') / 255.
        imsave('test_depth.png', image)

        assert(np.allclose(image, ref, atol=1e-2))

    def test_backward_case1(self):
        vertices = [
            [-0.9, -0.9, 2.],
            [-0.8, 0.8, 1.],
            [0.8, 0.8, 0.5]]
        faces = [[0, 1, 2]]

        renderer = neural_renderer.Renderer()
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False
        renderer.camera_mode = 'none'

        vertices = torch.from_numpy(np.array(vertices, 'float32')).cuda()
        faces = torch.from_numpy(np.array(faces, 'int32')).cuda()
        vertices, faces = utils.to_minibatch((vertices, faces))
        vertices.requires_grad = True

        images = renderer.render_depth(vertices, faces)
        loss = torch.sum((images[0, 15, 20] - 1)**2)
        loss.backward()
        grad = vertices.grad.clone()
        grad2 = np.zeros_like(grad)

        for i in range(3):
            for j in range(3):
                eps = 1e-3
                vertices2 = vertices.clone()
                vertices2[i, j] += eps
                images = renderer.render_depth(vertices2, faces)
                loss2 = torch.sum((images[0, 15, 20] - 1)**2)
                grad2[i, j] = ((loss2 - loss) / eps).item()

        assert(np.allclose(grad, grad2, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
