import unittest

import chainer
import chainer.functions as cf
import chainer.gradient_check
import chainer.testing
import cupy as cp
import numpy as np
import scipy.misc

import neural_renderer
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
        images = images.data.get()
        image = images[2]
        image = image != image.max()

        # load reference image by blender
        ref = scipy.misc.imread('./tests/data/teapot_blender.png')
        ref = ref.astype('float32')
        ref = (ref.min(-1) != 255).astype('float32')

        chainer.testing.assert_allclose(ref, image)

    def test_forward_case2(self):
        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer.render_depth(vertices, faces)
        images = images.data.get()
        image = images[2]
        image[image == image.max()] = image.min()
        image = (image - image.min()) / (image.max() - image.min())

        ref = scipy.misc.imread('./tests/data/test_depth.png')
        ref = ref.astype('float32') / 255.
        scipy.misc.toimage(image).save('../tmp/test_depth.png')

        chainer.testing.assert_allclose(image, ref, atol=1e-2)

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

        vertices = cp.array(vertices, 'float32')
        faces = cp.array(faces, 'int32')
        vertices, faces = utils.to_minibatch((vertices, faces))
        vertices = chainer.Variable(vertices)

        images = renderer.render_depth(vertices, faces)
        loss = cf.sum(cf.square(images[0, 15, 20] - 1))
        loss.backward()
        grad = vertices.grad.get()
        grad2 = np.zeros_like(grad)

        for i in range(3):
            for j in range(3):
                eps = 1e-3
                vertices2 = vertices.data.copy()
                vertices2[i, j] += eps
                images = renderer.render_depth(vertices2, faces)
                loss2 = cf.sum(cf.square(images[0, 15, 20] - 1))
                grad2[i, j] = ((loss2 - loss) / eps).data.get()

        chainer.testing.assert_allclose(grad, grad2, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
