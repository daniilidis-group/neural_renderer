import unittest

import chainer
import chainer.functions as cf
import chainer.gradient_check
import chainer.testing
import cupy as cp
import scipy.misc

import neural_renderer
import utils


class TestRasterizeSilhouettes(unittest.TestCase):
    def test_case1(self):
        """Whether a silhouette by neural renderer matches that by Blender."""

        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer.render_silhouettes(vertices, faces)
        images = images.data.get()
        image = images[2]

        # load reference image by blender
        ref = scipy.misc.imread('./tests/data/teapot_blender.png')
        ref = ref.astype('float32')
        ref = (ref.min(-1) != 255).astype('float32')

        chainer.testing.assert_allclose(ref, image)

    def test_backward_case1(self):
        """Backward if non-zero gradient is out of a face."""

        vertices = [
            [0.8, 0.8, 1.],
            [0.0, -0.5, 1.],
            [0.2, -0.4, 1.]]
        faces = [[0, 1, 2]]
        pxi = 35
        pyi = 25
        grad_ref = [
            [1.6725862, -0.26021874, 0.],
            [1.41986704, -1.64284933, 0.],
            [0., 0., 0.],
        ]

        renderer = neural_renderer.Renderer()
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False

        vertices = cp.array(vertices, 'float32')
        faces = cp.array(faces, 'int32')
        grad_ref = cp.array(grad_ref, 'float32')
        vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
        vertices = chainer.Variable(vertices)
        images = renderer.render_silhouettes(vertices, faces)
        loss = cf.sum(cf.absolute(images[:, pyi, pxi] - 1))
        loss.backward()

        chainer.testing.assert_allclose(vertices.grad, grad_ref, rtol=1e-2)

    def test_backward_case2(self):
        """Backward if non-zero gradient is on a face."""

        vertices = [
            [0.8, 0.8, 1.],
            [-0.5, -0.8, 1.],
            [0.8, -0.8, 1.]]
        faces = [[0, 1, 2]]
        pyi = 40
        pxi = 50
        grad_ref = [
            [0.98646867, 1.04628897, 0.],
            [-1.03415668, - 0.10403691, 0.],
            [3.00094461, - 1.55173182, 0.],
        ]

        renderer = neural_renderer.Renderer()
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False

        vertices = cp.array(vertices, 'float32')
        faces = cp.array(faces, 'int32')
        grad_ref = cp.array(grad_ref, 'float32')
        vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
        vertices = chainer.Variable(vertices)
        images = renderer.render_silhouettes(vertices, faces)
        loss = cf.sum(cf.absolute(images[:, pyi, pxi]))
        loss.backward()

        chainer.testing.assert_allclose(vertices.grad, grad_ref, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
