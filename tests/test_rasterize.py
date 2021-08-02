import unittest
import os

import torch
import numpy as np
from skimage.io import imread, imsave

import neural_renderer as nr
import utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class TestRasterize(unittest.TestCase):
    def test_forward_case1(self):
        """Rendering a teapot without anti-aliasing."""

        # load teapot
        vertices, faces, textures = utils.load_teapot_batch()
        vertices = vertices.cuda()
        faces = faces.cuda()
        textures = textures.cuda()

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 256
        renderer.anti_aliasing = False

        # render
        images, _, _ = renderer(vertices, faces, textures)
        images = images.detach().cpu().numpy()
        image = images[2]
        image = image.transpose((1, 2, 0))

        imsave(os.path.join(data_dir, 'test_rasterize1.png'), image)

    def test_forward_case2(self):
        """Rendering a teapot with anti-aliasing and another viewpoint."""

        # load teapot
        vertices, faces, textures = utils.load_teapot_batch()
        vertices = vertices.cuda()
        faces = faces.cuda()
        textures = textures.cuda()

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = [1, 1, -2.7]

        # render
        images, _, _ = renderer(vertices, faces, textures)
        images = images.detach().cpu().numpy()
        image = images[2]
        image = image.transpose((1, 2, 0))

        imsave(os.path.join(data_dir, 'test_rasterize2.png'), image)

    def test_forward_case3(self):
        """Whether a silhouette by neural renderer matches that by Blender."""

        # load teapot
        vertices, faces, textures = utils.load_teapot_batch()
        vertices = vertices.cuda()
        faces = faces.cuda()
        textures = textures.cuda()

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 256
        renderer.anti_aliasing = False
        renderer.light_intensity_ambient = 1.0
        renderer.light_intensity_directional = 0.0

        images, _, _ = renderer(vertices, faces, textures)
        images = images.detach().cpu().numpy()
        image = images[2].mean(0)

        # load reference image by blender
        ref = imread(os.path.join(data_dir, 'teapot_blender.png'))
        ref = utils.guess_foreground_mask(ref)

        image = image.astype(bool)
        iou = utils.calc_mask_iou(image.astype(bool), ref)
        self.assertGreater(iou, 0.999)

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

        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False
        renderer.light_intensity_ambient = 1.0
        renderer.light_intensity_directional = 0.0

        vertices = torch.from_numpy(np.array(vertices, dtype=np.float32)).cuda()
        faces = torch.from_numpy(np.array(faces, dtype=np.int32)).cuda()
        textures = torch.ones(faces.shape[0], 4, 4, 4, 3, dtype=torch.float32).cuda()
        grad_ref = torch.from_numpy(np.array(grad_ref, dtype=np.float32)).cuda()
        vertices, faces, textures, grad_ref = utils.to_minibatch((vertices, faces, textures, grad_ref))
        vertices, faces, textures, grad_ref = vertices.cuda(), faces.cuda(), textures.cuda(), grad_ref.cuda()
        vertices.requires_grad = True
        images, _, _ = renderer(vertices, faces, textures)
        images = torch.mean(images, dim=1)
        loss = torch.sum(torch.abs(images[:, pyi, pxi] - 1))
        loss.backward()

        assert(torch.allclose(vertices.grad, grad_ref, rtol=1e-2))

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

        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False
        renderer.light_intensity_ambient = 1.0
        renderer.light_intensity_directional = 0.0

        vertices = torch.from_numpy(np.array(vertices, dtype=np.float32)).cuda()
        faces = torch.from_numpy(np.array(faces, dtype=np.int32)).cuda()
        textures = torch.ones(faces.shape[0], 4, 4, 4, 3, dtype=torch.float32).cuda()
        grad_ref = torch.from_numpy(np.array(grad_ref, dtype=np.float32)).cuda()
        vertices, faces, textures, grad_ref = utils.to_minibatch((vertices, faces, textures, grad_ref))
        vertices.requires_grad=True

        images, _, _ = renderer(vertices, faces, textures)
        images = torch.mean(images, dim=1)
        loss = torch.sum(torch.abs(images[:, pyi, pxi]))
        loss.backward()

        assert(torch.allclose(vertices.grad, grad_ref, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
