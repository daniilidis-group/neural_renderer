import os
import unittest
import numpy as np
import neural_renderer
import chainer
import scipy

class TestCore(unittest.TestCase):
    def test_save_obj(self):
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        neural_renderer.save_obj('./tests/data/teapot2.obj', vertices, faces)
        vertices2, faces2 = neural_renderer.load_obj('./tests/data/teapot.obj')
        os.remove('./tests/data/teapot2.obj')
        assert np.allclose(vertices, vertices2)
        assert np.allclose(faces, faces2)

    def test_texture(self):
        pass
        # renderer = neural_renderer.Renderer()
        # renderer.eye = neural_renderer.get_points_from_angles(2, 15, 30)
        # renderer.eye = neural_renderer.get_points_from_angles(2, 15, -90)
        #
        # vertices, faces, textures = neural_renderer.load_obj(
        #     './tests/data/4e49873292196f02574b5684eaec43e9/model.obj', load_texture=True, texture_size=16, normalization=False)
        #
        # # vertices, faces, textures = neural_renderer.load_obj('./tests/data/1cde62b063e14777c9152a706245d48/model.obj')
        # neural_renderer.save_obj('./tests/data/tmp.obj', vertices, faces, textures)
        #
        # vertices, faces, textures = neural_renderer.load_obj('./tests/data/tmp.obj', load_texture=True, texture_size=16)
        # vertices = chainer.cuda.to_gpu(vertices)
        # faces = chainer.cuda.to_gpu(faces)
        # textures = chainer.cuda.to_gpu(textures)
        # images = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).data.get()
        # scipy.misc.imsave('./tests/data/car2.png', scipy.misc.toimage(images[0]))

if __name__ == '__main__':
    unittest.main()
