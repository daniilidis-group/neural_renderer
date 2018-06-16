"""
Example 2. Optimizing vertices.
"""
import argparse
import glob
import os
import subprocess

import chainer
import chainer.functions as cf
import numpy as np
import scipy.misc
import tqdm

import neural_renderer


class Model(chainer.Link):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        with self.init_scope():
            # load .obj
            vertices, faces = neural_renderer.load_obj(filename_obj)
            self.vertices = chainer.Parameter(vertices[None, :, :])
            self.faces = faces[None, :, :]

            # create textures
            texture_size = 2
            textures = np.ones((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3), 'float32')
            self.textures = textures

            # load reference image
            self.image_ref = scipy.misc.imread(filename_ref).astype('float32').mean(-1) / 255.

            # setup renderer
            renderer = neural_renderer.Renderer()
            self.renderer = renderer

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.textures = chainer.cuda.to_gpu(self.textures, device)
        self.image_ref = chainer.cuda.to_gpu(self.image_ref, device)

    def __call__(self):
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, 90)
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = cf.sum(cf.square(image - self.image_ref[None, :, :]))
        return loss


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./examples/data/example2_ref.png')
    parser.add_argument(
        '-oo', '--filename_output_optimization', type=str, default='./examples/data/example2_optimization.gif')
    parser.add_argument(
        '-or', '--filename_output_result', type=str, default='./examples/data/example2_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output_result)

    model = Model(args.filename_obj, args.filename_ref)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    loop = tqdm.tqdm(range(300))
    for i in loop:
        loop.set_description('Optimizing')
        optimizer.target.cleargrads()
        loss = model()
        loss.backward()
        optimizer.update()
        images = model.renderer.render_silhouettes(model.vertices, model.faces)
        image = images.data.get()[0]
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, i))
    make_gif(working_directory, args.filename_output_optimization)

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, azimuth)
        images = model.renderer.render(model.vertices, model.faces, model.textures)
        image = images.data.get()[0].transpose((1, 2, 0))
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))
    make_gif(working_directory, args.filename_output_result)


if __name__ == '__main__':
    run()
