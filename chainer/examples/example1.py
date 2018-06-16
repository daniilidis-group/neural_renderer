"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import argparse
import glob
import os
import subprocess

import chainer
import numpy as np
import scipy.misc
import tqdm

import neural_renderer


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-o', '--filename_output', type=str, default='./examples/data/example1.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    vertices, faces = neural_renderer.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = np.ones((1, faces.shape[1], texture_size, texture_size, texture_size, 3), 'float32')

    # to gpu
    chainer.cuda.get_device_from_id(args.gpu).use()
    vertices = chainer.cuda.to_gpu(vertices)
    faces = chainer.cuda.to_gpu(faces)
    textures = chainer.cuda.to_gpu(textures)

    # create renderer
    renderer = neural_renderer.Renderer()

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = neural_renderer.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer.render(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.data.get()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))

    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, args.filename_output), shell=True)

    # remove temporary files
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


if __name__ == '__main__':
    run()
