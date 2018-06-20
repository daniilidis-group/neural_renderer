"""
Example 2. Optimizing vertices.
"""
import argparse
import glob
import os

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio

from context import neural_renderer


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        # load .obj
        vertices, faces = neural_renderer.load_obj(filename_obj)
        self.vertices = nn.Parameter(vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype(np.float32).mean(-1) / 255.)[None, ::]
        self.register_buffer('image_ref', image_ref)

        # setup renderer
        renderer = neural_renderer.Renderer()
        self.renderer = renderer

    def forward(self):
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, 90)
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = torch.sum((image - self.image_ref[None, :, :])**2)
        return loss


def make_gif(working_directory, filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in glob.glob('%s/_tmp_*.png' % working_directory):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./data/teapot.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./data/example2_ref.png')
    parser.add_argument(
        '-oo', '--filename_output_optimization', type=str, default='./data/example2_optimization.gif')
    parser.add_argument(
        '-or', '--filename_output_result', type=str, default='./data/example2_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output_result)

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # optimizer.setup(model)
    loop = tqdm.tqdm(range(10))
    with torch.autograd.profiler.profile() as prof:
        for i in loop:
            loop.set_description('Optimizing')
            # optimizer.target.cleargrads()
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            images = model.renderer.render_silhouettes(model.vertices, model.faces)
            image = images.detach().cpu().numpy()[0]
            imsave('%s/_tmp_%04d.png' % (working_directory, i), image)
    print(prof.key_averages())
    make_gif(working_directory, args.filename_output_optimization)

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, azimuth)
        images = model.renderer.render(model.vertices, model.faces, model.textures)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('%s/_tmp_%04d.png' % (working_directory, num), image)
    make_gif(working_directory, args.filename_output_result)


if __name__ == '__main__':
    main()
