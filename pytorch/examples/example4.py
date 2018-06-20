"""
Example 4. Finding camera parameters.
"""
import argparse
import glob
import os

import torch
import torch.nn as nn

import numpy as np
import scipy.misc
import tqdm
import imageio

from context import neural_renderer


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        # load .obj
        vertices, faces = neural_renderer.load_obj(filename_obj)
        self.vertices = vertices[None, :, :]
        self.faces = faces[None, :, :]

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = textures.cuda()

        # load reference image
        if filename_ref is not None:
            self.image_ref = (scipy.misc.imread(filename_ref).max(-1) != 0).astype('float32')
            self.image_ref = torch.from_numpy(self.image_ref).cuda()
        else:
            self.image_ref = None

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))

        # setup renderer
        renderer = neural_renderer.Renderer()
        renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        return loss


def make_gif(working_directory, filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in glob.glob('%s/_tmp_*.png' % working_directory):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.to_gpu()

    model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 30, -15)
    images = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.data.get()[0]
    scipy.misc.toimage(image, cmin=0, cmax=1).save(filename_ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./data/teapot.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./data/example4_ref.png')
    parser.add_argument('-or', '--filename_output', type=str, default='./data/example4_result.gif')
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        images = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0]
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, i))
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.data < 70:
            break
    make_gif(working_directory, args.filename_output)


if __name__ == '__main__':
    main()
