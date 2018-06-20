"""
Example 3. Optimizing textures.
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

from .context import neural_renderer


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = neural_renderer.load_obj(filename_obj)
        self.vertices = nn.Parameter(vertices[None, :, :])
        self.faces = faces[None, :, :]
        # self.register_buffer('mesh_faces', self.faces)

        # create textures
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        self.image_ref = scipy.misc.imread(filename_ref).astype('float32') / 255.
        self.image_ref = torch.from_numpy(self.image_ref).cuda().permute(2,0,1)[None, ::]
        # self.register_buffer('m_image_ref', self.image_ref)

        # setup renderer
        renderer = neural_renderer.Renderer()
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer


    def forward(self):
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        image = self.renderer.render(self.vertices, self.faces, torch.tanh(self.textures))
        loss = torch.sum((image - self.image_ref) ** 2)
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
    parser.add_argument('-ir', '--filename_ref', type=str, default='./data/example3_ref.png')
    parser.add_argument('-or', '--filename_output', type=str, default='./data/example3_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        # from IPython.core.debugger import Pdb
        # Pdb().set_trace()
        optimizer.step()

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, azimuth)
        images = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))
    make_gif(working_directory, args.filename_output)


if __name__ == '__main__':
    main()
