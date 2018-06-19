import torch
import numpy as np
from torch.autograd import Function
from torch.utils.cpp_extension import load

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

create_texture_image_cuda = load(
                          'create_texture_image', [os.path.join(dir_path,'create_texture_image_cuda.cpp'),
                          os.path.join(dir_path, 'create_texture_image_cuda_kernel.cu')], verbose=True)

class CreateTextureImageFunction(Function):

    @staticmethod
    def forward(ctx, textures, texture_size_out=16):
        # argument order is swapped to follow the standard cuda conventions
        num_faces, texture_size_in = textures.shape[:2]
        tile_width = int((num_faces - 1.) ** 0.5) + 1
        tile_height = int((num_faces - 1.) / tile_width) + 1
        image = torch.zeros(tile_height * texture_size_out, tile_width * texture_size_out, 3, dtype=np.float32)
        vertices = np.zeros((num_faces, 3, 2), dtype=np.float32)  # [:, :, XY]
        face_nums = np.arange(num_faces)
        column = face_nums % tile_width
        row = face_nums / tile_width
        vertices[:, 0, 0] = column * texture_size_out
        vertices[:, 0, 1] = row * texture_size_out
        vertices[:, 1, 0] = column * texture_size_out
        vertices[:, 1, 1] = (row + 1) * texture_size_out - 1
        vertices[:, 2, 0] = (column + 1) * texture_size_out - 1
        vertices[:, 2, 1] = (row + 1) * texture_size_out - 1
        image = image.cuda()
        vertices = vertices.cuda()
        textures = textures.cuda()
        image = create_texture_image_cuda.create_texture_image(vertices, textures, image, 1e-5)

        vertices[:, :, 0] /= (image.shape[1] - 1)
        vertices[:, :, 1] /= (image.shape[0] - 1)

        return image, vertices
