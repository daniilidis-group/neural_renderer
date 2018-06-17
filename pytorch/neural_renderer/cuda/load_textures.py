import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

load_textures = load(
        'load_textures', [os.path.join(dir_path,'load_textures_cuda.cpp'),
                          os.path.join(dir_path, 'load_textures_cuda_kernel.cu')], verbose=True)
help(load_textures)

class LoadTexturesFunction(Function):

    @staticmethod
    def forward(ctx, image, faces, textures, is_update):
        # argument order is swapped to follow the standard cuda conventions
        textures = load_textures.forward(image, faces, is_update, textures)
        return textures

class LoadTextures(nn.Module):

    def forward(self, image, faces, textures, is_update):
        return LoadTexturesFunction.apply(image, faces, textures, is_update)

