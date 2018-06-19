from torch.utils.cpp_extension import load

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

load_textures_cuda = load(
        'load_textures_cuda', [os.path.join(dir_path,'load_textures_cuda.cpp'),
                          os.path.join(dir_path, 'load_textures_cuda_kernel.cu')], verbose=True)

def load_textures(image, faces, textures, is_update):
    textures = load_textures_cuda.load_textures(image, faces, is_update, textures)
    return textures
