import torch
import torch.nn as nn

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)

class Rasterize(nn.Module):

    def __init__(self, image_size, near, far, eps, background_color,
            return_rgb=False, return_alpha=False, return_depth=False):
        super(Rasterize, self).__init__()

        if not any((return_rgb, return_alpha, return_depth)):
            raise RuntimeError('Nothing to draw')

        # arguments
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

        # input buffers
        self.faces = None
        self.textures = None
        self.grad_rgb_map = None
        self.grad_alpha_map = None
        self.grad_depth_map = None

        # output buffers
        self.rgb_map = None
        self.alpha_map = None
        self.depth_map = None
        self.grad_faces = None
        self.grad_textures = None

        # intermediate buffers
        self.face_index_map = None
        self.weight_map = None
        self.face_inv_map = None
        self.sampling_index_map = None
        self.sampling_weight_map = None

        # input information
        self.xp = None
        self.batch_size = None
        self.num_faces = None
        self.texture_size = None

    def forward(self):
        return
