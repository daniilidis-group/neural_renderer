import torch.nn.functional as F

from .cuda.rasterize import Rasterize

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)

def rasterize_rgbad(
        faces,
        textures=None,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR,
        return_rgb=True,
        return_alpha=True,
        return_depth=True,
):
    """
    Generate RGB, alpha channel, and depth images from faces and textures (for RGB).

    Args:
        faces (torch.Tensor): Faces. The shape is [batch size, number of faces, 3 (vertices), 3 (XYZ)].
        textures (torch.Tensor): Textures.
            The shape is [batch size, number of faces, texture size, texture size, texture size, 3 (RGB)].
        image_size (int): Width and height of rendered images.
        anti_aliasing (bool): do anti-aliasing by super-sampling.
        near (float): nearest z-coordinate to draw.
        far (float): farthest z-coordinate to draw.
        eps (float): small epsilon for approximated differentiation.
        background_color (tuple): background color of RGB images.
        return_rgb (bool): generate RGB images or not.
        return_alpha (bool): generate alpha channels or not.
        return_depth (bool): generate depth images or not.

    Returns:
        dict:
            {
                'rgb': RGB images. The shape is [batch size, 3, image_size, image_size].
                'alpha': Alpha channels. The shape is [batch size, image_size, image_size].
                'depth': Depth images. The shape is [batch size, image_size, image_size].
            }

    """
    if textures is None:
        inputs = [faces, None]
    else:
        inputs = [faces, textures]

    if anti_aliasing:
        # 2x super-sampling
        rgb, alpha, depth = Rasterize(
            image_size * 2, near, far, eps, background_color, return_rgb, return_alpha, return_depth)(*inputs)
    else:
        rgb, alpha, depth = Rasterize(
            image_size, near, far, eps, background_color, return_rgb, return_alpha, return_depth)(*inputs)

    # transpose & vertical flip
    if return_rgb:
        rgb = rgb.permute((0, 3, 1, 2))
        # pytorch does not support negative slicing for the moment
        # may need to look at this again because it seems to be very slow
        # rgb = rgb[:, :, ::-1, :]
        rgb = rgb[:, :, list(reversed(range(rgb.shape[2]))), :]
    if return_alpha:
        # alpha = alpha[:, ::-1, :]
        alpha = alpha[:, list(reversed(range(alpha.shape[1]))), :]
    if return_depth:
        # depth = depth[:, ::-1, :]
        depth = depth[:, list(reversed(range(depth.shape[1]))), :]

    if anti_aliasing:
        # 0.5x down-sampling
        if return_rgb:
            rgb = F.avg_pool2d(rgb, kernel_size=(2,2))
        if return_alpha:
            alpha = F.avg_pool2d(alpha[:, None, :, :], kernel_size=(2, 2))[:, 0]
        if return_depth:
            depth = F.avg_pool2d(depth[:, None, :, :], kernel_size=(2, 2))[:, 0]

    ret = {
        'rgb': rgb if return_rgb else None,
        'alpha': alpha if return_alpha else None,
        'depth': depth if return_depth else None,
    }

    return ret


def rasterize(
        faces,
        textures,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR,
):
    """
    Generate RGB images from faces and textures.

    Args:
        faces: see `rasterize_rgbad`.
        textures: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.
        background_color: see `rasterize_rgbad`.

    Returns:
        ~torch.Tensor: RGB images. The shape is [batch size, 3, image_size, image_size].

    """
    return rasterize_rgbad(
        faces, textures, image_size, anti_aliasing, near, far, eps, background_color, True, False, False)['rgb']


def rasterize_silhouettes(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
):
    """
    Generate alpha channels from faces.

    Args:
        faces: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.

    Returns:
        ~torch.Tensor: Alpha channels. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, True, False)['alpha']


def rasterize_depth(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
):
    """
    Generate depth images from faces.

    Args:
        faces: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.

    Returns:
        ~torch.Tensor: Depth images. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, False, True)['depth']
