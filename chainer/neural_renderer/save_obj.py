import os
import string

import chainer
import cupy as cp
import numpy as np
import scipy.misc


def create_texture_image(textures, texture_size_out=16):
    num_faces, texture_size_in = textures.shape[:2]
    tile_width = int((num_faces - 1.) ** 0.5) + 1
    tile_height = int((num_faces - 1.) / tile_width) + 1
    image = np.zeros((tile_height * texture_size_out, tile_width * texture_size_out, 3), 'float32')

    vertices = np.zeros((num_faces, 3, 2), 'float32')  # [:, :, XY]
    face_nums = np.arange(num_faces)
    column = face_nums % tile_width
    row = face_nums / tile_width
    vertices[:, 0, 0] = column * texture_size_out
    vertices[:, 0, 1] = row * texture_size_out
    vertices[:, 1, 0] = column * texture_size_out
    vertices[:, 1, 1] = (row + 1) * texture_size_out - 1
    vertices[:, 2, 0] = (column + 1) * texture_size_out - 1
    vertices[:, 2, 1] = (row + 1) * texture_size_out - 1

    image = chainer.cuda.to_gpu(image)
    vertices = chainer.cuda.to_gpu(vertices)
    textures = chainer.cuda.to_gpu(textures)

    loop = cp.arange(image.size / 3).astype('int32')
    chainer.cuda.elementwise(
        'int32 j, raw float32 image, raw float32 vertices_all, raw float32 textures',
        '',
        string.Template('''
            const int x = i % (${tile_width} * ${texture_size_out});
            const int y = i / (${tile_width} * ${texture_size_out});
            const int row = x / ${texture_size_out};
            const int column = y / ${texture_size_out};
            const int fn = row + column * ${tile_width};
            const int tsi = ${texture_size_in};

            const float* texture = &textures[fn * tsi * tsi * tsi * 3];
            const float* vertices = &vertices_all[fn * 3 * 2];
            const float* p0 = &vertices[2 * 0];
            const float* p1 = &vertices[2 * 1];
            const float* p2 = &vertices[2 * 2];

            /* */
            // if ((y % ${texture_size_out}) < (x % ${texture_size_out})) continue;

            /* compute face_inv */
            float face_inv[9] = {
                p1[1] - p2[1], p2[0] - p1[0], p1[0] * p2[1] - p2[0] * p1[1],
                p2[1] - p0[1], p0[0] - p2[0], p2[0] * p0[1] - p0[0] * p2[1],
                p0[1] - p1[1], p1[0] - p0[0], p0[0] * p1[1] - p1[0] * p0[1]};
            float face_inv_denominator = (
                p2[0] * (p0[1] - p1[1]) +
                p0[0] * (p1[1] - p2[1]) +
                p1[0] * (p2[1] - p0[1]));
            for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

            /* compute w = face_inv * p */
            float weight[3];
            float weight_sum = 0;
            for (int k = 0; k < 3; k++) {
                weight[k] = face_inv[3 * k + 0] * x + face_inv[3 * k + 1] * y + face_inv[3 * k + 2];
                weight_sum += weight[k];
            }
            for (int k = 0; k < 3; k++) weight[k] /= (weight_sum + ${eps});

            /* get texture index (float) */
            float texture_index_float[3];
            for (int k = 0; k < 3; k++) {
                float tif = weight[k] * (tsi - 1);
                tif = max(tif, 0.);
                tif = min(tif, tsi - 1 - ${eps});
                texture_index_float[k] = tif;
            }

            /* blend */
            float new_pixel[3] = {0, 0, 0};
            for (int pn = 0; pn < 8; pn++) {
                float w = 1;                         // weight
                int texture_index_int[3];            // index in source (int)
                for (int k = 0; k < 3; k++) {
                    if ((pn >> k) % 2 == 0) {
                        w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                        texture_index_int[k] = (int)texture_index_float[k];
                    } else {
                        w *= texture_index_float[k] - (int)texture_index_float[k];
                        texture_index_int[k] = (int)texture_index_float[k] + 1;
                    }
                }
                int isc = texture_index_int[0] * tsi * tsi + texture_index_int[1] * tsi + texture_index_int[2];
                for (int k = 0; k < 3; k++) new_pixel[k] += w * texture[isc * 3 + k];
            }
            for (int k = 0; k < 3; k++) image[i * 3 + k] = new_pixel[k];
        ''').substitute(
            num_faces=num_faces,
            texture_size_in=texture_size_in,
            texture_size_out=texture_size_out,
            tile_width=tile_width,
            eps=1e-5,
        ),
        'function',
    )(loop, image, vertices, textures)

    chainer.cuda.elementwise(
        'int32 j, raw float32 image, raw float32 vertices_all, raw float32 textures',
        '',
        string.Template('''
            const int x = i % (${tile_width} * ${texture_size_out});
            const int y = i / (${tile_width} * ${texture_size_out});
            const int row = x / ${texture_size_out};
            const int column = y / ${texture_size_out};
            const int fn = row + column * ${tile_width};
            const int tsi = ${texture_size_in};

            const float* texture = &textures[fn * tsi * tsi * tsi * 3];
            const float* vertices = &vertices_all[fn * 3 * 2];
            const float* p0 = &vertices[2 * 0];
            const float* p1 = &vertices[2 * 1];
            const float* p2 = &vertices[2 * 2];

            /* */
            if ((y % ${texture_size_out} + 1) == (x % ${texture_size_out})) {
                for (int k = 0; k < 3; k++) image[i * 3 + k] = image[
                    (y * ${tile_width} * ${texture_size_out} + (x - 1))  * 3 + k];
            }

        ''').substitute(
            num_faces=num_faces,
            texture_size_in=texture_size_in,
            texture_size_out=texture_size_out,
            tile_width=tile_width,
            eps=1e-5,
        ),
        'function',
    )(loop, image, vertices, textures)

    vertices[:, :, 0] /= (image.shape[1] - 1)
    vertices[:, :, 1] /= (image.shape[0] - 1)

    image = image[::-1, ::1]
    image = image.get()
    vertices = vertices.get()
    return image, vertices


def save_obj(filename, vertices, faces, textures=None):
    assert vertices.ndim == 2
    assert faces.ndim == 2

    if textures is not None:
        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        texture_image, vertices_textures = create_texture_image(textures)
        scipy.misc.toimage(texture_image, cmin=0, cmax=1).save(filename_texture)

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        for vertex in vertices:
            f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n')

        if textures is not None:
            for vertex in vertices_textures.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, 3 * i + 1, face[1] + 1, 3 * i + 2, face[2] + 1, 3 * i + 3))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

    if textures is not None:
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))
