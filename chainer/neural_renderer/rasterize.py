import os
import string

import chainer
import chainer.functions as cf

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)
USE_UNSAFE_IMPLEMENTATION = False

if 'NEURAL_RENDERER_UNSAFE' in os.environ and int(os.environ['NEURAL_RENDERER_UNSAFE']):
    USE_UNSAFE_IMPLEMENTATION = True


class Rasterize(chainer.Function):
    def __init__(
            self, image_size, near, far, eps, background_color, return_rgb=False, return_alpha=False,
            return_depth=False):
        super(Rasterize, self).__init__()

        if not any((return_rgb, return_alpha, return_depth)):
            # nothing to draw
            raise Exception

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

    def check_type_forward(self, in_types):
        assert in_types.size() in [1, 2]

        # faces: [batch size, num of faces, v012, XYZ]
        faces_type = in_types[0]
        chainer.utils.type_check.expect(
            faces_type.dtype.kind == 'f',
            faces_type.ndim == 4,
            faces_type.shape[2] == 3,
            faces_type.shape[3] == 3,
        )

        if self.return_rgb:
            # textures: [batch size, num of faces, texture size, texture size, texture size, RGB]
            textures_type = in_types[1]
            chainer.utils.type_check.expect(
                textures_type.dtype.kind == 'f',
                textures_type.ndim == 6,
                2 <= textures_type.shape[2],
                textures_type.shape[2] == textures_type.shape[3],
                textures_type.shape[3] == textures_type.shape[4],
                textures_type.shape[5] == 3,
                faces_type.shape[0] == textures_type.shape[0],
                faces_type.shape[1] == textures_type.shape[1],
            )

    ####################################################################################################################
    # forward
    def forward_face_index_map_gpu(self):
        # inputs:
        #   faces
        # outputs:
        #   if rgb: face_index_map, weight_map, depth_map
        #   if alpha: face_index_map, weight_map, depth_map
        #   if depth: face_index_map, weight_map, depth_map, face_inv_map

        if USE_UNSAFE_IMPLEMENTATION:
            # very fast, but unable to run on some environments

            # for each face
            loop = self.xp.arange(self.batch_size * self.num_faces).astype('int32')
            lock = self.xp.zeros(self.face_index_map.shape, 'int32')
            chainer.cuda.elementwise(
                'int32 _, raw float32 faces, raw int32 face_index_map, raw float32 weight_map, ' +
                'raw float32 depth_map, raw float32 face_inv_map, raw int32 lock',
                '',
                string.Template('''
                    /* batch number, face, number, image size, face[v012][RGB] */
                    const int bn = i / ${num_faces};
                    const int fn = i % ${num_faces};
                    const int is = ${image_size};
                    const float* face = &faces[i * 9];

                    /* return if backside */
                    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) return;

                    /* pi[0], pi[1], pi[2] = leftmost, middle, rightmost points */
                    int pi[3];
                    if (face[0] < face[3]) {
                        if (face[6] < face[0]) pi[0] = 2; else pi[0] = 0;
                        if (face[3] < face[6]) pi[2] = 2; else pi[2] = 1;
                    } else {
                        if (face[6] < face[3]) pi[0] = 2; else pi[0] = 1;
                        if (face[0] < face[6]) pi[2] = 2; else pi[2] = 0;
                    }
                    for (int k = 0; k < 3; k++) if (pi[0] != k && pi[2] != k) pi[1] = k;

                    /* p[num][xyz]: x, y is normalized from [-1, 1] to [0, is - 1]. */
                    float p[3][3];
                    for (int num = 0; num < 3; num++) {
                        for (int dim = 0; dim < 3; dim++) {
                            if (dim != 2) {
                                p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);
                            } else {
                                p[num][dim] = face[3 * pi[num] + dim];
                            }
                        }
                    }
                    if (p[0][0] == p[2][0]) return; // line, not triangle

                    /* compute face_inv */
                    float face_inv[9] = {
                        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
                        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
                        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
                    float face_inv_denominator = (
                        p[2][0] * (p[0][1] - p[1][1]) +
                        p[0][0] * (p[1][1] - p[2][1]) +
                        p[1][0] * (p[2][1] - p[0][1]));
                    for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

                    /* from left to right */
                    const int xi_min = max(ceil(p[0][0]), 0.);
                    const int xi_max = min(p[2][0], is - 1.);
                    for (int xi = xi_min; xi <= xi_max; xi++) {
                        /* compute yi_min and yi_max */
                        float yi1, yi2;
                        if (xi <= p[1][0]) {
                            if (p[1][0] - p[0][0] != 0) {
                                yi1 = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];
                            } else {
                                yi1 = p[1][1];
                            }
                        } else {
                            if (p[2][0] - p[1][0] != 0) {
                                yi1 = (p[2][1] - p[1][1]) / (p[2][0] - p[1][0]) * (xi - p[1][0]) + p[1][1];
                            } else {
                                yi1 = p[1][1];
                            }
                        }
                        yi2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];

                        /* from up to bottom */
                        int yi_min = max(0., ceil(min(yi1, yi2)));
                        int yi_max = min(max(yi1, yi2), is - 1.);
                        for (int yi = yi_min; yi <= yi_max; yi++) {
                            /* index in output buffers */
                            int index = bn * is * is + yi * is + xi;

                            /* compute w = face_inv * p */
                            float w[3];
                            for (int k = 0; k < 3; k++)
                                w[k] = face_inv[3 * k + 0] * xi + face_inv[3 * k + 1] * yi + face_inv[3 * k + 2];

                            /* sum(w) -> 1, 0 < w < 1 */
                            float w_sum = 0;
                            for (int k = 0; k < 3; k++) {
                                w[k] = min(max(w[k], 0.), 1.);
                                w_sum += w[k];
                            }
                            for (int k = 0; k < 3; k++) w[k] /= w_sum;

                            /* compute 1 / zp = sum(w / z) */
                            const float zp = 1. / (w[0] / p[0][2] + w[1] / p[1][2] + w[2] / p[2][2]);
                            if (zp <= ${near} || ${far} <= zp) continue;

                            /* lock and update */
                            bool locked = false;
                            do {
                                if (locked = atomicCAS(&lock[index], 0, 1) == 0) {
                                    if (zp < atomicAdd(&depth_map[index], 0)) {
                                        float record = 0;
                                        atomicExch(&depth_map[index], zp);
                                        atomicExch(&face_index_map[index], fn);
                                        for (int k = 0; k < 3; k++) atomicExch(&weight_map[3 * index + pi[k]], w[k]);
                                        if (${return_depth}) {
                                            for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++)
                                                atomicExch(
                                                    &face_inv_map[9 * index + 3 * pi[l] + k], face_inv[3 * l + k]);
                                        }
                                        record += atomicAdd(&depth_map[index], 0.);
                                        record += atomicAdd(&face_index_map[index], 0.);
                                        if (0 < record) atomicExch(&lock[index], 0);
                                    } else {
                                        atomicExch(&lock[index], 0);
                                    }
                                }
                            } while (!locked);
                        }
                    }
                ''').substitute(
                    num_faces=self.num_faces,
                    image_size=self.image_size,
                    near=self.near,
                    far=self.far,
                    return_rgb=int(self.return_rgb),
                    return_alpha=int(self.return_alpha),
                    return_depth=int(self.return_depth),
                ),
                'function',
            )(loop, self.faces, self.face_index_map, self.weight_map, self.depth_map, self.face_inv_map, lock)

        else:
            # for each face
            faces_inv = self.xp.zeros_like(self.faces)
            loop = self.xp.arange(self.batch_size * self.num_faces).astype('int32')
            chainer.cuda.elementwise(
                'int32 _, raw float32 faces, raw float32 faces_inv',
                '',
                string.Template('''
                    /* face[v012][RGB] */
                    const int is = ${image_size};
                    const float* face = &faces[i * 9];
                    float* face_inv_g = &faces_inv[i * 9];

                    /* return if backside */
                    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
                        continue;

                    /* p[num][xy]: x, y is normalized from [-1, 1] to [0, is - 1]. */
                    float p[3][2];
                    for (int num = 0; num < 3; num++) for (int dim = 0; dim < 2; dim++)
                        p[num][dim] = 0.5 * (face[3 * num + dim] * is + is - 1);

                    /* compute face_inv */
                    float face_inv[9] = {
                        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
                        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
                        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
                    float face_inv_denominator = (
                        p[2][0] * (p[0][1] - p[1][1]) +
                        p[0][0] * (p[1][1] - p[2][1]) +
                        p[1][0] * (p[2][1] - p[0][1]));
                    for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

                    /* set to global memory */
                    for (int k = 0; k < 9; k++) face_inv_g[k] = face_inv[k];
                ''').substitute(
                    image_size=self.image_size,
                ),
                'function',
            )(loop, self.faces, faces_inv)

            # for each pixel
            loop = self.xp.arange(self.batch_size * self.image_size * self.image_size).astype('int32')
            chainer.cuda.elementwise(
                'int32 _, raw float32 faces, raw float32 faces_inv, raw int32 face_index_map, ' +
                'raw float32 weight_map, raw float32 depth_map, raw float32 face_inv_map',
                '',
                string.Template('''
                    const int is = ${image_size};
                    const int nf = ${num_faces};
                    const int bn = i / (is * is);
                    const int pn = i % (is * is);
                    const int yi = pn / is;
                    const int xi = pn % is;
                    const float yp = (2. * yi + 1 - is) / is;
                    const float xp = (2. * xi + 1 - is) / is;

                    float* face = &faces[bn * nf * 9] - 9;
                    float* face_inv = &faces_inv[bn * nf * 9] - 9;
                    float depth_min = ${far};
                    int face_index_min = -1;
                    float weight_min[3];
                    float face_inv_min[9];
                    for (int fn = 0; fn < nf; fn++) {
                        /* go to next face */
                        face += 9;
                        face_inv += 9;

                        /* return if backside */
                        if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
                            continue;

                        /* check [py, px] is inside the face */
                        if (((yp - face[1]) * (face[3] - face[0]) < (xp - face[0]) * (face[4] - face[1])) ||
                            ((yp - face[4]) * (face[6] - face[3]) < (xp - face[3]) * (face[7] - face[4])) ||
                            ((yp - face[7]) * (face[0] - face[6]) < (xp - face[6]) * (face[1] - face[7]))) continue;

                        /* compute w = face_inv * p */
                        float w[3];
                        for (int k = 0; k < 3; k++)
                        w[0] = face_inv[3 * 0 + 0] * xi + face_inv[3 * 0 + 1] * yi + face_inv[3 * 0 + 2];
                        w[1] = face_inv[3 * 1 + 0] * xi + face_inv[3 * 1 + 1] * yi + face_inv[3 * 1 + 2];
                        w[2] = face_inv[3 * 2 + 0] * xi + face_inv[3 * 2 + 1] * yi + face_inv[3 * 2 + 2];

                        /* sum(w) -> 1, 0 < w < 1 */
                        float w_sum = 0;
                        for (int k = 0; k < 3; k++) {
                            w[k] = min(max(w[k], 0.), 1.);
                            w_sum += w[k];
                        }
                        for (int k = 0; k < 3; k++) w[k] /= w_sum;

                        /* compute 1 / zp = sum(w / z) */
                        const float zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);
                        if (zp <= ${near} || ${far} <= zp) continue;

                        /* check z-buffer */
                        if (zp < depth_min) {
                            depth_min = zp;
                            face_index_min = fn;
                            for (int k = 0; k < 3; k++) weight_min[k] = w[k];
                            if (${return_depth}) for (int k = 0; k < 9; k++) face_inv_min[k] = face_inv[k];
                        }
                    }

                    /* set to global memory */
                    if (0 <= face_index_min) {
                        depth_map[i] = depth_min;
                        face_index_map[i] = face_index_min;
                        for (int k = 0; k < 3; k++) weight_map[3 * i + k] = weight_min[k];
                        if (${return_depth}) for (int k = 0; k < 9; k++) face_inv_map[9 * i + k] = face_inv_min[k];
                    }
                ''').substitute(
                    num_faces=self.num_faces,
                    image_size=self.image_size,
                    near=self.near,
                    far=self.far,
                    return_rgb=int(self.return_rgb),
                    return_alpha=int(self.return_alpha),
                    return_depth=int(self.return_depth),
                ),
                'function',
            )(loop, self.faces, faces_inv, self.face_index_map, self.weight_map, self.depth_map, self.face_inv_map)

    def forward_texture_sampling(self):
        # inputs:
        #   faces, textures, face_index_map, weight_map, depth_map
        # outputs:
        #   if rgb: rgb_map, sampling_index_map, sampling_weight_map

        if not self.return_rgb:
            return

        # for each pixel
        loop = self.xp.arange(self.batch_size * self.image_size * self.image_size).astype('int32')
        chainer.cuda.elementwise(
            'int32 _, raw float32 faces, raw float32 textures, raw int32 face_index_map, raw float32 weight_map, ' +
            'raw float32 depth_map, raw float32 rgb_map, raw int32 sampling_index_map, raw float32 sampling_weight_map',
            '',
            string.Template('''
                const int face_index = face_index_map[i];

                if (0 <= face_index) {
                    /*
                        from global variables:
                        batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
                        texture[ts][ts][ts][RGB], sampling indices[8], sampling_weights[8];
                    */

                    const int bn = i / (${image_size} * ${image_size});
                    const int nf = ${num_faces};
                    const int ts = ${texture_size};
                    const float* face = &faces[face_index * 9];
                    const float* texture = &textures[(bn * nf + face_index) * ts * ts * ts * 3];
                    float* pixel = &rgb_map[i * 3];
                    const float* weight = &weight_map[i * 3];
                    const float depth = depth_map[i];
                    int* sampling_indices = &sampling_index_map[i * 8];
                    float* sampling_weights = &sampling_weight_map[i * 8];

                    /* get texture index (float) */
                    float texture_index_float[3];
                    for (int k = 0; k < 3; k++) {
                        float tif = weight[k] * (ts - 1) * (depth / (face[3 * k + 2]));
                        tif = max(tif, 0.);
                        tif = min(tif, ts - 1 - ${eps});
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

                        int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
                        for (int k = 0; k < 3; k++) new_pixel[k] += w * texture[isc * 3 + k];
                        sampling_indices[pn] = isc;
                        sampling_weights[pn] = w;
                    }
                    for (int k = 0; k < 3; k++) pixel[k] = new_pixel[k];
                }
            ''').substitute(
                image_size=self.image_size,
                num_faces=self.num_faces,
                texture_size=self.texture_size,
                eps=self.eps,
            ),
            'function',
        )(
            loop, self.faces, self.textures, self.face_index_map, self.weight_map, self.depth_map, self.rgb_map,
            self.sampling_index_map, self.sampling_weight_map,
        )

    def forward_alpha_map_gpu(self):
        # inputs:
        #   face_index_map,
        # outputs:
        #   if alpha: alpha_map

        if not self.return_alpha:
            return

        self.alpha_map[0 <= self.face_index_map] = 1

    def forward_background_gpu(self):
        # inputs:
        #   face_index_map, rgb_map, background_color
        # outputs:
        #   if rgb: rgb_map

        if not self.return_rgb:
            return

        background_color = self.xp.array(self.background_color, 'float32')
        mask = (0 <= self.face_index_map).astype('float32')[:, :, :, None]
        if background_color.ndim == 1:
            self.rgb_map = self.rgb_map * mask + (1 - mask) * background_color[None, None, None, :]
        elif background_color.ndim == 2:
            self.rgb_map = self.rgb_map * mask + (1 - mask) * background_color[:, None, None, :]

    def forward_gpu(self, inputs):
        # get input information
        self.xp = chainer.cuda.get_array_module(inputs[0])
        self.faces = self.xp.ascontiguousarray(inputs[0].copy())
        self.batch_size, self.num_faces = self.faces.shape[:2]
        if self.return_rgb:
            textures = self.xp.ascontiguousarray(inputs[1])
            self.textures = textures
            self.texture_size = textures.shape[2]

        # initialize outputs
        self.face_index_map = -1 * self.xp.ones((self.batch_size, self.image_size, self.image_size), 'int32')
        self.weight_map = self.xp.zeros((self.batch_size, self.image_size, self.image_size, 3), 'float32')
        self.depth_map = self.xp.zeros(self.face_index_map.shape, 'float32') + self.far
        if self.return_rgb:
            self.rgb_map = self.xp.zeros((self.batch_size, self.image_size, self.image_size, 3), 'float32')
            self.sampling_index_map = self.xp.zeros((self.batch_size, self.image_size, self.image_size, 8), 'int32')
            self.sampling_weight_map = self.xp.zeros((self.batch_size, self.image_size, self.image_size, 8), 'float32')
        else:
            self.rgb_map = self.xp.zeros(1, 'float32')
            self.sampling_index_map = self.xp.zeros(1, 'int32')
            self.sampling_weight_map = self.xp.zeros(1, 'float32')
        if self.return_alpha:
            self.alpha_map = self.xp.zeros((self.batch_size, self.image_size, self.image_size), 'float32')
        else:
            self.alpha_map = self.xp.zeros(1, 'float32')
        if self.return_depth:
            self.face_inv_map = self.xp.zeros((self.batch_size, self.image_size, self.image_size, 3, 3), 'float32')
        else:
            self.face_inv_map = self.xp.zeros(1, 'float32')

        # forward pass
        self.forward_face_index_map_gpu()
        self.forward_texture_sampling()
        self.forward_background_gpu()
        self.forward_alpha_map_gpu()

        # return
        rgb_r, alpha_r, depth_r = None, None, None
        if self.return_rgb:
            rgb_r = self.rgb_map
        if self.return_alpha:
            alpha_r = self.alpha_map.copy()
        if self.return_depth:
            depth_r = self.depth_map.copy()

        return rgb_r, alpha_r, depth_r

    ####################################################################################################################
    # backward
    def backward_pixel_map_gpu(self):
        # inputs:
        #   face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map
        # outputs:
        #   if rgb or alpha: grad_faces

        if (not self.return_rgb) and (not self.return_alpha):
            return

        # for each face
        loop = self.xp.arange(self.batch_size * self.num_faces).astype('int32')
        chainer.cuda.elementwise(
            'int32 _, raw float32 faces, raw int32 face_index_map, raw float32 rgb_map, raw float32 alpha_map, ' +
            'raw float32 grad_rgb_map, raw float32 grad_alpha_map, raw float32 grad_faces',
            '',
            string.Template('''
                const int bn = i / ${num_faces};
                const int fn = i % ${num_faces};
                const int is = ${image_size};
                const float* face = &faces[i * 9];
                float grad_face[9] = {};

                /* check backside */
                if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) return;

                /* for each edge */
                for (int edge_num = 0; edge_num < 3; edge_num++) {
                    /* set points of target edge */
                    int pi[3];
                    float pp[3][2];
                    for (int num = 0; num < 3; num++) pi[num] = (edge_num + num) % 3;
                    for (int num = 0; num < 3; num++) for (int dim = 0; dim < 2; dim++)
                        pp[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);

                    /* for dy, dx */
                    for (int axis = 0; axis < 2; axis++) {
                        /* */
                        float p[3][2];
                        for (int num = 0; num < 3; num++) for (int dim = 0; dim < 2; dim++)
                            p[num][dim] = pp[num][(dim + axis) % 2];

                        /* set direction */
                        int direction;
                        if (axis == 0) {
                            if (p[0][0] < p[1][0]) direction = -1; else direction = 1;
                        } else {
                            if (p[0][0] < p[1][0]) direction = 1; else direction = -1;
                        }

                        /* along edge */
                        int d0_from, d0_to;
                        d0_from = max(ceil(min(p[0][0], p[1][0])), 0.);
                        d0_to = min(max(p[0][0], p[1][0]), is - 1.);
                        for (int d0 = d0_from; d0 <= d0_to; d0++) {
                            /* get cross point */
                            int d1_in, d1_out;
                            const float d1_cross = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                            if (0 < direction) d1_in = floor(d1_cross); else d1_in = ceil(d1_cross);
                            d1_out = d1_in + direction;

                            /* continue if cross point is not shown */
                            if (d1_in < 0 || is <= d1_in) continue;
                            if (d1_out < 0 || is <= d1_out) continue;

                            /* get color of in-pixel and out-pixel */
                            float alpha_in;
                            float alpha_out;
                            float *rgb_in;
                            float *rgb_out;
                            int map_index_in, map_index_out;
                            if (axis == 0) {
                                map_index_in = bn * is * is + d1_in * is + d0;
                                map_index_out = bn * is * is + d1_out * is + d0;
                            } else {
                                map_index_in = bn * is * is + d0 * is + d1_in;
                                map_index_out = bn * is * is + d0 * is + d1_out;
                            }
                            if (${return_alpha}) {
                                alpha_in = alpha_map[map_index_in];
                                alpha_out = alpha_map[map_index_out];
                            }
                            if (${return_rgb}) {
                                rgb_in = &rgb_map[map_index_in * 3];
                                rgb_out = &rgb_map[map_index_out * 3];
                            }

                            /* out */
                            bool is_in_fn = (face_index_map[map_index_in] == fn);
                            if (is_in_fn) {
                                int d1_limit;
                                if (0 < direction) d1_limit = is - 1; else d1_limit = 0;
                                int d1_from = max(min(d1_out, d1_limit), 0);
                                int d1_to = min(max(d1_out, d1_limit), is - 1);
                                float* alpha_map_p;
                                float* grad_alpha_map_p;
                                float* rgb_map_p;
                                float* grad_rgb_map_p;
                                int map_offset, map_index_from;
                                if (axis == 0) {
                                    map_offset = is;
                                    map_index_from = bn * is * is + d1_from * is + d0;
                                } else {
                                    map_offset = 1;
                                    map_index_from = bn * is * is + d0 * is + d1_from;
                                }
                                if (${return_alpha}) {
                                    alpha_map_p = &alpha_map[map_index_from];
                                    grad_alpha_map_p = &grad_alpha_map[map_index_from];
                                }
                                if (${return_rgb}) {
                                    rgb_map_p = &rgb_map[map_index_from * 3];
                                    grad_rgb_map_p = &grad_rgb_map[map_index_from * 3];
                                }
                                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                                    float diff_grad = 0;
                                    if (${return_alpha}) {
                                        diff_grad += (*alpha_map_p - alpha_in) * *grad_alpha_map_p;
                                    }
                                    if (${return_rgb}) {
                                        for (int k = 0; k < 3; k++)
                                            diff_grad += (rgb_map_p[k] - rgb_in[k]) * grad_rgb_map_p[k];
                                    }
                                    if (${return_alpha}) {
                                        alpha_map_p += map_offset;
                                        grad_alpha_map_p += map_offset;
                                    }
                                    if (${return_rgb}) {
                                        rgb_map_p += 3 * map_offset;
                                        grad_rgb_map_p += 3 * map_offset;
                                    }
                                    if (diff_grad <= 0) continue;
                                    if (p[1][0] != d0) {
                                        float dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                    if (p[0][0] != d0) {
                                        float dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                }
                            }

                            /* in */
                            {
                                int d1_limit;
                                float d0_cross2;
                                if ((d0 - p[0][0]) * (d0 - p[2][0]) < 0) {
                                    d0_cross2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                                } else {
                                    d0_cross2 = (p[1][1] - p[2][1]) / (p[1][0] - p[2][0]) * (d0 - p[2][0]) + p[2][1];
                                }
                                if (0 < direction) d1_limit = ceil(d0_cross2); else d1_limit = floor(d0_cross2);
                                int d1_from = max(min(d1_in, d1_limit), 0);
                                int d1_to = min(max(d1_in, d1_limit), is - 1);

                                int* face_index_map_p;
                                float* alpha_map_p;
                                float* grad_alpha_map_p;
                                float* rgb_map_p;
                                float* grad_rgb_map_p;
                                int map_index_from;
                                int map_offset;
                                if (axis == 0) map_offset = is; else map_offset = 1;
                                if (axis == 0) {
                                    map_index_from = bn * is * is + d1_from * is + d0;
                                } else {
                                    map_index_from = bn * is * is + d0 * is + d1_from;
                                }
                                face_index_map_p = &face_index_map[map_index_from] - map_offset;
                                if (${return_alpha}) {
                                    alpha_map_p = &alpha_map[map_index_from] - map_offset;
                                    grad_alpha_map_p = &grad_alpha_map[map_index_from] - map_offset;
                                }
                                if (${return_rgb}) {
                                    rgb_map_p = &rgb_map[map_index_from * 3] - 3 * map_offset;
                                    grad_rgb_map_p = &grad_rgb_map[map_index_from * 3] - 3 * map_offset;
                                }

                                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                                    face_index_map_p += map_offset;
                                    if (${return_alpha}) {
                                        alpha_map_p += map_offset;
                                        grad_alpha_map_p += map_offset;
                                    }
                                    if (${return_rgb}) {
                                        rgb_map_p += 3 * map_offset;
                                        grad_rgb_map_p += 3 * map_offset;
                                    }
                                    if (*face_index_map_p != fn) continue;

                                    float diff_grad = 0;
                                    if (${return_alpha}) {
                                        diff_grad += (*alpha_map_p - alpha_out) * *grad_alpha_map_p;
                                    }
                                    if (${return_rgb}) {
                                        for (int k = 0; k < 3; k++)
                                            diff_grad += (rgb_map_p[k] - rgb_out[k]) * grad_rgb_map_p[k];
                                    }
                                    if (diff_grad <= 0) continue;

                                    if (p[1][0] != d0) {
                                        float dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                    if (p[0][0] != d0) {
                                        float dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                                        dist = (0 < dist) ? dist + ${eps} : dist - ${eps};
                                        grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                                    }
                                }
                            }
                        }
                    }
                }

                /* set to global gradient variable */
                for (int k = 0; k < 9; k++) grad_faces[i * 9 + k] = grad_face[k];
            ''').substitute(
                image_size=self.image_size,
                num_faces=self.num_faces,
                eps=self.eps,
                return_rgb=int(self.return_rgb),
                return_alpha=int(self.return_alpha),
            ),
            'function',
        )(
            loop, self.faces, self.face_index_map, self.rgb_map, self.alpha_map, self.grad_rgb_map, self.grad_alpha_map,
            self.grad_faces,
        )

    def backward_textures_gpu(self):
        # inputs:
        #   face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map
        # outputs:
        #   if rgb: grad_textures

        if not self.return_rgb:
            return

        loop = self.xp.arange(self.batch_size * self.image_size * self.image_size).astype('int32')
        chainer.cuda.elementwise(
            'int32 _, raw int32 face_index_map, raw T sampling_weight_map, raw int32 sampling_index_map, ' +
            'raw T grad_rgb_map, raw T grad_textures',
            '',
            string.Template('''
                const int face_index = face_index_map[i];
                if (0 <= face_index) {
                int is = ${image_size};
                    int nf = ${num_faces};
                    int ts = ${texture_size};
                    int bn = i / (is * is);    // batch number [0 -> bs]

                    float* grad_texture = &grad_textures[(bn * nf + face_index) * ts * ts * ts * 3];
                    float* sampling_weight_map_p = &sampling_weight_map[i * 8];
                    int* sampling_index_map_p = &sampling_index_map[i * 8];
                    for (int pn = 0; pn < 8; pn++) {
                        float w = *sampling_weight_map_p++;
                        int isc = *sampling_index_map_p++;
                        float* grad_texture_p = &grad_texture[isc * 3];
                        float* grad_rgb_map_p = &grad_rgb_map[i * 3];
                        for (int k = 0; k < 3; k++) atomicAdd(grad_texture_p++, w * *grad_rgb_map_p++);
                    }
                }
            ''').substitute(
                image_size=self.image_size,
                num_faces=self.num_faces,
                texture_size=self.texture_size,
            ),
            'function',
        )(
            loop, self.face_index_map, self.sampling_weight_map, self.sampling_index_map, self.grad_rgb_map,
            self.grad_textures,
        )

    def backward_depth_map_gpu(self):
        # inputs:
        #   faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map
        # outputs:
        #   if depth: grad_faces

        if not self.return_depth:
            return

        # for each pixel
        loop = self.xp.arange(self.batch_size * self.image_size * self.image_size).astype('int32')
        chainer.cuda.elementwise(
            'int32 _, raw float32 faces, raw float32 depth_map, raw int32 face_index_map, ' +
            'raw float32 face_inv_map, raw float32 weight_map, raw float32 grad_depth_map, raw float32 grad_faces',
            '',
            string.Template('''
                const int fn = face_index_map[i];
                if (0 <= fn) {
                    const int nf = ${num_faces};
                    const int is = ${image_size};
                    const int bn = i / (is * is);
                    const float* face = &faces[(bn * nf + fn) * 9];
                    const float depth = depth_map[i];
                    const float depth2 = depth * depth;
                    const float* face_inv = &face_inv_map[i * 9];
                    const float* weight = &weight_map[i * 3];
                    const float grad_depth = grad_depth_map[i];
                    float* grad_face = &grad_faces[(bn * nf + fn) * 9];

                    /* derivative wrt z */
                    for (int k = 0; k < 3; k++) {
                        const float z_k = face[3 * k + 2];
                        atomicAdd(&grad_face[3 * k + 2], grad_depth * weight[k] * depth2 / (z_k * z_k));
                    }

                    /* derivative wrt x, y */
                    float tmp[3] = {};
                    for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) {
                        tmp[k] += -face_inv[3 * l + k] / face[3 * l + 2];
                    }
                    for (int k = 0; k < 3; k++) for (int l = 0; l < 2; l++) {
                        // k: point number, l: dimension
                        atomicAdd(&grad_face[3 * k + l], -grad_depth * tmp[l] * weight[k] * depth2 * is / 2);
                    }
                }
            ''').substitute(
                num_faces=self.num_faces,
                image_size=self.image_size,
            ),
            'function'
        )(
            loop, self.faces, self.depth_map, self.face_index_map, self.face_inv_map, self.weight_map,
            self.grad_depth_map, self.grad_faces,
        )

    def backward_gpu(self, inputs, grad_outputs):
        # initialize output buffers
        self.grad_faces = self.xp.ascontiguousarray(self.xp.zeros_like(self.faces, dtype='float32'))
        if self.return_rgb:
            self.grad_textures = self.xp.ascontiguousarray(self.xp.zeros_like(self.textures, dtype='float32'))
        else:
            self.grad_textures = self.xp.zeros(1, 'float32')

        # get grad_outputs
        if self.return_rgb:
            if grad_outputs[0] is not None:
                self.grad_rgb_map = self.xp.ascontiguousarray(grad_outputs[0])
            else:
                self.grad_rgb_map = self.xp.zeros_like(self.rgb_map)
        else:
            self.grad_rgb_map = self.xp.zeros(1, 'float32')
        if self.return_alpha:
            if grad_outputs[1] is not None:
                self.grad_alpha_map = self.xp.ascontiguousarray(grad_outputs[1])
            else:
                self.grad_alpha_map = self.xp.zeros_like(self.alpha_map)
        else:
            self.grad_alpha_map = self.xp.zeros(1, 'float32')
        if self.return_depth:
            if grad_outputs[2] is not None:
                self.grad_depth_map = self.xp.ascontiguousarray(grad_outputs[2])
            else:
                self.grad_depth_map = self.xp.zeros_like(self.depth_map)
        else:
            self.grad_depth_map = self.xp.zeros(1, 'float32')

        # backward pass
        self.backward_pixel_map_gpu()
        self.backward_textures_gpu()
        self.backward_depth_map_gpu()

        # return
        if len(self.inputs) == 1:
            return self.grad_faces,
        else:
            return self.grad_faces, self.grad_textures

    ####################################################################################################################
    # CPU
    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError


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
        faces (chainer.Variable): Faces. The shape is [batch size, number of faces, 3 (vertices), 3 (XYZ)].
        textures (chainer.Variable): Textures.
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
        inputs = [faces]
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
        rgb = rgb.transpose((0, 3, 1, 2))
        rgb = rgb[:, :, ::-1, :]
    if return_alpha:
        alpha = alpha[:, ::-1, :]
    if return_depth:
        depth = depth[:, ::-1, :]

    if anti_aliasing:
        # 0.5x down-sampling
        if return_rgb:
            rgb = cf.average_pooling_2d(rgb, 2, 2)
        if return_alpha:
            alpha = cf.average_pooling_2d(alpha[:, None, :, :], 2, 2)[:, 0]
        if return_depth:
            depth = cf.average_pooling_2d(depth[:, None, :, :], 2, 2)[:, 0]

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
        ~chainer.Variable: RGB images. The shape is [batch size, 3, image_size, image_size].

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
        ~chainer.Variable: Alpha channels. The shape is [batch size, image_size, image_size].

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
        ~chainer.Variable: Depth images. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, False, True)['depth']


def use_unsafe_rasterizer(flag):
    global USE_UNSAFE_IMPLEMENTATION
    USE_UNSAFE_IMPLEMENTATION = flag
