#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// implementation of atomicExch for double input
__device__ double atomicExch(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

template <typename scalar_t>
__global__ void forward_face_index_map_cuda_kernel(
        scalar_t* __restrict__ faces,
        int32_t* __restrict__ face_index_map,
        scalar_t* __restrict__ weight_map,
        scalar_t* __restrict__ depth_map,
        scalar_t* __restrict__ face_inv_map,
        int32_t* __restrict__ lock,
        int64_t num_faces,
        int image_size,
        scalar_t near,
        scalar_t far,
        int return_rgb,
        int return_alpha,
        int return_depth) {
     /* batch number, face, number, image size, face[v012][RGB] */
     const int i = blockIdx.x * blockDim.x + threadIdx.x;
     const int bn = i / num_faces;
     const int fn = i % num_faces;
     const int is = image_size;
     const scalar_t* face = &faces[i * 9];
     
     /* return if backside */
     if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
         return;
     
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
     scalar_t p[3][3];
     for (int num = 0; num < 3; num++) {
         for (int dim = 0; dim < 3; dim++) {
             if (dim != 2) {
                 p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);
             } else {
                 p[num][dim] = face[3 * pi[num] + dim];
             }
         }
     }
     if (p[0][0] == p[2][0])
         return; // line, not triangle 
     
     /* compute face_inv */
     scalar_t face_inv[9] = {
         p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
         p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
         p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
     scalar_t face_inv_denominator = (
         p[2][0] * (p[0][1] - p[1][1]) +
         p[0][0] * (p[1][1] - p[2][1]) +
         p[1][0] * (p[2][1] - p[0][1]));
     for (int k = 0; k < 9; k++)
         face_inv[k] /= face_inv_denominator;
     
     /* from left to right */
     const int xi_min = max(ceil(p[0][0]), 0.);
     const int xi_max = min(p[2][0], is - 1.);
     for (int xi = xi_min; xi <= xi_max; xi++) {
         /* compute yi_min and yi_max */
         scalar_t yi1, yi2;
         if (xi <= p[1][0]) {
             if (p[1][0] - p[0][0] != 0) {
                 yi1 = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];
             }
             else {
                 yi1 = p[1][1];
             }
         }
         else {
             if (p[2][0] - p[1][0] != 0) {
                 yi1 = (p[2][1] - p[1][1]) / (p[2][0] - p[1][0]) * (xi - p[1][0]) + p[1][1];
             }
             else {
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
             scalar_t w[3];
             for (int k = 0; k < 3; k++)
                 w[k] = face_inv[3 * k + 0] * xi + face_inv[3 * k + 1] * yi + face_inv[3 * k + 2];
     
             /* sum(w) -> 1, 0 < w < 1 */
             scalar_t w_sum = 0;
             for (int k = 0; k < 3; k++) {
                 w[k] = min(max(w[k], 0.), 1.);
                 w_sum += w[k];
             }
             for (int k = 0; k < 3; k++) w[k] /= w_sum;
     
             /* compute 1 / zp = sum(w / z) */
             const scalar_t zp = 1. / (w[0] / p[0][2] + w[1] / p[1][2] + w[2] / p[2][2]);
             if (zp <= near || far <= zp)
                 continue;
     
             /* lock and update */
             bool locked = false;
             do {
                 if (locked = atomicCAS(&lock[index], 0, 1) == 0) {
                     if (zp < atomicAdd(&depth_map[index], 0)) {
                         float record = 0;
                         atomicExch(&depth_map[index], zp);
                         atomicExch(&face_index_map[index], fn);
                         for (int k = 0; k < 3; k++)
                             atomicExch(&weight_map[3 * index + pi[k]], w[k]);
                         if (return_depth) {
                             for (int k = 0; k < 3; k++)
                                 for (int l = 0; l < 3; l++)
                                 atomicExch(
                                     &face_inv_map[9 * index + 3 * pi[l] + k], face_inv[3 * l + k]);
                         }
                         record += atomicAdd(&depth_map[index], 0.);
                         record += atomicAdd(&face_index_map[index], 0.);
                         if (0 < record) atomicExch(&lock[index], 0);
                     }
                     else {
                         atomicExch(&lock[index], 0);
                     }
                 }
             } while (!locked);
         }
     }
}

template <typename scalar_t> __global__ void forward_texture_sampling_cuda_kernel(
		const scalar_t* __restrict__ faces,
		const scalar_t* __restrict__ textures,
		const int32_t* __restrict__ face_index_map,
		const scalar_t* __restrict__ weight_map,
		const scalar_t* __restrict__ depth_map,
		scalar_t* __restrict__ rgb_map,
		int32_t* __restrict__ sampling_index_map,
        scalar_t* __restrict__ sampling_weight_map,
        int32_t* __restrict__ lock,
        int num_faces,
        int image_size,
        int texture_size,
        scalar_t eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int face_index = face_index_map[i];
    
    if (face_index >= 0) {
        /*
            from global variables:
            batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
            texture[ts][ts][ts][RGB], sampling indices[8], sampling_weights[8];
        */
        const int bn = i / (image_size * image_size);
        const int nf = num_faces;
        const int ts = texture_size;
        const scalar_t* face = &faces[face_index * 9];
        const scalar_t* texture = &textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* pixel = &rgb_map[i * 3];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t depth = depth_map[i];
        int32_t* sampling_indices = &sampling_index_map[i * 8];
        scalar_t* sampling_weights = &sampling_weight_map[i * 8];
    
        /* get texture index (float) */
        scalar_t texture_index_float[3];
        for (int k = 0; k < 3; k++) { scalar_t tif = weight[k] * (ts - 1) * (depth / (face[3 * k + 2]));
            tif = max(tif, 0.);
            tif = min(tif, ts - 1 - eps);
            texture_index_float[k] = tif;
        }
    
        /* blend */
        scalar_t new_pixel[3] = {0, 0, 0};
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = 1;                         // weight
            int texture_index_int[3];            // index in source (int)
            for (int k = 0; k < 3; k++) {
                if ((pn >> k) % 2 == 0) {
                    w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                    texture_index_int[k] = (int)texture_index_float[k];
                }
                else {
                    w *= texture_index_float[k] - (int)texture_index_float[k];
                    texture_index_int[k] = (int)texture_index_float[k] + 1;
                }
            }
    
            int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
            for (int k = 0; k < 3; k++)
                new_pixel[k] += w * texture[isc * 3 + k];
            sampling_indices[pn] = isc;
            sampling_weights[pn] = w;
        }
        for (int k = 0; k < 3; k++)
            pixel[k] = new_pixel[k];
    }
}

std::vector<at::Tensor> forward_face_index_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor face_inv_map,
        at::Tensor lock,
        int image_size,
        float near,
        float far,
        int return_rgb,
        int return_alpha,
        int return_depth) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 1024;
    const dim3 blocks(batch_size * num_faces);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_cuda", ([&] {
      forward_face_index_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          face_inv_map.data<scalar_t>(),
          lock.data<int32_t>(),
          num_faces,
          image_size,
          (scalar_t) near,
          (scalar_t) far,
          return_rgb,
          return_alpha,
          return_depth);
      }));
    return {face_index_map, weight_map, depth_map, face_inv_map};
}

std::vector<at::Tensor> forward_texture_sampling_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor lock,
        int image_size,
        float eps) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(1);
    const int threads = 1024;
    const dim3 blocks(batch_size * image_size * image_size);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_texture_sampling_cuda", ([&] {
      forward_texture_sampling_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          rgb_map.data<scalar_t>(),
		  sampling_index_map.data<int32_t>(),
		  sampling_weight_map.data<scalar_t>(),
          lock.data<int32_t>(),
		  num_faces,
          image_size,
          texture_size,
          eps);
      }));
    return {};
}
