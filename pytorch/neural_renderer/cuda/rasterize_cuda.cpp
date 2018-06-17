#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

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
        int return_depth);

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
        float eps);

// C++ interface

std::vector<at::Tensor> forward_face_index_map(
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

    return forward_face_index_map_cuda(faces, face_index_map, weight_map,
                                       depth_map, face_inv_map, lock,
                                       image_size, near, far,
                                       return_rgb, return_alpha, return_depth);
}

std::vector<at::Tensor> forward_texture_sampling(
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
        int texture_size,
        float eps) {

    return forward_texture_sampling_cuda(faces, textures, face_index_map,
                                    weight_map, depth_map, rgb_map,
                                    sampling_index_map, sampling_weight_map, lock,
                                    image_size, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_face_index_map", &forward_face_index_map, "FORWARD_FACE_INDEX_MAP (CUDA)");
    // m.def("forward_texture_sampling", &forward_texture_sampling_cuda, "FORWARD_TEXTURE_SAMPLING (CUDA)");
}
