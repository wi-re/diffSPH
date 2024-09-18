#pragma once
// #define __USE_ISOC11 1
// #include <time.h>
#ifdef __INTELLISENSE__
#define OMP_VERSION
#endif

// #define _OPENMP
#include <algorithm>
#ifdef OMP_VERSION
#include <omp.h>
// #include <ATen/ParallelOpenMP.h>
#endif
#ifdef TBB_VERSION
#include <ATen/ParallelNativeTBB.h>
#endif
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/core/TensorAccessor.h>


#if defined(__CUDACC__) || defined(__HIPCC__)
#define hostDeviceInline __device__ inline
#else
#define hostDeviceInline inline
#endif

// Define the traits for the pointer types based on the CUDA availability
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename T>
using traits = torch::RestrictPtrTraits<T>;
#else
template<typename T>
using traits = torch::DefaultPtrTraits<T>;
#endif

// Define tensor accessor aliases for different cases, primiarly use ptr_t when possible
template<typename T, std::size_t dim>
using ptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using cptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using tensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using ctensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using general_t = torch::TensorAccessor<T, dim>;


#include <torch/extension.h>
/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print32_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim, traits>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    try{
        return t.template packed_accessor32<scalar_t, dim, traits>();
    } catch(const c10::Error& e){
        throw std::runtime_error(name + " is not of the correct type " + e.what());
    }
}

hostDeviceInline void viscosityFn(int32_t index_i,
cptr_t<float,2> velocities_i, cptr_t<float,2> velocities_j,
cptr_t<float,1> masses_i, cptr_t<float,1> masses_j,
cptr_t<float,1> densities_i, cptr_t<float,1> densities_j,
cptr_t<float,1> supports_i, cptr_t<float,1> supports_j,
cptr_t<int64_t,1> indices_j, cptr_t<float,1> supports, cptr_t<float,1> kernels,
cptr_t<float,2> gradients, cptr_t<float,1> distances, cptr_t<float,2> vectors,
cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset,
cptr_t<float,2> output, float cs, float rho0, float alpha, float eps, bool pi_switch){
    float result_x = 0.f;
    float result_y = 0.f;

    int32_t numNeigh = numNeighbors[index_i];
    int32_t offset = neighborOffset[index_i];

    float velocity_x = velocities_i[index_i][0];
    float velocity_y = velocities_i[index_i][1];

    float density_i = densities_i[index_i];
    float support_i = supports_i[index_i];

    // Iterate over the neighbors
    for (int j = 0; j < numNeigh; j++) {
        // Get the neighbor index
        int32_t index_j = indices_j[offset + j];
        auto h_ij = supports[offset + j];
        auto v_ij_x = velocity_x - velocities_j[index_j][0];
        auto v_ij_y = velocity_y - velocities_j[index_j][1];
        auto x_ij_x = vectors[offset + j][0];
        auto x_ij_y = vectors[offset + j][1];
        auto r_ij = distances[offset + j] * h_ij;

        auto vr_ij = v_ij_x * x_ij_x + v_ij_y * x_ij_y;
        auto pi_ij = vr_ij / (r_ij + eps + h_ij * h_ij);
        if (pi_switch)
            pi_ij = vr_ij < 0 ? pi_ij : 0;

        auto V_j = masses_j[index_j] / (density_i + densities_j[index_j]);

        result_x += V_j * pi_ij * gradients[offset + j][0];
        result_y += V_j * pi_ij * gradients[offset + j][1];
    }

    auto factor = alpha * support_i * cs * rho0 / density_i;

    output[index_i][0] = result_x * factor;
    output[index_i][1] = result_y * factor;
    }

void viscosityKernel_cuda(int32_t numParticles,
torch::Tensor velocities_i, torch::Tensor velocities_j,
torch::Tensor masses_i, torch::Tensor masses_j,
torch::Tensor densities_i, torch::Tensor densities_j,
torch::Tensor supports_i, torch::Tensor supports_j,
torch::Tensor indices_j, torch::Tensor supports, torch::Tensor kernels,
torch::Tensor gradients, torch::Tensor distances, torch::Tensor vectors,
torch::Tensor numNeighbors, torch::Tensor neighborOffset,
torch::Tensor output, float cs, float rho0, float alpha, float eps, bool pi_switch);