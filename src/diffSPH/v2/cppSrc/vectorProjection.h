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




template<std::size_t dim, typename scalar_t>
hostDeviceInline auto modDistance2(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return sum;
}
template<typename scalar_t>
hostDeviceInline auto moduloOp(const scalar_t p, const scalar_t q, const scalar_t h){
    return ((p - q + h / 2.0) - std::floor((p - q + h / 2.0) / h) * h) - h / 2.0;
}

struct Mat3x3{
    float m_00, m_01, m_02;
    float m_10, m_11, m_12;
    float m_20, m_21, m_22;
};

 struct Float3{
    float x;
    float y;
    float z;};
 struct Float2{
    float x;
    float y;};
 
hostDeviceInline void vectorProjectionKernel(int32_t index_i,
    cptr_t<float,2> vectors, cptr_t<float,2> normals, ptr_t<float,2> projectedOutput,
    
    cptr_t<int64_t,1> indices_j, cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset){
            int32_t numNeigh = numNeighbors[index_i];
            int32_t offset = neighborOffset[index_i];

            Float2 vector{vectors[index_i][0], vectors[index_i][1]};

            for (int j = 0; j < numNeigh; j++) {
                // Get the neighbor index
                int32_t index_j = indices_j[offset + j];
                Float2 normal{normals[index_j][0], normals[index_j][1]};

// dx_parallel_to_bdy = dx_fi - torch.sum(dx_fi * n_bj, dim = -1, keepdim = True) * n_bj
// dx_orthogonal_to_bdy = torch.sum(dx_fi * n_bj, dim = -1, keepdim = True) * n_bj

                auto dotProduct = vector.x * normal.x + vector.y * normal.y;

                Float2 parallel = {vector.x - dotProduct * normal.x, vector.y - dotProduct * normal.y};
                Float2 orthogonal = {dotProduct * normal.x, dotProduct * normal.y};

                vector.x = parallel.x;
                vector.y = parallel.y;
            }
            projectedOutput[index_i][0] = vector.x;
            projectedOutput[index_i][1] = vector.y;
    }

void vectorProjection_cuda(
    int32_t numParticles,
    torch::Tensor vectors, torch::Tensor normals, torch::Tensor projectedOutput,
    torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset);