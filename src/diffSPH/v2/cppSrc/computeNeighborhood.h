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

hostDeviceInline void countingKernel(int32_t index_i,
    cptr_t<float,2> positions_i, cptr_t<float,1> supports_i, 
    cptr_t<float,2> positions_j, cptr_t<float,1> supports_j,
    cptr_t<float,1> minDomain, cptr_t<float,1> maxDomain, cptr_t<bool,1> periodicity,
    cptr_t<int64_t,1> indices_j, cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset,
    
    cptr_t<int32_t,1> output){
        float x_i = positions_i[index_i][0];
        float y_i = positions_i[index_i][1];
        float h_i = supports_i[index_i];

        int32_t numNeigh = numNeighbors[index_i];
        int32_t offset = neighborOffset[index_i];

        int32_t counter = 0;
        // Iterate over the neighbors
        for (int j = 0; j < numNeigh; j++) {
            int32_t index_j = indices_j[offset + j];
            auto x_j = positions_j[index_j][0];
            auto y_j = positions_j[index_j][1];
            auto h_j = supports_j[index_j];

            auto diff_x = periodicity[0] ? moduloOp(x_i, x_j, maxDomain[0] - minDomain[0]) : x_i - x_j;
            auto diff_y = periodicity[1] ? moduloOp(y_i, y_j, maxDomain[1] - minDomain[1]) : y_i - y_j;
            auto sum = diff_x * diff_x + diff_y * diff_y;
            if (sum <= h_j * h_j)
                counter++;

        }
        output[index_i] = counter;

    }

hostDeviceInline void updateKernel(int32_t index_i,
    cptr_t<float,2> positions_i, cptr_t<float,1> supports_i, 
    cptr_t<float,2> positions_j, cptr_t<float,1> supports_j,
    cptr_t<float,1> minDomain, cptr_t<float,1> maxDomain, cptr_t<bool,1> periodicity,
    cptr_t<int64_t,1> indices_j, cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset,
    
    cptr_t<int32_t, 1> newOffsets,
    cptr_t<int64_t,1> output_i, cptr_t<int64_t, 1> output_j, 
    cptr_t<float, 1> output_rij, cptr_t<float, 2> output_xij, cptr_t<float, 1> output_hij){
        float x_i = positions_i[index_i][0];
        float y_i = positions_i[index_i][1];
        float h_i = supports_i[index_i];

        int32_t numNeigh = numNeighbors[index_i];
        int32_t offset = neighborOffset[index_i];
        auto newOffset = newOffsets[index_i];

        int32_t counter = 0;
        // Iterate over the neighbors
        for (int j = 0; j < numNeigh; j++) {
            int32_t index_j = indices_j[offset + j];
            auto x_j = positions_j[index_j][0];
            auto y_j = positions_j[index_j][1];
            auto h_j = supports_j[index_j];

            auto diff_x = periodicity[0] ? moduloOp(x_i, x_j, maxDomain[0] - minDomain[0]) : x_i - x_j;
            auto diff_y = periodicity[1] ? moduloOp(y_i, y_j, maxDomain[1] - minDomain[1]) : y_i - y_j;
            auto sum = diff_x * diff_x + diff_y * diff_y;
            if (sum <= h_j * h_j){
                output_i[newOffset + counter] = index_i;
                output_j[newOffset + counter] = index_j;
                auto dist = std::sqrt(sum);
                output_rij[newOffset + counter] = dist / h_j;
                output_xij[newOffset + counter][0] = diff_x / (dist + 1e-6f * h_i);
                output_xij[newOffset + counter][1] = diff_y / (dist + 1e-6f * h_i);
                output_hij[newOffset + counter] = h_j;
                counter++;
            }
        }
    }


void countingKernel_cuda(int32_t numParticles,
    torch::Tensor positions_i, torch::Tensor supports_i, 
    torch::Tensor positions_j, torch::Tensor supports_j,
    torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    
    torch::Tensor output);
void updateKernel_cuda(int32_t numParticles,
    torch::Tensor positions_i, torch::Tensor supports_i, 
    torch::Tensor positions_j, torch::Tensor supports_j,
    torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    
    torch::Tensor newOffsets,
    torch::Tensor output_i, torch::Tensor output_j, 
    torch::Tensor output_rij, torch::Tensor output_xij, torch::Tensor output_hij);