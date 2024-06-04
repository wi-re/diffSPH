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
#define hostDeviceInline __device__ __host__ inline
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


hostDeviceInline void sphOperation_Density(
    int32_t i,
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset){
        float qInterpolated = 0.f;

        int32_t numNeigh = numNeighbors[i];
        int32_t offset = neighborOffset[i];

        // Iterate over the neighbors
        for (int j = 0; j < numNeigh; j++) {
            // Get the neighbor index
            int32_t index_j = indicies_j[offset + j];
            // Get the mass of the neighbor
            float mass_j = masses_B[index_j];
            // Get the kernel value
            float kernel = kernels[offset + j];

            // Compute the density contribution
            qInterpolated += mass_j * kernel;
        }

        output[i] = qInterpolated;
    }

hostDeviceInline void sphOperation_Interpolate(
    int32_t i,
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset){
        float qInterpolated = 0.f;

        int32_t numNeigh = numNeighbors[i];
        int32_t offset = neighborOffset[i];

        // Iterate over the neighbors
        for (int j = 0; j < numNeigh; j++) {
            // Get the neighbor index
            int32_t index_j = indicies_j[offset + j];
            // Get the mass of the neighbor
            float mass_j = masses_B[index_j];
            // Get the density of the neighbor
            float density_j = densities_B[index_j];
            // Get the quantity of the neighbor
            float quantity_j = quantities_B[index_j];
            // Get the kernel value
            float kernel = kernels[offset + j];

            // Compute the density contribution
            qInterpolated += mass_j / density_j * quantity_j * kernel;
        }

        output[i] = qInterpolated;
    }

void sphOperation_Density_cuda(
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset);
void sphOperation_Interpolation_cuda(
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset);