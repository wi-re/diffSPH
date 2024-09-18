#include "computeNeighborhood.h"

template<typename Func, typename... Ts>
__global__ void kernelWrapper(Func kernel, int32_t numThreads, Ts&&... args) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numThreads)
        return;
    kernel(i, std::forward<Ts>(args)...);
}



// // // #ifdef CUUDA_VERSION
// #ifdef CUDA_VERSION
#include <cuda_runtime.h>

void cuda_error_check() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
// #endif

template<typename Func, typename... Ts>
void launchKernel(Func kernel, int32_t numParticles, Ts&&... args) {
    int32_t blockSize;  // Number of threads per block
    int32_t minGridSize;  // Minimum number of blocks required for the kernel
    int32_t gridSize;  // Number of blocks to use

    // Compute the maximum potential block size for the kernel
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelWrapper<Func, Ts...>);
    // cuda_error_check();
    gridSize = (numParticles + blockSize - 1) / blockSize;

    kernelWrapper<<<gridSize, blockSize>>>(kernel, numParticles, std::forward<Ts>(args)...);
    // cuda_error_check();
}



void countingKernel_cuda(int32_t numParticles,
    torch::Tensor positions_i, torch::Tensor supports_i, 
    torch::Tensor positions_j, torch::Tensor supports_j,
    torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    
    torch::Tensor output){
        auto positions_i_a = positions_i.packed_accessor32<float,2, traits>();
        auto supports_i_a = supports_i.packed_accessor32<float,1, traits>();
        auto positions_j_a = positions_j.packed_accessor32<float,2, traits>();
        auto supports_j_a = supports_j.packed_accessor32<float,1, traits>();
        auto minDomain_a = minDomain.packed_accessor32<float,1, traits>();
        auto maxDomain_a = maxDomain.packed_accessor32<float,1, traits>();
        auto periodicity_a = periodicity.packed_accessor32<bool,1, traits>();

        auto indices_j_a = indices_j.packed_accessor32<int64_t,1, traits>();
        auto numNeighbors_a = numNeighbors.packed_accessor32<int32_t,1, traits>();
        auto neighborOffset_a = neighborOffset.packed_accessor32<int32_t,1, traits>();

        auto output_a = output.packed_accessor32<int32_t,1, traits>();


    launchKernel([=] __device__(int32_t index_i){
        countingKernel(index_i,
        positions_i_a, supports_i_a, positions_j_a, supports_j_a, minDomain_a, maxDomain_a, periodicity_a,
        indices_j_a, numNeighbors_a, neighborOffset_a, output_a);
    }, numParticles);
    }
void updateKernel_cuda(int32_t numParticles,
    torch::Tensor positions_i, torch::Tensor supports_i, 
    torch::Tensor positions_j, torch::Tensor supports_j,
    torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    
    torch::Tensor newOffsets,
    torch::Tensor output_i, torch::Tensor output_j, 
    torch::Tensor output_rij, torch::Tensor output_xij, torch::Tensor output_hij){
        auto positions_i_a = positions_i.packed_accessor32<float,2, traits>();
        auto supports_i_a = supports_i.packed_accessor32<float,1, traits>();
        auto positions_j_a = positions_j.packed_accessor32<float,2, traits>();
        auto supports_j_a = supports_j.packed_accessor32<float,1, traits>();
        auto minDomain_a = minDomain.packed_accessor32<float,1, traits>();
        auto maxDomain_a = maxDomain.packed_accessor32<float,1, traits>();
        auto periodicity_a = periodicity.packed_accessor32<bool,1, traits>();

        auto indices_j_a = indices_j.packed_accessor32<int64_t,1, traits>();
        auto numNeighbors_a = numNeighbors.packed_accessor32<int32_t,1, traits>();
        auto neighborOffset_a = neighborOffset.packed_accessor32<int32_t,1, traits>();

        auto newOffsets_a = newOffsets.packed_accessor32<int32_t,1, traits>();

        auto output_i_a = output_i.packed_accessor32<int64_t,1, traits>();
        auto output_j_a = output_j.packed_accessor32<int64_t,1, traits>();
        auto output_rij_a = output_rij.packed_accessor32<float,1, traits>();
        auto output_xij_a = output_xij.packed_accessor32<float,2, traits>();
        auto output_hij_a = output_hij.packed_accessor32<float,1, traits>();

    launchKernel([=] __device__(int32_t index_i){
        updateKernel(index_i,
        positions_i_a, supports_i_a, positions_j_a, supports_j_a, minDomain_a, maxDomain_a, periodicity_a,
        indices_j_a, numNeighbors_a, neighborOffset_a,
        newOffsets_a,
        output_i_a, output_j_a, output_rij_a, output_xij_a, output_hij_a);
    }, numParticles);
    }