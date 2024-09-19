#include "vectorProjection.h"

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

void vectorProjection_cuda(
    int32_t numParticles,
    torch::Tensor vectors, torch::Tensor normals, torch::Tensor projectedOutput,
    torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset){

        auto vectors_a = vectors.packed_accessor32<float,2, traits>();
        auto normals_a = normals.packed_accessor32<float,2, traits>();
        auto projectedOutput_a = projectedOutput.packed_accessor32<float,2, traits>();
        auto indices_j_a = indices_j.packed_accessor32<int64_t,1, traits>();
        auto numNeighbors_a = numNeighbors.packed_accessor32<int32_t,1, traits>();
        auto neighborOffset_a = neighborOffset.packed_accessor32<int32_t,1, traits>();

    auto vectorProjectionKernelFn = [=] __device__ (int32_t i){
        vectorProjectionKernel(i, vectors_a, normals_a, projectedOutput_a, indices_j_a, numNeighbors_a, neighborOffset_a);
    };

    launchKernel(vectorProjectionKernelFn, numParticles);
    }