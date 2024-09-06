#include "mDBC.h"

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

void mDBCDensity_cuda(
    int32_t numParticles,
    torch::Tensor masses_j, torch::Tensor densities_j, torch::Tensor positions_i, torch::Tensor positions_j,
    torch::Tensor indices_j, torch::Tensor kernels, torch::Tensor kernelGradients, torch::Tensor distances, torch::Tensor vectors, torch::Tensor supports,
    torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    torch::Tensor mDBCDensity, torch::Tensor shepardDensity, float rho0){
    auto masses_j_a = masses_j.packed_accessor32<float,1, traits>();
    auto densities_j_a = densities_j.packed_accessor32<float,1, traits>();
    auto positions_i_a = positions_i.packed_accessor32<float,2, traits>();
    auto positions_j_a = positions_j.packed_accessor32<float,2, traits>();
    auto indices_j_a = indices_j.packed_accessor32<int64_t,1, traits>();
    auto kernels_a = kernels.packed_accessor32<float,1, traits>();
    auto kernelGradients_a = kernelGradients.packed_accessor32<float,2, traits>();
    auto distances_a = distances.packed_accessor32<float,1, traits>();
    auto vectors_a = vectors.packed_accessor32<float,2, traits>();
    auto supports_a = supports.packed_accessor32<float,1, traits>();
    auto numNeighbors_a = numNeighbors.packed_accessor32<int32_t,1, traits>();
    auto neighborOffset_a = neighborOffset.packed_accessor32<int32_t,1, traits>();
    auto mDBCDensity_a = mDBCDensity.packed_accessor32<float,1, traits>();
    auto shepardDensity_a = shepardDensity.packed_accessor32<float,1, traits>();

    auto mDBCDensityFn = [=] __device__ (int32_t i){
        mDBCDensityKernel(i, masses_j_a, densities_j_a, positions_i_a, positions_j_a, indices_j_a, kernels_a, kernelGradients_a, distances_a, vectors_a, supports_a, numNeighbors_a, neighborOffset_a, mDBCDensity_a, shepardDensity_a, rho0);
    };

    launchKernel(mDBCDensityFn, numParticles);
    }