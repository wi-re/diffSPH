#include "viscosityKernel.h"

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




void viscosityKernel_cuda(int32_t numParticles,
torch::Tensor velocities_i, torch::Tensor velocities_j,
torch::Tensor masses_i, torch::Tensor masses_j,
torch::Tensor densities_i, torch::Tensor densities_j,
torch::Tensor supports_i, torch::Tensor supports_j,
torch::Tensor indices_j, torch::Tensor supports, torch::Tensor kernels,
torch::Tensor gradients, torch::Tensor distances, torch::Tensor vectors,
torch::Tensor numNeighbors, torch::Tensor neighborOffset,
torch::Tensor output, float cs, float rho0, float alpha, float eps, bool pi_switch){

    auto velocities_i_a = velocities_i.packed_accessor32<float,2, traits>();
    auto velocities_j_a = velocities_j.packed_accessor32<float,2, traits>();
    auto masses_i_a = masses_i.packed_accessor32<float,1, traits>();
    auto masses_j_a = masses_j.packed_accessor32<float,1, traits>();
    auto densities_i_a = densities_i.packed_accessor32<float,1, traits>();
    auto densities_j_a = densities_j.packed_accessor32<float,1, traits>();
    auto supports_i_a = supports_i.packed_accessor32<float,1, traits>();
    auto supports_j_a = supports_j.packed_accessor32<float,1, traits>();
    auto indices_j_a = indices_j.packed_accessor32<int64_t,1, traits>();
    auto supports_a = supports.packed_accessor32<float,1, traits>();
    auto kernels_a = kernels.packed_accessor32<float,1, traits>();
    auto gradients_a = gradients.packed_accessor32<float,2, traits>();
    auto distances_a = distances.packed_accessor32<float,1, traits>();
    auto vectors_a = vectors.packed_accessor32<float,2, traits>();
    auto numNeighbors_a = numNeighbors.packed_accessor32<int32_t,1, traits>();
    auto neighborOffset_a = neighborOffset.packed_accessor32<int32_t,1, traits>();
    auto output_a = output.packed_accessor32<float,2, traits>();

    launchKernel([=] __device__(int32_t index_i){
        viscosityFn(index_i,
        velocities_i_a, velocities_j_a, masses_i_a, masses_j_a, densities_i_a, densities_j_a, supports_i_a, supports_j_a,
        indices_j_a, supports_a, kernels_a, gradients_a, distances_a, vectors_a, numNeighbors_a, neighborOffset_a, output_a, cs, rho0, alpha, eps, pi_switch);
    }, numParticles);
}