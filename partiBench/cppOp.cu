#include "cppOp.h"


__global__ void sphOperation_Density_kernel(
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) {
        return;
    }

    sphOperation_Density(i, output, masses_A, masses_B, densities_A, densities_B, quantities_A, quantities_B, indices_i, indicies_j, kernels, numParticles, numNeighbors, neighborOffset);
}
__global__ void sphOperation_Interpolation_kernel(
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) {
        return;
    }
    extern __shared__ char dynamicSM[];
    float *outputBuffer = (double *)dynamicSM;

            // sphOperation_Interpolate(
            //     index_i,
            //     output_ptr,
            //     masses_i_ptr, masses_j_ptr,
            //     densities_i_ptr, densities_j_ptr,
            //     quantities_i_ptr, quantities_j_ptr,
            //     indices_i_ptr, indices_j_ptr,
            //     kernels_ptr,
            //     numParticles,
            //     numNeighbors_ptr, neighborOffset_ptr);
    sphOperation_Interpolate(i, output, masses_A, masses_B, densities_A, densities_B, quantities_A, quantities_B, indices_i, indicies_j, kernels, numParticles, numNeighbors, neighborOffset);
}


// // #ifdef CUUDA_VERSION
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
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
    // cuda_error_check();
    gridSize = (numParticles + blockSize - 1) / blockSize;

    kernel<<<gridSize, blockSize>>>(std::forward<Ts>(args)...);
    // cuda_error_check();
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
    int32_t* neighborOffset){
        launchKernel(sphOperation_Density_kernel, 
            numParticles, 
            output,
            masses_A, masses_B, 
            densities_A, densities_B, 
            quantities_A, quantities_B, 
            indices_i, indicies_j, 
            kernels, 
            numParticles, numNeighbors, neighborOffset);

    }
void sphOperation_Interpolation_cuda(
    float* output,
    float* masses_A, float* masses_B,
    float* densities_A, float* densities_B,
    float* quantities_A, float* quantities_B,
    
    int64_t *indices_i, int64_t *indicies_j,
    float* kernels,
    int32_t numParticles,
    int32_t* numNeighbors,
    int32_t* neighborOffset){
        launchKernel(sphOperation_Interpolation_kernel, 
            numParticles, 
            output,
            masses_A, masses_B, 
            densities_A, densities_B, 
            quantities_A, quantities_B, 
            indices_i, indicies_j, 
            kernels, 
            numParticles, numNeighbors, neighborOffset);
}