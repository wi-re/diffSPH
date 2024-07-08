#include "vectorProjection.h"
 
 
at::Tensor projectVectors(
    torch::Tensor vectors_,
    torch::Tensor normals_,
    
    int32_t numBoundaryParticles, int32_t numFluidParticles,
    std::pair<torch::Tensor, torch::Tensor> indices_,
    torch::Tensor numNeighbors_,
    torch::Tensor neighborOffset_){
    bool useCuda = numNeighbors_.is_cuda();
    auto defaultOptions = at::TensorOptions().device(numNeighbors_.device());
    auto hostOptions = at::TensorOptions();
    
    // Get the pointers to the particle data

    auto normals = getAccessor<float, 2>(normals_, "normals", useCuda);
    auto vectors = getAccessor<float, 2>(vectors_, "vectors", useCuda);

    // Get  the pointers to the neighborhood data
    auto indices_i = getAccessor<int64_t, 1>(indices_.first, "indices (A)", useCuda);
    auto indices_j = getAccessor<int64_t, 1>(indices_.second, "indices (B)", useCuda);
    auto numNeighbors = getAccessor<int32_t, 1>(numNeighbors_, "numNeighbors", useCuda);
    auto neighborOffset = getAccessor<int32_t, 1>(neighborOffset_, "neighborOffset", useCuda);

     
    auto projectedOutput_ = torch::zeros({numFluidParticles, 2}, defaultOptions.dtype(torch::kFloat32));
    auto projectedOutput = getAccessor<float, 2>(projectedOutput_, "output", useCuda);


        // return projectedOutput_;
    // return {mDBCDensity_, shepardDensity_};
    if(useCuda){
        vectorProjection_cuda(
            numFluidParticles,
            vectors_, normals_, projectedOutput_,
            indices_.second, numNeighbors_, neighborOffset_);
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numFluidParticles; index_i++) {
            vectorProjectionKernel(index_i, vectors, normals, projectedOutput, indices_j, numNeighbors, neighborOffset);
        }

    }
    return projectedOutput_;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("projectVectors", &projectVectors, "SPH Density Interpolation");
}