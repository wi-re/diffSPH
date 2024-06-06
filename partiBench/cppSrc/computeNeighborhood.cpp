#include "computeNeighborhood.h"

 

std::vector<at::Tensor> computeNeighborhood(
    std::pair<torch::Tensor, torch::Tensor> indices_,
    int32_t numParticles,
    torch::Tensor numNeighbors_,
    torch::Tensor neighborOffset_,

    std::pair<torch::Tensor, torch::Tensor> particle_positions_,
    std::pair<torch::Tensor, torch::Tensor> particle_supports_,
    
    torch::Tensor minDomain_, torch::Tensor maxDomain_,
    torch::Tensor periodicity_

    ) {
    bool useCuda = indices_.first.is_cuda();
    auto defaultOptions = at::TensorOptions().device(indices_.first.device());
    auto hostOptions = at::TensorOptions();
    
    // Get the pointers to the particle data
    auto positions_i = getAccessor<float, 2>(particle_positions_.first, "particle positions (A)", useCuda);
    auto positions_j = getAccessor<float, 2>(particle_positions_.second, "particle positions (B)", useCuda);
    auto supports_i = getAccessor<float, 1>(particle_supports_.first, "particle supports (A)", useCuda);
    auto supports_j = getAccessor<float, 1>(particle_supports_.second, "particle supports (B)", useCuda);
    
    // Get  the pointers to the neighborhood data
    auto indices_i = getAccessor<int64_t, 1>(indices_.first, "indices (A)", useCuda);
    auto indices_j = getAccessor<int64_t, 1>(indices_.second, "indices (B)", useCuda);
    auto numNeighbors = getAccessor<int32_t, 1>(numNeighbors_, "numNeighbors", useCuda);
    auto neighborOffset = getAccessor<int32_t, 1>(neighborOffset_, "neighborOffset", useCuda);
    
    auto minDomain = getAccessor<float, 1>(minDomain_, "minDomain", useCuda);
    auto maxDomain = getAccessor<float, 1>(maxDomain_, "maxDomain", useCuda);
    auto periodicity = getAccessor<bool, 1>(periodicity_, "periodicity", useCuda);


    auto output_ = torch::zeros(particle_supports_.first.sizes(), defaultOptions.dtype(torch::kInt32));
    auto output = getAccessor<int32_t, 1>(output_, "output", useCuda);
    // auto numParticles = particle_positions_.first.size(0);

    if(useCuda){
        countingKernel_cuda(numParticles, 
        particle_positions_.first, particle_supports_.first,
        particle_positions_.second, particle_supports_.second,
        minDomain_, maxDomain_, periodicity_, indices_.second, numNeighbors_, neighborOffset_, output_);
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numParticles; index_i++) {
            countingKernel(index_i, positions_i, supports_i, positions_j, supports_j, minDomain, maxDomain, periodicity, indices_j, numNeighbors, neighborOffset, output);
        }
    }

    auto cumsum_ = torch::hstack({torch::zeros(1, defaultOptions.dtype(torch::kInt32)), torch::cumsum(output_, 0, torch::kInt32)});
    auto lastEntry = cumsum_[-1].item().toInt();

    cumsum_ = cumsum_.slice(0, 0, cumsum_.size(0) - 1);
    auto newOffsets = getAccessor<int32_t, 1>(cumsum_, "cumsum", useCuda);
    // return {cumsum_};

    auto output_i_ = torch::zeros({lastEntry}, defaultOptions.dtype(torch::kInt64));
    auto output_j_ = torch::zeros({lastEntry}, defaultOptions.dtype(torch::kInt64));
    auto output_rij_ = torch::zeros({lastEntry}, defaultOptions.dtype(torch::kFloat32));
    auto output_xij_ = torch::zeros({lastEntry, 2}, defaultOptions.dtype(torch::kFloat32));
    auto output_hij_ = torch::zeros({lastEntry}, defaultOptions.dtype(torch::kFloat32));

    auto output_i = getAccessor<int64_t, 1>(output_i_, "output_i", useCuda);
    auto output_j = getAccessor<int64_t, 1>(output_j_, "output_j", useCuda);
    auto output_rij = getAccessor<float, 1>(output_rij_, "output_rij", useCuda);
    auto output_xij = getAccessor<float, 2>(output_xij_, "output_xij", useCuda);
    auto output_hij = getAccessor<float, 1>(output_hij_, "output_hij", useCuda);


    if(useCuda){
        updateKernel_cuda(numParticles, 
        particle_positions_.first, particle_supports_.first,
        particle_positions_.second, particle_supports_.second,
        minDomain_, maxDomain_, periodicity_, indices_.second, numNeighbors_, neighborOffset_, cumsum_, output_i_, output_j_, output_rij_, output_xij_, output_hij_);
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numParticles; index_i++) {
            updateKernel(index_i, positions_i, supports_i, positions_j, supports_j, minDomain, maxDomain, periodicity, indices_j, numNeighbors, neighborOffset, newOffsets, output_i, output_j, output_rij, output_xij, output_hij);
        }
    }


    return {output_, cumsum_, output_i_, output_j_, output_rij_, output_xij_, output_hij_};
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("computeNeighborhood", &computeNeighborhood, "SPH Density Interpolation");
}