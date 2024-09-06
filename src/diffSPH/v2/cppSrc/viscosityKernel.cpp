#include "viscosityKernel.h"


 
at::Tensor viscosityKernel(
    std::pair<torch::Tensor, torch::Tensor> indices_,
    torch::Tensor supports_,
    torch::Tensor kernels_,
    torch::Tensor gradients_,
    torch::Tensor distances_,
    torch::Tensor vectors_,
    int32_t numParticles,
    torch::Tensor numNeighbors_,
    torch::Tensor neighborOffset_,

    std::pair<torch::Tensor, torch::Tensor> particle_velocities_,
    std::pair<torch::Tensor, torch::Tensor> particle_densities_,
    std::pair<torch::Tensor, torch::Tensor> particle_masses_,
    std::pair<torch::Tensor, torch::Tensor> particle_supports_,

    float cs, float rho0, float alpha, float eps, bool pi_switch) {
    bool useCuda = kernels_.is_cuda();
    auto defaultOptions = at::TensorOptions().device(kernels_.device());
    auto hostOptions = at::TensorOptions();
    
    // Get the pointers to the particle data
    auto velocities_i = getAccessor<float, 2>(particle_velocities_.first, "particle velocities (A)", useCuda);
    auto velocities_j = getAccessor<float, 2>(particle_velocities_.second, "particle velocities (B)", useCuda);
    auto masses_i = getAccessor<float, 1>(particle_masses_.first, "particle masses (A)", useCuda);
    auto masses_j = getAccessor<float, 1>(particle_masses_.second, "particle masses (B)", useCuda);
    auto densities_i = getAccessor<float, 1>(particle_densities_.first, "particle densities (A)", useCuda);
    auto densities_j = getAccessor<float, 1>(particle_densities_.second, "particle densities (B)", useCuda);
    auto supports_i = getAccessor<float, 1>(particle_supports_.first, "particle supports (A)", useCuda);
    auto supports_j = getAccessor<float, 1>(particle_supports_.second, "particle supports (B)", useCuda);
    
    // Get  the pointers to the neighborhood data
    auto indices_i = getAccessor<int64_t, 1>(indices_.first, "indices (A)", useCuda);
    auto indices_j = getAccessor<int64_t, 1>(indices_.second, "indices (B)", useCuda);
    auto supports = getAccessor<float, 1>(supports_, "supports", useCuda);
    auto kernels = getAccessor<float, 1>(kernels_, "kernels", useCuda);
    auto gradients = getAccessor<float, 2>(gradients_, "gradients", useCuda);
    auto distances = getAccessor<float, 1>(distances_, "distances", useCuda);
    auto vectors = getAccessor<float, 2>(vectors_, "vectors", useCuda);

    auto numNeighbors = getAccessor<int32_t, 1>(numNeighbors_, "numNeighbors", useCuda);
    auto neighborOffset = getAccessor<int32_t, 1>(neighborOffset_, "neighborOffset", useCuda);

    auto output_ = torch::zeros(particle_velocities_.first.sizes(), defaultOptions.dtype(torch::kFloat32));
    auto output = getAccessor<float, 2>(output_, "output", useCuda);


    if(useCuda){
// void viscosityKernel_cuda(int32_t numParticles,
// torch::Tensor velocities_i, torch::Tensor velocities_j,
// torch::Tensor masses_i, torch::Tensor masses_j,
// torch::Tensor densities_i, torch::Tensor densities_j,
// torch::Tensor supports_i, torch::Tensor supports_j,
// torch::Tensor indices_j, torch::Tensor supports, torch::Tensor kernels,
// torch::Tensor gradients, torch::Tensor distances, torch::Tensor vectors,
// torch::Tensor numNeighbors, torch::Tensor neighborOffset,
// torch::Tensor output, float cs, float rho0, float alpha, float eps, bool pi_switch);
        viscosityKernel_cuda(numParticles, 
        particle_velocities_.first, particle_velocities_.second,
        particle_masses_.first, particle_masses_.second,
        particle_densities_.first, particle_densities_.second,
        particle_supports_.first, particle_supports_.second,
        indices_.second, supports_, kernels_,
        gradients_, distances_, vectors_,
        numNeighbors_, neighborOffset_,
        output_, cs, rho0, alpha, eps, pi_switch);
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numParticles; index_i++) {
            viscosityFn(index_i, velocities_i, velocities_j, masses_i, masses_j, densities_i, densities_j, supports_i, supports_j, indices_j, supports, kernels, gradients, distances, vectors, numNeighbors, neighborOffset, output, cs, rho0, alpha, eps, pi_switch);
        }
    }
    return output_;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("viscosityKernel", &viscosityKernel, "SPH Density Interpolation");
}