#include "cppOp.h"



at::Tensor sphDensity(
    std::pair<torch::Tensor, torch::Tensor> masses,
    std::pair<torch::Tensor, torch::Tensor> densities,
    std::pair<torch::Tensor, torch::Tensor> quantities,


    std::pair<torch::Tensor, torch::Tensor> indices,
    torch::Tensor kernels,

    int32_t numParticles,
    torch::Tensor numNeighbors,
    torch::Tensor neighborOffset) {

    // Get the pointers to the data
    auto masses_i_ptr = masses.first.data_ptr<float>();
    auto masses_j_ptr = masses.second.data_ptr<float>();
    auto densities_i_ptr = densities.first.data_ptr<float>();
    auto densities_j_ptr = densities.second.data_ptr<float>();
    auto quantities_i_ptr = quantities.first.data_ptr<float>();
    auto quantities_j_ptr = quantities.second.data_ptr<float>();

    auto indices_i_ptr = indices.first.data_ptr<int64_t>();
    auto indices_j_ptr = indices.second.data_ptr<int64_t>();
    auto kernels_ptr = kernels.data_ptr<float>();

    auto numNeighbors_ptr = numNeighbors.data_ptr<int32_t>();
    auto neighborOffset_ptr = neighborOffset.data_ptr<int32_t>();

    // Create the output tensor
    auto defaultOptions = at::TensorOptions().device(masses.first.device());
    auto hostOptions = at::TensorOptions();

    auto output = torch::zeros({numParticles}, defaultOptions.dtype(torch::kFloat32));
    auto output_ptr = output.data_ptr<float>();

    // Create the output tensor
    // auto output = torch::zeros({numParticles}, torch::dtype(torch::kFloat32));
    // auto output_ptr = output.data_ptr<float>();

    if(masses.first.is_cuda()){
        sphOperation_Density_cuda(
            output_ptr,
            masses_i_ptr, masses_j_ptr,
            densities_i_ptr, densities_j_ptr,
            quantities_i_ptr, quantities_j_ptr,
            indices_i_ptr, indices_j_ptr,
            kernels_ptr,
            numParticles,
            numNeighbors_ptr, neighborOffset_ptr);
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numParticles; index_i++) {
            sphOperation_Density(
                index_i,
                output_ptr,
                masses_i_ptr, masses_j_ptr,
                densities_i_ptr, densities_j_ptr,
                quantities_i_ptr, quantities_j_ptr,
                indices_i_ptr, indices_j_ptr,
                kernels_ptr,
                numParticles,
                numNeighbors_ptr, neighborOffset_ptr);

            // // Get the number of neighbors
            // int32_t numNeigh = numNeighbors_ptr[index_i];
            // // Get the neighbor offset
            // int32_t offset = neighborOffset_ptr[index_i];

            // // Get the mass of the particle
            // float mass_i = masses_i_ptr[index_i];
            // // Get the density of the particle
            // // float density_i = densities_i_ptr[index_i];
            // // Get the quantity of the particle
            // // float quantity_i = quantities_i_ptr[index_i];

            // // Initialize the density
            // float density = 0.0f;
            // // Iterate over the neighbors
            // for (int j = 0; j < numNeigh; j++) {
            //     // Get the neighbor index
            //     int32_t index_j = indices_j_ptr[offset + j];
            //     // Get the mass of the neighbor
            //     float mass_j = masses_j_ptr[index_j];
            //     // Get the density of the neighbor
            //     // float density_j = densities_j_ptr[index_j];
            //     // Get the quantity of the neighbor
            //     // float quantity_j = quantities_j_ptr[index_j];
            //     // Get the kernel value
            //     float kernel = kernels_ptr[offset + j];

            //     // Compute the density contribution
            //     density += mass_j * kernel;
            // }

            // // Compute the density
            // output_ptr[index_i] = density;
        }
    }
    return output;
}



at::Tensor sphInterpolation(
    std::pair<torch::Tensor, torch::Tensor> masses,
    std::pair<torch::Tensor, torch::Tensor> densities,
    std::pair<torch::Tensor, torch::Tensor> quantities,


    std::pair<torch::Tensor, torch::Tensor> indices,
    torch::Tensor kernels,

    int32_t numParticles,
    torch::Tensor numNeighbors,
    torch::Tensor neighborOffset) {

    // Get the pointers to the data
    auto masses_i_ptr = masses.first.data_ptr<float>();
    auto masses_j_ptr = masses.second.data_ptr<float>();
    auto densities_i_ptr = densities.first.data_ptr<float>();
    auto densities_j_ptr = densities.second.data_ptr<float>();
    auto quantities_i_ptr = quantities.first.data_ptr<float>();
    auto quantities_j_ptr = quantities.second.data_ptr<float>();

    auto indices_i_ptr = indices.first.data_ptr<int64_t>();
    auto indices_j_ptr = indices.second.data_ptr<int64_t>();
    auto kernels_ptr = kernels.data_ptr<float>();

    auto numNeighbors_ptr = numNeighbors.data_ptr<int32_t>();
    auto neighborOffset_ptr = neighborOffset.data_ptr<int32_t>();

    // Create the output tensor
    auto defaultOptions = at::TensorOptions().device(masses.first.device());
    auto hostOptions = at::TensorOptions();


    auto quantities_shape = quantities.first.sizes();
    auto numEntries = quantities_shape[0];
    auto entriesPerElement = 1;

    if (quantities_shape.size() > 1) {
        entriesPerElement *= quantities_shape[1];
    }

    auto output_shape = std::vector<int64_t>{numParticles};
    for (int i = 1; i < quantities_shape.size(); i++) {
        output_shape.push_back(quantities_shape[i]);
    }
    
    auto output = torch::zeros(output_shape, defaultOptions.dtype(torch::kFloat32));
    auto output_ptr = output.data_ptr<float>();


    if(masses.first.is_cuda()){
        sphOperation_Density_cuda(
            output_ptr,
            masses_i_ptr, masses_j_ptr,
            densities_i_ptr, densities_j_ptr,
            quantities_i_ptr, quantities_j_ptr,
            indices_i_ptr, indices_j_ptr,
            kernels_ptr,
            numParticles,
            numNeighbors_ptr, neighborOffset_ptr);
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numParticles; index_i++) {
            sphOperation_Interpolate(
                index_i,
                output_ptr,
                masses_i_ptr, masses_j_ptr,
                densities_i_ptr, densities_j_ptr,
                quantities_i_ptr, quantities_j_ptr,
                indices_i_ptr, indices_j_ptr,
                kernels_ptr,
                numParticles,
                numNeighbors_ptr, neighborOffset_ptr);

            // // Get the number of neighbors
            // int32_t numNeigh = numNeighbors_ptr[index_i];
            // // Get the neighbor offset
            // int32_t offset = neighborOffset_ptr[index_i];

            // // Get the mass of the particle
            // float mass_i = masses_i_ptr[index_i];
            // // Get the density of the particle
            // float density_i = densities_i_ptr[index_i];
            // // Get the quantity of the particle
            // float quantity_i = quantities_i_ptr[index_i];

            // // Initialize the density
            // float density = 0.0f;
            // // Iterate over the neighbors
            // for (int j = 0; j < numNeigh; j++) {
            //     // Get the neighbor index
            //     int32_t index_j = indices_j_ptr[offset + j];
            //     // Get the mass of the neighbor
            //     float mass_j = masses_j_ptr[index_j];
            //     // Get the density of the neighbor
            //     float density_j = densities_j_ptr[index_j];
            //     // Get the quantity of the neighbor
            //     float quantity_j = quantities_j_ptr[index_j];
            //     // Get the kernel value
            //     float kernel = kernels_ptr[offset + j];

            //     // Compute the density contribution
            //     density += mass_j / density_j * quantity_j * kernel;
            // }

            // // Compute the density
            // output_ptr[index_i] = density;
        }
    }
    return output;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sphDensity", &sphDensity, "SPH Density Interpolation");
  m.def("sphInterpolation", &sphInterpolation, "SPH Interpolation");
}