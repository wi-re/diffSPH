#include "cppOp.h"



template<typename scalar_t = float, std::size_t qDim = 1>
at::Tensor sphInterpolation_t(
    std::pair<torch::Tensor, torch::Tensor> masses,
    std::pair<torch::Tensor, torch::Tensor> densities,
    std::pair<torch::Tensor, torch::Tensor> quantities,

    std::pair<torch::Tensor, torch::Tensor> indices,
    torch::Tensor kernels_,

    int32_t numParticles,
    torch::Tensor numNeighbors_,
    torch::Tensor neighborOffset_
){
    bool verbose = false;
    bool useCuda = kernels_.is_cuda();        

    // Get the pointers to the data
    auto masses_b = getAccessor<scalar_t, 1>(masses.second, "masses_B", useCuda, verbose);
    auto densities_b = getAccessor<scalar_t, 1>(densities.second, "densities_B", useCuda, verbose);

    auto quantities_b = getAccessor<scalar_t, qDim>(quantities.second, "quantities_B", useCuda, verbose);

    auto indices_i = getAccessor<int64_t, 1>(indices.first, "indices_i", useCuda, verbose);
    auto indices_j = getAccessor<int64_t, 1>(indices.second, "indices_j", useCuda, verbose);
    auto kernels = getAccessor<scalar_t, 1>(kernels_, "kernels", useCuda, verbose);

    auto numNeighbors = getAccessor<int32_t, 1>(numNeighbors_, "numNeighbors", useCuda, verbose);
    auto neighborOffset = getAccessor<int32_t, 1>(neighborOffset_, "neighborOffset", useCuda, verbose);

    // Create the output tensor
    auto defaultOptions = at::TensorOptions().device(masses.first.device());
    auto hostOptions = at::TensorOptions();

    auto quantities_shape = quantities.first.sizes();
    auto numEntries = quantities_shape[0];
    auto entriesPerElement = 1;

    if (quantities_shape.size() > 1) {
        entriesPerElement *= quantities_shape[1];
    }
    auto masses_i_ptr = masses.first.data_ptr<float>();
    auto masses_j_ptr = masses.second.data_ptr<float>();
    auto densities_i_ptr = densities.first.data_ptr<float>();
    auto densities_j_ptr = densities.second.data_ptr<float>();
    auto quantities_i_ptr = quantities.first.data_ptr<float>();
    auto quantities_j_ptr = quantities.second.data_ptr<float>();

    auto indices_i_ptr = indices.first.data_ptr<int64_t>();
    auto indices_j_ptr = indices.second.data_ptr<int64_t>();
    auto kernels_ptr = kernels_.data_ptr<float>();

    auto numNeighbors_ptr = numNeighbors_.data_ptr<int32_t>();
    auto neighborOffset_ptr = neighborOffset_.data_ptr<int32_t>();

    auto output_shape = std::vector<int64_t>{numParticles};
    std::vector<int64_t> tensor_shape;
    for (int i = 1; i < quantities_shape.size(); i++) {
        output_shape.push_back(quantities_shape[i]);
        tensor_shape.push_back(quantities_shape[i]);
    }

    auto output = torch::zeros(output_shape, defaultOptions.dtype(quantities.first.dtype()));

    auto output_accessor = getAccessor<scalar_t, qDim>(output, "output", useCuda, verbose);

    if(masses.first.is_cuda()){
        sphOperation_Interpolation_cuda(
            output.data_ptr<float>(),
            masses_i_ptr, masses_j_ptr,
            densities_i_ptr, densities_j_ptr,
            quantities_i_ptr, quantities_j_ptr,
            indices_i_ptr, indices_j_ptr,
            kernels_ptr,
            numParticles,
            numNeighbors_ptr, neighborOffset_ptr);
    }
    else{
        #pragma omp parallel for
        for (int index_i = 0; index_i < numParticles; index_i++) {
            if constexpr (qDim == 1) {
                float qInterpolated = 0.f;

                int32_t numNeigh = numNeighbors[index_i];
                int32_t offset = neighborOffset[index_i];

                // Iterate over the neighbors
                for (int j = 0; j < numNeigh; j++) {
                    // Get the neighbor index
                    int32_t index_j = indices_j[offset + j];
                    // Get the mass of the neighbor
                    scalar_t mass_j = masses_b[index_j];
                    // Get the kernel value
                    scalar_t kernel = kernels[offset + j];

                    // Compute the density contribution
                    qInterpolated += mass_j * kernel;
                }

                output_accessor[index_i] = qInterpolated;
            }else{
            }
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
        torch::Tensor output;
        if(quantities.first.dim() == 1)
            AT_DISPATCH_FLOATING_TYPES(quantities.first.scalar_type(), "sphInterpolation", ([&] {
                output = sphInterpolation_t<scalar_t, 1>(
                    masses, densities, quantities, indices, kernels, numParticles, numNeighbors, neighborOffset);
            }));
        return output;

    }



// at::Tensor sphDensity(
//     std::pair<torch::Tensor, torch::Tensor> masses,
//     std::pair<torch::Tensor, torch::Tensor> densities,
//     std::pair<torch::Tensor, torch::Tensor> quantities,


//     std::pair<torch::Tensor, torch::Tensor> indices,
//     torch::Tensor kernels,

//     int32_t numParticles,
//     torch::Tensor numNeighbors,
//     torch::Tensor neighborOffset) {

//         bool verbose = false;
//         bool useCuda = kernels.is_cuda();        

//     // Get the pointers to the data
//     auto masses_i_ptr = masses.first.data_ptr<float>();
//     auto masses_j_ptr = masses.second.data_ptr<float>();
//     auto densities_i_ptr = densities.first.data_ptr<float>();
//     auto densities_j_ptr = densities.second.data_ptr<float>();

//     ptr_t<float, 1> quantities_i_accessor = getAccessor<float, 1>(quantities.first, "queryPositions", useCuda, verbose);

//     auto quantities_i_ptr = quantities.first.data_ptr<float>();
//     auto quantities_j_ptr = quantities.second.data_ptr<float>();

//     auto indices_i_ptr = indices.first.data_ptr<int64_t>();
//     auto indices_j_ptr = indices.second.data_ptr<int64_t>();
//     auto kernels_ptr = kernels.data_ptr<float>();

//     auto numNeighbors_ptr = numNeighbors.data_ptr<int32_t>();
//     auto neighborOffset_ptr = neighborOffset.data_ptr<int32_t>();

//     // Create the output tensor
//     auto defaultOptions = at::TensorOptions().device(masses.first.device());
//     auto hostOptions = at::TensorOptions();

//     auto output = torch::zeros({numParticles}, defaultOptions.dtype(torch::kFloat32));
//     auto output_ptr = output.data_ptr<float>();

//     // Create the output tensor
//     // auto output = torch::zeros({numParticles}, torch::dtype(torch::kFloat32));
//     // auto output_ptr = output.data_ptr<float>();

//     if(masses.first.is_cuda()){
//         sphOperation_Density_cuda(
//             output_ptr,
//             masses_i_ptr, masses_j_ptr,
//             densities_i_ptr, densities_j_ptr,
//             quantities_i_ptr, quantities_j_ptr,
//             indices_i_ptr, indices_j_ptr,
//             kernels_ptr,
//             numParticles,
//             numNeighbors_ptr, neighborOffset_ptr);
//     }
//     else{
//         // Iterate over the particles
//         #pragma omp parallel for
//         for (int index_i = 0; index_i < numParticles; index_i++) {
//             sphOperation_Density(
//                 index_i,
//                 output_ptr,
//                 masses_i_ptr, masses_j_ptr,
//                 densities_i_ptr, densities_j_ptr,
//                 quantities_i_ptr, quantities_j_ptr,
//                 indices_i_ptr, indices_j_ptr,
//                 kernels_ptr,
//                 numParticles,
//                 numNeighbors_ptr, neighborOffset_ptr);

//             // // Get the number of neighbors
//             // int32_t numNeigh = numNeighbors_ptr[index_i];
//             // // Get the neighbor offset
//             // int32_t offset = neighborOffset_ptr[index_i];

//             // // Get the mass of the particle
//             // float mass_i = masses_i_ptr[index_i];
//             // // Get the density of the particle
//             // // float density_i = densities_i_ptr[index_i];
//             // // Get the quantity of the particle
//             // // float quantity_i = quantities_i_ptr[index_i];

//             // // Initialize the density
//             // float density = 0.0f;
//             // // Iterate over the neighbors
//             // for (int j = 0; j < numNeigh; j++) {
//             //     // Get the neighbor index
//             //     int32_t index_j = indices_j_ptr[offset + j];
//             //     // Get the mass of the neighbor
//             //     float mass_j = masses_j_ptr[index_j];
//             //     // Get the density of the neighbor
//             //     // float density_j = densities_j_ptr[index_j];
//             //     // Get the quantity of the neighbor
//             //     // float quantity_j = quantities_j_ptr[index_j];
//             //     // Get the kernel value
//             //     float kernel = kernels_ptr[offset + j];

//             //     // Compute the density contribution
//             //     density += mass_j * kernel;
//             // }

//             // // Compute the density
//             // output_ptr[index_i] = density;
//         }
//     }
//     return output;
// }



// at::Tensor sphInterpolation(
//     std::pair<torch::Tensor, torch::Tensor> masses,
//     std::pair<torch::Tensor, torch::Tensor> densities,
//     std::pair<torch::Tensor, torch::Tensor> quantities,


//     std::pair<torch::Tensor, torch::Tensor> indices,
//     torch::Tensor kernels,

//     int32_t numParticles,
//     torch::Tensor numNeighbors,
//     torch::Tensor neighborOffset) {

//     // Get the pointers to the data
//     auto masses_i_ptr = masses.first.data_ptr<float>();
//     auto masses_j_ptr = masses.second.data_ptr<float>();
//     auto densities_i_ptr = densities.first.data_ptr<float>();
//     auto densities_j_ptr = densities.second.data_ptr<float>();
//     auto quantities_i_ptr = quantities.first.data_ptr<float>();
//     auto quantities_j_ptr = quantities.second.data_ptr<float>();

//     auto indices_i_ptr = indices.first.data_ptr<int64_t>();
//     auto indices_j_ptr = indices.second.data_ptr<int64_t>();
//     auto kernels_ptr = kernels.data_ptr<float>();

//     auto numNeighbors_ptr = numNeighbors.data_ptr<int32_t>();
//     auto neighborOffset_ptr = neighborOffset.data_ptr<int32_t>();

//     // Create the output tensor
//     auto defaultOptions = at::TensorOptions().device(masses.first.device());
//     auto hostOptions = at::TensorOptions();


//     auto quantities_shape = quantities.first.sizes();
//     auto numEntries = quantities_shape[0];
//     auto entriesPerElement = 1;

//     if (quantities_shape.size() > 1) {
//         entriesPerElement *= quantities_shape[1];
//     }

//     auto output_shape = std::vector<int64_t>{numParticles};
//     for (int i = 1; i < quantities_shape.size(); i++) {
//         output_shape.push_back(quantities_shape[i]);
//     }
    
//     auto output = torch::zeros(output_shape, defaultOptions.dtype(torch::kFloat32));
//     auto output_ptr = output.data_ptr<float>();


//     if(masses.first.is_cuda()){
//         sphOperation_Density_cuda(
//             output_ptr,
//             masses_i_ptr, masses_j_ptr,
//             densities_i_ptr, densities_j_ptr,
//             quantities_i_ptr, quantities_j_ptr,
//             indices_i_ptr, indices_j_ptr,
//             kernels_ptr,
//             numParticles,
//             numNeighbors_ptr, neighborOffset_ptr);
//     }
//     else{
//         // Iterate over the particles
//         #pragma omp parallel for
//         for (int index_i = 0; index_i < numParticles; index_i++) {
//             sphOperation_Interpolate(
//                 index_i,
//                 output_ptr,
//                 masses_i_ptr, masses_j_ptr,
//                 densities_i_ptr, densities_j_ptr,
//                 quantities_i_ptr, quantities_j_ptr,
//                 indices_i_ptr, indices_j_ptr,
//                 kernels_ptr,
//                 numParticles,
//                 numNeighbors_ptr, neighborOffset_ptr);

//             // // Get the number of neighbors
//             // int32_t numNeigh = numNeighbors_ptr[index_i];
//             // // Get the neighbor offset
//             // int32_t offset = neighborOffset_ptr[index_i];

//             // // Get the mass of the particle
//             // float mass_i = masses_i_ptr[index_i];
//             // // Get the density of the particle
//             // float density_i = densities_i_ptr[index_i];
//             // // Get the quantity of the particle
//             // float quantity_i = quantities_i_ptr[index_i];

//             // // Initialize the density
//             // float density = 0.0f;
//             // // Iterate over the neighbors
//             // for (int j = 0; j < numNeigh; j++) {
//             //     // Get the neighbor index
//             //     int32_t index_j = indices_j_ptr[offset + j];
//             //     // Get the mass of the neighbor
//             //     float mass_j = masses_j_ptr[index_j];
//             //     // Get the density of the neighbor
//             //     float density_j = densities_j_ptr[index_j];
//             //     // Get the quantity of the neighbor
//             //     float quantity_j = quantities_j_ptr[index_j];
//             //     // Get the kernel value
//             //     float kernel = kernels_ptr[offset + j];

//             //     // Compute the density contribution
//             //     density += mass_j / density_j * quantity_j * kernel;
//             // }

//             // // Compute the density
//             // output_ptr[index_i] = density;
//         }
//     }
//     return output;
// }




template<typename scalar_t = float, std::size_t qDim = 1>
at::Tensor scatterAdd(
    torch::Tensor input,
    int32_t numParticles,
    torch::Tensor numNeighbors_,
    torch::Tensor neighborOffset_
){
    bool verbose = false;
    bool useCuda = input.is_cuda();        

    auto defaultOptions = at::TensorOptions().device(input.device());
    auto hostOptions = at::TensorOptions();

    auto numNeighbors = getAccessor<int32_t, 1>(numNeighbors_, "numNeighbors", useCuda, verbose);
    auto neighborOffset = getAccessor<int32_t, 1>(neighborOffset_, "neighborOffset", useCuda, verbose);
    
    auto quantities_shape = input.sizes();
    auto numEntries = quantities_shape[0];
    auto entriesPerElement = 1;

    auto output_shape = std::vector<int64_t>{numParticles};
    std::vector<int64_t> tensor_shape;
    for (int i = 1; i < quantities_shape.size(); i++) {
        output_shape.push_back(quantities_shape[i]);
        tensor_shape.push_back(quantities_shape[i]);
    }

    auto output = torch::zeros(output_shape, defaultOptions.dtype(input.dtype()));

    auto input_accessor = getAccessor<scalar_t, 1>(input, "input", useCuda, verbose);
    auto output_accessor = getAccessor<scalar_t, 1>(output, "output", useCuda, verbose);

    if(useCuda){
        scatterAdd_cu(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            numParticles,
            numNeighbors_.data_ptr<int32_t>(), neighborOffset_.data_ptr<int32_t>());
    }
    else{

    #pragma omp parallel for
    for(int i = 0; i < numParticles; i++){
        int32_t numNeigh = numNeighbors[i];
        int32_t offset = neighborOffset[i];
        float temporary = 0.f;
        // auto temporary = torch::zeros(tensor_shape, defaultOptions.dtype(input.dtype()));

        for(int j = 0; j < numNeigh; j++){
            int32_t index_j = offset + j;
            scalar_t input_j = input_accessor[index_j];
            temporary += input_j;
        }

        output_accessor[i] = temporary;
    }
    }

    return output;
}


#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("sphDensity", &sphDensity, "SPH Density Interpolation");
  m.def("sphInterpolation", &sphInterpolation, "SPH Interpolation");
  m.def("scatterAdd", &scatterAdd, "Scatter Add");
}