#include "mDBC.h"
 
 
std::vector<at::Tensor> mDBC(
    std::pair<torch::Tensor, torch::Tensor> masses_,
    std::pair<torch::Tensor, torch::Tensor> densities_,
    std::pair<torch::Tensor, torch::Tensor> positions_,
    
    int32_t numBoundaryParticles, int32_t numFluidParticles,
    std::pair<torch::Tensor, torch::Tensor> indices_,
    torch::Tensor numNeighbors_,
    torch::Tensor neighborOffset_,
    torch::Tensor kernels_,
    torch::Tensor kernelGradients_,
    torch::Tensor distances_, 
    torch::Tensor vectors_,
    torch::Tensor supports_,


    float rho0){
    bool useCuda = numNeighbors_.is_cuda();
    auto defaultOptions = at::TensorOptions().device(numNeighbors_.device());
    auto hostOptions = at::TensorOptions();
    
    // Get the pointers to the particle data
    auto masses_i = getAccessor<float, 1>(masses_.first, "particle masses (A)", useCuda);
    auto masses_j = getAccessor<float, 1>(masses_.second, "particle masses (B)", useCuda);
    auto densities_i = getAccessor<float, 1>(densities_.first, "particle densities (A)", useCuda);
    auto densities_j = getAccessor<float, 1>(densities_.second, "particle densities (B)", useCuda);
    auto positions_i = getAccessor<float, 2>(positions_.first, "particle positions (A)", useCuda);
    auto positions_j = getAccessor<float, 2>(positions_.second, "particle positions (B)", useCuda);

    // Get  the pointers to the neighborhood data
    auto indices_i = getAccessor<int64_t, 1>(indices_.first, "indices (A)", useCuda);
    auto indices_j = getAccessor<int64_t, 1>(indices_.second, "indices (B)", useCuda);
    auto numNeighbors = getAccessor<int32_t, 1>(numNeighbors_, "numNeighbors", useCuda);
    auto neighborOffset = getAccessor<int32_t, 1>(neighborOffset_, "neighborOffset", useCuda);
    auto kernels = getAccessor<float, 1>(kernels_, "kernels", useCuda);
    auto kernelGradients = getAccessor<float, 2>(kernelGradients_, "kernelGradients", useCuda);
    auto distances = getAccessor<float, 1>(distances_, "distances", useCuda);
    auto vectors = getAccessor<float, 2>(vectors_, "vectors", useCuda);
    auto supports = getAccessor<float, 1>(supports_, "supports", useCuda);

     
    auto mDBCDensity_ = torch::zeros(numBoundaryParticles, defaultOptions.dtype(torch::kFloat32));
    auto mDBCDensity = getAccessor<float, 1>(mDBCDensity_, "output", useCuda);
    auto shepardDensity_ = torch::zeros(numBoundaryParticles, defaultOptions.dtype(torch::kFloat32));
    auto shepardDensity = getAccessor<float, 1>(shepardDensity_, "output", useCuda);

    // auto b_vec_ = torch::zeros({numBoundaryParticles,3}, defaultOptions.dtype(torch::kFloat32));
    // auto b_vec = getAccessor<float, 2>(b_vec_, "output", useCuda);

    // auto A_g_output_ = torch::zeros({numBoundaryParticles,3,3}, defaultOptions.dtype(torch::kFloat32));
    // auto A_g_output = getAccessor<float, 3>(A_g_output_, "output", useCuda);

    // auto A_g_inv_output_ = torch::zeros({numBoundaryParticles,3,3}, defaultOptions.dtype(torch::kFloat32));
    // auto A_g_inv_output = getAccessor<float, 3>(A_g_inv_output_, "output", useCuda);

    // auto numParticles = particle_positions_.first.size(0);


    // return {mDBCDensity_, shepardDensity_};
    if(useCuda){
        mDBCDensity_cuda(numBoundaryParticles,
        masses_.second, densities_.second, positions_.first, positions_.second,
        indices_.second, kernels_, kernelGradients_, distances_, vectors_, supports_,
        numNeighbors_, neighborOffset_, mDBCDensity_, shepardDensity_, rho0);
        // throw std::runtime_error("CUDA not supported");
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numBoundaryParticles; index_i++) {
            mDBCDensityKernel(index_i, masses_j, densities_j, positions_i, positions_j, indices_j, kernels, kernelGradients, distances, vectors, supports, numNeighbors, neighborOffset, mDBCDensity, shepardDensity, rho0);
        }

        // auto A_g_inv = torch::linalg::pinv(A_g_output_);
        


        // auto res = torch::matmul(A_g_inv, b_vec_.unsqueeze(2)).slice(2,0,1);

        // Iterate over the particles
        // #pragma omp parallel for
        // for (int index_i = 0; index_i < numBoundaryParticles; index_i++) {

        //     // A_g_output_[index_i][0][0] = A_g.m_00;
        //     // A_g_output_[index_i][0][1] = A_g.m_01;
        //     // A_g_output_[index_i][0][2] = A_g.m_02;

        //     // A_g_output_[index_i][1][0] = A_g.m_10;
        //     // A_g_output_[index_i][1][1] = A_g.m_11;
        //     // A_g_output_[index_i][1][2] = A_g.m_12;

        //     // A_g_output_[index_i][2][0] = A_g.m_20;
        //     // A_g_output_[index_i][2][1] = A_g.m_21;
        //     // A_g_output_[index_i][2][2] = A_g.m_22;


    	// 	SVD::Mat3x3 V_(jacobiEigenanlysis(A_g.transpose() * A_g));
        //     auto B = A_g * V_; 
        //     // sortSingularValues(B, V_);
        //     SVD::QR qr = QRDecomposition(B);
        //     SVD::SVDSet svd{ qr.Q, qr.R, V_ };

        //     auto U = svd.U;
        //     auto S = svd.S;
        //     auto V = svd.V;
        //     S.m_00 = (fabsf(S.m_00) > 1e-7f ? 1.f / S.m_00 : 0.f);
        //     S.m_11 = (fabsf(S.m_11) > 1e-7f ? 1.f / S.m_11 : 0.f);
        //     S.m_22 = (fabsf(S.m_22) > 1e-7f ? 1.f / S.m_22 : 0.f);
        //     S.m_01 = S.m_02 = S.m_10 = S.m_12 = S.m_20 = S.m_21 = 0.f;
        //     auto A_g_inv = V * S * U.transpose();

        //     // A_g_inv = A_g * A_g_inv;


        //     // A_g_inv_output[index_i][0][0] = A_g_inv.m_00;
        //     // A_g_inv_output[index_i][0][1] = A_g_inv.m_01;
        //     // A_g_inv_output[index_i][0][2] = A_g_inv.m_02;

        //     // A_g_inv_output[index_i][1][0] = A_g_inv.m_10;
        //     // A_g_inv_output[index_i][1][1] = A_g_inv.m_11;
        //     // A_g_inv_output[index_i][1][2] = A_g_inv.m_12;

        //     // A_g_inv_output[index_i][2][0] = A_g_inv.m_20;
        //     // A_g_inv_output[index_i][2][1] = A_g_inv.m_21;
        //     // A_g_inv_output[index_i][2][2] = A_g_inv.m_22;


        //     auto res_x = A_g_inv.m_00 * b.x + A_g_inv.m_01 * b.y + A_g_inv.m_02 * b.z;
        //     auto res_y = A_g_inv.m_10 * b.x + A_g_inv.m_11 * b.y + A_g_inv.m_12 * b.z;
        //     auto res_z = A_g_inv.m_20 * b.x + A_g_inv.m_21 * b.y + A_g_inv.m_22 * b.z;

        //     auto relPos_x = positions_i[index_i][0] - positions_j[index_i][0];
        //     auto relPos_y = positions_i[index_i][1] - positions_j[index_i][1];

        //     auto boundaryDensity = res_x + (relPos_x * res_y + relPos_y * res_z);

        //     boundaryDensity = numNeigh <= 5 ? shepard : boundaryDensity;
        //     mDBCDensity[index_i] = std::max(boundaryDensity, rho0);

        // }
    }
    return {mDBCDensity_, shepardDensity_};
    // return {mDBCDensity_, shepardDensity_};
    // return {mDBCDensity_, shepardDensity_, A_g_output_, A_g_inv_output_};
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mDBC", &mDBC, "SPH Density Interpolation");
}