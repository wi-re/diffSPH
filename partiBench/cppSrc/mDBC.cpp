#include "mDBC.h"

 struct float3{
    float x;
    float y;
    float z;};
 struct float2{
    float x;
    float y;};
 
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
        throw std::runtime_error("CUDA not supported");
    }
    else{
        // Iterate over the particles
        #pragma omp parallel for
        for (int index_i = 0; index_i < numBoundaryParticles; index_i++) {
            float shepardNominator = 0.f;
            float shepardDenominator = 0.f;

            SVD::Mat3x3 A_g(0.f,0.f,0.f,     0.f,0.f,0.f,        0.f,0.f,0.f);
            float3 b{0.f,0.f,0.f};


            int32_t numNeigh = numNeighbors[index_i];
            int32_t offset = neighborOffset[index_i];

            for (int j = 0; j < numNeigh; j++) {
                // Get the neighbor index
                int32_t index_j = indices_j[offset + j];
                float m_j = masses_j[index_j];
                float rho_j = densities_j[index_j];

                // Get the kernel value
                float W_ij = kernels[offset + j];
                // Get the kernel gradient
                float2 gradW_ij{kernelGradients[offset + j][0], kernelGradients[offset + j][1]};

                // Get the distance vector
                float r_ij = distances[offset + j];
                float h_ij = supports[offset + j];
                float2 x_ij{- r_ij * h_ij * vectors[offset + j][0], - r_ij * h_ij * vectors[offset + j][1]};

                shepardNominator += m_j * W_ij;
                shepardDenominator += m_j / rho_j * W_ij;


                float volumeTerm = m_j / rho_j * W_ij;
                float volumeGradTerm_x = m_j / rho_j * gradW_ij.x;
                float volumeGradTerm_y = m_j / rho_j * gradW_ij.y;

                float position_x = m_j / rho_j * W_ij * x_ij.x;
                float position_y = m_j / rho_j * W_ij * x_ij.y;

                float pos_grad_xx = m_j / rho_j * gradW_ij.x * x_ij.x;
                float pos_grad_xy = m_j / rho_j * gradW_ij.x * x_ij.y;
                float pos_grad_yx = m_j / rho_j * gradW_ij.y * x_ij.x;
                float pos_grad_yy = m_j / rho_j * gradW_ij.y * x_ij.y;

                A_g.m_00 += volumeTerm;
                A_g.m_10 += volumeGradTerm_x;
                A_g.m_20 += volumeGradTerm_y;

                A_g.m_01 += position_x;
                A_g.m_11 += pos_grad_xx;
                A_g.m_21 += pos_grad_xy;

                A_g.m_02 += position_y;
                A_g.m_12 += pos_grad_yx;
                A_g.m_22 += pos_grad_yy; 

                b.x += m_j * W_ij;
                b.y += m_j * gradW_ij.x;
                b.z += m_j * gradW_ij.y;
            }

            float shepard = shepardNominator / (shepardDenominator + 1e-7f);

            shepardDensity[index_i] = shepard;

            if (numNeigh <= 5) {
                mDBCDensity[index_i] = std::max(shepard, rho0);
                // mDBCDensity[index_i] = shepard;
                continue;
            }

            auto determinant = 
            []( float m_11, float m_12, float m_13, 
                float m_21, float m_22, float m_23, 
                float m_31, float m_32, float m_33){

                    float a = m_11 * (m_22 * m_33 - m_23 * m_32);
                    float b = m_12 * (m_21 * m_33 - m_23 * m_31);
                    float c = m_13 * (m_21 * m_32 - m_22 * m_31);
                    return a - b + c;
                };

            auto d0 = determinant(
                A_g.m_00, A_g.m_01, A_g.m_02, 
                A_g.m_10, A_g.m_11, A_g.m_12, 
                A_g.m_20, A_g.m_21, A_g.m_22);

            auto res_x = determinant(
                b.x, A_g.m_01, A_g.m_02, 
                b.y, A_g.m_11, A_g.m_12, 
                b.z, A_g.m_21, A_g.m_22) / (d0 + 1e-7f);

            auto res_y = determinant(
                A_g.m_00, b.x, A_g.m_02, 
                A_g.m_10, b.y, A_g.m_12, 
                A_g.m_20, b.z, A_g.m_22) / (d0 + 1e-7f);

            auto res_z = determinant(
                A_g.m_00, A_g.m_01, b.x, 
                A_g.m_10, A_g.m_11, b.y, 
                A_g.m_20, A_g.m_21, b.z) / (d0 + 1e-7f);

            
            auto relPos_x = positions_i[index_i][0] - positions_j[index_i][0];
            auto relPos_y = positions_i[index_i][1] - positions_j[index_i][1];

            auto boundaryDensity = res_x + (relPos_x * res_y + relPos_y * res_z);

            boundaryDensity = numNeigh <= 5 ? shepard : boundaryDensity;
            mDBCDensity[index_i] = std::max(boundaryDensity, rho0);

            // if (numNeigh <= 5) {
            //     mDBCDensity[index_i] = shepard;
            //     continue;
            // }
    		// SVD::Mat3x3 V_(jacobiEigenanlysis(A_g.transpose() * A_g));
            // auto B = A_g * V_; 
            // // sortSingularValues(B, V_);
            // SVD::QR qr = QRDecomposition(B);
            // SVD::SVDSet svd{ qr.Q, qr.R, V_ };

            // auto U = svd.U;
            // auto S = svd.S;
            // auto V = svd.V;
            // S.m_00 = (fabsf(S.m_00) > 1e-7f ? 1.f / S.m_00 : 0.f);
            // S.m_11 = (fabsf(S.m_11) > 1e-7f ? 1.f / S.m_11 : 0.f);
            // S.m_22 = (fabsf(S.m_22) > 1e-7f ? 1.f / S.m_22 : 0.f);
            // S.m_01 = S.m_02 = S.m_10 = S.m_12 = S.m_20 = S.m_21 = 0.f;
            // auto A_g_inv = V * S * U.transpose();

            // // A_g_inv = A_g * A_g_inv;


            // // A_g_inv_output[index_i][0][0] = A_g_inv.m_00;
            // // A_g_inv_output[index_i][0][1] = A_g_inv.m_01;
            // // A_g_inv_output[index_i][0][2] = A_g_inv.m_02;

            // // A_g_inv_output[index_i][1][0] = A_g_inv.m_10;
            // // A_g_inv_output[index_i][1][1] = A_g_inv.m_11;
            // // A_g_inv_output[index_i][1][2] = A_g_inv.m_12;

            // // A_g_inv_output[index_i][2][0] = A_g_inv.m_20;
            // // A_g_inv_output[index_i][2][1] = A_g_inv.m_21;
            // // A_g_inv_output[index_i][2][2] = A_g_inv.m_22;

            
            // auto res_x = A_g_inv.m_00 * b.x + A_g_inv.m_01 * b.y + A_g_inv.m_02 * b.z;
            // auto res_y = A_g_inv.m_10 * b.x + A_g_inv.m_11 * b.y + A_g_inv.m_12 * b.z;
            // auto res_z = A_g_inv.m_20 * b.x + A_g_inv.m_21 * b.y + A_g_inv.m_22 * b.z;

            // auto relPos_x = positions_i[index_i][0] - positions_j[index_i][0];
            // auto relPos_y = positions_i[index_i][1] - positions_j[index_i][1];

            // auto boundaryDensity = res_x + (relPos_x * res_y + relPos_y * res_z);

            // boundaryDensity = numNeigh <= 5 ? shepard : boundaryDensity;
            // mDBCDensity[index_i] = std::max(boundaryDensity, rho0);
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