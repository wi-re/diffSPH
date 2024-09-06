#pragma once
// #define __USE_ISOC11 1
// #include <time.h>
#ifdef __INTELLISENSE__
#define OMP_VERSION
#endif

// #define _OPENMP
#include <algorithm>
#ifdef OMP_VERSION
#include <omp.h>
// #include <ATen/ParallelOpenMP.h>
#endif
#ifdef TBB_VERSION
#include <ATen/ParallelNativeTBB.h>
#endif
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/core/TensorAccessor.h>


#if defined(__CUDACC__) || defined(__HIPCC__)
#define hostDeviceInline __device__ inline
#else
#define hostDeviceInline inline
#endif

// Define the traits for the pointer types based on the CUDA availability
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename T>
using traits = torch::RestrictPtrTraits<T>;
#else
template<typename T>
using traits = torch::DefaultPtrTraits<T>;
#endif

// Define tensor accessor aliases for different cases, primiarly use ptr_t when possible
template<typename T, std::size_t dim>
using ptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using cptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using tensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using ctensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using general_t = torch::TensorAccessor<T, dim>;


#include <torch/extension.h>
/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print32_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim, traits>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    try{
        return t.template packed_accessor32<scalar_t, dim, traits>();
    } catch(const c10::Error& e){
        throw std::runtime_error(name + " is not of the correct type " + e.what());
    }
}




template<std::size_t dim, typename scalar_t>
hostDeviceInline auto modDistance2(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return sum;
}
template<typename scalar_t>
hostDeviceInline auto moduloOp(const scalar_t p, const scalar_t q, const scalar_t h){
    return ((p - q + h / 2.0) - std::floor((p - q + h / 2.0) / h) * h) - h / 2.0;
}

struct Mat3x3{
    float m_00, m_01, m_02;
    float m_10, m_11, m_12;
    float m_20, m_21, m_22;
};

 struct Float3{
    float x;
    float y;
    float z;};
 struct Float2{
    float x;
    float y;};
 
hostDeviceInline void mDBCDensityKernel(int32_t index_i,
    cptr_t<float,1> masses_j, cptr_t<float,1> densities_j, cptr_t<float,2> positions_i, cptr_t<float,2> positions_j,

    cptr_t<int64_t,1> indices_j, cptr_t<float,1> kernels, cptr_t<float,2> kernelGradients, cptr_t<float,1> distances, cptr_t<float,2> vectors, cptr_t<float,1> supports,
    cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset,

    ptr_t<float,1> mDBCDensity, ptr_t<float,1> shepardDensity, float rho0){

            float shepardNominator = 0.f;
            float shepardDenominator = 0.f;

            Mat3x3 A_g{0.f,0.f,0.f,     0.f,0.f,0.f,        0.f,0.f,0.f};
            Float3 b{0.f,0.f,0.f};


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
                Float2 gradW_ij{kernelGradients[offset + j][0], kernelGradients[offset + j][1]};

                // Get the distance vector
                float r_ij = distances[offset + j];
                float h_ij = supports[offset + j];
                Float2 x_ij{- r_ij * h_ij * vectors[offset + j][0], - r_ij * h_ij * vectors[offset + j][1]};

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
                return;
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

void mDBCDensity_cuda(
    int32_t numParticles,
    torch::Tensor masses_j, torch::Tensor densities_j, torch::Tensor positions_i, torch::Tensor positions_j,
    torch::Tensor indices_j, torch::Tensor kernels, torch::Tensor kernelGradients, torch::Tensor distances, torch::Tensor vectors, torch::Tensor supports,
    torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    torch::Tensor mDBCDensity, torch::Tensor shepardDensity, float rho0);