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

// hostDeviceInline void countingKernel(int32_t index_i,
//     cptr_t<float,2> positions_i, cptr_t<float,1> supports_i, 
//     cptr_t<float,2> positions_j, cptr_t<float,1> supports_j,
//     cptr_t<float,1> minDomain, cptr_t<float,1> maxDomain, cptr_t<bool,1> periodicity,
//     cptr_t<int64_t,1> indices_j, cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset,
    
//     cptr_t<int32_t,1> output){
//         float x_i = positions_i[index_i][0];
//         float y_i = positions_i[index_i][1];
//         float h_i = supports_i[index_i];

//         int32_t numNeigh = numNeighbors[index_i];
//         int32_t offset = neighborOffset[index_i];

//         int32_t counter = 0;
//         // Iterate over the neighbors
//         for (int j = 0; j < numNeigh; j++) {
//             int32_t index_j = indices_j[offset + j];
//             auto x_j = positions_j[index_j][0];
//             auto y_j = positions_j[index_j][1];
//             auto h_j = supports_j[index_j];

//             auto diff_x = periodicity[0] ? moduloOp(x_i, x_j, maxDomain[0] - minDomain[0]) : x_i - x_j;
//             auto diff_y = periodicity[1] ? moduloOp(y_i, y_j, maxDomain[1] - minDomain[1]) : y_i - y_j;
//             auto sum = diff_x * diff_x + diff_y * diff_y;
//             if (sum <= h_j * h_j)
//                 counter++;

//         }
//         output[index_i] = counter;

//     }

// hostDeviceInline void updateKernel(int32_t index_i,
//     cptr_t<float,2> positions_i, cptr_t<float,1> supports_i, 
//     cptr_t<float,2> positions_j, cptr_t<float,1> supports_j,
//     cptr_t<float,1> minDomain, cptr_t<float,1> maxDomain, cptr_t<bool,1> periodicity,
//     cptr_t<int64_t,1> indices_j, cptr_t<int32_t,1> numNeighbors, cptr_t<int32_t,1> neighborOffset,
    
//     cptr_t<int32_t, 1> newOffsets,
//     cptr_t<int64_t,1> output_i, cptr_t<int64_t, 1> output_j, 
//     cptr_t<float, 1> output_rij, cptr_t<float, 2> output_xij, cptr_t<float, 1> output_hij){
//         float x_i = positions_i[index_i][0];
//         float y_i = positions_i[index_i][1];
//         float h_i = supports_i[index_i];

//         int32_t numNeigh = numNeighbors[index_i];
//         int32_t offset = neighborOffset[index_i];
//         auto newOffset = newOffsets[index_i];

//         int32_t counter = 0;
//         // Iterate over the neighbors
//         for (int j = 0; j < numNeigh; j++) {
//             int32_t index_j = indices_j[offset + j];
//             auto x_j = positions_j[index_j][0];
//             auto y_j = positions_j[index_j][1];
//             auto h_j = supports_j[index_j];

//             auto diff_x = periodicity[0] ? moduloOp(x_i, x_j, maxDomain[0] - minDomain[0]) : x_i - x_j;
//             auto diff_y = periodicity[1] ? moduloOp(y_i, y_j, maxDomain[1] - minDomain[1]) : y_i - y_j;
//             auto sum = diff_x * diff_x + diff_y * diff_y;
//             if (sum <= h_j * h_j){
//                 output_i[newOffset + counter] = index_i;
//                 output_j[newOffset + counter] = index_j;
//                 auto dist = std::sqrt(sum);
//                 output_rij[newOffset + counter] = dist / h_j;
//                 output_xij[newOffset + counter][0] = diff_x / (dist + 1e-6f * h_i);
//                 output_xij[newOffset + counter][1] = diff_y / (dist + 1e-6f * h_i);
//                 output_hij[newOffset + counter] = h_j;
//                 counter++;
//             }
//         }
//     }


// void countingKernel_cuda(int32_t numParticles,
//     torch::Tensor positions_i, torch::Tensor supports_i, 
//     torch::Tensor positions_j, torch::Tensor supports_j,
//     torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
//     torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    
//     torch::Tensor output);
// void updateKernel_cuda(int32_t numParticles,
//     torch::Tensor positions_i, torch::Tensor supports_i, 
//     torch::Tensor positions_j, torch::Tensor supports_j,
//     torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
//     torch::Tensor indices_j, torch::Tensor numNeighbors, torch::Tensor neighborOffset,
    
//     torch::Tensor newOffsets,
//     torch::Tensor output_i, torch::Tensor output_j, 
//     torch::Tensor output_rij, torch::Tensor output_xij, torch::Tensor output_hij);


#include <cmath>
#if __has_include(<cuda.h>)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <utility>
#include <tuple>
#include <xmmintrin.h>
#ifndef NO_CUDA_SUPPORT
#if __has_include(<cuda.h>)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#else
#define __host__
#define __device__
#define __forceinline__
#endif



#if defined(__CUDACC__) || defined(__HIPCC__)
#else
#define __host__
#define __device__
#define __inline__ inline
#define __forceinline__ inline
#endif


namespace SVD {
	// Constants used for calculation of givens quaternions
#define _gamma 5.828427124f // sqrt(8)+3;
#define _cstar 0.923879532f // cos(pi/8)
#define _sstar 0.3826834323f // sin(p/8)
// Threshold value
#define _SVD_EPSILON 1e-6f
// Iteration counts for Jacobi Eigenanlysis and reciprocal square root functions, influence precision
#define JACOBI_STEPS 12
#define RSQRT_STEPS 4
#define RSQRT1_STEPS 6
// Used to control Denormals Are Zero and Flush To Zero settings on CPUs, surround code with these macros
#ifndef NO_CPU_SUPPORT
#define SET_MXCSR_REGISTER \
int oldMXCSR__ = _mm_getcsr();\
int newMXCSR__ = oldMXCSR__ | 0x8040;\
_mm_setcsr( newMXCSR__ ); 
#define RESET_MXCSR_REGISTER \
_mm_setcsr( oldMXCSR__ ); 
#endif
	// Calculates the result of x / y. Required as the accurate square root function otherwise uses a reciprocal approximation when using optimizations on a GPU which can lead to slightly different results. If non exact matching results are acceptable a simple division can be used. Maps to __fdiv_rn(x,y) internally for CUDA.
	__host__ __device__ __forceinline__ float fdiv(float x, float y) {
#ifndef __CUDA_ARCH__
		return x / y;
#else
		return __fdiv_rn(x, y);
#endif
	}
	// Calculates the reciprocal square root of x using a fast approximation. The number of newton iterations can be controlled using RSQRT_STEPS. A built-in rsqrt function or 1.f/sqrt(x) could be used, however doing this manually allows for exact matching results on CPU and GPU code.
	__host__ __device__ __forceinline__ float rsqrt(float x) {
		float xhalf = -0.5f * x;
		int i = *(int *)&x;
		i = 0x5f375a82 - (i >> 1);
		x = *(float *)&i;
		for (int32_t i = 0; i < RSQRT_STEPS; ++i) {
			x = x * fmaf(x * x, xhalf, 1.5f);
		}
		return x;
	}
	// See rsqrt. Uses RSQRT1_STEPS to offer a higher precision alternative
	__host__ __device__ __forceinline__ float rsqrt1(float x) {
		float xhalf = -0.5f*x;
		int i = *(int *)&x;
		i = 0x5f37599e - (i >> 1);
		x = *(float *)&i;
		for (int32_t i = 0; i < RSQRT1_STEPS; ++i) {
			x = x * fmaf(x*x, xhalf, 1.5f);
		}
		return x;
	}
	// Calculates the square root of x using 1.f/rsqrt1(x) to give a square root with controllable and consistent precision on CPU and GPU code.
	__host__ __device__ __forceinline__ float accurateSqrt(float x) {
		return fdiv(1.f, rsqrt1(x));
	}
	// Helper function used to swap X with Y and Y with  X if c == true
	__host__ __device__ __forceinline__ void condSwap(bool c, float &X, float &Y){
		float Z = X;
		X = c ? Y : X;
		Y = c ? Z : Y;
	}
	// Helper function used to swap X with Y and Y with -X if c == true
	__host__ __device__ __forceinline__ void condNegSwap(bool c, float &X, float &Y){
		float Z = -X;
		X = c ? Y : X;
		Y = c ? Z : Y;
	}
	// Helper class to contain a quaternion. Could be replaced with float4 (CUDA based type) but this might lead to unintended conversions when using the supplied matrices
	struct quaternion {
		float x = 0.f, y = 0.f, z = 0.f, w = 1.f;
		__host__ __device__ __forceinline__ float& operator[](int32_t arg) {
			return ((float*)this)[arg];
		}
	};
	// A simple 3x3 Matrix class
	struct Mat3x3 {
		float m_00 = 1.f, m_01 = 0.f, m_02 = 0.f;
		float m_10 = 0.f, m_11 = 1.f, m_12 = 0.f;
		float m_20 = 0.f, m_21 = 0.f, m_22 = 1.f;
		static __host__ __device__ __forceinline__ Mat3x3 fromPtr(float* ptr, int32_t i, int32_t offset) {
                  return Mat3x3{ptr[i * 1 + 0 * offset], ptr[i * 1 + 1 * offset], ptr[i * 1 + 2 * offset],
                                ptr[i * 1 + 3 * offset], ptr[i * 1 + 4 * offset], ptr[i * 1 + 5 * offset],
                                ptr[i * 1 + 6 * offset], ptr[i * 1 + 7 * offset], ptr[i * 1 + 8 * offset]
			};
		}
		__host__ __device__ __forceinline__ auto det() const {
			return fmaf(m_00, fmaf(m_11, m_22, -m_21 * m_12), fmaf(-m_01, fmaf(m_10, m_22, -m_20 * m_12), m_02 *        fmaf(m_10, m_21, -m_20 * m_11)));
		}
		__host__ __device__ __forceinline__ auto norm2() const {
			return		sqrtf(m_00 * m_00 + m_01 * m_01 + m_02 * m_02) +
						sqrtf(m_10 * m_10 + m_11 * m_11 + m_12 * m_12) +
						sqrtf(m_20 * m_20 + m_21 * m_21 + m_22 * m_22);
		}
		__host__ __device__ __forceinline__ auto normF() const {
			return	sqrtf(
				m_00 * m_00 + m_01 * m_01 + m_02 * m_02 +
				m_10 * m_10 + m_11 * m_11 + m_12 * m_12 +
				m_20 * m_20 + m_21 * m_21 + m_22 * m_22);
		}
		// __host__ __device__ __forceinline__ auto normMax() const {
		// 	return 
		// 	math::max(	math::max(m_00, math::max(m_01, m_02)),
		// 	math::max(
		// 				math::max(m_00, math::max(m_01, m_02)),
		// 				math::max(m_00, math::max(m_01, m_02)))
		// 		);
		// }
		__host__ __device__ __forceinline__ void toPtr(float* ptr, int32_t i, int32_t offset) const {
			ptr[i * 1 + 0 * offset] = m_00;
                  ptr[i * 1 + 1 * offset] = m_01;
                        ptr[i * 1 + 2 * offset] = m_02;
			ptr[i * 1 + 3 * offset] = m_10;
                        ptr[i * 1 + 4 * offset] = m_11;
                        ptr[i * 1 + 5 * offset] = m_12;
                        ptr[i * 1 + 6 * offset] = m_20;
                        ptr[i * 1 + 7 * offset] = m_21;
                        ptr[i * 1 + 8 * offset] = m_22;
		}
		__host__ __device__ __forceinline__ Mat3x3(float a11 = 1.f, float a12 = 0.f, float a13 = 0.f, float a21 = 0.f, float a22 = 1.f, float a23 = 0.f, float  a31 = 0.f, float a32 = 0.f, float a33 = 1.f) :
			m_00(a11), m_01(a12), m_02(a13), m_10(a21), m_11(a22), m_12(a23), m_20(a31), m_21(a32), m_22(a33) {}
		__host__ __device__ __forceinline__ Mat3x3(const quaternion& q) {
			m_00 = 1.f - 2.f * (fmaf(q.y, q.y, q.z * q.z)); m_01 = 2 * fmaf(q.x, q.y, -q.w * q.z); m_02 = 2 * fmaf(q.x, q.z, q.w * q.y);
			m_10 = 2.f * fmaf(q.x, q.y, +q.w * q.z); m_11 = 1 - 2 * fmaf(q.x, q.x, q.z * q.z); m_12 = 2 * fmaf(q.y, q.z, -q.w * q.x);
			m_20 = 2.f * fmaf(q.x, q.z, -q.w * q.y); m_21 = 2 * fmaf(q.y, q.z, q.w * q.x); m_22 = 1 - 2 * fmaf(q.x, q.x, q.y * q.y);
		}
		__host__ __device__ __forceinline__ Mat3x3 transpose() const {
			return Mat3x3{
				m_00, m_10, m_20,
				m_01, m_11, m_21,
				m_02, m_12, m_22
			};
		}
		__host__ __device__ __forceinline__ Mat3x3 operator*(const float& o)  const {
			return Mat3x3{
				m_00 * o, m_01 * o, m_02 * o,
				m_10 * o, m_11 * o, m_12 * o,
				m_20 * o, m_21 * o, m_22 * o,
			};
		}
		__host__ __device__ __forceinline__ Mat3x3& operator*=(const float& o) {
			m_00 *= o; m_01 *= o; m_02 *= o;
			m_10 *= o; m_11 *= o; m_12 *= o;
			m_20 *= o; m_21 *= o; m_22 *= o;
			return *this;
		}
		__host__ __device__ __forceinline__ Mat3x3 operator-(const Mat3x3& o) const {
			return Mat3x3{
			m_00 - o.m_00, m_01 - o.m_01, m_02 - o.m_02,
			m_10 - o.m_10, m_11 - o.m_11, m_12 - o.m_12,
			m_20 - o.m_20, m_21 - o.m_21, m_22 - o.m_22
			};
		}
		__host__ __device__ __forceinline__ Mat3x3 operator+(const Mat3x3& o) const {
			return Mat3x3{
			m_00 + o.m_00, m_01 + o.m_01, m_02 + o.m_02,
			m_10 + o.m_10, m_11 + o.m_11, m_12 + o.m_12,
			m_20 + o.m_20, m_21 + o.m_21, m_22 + o.m_22
			};
		}
		__host__ __device__ __forceinline__ Mat3x3 operator*(const Mat3x3& o)  const {
			return Mat3x3{
				fmaf(m_00, o.m_00, fmaf(m_01, o.m_10, m_02 * o.m_20)), fmaf(m_00, o.m_01, fmaf(m_01, o.m_11, m_02 * o.m_21)), fmaf(m_00, o.m_02, fmaf(m_01, o.m_12, m_02 * o.m_22)),
				fmaf(m_10, o.m_00, fmaf(m_11, o.m_10, m_12 * o.m_20)), fmaf(m_10, o.m_01, fmaf(m_11, o.m_11, m_12 * o.m_21)), fmaf(m_10, o.m_02, fmaf(m_11, o.m_12, m_12 * o.m_22)),
				fmaf(m_20, o.m_00, fmaf(m_21, o.m_10, m_22 * o.m_20)), fmaf(m_20, o.m_01, fmaf(m_21, o.m_11, m_22 * o.m_21)), fmaf(m_20, o.m_02, fmaf(m_21, o.m_12, m_22 * o.m_22))
			};
		}
	};
	// A simple symmetrix 3x3 Matrix class (contains no storage for (0, 1) (0, 2) and (1, 2)
	struct Symmetric3x3 {
		float m_00 = 1.f;
		float m_10 = 0.f, m_11 = 1.f;
		float m_20 = 0.f, m_21 = 0.f, m_22 = 1.f;
		__host__ __device__ __forceinline__ Symmetric3x3(float a11 = 1.f, float a21 = 0.f, float a22 = 1.f, float  a31 = 0.f, float a32 = 0.f, float a33 = 1.f) :
			m_00(a11), m_10(a21), m_11(a22), m_20(a31), m_21(a32), m_22(a33) {}
		__host__ __device__ __forceinline__ Symmetric3x3(Mat3x3 o) :
			m_00(o.m_00), m_10(o.m_10), m_11(o.m_11), m_20(o.m_20), m_21(o.m_21), m_22(o.m_22) {}
	};
	// Helper struct to store 2 floats to avoid OUT parameters on functions
	struct givens {
		float ch = _cstar;
		float sh = _sstar;
	};
	// Helper struct to store 2 Matrices to avoid OUT parameters on functions
	struct QR {
		Mat3x3 Q;
		Mat3x3 R;
	};
	// Helper struct to store 3 Matrices to avoid OUT parameters on functions
	struct SVDSet {
		Mat3x3 U, S, V;
	};
	// Calculates the squared norm of the vector [x y z] using a standard scalar product d = x * x + y *y + z * z
	__host__ __device__ __forceinline__ float dist2(float x, float y, float z) {
		return fmaf(x, x, fmaf(y, y, z * z));
	}
	// For an explanation of the math see http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf 
	// Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
	// See Algorithm 2 in reference. Given a matrix A this function returns the givens quaternion (x and w component, y and z are 0)
	__host__ __device__ __forceinline__ givens approximateGivensQuaternion(Symmetric3x3& A) {
		givens g{ 2.f * (A.m_00 - A.m_11), A.m_10 };
		bool b = _gamma * g.sh*g.sh < g.ch*g.ch;
		float w = rsqrt(fmaf(g.ch, g.ch, g.sh * g.sh));
		if (w != w) b = 0;
		return givens{ b ? w * g.ch : (float)_cstar,b ? w * g.sh : (float)_sstar };
	}
	// Function used to apply a givens rotation S. Calculates the weights and updates the quaternion to contain the cumultative rotation
	__host__ __device__ __forceinline__ void jacobiConjugation(const int32_t x, const int32_t y, const int32_t z, Symmetric3x3& S, quaternion& q) {
		auto g = approximateGivensQuaternion(S);
		float scale = 1.f / fmaf(g.ch, g.ch, g.sh *  g.sh);
		float a = fmaf(g.ch, g.ch, -g.sh * g.sh) * scale;
		float b = 2.f * g.sh * g.ch * scale;
		Symmetric3x3 _S = S;
		// perform conjugation S = Q'*S*Q
		S.m_00 = fmaf(a, fmaf(a, _S.m_00, b * _S.m_10), b * (fmaf(a, _S.m_10, b * _S.m_11)));
		S.m_10 = fmaf(a, fmaf(-b, _S.m_00, a * _S.m_10), b * (fmaf(-b, _S.m_10, a * _S.m_11)));
		S.m_11 = fmaf(-b, fmaf(-b, _S.m_00, a * _S.m_10), a * (fmaf(-b, _S.m_10, a * _S.m_11)));
		S.m_20 = fmaf(a, _S.m_20, b * _S.m_21);
		S.m_21 = fmaf(-b, _S.m_20, a * _S.m_21);
		S.m_22 = _S.m_22;
		// update cumulative rotation qV
		float tmp[3];
		tmp[0] = q[0] * g.sh;
		tmp[1] = q[1] * g.sh;
		tmp[2] = q[2] * g.sh;
		g.sh *= q[3];
		// (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1)) for (p,q) = ((0,1),(1,2),(0,2))
		q[z] = fmaf(q[z], g.ch, g.sh);
		q[3] = fmaf(q[3], g.ch, -tmp[z]); // w
		q[x] = fmaf(q[x], g.ch, tmp[y]);
		q[y] = fmaf(q[y], g.ch, -tmp[x]);
		// re-arrange matrix for next iteration
		_S.m_00 = S.m_11;
		_S.m_10 = S.m_21; _S.m_11 = S.m_22;
		_S.m_20 = S.m_10; _S.m_21 = S.m_20; _S.m_22 = S.m_00;
		S.m_00 = _S.m_00;
		S.m_10 = _S.m_10; S.m_11 = _S.m_11;
		S.m_20 = _S.m_20; S.m_21 = _S.m_21; S.m_22 = _S.m_22;
	}
	// Function used to contain the givens permutations and the loop of the jacobi steps controlled by JACOBI_STEPS
	// Returns the quaternion q containing the cumultative result used to reconstruct S
	__host__ __device__ __forceinline__ quaternion jacobiEigenanlysis(Symmetric3x3 S) {
		quaternion q;
		for (int32_t i = 0; i < JACOBI_STEPS; i++) {
			jacobiConjugation(0, 1, 2, S, q);
			jacobiConjugation(1, 2, 0, S, q);
			jacobiConjugation(2, 0, 1, S, q);
		}
		return q;
	}
	// Implementation of Algorithm 3
	__host__ __device__ __forceinline__ void sortSingularValues(Mat3x3& B, Mat3x3& V){
		float rho1 = dist2(B.m_00, B.m_10, B.m_20);
		float rho2 = dist2(B.m_01, B.m_11, B.m_21);
		float rho3 = dist2(B.m_02, B.m_12, B.m_22);
		bool c;
		c = rho1 < rho2;
		condNegSwap(c, B.m_00, B.m_01); condNegSwap(c, V.m_00, V.m_01);
		condNegSwap(c, B.m_10, B.m_11); condNegSwap(c, V.m_10, V.m_11);
		condNegSwap(c, B.m_20, B.m_21); condNegSwap(c, V.m_20, V.m_21);
		condSwap(c, rho1, rho2);
		c = rho1 < rho3;
		condNegSwap(c, B.m_00, B.m_02); condNegSwap(c, V.m_00, V.m_02);
		condNegSwap(c, B.m_10, B.m_12); condNegSwap(c, V.m_10, V.m_12);
		condNegSwap(c, B.m_20, B.m_22); condNegSwap(c, V.m_20, V.m_22);
		condSwap(c, rho1, rho3);
		c = rho2 < rho3;
		condNegSwap(c, B.m_01, B.m_02); condNegSwap(c, V.m_01, V.m_02);
		condNegSwap(c, B.m_11, B.m_12); condNegSwap(c, V.m_11, V.m_12);
		condNegSwap(c, B.m_21, B.m_22); condNegSwap(c, V.m_21, V.m_22);
	}
	// Implementation of Algorithm 4
	__host__ __device__ __forceinline__ givens QRGivensQuaternion(float a1, float a2){
		// a1 = pivot point on diagonal
		// a2 = lower triangular entry we want to annihilate
		float epsilon = (float)_SVD_EPSILON;
		float rho = accurateSqrt(fmaf(a1, a1, +a2 * a2));
		givens g{ fabsf(a1) + fmaxf(rho, epsilon), rho > epsilon ? a2 : 0 };
		bool b = a1 < 0.f;
		condSwap(b, g.sh, g.ch);
		float w = rsqrt(fmaf(g.ch, g.ch, g.sh *  g.sh));
		g.ch *= w;
		g.sh *= w;
		return g;
	}
	// Implements a QR decomposition of a Matrix, see Sec 4.2
	__host__ __device__ __forceinline__ QR QRDecomposition(Mat3x3& B){
		Mat3x3 Q, R;
		// first givens rotation (ch,0,0,sh)
		auto g1 = QRGivensQuaternion(B.m_00, B.m_10);
		auto a = fmaf(-2.f, g1.sh*g1.sh, 1.f);
		auto b = 2.f * g1.ch*g1.sh;
		// apply B = Q' * B
		R.m_00 = fmaf(a, B.m_00, b * B.m_10);	R.m_01 = fmaf(a, B.m_01, b * B.m_11);	R.m_02 = fmaf(a, B.m_02, b * B.m_12);
		R.m_10 = fmaf(-b, B.m_00, a * B.m_10);	R.m_11 = fmaf(-b, B.m_01, a * B.m_11);	R.m_12 = fmaf(-b, B.m_02, a * B.m_12);
		R.m_20 = B.m_20;						R.m_21 = B.m_21;						R.m_22 = B.m_22;
		// second givens rotation (ch,0,-sh,0)
		auto g2 = QRGivensQuaternion(R.m_00, R.m_20);
		a = fmaf(-2.f, g2.sh*g2.sh, 1.f);
		b = 2.f * g2.ch*g2.sh;
		// apply B = Q' * B;
		B.m_00 = fmaf(a, R.m_00, b * R.m_20);	B.m_01 = fmaf(a, R.m_01, b * R.m_21);	B.m_02 = fmaf(a, R.m_02, b * R.m_22);
		B.m_10 = R.m_10;						B.m_11 = R.m_11;						B.m_12 = R.m_12;
		B.m_20 = fmaf(-b, R.m_00, a * R.m_20);	B.m_21 = fmaf(-b, R.m_01, a * R.m_21);	B.m_22 = fmaf(-b, R.m_02, a * R.m_22);
		// third givens rotation (ch,sh,0,0)
		auto g3 = QRGivensQuaternion(B.m_11, B.m_21);
		a = fmaf(-2.f, g3.sh*g3.sh, 1.f);
		b = 2.f * g3.ch*g3.sh;
		// R is now set to desired value
		R.m_00 = B.m_00;						R.m_01 = B.m_01;						R.m_02 = B.m_02;
		R.m_10 = fmaf(a, B.m_10, b * B.m_20);  R.m_11 = fmaf(a, B.m_11, b * B.m_21);  R.m_12 = fmaf(a, B.m_12, b * B.m_22);
		R.m_20 = fmaf(-b, B.m_10, a * B.m_20);  R.m_21 = fmaf(-b, B.m_11, a * B.m_21);  R.m_22 = fmaf(-b, B.m_12, a * B.m_22);
		// construct the cumulative rotation Q=Q1 * Q2 * Q3
		// the number of floating point operations for three quaternion multiplications
		// is more or less comparable to the explicit form of the joined matrix.
		// certainly more memory-efficient!
		auto sh12 = 2.f * fmaf(g1.sh, g1.sh, -0.5f);
		auto sh22 = 2.f * fmaf(g2.sh, g2.sh, -0.5f);
		auto sh32 = 2.f * fmaf(g3.sh, g3.sh, -0.5f);
		Q.m_00 = sh12 * sh22;
		Q.m_01 = fmaf(4.f * g2.ch * g3.ch, sh12 * g2.sh * g3.sh, 2.f * g1.ch * g1.sh * sh32);
		Q.m_02 = fmaf(4.f * g1.ch * g3.ch, g1.sh * g3.sh, -2.f * g2.ch * sh12 * g2.sh * sh32);

		Q.m_10 = -2.f * g1.ch * g1.sh * sh22;
		Q.m_11 = fmaf(-8.f * g1.ch * g2.ch * g3.ch, g1.sh * g2.sh * g3.sh, sh12 * sh32);
		Q.m_12 = fmaf(-2.f * g3.ch, g3.sh, 4.f * g1.sh * fmaf(g3.ch * g1.sh, g3.sh, g1.ch * g2.ch*g2.sh*sh32));

		Q.m_20 = 2.f * g2.ch * g2.sh;
		Q.m_21 = -2.f * g3.ch * sh22 * g3.sh;
		Q.m_22 = sh22 * sh32;
		return QR{ Q,R };
	}
	// Wrapping function used to contain all of the required sub calls
	// Also sets CPU floating point mode to Denormals Are Zero and Flush To Zero 
	// in order to avoid certain issues arising from the quaternions becoming too small
	// which causes the sqrt to become nan due to it being implemented as a reciprocal square 
	// root thus causing a division by 0 in the wrong place.
	__host__ __device__ __forceinline__ SVDSet svd(Mat3x3 A){
#ifndef __CUDA_ARCH__
//		SET_MXCSR_REGISTER;
#endif
		Mat3x3 V(jacobiEigenanlysis(A.transpose() * A));
		auto B = A * V;
		sortSingularValues(B, V);
		QR qr = QRDecomposition(B);
#ifndef __CUDA_ARCH__
//		RESET_MXCSR_REGISTER;
#endif
		return SVDSet{ qr.Q, qr.R, V };
	}
	
#undef _gamma
#undef _cstar
#undef _sstar
#undef _SVD_EPSILON
#undef JACOBI_STEPS
#undef RSQRT_STEPS
#undef RSQRT1_STEPS
}	