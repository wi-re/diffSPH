import inspect
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
from torch.profiler import profile, record_function, ProfilerActivity

from ..kernels import kernel, kernelGradient
from ..module import Module
from ..parameter import Parameter


from sympy import nextprime
from typing import Dict, Optional
from torch.utils.cpp_extension import load

from torch.utils.cpp_extension import load, load_inline

from pathlib import Path
import os

directory = Path(__file__).resolve().parent


import subprocess
import sys

IS_WINDOWS = sys.platform == 'win32'

def find_cuda_home():
    '''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
#             print('.', which)
            nvcc = subprocess.check_output(
                [which, 'nvcc'], env = dict(PATH='%s:%s/bin' % (os.environ['PATH'], sys.exec_prefix))).decode().rstrip('\r\n')
#             print(nvcc)
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        print("No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home))
    os.environ['CUDA_HOME'] = cuda_home
    return cuda_home

find_cuda_home()

neighborSearch = load(name="neighborSearch", 
    sources=[os.path.join(directory, "neighSearch.cpp"), os.path.join(directory, "neighSearch_cuda.cu")], 
    verbose=False, extra_cflags=['-fopenmp', '-O3', '-march=native'])
# neighborSearch = load_inline(name="neighborSearch", cpp_sources=['''// #define _OPENMP
# #include <algorithm>
# #include <ATen/Parallel.h>
# #include <ATen/ParallelOpenMP.h>
# // #include <ATen/ParallelNativeTBB.h>
# #include <torch/extension.h>

# #include <vector>

# /*
# std::vector<torch::Tensor> sortPointSet( torch::Tensor points, torch::Tensor supports){
#   auto hMax = at::max(supports);
#   // std::cout << "Output from pytorch module" << std::endl;
#   // std::cout << "hMax " << hMax << std::endl;
#   auto qMin = std::get<0>(at::min(points,0)) - hMax;
#   auto qMax = std::get<0>(at::max(points,0)) + 2 * hMax;
#   // std::cout << "qMin " << qMin << std::endl;
#   // std::cout << "qMax " << qMax << std::endl;

#   auto qEx = qMax - qMin;
#   // std::cout << "qEx: " << qEx;
  
#   auto cells = at::ceil(qEx / hMax).to(torch::kInt);
#   // std::cout << "Cells: " << cells;
#   auto indices = at::ceil((points - qMin) / hMax).to(torch::kInt);

#   // auto linearIndices = at::empty({points.size(0)}, torch::TensorOptions().dtype(torch::kInt));

#   auto linearIndices = indices.index({torch::indexing::Slice(), 0}) + cells[0] * indices.index({torch::indexing::Slice(), 1});
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << linearIndices << std::endl;

#   auto indexAccessor = indices.accessor<int32_t, 2>();
#   auto linearIndexAccessor = linearIndices.accessor<int32_t, 1>();
#   auto cols = cells[0].item<int32_t>();
#   int32_t batch_size = indices.size(0); 
#   // at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
#   //   for (int32_t b = start; b < end; b++) {
#   //     linearIndexAccessor[b] = indexAccessor[b][0] + cols * indexAccessor[b][1];
#   //     // linearIndices[b] = indices[b][0] + cells[0] * indices[b][1];
#   //   }
#   // });

#   auto sorted = torch::argsort(linearIndices);
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;

#   auto sortedIndices = torch::clone(linearIndices);
#   auto sortedPositions = torch::clone(points);
#   auto sortedSupport = torch::clone(supports);

#   auto sort_ = sorted.accessor<int32_t, 1>();
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
#   auto sortedIndex_ = sortedIndices.accessor<int32_t, 1>();
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
#   auto sortedPosition_ = sortedPositions.accessor<float, 2>();
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
#   auto sortedSupport_ = sortedSupport.accessor<float, 1>();
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
#   auto points_ = points.accessor<float, 2>();
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
#   auto supports_ = supports.accessor<float,1>();
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;

#   at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
#     for (int32_t b = start; b < end; b++) {
#       auto i = sort_[b];
#       sortedIndex_[b] = linearIndexAccessor[i];
#       sortedPosition_[b][0] = points_[i][0];
#       sortedPosition_[b][1] = points_[i][1];
#       sortedSupport_[b] = supports_[i];
#     }
#   });
#   // auto b = 0;
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << sorted[b] << std::endl;
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << points[sort_[b]] << std::endl;
#   // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
#   // sortedPositions[sorted] = points;

#   // auto sortedIndices = linearIndices[sorted];
#   // auto sortedPositions = points[sorted];
#   // auto sortedSupport = supports[sorted];
#   return {qMin, hMax, cells, sortedPositions, sortedSupport, sortedIndices, sorted};


#   // std::cout << "indices: " << indices;


#   torch::Tensor z_out = at::empty({points.size(0)}, points.options());

#   return {z_out};
# }
# */

# std::pair<int32_t, int32_t> queryHashMap(int32_t qIDx, int32_t qIDy, at::TensorAccessor<int32_t,2> hashTable, at::TensorAccessor<int32_t,1> cellIndices, at::TensorAccessor<int32_t,1> cumCell,  at::TensorAccessor<int32_t,1> cellSpan, int32_t cellsX, 
#   int32_t hashMapLength){
#     if(qIDx < 0 || qIDy < 0) return {-1,-1};
#     auto qLin = qIDx + cellsX * qIDy;
#     auto qHash = (qIDx * 3 +  qIDy * 5)%  hashMapLength;
#     auto hashEntries = hashTable[qHash];
#     if(hashEntries[0] == -1)
#       return {-1,-1};
#     auto hashIndices = hashEntries[0];
#     auto minIter = hashEntries[0];
#     auto maxIter = (hashEntries[0] + hashEntries[1]);
#     for(int32_t i = minIter; i < maxIter; i++){
#         auto hashIndex = i;
#         auto cellIndex = cellIndices[hashIndex];
#         if(cellIndex == qLin){
#             auto minCellIter = cellSpan[hashIndex];
#             auto maxCellIter = (cellSpan[hashIndex] + cumCell[hashIndex]);  
#             return {minCellIter,maxCellIter};
#         }
#     }
#     return {-1,-1}; 
# }


# #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
# #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
# #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

# std::pair<torch::Tensor,torch::Tensor> countNeighborsCUDAImpl(
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor sortedParticles, torch::Tensor sortedSupport,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     torch::Tensor qMin_,
#     float hMax,
#     int32_t cellsX, int32_t hashMapLength, int32_t searchRadius);

# std::pair<torch::Tensor,torch::Tensor> constructNeighborsCUDAImpl(
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor sortedParticles, torch::Tensor sortedSupport,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     torch::Tensor qMin_,
#     torch::Tensor counters,
#     torch::Tensor offsets,
#     float hMax,
#     int32_t cellsX, int32_t hashMapLength, int32_t searchRadius);

# std::vector<torch::Tensor> buildNeighborListCUDA(
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor sortedParticles, torch::Tensor sortedSupport,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     torch::Tensor qMin_,
#     float hMax,
#     int32_t cellsX, int32_t hashMapLength, int32_t searchRadius)
# {
#   CHECK_INPUT(queryParticles_);
#   CHECK_INPUT(support_);
#   CHECK_INPUT(sortedParticles);
#   CHECK_INPUT(sortedSupport);
#   CHECK_INPUT(hashTable_);
#   CHECK_INPUT(cellIndices_);
#   CHECK_INPUT(cumCell_);
#   CHECK_INPUT(cellSpan_);
#   CHECK_INPUT(sort_);
#   CHECK_INPUT(qMin_);

#   auto neighCount = countNeighborsCUDAImpl(queryParticles_, support_, sortedParticles, sortedSupport, hashTable_, cellIndices_, cumCell_, cellSpan_, sort_, qMin_, hMax, cellsX, hashMapLength, searchRadius);
#   // return {neighCount.first, neighCount.second};

#   auto neighborList = constructNeighborsCUDAImpl(queryParticles_, support_, sortedParticles, sortedSupport, hashTable_, cellIndices_, cumCell_, cellSpan_, sort_, qMin_, neighCount.first, neighCount.second, hMax, cellsX, hashMapLength, searchRadius);

#   return {neighborList.first, neighborList.second};
# }

# std::vector<torch::Tensor> buildNeighborList(
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     int32_t cellsX, int32_t hashMapLength, int32_t searchRadius)
# {
 
#     auto queryParticles = queryParticles_.accessor<float, 2>();
#     auto support = support_.accessor<float, 1>();
#     auto hashTable = hashTable_.accessor<int32_t, 2>();
#     auto cumCell = cumCell_.accessor<int32_t, 1>();
#     auto cellIndices = cellIndices_.accessor<int32_t, 1>();
#     auto cellSpan = cellSpan_.accessor<int32_t, 1>();
#     auto sort = sort_.accessor<int32_t, 1>();

#     std::mutex m;
#     std::vector<std::vector<int32_t>> globalRows;
#     std::vector<std::vector<int32_t>> globalCols;
#     // std::vector<std::vector<int32_t>> neighborCounters;

#     int32_t batch_size = cellIndices.size(0);
#     at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end){
#       std::vector<int32_t> rows, cols;
#       for (int32_t b = start; b < end; b++) {
#         auto cell = cellIndices[b];
#         auto qIDx = cell % cellsX;
#         auto qIDy = cell / cellsX;
#         auto indexPair = queryHashMap(qIDx, qIDy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
#         for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
#           for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
#             auto currentIndexPair = queryHashMap(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
#             if(currentIndexPair.first == -1) continue;
#             for(int32_t i = indexPair.first; i < indexPair.second; ++i){
#               auto xi = queryParticles[i];
#               auto hi = support[i];
#               for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
#                 auto xj = queryParticles[j];
#                 auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
#                 if( dist < hi){
#                   rows.push_back(sort[i]);
#                   cols.push_back(sort[j]);
#                 }
#               }
#             }
#           }
#         }
#       }
#       if(rows.size() > 0)
#       {
#     // std::cout << rows.size() << std::endl;
#         std::lock_guard<std::mutex> lg(m);
#         globalRows.push_back(rows);
#         globalCols.push_back(cols);
#       } 
#     });

#     int32_t totalElements = 0;
#     for (const auto &v : globalRows)
#         totalElements += (int32_t)v.size();
#     if (totalElements == 0)
#       return {at::empty({0},  torch::TensorOptions().dtype(torch::kInt)), at::empty({0},  torch::TensorOptions().dtype(torch::kInt))};

#     // std::cout << totalElements << std::endl;
    
#         // return {at::empty({0},  torch::TensorOptions().dtype(torch::kInt)), at::empty({0},  torch::TensorOptions().dtype(torch::kInt))};
#     auto rowTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
#     auto colTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
#     std::size_t offset = 0;
#     for (std::size_t i = 0; i < globalRows.size(); ++i){
#         memcpy(rowTensor.data_ptr() + offset, globalRows[i].data(), globalRows[i].size() * sizeof(int32_t));
#         memcpy(colTensor.data_ptr() + offset, globalCols[i].data(), globalCols[i].size() * sizeof(int32_t));
#         offset += globalCols[i].size() * sizeof(int32_t);
#     }
#     return {rowTensor, colTensor};
# }

# std::vector<torch::Tensor> buildNeighborListUnsortedPerParticle(
#     torch::Tensor inputParticles_, torch::Tensor inputSupport_,
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     int32_t cellsX, int32_t hashMapLength, torch::Tensor qMin_, float hMax,int32_t searchRadius)
# {
#     auto inputParticles = inputParticles_.accessor<float, 2>();
#     auto inputSupport = inputSupport_.accessor<float, 1>();
#     auto queryParticles = queryParticles_.accessor<float, 2>();
#     auto support = support_.accessor<float, 1>();
#     auto hashTable = hashTable_.accessor<int32_t, 2>();
#     auto cumCell = cumCell_.accessor<int32_t, 1>();
#     auto cellIndices = cellIndices_.accessor<int32_t, 1>();
#     auto cellSpan = cellSpan_.accessor<int32_t, 1>();
#     auto sort = sort_.accessor<int32_t, 1>();
#     auto qMin = qMin_.accessor<float, 1>();

#     std::mutex m;
#     std::vector<std::vector<int32_t>> globalRows;
#     std::vector<std::vector<int32_t>> globalCols;

#     int32_t batch_size = inputParticles.size(0);
#     at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end){
#       std::vector<int32_t> rows, cols;
#       for (int32_t b = start; b < end; b++) {
#         auto xi = inputParticles[b];
#         auto hi = inputSupport[b];
#         int32_t qIDx = ceil((xi[0] - qMin[0]) / hMax);
#         int32_t qIDy = ceil((xi[1] - qMin[1]) / hMax);

#         for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
#           for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
#             auto currentIndexPair = queryHashMap(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
#             if(currentIndexPair.first == -1) continue;
#             for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
#               auto xj = queryParticles[j];
#               auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
#               if( dist < hi){
#                 rows.push_back(b);
#                 cols.push_back(sort[j]);
#               }
#             }
#           }
#         }
#       }
#       if(rows.size() > 0)
#       {
#         std::lock_guard<std::mutex> lg(m);
#         globalRows.push_back(rows);
#         globalCols.push_back(cols);
#       } 
#     });

#     int32_t totalElements = 0;
#     for (const auto &v : globalRows)
#         totalElements += (int32_t)v.size();
#     if (totalElements == 0)
#       return {at::empty({0},  torch::TensorOptions().dtype(torch::kInt)), at::empty({0},  torch::TensorOptions().dtype(torch::kInt))};

#     auto rowTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
#     auto colTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
#     std::size_t offset = 0;
#     for (std::size_t i = 0; i < globalRows.size(); ++i){
#         memcpy(rowTensor.data_ptr() + offset, globalRows[i].data(), globalRows[i].size() * sizeof(int32_t));
#         memcpy(colTensor.data_ptr() + offset, globalCols[i].data(), globalCols[i].size() * sizeof(int32_t));
#         offset += globalCols[i].size() * sizeof(int32_t);
#     }
#     return {rowTensor, colTensor};
# }

# std::vector<torch::Tensor> buildNeighborListAsymmetric(
#     torch::Tensor queryParticlesA_, torch::Tensor supportA_,
#     torch::Tensor hashTableA_,
#     torch::Tensor cellIndicesA_, torch::Tensor cumCellA_, torch::Tensor cellSpanA_,
#     torch::Tensor sortA_,
#     torch::Tensor queryParticlesB_, torch::Tensor supportB_,
#     torch::Tensor hashTableB_,
#     torch::Tensor cellIndicesB_, torch::Tensor cumCellB_, torch::Tensor cellSpanB_,
#     torch::Tensor sortB_,
#     int32_t cellsX, int32_t hashMapLength)
# {
#     auto queryParticlesA = queryParticlesA_.accessor<float, 2>();
#     auto supportA = supportA_.accessor<float, 1>();
#     auto hashTableA = hashTableA_.accessor<int32_t, 2>();
#     auto cumCellA = cumCellA_.accessor<int32_t, 1>();
#     auto cellIndicesA = cellIndicesA_.accessor<int32_t, 1>();
#     auto cellSpanA = cellSpanA_.accessor<int32_t, 1>();
#     auto sortA = sortA_.accessor<int32_t, 1>();
#     auto queryParticlesB = queryParticlesB_.accessor<float, 2>();
#     auto supportB = supportB_.accessor<float, 1>();
#     auto hashTableB = hashTableB_.accessor<int32_t, 2>();
#     auto cumCellB = cumCellB_.accessor<int32_t, 1>();
#     auto cellIndicesB = cellIndicesB_.accessor<int32_t, 1>();
#     auto cellSpanB = cellSpanB_.accessor<int32_t, 1>();
#     auto sortB = sortB_.accessor<int32_t, 1>();

#     std::mutex m;
#     std::vector<std::vector<int32_t>> globalRows;
#     std::vector<std::vector<int32_t>> globalCols;

#     int32_t batch_size = cellIndicesA.size(0);
#     at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end){
#       std::vector<int32_t> rows, cols;
#       for (int32_t b = start; b < end; b++) {
#         auto cell = cellIndicesA[b];
#         auto qIDx = cell % cellsX;
#         auto qIDy = cell / cellsX;
#         auto indexPair = queryHashMap(qIDx, qIDy, hashTableA, cellIndicesA, cumCellA, cellSpanA, cellsX, hashMapLength);
#         for (int32_t xx = -1; xx<= 1; ++xx){
#           for (int32_t yy = -1; yy<= 1; ++yy){
#             auto currentIndexPair = queryHashMap(qIDx + xx, qIDy + yy, hashTableB, cellIndicesB, cumCellB, cellSpanB, cellsX, hashMapLength);
#             if(currentIndexPair.first == -1) continue;
#             for(int32_t i = indexPair.first; i < indexPair.second; ++i){
#               auto xi = queryParticlesA[i];
#               auto hi = supportA[i];
#               for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
#                 auto xj = queryParticlesB[j];
#                 auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
#                 if( dist < hi){
#                   rows.push_back(sortA[i]);
#                   cols.push_back(sortB[j]);
#                 }
#               }
#             }
#           }
#         }
#       }
#       {
#         std::lock_guard<std::mutex> lg(m);
#         globalRows.push_back(rows);
#         globalCols.push_back(cols);
#       } 
#     });

#     int32_t totalElements = 0;
#     for (const auto &v : globalRows)
#         totalElements += (int32_t)v.size();

#     auto rowTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
#     auto colTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
#     std::size_t offset = 0;
#     for (std::size_t i = 0; i < globalRows.size(); ++i){
#         memcpy(rowTensor.data_ptr() + offset, globalRows[i].data(), globalRows[i].size() * sizeof(int32_t));
#         memcpy(colTensor.data_ptr() + offset, globalCols[i].data(), globalCols[i].size() * sizeof(int32_t));
#         offset += globalCols[i].size() * sizeof(int32_t);
#     }
#     return {rowTensor, colTensor};
# }


# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#   m.def("buildNeighborListCUDA", &buildNeighborListCUDA, "LLTM backward (CUDA)");
#   m.def("buildNeighborList", &buildNeighborList, "LLTM backward (CUDA)");
#   m.def("buildNeighborListAsymmetric", &buildNeighborListAsymmetric, "LLTM backward (CUDA)");
#   m.def("buildNeighborListUnsortedPerParticle", &buildNeighborListUnsortedPerParticle, "LLTM backward (CUDA)");
# }
# '''], cuda_sources=['''
# #ifdef __INTELLISENSE__
# #define __CUDACC__
# #define __device__
# #endif


# #include <torch/extension.h>

# #include <cuda.h>
# #include <cuda_runtime.h>

# #include <utility>
# #include <vector>

# namespace {
# // template <typename scalar_t>
# // __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
# //   return 1.0 / (1.0 + exp(-z));
# // }

# // template <typename scalar_t>
# // __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
# //   const auto s = sigmoid(z);
# //   return (1.0 - s) * s;
# // }

# // template <typename scalar_t>
# // __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
# //   const auto t = tanh(z);
# //   return 1 - (t * t);
# // }

# // template <typename scalar_t>
# // __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
# //   return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
# // }

# // template <typename scalar_t>
# // __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
# //   const auto e = exp(z);
# //   const auto d_relu = z < 0.0 ? 0.0 : 1.0;
# //   return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
# // }

# // template <typename scalar_t>
# // __global__ void lltm_cuda_forward_kernel(
# //     const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
# //     const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
# //     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
# //     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
# //     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
# //     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
# //     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
# //   //batch index
# //   const int n = blockIdx.y;
# //   // column index
# //   const int c = blockIdx.x * blockDim.x + threadIdx.x;
# //   if (c < gates.size(2)){
# //     input_gate[n][c] = sigmoid(gates[n][0][c]);
# //     output_gate[n][c] = sigmoid(gates[n][1][c]);
# //     candidate_cell[n][c] = elu(gates[n][2][c]);
# //     new_cell[n][c] =
# //         old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
# //     new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
# //   }
# // }

# using int2Ptr_t = torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits>;
# using intPtr_t = torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits>;
# template<typename scalar_t>
# using scalar2Ptr_t = torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>;
# template<typename scalar_t>
# using scalarPtr_t = torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits>;

# __device__ ::std::pair<int64_t, int64_t> queryHashMapCUDA(int32_t qIDx, int32_t qIDy, 
#   const int2Ptr_t hashTable, const intPtr_t cellIndices, const intPtr_t cumCell,  const intPtr_t cellSpan, int32_t cellsX, 
#   int32_t hashMapLength){
    
#     if(qIDx < 0 || qIDy < 0) return {-1,-1};
#     auto qLin = qIDx + cellsX * qIDy;
#     auto qHash = (qIDx * 3 +  qIDy * 5)%  hashMapLength;
#     auto hashEntries = hashTable[qHash];
#     if(hashEntries[0] == -1)
#       return {-1,-1};
#     auto hashIndices = hashEntries[0];
#     auto minIter = hashEntries[0];
#     auto maxIter = (hashEntries[0] + hashEntries[1]);
#     for(int32_t i = minIter; i < maxIter; i++){
#         auto hashIndex = i;
#         auto cellIndex = cellIndices[hashIndex];
#         if(cellIndex == qLin){
#             auto minCellIter = cellSpan[hashIndex];
#             auto maxCellIter = (cellSpan[hashIndex] + cumCell[hashIndex]);  
#             return {minCellIter,maxCellIter};
#         }
#     }
#     return {-1,-1}; 
# }

# template <typename scalar_t>
# __global__ void neighborSearchCUDAKernel(
#     const scalar2Ptr_t<scalar_t> queryParticles,
#     const scalarPtr_t<scalar_t> support,
#     const scalar2Ptr_t<scalar_t> sortedParticles,
#     const scalarPtr_t<scalar_t> sortedSupport,
#     const int2Ptr_t hashTable,
#     const intPtr_t cellIndices,
#     const intPtr_t cumCell,
#     const intPtr_t cellSpan,
#     const intPtr_t sort,
#     intPtr_t counter,
#     const scalarPtr_t<scalar_t> qMin,
#     float hMax,
#     const int32_t cellsX, int32_t hashMapLength, int32_t numParticles, int32_t searchRadius) {

#   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
#   if(idx >= numParticles)
#     return;
    
#       auto xi = queryParticles[idx];
#       auto hi = support[idx];
#       int32_t qIDx = ceil((xi[0] - qMin[0]) / hMax);
#       int32_t qIDy = ceil((xi[1] - qMin[1]) / hMax);

#       int32_t numNeighbors = 0;

#       for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
#         for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
#           auto currentIndexPair = queryHashMapCUDA(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
#           if(currentIndexPair.first == -1) continue;
#           for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
#             auto xj = sortedParticles[j];
#             auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
#             if( dist < hi){
#               ++numNeighbors;
#               // rows.push_back(b);
#               // cols.push_back(sort[j]);
#             }
#           }
#         }
#       }

#     counter[idx] = numNeighbors;
#     }

#         // queryParticles_.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
#         // support_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
#         // sortedParticles.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
#         // sortedSupport.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
#         // hashTable_.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
#         // cellIndices_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         // cumCell_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         // cellSpan_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         // sort_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         // qMin_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),    
#         // offset.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
#         // neighborListI.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
#         // neighborListJ.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),        
# template <typename scalar_t>
# __global__ void constructNeighborhoodsCUDA(
#     const scalar2Ptr_t<scalar_t> queryParticles,
#     const scalarPtr_t<scalar_t> support,
#     const scalar2Ptr_t<scalar_t> sortedParticles,
#     const scalarPtr_t<scalar_t> sortedSupport,
#     const int2Ptr_t hashTable,
#     const intPtr_t cellIndices,
#     const intPtr_t cumCell,
#     const intPtr_t cellSpan,
#     const intPtr_t sort,
#     const scalarPtr_t<scalar_t> qMin,
#     const intPtr_t offsets,
#     intPtr_t neighborListI,
#     intPtr_t neighborListJ,
#     float hMax,
#     const int32_t cellsX, int32_t hashMapLength, int32_t numParticles, int32_t searchRadius) {

#   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
#   if(idx >= numParticles)
#     return;
    
#       auto xi = queryParticles[idx];
#       auto hi = support[idx];
#       auto offset = idx == 0 ? 0 : offsets[idx-1];
#       int32_t qIDx = ceil((xi[0] - qMin[0]) / hMax);
#       int32_t qIDy = ceil((xi[1] - qMin[1]) / hMax);

#       int32_t numNeighbors = 0;

#       for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
#         for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
#           auto currentIndexPair = queryHashMapCUDA(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
#           if(currentIndexPair.first == -1) continue;
#           for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
#             auto xj = sortedParticles[j];
#             auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
#             if( dist < hi){
#               neighborListI[offset + numNeighbors] = idx;
#               neighborListJ[offset + numNeighbors] = sort[j];
#               // rows.push_back(b);
#               // cols.push_back(sort[j]);
#               ++numNeighbors;
#             }
#           }
#         }
#       }

#     // counter[idx] = numNeighbors;
#     }
  
# } // namespace


# std::pair<torch::Tensor,torch::Tensor> countNeighborsCUDAImpl(
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor sortedParticles, torch::Tensor sortedSupport,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     torch::Tensor qMin_,
#     float hMax,
#     int32_t cellsX, int32_t hashMapLength, int32_t searchRadius){
#       auto counter = torch::zeros({support_.size(0)}, torch::TensorOptions()
#           .dtype(torch::kInt32)
#           .layout(torch::kStrided)
#         .device(torch::kCUDA, queryParticles_.get_device()));

#         auto numParticles = queryParticles_.size(0);
#         // std::cout << "Number of particles: " << numParticles << std::endl;

#   int32_t threads = 1024;
#   int32_t blocks = (int32_t) floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);
#   // std::cout << "Launching with: " << blocks << " @ " << threads << " tpb for " << numParticles << " particles." << std::endl;

#   AT_DISPATCH_FLOATING_TYPES(queryParticles_.type(), "lltm_forward_cuda", ([&] {
#     neighborSearchCUDAKernel<scalar_t><<<blocks, threads>>>(
#         queryParticles_.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
#         support_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
#         sortedParticles.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
#         sortedSupport.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
#         hashTable_.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
#         cellIndices_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         cumCell_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         cellSpan_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         sort_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         counter.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         qMin_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),        
#         hMax, cellsX, hashMapLength, numParticles, searchRadius);
#   }));
#   auto offsets  = torch::cumsum(counter, 0);
#   offsets = offsets.to(torch::kInt);
#         return {counter, offsets};
#     }

# std::pair<torch::Tensor,torch::Tensor> constructNeighborsCUDAImpl(
#     torch::Tensor queryParticles_, torch::Tensor support_,
#     torch::Tensor sortedParticles, torch::Tensor sortedSupport,
#     torch::Tensor hashTable_,
#     torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
#     torch::Tensor sort_,
#     torch::Tensor qMin_,
#     torch::Tensor counter,
#     torch::Tensor offset,
#     float hMax,
#     int32_t cellsX, int32_t hashMapLength, int32_t searchRadius){

#       int32_t numElements = torch::max(offset).item<int32_t>();

#       auto neighborListI = torch::zeros({numElements}, torch::TensorOptions()
#           .dtype(torch::kInt32)
#           .layout(torch::kStrided)
#         .device(torch::kCUDA, queryParticles_.get_device()));
#       auto neighborListJ = torch::zeros({numElements}, torch::TensorOptions()
#           .dtype(torch::kInt32)
#           .layout(torch::kStrided)
#         .device(torch::kCUDA, queryParticles_.get_device()));



#         auto numParticles = queryParticles_.size(0);
#         // std::cout << "Number of particles: " << numParticles << std::endl;

#   int32_t threads = 1024;
#   int32_t blocks = (int32_t) floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);
#   // std::cout << "Launching with: " << blocks << " @ " << threads << " tpb for " << numParticles << " particles." << std::endl;
  

#   AT_DISPATCH_FLOATING_TYPES(queryParticles_.type(), "lltm_forward_cuda", ([&] {
#     constructNeighborhoodsCUDA<scalar_t><<<blocks, threads>>>(
#         queryParticles_.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
#         support_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
#         sortedParticles.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
#         sortedSupport.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
#         hashTable_.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
#         cellIndices_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         cumCell_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         cellSpan_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         sort_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
#         qMin_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),    
#         offset.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
#         neighborListI.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
#         neighborListJ.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),        
#         hMax, cellsX, hashMapLength, numParticles, searchRadius);
#   }));
#   // auto offsets  = torch::cumsum(counter, 0) - counter[0];
#         return {neighborListI, neighborListJ};
#     }

# // std::vector<torch::Tensor> lltm_cuda_forward(
# //     torch::Tensor input,
# //     torch::Tensor weights,
# //     torch::Tensor bias,
# //     torch::Tensor old_h,
# //     torch::Tensor old_cell) {
# //   auto X = torch::cat({old_h, input}, /*dim=*/1);
# //   auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

# //   const auto batch_size = old_cell.size(0);
# //   const auto state_size = old_cell.size(1);

# //   auto gates = gate_weights.reshape({batch_size, 3, state_size});
# //   auto new_h = torch::zeros_like(old_cell);
# //   auto new_cell = torch::zeros_like(old_cell);
# //   auto input_gate = torch::zeros_like(old_cell);
# //   auto output_gate = torch::zeros_like(old_cell);
# //   auto candidate_cell = torch::zeros_like(old_cell);

# //   const int threads = 1024;
# //   const dim3 blocks((state_size + threads - 1) / threads, batch_size);

# //   AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
# //     lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
# //         gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
# //         old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
# //         new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
# //         new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
# //         input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
# //         output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
# //         candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
# //   }));

# //   return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
# // }
 
# // std::vector<torch::Tensor> lltm_cuda_backward(
# //     torch::Tensor grad_h,
# //     torch::Tensor grad_cell,
# //     torch::Tensor new_cell,
# //     torch::Tensor input_gate,
# //     torch::Tensor output_gate,
# //     torch::Tensor candidate_cell,
# //     torch::Tensor X,
# //     torch::Tensor gates,
# //     torch::Tensor weights) {
# //   auto d_old_cell = torch::zeros_like(new_cell);
# //   auto d_gates = torch::zeros_like(gates);

# //   const auto batch_size = new_cell.size(0);
# //   const auto state_size = new_cell.size(1);

# //   const int threads = 1024;
# //   const dim3 blocks((state_size + threads - 1) / threads, batch_size);

# //   AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_forward_cuda", ([&] {
# //     lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
# //         d_old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         d_gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
# //         grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         grad_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
# //         gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
# //   }));

# //   auto d_gate_weights = d_gates.flatten(1, 2);
# //   auto d_weights = d_gate_weights.t().mm(X);
# //   auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

# //   auto d_X = d_gate_weights.mm(weights);
# //   auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
# //   auto d_input = d_X.slice(/*dim=*/1, state_size);

# //   return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
# // }
# '''], verbose=False, extra_cflags=['-fopenmp', '-O3', '-march=native'])

@torch.jit.script
def sortPositions(queryParticles, querySupport, supportScale :float = 1.0, qMin : Optional[torch.Tensor]  = None, qMax : Optional[torch.Tensor]  = None):
    with record_function("sort"): 
        with record_function("sort - bound Calculation"): 
            hMax = torch.max(querySupport)
            if qMin is None:
                qMin = torch.min(queryParticles,dim=0)[0] - hMax * supportScale
            else:
                qMin = qMin  - hMax * supportScale
            if qMax is None:
                qMax = torch.max(queryParticles,dim=0)[0] + 2 * hMax * supportScale
            else:
                qMax = qMax + 2 * hMax * supportScale
        with record_function("sort - index Calculation"): 
            qExtent = qMax - qMin
            cellCount = torch.ceil(qExtent / (hMax * supportScale)).to(torch.int32)
            indices = torch.ceil((queryParticles - qMin) / hMax).to(torch.int32)
            linearIndices = indices[:,0] + cellCount[0] * indices[:,1]
        with record_function("sort - actual argsort"): 
            sortingIndices = torch.argsort(linearIndices)
        with record_function("sort - sorting data"): 
            sortedLinearIndices = linearIndices[sortingIndices]
            sortedPositions = queryParticles[sortingIndices,:]
            sortedSupport = querySupport[sortingIndices]
    return sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, \
            int(cellCount[0]), qMin, float(hMax)


@torch.jit.script
def constructHashMap(sortedPositions, sortedSupport, sortedIndices, sort, hashMapLength : int, cellCount : int):
    with record_function("hashmap"): 
        # First create a list of occupied cells and then create a cumsum to figure out where each cell starts in the data
        with record_function("hashmap - cell cumulation"): 
            cellIndices, cellCounters = torch.unique_consecutive(sortedIndices, return_counts=True, return_inverse=False)
            cellCounters = cellCounters.to(torch.int32)
            cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0)))[:-1].to(torch.int32)
            
        # Now compute the hash indices of all particles by reversing the linearIndices            
        with record_function('hashmap - compute indices'): 
            xIndices = cellIndices % cellCount
            yIndices = torch.div(cellIndices, cellCount, rounding_mode='trunc')
            hashedIndices = (xIndices * 3 + yIndices * 5) % hashMapLength
        # Sort the hashes and use unique consecutive to find hash collisions. Then resort the cell indices based on the hash indices
        with record_function('hashmap - sort hashes'): 
            hashIndexSorting = torch.argsort(hashedIndices)
        with record_function('hashmap - collision detection'): 
            hashMap, hashMapCounters = torch.unique_consecutive(hashedIndices[hashIndexSorting], return_counts=True, return_inverse=False)
            hashMapCounters = hashMapCounters.to(torch.int32)
            cellIndices = cellIndices[hashIndexSorting]
            cellSpan = cumCell[hashIndexSorting]
            cumCell = cellCounters[hashIndexSorting]
        # Now construct the hashtable
        with record_function('hashmap - hashmap construction'):
            hashTable = hashMap.new_ones(hashMapLength,2) * -1
            hashTable[:,1] = 0
            hashMap64 = hashMap.to(torch.int64)
            hashTable[hashMap64,0] = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellIndices.dtype),torch.cumsum(hashMapCounters,dim=0)))[:-1].to(torch.int32) #torch.cumsum(hashMapCounters, dim = 0) #torch.arange(hashMap.shape[0], device=hashMap.device)
            hashTable[hashMap64,1] = hashMapCounters
    return hashTable, cellIndices, cumCell, cellSpan

# @torch.jit.script
def constructNeighborhoods(queryPositions, querySupports, hashMapLength :int = -1, supportScale : float = 1.0, minCoord : Optional[torch.Tensor]  = None, maxCoord : Optional[torch.Tensor]  = None, searchRadius : int = 1):
    with record_function('sortPositions'):
        sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, cellCount, qMin, hMax = sortPositions(queryPositions, querySupports, 1.0, minCoord, maxCoord)
    if hashMapLength == -1:
        hashMapLength = nextprime(queryPositions.shape[0])
#     return None, None, None, None, None
    with record_function('constructHashMap'):
        hashTable, cellLinearIndices, cellOffsets, cellParticleCounters = constructHashMap(sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, hashMapLength, cellCount)
        sortingIndices = sortingIndices.to(torch.int32)

    # print(hashMapLength)
    # print('sortedPositions', sortedPositions.shape, sortedPositions.dtype, sortedPositions)
    # print('sortedSupport', sortedSupport.shape, sortedSupport.dtype, sortedSupport)
    # print('sortedLinearIndices', sortedLinearIndices.shape, sortedLinearIndices.dtype, sortedLinearIndices)
    # print('cellCount', cellCount)
    # print('hashTable', hashTable.shape, hashTable.dtype, hashTable)
    # print('cellLinearIndices', cellLinearIndices.shape, cellLinearIndices.dtype, cellLinearIndices)
    # print('cellOffsets', cellOffsets.shape, cellOffsets.dtype, cellOffsets)
    # print('cellParticleCounters', cellParticleCounters.shape, cellParticleCounters.dtype, cellParticleCounters)

    # return None
    with record_function('buildNeighborList'):
        rows, cols = neighborSearch.buildNeighborList(sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, cellCount, hashMapLength, searchRadius)
    
    # return None
    return rows.to(torch.int64), cols.to(torch.int64), None, None, (qMin, hMax, sortingIndices, sortedPositions, sortedSupport), (hashTable, hashMapLength), (cellLinearIndices, cellOffsets, cellParticleCounters, cellCount)
    

def constructNeighborhoodsPreSorted(queryPositions, querySupports, particleState, hashMap, cellMap, searchRadius : int = 1):
    qMin, hMax, sortingIndices, sortedPositions, sortedSupport = particleState
    hashTable, hashMapLength = hashMap
    cellLinearIndices, cellOffsets, cellParticleCounters, cellCount = cellMap

    # print(qMin.shape, qMin.dtype)
    # print(sortedPositions.shape, sortedPositions.dtype)
    # print(sortedSupport.shape, sortedSupport.dtype)
    # print(queryPositions.shape, querySupports.dtype)
    # debugPrint(hMax)
    # debugPrint()

    with record_function('buildNeighborList'):
        rows, cols = neighborSearch.buildNeighborListUnsortedPerParticle(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, cellCount, hashMapLength, qMin.type(torch.float32), np.float32(hMax), searchRadius)
        
    
    return rows.to(torch.int64), cols.to(torch.int64)
    


def constructNeighborhoodsCUDA(queryPositions, querySupports, hashMapLength :int = -1, supportScale : float = 1.0, minCoord : Optional[torch.Tensor]  = None, maxCoord : Optional[torch.Tensor]  = None, searchRadius : int = 1):
    with record_function('sortPositions'):
        sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, cellCount, qMin, hMax = sortPositions(queryPositions, querySupports, 1.0, minCoord, maxCoord)
    if hashMapLength == -1:
        hashMapLength = nextprime(queryPositions.shape[0])
#     return None, None, None, None, None
    with record_function('constructHashMap'):
        hashTable, cellLinearIndices, cellOffsets, cellParticleCounters = constructHashMap(sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, hashMapLength, cellCount)
    sortingIndices = sortingIndices.to(torch.int32)
# 

    with record_function('buildNeighborList'):
        # ctr, offsets, i, j = neighborSearch.buildNeighborListCUDA(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin, hMax, cellCount, hashMapLength, searchRadius)
        i, j = neighborSearch.buildNeighborListCUDA(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin, hMax, cellCount, hashMapLength, searchRadius)
    with record_function('finalize'):        
        j, jj = torch.sort(j, dim = 0, stable = True)
        i = i[jj]
        i, ii = torch.sort(i, dim = 0, stable = True)
        j = j[ii]
       
    return i.to(torch.int64), j.to(torch.int64), None, None, (qMin, hMax, sortingIndices, sortedPositions, sortedSupport), (hashTable, hashMapLength), (cellLinearIndices, cellOffsets, cellParticleCounters, cellCount)

    return i.to(torch.int64), j.to(torch.int64), ctr, offsets, (qMin, hMax, sortingIndices, sortedPositions, sortedSupport), (hashTable, hashMapLength), (cellLinearIndices, cellOffsets, cellParticleCounters, cellCount)

def constructNeighborhoodsPreSortedCUDA(queryPositions, querySupports, particleState, hashMap, cellMap, searchRadius : int = 1):
    qMin, hMax, sortingIndices, sortedPositions, sortedSupport = particleState
    hashTable, hashMapLength = hashMap
    cellLinearIndices, cellOffsets, cellParticleCounters, cellCount = cellMap
    with record_function('buildNeighborList'):
        rows, cols = neighborSearch.buildNeighborListCUDA(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin.type(torch.float32), np.float32(hMax), cellCount, hashMapLength, searchRadius)
        
    
    return rows.to(torch.int64), cols.to(torch.int64)
    
    
from .periodicBC import createGhostParticlesKernel

def periodicNeighborSearch(ptcls, minDomain, maxDomain, support, periodicX, periodicY, useCompactHashMap = True):
    minD = torch.tensor(minDomain).to(ptcls.device).type(ptcls.dtype)
    maxD = torch.tensor(maxDomain).to(ptcls.device).type(ptcls.dtype)
    x = torch.remainder(ptcls - minD, maxD - minD) + minD

    # x = torch.tensor(ptcls).type(torch.float32)
    if periodicX or periodicY:
        ghostIndices, ghostOffsets = createGhostParticlesKernel(x, minDomain, maxDomain, 1, support, periodicX, periodicY)

        indices = torch.cat(ghostIndices)
        positions = torch.cat([x[g] + offset for g, offset in zip(ghostIndices, ghostOffsets)])

        indices = torch.hstack((torch.arange(x.shape[0]).to(x.device), indices))
        y = torch.vstack((x, positions))
    else:
        y = x
        indices = torch.arange(x.shape[0]).to(x.device)

    if useCompactHashMap:
        i, j = radiusCompactHashMap(x, y, support)
    else:
        i, j = radius(x,y,support)
    i_t = indices[i]
    # ii, ni = torch.unique(i, return_counts = True)
    # jj, nj = torch.unique(j, return_counts = True)
    # i_ti, ni_t = torch.unique(i_t, return_counts = True)

#     print(x.shape,y.shape)
#     print(i, torch.min(i), torch.max(i), torch.unique(i).shape)
#     print(j, torch.min(j), torch.max(j), torch.unique(j).shape)
#     print(i_t, torch.min(i_t), torch.max(i_t), torch.unique(i_t).shape)
#     print(ni, torch.min(ni), torch.median(ni), torch.max(ni))
#     print(nj, torch.min(nj), torch.median(nj), torch.max(nj))
#     print(ni_t, torch.min(ni_t), torch.median(ni_t), torch.max(ni_t))
    
    fluidDistances = y[i] - x[j]
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidDistances[fluidRadialDistances < 1e-4 * support,:] = 0
    fluidDistances[fluidRadialDistances >= 1e-4 * support,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-4 * support,None]
    fluidRadialDistances /= support

    return i_t, j, fluidDistances, fluidRadialDistances#, x, y, ii, ni, jj, nj


def periodicNeighborSearchXY(x, ptcls, minDomain, maxDomain, support, periodicX, periodicY, useCompactHashMap = True, searchRadius = 1):
    minD = torch.tensor(minDomain).to(ptcls.device).type(ptcls.dtype)
    maxD = torch.tensor(maxDomain).to(ptcls.device).type(ptcls.dtype)
    y = torch.remainder(ptcls - minD, maxD - minD) + minD

    xx = torch.remainder(x - minD, maxD - minD) + minD

    # x = torch.tensor(ptcls).type(torch.float32)
    if periodicX or periodicY:
        ghostIndices, ghostOffsets = createGhostParticlesKernel(y, minDomain, maxDomain, 1, support, periodicX, periodicY)

        indices = torch.cat(ghostIndices)
        positions = torch.cat([y[g] + offset for g, offset in zip(ghostIndices, ghostOffsets)])

        indices = torch.hstack((torch.arange(y.shape[0]).to(y.device), indices))
        y = torch.vstack((y, positions))
    else:
        y = y
        indices = torch.arange(y.shape[0]).to(y.device)

    if useCompactHashMap:
        i, j = radiusCompactHashMap(xx, y, support, searchRadius = searchRadius)
    else:
        i, j = radius(xx,y,support)
    i_t = indices[i]
    # ii, ni = torch.unique(i, return_counts = True)
    # jj, nj = torch.unique(j, return_counts = True)
    # i_ti, ni_t = torch.unique(i_t, return_counts = True)

#     print(x.shape,y.shape)
#     print(i, torch.min(i), torch.max(i), torch.unique(i).shape)
#     print(j, torch.min(j), torch.max(j), torch.unique(j).shape)
#     print(i_t, torch.min(i_t), torch.max(i_t), torch.unique(i_t).shape)
#     print(ni, torch.min(ni), torch.median(ni), torch.max(ni))
#     print(nj, torch.min(nj), torch.median(nj), torch.max(nj))
#     print(ni_t, torch.min(ni_t), torch.median(ni_t), torch.max(ni_t))
    
    fluidDistances = y[i] - xx[j]
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidDistances[fluidRadialDistances < 1e-4 * support,:] = 0
    fluidDistances[fluidRadialDistances >= 1e-4 * support,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-4 * support,None]
    fluidRadialDistances /= support

    return i_t, j, fluidDistances, fluidRadialDistances#, x, y, ii, ni, jj, nj

def periodicNeighborSearchXYNoNorm(x, ptcls, minDomain, maxDomain, support, periodicX, periodicY, useCompactHashMap = True, searchRadius = 1):
    minD = torch.tensor(minDomain).to(ptcls.device).type(ptcls.dtype)
    maxD = torch.tensor(maxDomain).to(ptcls.device).type(ptcls.dtype)
    y = torch.remainder(ptcls - minD, maxD - minD) + minD

    xx = torch.remainder(x - minD, maxD - minD) + minD

    # x = torch.tensor(ptcls).type(torch.float32)
    if periodicX or periodicY:
        ghostIndices, ghostOffsets = createGhostParticlesKernel(y, minDomain, maxDomain, 1, support, periodicX, periodicY)

        indices = torch.cat(ghostIndices)
        positions = torch.cat([y[g] + offset for g, offset in zip(ghostIndices, ghostOffsets)])

        indices = torch.hstack((torch.arange(y.shape[0]).to(y.device), indices))
        y = torch.vstack((y, positions))
    else:
        y = y
        indices = torch.arange(y.shape[0]).to(y.device)

    with torch.no_grad():
      if useCompactHashMap:
          i, j = radiusCompactHashMap(xx, y, support, searchRadius = searchRadius)
      else:
          i, j = radius(xx,y,support)
    i_t = indices[i]
    # ii, ni = torch.unique(i, return_counts = True)
    # jj, nj = torch.unique(j, return_counts = True)
    # i_ti, ni_t = torch.unique(i_t, return_counts = True)

#     print(x.shape,y.shape)
#     print(i, torch.min(i), torch.max(i), torch.unique(i).shape)
#     print(j, torch.min(j), torch.max(j), torch.unique(j).shape)
#     print(i_t, torch.min(i_t), torch.max(i_t), torch.unique(i_t).shape)
#     print(ni, torch.min(ni), torch.median(ni), torch.max(ni))
#     print(nj, torch.min(nj), torch.median(nj), torch.max(nj))
#     print(ni_t, torch.min(ni_t), torch.median(ni_t), torch.max(ni_t))
    
    # fluidDistances = (y[i] - xx[j]) / support
    # fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    # fluidDistances[fluidRadialDistances < 1e-4 * support,:] = 0
    # fluidDistances[fluidRadialDistances >= 1e-4 * support,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-4 * support,None]
    # fluidRadialDistances /= support

    return i_t, j#, fluidDistances, fluidRadialDistances#, x, y, ii, ni, jj, nj

class neighborSearchModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
        
    def getParameters(self):
        return [
            Parameter('neighborSearch', 'gradientThreshold', 'float', 1e-7, required = False, export = True, hint = ''),
            Parameter('neighborSearch', 'supportScale', 'float', 1.0, required = False, export = True, hint = ''),
            Parameter('neighborSearch', 'sortNeighborhoods', 'bool', True, required = False, export = True, hint = '')
        ]
        
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.maxNeighbors = simulationConfig['compute']['maxNeighbors']
        self.threshold = simulationConfig['neighborSearch']['gradientThreshold']
        self.supportScale = simulationConfig['neighborSearch']['supportScale']
        self.sortNeighborhoods = simulationConfig['neighborSearch']['sortNeighborhoods']
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']

        self.minDomain = simulationConfig['domain']['min']
        self.maxDomain = simulationConfig['domain']['max']

        self.periodicX = simulationConfig['periodicBC']['periodicX']
        self.periodicY = simulationConfig['periodicBC']['periodicY']
        
    def resetState(self, simulationState):
        simulationState.pop('fluidNeighbors', None)
        simulationState.pop('fluidDistances', None)
        simulationState.pop('fluidRadialDistances', None)

    def search(self, simulationState, simulation):
        with record_function("neighborhood - fluid neighbor search"): 
            row, col, fluidDistances, fluidRadialDistances = periodicNeighborSearch(
                simulationState['fluidPosition'], self.minDomain, self.maxDomain, 
                self.support * self.supportScale, self.periodicX, self.periodicY, True)
            
            queryPositions = simulationState['fluidPosition']
            querySupports = simulationState['fluidSupport']

            # _ = constructNeighborhoods(queryPositions, querySupports, -1, minCoord = torch.tensor(self.minDomain), maxCoord = torch.tensor(self.maxDomain))


            # if queryPositions.is_cuda:
            #     row, col, ctr, offsets, self.sortInfo, self.hashMap, self.cellMap = constructNeighborhoodsCUDA(queryPositions, querySupports, -1, minCoord = torch.tensor(self.minDomain,device=self.device,dtype=self.dtype), maxCoord = torch.tensor(self.maxDomain,device=self.device,dtype=self.dtype))
            # else:
            #     row, col, ctr, offsets, self.sortInfo, self.hashMap, self.cellMap = constructNeighborhoods(queryPositions, querySupports, -1, minCoord = torch.tensor(self.minDomain), maxCoord = torch.tensor(self.maxDomain))

            fluidNeighbors = torch.stack([row, col], dim = 0)

            # fluidDistances = (simulationState['fluidPosition'][fluidNeighbors[0]] - simulationState['fluidPosition'][fluidNeighbors[1]])
            # fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

            # fluidDistances[fluidRadialDistances < self.threshold,:] = 0
            # fluidDistances[fluidRadialDistances >= self.threshold,:] /= fluidRadialDistances[fluidRadialDistances >= self.threshold,None]
            # fluidRadialDistances /= self.support

            simulationState['fluidNeighbors'] = fluidNeighbors
            simulationState['fluidDistances'] = fluidDistances
            simulationState['fluidRadialDistances'] = fluidRadialDistances

            return fluidNeighbors, fluidDistances, fluidRadialDistances
        
    def searchExisting(self, queryPositions, querySupports, simulationState, simulation, searchRadius :int = 1):
        with record_function("neighborhood - searching existing"): 
            rows, cols = periodicNeighborSearchXY(queryPositions, simulationState['fluidPosition'], self.minDomain, self.maxDomain, 
                self.support * self.supportScale, self.periodicX, self.periodicY, useCompactHashMap = True, searchRadius = searchRadius)
            
            # queryPositions = simulationState['fluidPosition'].to('cpu')
            # querySupports = simulationState['fluidSupport'].to('cpu')
            # if queryPositions.is_cuda:
            #     rows, cols = constructNeighborhoodsPreSortedCUDA(queryPositions, querySupports,  self.sortInfo, self.hashMap, self.cellMap, searchRadius = searchRadius)
            # else:
            #     rows, cols = constructNeighborhoodsPreSorted(queryPositions, querySupports,  self.sortInfo, self.hashMap, self.cellMap, searchRadius = searchRadius)
            # rows = rows.to(self.device)
            # cols = cols.to(self.device)
            
#             row, col = radius(simulationState['fluidPosition'], simulationState['fluidPosition'], self.support, max_num_neighbors = self.maxNeighbors)
            fluidNeighbors = torch.stack([rows, cols], dim = 0)

            fluidDistances = (queryPositions[fluidNeighbors[0]] - simulationState['fluidPosition'][fluidNeighbors[1]])
            fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

            fluidDistances[fluidRadialDistances < self.threshold,:] = 0
            fluidDistances[fluidRadialDistances >= self.threshold,:] /= fluidRadialDistances[fluidRadialDistances >= self.threshold,None]
            fluidRadialDistances /= querySupports[rows]

            return fluidNeighbors, fluidDistances, fluidRadialDistances


def radiusCompactHashMap(x: torch.Tensor, y: torch.Tensor, r: float, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None, searchRadius = 1):
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.
    """
    if batch_x is not None:
        batches = torch.unique(batch_x)
        iList = []
        jList = []
        for i in batches:
            xBatch = x[batch_x == i,:]
            yBatch = y[batch_y == i,:]
            xSupport = torch.ones_like(xBatch[:,0]) * r
            ySupport = torch.ones_like(yBatch[:,0]) * r

            minPos = torch.minimum(torch.min(xBatch,dim=0)[0], torch.min(yBatch,dim=0)[0])
            maxPos = torch.maximum(torch.max(xBatch,dim=0)[0], torch.max(yBatch,dim=0)[0])

            hashMapLength = nextprime(xBatch.shape[0])

            sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, cellCount, qMin, hMax = sortPositions(xBatch, xSupport, 1.0, minPos, maxPos)
            hashTable, cellLinearIndices, cellOffsets, cellParticleCounters = constructHashMap(sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, hashMapLength, cellCount)
            sortingIndices = sortingIndices.to(torch.int32)

            if not x.is_cuda:
                i, j = neighborSearch.buildNeighborListUnsortedPerParticle(
                    yBatch, ySupport, 
                    sortedPositions, sortedSupport,
                    hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, cellCount, hashMapLength, 
                    qMin.type(torch.float32), np.float32(hMax), searchRadius)
            else:
                i, j = neighborSearch.buildNeighborListCUDA(yBatch, ySupport, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin, hMax, cellCount, hashMapLength, searchRadius)
            iList.append(i)
            jList.append(j)
        return torch.hstack(iList).to(torch.int64), torch.hstack(jList).to(torch.int64)
        
    xSupport = torch.ones_like(x[:,0]) * r
    ySupport = torch.ones_like(y[:,0]) * r

    minPos = torch.minimum(torch.min(x,dim=0)[0], torch.min(y,dim=0)[0])
    maxPos = torch.maximum(torch.max(x,dim=0)[0], torch.max(y,dim=0)[0])

    hashMapLength = nextprime(x.shape[0])

    sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, cellCount, qMin, hMax = sortPositions(x, xSupport, 1.0, minPos, maxPos)
    hashTable, cellLinearIndices, cellOffsets, cellParticleCounters = constructHashMap(sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, hashMapLength, cellCount)
    sortingIndices = sortingIndices.to(torch.int32)

    if not x.is_cuda:
        i, j = neighborSearch.buildNeighborListUnsortedPerParticle(
            y, ySupport, 
            sortedPositions, sortedSupport,
            hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, cellCount, hashMapLength, 
            qMin.type(torch.float32), np.float32(hMax), searchRadius)
    else:
        i, j = neighborSearch.buildNeighborListCUDA(y, ySupport, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin, hMax, cellCount, hashMapLength, searchRadius)
    return i.to(torch.int64), j.to(torch.int64)
# i are indices in x, j are indices in y
