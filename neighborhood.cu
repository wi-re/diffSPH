#include "neighborhood.h"
template<std::size_t dim>
__global__ void buildNeighborhoodCudaDispatcher(int32_t numParticles,
                                                ptr_t<int32_t, 1> neighborOffsets, ptr_t<int32_t, 1> neighborList_i, ptr_t<int32_t, 1> neighborList_j,
                                                ptr_t<float, 2> queryPositions, ptr_t<float, 1> querySupport, int searchRange,
                                                ptr_t<float, 2> sortedPositions, ptr_t<float, 1> sortedSupport,
                                                ptr_t<int32_t, 2> hashTable, int hashMapLength,
                                                ptr_t<int64_t, 2> cellTable, ptr_t<int32_t, 1> numCells,
                                                ptr_t<int32_t, 2> offsets, float hCell, ptr_t<float, 1> minDomain, ptr_t<float, 1> maxDomain, ptr_t<int32_t, 1> periodicity,
                                                supportMode searchMode, c10::TensorOptions defaultOptions) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        buildNeighborhood<dim>(i, neighborOffsets, neighborList_i, neighborList_j, queryPositions, querySupport, searchRange, sortedPositions, sortedSupport, hashTable, hashMapLength, cellTable, numCells, offsets, hCell, minDomain, maxDomain, periodicity, searchMode, defaultOptions);
    }
}
template<std::size_t dim>
__global__ void countNeighborsForParticleCudaDispatcher(int32_t numParticles,
                                                        ptr_t<int32_t, 1> neighborCounters,
                                                        ptr_t<float, 2> queryPositions, ptr_t<float, 1> querySupport, int searchRange,
                                                        ptr_t<float, 2> sortedPositions, ptr_t<float, 1> sortedSupport,
                                                        ptr_t<int32_t, 2> hashTable, int hashMapLength,
                                                        ptr_t<int64_t, 2> cellTable, ptr_t<int32_t, 1> numCellsVec,
                                                        ptr_t<int32_t, 2> offsets,
                                                        float hCell, ptr_t<float, 1> minDomain, ptr_t<float, 1> maxDomain, ptr_t<int32_t, 1> periodicity,
                                                        supportMode searchMode, c10::TensorOptions defaultOptions) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        countNeighborsForParticle<dim>(i, neighborCounters, queryPositions, querySupport, searchRange, sortedPositions, sortedSupport, hashTable, hashMapLength, cellTable, numCellsVec, offsets, hCell, minDomain, maxDomain, periodicity, searchMode, defaultOptions);
    }
}


void buildNeighborhoodCuda(torch::Tensor neighborOffsets, torch::Tensor neighborList_i, torch::Tensor neighborList_j,
                       torch::Tensor queryPositions, torch::Tensor querySupport, int searchRange,
                       torch::Tensor sortedPositions, torch::Tensor sortedSupport,
                       torch::Tensor hashTable, int hashMapLength,
                       torch::Tensor cellTable, torch::Tensor numCells,
                       torch::Tensor offsets, float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
                       supportMode searchMode, c10::TensorOptions defaultOptions) {
    int32_t numParticles = queryPositions.size(0);
    int32_t threads = 32;
    int32_t blocks = (int32_t)floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);

#define args numParticles, \
neighborOffsets.packed_accessor32<int32_t,1, traits>(), neighborList_i.packed_accessor32<int32_t,1, traits>(), neighborList_j.packed_accessor32<int32_t,1, traits>(), \
queryPositions.packed_accessor32<float, 2, traits>(), querySupport.packed_accessor32<float,1, traits>(), searchRange, \
sortedPositions.packed_accessor32<float, 2, traits>(), sortedSupport.packed_accessor32<float,1, traits>(), \
hashTable.packed_accessor32<int32_t,2, traits>(), hashMapLength, \
cellTable.packed_accessor32<int64_t,2, traits>(), numCells.packed_accessor32<int32_t,1, traits>(), \
offsets.packed_accessor32<int32_t,2, traits>(), \
hCell, minDomain.packed_accessor32<float,1, traits>(), maxDomain.packed_accessor32<float,1, traits>(), periodicity.packed_accessor32<int32_t,1, traits>(), searchMode, defaultOptions

    int32_t dim = queryPositions.size(1);
    if(dim == 1)
        buildNeighborhoodCudaDispatcher<1><<<blocks, threads>>>(args);
    else if(dim == 2)
        buildNeighborhoodCudaDispatcher<2><<<blocks, threads>>>(args);
    else if(dim == 3)
        buildNeighborhoodCudaDispatcher<3><<<blocks, threads>>>(args);
    else throw std::runtime_error("Unsupported dimensionality");

#undef args
}

void countNeighborsForParticleCuda(
    torch::Tensor neighborCounters, 
    torch::Tensor queryPositions, torch::Tensor querySupport, int searchRange, 
    torch::Tensor sortedPositions, torch::Tensor sortedSupport,
    torch::Tensor hashTable, int hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCellsVec, 
    torch::Tensor offsets,
    float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    supportMode searchMode, c10::TensorOptions defaultOptions) {
    int32_t numParticles = queryPositions.size(0);
    int32_t threads = 32;
    int32_t blocks = (int32_t)floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);

#define args \
        numParticles, \
        neighborCounters.packed_accessor32<int32_t,1, traits>(), \
        queryPositions.packed_accessor32<float,2, traits>(), querySupport.packed_accessor32<float,1, traits>(), searchRange, \
        sortedPositions.packed_accessor32<float,2, traits>(), sortedSupport.packed_accessor32<float,1, traits>(), \
        hashTable.packed_accessor32<int32_t,2, traits>(), hashMapLength, \
        cellTable.packed_accessor32<int64_t,2, traits>(), numCellsVec.packed_accessor32<int32_t,1, traits>(), \
        offsets.packed_accessor32<int32_t,2, traits>(), \
        hCell, minDomain.packed_accessor32<float, 1, traits>(), maxDomain.packed_accessor32<float, 1, traits>(), periodicity.packed_accessor32<int32_t, 1, traits>(), searchMode, defaultOptions

    int32_t dim = queryPositions.size(1);
    // std::cout << "dim: " << dim << std::endl;
    if (dim == 1)
        countNeighborsForParticleCudaDispatcher<1><<<blocks, threads>>>(args);
    else if (dim == 2)
        countNeighborsForParticleCudaDispatcher<2><<<blocks, threads>>>(args);
    else if (dim == 3)
        countNeighborsForParticleCudaDispatcher<3><<<blocks, threads>>>(args);
    else throw std::runtime_error("Unsupported dimensionality");

#undef args
}
