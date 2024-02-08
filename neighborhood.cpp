// #define _OPENMP
#include <algorithm>
#include <ATen/Parallel.h>
#include <ATen/ParallelOpenMP.h>
// #include <ATen/ParallelNativeTBB.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/core/TensorAccessor.h>

// Define __device__ and __host__ macros for non-CUDA builds
#if !(defined(__CUDACC__) || defined(__HIPCC__))
#define __device__
#define __host__
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
using tensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using tensor32_t = torch::PackedTensorAccessor32<T, dim, traits>;

// Simple enum to specify the support mode
enum struct supportMode{
    symmetric, gather, scatter
};

// Simple helper math functions
/**
 * Calculates an integer power of a given base and exponent.
 * 
 * @param base The base.
 * @param exponent The exponent.
 * @return The calculated power.
*/
int power(int base, int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}
/**
 * Calculates the modulo of a given number n with respect to a given modulus m.
 * Works using python modulo semantics NOT C++ modulo semantics.
 * 
 * @param n The number.
 * @param m The modulus.
 * @return The calculated modulo.
 */
auto pymod(int n, int m){
    return n >= 0 ? n % m : ((n % m) + m) % m;
}
/**
 * Calculates the modulo of a given number n with respect to a given modulus m.
 * Works using python modulo semantics NOT C++ modulo semantics.
 * 
 * @param n The number.
 * @param m The modulus.
 * @return The calculated modulo.
 */
auto moduloOp(float p, float q, float h){
    return ((p - q + h / 2.0) - std::floor((p - q + h / 2.0) / h) * h) - h / 2.0;
}

/**
 * Calculates the distance between two points in a periodic domain.
 * 
 * @param x_i The first point.
 * @param x_j The second point.
 * @param minDomain The minimum domain bounds.
 * @param maxDomain The maximum domain bounds.
 * @param periodicity The periodicity flags.
 * @return The calculated distance.
 */
auto modDistance(tensor_t<float,1> x_i, tensor_t<float,1> x_j, ptr_t<float,1> minDomain, ptr_t<float,1> maxDomain, ptr_t<bool,1> periodicity){
    auto dim = x_i.size(0);
    float sum = 0.f;
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


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
 * @param verbose Flag indicating whether to print verbose information.
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
        return t.template packed_accessor32<scalar_t, dim>();
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
    return t.template packed_accessor32<scalar_t, dim>();
}

/**
 * Calculates the hash index for a given set of cell indices.
 * The hash index is used for indexing into a hash map.
 *
 * @param cellIndices The cell indices.
 * @param hashMapLength The length of the hash map.
 * @return The hash index.
 * @throws std::runtime_error if the dimension is not supported (only 1D, 2D, and 3D supported).
 */
auto hashIndexing(ptr_t<int32_t, 1> cellIndices, int32_t hashMapLength) {
    auto dim = cellIndices.size(0);
    constexpr auto primes = std::array<int32_t, 3>{73856093, 19349663, 83492791};
    if (dim == 1) {
        return cellIndices[0] % hashMapLength;
    } else if (dim == 2) {
        return (cellIndices[0] * primes[0] + cellIndices[1] * primes[1]) % hashMapLength;
    } else if (dim == 3) {
        return (cellIndices[0] * primes[0] + cellIndices[1] * primes[1] + cellIndices[2] * primes[2]) % hashMapLength;
    } else {
        throw std::runtime_error("Only 1D, 2D and 3D supported");
    }
}

/**
 * Calculates the linear index based on the given cell indices and cell counts.
 * 
 * @param cellIndices The array of cell indices.
 * @param cellCounts The array of cell counts.
 * @return The calculated linear index.
 */
auto linearIndexing(ptr_t<int32_t, 1> cellIndices, ptr_t<int32_t, 1> cellCounts) {
    auto dim = cellIndices.size(0);
    int32_t linearIndex = 0;
    int32_t product = 1;
    for (int32_t i = 0; i < dim; i++) {
        linearIndex += cellIndices[i] * product;
        product *= cellCounts[i];
    }
    return linearIndex;
}

/**
 * Queries the hash map for a given cell index and returns the corresponding cell table entry.
 * 
 * @param cellID The cell index.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCells The number of cells.
 * @return The cell table entry.
 */
std::pair<int32_t, int32_t> queryHashMap(
    ptr_t<int32_t, 1> cellID,
    ptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
    ptr_t<int64_t, 2> cellTable,
    ptr_t<int32_t, 1> numCells) {
    auto linearIndex = linearIndexing(cellID, numCells);
    auto hashedIndex = hashIndexing(cellID, hashMapLength);

    auto tableEntry = hashTable[hashedIndex];
    auto hBegin = tableEntry[0];
    auto hLength = tableEntry[1];
    if (hBegin != -1) {
        for (int32_t i = hBegin; i < hBegin + hLength; i++) {
            auto cell = cellTable[i];
            if (cell[0] == linearIndex) {
                auto cBegin = cell[1];
                auto cLength = cell[2];
                return std::pair{cBegin, cBegin + cLength};
            }
        }
    }
    return std::pair{-1, -1};
}

/**
 * Iterates over the cells in the neighborhood of a given cell and calls a given function for each cell.
 * 
 * @tparam Func The function type.
 * @param centralCell The central cell.
 * @param cellOffsets The cell offsets.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCells The number of cells.
 * @param periodicity The periodicity flags.
 * @param queryFunction The query function.
 * @param defaultOptions The default tensor options.
 */
template<typename Func>
auto iterateOffsetCells(
    ptr_t<int32_t, 1> centralCell, ptr_t<int32_t, 2> cellOffsets, 
    ptr_t<int32_t, 2> hashTable, int32_t hashMapLength, 
    ptr_t<int64_t, 2> cellTable, ptr_t<int32_t, 1> numCells, ptr_t<bool,1> periodicity, Func queryFunction, c10::TensorOptions defaultOptions){
    auto nOffsets = cellOffsets.size(0);
    auto dim = centralCell.size(0);

    for(int32_t c = 0; c < nOffsets; ++c){
        auto offset = cellOffsets[c];
        auto offsetCell = torch::zeros({centralCell.size(0)}, defaultOptions.dtype(torch::kInt32));

        for(int32_t d = 0; d < dim; ++d){
            offsetCell[d] = periodicity[d] ? pymod(centralCell[d] + offset[d],  numCells[d]) : centralCell[d] + offset[d];
        }
        auto queried = queryHashMap(offsetCell.packed_accessor32<int32_t,1>(), hashTable, hashMapLength, cellTable, numCells);
        if(queried.first != -1){
            queryFunction(queried.first, queried.second);
        }
    }
}


/***
 * @brief Counts the number of neighbors for a given particle.
 * 
 * This function counts the number of neighbors for a given particle based on the given search mode.
 * 
 * @param xi The position of the particle.
 * @param hi The support radius of the particle.
 * @param searchRange The search range.
 * @param sortedPositions The sorted positions of the particles.
 * @param sortedSupport The sorted support radii of the particles.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCellsVec The number of cells.
 * @param offsets The cell offsets.
 * @param hCell The cell size.
 * @param minDomain The minimum domain bounds.
 * @param maxDomain The maximum domain bounds.
 * @param periodicity The periodicity flags.
 * @param searchMode The search mode.
 * @param defaultOptions The default tensor options.
 * @return The number of neighbors.
*/
auto countNeighborsForParticle(
    tensor_t<float, 1> xi, float hi, int searchRange, 
    ptr_t<float, 2> sortedPositions, ptr_t<float,1> sortedSupport,
    ptr_t<int32_t, 2> hashTable, int hashMapLength,
    ptr_t<int64_t, 2> cellTable, ptr_t<int32_t,1> numCellsVec, 
    ptr_t<int32_t, 2> offsets,
    float hCell, ptr_t<float,1> minDomain, ptr_t<float,1> maxDomain, ptr_t<bool,1> periodicity,
    supportMode searchMode, c10::TensorOptions defaultOptions){
    auto dim = xi.size(0);

    auto queryCell = torch::zeros({dim}, defaultOptions.dtype(torch::kInt32));
    for(int d = 0; d < dim; d++){
        queryCell[d] = std::floor((xi[d] - minDomain[d]) / hCell);
    }    
    int32_t neighborCounter = 0;
    iterateOffsetCells(queryCell.packed_accessor32<int32_t,1>(), offsets, 
        hashTable, hashMapLength, 
        cellTable, numCellsVec, periodicity,
        [&](int32_t cBegin, int32_t cEnd){
            // std::cout << "queried: " << cBegin << " " << cEnd << " -> " << cEnd - cBegin << std::endl;

            for(int32_t j = cBegin; j < cEnd; j++){
                auto xj = sortedPositions[j];
                auto dist = modDistance(xi, xj, minDomain, maxDomain, periodicity);
                if( searchMode == supportMode::scatter && dist < sortedSupport[j])
                    neighborCounter++;
                else if( searchMode == supportMode::gather && dist < hi)
                    neighborCounter++;
                else if(searchMode == supportMode::symmetric && dist < (hi + sortedSupport[j]) / 2.f)
                    neighborCounter++;
            }
        }, defaultOptions);
    return neighborCounter;
}

/**
 * 
 * @brief C++ based neighborhood counting for all particles in the query set relative to the sorted set.
 * 
 * This function counts the number of neighbors for all particles in the query set relative to the sorted set.
 * 
 * @param queryPositions_ The positions of the query particles.
 * @param querySupport_ The support radii of the query particles.
 * @param searchRange The search range.
 * @param sortedPositions_ The sorted positions of the particles.
 * @param sortedSupport_ The sorted support radii of the particles.
 * @param hashTable_ The hash table.
 * @param hashMapLength The length of the hash map.
 * @param numCells_ The number of cells.
 * @param cellTable_ The cell table.
 * @param qMin_ The minimum domain bounds.
 * @param hCell The cell size.
 * @param maxDomain_ The maximum domain bounds.
 * @param minDomain_ The minimum domain bounds.
 * @param periodicity_ The periodicity flags.
 * @param mode The support mode.
 * @param verbose Flag indicating whether to print verbose information.
 * @return The number of neighbors for each particle in the query set.
 */
torch::Tensor countNeighbors(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, float hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false){
    if(verbose)
        std::cout << "C++: countNeighbors" << std::endl;
    // Convert the mode to an enum for easier handling
    supportMode searchMode = supportMode::symmetric;
    if(mode == "symmetric"){
        searchMode = supportMode::symmetric;
    } else if(mode == "gather"){
        searchMode = supportMode::gather;
    } else if(mode == "scatter"){
        searchMode = supportMode::scatter;
    } else {
        throw std::runtime_error("Invalid support mode: " + mode);
    }

    // Check if the input tensors are defined and contiguous and have the correct dimensions
    auto queryPositions = getAccessor<float, 2>(queryPositions_, "queryPositions", false, verbose);
    auto querySupport = getAccessor<float, 1>(querySupport_, "querySupport", false, verbose, supportMode::scatter == searchMode);
    auto sortedPositions = getAccessor<float, 2>(sortedPositions_, "sortedPositions", false, verbose);
    auto sortedSupport = getAccessor<float, 1>(sortedSupport_, "sortedSupport", false, verbose, supportMode::gather == searchMode);

    // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
    auto hashTable = getAccessor<int, 2>(hashTable_, "hashTable", false, verbose);
    auto numCells = getAccessor<int, 1>(numCells_, "numCells", false, verbose);
    auto cellTable = getAccessor<int64_t, 2>(cellTable_, "cellTable", false, verbose);
    auto qMin = getAccessor<float, 1>(qMin_, "qMin", false, verbose);
    auto maxDomain = getAccessor<float, 1>(maxDomain_, "maxDomain", false, verbose);
    auto minDomain = getAccessor<float, 1>(minDomain_, "minDomain", false, verbose);
    auto periodicity = getAccessor<bool, 1>(periodicity_, "periodicity", false, verbose);

    // Get the dimensions of the input tensors
    int nQuery = queryPositions.size(0);
    int dim = queryPositions.size(1);
    int nSorted = sortedPositions.size(0);

    // Output input state to console for debugging, enable via verbose flag
    if (verbose) {
        std::cout << "Search Parameters:" << std::endl;
        std::cout << "\tnQuery: " << nQuery << std::endl;
        std::cout << "\tdim: " << dim << std::endl;
        std::cout << "\tnSorted: " << nSorted << std::endl;
        std::cout << "\tsearchRange: " << searchRange << std::endl;
        std::cout << "\thashMapLength: " << hashMapLength << std::endl;
        std::cout << "\thCell: " << hCell << std::endl;
        std::cout << "\tMode: " << mode << std::endl;

        std::cout << "\nInput Tensors:"   << std::endl;
        std::cout << "\tqueryPositions: " << queryPositions.size(0) << "x" << queryPositions.size(1) << std::endl;
        std::cout << "\tquerySupport: " << querySupport.size(0) << std::endl;
        std::cout << "\tsortedPositions: " << sortedPositions.size(0) << "x" << sortedPositions.size(1) << std::endl;
        std::cout << "\tsortedSupport: " << sortedSupport.size(0) << std::endl;

        std::cout << "\nDatastructure Tensors:" << std::endl;
        std::cout << "\thashTable: " << hashTable.size(0) << "x" << hashTable.size(1) << std::endl;
        std::cout << "\tnumCells: " << numCells.size(0) << std::endl;
        std::cout << "\tcellTable: " << cellTable.size(0) << "x" << cellTable.size(1) << std::endl;

        std::cout << "\nDomain Tensors:" << std::endl;
        std::cout << "\tqMin: " << qMin.size(0) << std::endl;
        std::cout << "\tmaxDomain: " << maxDomain.size(0) << std::endl;
        std::cout << "\tminDomain: " << minDomain.size(0) << std::endl;
        std::cout << "\tperiodicity: " << periodicity.size(0) << std::endl;

        std::cout << "\n";
    }

    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
    auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
    for (int32_t d = 0; d < dim; d++){
        int32_t itr = -searchRange;
        int32_t ctr = 0;
        for(int32_t o = 0; o < offsets.size(0); ++o){
            int32_t c = o % power(1 + 2 * searchRange, d);
            if(c == 0 && ctr > 0)
                itr++;
            if(itr > searchRange)
                itr = -searchRange;
            offsets[o][dim - d - 1] = itr;
            ctr++;
        }
    }
    offsets = offsets.to(queryPositions_.device());
    // Output the cell offsets to the console for debugging, enable via verbose flag
    if(verbose){
        std::cout << "Cell Offsets:" << std::endl;
        for (int32_t i = 0; i < offsets.size(0); i++){
            std::cout << "\t[" << i << "]: ";
            for (int32_t d = 0; d < dim; d++){
                std::cout << offsets[i][d].item<int32_t>() << " ";
            }
            std::cout << std::endl;
        }
    }

    // Allocate output tensor for the neighbor counters
    auto neighborCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));

    // Create the accessors for the input tensors as packed accessors
    auto referencePositionAccessor = sortedPositions_.packed_accessor32<float, 2, traits>();
    auto referenceSupportAccessor = sortedSupport_.packed_accessor32<float, 1, traits>();
    auto hashTableAccessor = hashTable_.packed_accessor32<int32_t, 2, traits>();
    auto celTableAccessor = cellTable_.packed_accessor32<int64_t, 2, traits>();
    auto offsetAccessor = offsets.packed_accessor32<int32_t, 2, traits>();
    auto numCellsAccessor = numCells_.packed_accessor32<int32_t, 1, traits>();

    // Loop over all query particles and count the number of neighbors per particle
    at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
        for(int32_t i = start; i < end; ++i){
            auto xi = queryPositions[i];
            auto hi = querySupport_.defined() ? querySupport[i] : 0.f;

            int32_t neighborCounter = countNeighborsForParticle(
                xi, hi, searchRange, 
                referencePositionAccessor, referenceSupportAccessor,
                hashTableAccessor, hashMapLength, 
                celTableAccessor, numCellsAccessor,
                offsetAccessor,
                hCell, minDomain, maxDomain, periodicity, searchMode, defaultOptions);
            neighborCounters[i] = neighborCounter;
        }
    });
    // Return the neighbor counters
    return neighborCounters;
}


/**
 * 
 * @brief C++ based neighborhood search for all particles in the query set relative to the sorted set.
 * 
 * This function searches the neighbors for all particles in the query set relative to the sorted set.
 * 
 * @param queryPositions_ The positions of the query particles.
 * @param querySupport_ The support radii of the query particles.
 * @param searchRange The search range.
 * @param sortedPositions_ The sorted positions of the particles.
 * @param sortedSupport_ The sorted support radii of the particles.
 * @param hashTable_ The hash table.
 * @param hashMapLength The length of the hash map.
 * @param numCells_ The number of cells.
 * @param cellTable_ The cell table.
 * @param qMin_ The minimum domain bounds.
 * @param hCell The cell size.
 * @param maxDomain_ The maximum domain bounds.
 * @param minDomain_ The minimum domain bounds.
 * @param periodicity_ The periodicity flags.
 * @param mode The support mode.
 * @param verbose Flag indicating whether to print verbose information.
 * @return The neighbor list as a pair of tensors
 */
std::pair<torch::Tensor, torch::Tensor> buildNeighborList(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int neighborListLength,
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, float hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false){
    if(verbose)
        std::cout << "C++: countNeighbors" << std::endl;
    // Convert the mode to an enum for easier handling
    supportMode searchMode = supportMode::symmetric;
    if(mode == "symmetric"){
        searchMode = supportMode::symmetric;
    } else if(mode == "gather"){
        searchMode = supportMode::gather;
    } else if(mode == "scatter"){
        searchMode = supportMode::scatter;
    } else {
        throw std::runtime_error("Invalid support mode: " + mode);
    }

    // Check if the input tensors are defined and contiguous and have the correct dimensions
    auto queryPositions = getAccessor<float, 2>(queryPositions_, "queryPositions", false, verbose);
    auto querySupport = getAccessor<float, 1>(querySupport_, "querySupport", false, verbose, supportMode::scatter == searchMode);
    auto sortedPositions = getAccessor<float, 2>(sortedPositions_, "sortedPositions", false, verbose);
    auto sortedSupport = getAccessor<float, 1>(sortedSupport_, "sortedSupport", false, verbose, supportMode::gather == searchMode);

    // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
    auto hashTable = getAccessor<int, 2>(hashTable_, "hashTable", false, verbose);
    auto numCells = getAccessor<int, 1>(numCells_, "numCells", false, verbose);
    auto cellTable = getAccessor<int64_t, 2>(cellTable_, "cellTable", false, verbose);
    auto qMin = getAccessor<float, 1>(qMin_, "qMin", false, verbose);
    auto maxDomain = getAccessor<float, 1>(maxDomain_, "maxDomain", false, verbose);
    auto minDomain = getAccessor<float, 1>(minDomain_, "minDomain", false, verbose);
    auto periodicity = getAccessor<bool, 1>(periodicity_, "periodicity", false, verbose);

    // Check if the neighbor counter tensor is defined and contiguous
    auto neighborCounter = getAccessor<int, 1>(neighborCounter_, "neighborCounter", false, verbose);
    auto neighborOffsets = getAccessor<int, 1>(neighborOffsets_, "neighborOffsets", false, verbose);

    // Get the dimensions of the input tensors
    int nQuery = queryPositions.size(0);
    int dim = queryPositions.size(1);
    int nSorted = sortedPositions.size(0);

    // Output input state to console for debugging, enable via verbose flag
    if (verbose) {
        std::cout << "Search Parameters:" << std::endl;
        std::cout << "\tnQuery: " << nQuery << std::endl;
        std::cout << "\tdim: " << dim << std::endl;
        std::cout << "\tnSorted: " << nSorted << std::endl;
        std::cout << "\tsearchRange: " << searchRange << std::endl;
        std::cout << "\thashMapLength: " << hashMapLength << std::endl;
        std::cout << "\thCell: " << hCell << std::endl;
        std::cout << "\tMode: " << mode << std::endl;
        std::cout << "\tneighborListLength: " << neighborListLength << std::endl;

        std::cout << "\nInput Tensors:"   << std::endl;
        std::cout << "\tqueryPositions: " << queryPositions.size(0) << "x" << queryPositions.size(1) << std::endl;
        std::cout << "\tquerySupport: " << querySupport.size(0) << std::endl;
        std::cout << "\tsortedPositions: " << sortedPositions.size(0) << "x" << sortedPositions.size(1) << std::endl;
        std::cout << "\tsortedSupport: " << sortedSupport.size(0) << std::endl;

        std::cout << "\nDatastructure Tensors:" << std::endl;
        std::cout << "\thashTable: " << hashTable.size(0) << "x" << hashTable.size(1) << std::endl;
        std::cout << "\tnumCells: " << numCells.size(0) << std::endl;
        std::cout << "\tcellTable: " << cellTable.size(0) << "x" << cellTable.size(1) << std::endl;

        std::cout << "\nDomain Tensors:" << std::endl;
        std::cout << "\tqMin: " << qMin.size(0) << std::endl;
        std::cout << "\tmaxDomain: " << maxDomain.size(0) << std::endl;
        std::cout << "\tminDomain: " << minDomain.size(0) << std::endl;
        std::cout << "\tperiodicity: " << periodicity.size(0) << std::endl;

        std::cout << "\nOffsets Tensors:" << std::endl;
        std::cout << "\tneighborCounter: " << neighborCounter.size(0) << std::endl;
        std::cout << "\tneighborOffsets: " << neighborOffsets.size(0) << std::endl;
        
        std::cout << "\n";
    }

    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
    auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
    for (int32_t d = 0; d < dim; d++){
        int32_t itr = -searchRange;
        int32_t ctr = 0;
        for(int32_t o = 0; o < offsets.size(0); ++o){
            int32_t c = o % power(1 + 2 * searchRange, d);
            if(c == 0 && ctr > 0)
                itr++;
            if(itr > searchRange)
                itr = -searchRange;
            offsets[o][dim - d - 1] = itr;
            ctr++;
        }
    }
    offsets = offsets.to(queryPositions_.device());
    // Output the cell offsets to the console for debugging, enable via verbose flag
    if(verbose){
        std::cout << "Cell Offsets:" << std::endl;
        for (int32_t i = 0; i < offsets.size(0); i++){
            std::cout << "\t[" << i << "]: ";
            for (int32_t d = 0; d < dim; d++){
                std::cout << offsets[i][d].item<int32_t>() << " ";
            }
            std::cout << std::endl;
        }
    }

    // Allocate output tensor for the neighbor counters
    auto neighborList_i = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt32));
    auto neighborList_j = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt32));

    // Create the accessors for the input tensors as packed accessors
    auto referencePositionAccessor = sortedPositions_.packed_accessor32<float, 2, traits>();
    auto referenceSupportAccessor = sortedSupport_.packed_accessor32<float, 1, traits>();
    auto hashTableAccessor = hashTable_.packed_accessor32<int32_t, 2, traits>();
    auto cellTableAccessor = cellTable_.packed_accessor32<int64_t, 2, traits>();
    auto offsetAccessor = offsets.packed_accessor32<int32_t, 2, traits>();
    auto numCellsAccessor = numCells_.packed_accessor32<int32_t, 1, traits>();
    auto neighborCounterAccessor = neighborCounter_.packed_accessor32<int32_t, 1, traits>();
    auto neighborOffsetsAccessor = neighborOffsets_.packed_accessor32<int32_t, 1, traits>();

    // Loop over all query particles and count the number of neighbors per particle
    // auto dim = queryPositions.size(1);

    at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
        for(int32_t i = start; i < end; ++i){
            auto xi = queryPositions[i];
            auto hi = querySupport_.defined() ? querySupport[i] : 0.f;

            int32_t offset = neighborOffsetsAccessor[i];
            int32_t currentOffset = offset;

            auto queryCell = torch::zeros({dim}, defaultOptions.dtype(torch::kInt32));
            for(int d = 0; d < dim; d++){
                queryCell[d] = std::floor((xi[d] - minDomain[d]) / hCell);
            }    
            int32_t neighborCounter = 0;
            iterateOffsetCells(queryCell.packed_accessor32<int32_t,1>(), offsetAccessor, 
                hashTableAccessor, hashMapLength, 
                cellTableAccessor, numCellsAccessor, periodicity,
                [&](int32_t cBegin, int32_t cEnd){
                    for(int32_t j = cBegin; j < cEnd; j++){
                        auto xj = sortedPositions[j];
                        auto dist = modDistance(xi, xj, minDomain, maxDomain, periodicity);
                        if( (searchMode == supportMode::scatter && dist < sortedSupport[j])||
                            (searchMode == supportMode::gather && dist < hi) ||
                            (searchMode == supportMode::symmetric && dist < (hi + sortedSupport[j]) / 2.f)){
                            neighborList_i[currentOffset] = i;
                            neighborList_j[currentOffset] = j;
                            currentOffset++;
                        }
                    }
                }, defaultOptions);
        }
    });
    // Return the neighbor counters
    return std::make_pair(neighborList_i, neighborList_j);
}

// Create the python bindings for the C++ functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("countNeighbors", &countNeighbors, "Count the Number of Neighbors (C++) using a precomputed hash table and cell map");
  m.def("buildNeighborList", &buildNeighborList, "Build the Neighborlist (C++) using a precomputed hash table and cell map as well as neighbor counts");
}