#include <iostream>

template<typename T, std::size_t... sizes>
struct Tensor {
    static constexpr std::size_t num_elements = (... * sizes);

    Tensor() : data(new T[num_elements]) {}

    ~Tensor() {
        delete[] data;
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) == sizeof...(sizes), "Incorrect number of indices");
        return data[calculateIndex(indices...)];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(sizes), "Incorrect number of indices");
        return data[calculateIndex(indices...)];
    }

    T* begin() {
        return data;
    }

    T* end() {
        return data + num_elements;
    }

    const T* begin() const {
        return data;
    }

    const T* end() const {
        return data + num_elements;
    }

    std::size_t numElements() const {
        return num_elements;
    }

    void print() const {
        for (std::size_t i = 0; i < num_elements; ++i) {
            std::cout << data[i] << ' ';
        }
        std::cout << '\n';
    }



private:
    // ...

    template<typename... Indices>
    std::size_t calculateIndex(std::size_t first, Indices... rest) {
        return first * calculateStride(0) + calculateIndex(rest...);
    }

    std::size_t calculateIndex() {
        return 0;
    }

    template<std::size_t N>
    std::size_t calculateStride() {
        return (... * (N < sizeof...(sizes) ? sizes : 1));
    }

    T* data;

};

template<typename T, std::size_t... sizes>
auto makeIdentityTensor() {
    Tensor<T, sizes...> tensor;
    
}


int main(){
    std::cout << "Hello World!" << std::endl;

    // Tensor<float> scalarTensor;
    Tensor<float, 3> vectorTensor;
    Tensor<float, 3, 3> matrixTensor;
    Tensor<float, 3, 3, 3> tensorTensor;

    return 0;
}