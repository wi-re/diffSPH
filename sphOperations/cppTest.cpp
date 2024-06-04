#include <iostream>

template<typename T, std::size_t... sizes>
struct general_t {
    using type = torch::TensorAccessor<T, sizes...>;
};


int main(){
    std::cout << "Hello World!" << std::endl;

    return 0;
}