#include "HipExample.h"
#include <iostream>

int main() {
    GpuProcessor proc(0);  // device 0
    proc.initialize();

    std::vector<std::vector<float>> a = {{1.f, 2.f, 3.f, 4.f},
                                         {1.f, 3.f, 3.f, 7.f},
                                         {4.f, 3.f, 8.f, 7.f}};
    std::vector<std::vector<float>> b = {{1.f, 2.f, 3.f, 4.f},
                                         {1.f, 3.f, 3.f, 7.f},
                                         {4.f, 19.f, 8.f, 23.f}};
    auto result = proc.run(a, b);

    for (const auto& row : result) {
        for (float v : row) std::cout << v << " ";
        std::cout << "\n";
    }
}