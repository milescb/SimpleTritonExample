#pragma once

#include <vector>
#include <stdexcept>

class StandaloneProcessor {
public:

    StandaloneProcessor(int device_id) : device_id_(device_id) {}

    void initialize() {
        return;
    }

    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& a,
                                        const std::vector<std::vector<float>>& b) {

        size_t rows = a.size();
        size_t n    = a[0].size();

        std::vector<std::vector<float>> result(rows, std::vector<float>(n));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
    
        return result;
    }

private:
    int device_id_;
};