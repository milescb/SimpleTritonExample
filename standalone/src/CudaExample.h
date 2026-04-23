#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess)                                       \
            throw std::runtime_error(cudaGetErrorString(err));        \
    } while (0)

__global__ void kernel_add(const float* a, const float* b, float* out, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

class GpuProcessor {
public:
    GpuProcessor(int device_id) : device_id_(device_id) {}

    void initialize() {
        CUDA_CHECK(cudaSetDevice(device_id_));
    }

    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& a,
                                        const std::vector<std::vector<float>>& b) {

        size_t rows = a.size();
        size_t n    = a[0].size();

        size_t bytes = n * sizeof(float);

        float *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a,   bytes));
        CUDA_CHECK(cudaMalloc(&d_b,   bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));

        int block = 256;
        int grid  = (n + block - 1) / block;

        std::vector<std::vector<float>> result(rows, std::vector<float>(n));
        for (size_t i = 0; i < rows; ++i) {
            CUDA_CHECK(cudaMemcpy(d_a, a[i].data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, b[i].data(), bytes, cudaMemcpyHostToDevice));
            kernel_add<<<grid, block>>>(d_a, d_b, d_out, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(result[i].data(), d_out, bytes, cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_out));
        return result;
    }

private:
    int device_id_;
};
