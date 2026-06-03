#pragma once

#include <hip/hip_runtime.h>
#include <vector>
#include <stdexcept>

#define HIP_CHECK(call)                                          \
    do {                                                         \
        hipError_t err = (call);                                 \
        if (err != hipSuccess)                                   \
            throw std::runtime_error(hipGetErrorString(err));    \
    } while (0)

__global__ void kernel_add(const float* a, const float* b, float* out, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

class StandaloneProcessor {
public:
    StandaloneProcessor(int device_id) : device_id_(device_id) {}

    void initialize() {
        HIP_CHECK(hipSetDevice(device_id_));
    }

    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& a,
                                        const std::vector<std::vector<float>>& b) {

        size_t rows = a.size();
        size_t n    = a[0].size();

        size_t bytes = n * sizeof(float);

        float *d_a, *d_b, *d_out;
        HIP_CHECK(hipMalloc(&d_a,   bytes));
        HIP_CHECK(hipMalloc(&d_b,   bytes));
        HIP_CHECK(hipMalloc(&d_out, bytes));

        int block = 256;
        int grid  = (n + block - 1) / block;

        std::vector<std::vector<float>> result(rows, std::vector<float>(n));
        for (size_t i = 0; i < rows; ++i) {
            HIP_CHECK(hipMemcpy(d_a, a[i].data(), bytes, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_b, b[i].data(), bytes, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(kernel_add, dim3(grid), dim3(block), 0, 0, d_a, d_b, d_out, n);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(result[i].data(), d_out, bytes, hipMemcpyDeviceToHost));
        }

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_out));
        return result;
    }

private:
    int device_id_;
};