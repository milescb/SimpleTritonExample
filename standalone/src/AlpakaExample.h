#pragma once

#include <alpaka/alpaka.hpp>

#include <vector>

struct AddKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                   float const* __restrict__ a,
                                   float const* __restrict__ b,
                                   float* __restrict__ out,
                                   size_t n) const {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        if (i < n)
            out[i] = a[i] + b[i];
    }
};

class GpuProcessor {
public:
    using Dim = alpaka::DimInt<1u>;
    using Idx = uint32_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    // Backend selected at configure time via -DALPAKA_BACKEND_HIP / _CUDA / _CPU
    #if defined(ALPAKA_BACKEND_CUDA)
        using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
    #elif defined(ALPAKA_BACKEND_CPU)
        using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
    #else
        // Default: HIP (AMD GPU)
        using Acc = alpaka::AccGpuHipRt<Dim, Idx>;
    #endif

    using Dev    = alpaka::Dev<Acc>;
    using Queue  = alpaka::Queue<Acc, alpaka::Blocking>;
    using BufDev = alpaka::Buf<Dev, float, Dim, Idx>;

    GpuProcessor(int device_id) : device_id_(static_cast<Idx>(device_id)) {}

    void initialize() {}

    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& a,
                                        const std::vector<std::vector<float>>& b) {
        alpaka::Platform<Acc> platform{};
        Dev dev = alpaka::getDevByIdx(platform, device_id_);
        Queue queue(dev);

        size_t rows = a.size();
        Idx    n    = static_cast<Idx>(a[0].size());
        Vec    ext  = Vec::all(n);

        BufDev d_a   = alpaka::allocBuf<float, Idx>(dev, ext);
        BufDev d_b   = alpaka::allocBuf<float, Idx>(dev, ext);
        BufDev d_out = alpaka::allocBuf<float, Idx>(dev, ext);

        Idx const blockSize = 256u;
        Idx const gridSize  = (n + blockSize - 1u) / blockSize;
        auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{
            Vec::all(gridSize), Vec::all(blockSize), Vec::all(1u)};

        alpaka::PlatformCpu cpuPlatform{};
        auto cpuDev = alpaka::getDevByIdx(cpuPlatform, 0u);

        std::vector<std::vector<float>> result(rows, std::vector<float>(n));
        for (size_t i = 0; i < rows; ++i) {
            auto h_a   = alpaka::createView(cpuDev, const_cast<float*>(a[i].data()), ext);
            auto h_b   = alpaka::createView(cpuDev, const_cast<float*>(b[i].data()), ext);
            auto h_out = alpaka::createView(cpuDev, result[i].data(), ext);

            alpaka::memcpy(queue, d_a, h_a);
            alpaka::memcpy(queue, d_b, h_b);

            alpaka::exec<Acc>(queue, workDiv, AddKernel{},
                              alpaka::getPtrNative(d_a),
                              alpaka::getPtrNative(d_b),
                              alpaka::getPtrNative(d_out),
                              static_cast<size_t>(n));

            alpaka::memcpy(queue, h_out, d_out);
        }

        return result;
    }

private:
    Idx device_id_;
};
