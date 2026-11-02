#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

// Run fn() `warmup` times untimed, then return the median of `iters` timed runs, in ms.
template <typename F>
float bench_ms(F fn, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; i++) fn();
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    std::vector<float> times(iters);
    for (int i = 0; i < iters; i++) {
        hipEventRecord(start);
        fn();
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&times[i], start, stop);
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
    std::nth_element(times.begin(), times.begin() + iters / 2, times.end());
    return times[iters / 2];
}

// Print time plus GB/s, and TFLOP/s if flops is given.
inline void report(const char* name, float ms, double bytes, double flops = 0) {
    printf("%-24s %8.3f ms  %8.1f GB/s", name, ms, bytes / (ms * 1e6));
    if (flops > 0) printf("  %8.2f TFLOP/s", flops / (ms * 1e9));
    printf("\n");
}