#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                     \
    do {                                                                    \
        hipError_t err_ = (expr);                                           \
        if (err_ != hipSuccess) {                                           \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,   \
                    #expr, hipGetErrorString(err_));                        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void write_ids(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i;
}

int main() {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    printf("Device: %s (%s)\n", p.name, p.gcnArchName);

    const int N = 256;
    int h[N], *d;
    HIP_CHECK(hipMalloc(&d, N * sizeof(int)));
    write_ids<<<N / 64, 64>>>(d);           // 4 workgroups x 64 threads
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d));

    for (int i = 0; i < N; i++)
        if (h[i] != i) { printf("FAIL at %d\n", i); return 1; }
    printf("PASS: %d threads wrote their IDs\n", N);
}