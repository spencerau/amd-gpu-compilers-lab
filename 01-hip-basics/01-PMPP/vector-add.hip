// Vector Addition Kernel Code ported to HIP 

#include <iostream>
// #include <cstdlib>
// #include <cstdio>
// #include <hip/hip_runtime.h>
#include "../../common/hip_check.hpp"


// kernel function
// each thread performs one pair-wise addition
__global__ 
void vecAddKernel(float* A, float* B, float* C, int n) {
    // unique identifier for the specific thread & block combo
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device (GPU) memory for A, B, and C
    // hipMalloc writes new device address into the ptr provided and 
    // returns a status code 
    HIP_CHECK(hipMalloc(&A_d, size));
    HIP_CHECK(hipMalloc(&B_d, size));
    HIP_CHECK(hipMalloc(&C_d, size));

    // copy A and B to device memory
    // hipMemcpy(dst, src, sizeBytes, kind) - 
    // copy into dst, from src, this many bytes, in what direction (device <-> host)
    HIP_CHECK(hipMemcpy(A_d, A_h, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h, size, hipMemcpyHostToDevice));

    // Part 2: Call kernel to launch a grid of threads to
    // perform the actual vector addition
    // vecAddKernel(A_d, B_d, C_d, n);
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    HIP_CHECK(hipGetLastError());
    
    // Part 3: Copy C from the device memory
    HIP_CHECK(hipMemcpy(C_h, C_d, size, hipMemcpyDeviceToHost));
    
    // Free device vectors
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(C_d));

}

int main() {
    // memory allocation for arrays A, B, and C
    // I/O to read A and B, N elements each
    int N;
    std::cout << "Enter cols (N) ";
    std::cin >> N;

    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    // fill in values for A and B
    for (int i = 0; i < N; ++i) {
        A[i] = 2 * (i + 1);  // 2, 4, 6, ...
        B[i] = 2 * i + 1;    // 1, 3, 5, ...
    }

    vecAdd(A, B, C, N);

    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    free(A);
    free(B);
    free(C);
}