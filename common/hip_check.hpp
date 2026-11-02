#pragma once

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