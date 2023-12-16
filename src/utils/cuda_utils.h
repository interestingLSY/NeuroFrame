#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, cudaGetErrorString(result)); \
        exit(-1); \
    } \
} while(0)

inline void syncAndCheck(const char* const file, int const line, bool force_check = false) {
#ifdef DEBUG
    force_check = true;
#endif
    if (force_check) {
        cudaDeviceSynchronize();
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ST] CUDA runtime error: ") + cudaGetErrorString(result) + " "
                                    + file + ":" + std::to_string(line) + " \n");
        }
    }
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__, false)
#define sync_check_cuda_error_force() syncAndCheck(__FILE__, __LINE__, true)
