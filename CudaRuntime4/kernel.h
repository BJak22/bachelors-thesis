#pragma once
#pragma once
#include "matrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

//KERNEL FUNCTIONS
//multiply
__global__ void thread_row_classic_GPU_no_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1);
__global__ void thread_row_classic_GPU_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1);
//add
__global__ void thread_add_classic_GPU_no_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1);
__global__ void thread_add_classic_GPU_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1);
//determinant
__global__ void kernel_swap_rows(float* m, int n, int row1, int row2);
__global__ void kernel_eliminate_row(float* m, int n, int pivot_row, int target_row);

__global__ void eliminate_rows_shared(float* m, int n, int pivot_row);
__global__ void eliminate_rows_shared(float* m, int n, int pivot_row);