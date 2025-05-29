//GPU code based on:
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/,
//https://github.com/NVIDIA/CUDALibrarySamples,
//https://github.com/NVIDIA/cuda-samples,
//https://docs.nvidia.com/cuda/cublas/index.html,
//https://docs.nvidia.com/cuda/cusolver/,
//https://uts.nipissingu.ca/haibinz/1557/l1557-7.pdf - Zhang. H,
//https://people.cs.rutgers.edu/~venugopa/parallel_summer2012 - Rutgers University. (n.d.). Matrix Computation Project.
//https://cpp0x.pl/dokumentacja/CUDA/cudaMalloc/1187
// and book "CUDA by Example: An Introduction to General_Purpose GPU Programming" J.Sanders, E.Kandrot
#include "kernel.h"

//CUDA_CHECK, CUBLAS_CHECK and CUSOLVER_CHECK are standard macros to check errors in code run on GPU
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }
#define CUBLAS_CHECK(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CUSOLVER_CHECK(call) \
    { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

//-------------------------AUXILIARY FUNCTIONS-------------------------//

//make 1D vector from matrix
std::vector<float> Matrix::flat_matrix(Matrix& matrix) {
    std::vector<float> flat;
    for (std::vector<std::vector<float>>::iterator it = matrix.get_matrix().begin(); it != matrix.get_matrix().end(); it++) {
        flat.insert(flat.end(), it->begin(), it->end());
    }
    return flat;
}

//return from 1D vector to Matrix
Matrix Matrix::unflat(std::vector<float> matrix, int n) {
    Matrix mat(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat.get_matrix()[i][j] = matrix[i * n + j];
        }
    }
    return mat;
}

//create flat version of identity matrix
std::vector<float> Matrix::flat_I(int size) {
    std::vector<float> flat(size * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                flat[i * size + j] = 1;
            }
            else {
                flat[i * size + j] = 0;
            }
        }
    }
    return flat;
}

//function used to calculate ceil of division of cols/rows and threads_per_block
int ceil(int rc, int tpb) {
    int rest = rc % tpb;
    int div = rc / tpb;
    if (rest > 0) {
        div++;
    }
    return div;
}

//-------------------------Functions to call threads/cublas functions-------------------------//

Matrix Matrix::GPU_threads_func(Matrix& m1, Matrix& m2, void (*func)(const float*, const float*, float*, int, int, int)) {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    if (m1.get_cols() != m2.get_rows()) {
        std::cout << "can't multiply this matrix" << std::endl;
        exit(0);
    }

    int size_m1 = m1.get_rows() * m1.get_cols() * sizeof(float);
    int size_m2 = m2.get_rows() * m2.get_cols() * sizeof(float);
    int size_res = m1.get_rows() * m2.get_cols();

    std::vector<float> flat_m1 = flat_matrix(m1);
    std::vector<float> flat_m2 = flat_matrix(m2);
    std::vector<float> flat_result(m1.get_rows() * m2.get_cols(), 0);

    size_res *= sizeof(float);

    float* d_m1;
    float* d_m2;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_m1, size_m1));
    CUDA_CHECK(cudaMalloc(&d_m2, size_m2));
    CUDA_CHECK(cudaMalloc(&d_result, size_res));
    CUDA_CHECK(cudaMemcpy(d_m1, flat_m1.data(), size_m1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m2, flat_m2.data(), size_m2, cudaMemcpyHostToDevice));
    dim3 threads_per_block(16, 16); //I use 16x16 becouse it is standard size of threads per block
    dim3 blocks_per_grid(ceil(m2.get_cols(), threads_per_block.x), ceil(m1.get_rows(), threads_per_block.y));
    func << < blocks_per_grid, threads_per_block >> > (d_m1, d_m2, d_result, m1.get_rows(), m2.get_cols(), m1.get_cols());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaMemcpy(flat_result.data(), d_result, m1.get_rows() * m2.get_cols() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    std::vector<std::vector<float>> result(m1.get_rows(), std::vector<float>(m2.get_cols(), 0));
    for (int i = 0; i < m1.get_rows(); i++) {
        for (int j = 0; j < m2.get_cols(); j++) {
            result[i][j] = flat_result[i * m2.get_cols() + j];
        }
    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "multiply_matrix_classic_threads_GPU time (microseconds): " << elapsed.count() << std::endl;
    return Matrix(result, m1.get_rows(), m2.get_cols());
}

Matrix Matrix::run_cublas(std::vector<float> h_A, std::vector<float> h_B, std::vector<float> h_C, std::string operation, int N, void (*func)(const float*, const float*, float*, int)) {

    //First call
    auto start1 = std::chrono::high_resolution_clock::now();
    func(h_A.data(), h_B.data(), h_C.data(), N);
    auto stop1 = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::cout << "First cuBLAS " << operation << " time(microseconds) : " << elapsed1.count() << std::endl;

    //Second call
    auto start2 = std::chrono::high_resolution_clock::now();
    func(h_A.data(), h_B.data(), h_C.data(), N);
    auto stop2 = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout << "Second cuBLAS " << operation << " time(microseconds) : " << elapsed2.count() << std::endl;

    return unflat(h_C, N);
}

//-------------------------MULTIPLYING-------------------------//

__global__ void thread_row_classic_GPU_no_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (i < rows1 && j < cols2) {
        sum = 0;
        for (int k = 0; k < cols1; k++) {
            sum += m1[i * cols1 + k] * m2[k * cols2 + j];
        }
        result[i * cols2 + j] = sum;
    }
}

__global__ void thread_row_classic_GPU_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1) {
    __shared__ float shared_m1[16][16];
    __shared__ float shared_m2[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    int rest = cols1 % blockDim.x;
    int div = cols1 / blockDim.x;
    if (rest > 0) {
        div++;
    }

    for (int k = 0; k < div; k++) {
        if (row < rows1 && (k * blockDim.x + threadIdx.x) < cols1) {
            shared_m1[threadIdx.y][threadIdx.x] = m1[row * cols1 + k * blockDim.x + threadIdx.x];
        }
        else {
            shared_m1[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < cols2 && (k * blockDim.y + threadIdx.y) < cols1) {
            shared_m2[threadIdx.y][threadIdx.x] = m2[(k * blockDim.y + threadIdx.y) * cols2 + col];
        }
        else {
            shared_m2[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_m1[threadIdx.y][i] * shared_m2[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows1 && col < cols2) {
        result[row * cols2 + col] = sum;
    }
}

std::vector<float> Matrix::threads_GPU_no_flat(std::vector<float>& m1, std::vector<float>& m2, void (*func)(const float*, const float*, float*, int, int, int)) {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();

    int n = static_cast<int>(sqrt(m1.size()));
    int rows1 = n;
    int cols2 = n;
    int cols1 = n;


    int size_m1 = m1.size() * sizeof(float);
    int size_m2 = m2.size() * sizeof(float);

    std::vector<float> flat_result(m1.size(), 0);

    int size_res = m1.size() * sizeof(float);

    float* d_m1;
    float* d_m2;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_m1, size_m1));
    CUDA_CHECK(cudaMalloc(&d_m2, size_m2));
    CUDA_CHECK(cudaMalloc(&d_result, size_res));
    CUDA_CHECK(cudaMemcpy(d_m1, m1.data(), size_m1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m2, m2.data(), size_m2, cudaMemcpyHostToDevice));
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        ceil(cols2, threads_per_block.x),
        ceil(rows1, threads_per_block.y)
    );
    std::chrono::steady_clock::time_point start_mo =
        std::chrono::high_resolution_clock::now();

    func << <blocks_per_grid, threads_per_block >> > (
        d_m1, d_m2, d_result,
        rows1, cols2, cols1
        );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point stop_mo = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed_mo = std::chrono::duration_cast<std::chrono::microseconds>(stop_mo - start_mo);
    std::cout << "multiply only time (microseconds): " << elapsed_mo.count() << std::endl;
    cudaMemcpy(flat_result.data(), d_result, m1.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "multiply_matrix_classic_threads_GPU time (microseconds): " << elapsed.count() << std::endl;
    return flat_result;
}


void Matrix::multiply_cublas(const float* h_A, const float* h_B, float* h_C, int n) {
    size_t size = n * n * sizeof(float);

    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0, beta = 0.0;

    std::chrono::steady_clock::time_point start_mo = std::chrono::high_resolution_clock::now();
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_A, n,
        d_B, n,
        &beta,
        d_C, n
    ));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point stop_mo = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed_mo = std::chrono::duration_cast<std::chrono::microseconds>(stop_mo - start_mo);
    std::cout << "multiply only time (microseconds): " << elapsed_mo.count() << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(handle));
}

//-------------------------ADD-------------------------//

__global__ void thread_add_classic_GPU_no_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows1 && j < cols2) {
        result[i * cols2 + j] = m1[i * cols2 + j] + m2[i * cols2 + j];
    }
}

__global__ void thread_add_classic_GPU_shared_memory(const float* m1, const float* m2, float* result, int rows1, int cols2, int cols1) {
    __shared__ float shared_m1[16][16];
    __shared__ float shared_m2[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    int rest = cols1 % blockDim.x;
    int div = cols1 / blockDim.x;
    if (rest > 0) {
        div++;
    }

    for (int k = 0; k < div; k++) {
        if (row < rows1 && (k * blockDim.x + threadIdx.x) < cols1) {
            shared_m1[threadIdx.y][threadIdx.x] = m1[row * cols1 + k * blockDim.x + threadIdx.x];
        }
        else {
            shared_m1[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < cols2 && (k * blockDim.y + threadIdx.y) < cols1) {
            shared_m2[threadIdx.y][threadIdx.x] = m2[(k * blockDim.y + threadIdx.y) * cols2 + col];
        }
        else {
            shared_m2[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();
    }

    if (row < rows1 && col < cols2) {
        result[row * cols2 + col] = m1[row * cols2 + col] + m2[row * cols2 + col];
    }
}

void Matrix::add_cublas(const float* h_A, const float* h_B, float* h_C, int n) {
    size_t size = n * n * sizeof(float);

    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0, beta = 1.0;

    std::chrono::steady_clock::time_point start_mo = std::chrono::high_resolution_clock::now();
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_A, n,
        d_B, n,
        &beta,
        d_C, n
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point stop_mo = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed_mo = std::chrono::duration_cast<std::chrono::microseconds>(stop_mo - start_mo);
    std::cout << "multiply only time (microseconds): " << elapsed_mo.count() << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(handle));
}

//-------------------------DETERMINANT-------------------------//

__global__ void kernel_swap_rows(float* m, int n, int row1, int row2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float tmp = m[row1 * n + j];
        m[row1 * n + j] = m[row2 * n + j];
        m[row2 * n + j] = tmp;
    }
}

__global__ void kernel_eliminate_row(float* m, int n, int pivot_row, int target_row) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_row < n && j < n && j >= pivot_row) {
        float pivot = m[pivot_row * n + pivot_row];
        float factor = m[target_row * n + pivot_row] / pivot;
        m[target_row * n + j] -= factor * m[pivot_row * n + j];
    }
}

double Matrix::GPU_determinant() {
    int n = get_rows();
    if (n != get_cols()) {
        std::cerr << "Determinant only for square matrices\n";
        return 0.0;
    }

    std::vector<float> flat = flat_matrix(*this);
    size_t bytes = n * n * sizeof(float);
    float* d_m = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMemcpy(d_m, flat.data(), bytes, cudaMemcpyHostToDevice));

    int swap_sign = 1;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    for (int i = 0; i < n; ++i) {
        std::vector<float> col(n);
        CUDA_CHECK(cudaMemcpy2D(
            col.data(), sizeof(float),
            d_m + i, n * sizeof(float),
            sizeof(float), n,
            cudaMemcpyDeviceToHost
        ));
        int pivot_row = i;
        float maxv = fabs(col[i]);
        for (int r = i + 1; r < n; ++r) {
            if (fabs(col[r]) > maxv) {
                maxv = fabs(col[r]);
                pivot_row = r;
            }
        }
        if (maxv < 1e-12f) {
            cudaFree(d_m);
            return 0.0;
        }
        if (pivot_row != i) {
            swap_sign *= -1;
            kernel_swap_rows << <grid, block >> > (d_m, n, i, pivot_row);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int target = i + 1; target < n; ++target) {
            kernel_eliminate_row << <grid, block >> > (d_m, n, i, target);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(flat.data(), d_m, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_m);

    double det = swap_sign;
    for (int i = 0; i < n; ++i) {
        det *= flat[i * n + i];
    }
    return det;
}

//shared versions of gpu determinant
__global__ void eliminate_rows_shared(float* m, int n, int pivot_row) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int target_row = blockIdx.y + pivot_row + 1;

    extern __shared__ float pivot_row_data[];

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        pivot_row_data[i] = m[pivot_row * n + i];
    }

    __syncthreads();

    if (target_row < n && col >= pivot_row && col < n) {
        float pivot_val = pivot_row_data[pivot_row];
        if (fabsf(pivot_val) < 1e-12f) return;
        float factor = m[target_row * n + pivot_row] / pivot_val;
        m[target_row * n + col] -= factor * pivot_row_data[col];
    }
}

double Matrix::GPU_determinant_shared() {
    int n = get_rows();
    if (n != get_cols()) {
        std::cerr << "Determinant only for square matrices\n";
        return 0.0;
    }

    std::vector<float> flat = flat_matrix(*this);
    size_t bytes = n * n * sizeof(float);
    float* d_m = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMemcpy(d_m, flat.data(), bytes, cudaMemcpyHostToDevice));

    int swap_sign = 1;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    for (int i = 0; i < n; ++i) {
        std::vector<float> column(n);
        CUDA_CHECK(cudaMemcpy2D(
            column.data(), sizeof(float),
            d_m + i, n * sizeof(float),
            sizeof(float), n,
            cudaMemcpyDeviceToHost
        ));
        int pivot_row = i;
        float max_val = fabsf(column[0]);
        for (int r = 1; r < n - i; ++r) {
            if (fabsf(column[r]) > max_val) {
                max_val = fabsf(column[r]);
                pivot_row = i + r;
            }
        }

        if (max_val < 1e-12f) {
            cudaFree(d_m);
            return 0.0;
        }

        if (pivot_row != i) {
            swap_sign *= -1;
            kernel_swap_rows << <grid, block >> > (d_m, n, i, pivot_row);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int rows_to_eliminate = n - i - 1;
        if (rows_to_eliminate > 0) {
            dim3 grid2((n + block.x - 1) / block.x, rows_to_eliminate);
            eliminate_rows_shared << <grid2, block, n * sizeof(float) >> > (d_m, n, i);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(flat.data(), d_m, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_m);

    double det = swap_sign;
    for (int i = 0; i < n; ++i) {
        float val = flat[i * n + i];
        if (std::isnan(val)) {
            std::cerr << "NaN on diagonal at row " << i << std::endl;
            return std::numeric_limits<double>::quiet_NaN();
        }
        det *= val;
    }

    return det;
}


double Matrix::GPU_determinant_shared_no_flat(std::vector<float> flat, int n) {
    size_t bytes = n * n * sizeof(float);
    float* d_m = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMemcpy(d_m, flat.data(), bytes, cudaMemcpyHostToDevice));

    int swap_sign = 1;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    for (int i = 0; i < n; ++i) {
        std::vector<float> column(n);
        CUDA_CHECK(cudaMemcpy2D(
            column.data(), sizeof(float),
            d_m + i, n * sizeof(float),
            sizeof(float), n,
            cudaMemcpyDeviceToHost
        ));

        int pivot_row = i;
        float max_val = fabsf(column[0]);
        for (int r = 1; r < n - i; ++r) {
            if (fabsf(column[r]) > max_val) {
                max_val = fabsf(column[r]);
                pivot_row = i + r;
            }
        }

        if (max_val < 1e-12f) {
            cudaFree(d_m);
            return 0.0;
        }

        if (pivot_row != i) {
            swap_sign *= -1;
            kernel_swap_rows << <grid, block >> > (d_m, n, i, pivot_row);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int rows_to_eliminate = n - i - 1;
        if (rows_to_eliminate > 0) {
            dim3 grid2((n + block.x - 1) / block.x, rows_to_eliminate);
            eliminate_rows_shared << <grid2, block, n * sizeof(float) >> > (d_m, n, i);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(flat.data(), d_m, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_m);

    double det = swap_sign;
    for (int i = 0; i < n; ++i) {
        float val = flat[i * n + i];
        if (std::isnan(val)) {
            std::cerr << "NaN on diagonal at row " << i << std::endl;
            return std::numeric_limits<double>::quiet_NaN();
        }
        det *= val;
    }

    return det;
}

//cublas
double Matrix::determinant_cublas(std::vector<float>& h_A, int N) {
    size_t bytes = N * N * sizeof(float);
    float* d_A = nullptr;
    int* d_Ipiv = nullptr;
    int* d_info = nullptr;
    void* d_work = nullptr;
    int lwork = 0;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_Ipiv, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));

    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(solver, N, N, d_A, N, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

    CUSOLVER_CHECK(cusolverDnSgetrf(solver, N, N, d_A, N, (float*)d_work, d_Ipiv, d_info));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_LU(N * N);
    std::vector<int>   h_Ipiv(N);
    CUDA_CHECK(cudaMemcpy(h_LU.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ipiv.data(), d_Ipiv, N * sizeof(int), cudaMemcpyDeviceToHost));

    double det = 1.0;
    for (int i = 0; i < N; ++i) {
        det *= static_cast<double>(h_LU[i * N + i]);
        if (h_Ipiv[i] != i + 1) {
            det = -det;
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));

    return det;
}