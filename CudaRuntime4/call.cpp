#include "call.h"

void print_if_small(Matrix result) {
    if (result.get_rows() <= 8 && result.get_cols() <= 8) {
        result.print();
    }
}

void Call::call_multiply(Matrix& m1, Matrix& m2) {
    std::cout << "//-------------------------MULTIPLYING-------------------------//" << std::endl << std::endl;

    std::cout << "NO THREADS METHODS:" << std::endl<<std::endl;

    std::cout << "multiply_matrix_classic_no_threads_no_ref" << std::endl;
    Matrix result = m1.multiply_matrix_classic_no_threads_no_ref(m1, false);
    print_if_small(result);

    
    std::cout << "multiply Strassen_alghoritm" << std::endl;
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    result = Matrix::Strassen_alghoritm(m1, m2, m1.get_rows());
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "multiply_matrix_strassen time (microseconds): " << elapsed.count() << std::endl;
    print_if_small(result);

    result = m1.multiply_matrix_classic_no_threads(m2);
    std::cout << "multiply_matrix_classic_ref" << std::endl;
    print_if_small(result);

    std::cout << std::endl << "THREADS METHODS:" << std::endl<< std::endl;

    std::cout << "CPU:" << std::endl;
    std::cout << "multiply_matrix_classic_threads" << std::endl;
    result = m1.CPU_threads_func(m2, "multiplication CPU threads", &Matrix::multiply_thread_row);
    print_if_small(result);

    std::cout <<std::endl<< "GPU:" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "no shared memory first try: " << std::endl;
    result = Matrix::GPU_threads_func(m1, m2, thread_row_classic_GPU_no_shared_memory);
    std::cout << "no shared memory second try: " << std::endl;
    result = Matrix::GPU_threads_func(m1, m2, thread_row_classic_GPU_no_shared_memory);
    print_if_small(result);

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "shared memory first try: " << std::endl;
    result = Matrix::GPU_threads_func(m1, m2, thread_row_classic_GPU_shared_memory);
    std::cout << "shared memory second try: " << std::endl;
    result = Matrix::GPU_threads_func(m1, m2, thread_row_classic_GPU_shared_memory);
    print_if_small(result);

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::vector<float> flat_m1 = Matrix::flat_matrix(m1);
    std::vector<float> flat_m2 = Matrix::flat_matrix(m2);
    std::cout << "no shared memory flat matrix first try: " << std::endl;
    std::vector<float> result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_row_classic_GPU_no_shared_memory);
    std::cout << "no shared memory flat matrix second try: " << std::endl;
    result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_row_classic_GPU_no_shared_memory);
    result = Matrix::unflat(result_flat, m1.get_rows());
    print_if_small(result);

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "shared memory flat matrix first try: " << std::endl;
    result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_row_classic_GPU_shared_memory);
    std::cout << "shared memory flat matrix second try: " << std::endl;
    result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_row_classic_GPU_shared_memory);
    result = Matrix::unflat(result_flat, m1.get_rows());
    print_if_small(result);
    std::cout << "--------------------------------------------------------------" << std::endl;

    std::cout << std::endl << "CUBLAS:" << std::endl;
    int N, M;
    N = m1.get_rows();
    M = m2.get_cols();
    size_t size = N * M * sizeof(float);

    std::vector<float> h_C(N * M);

    std::vector<float> h_A = Matrix::flat_matrix(m1);
    std::vector<float> h_B = Matrix::flat_matrix(m2);

    result = Matrix::run_cublas(h_A, h_B, h_C, "multiply", N, &Matrix::multiply_cublas);
    print_if_small(result);
    std::cout << std::endl;
}


void Call::call_add(Matrix& m1, Matrix& m2) {
    std::cout << "//-------------------------ADD-------------------------//" << std::endl << std::endl;

    std::cout << "NO THREADS METHODS:" << std::endl << std::endl;

    std::cout << "add_no_ref:" << std::endl;
    Matrix result = m1.add_classic_no_ref(m2);
    print_if_small(result);

    std::cout << "add_ref:" << std::endl;
    result = m1.add_classic_ref(m2, false);
    print_if_small(result);

    std::cout << std::endl << "THREADS METHODS:" << std::endl << std::endl;
    std::cout << "CPU:" << std::endl;
    result = m1.CPU_threads_func(m2, "add CPU threads", &Matrix::add_thread_row);
    print_if_small(result);

    std::cout << std::endl << "GPU:" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "no shared memory flat first try: " << std::endl;
    std::vector<float> flat_m1 = Matrix::flat_matrix(m1);
    std::vector<float> flat_m2 = Matrix::flat_matrix(m2);
    std::vector<float> result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_add_classic_GPU_no_shared_memory);
    std::cout << "no shared memory flat second try: " << std::endl;
    result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_add_classic_GPU_no_shared_memory);
    result = Matrix::unflat(result_flat, m1.get_rows());
    print_if_small(result);

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "shared memory flat first try: " << std::endl;
    result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_add_classic_GPU_shared_memory);
    std::cout << "shared memory flat second try: " << std::endl;
    result_flat = Matrix::threads_GPU_no_flat(flat_m1, flat_m2, thread_add_classic_GPU_shared_memory);
    result = Matrix::unflat(result_flat, m1.get_rows());
    print_if_small(result);
    std::cout << "--------------------------------------------------------------" << std::endl;

    std::cout << std::endl << "CUBLAS:" << std::endl;
    int N, M;
    N = m1.get_rows();
    M = m2.get_cols();
    size_t size = N * M * sizeof(float);

    std::vector<float> h_B = Matrix::flat_I(N);

    std::vector<float> h_A = Matrix::flat_matrix(m1);
    std::vector<float> h_C = Matrix::flat_matrix(m2);

    Matrix test = Matrix::run_cublas(h_A, h_B, h_C, "add", N, &Matrix::multiply_cublas);
    print_if_small(result);
    std::cout << std::endl;
}

//this time printed are only second calls of gpu methods
void Call::call_determinant(Matrix& m1) {
    std::cout << "//-------------------------DETERMINANT-------------------------//" << std::endl << std::endl;
    double det_gauss = m1.determinant_gauss();
    double det_thread_cpu = m1.determinant_gauss_call_thread_func(Matrix::determinant_thread);
    double det_gpu = m1.GPU_determinant();
    double det_gpu_shared = m1.GPU_determinant_shared();

    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    det_gpu = m1.GPU_determinant();
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "det gpu second try time (microseconds): " << elapsed.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    det_gpu_shared = m1.GPU_determinant_shared();
    stop = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "det gpu shared second try time (microseconds): " << elapsed.count() << std::endl;

    std::vector<float> flat_m1 = Matrix::flat_matrix(m1);
    

    double det_gpu_shared_no_flat = Matrix::GPU_determinant_shared_no_flat(flat_m1, m1.get_rows());

    start = std::chrono::high_resolution_clock::now();
    det_gpu_shared_no_flat = Matrix::GPU_determinant_shared_no_flat(flat_m1,m1.get_rows());
    stop = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "det gpu shared no flat determinant second try time(microseconds) : " << elapsed.count() << std::endl;

    int N = m1.get_rows();
    double det_cublas = Matrix::determinant_cublas(flat_m1, N);

    start = std::chrono::high_resolution_clock::now();
    det_cublas = Matrix::determinant_cublas(flat_m1, N);
    stop = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "cuBLAS determinant second try time(microseconds) : " << elapsed.count() << std::endl;

    std::cout << "det gauss: " << det_gauss << std::endl << "det thread cpu: " << det_thread_cpu << std::endl << "det gpu: " << det_gpu << std::endl
        << "det gpu shared: " << det_gpu_shared << std::endl << "det cublas: " << det_cublas << std::endl << "det gpu shared no flat: " << det_gpu_shared_no_flat << std::endl;
}