//Biblography with similiar solutions:
//https://www.geeksforgeeks.org/c-program-multiply-two-matrices,
//https://github.com/mohamedhassan279/Matrix-Multiplication,
//https://www.youtube.com/watch?v=2IgZuVGwEb0,
//https://www.geeksforgeeks.org/cpp-program-for-determinant-of-a-matrix

#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <ctime>
#include <chrono>

/*
 Functions which are caaling GPU kernels are implemented in kernel.cu File and declared in Matrix class as methods
*/

//structure used in split and merge functions
struct Struc_matrixes {
    std::vector<std::vector<float>> m1;
    std::vector<std::vector<float>> m2;
    std::vector<std::vector<float>> m3;
    std::vector<std::vector<float>> m4;
};

class Call; //declaration of class test to use it as a friend of Matrix

class Matrix {
private:
    int rows;
    int cols;
    std::vector<std::vector<float>> matrix;

    //auxiliary functions for multiplying
    Struc_matrixes split(int rows);
    static Matrix merge(Struc_matrixes ms);

    //auxiliary functions for caluclating determinant
    static int find_max(const std::vector<std::vector<float>>& matrix, int column, int start);
    double multiply_diagonal(const std::vector<std::vector<float>>& matrix);

    //auxiliary functions for GPU kernels
    
    static Matrix unflat(std::vector<float> matrix, int n);
    static std::vector<float> flat_I(int size);
    static std::vector<float> flat_matrix(Matrix& matrix);
public:


    //Constructors
    Matrix(int rows_, int cols_);
    Matrix(std::vector<std::vector<float>>& m, int rows_, int cols_);
    Matrix(std::vector<std::vector<float>>& m);

    //getters
    std::vector<std::vector<float>>& get_matrix();
    int get_rows() const;
    int get_cols() const;

    void print() const;

    //operators
    Matrix operator +(Matrix m2);
    Matrix operator -(Matrix m2);
    operator std::vector<std::vector<float>>();

    //Functions to call threads
    Matrix CPU_threads_func(Matrix& m2, std::string operation, void(*func)(const Matrix&, const Matrix&, std::vector<std::vector<float>>&, int)) const;
    static Matrix GPU_threads_func(Matrix& m1, Matrix& m2, void (*func)(const float*, const float*, float*, int, int, int));
    static Matrix run_cublas(std::vector<float> h_A, std::vector<float> h_B, std::vector<float> h_C, std::string operation, int N, void (*func)(const float*, const float*, float*, int));
    static std::vector<float> threads_GPU_no_flat(std::vector<float>& m1, std::vector<float>& m2, void (*func)(const float*, const float*, float*, int, int, int));

    //MULTIPLYING
    Matrix multiply_matrix_classic_no_threads_no_ref(Matrix m2, bool quiet) const;
    Matrix multiply_matrix_classic_no_threads(Matrix& m2) const;
    static Matrix Strassen_alghoritm(Matrix m1, Matrix m2, int rows);
    //cpu threads
    static void __cdecl multiply_thread_row(const Matrix& m1, const Matrix& m2, std::vector<std::vector<float>>& result, int i);
    //gpu
    static void multiply_cublas(const float* h_A, const float* h_B, float* h_C, int n);


    //ADD
    Matrix add_classic_no_ref(Matrix m2);
    Matrix add_classic_ref(Matrix& m2, bool quiet);
    //cpu threads
    static void Matrix::add_thread_row(const Matrix& m1, const Matrix& m2, std::vector<std::vector<float>>& result, int i);
    //gpu
    static void Matrix::add_cublas(const float* h_A, const float* h_B, float* h_C, int n);


    //DETERMINANT
    double determinant_gauss();
    //cpu threads
    static void Matrix::determinant_thread(Matrix& m, int row, int pivot_row);
    double Matrix::determinant_gauss_call_thread_func(void(*func)(Matrix& m, int row, int pivot_row));
    //gpu
    double GPU_determinant();
    static double determinant_cublas(std::vector<float>& h_A, int N);

    double GPU_determinant_shared();
    static double Matrix::GPU_determinant_shared_no_flat(std::vector<float> flat, int rows);

    friend Call;
};
