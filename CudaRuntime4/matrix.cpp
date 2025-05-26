//Biblography with similiar solutions:
//https://www.geeksforgeeks.org/c-program-multiply-two-matrices,
//https://github.com/mohamedhassan279/Matrix-Multiplication,
//https://www.youtube.com/watch?v=2IgZuVGwEb0,
//https://www.geeksforgeeks.org/cpp-program-for-determinant-of-a-matrix

#include "matrix.h"
#include <algorithm>
#include <stdexcept>
#include<iomanip>

//-------------------------AUXILIARY FUNCTIONS-------------------------//

//auxiliary functions for multiplying

//split matrix to 4 equal blocks
Struc_matrixes Matrix::split(int rows) {
    Struc_matrixes tmp;
    tmp.m1.resize(rows / 2, std::vector<float>(rows / 2));
    tmp.m2.resize(rows / 2, std::vector<float>(rows / 2));
    tmp.m3.resize(rows / 2, std::vector<float>(rows / 2));
    tmp.m4.resize(rows / 2, std::vector<float>(rows / 2));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            if (i < rows / 2 && j < rows / 2) {
                tmp.m1[i][j] = matrix[i][j];
            }
            else if (i >= rows / 2 && j < rows / 2) {
                tmp.m3[i - rows / 2][j] = matrix[i][j];
            }
            else if (i < rows / 2 && j >= rows / 2) {
                tmp.m2[i][j - rows / 2] = matrix[i][j];
            }
            else {
                tmp.m4[i - rows / 2][j - rows / 2] = matrix[i][j];
            }
        }
    }
    return tmp;
}

Matrix Matrix::merge(Struc_matrixes ms) {
    Matrix matrix(ms.m1.size() * 2, ms.m1.size() * 2);
    for (int i = 0; i < ms.m1.size() * 2; i++) {
        for (int j = 0; j < ms.m1.size() * 2; j++) {
            if (i < ms.m1.size() && j < ms.m1.size()) {
                matrix.matrix[i][j] = ms.m1[i][j];
            }
            else if (i >= ms.m1.size() && j < ms.m1.size()) {
                matrix.matrix[i][j] = ms.m3[i - ms.m1.size()][j];
            }
            else if (i < ms.m1.size() && j >= ms.m1.size()) {
                matrix.matrix[i][j] = ms.m2[i][j - ms.m1.size()];
            }
            else {
                matrix.matrix[i][j] = ms.m4[i - ms.m1.size()][j - ms.m1.size()];
            }
        }
    }
    return matrix;
}

//auxiliary functions for caluclating determinant

int Matrix::find_max(const std::vector<std::vector<float>>& matrix, int column, int start) {
    int max_row = start;
    float max_value = std::abs(matrix[start][column]);

    for (int i = start + 1; i < matrix.size(); ++i) {
        if (std::abs(matrix[i][column]) > max_value) {
            max_value = std::abs(matrix[i][column]);
            max_row = i;
        }
    }

    return max_row;
}

double Matrix::multiply_diagonal(const std::vector<std::vector<float>>& matrix) {
    float product = 1.0f;
    for (int i = 0; i < matrix.size(); ++i) {
        product *= matrix[i][i];
    }
    return product;
}

//-------------------------CONSTRUCTORS-------------------------//

Matrix::Matrix(std::vector<std::vector<float>>& m, int rows_, int cols_) {
    matrix = m;
    rows = rows_;
    cols = cols_;
}

Matrix::Matrix(std::vector<std::vector<float>>& m) : matrix(m) {
    rows = m[0].size();
    cols = m[0].size();

}

Matrix::Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
    matrix.resize(rows_, std::vector<float>(cols_));

    std::random_device random_dev;
    std::mt19937 gen(random_dev());
    std::uniform_real_distribution<> dist(0.0, 10.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dist(gen);
        }
    }
}

//-------------------------GETTERS-------------------------//

std::vector<std::vector<float>>& Matrix::get_matrix() {
    return matrix;
}

int Matrix::get_rows() const {
    return rows;
}

int Matrix::get_cols() const {
    return cols;
}



void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(4) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

//-------------------------OPERATORS-------------------------//

Matrix Matrix::operator +(Matrix m2) {
    return (*this).add_classic_ref(m2, true);
}

Matrix Matrix::operator -(Matrix m2) {
    for (int i = 0; i < get_rows(); i++) {
        for (int j = 0; j < get_cols(); j++) {
            matrix[i][j] -= m2.get_matrix()[i][j];
        }
    }
    return matrix;
}

Matrix::operator std::vector<std::vector<float>>() {
    return matrix;
}

std::vector<std::vector<float>> operator +(std::vector<std::vector<float>>& m1, const std::vector<std::vector<float>>& m2) {
    for (int i = 0; i < m1[0].size(); i++) {
        for (int j = 0; j < m1[0].size(); j++) {
            m1[i][j] += m2[i][j];
        }
    }
    return m1;
}

std::vector<std::vector<float>> operator -(std::vector<std::vector<float>>& m1, const std::vector<std::vector<float>>& m2) {
    for (int i = 0; i < m1[0].size(); i++) {
        for (int j = 0; j < m1[0].size(); j++) {
            m1[i][j] -= m2[i][j];
        }
    }
    return m1;
}


//Function to call thrads
Matrix Matrix::CPU_threads_func(Matrix& m2, std::string operation, void(*func)(const Matrix&, const Matrix&, std::vector<std::vector<float>>&, int)) const {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> result(rows, std::vector<float>(m2.cols, 0));
    std::vector<std::thread> threads;
    for (int i = 0; i < rows; i++) {
        threads.emplace_back(func, *this, std::cref(m2), std::ref(result), i);
    }
    for (std::vector<std::thread>::iterator it = threads.begin(); it != threads.end(); it++) {
        if (it->joinable()) {
            it->join();
        }
    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << operation << " time (microseconds): " << elapsed.count() << std::endl;
    return Matrix(result, rows, m2.cols);
}

//-------------------------MULTIPLYING-------------------------//

Matrix Matrix::multiply_matrix_classic_no_threads_no_ref(Matrix m2, bool quiet) const {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    if (cols != m2.rows) {
        std::cout << "can't multiply this matrix" << std::endl;
        exit(0);
    }
    std::vector<std::vector<float>> result(rows, std::vector<float>(m2.cols, 0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < m2.cols; j++) {
            for (int k = 0; k < m2.rows; k++) {
                result[i][j] += matrix[i][k] * m2.matrix[k][j];
            }
        }
    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (!quiet) {
        std::cout << "multiply_matrix_classic_no_threads_no_ref time (microseconds): " << elapsed.count() << std::endl;
    }
    return Matrix(result, rows, m2.cols);
}

//classic approach to multiplyaing matrixes with reference
Matrix Matrix::multiply_matrix_classic_no_threads(Matrix& m2) const {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    if (cols != m2.rows) {
        std::cout << "can't multiply this matrix" << std::endl;
        exit(0);
    }
    std::vector<std::vector<float>> result(rows, std::vector<float>(m2.cols, 0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < m2.cols; j++) {
            for (int k = 0; k < m2.rows; k++) {
                result[i][j] += matrix[i][k] * m2.matrix[k][j];
            }
        }
    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "multiply_matrix_classic_no_threads time (microseconds): " << elapsed.count() << std::endl;
    return Matrix(result, rows, m2.cols);
}

Matrix Matrix::Strassen_alghoritm(Matrix m1, Matrix m2, int rows) {
    if (rows <= 128) {
        return m1.multiply_matrix_classic_no_threads_no_ref(m2, true);
    }
    Struc_matrixes A = m1.split(rows);
    Struc_matrixes B = m2.split(rows);
    Matrix P1 = Strassen_alghoritm(A.m1 + A.m4, B.m1 + B.m4, A.m1[0].size());
    Matrix P2 = Strassen_alghoritm(A.m4, B.m3 - B.m1, A.m1[0].size());
    Matrix P3 = Strassen_alghoritm(A.m1 + A.m2, B.m4, A.m1[0].size());
    Matrix P4 = Strassen_alghoritm(A.m2 - A.m4, B.m3 + B.m4, A.m1[0].size());
    Matrix P5 = Strassen_alghoritm(A.m1, B.m2 - B.m4, A.m1[0].size());
    Matrix P6 = Strassen_alghoritm(A.m3 + A.m4, B.m1, A.m1[0].size());
    Matrix P7 = Strassen_alghoritm(A.m1 - A.m3, B.m1 + B.m2, A.m1[0].size());
    Struc_matrixes R;
    R.m1 = P1 + P2 - P3 + P4;
    R.m2 = P5 + P3;
    R.m3 = P6 + P2;
    R.m4 = P5 + P1 - P6 - P7;
    return merge(R);
}

//CPU  threads

//function for threading which will calculate results for one row
void Matrix::multiply_thread_row(const Matrix& m1, const Matrix& m2, std::vector<std::vector<float>>& result, int i)
{
    for (int j = 0; j < m2.cols; j++) {
        for (int k = 0; k < m2.rows; k++) {
            result[i][j] += m1.matrix[i][k] * m2.matrix[k][j];
        }
    }
}

//-------------------------ADD-------------------------//

Matrix Matrix::add_classic_no_ref(Matrix m2) {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    Matrix result(get_rows(), get_cols());
    for (int i = 0; i < get_rows(); i++) {
        for (int j = 0; j < get_cols(); j++) {
            result.get_matrix()[i][j] = get_matrix()[i][j] + m2.get_matrix()[i][j];
        }
    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "add_classic_no_ref time (microseconds): " << elapsed.count() << std::endl;
    return result;
}

Matrix Matrix::add_classic_ref(Matrix& m2, bool quiet) {
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    Matrix result(get_rows(), get_cols());
    for (int i = 0; i < get_rows(); i++) {
        for (int j = 0; j < get_cols(); j++) {
            result.get_matrix()[i][j] = get_matrix()[i][j] + m2.get_matrix()[i][j];
        }
    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (!quiet) {
        std::cout << "add_classic_ref time (microseconds): " << elapsed.count() << std::endl;
    }
    return result;
}

//CPU threads

void Matrix::add_thread_row(const Matrix& m1, const Matrix& m2, std::vector<std::vector<float>>& result, int i)
{
    for (int j = 0; j < m2.cols; j++) {
        result[i][j] = m1.matrix[i][j] + m2.matrix[i][j];
    }
}

//-------------------------DETERMINANT-------------------------//

double Matrix::determinant_gauss() {
    std::vector<std::vector<float>> m = matrix;
    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
    int swapper = 1;
    for (int i = 0; i < rows; i++) {
        int pivot_row = find_max(m, i, i);
        double pivot = m[pivot_row][i];
        if (fabs(pivot) < 1e-12f) {
            return 0;
        }
        if (pivot_row != i) {
            swapper *= -1;
            std::swap(m[pivot_row], m[i]);
        }
        for (int j = i + 1; j < rows; j++) {
            double factor = m[j][i] / pivot;
            for (int k = i; k < cols; k++) {
                m[j][k] -= factor * m[i][k];
            }
        }

    }
    std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "determinant_gauss time (microseconds): " << elapsed.count() << std::endl;
    return swapper * multiply_diagonal(m);
}

//CPU threads
void Matrix::determinant_thread(Matrix& m, int row, int pivot_row) {
    double pivot = m.get_matrix()[pivot_row][pivot_row];
    double factor = m.get_matrix()[row][pivot_row] / pivot;

    for (int k = pivot_row; k < m.get_cols(); ++k) {
        m.get_matrix()[row][k] -= factor * m.get_matrix()[pivot_row][k];
    }
}

double Matrix::determinant_gauss_call_thread_func(void(*func)(Matrix& m, int row, int pivot_row)) {
    Matrix m_copy(matrix);
    int swapper = 1;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < m_copy.get_rows(); ++i) {
        int pivot_row = find_max(m_copy.get_matrix(), i, i);
        double pivot = m_copy.get_matrix()[pivot_row][i];

        if (fabs(pivot) < 1e-12f) return 0;

        if (pivot_row != i) {
            swapper *= -1;
            std::swap(m_copy.get_matrix()[pivot_row], m_copy.get_matrix()[i]);
        }

        std::vector<std::thread> threads;
        for (int j = i + 1; j < m_copy.get_rows(); ++j) {
            threads.emplace_back(func, std::ref(m_copy), j, i);
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "determinant_gauss cpu threads time (microseconds): "
        << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
        << std::endl;

    return swapper * multiply_diagonal(m_copy.get_matrix());
}