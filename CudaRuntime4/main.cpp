#include "call.h"

/* 
  This code was written by Bartosz Jakubowski as part of his bachelor's thesis. 

  The above application is used to compare the execution time of algorithms performing selected actions on arrays. 
  The operation of the mentioned algorithms, as well as their computational complexity, is discussed in the bachelor's thesis to which this code is attached.
  
  The implemented algorithms, in addition to their computational complexity, also differ in the way they are executed, i.e.: a selected part of them has been implemented
  in the version that executes on a single thread, multiple threads of the processor and on the graphics card, in order to show the differences in the execution time
  of operations on different devices.
  
  In order to test the relevant operations, a special interface Call was created, which performs all types of implemented algorithms for a given operation 
  (e.g.: Call::call_multiply() performs all available types of matrix multiplication).
  The method will also automatically print the results of the operation, but only if the resulting matrix is relatively small (no more than 8x8).
  It is also possible to call individual methods separately. 
  
  ATTENTION: Some of the algorithms (e.g. Strassen's algorithm) work correctly only for square matrices with sizes equal to the next powers of two (e.g. 32, 64, 128, etc.),
  so it is recommended to test the algorithms only on matrices of the specified sizes. 
*/


int main() {
    int rows = 64, cols = 64;
    Matrix matrix1(rows, cols);
    Matrix matrix2(matrix1);

    //print matrix if it is small
    if(rows<=8)
    std::cout << "matrix1:" << std::endl;
    print_if_small(matrix1);
    if (rows <= 8)
    std::cout << std::endl << "matrix2:" << std::endl;
    print_if_small(matrix2);

    //TESTS
    Call::call_multiply(matrix1, matrix2);
    Call::call_add(matrix1, matrix2);
    Call::call_determinant(matrix1);

    return 0;
}