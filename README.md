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

ATTENTION: For proper operation of the application, it is necessary to have an NVIDIA graphics card and install the CUDA toolkit. You can download it at the following link: https://developer.nvidia.com/cuda-12-8-0-download-archive.
Program was written on CUDA version 12.8, so it is recomended using this version of CUDA for the correct operation of the application.
