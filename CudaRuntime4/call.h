#pragma once
#include "kernel.h"
#include <functional>

class Call {
public:
	static void call_multiply(Matrix& m1, Matrix& m2);
	static void call_add(Matrix& m1, Matrix& m2);
	static void call_determinant(Matrix& m1);
};

void print_if_small(Matrix m);