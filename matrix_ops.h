//
// MIT License
// 
// Copyright (c) 2025 Timo Heimonen <timo.heimonen@gmail.com>
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Assembly functions for matrix multiplication
// Floating-point matrix multiplication (double precision)
void matrix_multiply_fp_asm(const double* A, const double* B, double* C, int N);

// 32-bit integer matrix multiplication
void matrix_multiply_int_asm(const int32_t* A, const int32_t* B, int32_t* C, int N);

// 64-bit integer matrix multiplication
void matrix_multiply_int64_asm(const int64_t* A, const int64_t* B, int64_t* C, int N);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPS_H