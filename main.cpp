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
// This program demonstrates the use of assembly functions for matrix multiplication
// with different data types (int32_t, int64_t, double). It includes a C++ program
// that initializes matrices, multiplies them using both C++ and assembly
// implementations, and compares performance and accuracy.

#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include "src/matrix_ops.h"
#include <random>
#include <chrono> 

// Helper function to print a matrix with floating point values
void print_matrix(const std::vector<double>& matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to generate a random matrix of int32_t values
std::vector<int32_t> generateRandomMatrix32(int size) {
    std::vector<int32_t> matrix(size * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(-100, 100);
    
    for (auto& val : matrix) {
        val = dist(gen);
    }
    
    return matrix;
}

// Function to generate a random matrix of int64_t values
std::vector<int64_t> generateRandomMatrix64(int size) {
    std::vector<int64_t> matrix(size * size);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int64_t> dist(-100, 100);
    
    for (auto& val : matrix) {
        val = dist(gen);
    }
    
    return matrix;
}

// Function to generate a random matrix of double values
std::vector<double> generateRandomMatrixDouble(int size) {
    std::vector<double> matrix(size * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    
    for (auto& val : matrix) {
        val = dist(gen);
    }
    
    return matrix;
}

// Native C++ 32-bit integer matrix multiplication for comparison
void matrixMultiplyCpp32(const std::vector<int32_t>& A, const std::vector<int32_t>& B, 
                       std::vector<int32_t>& C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int32_t sum = 0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Native C++ 64-bit integer matrix multiplication for comparison
void matrixMultiplyCpp64(const std::vector<int64_t>& A, const std::vector<int64_t>& B, 
                       std::vector<int64_t>& C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int64_t sum = 0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Native C++ floating-point matrix multiplication for comparison
void matrixMultiplyCppDouble(const std::vector<double>& A, const std::vector<double>& B, 
                           std::vector<double>& C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Function to print a matrix (for small matrices)
template<typename T>
void printMatrix(const std::vector<T>& matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << std::setw(8) << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to verify matrix multiplication results
template<typename T>
bool verifyResults(const std::vector<T>& expected, const std::vector<T>& actual, int size) {
    for (int i = 0; i < size * size; i++) {
        if (expected[i] != actual[i]) {
            std::cout << "Mismatch at [" << i / size << "][" << i % size 
                      << "]: Expected " << expected[i] << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Function to benchmark matrix multiplication (templated for different data types)
template<typename T>
double benchmark(void (*func)(const std::vector<T>&, const std::vector<T>&, std::vector<T>&, int),
                const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int size, int runs = 5) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run multiple times to get a more accurate measurement
    for (int i = 0; i < runs; i++) {
        func(A, B, C, size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Return average time per run in milliseconds
    return duration.count() / runs;
}

// Function to benchmark ASM matrix multiplication for 32-bit integers
double benchmark_asm_int32(const std::vector<int32_t>& A, const std::vector<int32_t>& B, 
                          std::vector<int32_t>& C, int size, int runs = 5) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < runs; i++) {
        matrix_multiply_int_asm(A.data(), B.data(), C.data(), size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / runs;
}

// Function to benchmark ASM matrix multiplication for 64-bit integers
double benchmark_asm_int64(const std::vector<int64_t>& A, const std::vector<int64_t>& B, 
                          std::vector<int64_t>& C, int size, int runs = 5) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < runs; i++) {
        matrix_multiply_int64_asm(A.data(), B.data(), C.data(), size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / runs;
}

// Function to benchmark ASM matrix multiplication for floating-point (double)
double benchmark_asm_double(const std::vector<double>& A, const std::vector<double>& B, 
                          std::vector<double>& C, int size, int runs = 5) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < runs; i++) {
        matrix_multiply_fp_asm(A.data(), B.data(), C.data(), size);  // Use consistent function name
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / runs;
}

int main() {
    std::cout << "SiliconMatrix: High-Performance Matrix Operations\n";
    std::cout << "================================================\n\n";
    
    // Example Matrix Output for small matrices
    std::cout << "Matrix Multiplication Examples (3x3)\n";
    std::cout << "==================================\n\n";
    
    // 32-bit Integer Example
    {
        int example_size = 3;
        auto A32 = generateRandomMatrix32(example_size);
        auto B32 = generateRandomMatrix32(example_size);
        std::vector<int32_t> C32_cpp(example_size * example_size, 0);
        std::vector<int32_t> C32_asm(example_size * example_size, 0);
        
        matrixMultiplyCpp32(A32, B32, C32_cpp, example_size);
        matrix_multiply_int_asm(A32.data(), B32.data(), C32_asm.data(), example_size);
        
        std::cout << "32-bit Integer Matrix Example:\n";
        std::cout << "Matrix A:\n";
        printMatrix(A32, example_size);
        std::cout << "Matrix B:\n";
        printMatrix(B32, example_size);
        std::cout << "Result Matrix C (A × B) - C++ implementation:\n";
        printMatrix(C32_cpp, example_size);
        std::cout << "Result Matrix C (A × B) - ASM implementation:\n";
        printMatrix(C32_asm, example_size);
        
        bool correct = verifyResults(C32_cpp, C32_asm, example_size);
        std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;
        std::cout << "-----------------------------------\n\n";
    }
    
    // 64-bit Integer Example
    {
        int example_size = 3;
        auto A64 = generateRandomMatrix64(example_size);
        auto B64 = generateRandomMatrix64(example_size);
        std::vector<int64_t> C64_cpp(example_size * example_size, 0);
        std::vector<int64_t> C64_asm(example_size * example_size, 0);
        
        matrixMultiplyCpp64(A64, B64, C64_cpp, example_size);
        matrix_multiply_int64_asm(A64.data(), B64.data(), C64_asm.data(), example_size);
        
        std::cout << "64-bit Integer Matrix Example:\n";
        std::cout << "Matrix A:\n";
        printMatrix(A64, example_size);
        std::cout << "Matrix B:\n";
        printMatrix(B64, example_size);
        std::cout << "Result Matrix C (A × B) - C++ implementation:\n";
        printMatrix(C64_cpp, example_size);
        std::cout << "Result Matrix C (A × B) - ASM implementation:\n";
        printMatrix(C64_asm, example_size);
        
        bool correct = verifyResults(C64_cpp, C64_asm, example_size);
        std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;
        std::cout << "-----------------------------------\n\n";
    }
    
    // Floating-point Example
    {
        int example_size = 3;
        auto A_double = generateRandomMatrixDouble(example_size);
        auto B_double = generateRandomMatrixDouble(example_size);
        std::vector<double> C_double_cpp(example_size * example_size, 0.0);
        std::vector<double> C_double_asm(example_size * example_size, 0.0);
        
        matrixMultiplyCppDouble(A_double, B_double, C_double_cpp, example_size);
        matrix_multiply_fp_asm(A_double.data(), B_double.data(), C_double_asm.data(), example_size);
        
        std::cout << "Floating-point Matrix Example:\n";
        std::cout << "Matrix A:\n";
        printMatrix(A_double, example_size);
        std::cout << "Matrix B:\n";
        printMatrix(B_double, example_size);
        std::cout << "Result Matrix C (A × B) - C++ implementation:\n";
        printMatrix(C_double_cpp, example_size);
        std::cout << "Result Matrix C (A × B) - ASM implementation:\n";
        printMatrix(C_double_asm, example_size);
        
        // Verify results - need special comparison for floating point
        bool correct = true;
        for (int i = 0; i < example_size * example_size; i++) {
            double diff = std::abs(C_double_cpp[i] - C_double_asm[i]);
            double tolerance = 1e-10 * (std::abs(C_double_cpp[i]) + std::abs(C_double_asm[i]));
            if (diff > tolerance) {
                std::cout << "Mismatch at [" << i / example_size << "][" << i % example_size 
                          << "]: Expected " << C_double_cpp[i] << ", got " << C_double_asm[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;
        std::cout << "-----------------------------------\n\n";
    }
    
    // Add benchmarking section
    std::cout << "Performance Benchmarks\n";
    std::cout << "=====================\n\n";
    
    // Matrix sizes to benchmark
    std::vector<int> matrix_sizes = {200, 500, 1000};
    
    // For each matrix size
    for (int size : matrix_sizes) {
        std::cout << "Matrix size: " << size << "x" << size << std::endl;
        
        // 32-bit integer benchmark
        {
            auto A32 = generateRandomMatrix32(size);
            auto B32 = generateRandomMatrix32(size);
            std::vector<int32_t> C32_cpp(size * size, 0);
            std::vector<int32_t> C32_asm(size * size, 0);
            
            // Warmup run
            matrixMultiplyCpp32(A32, B32, C32_cpp, size);
            matrix_multiply_int_asm(A32.data(), B32.data(), C32_asm.data(), size);
            
            // Benchmark C++ implementation
            double cpp_time = benchmark(matrixMultiplyCpp32, A32, B32, C32_cpp, size);
            
            // Benchmark ASM implementation
            double asm_time = benchmark_asm_int32(A32, B32, C32_asm, size);
            
            // Calculate speedup
            double speedup = cpp_time / asm_time;
            
            // Verify results
            bool correct = verifyResults(C32_cpp, C32_asm, size);
            
            // Print results
            std::cout << "  32-bit Integer Matrix Multiplication:" << std::endl;
            std::cout << "  C++ implementation: " << cpp_time << " ms" << std::endl;
            std::cout << "  ASM implementation: " << asm_time << " ms" << std::endl;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
            std::cout << "  Results match: " << (correct ? "Yes" : "No") << std::endl << std::endl;
        }
        
        // 64-bit integer benchmark
        {
            auto A64 = generateRandomMatrix64(size);
            auto B64 = generateRandomMatrix64(size);
            std::vector<int64_t> C64_cpp(size * size, 0);
            std::vector<int64_t> C64_asm(size * size, 0);
            
            // Warmup run
            matrixMultiplyCpp64(A64, B64, C64_cpp, size);
            matrix_multiply_int64_asm(A64.data(), B64.data(), C64_asm.data(), size);
            
            // Benchmark C++ implementation
            double cpp_time = benchmark(matrixMultiplyCpp64, A64, B64, C64_cpp, size);
            
            // Benchmark ASM implementation
            double asm_time = benchmark_asm_int64(A64, B64, C64_asm, size);
            
            // Calculate speedup
            double speedup = cpp_time / asm_time;
            
            // Verify results
            bool correct = verifyResults(C64_cpp, C64_asm, size);
            
            // Print results
            std::cout << "  64-bit Integer Matrix Multiplication:" << std::endl;
            std::cout << "  C++ implementation: " << cpp_time << " ms" << std::endl;
            std::cout << "  ASM implementation: " << asm_time << " ms" << std::endl;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
            std::cout << "  Results match: " << (correct ? "Yes" : "No") << std::endl << std::endl;
        }
        
        // Floating-point (double) benchmark
        {
            auto A_double = generateRandomMatrixDouble(size);
            auto B_double = generateRandomMatrixDouble(size);
            std::vector<double> C_double_cpp(size * size, 0.0);
            std::vector<double> C_double_asm(size * size, 0.0);
            
            // Warmup run
            matrixMultiplyCppDouble(A_double, B_double, C_double_cpp, size);
            matrix_multiply_fp_asm(A_double.data(), B_double.data(), C_double_asm.data(), size);
            
            // Benchmark C++ implementation
            double cpp_time = benchmark(matrixMultiplyCppDouble, A_double, B_double, C_double_cpp, size);
            
            // Benchmark ASM implementation
            double asm_time = benchmark_asm_double(A_double, B_double, C_double_asm, size);
            
            // Calculate speedup
            double speedup = cpp_time / asm_time;
            
            // Verify results with proper floating-point comparison
            bool correct = true;
            for (int i = 0; i < size * size; i++) {
                double diff = std::abs(C_double_cpp[i] - C_double_asm[i]);
                double tolerance = 1e-10 * (std::abs(C_double_cpp[i]) + std::abs(C_double_asm[i]));
                if (diff > tolerance) {
                    correct = false;
                    break;
                }
            }
            
            // Print results
            std::cout << "  Floating-point (double) Matrix Multiplication:" << std::endl;
            std::cout << "  C++ implementation: " << cpp_time << " ms" << std::endl;
            std::cout << "  ASM implementation: " << asm_time << " ms" << std::endl;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
            std::cout << "  Results match: " << (correct ? "Yes" : "No") << std::endl << std::endl;
        }
        
        // Add a separator between different matrix sizes
        std::cout << "--------------------------------------------" << std::endl << std::endl;
    }
    
    return 0;
}
