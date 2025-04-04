# SiliconMatrix

A high-performance matrix multiplication library implemented in ARM64 assembly for Apple Silicon processors.

## Overview

This project demonstrates optimized matrix multiplication for Apple Silicon using ARM64 assembly code enhanced with SIMD, cache prefetching, and loop unrolling.

The project includes:
- Floating-point matrix multiplication (64-bit doubles)
- Integer matrix multiplication (32-bit integers)
- Integer matrix multiplication (64-bit integers)
- A simple C++ interface for easy integration

## Features

- **SIMD Optimization**: Uses Advanced SIMD (NEON) instructions for parallel processing
- **Cache Prefetching**: Strategic prefetching to minimize cache misses
- **Loop Unrolling**: Manual loop unrolling for improved instruction-level parallelism
- **Clean C++ Interface**: Simple header-only integration for C++ applications

## Requirements

- macOS running on Apple Silicon (M1/M2/M3/M4..)
- Clang compiler (comes with Xcode)
- Make (optional, for building with the provided Makefile)

## Building

To build the project:

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd matrixcalculator

# Build with make (using -O2 optimization by default)
make

# Build with specific optimization levels
make o0    # No optimization (-O0)
make o1    # Basic optimization (-O1)
make o3    # High optimization (-O3)
make ofast # Maximum optimization (-Ofast)

# Or manually with clang
clang++ -std=c++17 -O2 -Wall -Wextra main.cpp matrix_multiply_fp_asm.s matrix_multiply_int_asm.s matrix_multiply_int64_asm.s -o SiliconMatrix
```

## Usage Example

The project includes a simple demonstration program showing how to use the assembly-optimized matrix multiplication functions:

```cpp
#include <vector>
#include "matrix_ops.h"

// Create matrices
std::vector<double> A = {...}; // Your matrix data
std::vector<double> B = {...}; // Your matrix data
std::vector<double> C(N * N, 0.0); // Result matrix

// Multiply matrices using the assembly function
matrix_multiply_fp_asm(A.data(), B.data(), C.data(), N);
```

## Performance

The assembly implementations are designed for maximum performance on Apple Silicon processors. Key optimizations include:

1. **SIMD Processing**: Processes multiple elements simultaneously
2. **Prefetching**: Reduces memory access latency
3. **Register Usage**: Careful register allocation to minimize memory operations
4. **Loop Structure**: Optimized loop nesting and unrolling for better instruction pipelining

### Benchmark Results

Below are actual performance measurements comparing the assembly implementations against equivalent C++ implementations at different optimization levels:

#### Matrix size: 200×200 with -O3 optimization
| Implementation Type | 32-bit Integer | 64-bit Integer | Floating-point |
|---------------------|----------------|----------------|----------------|
| C++ implementation  | 3.41 ms        | 2.23 ms        | 4.17 ms        |
| ASM implementation  | 1.99 ms        | 3.98 ms        | 2.30 ms        |
| **Speedup**         | **1.72x**      | **0.56x**      | **1.81x**      |

#### Matrix size: 1000×1000 with -O3 optimization
| Implementation Type | 32-bit Integer | 64-bit Integer | Floating-point |
|---------------------|----------------|----------------|----------------|
| C++ implementation  | 228.94 ms      | 471.55 ms      | 711.10 ms      |
| ASM implementation  | 164.41 ms      | 255.81 ms      | 392.67 ms      |
| **Speedup**         | **1.39x**      | **1.84x**      | **1.81x**      |

#### Effect of Compiler Optimization Levels on Performance

Our testing revealed interesting patterns about how compiler optimization levels affect both C++ and assembly implementations:

##### C++ Implementation Performance Across Optimization Levels (1000×1000 matrix)
| Optimization | 32-bit Integer | 64-bit Integer | Floating-point |
|--------------|----------------|----------------|----------------|
| -O0 (none)   | 2280.83 ms     | 2366.46 ms     | 2366.08 ms     |
| -O1 (basic)  | 367.55 ms      | 510.87 ms      | 711.56 ms      |
| -O3 (high)   | 228.94 ms      | 471.55 ms      | 711.10 ms      |
| -Ofast (max) | 227.94 ms      | 470.75 ms      | 446.99 ms      |

##### Assembly Implementation Performance Across Optimization Levels (1000×1000 matrix)
| Optimization | 32-bit Integer | 64-bit Integer | Floating-point |
|--------------|----------------|----------------|----------------|
| -O0 (none)   | 165.67 ms      | 264.13 ms      | 394.00 ms      |
| -O1 (basic)  | 165.30 ms      | 260.40 ms      | 395.44 ms      |
| -O3 (high)   | 164.41 ms      | 255.81 ms      | 392.67 ms      |
| -Ofast (max) | 166.98 ms      | 254.87 ms      | 392.81 ms      |

#### Key Observations:

1. **C++ Performance**: The C++ implementation's performance varies dramatically with optimization levels, showing up to 10x improvement from -O0 to -O3.

2. **Assembly Stability**: The assembly implementations show relatively consistent performance across optimization levels, demonstrating the value of hand-optimized code.

3. **Small vs. Large Matrices**: For 64-bit integers, our assembly implementation:
   - Is slower than C++ for small matrices (200×200)
   - Shows significant speedup (1.84x) for large matrices (1000×1000)

4. **SIMD Benefits**: The 32-bit integer and floating-point implementations benefit from NEON SIMD instructions and consistently outperform their C++ counterparts.

5. **-Ofast Impact**: The -Ofast optimization level shows particular improvement for floating-point C++ code, but can compromise numerical accuracy.

## Implementation Details

### 64-bit Integer Implementation

The 64-bit integer implementation uses several advanced optimization techniques:

1. **Multi-column Processing**: Processes 4 columns at once in the main loop with fallbacks to 2 columns and single column processing for remainders
2. **Aggressive Loop Unrolling**: Reduces branch mispredictions and instruction fetch overhead
3. **Strategic Prefetching**: Looks ahead in both A and B matrices to ensure data is in cache before needed
4. **Paired Load/Store Operations**: Uses `ldp`/`stp` instructions to improve memory bandwidth utilization
5. **Multiply-Accumulate Instructions**: Uses `madd` instructions for efficient computation
6. **Size-Adaptive Algorithms**: Employs different algorithms based on matrix size for optimal performance

### 32-bit Integer Implementation

The 32-bit implementation leverages NEON SIMD instructions to process multiple elements in parallel:

1. **SIMD Vectorization**: Processes 4 int32 values simultaneously using NEON
2. **Vector Broadcast**: Efficiently reuses A matrix values across calculations
3. **Aligned Memory Access**: Ensures optimal memory throughput

### Floating-point Implementation

The floating-point implementation combines SIMD operations with efficient memory access:

1. **SIMD Processing**: Uses NEON to process pairs of double-precision values
2. **FMA Instructions**: Employs fused multiply-add for improved accuracy and performance
3. **Advanced Loop Structure**: Adapts to optimal processing width based on remaining elements

## License

MIT License - See the LICENSE file for details.

## Author

Timo Heimonen <timo.heimonen@gmail.com>
