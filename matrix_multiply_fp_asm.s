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
// ARM64 assembly implementation of matrix multiplication for floating point
// Function: void matrix_multiply_fp_asm(const double* A, const double* B, double* C, int N)
// 
// Parameters:
// x0 - pointer to matrix A
// x1 - pointer to matrix B
// x2 - pointer to result matrix C
// w3 - size of matrices (N)

.global _matrix_multiply_fp_asm
.align 4

_matrix_multiply_fp_asm:
    // Stack setup - save regs
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!        // SIMD regs
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    // Save args to callee-saved regs
    mov x19, x0                    // A
    mov x20, x1                    // B
    mov x21, x2                    // C
    mov w22, w3                    // N

    // N*8 for faster addressing (doubles = 8 bytes)
    lsl x27, x22, #3               // row stride = N*8

    mov x23, xzr                   // i = 0 (row counter)

outer_loop_i:
    cmp x23, x22
    b.ge end_outer_loop_i

    // i*N*8 for faster A row access
    mul x28, x23, x27

    // Prefetch next A row
    add x10, x23, #1
    cmp x10, x22
    b.ge skip_pf_a_row
    mul x10, x10, x27
    add x10, x19, x10
    prfm pldl1keep, [x10]
skip_pf_a_row:

    mov x24, xzr                   // j = 0 (col counter)

inner_loop_j:
    cmp x24, x22
    b.ge end_inner_loop_j

    // Get C[i][j] addr
    add x5, x21, x28
    add x5, x5, x24, lsl #3

    // Check if we can use SIMD (need 2+ columns)
    add x10, x24, #1
    cmp x10, x22
    b.ge standard_compute          // not enough cols left, use scalar

    // --- SIMD path (2 columns at once) ---
    movi v14.2d, #0                // zero accumulator

    mov x25, xzr                   // k = 0

    .align 7                       // 128-byte alignment for Apple Silicon cache lines
simd2_loop_k:
    // Check if we can unroll
    add x10, x25, #3
    cmp x10, x22
    b.ge simd2_remainder_loop

    // Unroll 4 k iterations
    // k=0
    add x6, x19, x28
    add x6, x6, x25, lsl #3
    ldr d0, [x6]                   // A[i][k]

    // Prefetch A
    add x10, x25, #8
    cmp x10, x22
    b.ge skip_pf_a1
    add x10, x6, #64
    prfm pldl1keep, [x10]
skip_pf_a1:

    mul x11, x25, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3

    // Prefetch B
    add x10, x25, #4
    cmp x10, x22
    b.ge skip_pf_b1
    add x12, x25, #4
    mul x12, x12, x27
    add x12, x20, x12
    add x12, x12, x24, lsl #3
    prfm pldl1keep, [x12]
skip_pf_b1:

    ld1 {v1.2d}, [x8]              // B[k][j,j+1]

    dup v0.2d, v0.d[0]             // broadcast A[i][k]
    fmla v14.2d, v1.2d, v0.2d      // v14 += A[i][k] * B[k][j,j+1]

    // k=1
    add x6, x6, #8
    ldr d0, [x6]

    add x11, x11, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ld1 {v1.2d}, [x8]

    dup v0.2d, v0.d[0]
    fmla v14.2d, v1.2d, v0.2d

    // k=2
    add x6, x6, #8
    ldr d0, [x6]

    add x11, x11, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ld1 {v1.2d}, [x8]

    dup v0.2d, v0.d[0]
    fmla v14.2d, v1.2d, v0.2d

    // k=3
    add x6, x6, #8
    ldr d0, [x6]

    add x11, x11, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ld1 {v1.2d}, [x8]

    dup v0.2d, v0.d[0]
    fmla v14.2d, v1.2d, v0.2d

    add x25, x25, #4
    b simd2_loop_k

simd2_remainder_loop:
    // Handle remaining k iterations
    cmp x25, x22
    b.ge end_simd2_loop_k

    add x6, x19, x28
    add x6, x6, x25, lsl #3
    ldr d0, [x6]

    mul x11, x25, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ld1 {v1.2d}, [x8]

    dup v0.2d, v0.d[0]
    fmla v14.2d, v1.2d, v0.2d

    add x25, x25, #1
    b simd2_remainder_loop

end_simd2_loop_k:
    // Store results
    st1 {v14.2d}, [x5]             // C[i][j,j+1]

    add x24, x24, #2               // j += 2
    b inner_loop_j

standard_compute:
    // Scalar path (single column)
    fmov d8, #0.0                  // acc = 0

    mov x25, xzr                   // k = 0

    .align 7                       // 128-byte alignment for Apple Silicon cache lines
standard_loop:
    add x10, x25, #3
    cmp x10, x22
    b.ge standard_remainder_loop

    // Unroll 4 k iterations
    // k=0
    add x6, x19, x28
    add x6, x6, x25, lsl #3

    add x10, x25, #8
    cmp x10, x22
    b.ge skip_pf_a2
    add x10, x6, #64
    prfm pldl1keep, [x10]
skip_pf_a2:

    ldr d0, [x6]                   // A[i][k]

    mul x11, x25, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ldr d1, [x8]                   // B[k][j]

    fmadd d8, d0, d1, d8           // acc += A[i][k] * B[k][j]

    // k=1
    add x6, x6, #8
    ldr d0, [x6]

    add x11, x11, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ldr d1, [x8]

    fmadd d8, d0, d1, d8

    // k=2
    add x6, x6, #8
    ldr d0, [x6]

    add x11, x11, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ldr d1, [x8]

    fmadd d8, d0, d1, d8

    // k=3
    add x6, x6, #8
    ldr d0, [x6]

    add x11, x11, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ldr d1, [x8]

    fmadd d8, d0, d1, d8

    add x25, x25, #4
    b standard_loop

standard_remainder_loop:
    // Handle remaining k iterations
    cmp x25, x22
    b.ge end_standard_loop

    add x6, x19, x28
    add x6, x6, x25, lsl #3
    ldr d0, [x6]

    mul x11, x25, x27
    add x8, x20, x11
    add x8, x8, x24, lsl #3
    ldr d1, [x8]

    fmadd d8, d0, d1, d8

    add x25, x25, #1
    b standard_remainder_loop

end_standard_loop:
    // Store result
    str d8, [x5]

    add x24, x24, #1               // j++
    b inner_loop_j

end_inner_loop_j:
    add x23, x23, #1               // i++
    b outer_loop_i

end_outer_loop_i:
    // Cleanup and return
    mov sp, x29
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret