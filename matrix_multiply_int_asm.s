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
// ARM64 assembly implementation of matrix multiplication for integers
// Function: void matrix_multiply_int_asm(const int32_t* A, const int32_t* B, int32_t* C, int N)
// 
// Parameters:
// x0 - pointer to matrix A
// x1 - pointer to matrix B
// x2 - pointer to result matrix C
// w3 - size of matrices (N)

.global _matrix_multiply_int_asm
.align 4

_matrix_multiply_int_asm:
    // Stack setup - save regs
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save args to callee-saved regs
    mov     x19, x0                // A
    mov     x20, x1                // B
    mov     x21, x2                // C
    mov     w22, w3                // N

    // N*4 for faster addressing (integers = 4 bytes)
    lsl     x27, x22, #2           // row stride = N*4
    
    // Loop counters: i, j, k for the three nested loops
    mov     x23, xzr               // i = 0 (row counter)

outer_loop_i:
    cmp     x23, x22
    b.ge    end_outer_loop_i       // if i >= N, end loop

    // i*N*4 for faster A row access
    mul     x28, x23, x27

    // Prefetch next A row
    add     x10, x23, #1
    cmp     x10, x22
    b.ge    skip_pf_a_row
    mul     x10, x10, x27
    add     x10, x19, x10
    prfm    pldl1keep, [x10]
skip_pf_a_row:

    mov     x24, xzr               // j = 0 (column counter)

inner_loop_j:
    cmp     x24, x22
    b.ge    end_inner_loop_j       // if j >= N, end loop

    // Calculate the position in C: C[i*N + j]
    add     x5, x21, x28
    add     x5, x5, x24, lsl #2    // &C[i*N + j]
    
    // Check if we can use SIMD (need 4+ columns for int32)
    add     x10, x24, #3
    cmp     x10, x22
    b.ge    standard_compute       // not enough cols left, use scalar

    // --- SIMD path (4 integers at once) ---
    movi    v14.4s, #0             // zero accumulator

    mov     x25, xzr               // k = 0 (depth counter)

    .align  7                      // 128-byte alignment for Apple Silicon cache lines
simd4_loop_k:
    // Check if we can unroll
    add     x10, x25, #3
    cmp     x10, x22
    b.ge    simd4_remainder_loop

    // Unroll 4 k iterations
    // k=0
    add     x6, x19, x28           // A + i*N*4
    add     x6, x6, x25, lsl #2    // A + i*N*4 + k*4
    ldr     w0, [x6]               // A[i][k]

    // Prefetch A
    add     x10, x25, #8
    cmp     x10, x22
    b.ge    skip_pf_a1
    add     x10, x6, #32
    prfm    pldl1keep, [x10]
skip_pf_a1:

    mul     x11, x25, x27          // k * N*4
    add     x8, x20, x11           // B + k*N*4 (start of row k)
    add     x8, x8, x24, lsl #2    // B + k*N*4 + j*4 (element j in row k)

    // Prefetch B
    add     x10, x25, #4
    cmp     x10, x22
    b.ge    skip_pf_b1
    add     x12, x25, #4
    mul     x12, x12, x27
    add     x12, x20, x12
    add     x12, x12, x24, lsl #2
    prfm    pldl1keep, [x12]
skip_pf_b1:

    ld1     {v1.4s}, [x8]          // B[k][j,j+1,j+2,j+3]
    dup     v0.4s, w0              // broadcast A[i][k]
    mla     v14.4s, v1.4s, v0.4s   // v14 += A[i][k] * B[k][j,j+1,j+2,j+3]

    // k=1
    add     x6, x6, #4             // A[i][k+1]
    ldr     w0, [x6]
    
    add     x11, x11, x27          // Add one more row stride
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ld1     {v1.4s}, [x8]          // B[k+1][j,j+1,j+2,j+3]
    
    dup     v0.4s, w0
    mla     v14.4s, v1.4s, v0.4s
    
    // k=2
    add     x6, x6, #4             // A[i][k+2]
    ldr     w0, [x6]
    
    add     x11, x11, x27          // Add one more row stride
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ld1     {v1.4s}, [x8]          // B[k+2][j,j+1,j+2,j+3]
    
    dup     v0.4s, w0
    mla     v14.4s, v1.4s, v0.4s
    
    // k=3
    add     x6, x6, #4             // A[i][k+3]
    ldr     w0, [x6]
    
    add     x11, x11, x27          // Add one more row stride
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ld1     {v1.4s}, [x8]          // B[k+3][j,j+1,j+2,j+3]
    
    dup     v0.4s, w0
    mla     v14.4s, v1.4s, v0.4s

    add     x25, x25, #4
    b       simd4_loop_k

simd4_remainder_loop:
    // Handle remaining k iterations
    cmp     x25, x22
    b.ge    end_simd4_loop_k

    // Process one k at a time
    add     x6, x19, x28
    add     x6, x6, x25, lsl #2
    ldr     w0, [x6]               // A[i][k]

    mul     x11, x25, x27          // k * N*4
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2

    ld1     {v1.4s}, [x8]          // B[k][j,j+1,j+2,j+3]
    dup     v0.4s, w0              // broadcast A[i][k]
    mla     v14.4s, v1.4s, v0.4s   // v14 += A[i][k] * B[k][j,j+1,j+2,j+3]

    add     x25, x25, #1
    b       simd4_remainder_loop

end_simd4_loop_k:
    // Store results
    st1     {v14.4s}, [x5]         // C[i][j,j+1,j+2,j+3]

    add     x24, x24, #4           // j += 4
    b       inner_loop_j

standard_compute:
    // Scalar path (single column)
    mov     w26, #0                // sum = 0
    mov     x25, xzr               // k = 0

    .align  7                      // 128-byte alignment for Apple Silicon cache lines
standard_loop:
    add     x10, x25, #3
    cmp     x10, x22
    b.ge    standard_remainder_loop

    // Unroll 4 k iterations
    // k=0
    add     x6, x19, x28
    add     x6, x6, x25, lsl #2

    // Prefetch A
    add     x10, x25, #8
    cmp     x10, x22
    b.ge    skip_pf_a2
    add     x10, x6, #32
    prfm    pldl1keep, [x10]
skip_pf_a2:

    ldr     w0, [x6]               // A[i][k]

    mul     x11, x25, x27          // k * N*4
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ldr     w1, [x8]               // B[k][j]

    mul     w1, w0, w1             // A[i][k] * B[k][j]
    add     w26, w26, w1           // sum += A[i][k] * B[k][j]

    // k=1
    add     x6, x6, #4             // A[i][k+1]
    ldr     w0, [x6]
    
    add     x11, x11, x27          // Add one more row stride
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ldr     w1, [x8]               // B[k+1][j]
    
    mul     w1, w0, w1             // A[i][k+1] * B[k+1][j]
    add     w26, w26, w1           // sum += A[i][k+1] * B[k+1][j]
    
    // k=2
    add     x6, x6, #4             // A[i][k+2]
    ldr     w0, [x6]
    
    add     x11, x11, x27          // Add one more row stride
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ldr     w1, [x8]               // B[k+2][j]
    
    mul     w1, w0, w1             // A[i][k+2] * B[k+2][j]
    add     w26, w26, w1           // sum += A[i][k+2] * B[k+2][j]
    
    // k=3
    add     x6, x6, #4             // A[i][k+3]
    ldr     w0, [x6]
    
    add     x11, x11, x27          // Add one more row stride
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ldr     w1, [x8]               // B[k+3][j]
    
    mul     w1, w0, w1             // A[i][k+3] * B[k+3][j]
    add     w26, w26, w1           // sum += A[i][k+3] * B[k+3][j]

    add     x25, x25, #4
    b       standard_loop

standard_remainder_loop:
    // Handle remaining k iterations
    cmp     x25, x22
    b.ge    end_standard_loop

    // Process one k at a time
    add     x6, x19, x28
    add     x6, x6, x25, lsl #2
    ldr     w0, [x6]               // A[i][k]

    mul     x11, x25, x27          // k * N*4
    add     x8, x20, x11
    add     x8, x8, x24, lsl #2
    ldr     w1, [x8]               // B[k][j]

    mul     w1, w0, w1             // A[i][k] * B[k][j]
    add     w26, w26, w1           // sum += A[i][k] * B[k][j]

    add     x25, x25, #1
    b       standard_remainder_loop

end_standard_loop:
    // Store result
    str     w26, [x5]

    add     x24, x24, #1           // j++
    b       inner_loop_j

end_inner_loop_j:
    add     x23, x23, #1           // i++
    b       outer_loop_i

end_outer_loop_i:
    // Restore callee-saved registers
    mov     sp, x29
    ldp     x29, x30, [sp], #16
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    
    ret                            // Return
