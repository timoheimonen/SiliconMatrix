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


/*
 * Function: _matrix_multiply_fp_asm
 * ---------------------------------
 * Performs matrix multiplication for floating-point matrices in assembly.
 *
 * Stack Setup:
 * - Saves general-purpose registers (x19-x30) and SIMD registers (d8-d15) onto the stack.
 * - Sets up the frame pointer (x29) to the current stack pointer.
 *
 * Arguments:
 * - x0 (x19): Pointer to matrix A.
 * - x1 (x20): Pointer to matrix B.
 * - x2 (x21): Pointer to matrix C (result matrix).
 * - w3 (w22): Dimension N (assumes square matrices of size N x N).
 *
 * Precomputations:
 * - Calculates row stride (N * 8) for efficient addressing of double-precision elements.
 *
 * Initialization:
 * - Initializes row counter (i) to 0 (stored in x23).
 */
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


// This section of the code implements the outer loop for iterating over rows of matrix A.
// - `outer_loop_i`: Label for the start of the outer loop.
// - Compares the current row index `x23` with the total number of rows `x22`.
// - If `x23 >= x22`, the loop ends and branches to `end_outer_loop_i`.
//
// Inside the loop:
// - Calculates the offset for the current row of matrix A by multiplying `x23` (current row index) 
//   by `x27` (N * 8, where N is the number of columns in A, and 8 is the size of each element).
//   The result is stored in `x28`.
//
// Prefetching:
// - Prepares for the next iteration by prefetching the next row of matrix A into the L1 cache.
// - Increments `x23` by 1 to get the next row index and checks if it is within bounds (`x10 < x22`).
// - If within bounds, calculates the memory address of the next row by multiplying the next row index
//   by `x27` and adding the base address of matrix A (`x19`).
// - Uses the `prfm` instruction to prefetch the calculated address into the L1 cache for faster access.
//
// If the next row index is out of bounds, the prefetching step is skipped.
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

/**
 * Label: skip_pf_a_row
 * 
 * This label marks the beginning of a section of code responsible for skipping
 * the prefetching of a row in matrix A. It initializes the column counter `j`
 * to 0 by moving the zero register (`xzr`) into register `x24`.
 * 
 * Registers:
 * - x24: Used as the column counter (`j`), initialized to 0.
 * - xzr: Zero register, always contains the value 0.
 */
skip_pf_a_row:

    mov x24, xzr                   // j = 0 (col counter)

// inner_loop_j:
// This label marks the beginning of the inner loop for iterating over columns (j).
// 
// cmp x24, x22
// Compares the current column index (x24) with the total number of columns (x22).
// If x24 >= x22, the loop ends and branches to end_inner_loop_j.
//
// add x5, x21, x28
// add x5, x5, x24, lsl #3
// Calculates the address of the element C[i][j] in the result matrix.
// x21 holds the base address of matrix C, x28 is the row offset, and x24 is the column index.
// The column index is left-shifted by 3 (multiplied by 8) to account for 64-bit elements.
//
// add x10, x24, #1
// cmp x10, x22
// b.ge standard_compute
// Checks if there are at least 2 columns remaining to process.
// If not, branches to standard_compute to handle the computation using scalar operations.
//
// movi v14.2d, #0
// Initializes the SIMD accumulator register (v14) to zero for accumulating results
// when processing two columns at a time.
//
// mov x25, xzr
// Initializes the loop counter (k) to 0 for iterating over the shared dimension.
//
// .align 7
// Aligns the following instructions to a 128-byte boundary for optimal performance
// on Apple Silicon, which benefits from cache line alignment.
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


/*
    simd2_loop_k:
    This label marks the start of a loop that processes matrix multiplication using SIMD instructions.

    - The loop first checks if there are enough remaining iterations to unroll the loop by 4.
      - Adds 3 to the current loop index (x25) and compares it with the loop limit (x22).
      - If the unrolling is not possible (x10 >= x22), it branches to `simd2_remainder_loop`.

    - If unrolling is possible:
      - Performs the first iteration of the unrolled loop (k=0):
        - Calculates the address of the current matrix element A[i][k] using x19 (base address of A), x28 (row offset), and x25 (column offset).
        - Loads the value of A[i][k] into SIMD register d0.

      - Prefetches the next block of matrix A data to optimize memory access:
        - Adds 8 to the current loop index (x25) and checks if it exceeds the loop limit (x22).
        - If prefetching is valid, calculates the address of the next block and issues a prefetch instruction (prfm) to load it into the L1 cache.
*/
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


/*
    skip_pf_a1:
    This label marks the beginning of a section of code responsible for calculating
    memory addresses and optionally prefetching data for matrix multiplication.

    - The first part calculates the address for a specific element in matrix A:
        - `mul x11, x25, x27`: Multiplies the row index (x25) by the row stride (x27).
        - `add x8, x20, x11`: Adds the base address of matrix A (x20) to the result.
        - `add x8, x8, x24, lsl #3`: Adds the column offset (x24) shifted left by 3 (multiplied by 8).

    - The second part conditionally prefetches data from matrix B:
        - `add x10, x25, #4`: Calculates the next row index (x25 + 4).
        - `cmp x10, x22`: Compares the next row index with the total number of rows (x22).
        - `b.ge skip_pf_b1`: Skips prefetching if the next row index exceeds the total rows.
        - If prefetching is not skipped:
            - `add x12, x25, #4`: Calculates the next row index again.
            - `mul x12, x12, x27`: Multiplies the next row index by the row stride.
            - `add x12, x20, x12`: Adds the base address of matrix B (x20).
            - `add x12, x12, x24, lsl #3`: Adds the column offset shifted left by 3.
            - `prfm pldl1keep, [x12]`: Prefetches the calculated address into the L1 cache.
*/
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



/*
    This section of the code performs a portion of a matrix multiplication operation
    using ARM NEON SIMD instructions. The operation involves multiplying elements
    from matrix A and matrix B, and accumulating the results into a destination
    register.

    - `skip_pf_b1`: Label marking the start of this section.
    - `ld1 {v1.2d}, [x8]`: Loads two double-precision floating-point values from
      matrix B into vector register v1.
    - `dup v0.2d, v0.d[0]`: Broadcasts the first double-precision value from
      vector register v0 to all elements of v0.
    - `fmla v14.2d, v1.2d, v0.2d`: Performs a fused multiply-add operation, 
      accumulating the product of v1 and v0 into v14.

    The loop iterates over four values of `k` (k=0, k=1, k=2, k=3), performing the
    following steps for each iteration:
    1. Increment the pointer to matrix A (`x6`) to load the next element.
    2. Load the next value from matrix A into register `d0`.
    3. Update the pointer to matrix B (`x8`) using offsets and strides.
    4. Load the next two values from matrix B into vector register `v1`.
    5. Broadcast the loaded value from matrix A and perform the fused multiply-add
       operation with the corresponding values from matrix B.

    After processing all four iterations, the loop increments the pointer `x25`
    and branches back to the main loop (`simd2_loop_k`) for further processing.
*/
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


/*
    simd2_remainder_loop:
    This loop handles the remaining iterations of the matrix multiplication
    process when the main SIMD loop cannot fully process all elements due to
    the size of the input. It performs the following steps:

    - Compares the current loop counter (x25) with the total number of iterations (x22).
      If the counter exceeds or equals the total, the loop exits (branch to end_simd2_loop_k).

    - Calculates the address of the current element in the first matrix (x6) using
      base address (x19), row offset (x28), and column offset (x25, scaled by 8 bytes).

    - Loads a double-precision floating-point value (d0) from the calculated address.

    - Computes the offset for the second matrix (x11) using the loop counter (x25)
      and the column stride (x27). Adds this offset to the base address (x20) and
      row offset (x24, scaled by 8 bytes) to calculate the address (x8).

    - Loads a 128-bit SIMD register (v1.2d) containing two double-precision values
      from the calculated address (x8).

    - Duplicates the loaded scalar value (v0.d[0]) into both lanes of the SIMD register (v0.2d).

    - Performs a fused multiply-add operation (FMLA) on the accumulator register (v14.2d),
      multiplying the elements of v1.2d and v0.2d and adding the result to v14.2d.

    - Increments the loop counter (x25) and repeats the process until all remaining
      iterations are processed.
*/
simd2_remainder_loop:
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



// This section of the code performs the following:
// - Stores the computed results from SIMD register v14 (containing two double-precision floating-point values) 
//   into the memory location pointed to by x5, which corresponds to the matrix C at position [i][j, j+1].
// - Increments the loop index j by 2 (x24 is used as the loop index variable).
// - Branches back to the start of the inner loop (inner_loop_j) to process the next set of columns in the matrix.
end_simd2_loop_k:
    // Store results
    st1 {v14.2d}, [x5]             // C[i][j,j+1]

    add x24, x24, #2               // j += 2
    b inner_loop_j


// Function: standard_compute
// Description: This function performs a scalar computation path for matrix multiplication, 
//              processing a single column at a time. It initializes an accumulator to zero 
//              and sets up a loop counter for iterating through the elements of the matrix.
// 
// Instructions:
// - fmov d8, #0.0: Initializes the floating-point accumulator (d8) to 0.0.
// - mov x25, xzr: Sets the loop counter (x25) to zero, using xzr (zero register).
// - .align 7: Aligns the following code to a 128-byte boundary, optimizing for 
//             Apple Silicon cache line alignment.
standard_compute:
    // Scalar path (single column)
    fmov d8, #0.0                  // acc = 0

    mov x25, xzr                   // k = 0

    .align 7                       // 128-byte alignment for Apple Silicon cache lines


// standard_loop:
// This loop performs matrix multiplication using an unrolled approach for optimization.
// It begins by incrementing the loop counter (x25) and checking if the loop should continue
// based on the comparison with the upper bound (x22). If the counter exceeds the bound,
// it branches to the remainder loop (standard_remainder_loop).
//
// Unrolling 4 iterations of the inner loop:
// - For k=0, it calculates the address of the current matrix element using x19 (base address),
//   x28 (row offset), and x25 (column offset shifted by 3 for 64-bit elements).
// - Prefetching is used to optimize memory access. If the next iteration is within bounds,
//   it prefetches the next block of data into the L1 cache using the `prfm` instruction.
//
// Branching:
// - If the loop counter exceeds the bound after incrementing, it skips the prefetching step
//   and branches to `skip_pf_a2`.
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


/**
 * This section of assembly code performs a partial matrix multiplication
 * for floating-point values. It calculates the dot product of a row from
 * matrix A and a column from matrix B, accumulating the result in a 
 * floating-point register (d8). The loop processes four iterations (k=0 to k=3),
 * corresponding to four elements in the row and column.
 *
 * Key operations:
 * - Load elements from matrix A and matrix B into floating-point registers (d0 and d1).
 * - Perform fused multiply-add (fmadd) to accumulate the product of A[i][k] and B[k][j] into d8.
 * - Update pointers and indices for the next iteration.
 *
 * Registers used:
 * - x6: Pointer to the current element in matrix A's row.
 * - x8: Pointer to the current element in matrix B's column.
 * - x25: Index for the current column in matrix B.
 * - x27: Stride for accessing elements in matrix B.
 * - x20, x24: Base addresses and offsets for matrix B.
 * - d0, d1: Floating-point registers for elements of A and B.
 * - d8: Accumulator for the dot product.
 *
 * After processing four iterations, the code increments the column index (x25)
 * and branches back to the main loop (standard_loop) for further processing.
 */
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


// This loop handles the remaining iterations of the matrix multiplication
// when the main loop cannot process all elements due to a non-divisible size.
// 
// Registers used:
// - x25: Loop counter for the remaining iterations.
// - x22: Total number of iterations (limit).
// - x19: Base address of matrix A.
// - x20: Base address of matrix B.
// - x28: Offset for the current row in matrix A.
// - x27: Stride for accessing elements in matrix B.
// - x24: Offset for the current column in matrix B.
// - d8: Accumulator for the result of the dot product.
// 
// Instructions:
// 1. Compare the loop counter (x25) with the total iterations (x22).
//    If x25 >= x22, exit the loop.
// 2. Calculate the address of the current element in matrix A and load it into d0.
// 3. Calculate the address of the current element in matrix B and load it into d1.
// 4. Perform a fused multiply-add operation (d8 += d0 * d1).
// 5. Increment the loop counter (x25) and repeat the loop.
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



// End of the standard loop for matrix multiplication.
// The result of the computation is stored in memory at the address pointed to by x5.
// Increment the loop counter for the inner loop (j++).
// Branch back to the start of the inner loop to process the next iteration.
end_standard_loop:
    // Store result
    str d8, [x5]

    add x24, x24, #1               // j++
    b inner_loop_j


// Increment the loop counter `x23` (i++) to move to the next iteration
// of the inner loop and branch back to the outer loop label `outer_loop_i`.
end_inner_loop_j:
    add x23, x23, #1               // i++
    b outer_loop_i

/**
 * Cleans up the stack and returns from the function.
 *
 * This section of code restores the stack pointer and pops the saved 
 * registers from the stack in reverse order of their storage. It ensures 
 * that the function's execution context is properly restored before 
 * returning to the caller.
 *
 * Steps performed:
 * 1. Restores the stack pointer (SP) from the frame pointer (X29).
 * 2. Pops the saved floating-point registers (D8-D15) from the stack.
 * 3. Pops the saved general-purpose registers (X19-X30) from the stack.
 * 4. Returns to the caller using the `ret` instruction.
 */
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