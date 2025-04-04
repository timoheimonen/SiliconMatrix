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


// _matrix_multiply_int_asm:
// This function performs matrix multiplication for integer matrices.
// It assumes matrices A, B, and C are stored in row-major order.
// Parameters:
//   x0 (A): Pointer to the first matrix (NxN).
//   x1 (B): Pointer to the second matrix (NxN).
//   x2 (C): Pointer to the result matrix (NxN).
//   w3 (N): Dimension of the square matrices (N x N).
//
// Stack setup:
//   - Saves callee-saved registers (x19-x30) to the stack.
//   - Sets up the frame pointer (x29).
//
// Argument handling:
//   - Moves the input arguments (A, B, C, N) into callee-saved registers (x19-x22).
//
// Pre-computation:
//   - Calculates the row stride (N * 4 bytes) for efficient addressing of matrix rows.
//   - Initializes the loop counter (i = 0) for iterating over rows of matrix A.
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

/*
 * outer_loop_i:
 * This label marks the beginning of the outer loop for iterating over rows of matrix A.
 *
 * cmp x23, x22
 * - Compares the current row index (x23) with the total number of rows (N, stored in x22).
 *
 * b.ge end_outer_loop_i
 * - If the current row index (i) is greater than or equal to N, the loop ends by branching to end_outer_loop_i.
 *
 * mul x28, x23, x27
 * - Calculates the offset for the current row in matrix A by multiplying the row index (i) by the row stride (N * 4, stored in x27).
 * - The result is stored in x28 for faster access to the current row of matrix A.
 *
 * Prefetching:
 * add x10, x23, #1
 * - Calculates the next row index (i + 1) and stores it in x10.
 *
 * cmp x10, x22
 * - Compares the next row index (i + 1) with the total number of rows (N).
 *
 * b.ge skip_pf_a_row
 * - If the next row index (i + 1) is greater than or equal to N, skips the prefetching step by branching to skip_pf_a_row.
 *
 * mul x10, x10, x27
 * - Calculates the memory offset for the next row in matrix A by multiplying the next row index (i + 1) by the row stride (N * 4).
 *
 * add x10, x19, x10
 * - Adds the base address of matrix A (stored in x19) to the calculated offset to get the address of the next row.
 *
 * prfm pldl1keep, [x10]
 * - Prefetches the next row of matrix A into the L1 cache to optimize memory access for subsequent iterations.
 */
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


/**
 * skip_pf_a_row:
 * This label marks the beginning of a routine or section of code.
 * 
 * - `mov x24, xzr`: Initializes the column counter `j` to 0 by moving the value 
 *   of the zero register (`xzr`) into register `x24`. This is likely used to 
 *   iterate over columns in a matrix or similar data structure.
 */
skip_pf_a_row:
    mov     x24, xzr               // j = 0 (column counter)


// inner_loop_j:
// This label marks the beginning of the inner loop over the columns (j).
// The loop iterates through the columns of the result matrix C.
//
// cmp x24, x22
// Compares the current column index (j) stored in x24 with the total number of columns (N) stored in x22.
//
// b.ge end_inner_loop_j
// If j >= N, the loop ends and execution jumps to the end_inner_loop_j label.
//
// add x5, x21, x28
// add x5, x5, x24, lsl #2
// Calculates the memory address of the element C[i*N + j] in the result matrix.
// - x21: Base address of matrix C.
// - x28: Offset for the current row (i*N).
// - x24: Current column index (j), scaled by 4 (size of int32).
//
// add x10, x24, #3
// cmp x10, x22
// b.ge standard_compute
// Checks if there are at least 4 columns remaining from the current column index (j).
// If not, jumps to the scalar computation path (standard_compute).
//
// movi v14.4s, #0
// Initializes a SIMD vector register (v14) with zeros to accumulate results for 4 integers at once.
//
// mov x25, xzr
// Initializes the depth counter (k) to 0. This counter is used for iterating over the depth dimension.
//
// .align 7
// Aligns the following instructions to a 128-byte boundary for optimal cache performance on Apple Silicon.
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



// simd4_loop_k:
// This label marks the start of a loop that processes 4 iterations of the k-dimension
// in a matrix multiplication operation using SIMD instructions.
//
// - The loop first checks if there are at least 4 remaining k iterations to process.
//   If not, it branches to the `simd4_remainder_loop` to handle the remaining iterations.
//
// - If unrolling is possible:
//   - It calculates the address of the current element in matrix A for the given i and k indices.
//   - Loads the value of A[i][k] into register w0.
//
// - Prefetching:
//   - Prefetches the next block of matrix A into the L1 cache to optimize memory access.
//   - Skips prefetching if the remaining k iterations are less than 8.
//
// Registers used:
// - x10: Temporary register for address and bounds checking.
// - x6:  Holds the address of the current element in matrix A.
// - x25: Current k index.
// - x22: Upper bound of k.
// - x19: Base address of matrix A.
// - x28: Offset for the i-th row of matrix A.
//
// Branches:
// - `simd4_remainder_loop`: Handles the remaining k iterations when unrolling is not possible.
// - `skip_pf_a1`: Skips prefetching if the remaining k iterations are insufficient.
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


// This section of the code performs the following operations:
// 
// 1. Calculates the memory address for an element in matrix B:
//    - `mul x11, x25, x27`: Multiplies `k` (x25) by `N*4` (x27) to compute the offset for row `k`.
//    - `add x8, x20, x11`: Adds the base address of matrix B (x20) to the row offset.
//    - `add x8, x8, x24, lsl #2`: Adds the column offset (`j*4`, where `j` is in x24) to get the address of element `B[k][j]`.
//
// 2. Prefetches the next row of matrix B into the L1 cache:
//    - `add x10, x25, #4`: Calculates `k + 4` to check the next row.
//    - `cmp x10, x22`: Compares `k + 4` with the total number of rows (x22).
//    - `b.ge skip_pf_b1`: Skips prefetching if `k + 4` exceeds the number of rows.
//    - `add x12, x25, #4`: Calculates the row index for `k + 4`.
//    - `mul x12, x12, x27`: Computes the offset for row `k + 4`.
//    - `add x12, x20, x12`: Adds the base address of matrix B to the row offset.
//    - `add x12, x12, x24, lsl #2`: Adds the column offset (`j*4`) to get the address of element `B[k+4][j]`.
//    - `prfm pldl1keep, [x12]`: Prefetches the calculated address into the L1 cache to optimize memory access.
//
// Note: The prefetching step is conditional and only executed if the next row index (`k + 4`) is within bounds.
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



/**
 * This section of assembly code performs a portion of a matrix multiplication
 * operation using SIMD (Single Instruction, Multiple Data) instructions. It
 * computes the dot product of a row from matrix A and a column from matrix B,
 * accumulating the results into a vector register (v14).
 *
 * Key operations:
 * - Loads elements of matrix B into SIMD registers (v1.4s) for four consecutive
 *   columns at a time.
 * - Broadcasts a single element from matrix A (A[i][k]) into all lanes of a SIMD
 *   register (v0.4s).
 * - Performs Multiply-Accumulate (mla) operations to compute partial sums for
 *   the dot product.
 *
 * Loop structure:
 * - Iterates over four consecutive elements of matrix A (A[i][k], A[i][k+1],
 *   A[i][k+2], A[i][k+3]) and their corresponding rows in matrix B.
 * - Updates pointers to access the next elements of matrix A and matrix B.
 *
 * Registers used:
 * - x6: Pointer to the current element of matrix A.
 * - x8: Pointer to the current row of matrix B.
 * - x11: Offset for accessing rows of matrix B.
 * - x20, x24, x27: Stride values for navigating matrix B.
 * - v0.4s: SIMD register holding broadcasted elements of matrix A.
 * - v1.4s: SIMD register holding elements of matrix B.
 * - v14.4s: Accumulator register for the dot product.
 *
 * Notes:
 * - The code assumes matrices are stored in row-major order.
 * - The loop processes four columns of matrix B at a time.
 * - The final result of the dot product is accumulated in v14.4s.
 */
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



/**
 * simd4_remainder_loop:
 * This loop handles the remaining iterations of the matrix multiplication
 * for cases where the number of iterations (k) is not a multiple of 4.
 * 
 * - Compares the current iteration index (x25) with the total iterations (x22).
 * - If all iterations are processed, it branches to `end_simd4_loop_k`.
 * - For each remaining iteration:
 *   1. Computes the address of the current element A[i][k] in matrix A.
 *   2. Loads the value of A[i][k] into register w0.
 *   3. Computes the address of the current row of matrix B (B[k][j, j+1, j+2, j+3]).
 *   4. Loads the 4-element vector from matrix B into SIMD register v1.
 *   5. Broadcasts the scalar value A[i][k] into all lanes of SIMD register v0.
 *   6. Performs a fused multiply-add operation (mla) to accumulate the product
 *      of A[i][k] and B[k][j, j+1, j+2, j+3] into SIMD register v14.
 *   7. Increments the iteration index (x25) and loops back to process the next k.
 */
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

// End of the SIMD4 loop for the k dimension in matrix multiplication.
// Stores the computed results from vector register v14 (containing 4 integer results)
// into the memory location pointed to by x5, which corresponds to C[i][j, j+1, j+2, j+3].
// Increments the column index j by 4 (x24 += 4) to process the next set of columns.
// Branches back to the inner loop for the j dimension to continue processing.
end_simd4_loop_k:
    // Store results
    st1     {v14.4s}, [x5]         // C[i][j,j+1,j+2,j+3]

    add     x24, x24, #4           // j += 4
    b       inner_loop_j

// This function, `standard_compute`, implements a scalar path for matrix multiplication,
// processing a single column at a time. It initializes the sum accumulator (`w26`) to 0
// and the loop counter (`x25`) to 0. The `.align 7` directive ensures 128-byte alignment
// for optimal performance on Apple Silicon processors, as this matches the cache line size.
standard_compute:
    // Scalar path (single column)
    mov     w26, #0                // sum = 0
    mov     x25, xzr               // k = 0

    .align  7                      // 128-byte alignment for Apple Silicon cache lines


/*
 * standard_loop:
 * This label marks the beginning of the standard loop for matrix multiplication.
 * 
 * - Adds 3 to the value in x25 and stores the result in x10.
 * - Compares x10 with x22 to determine if the loop should continue or branch to the remainder loop.
 * 
 * Unroll 4 k iterations:
 * - k=0:
 *   - Computes the address for the current element in matrix A by adding x19, x28, and x25 shifted left by 2.
 *   - Stores the result in x6.
 * 
 * Prefetch A:
 * - Checks if prefetching is necessary by adding 8 to x25 and comparing the result with x22.
 * - If the condition is met, skips the prefetch operation.
 * - Otherwise, computes the address for prefetching by adding 32 to x6 and issues a prefetch instruction (prfm) to load data into L1 cache.
 */
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

    
/**
 * This assembly code performs a partial matrix multiplication for a 4x4 block
 * of matrices A and B, accumulating the result into a scalar sum.
 *
 * Registers:
 * - x6: Pointer to the current element in matrix A.
 * - x20: Base address of matrix B.
 * - x24: Column index (j) for matrix B.
 * - x25: Row index (i) for matrix A.
 * - x26: Accumulator for the sum of products.
 * - x27: Row stride for matrix B (N * 4, where N is the number of columns in B).
 * - x8: Temporary pointer for accessing elements in matrix B.
 * - x11: Temporary variable for calculating offsets in matrix B.
 * - w0: Temporary variable for loading elements from matrix A.
 * - w1: Temporary variable for loading elements from matrix B and storing intermediate products.
 *
 * The code processes four iterations (k = 0, 1, 2, 3) of the inner loop of
 * matrix multiplication, where:
 *   sum += A[i][k] * B[k][j]
 *
 * For each iteration:
 * 1. Load the current element of A (A[i][k]) into w0.
 * 2. Compute the address of the corresponding element in B (B[k][j]) using
 *    the base address, row stride, and column index, and load it into w1.
 * 3. Multiply A[i][k] and B[k][j], and accumulate the result into w26.
 * 4. Update pointers and offsets for the next iteration.
 *
 * After processing the 4x4 block, the row index (x25) is incremented, and
 * control is transferred back to the main loop (standard_loop).
 */
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

// This assembly code implements a loop to handle the remaining iterations
// of a matrix multiplication operation for the case where the number of
// iterations (k) is not evenly divisible by the vectorized loop size.
//
// standard_remainder_loop:
// - Compares the current iteration index (x25) with the total number of
//   iterations (x22). If x25 >= x22, the loop ends and branches to
//   `end_standard_loop`.
// - For each iteration:
//   1. Computes the address of the current element A[i][k] in matrix A
//      using base address (x19), row offset (x28), and column offset (x25).
//   2. Loads the value of A[i][k] into register w0.
//   3. Computes the address of the current element B[k][j] in matrix B
//      using base address (x20), row offset (k * N*4), and column offset (x24).
//   4. Loads the value of B[k][j] into register w1.
//   5. Multiplies A[i][k] (w0) and B[k][j] (w1), storing the result in w1.
//   6. Accumulates the product into the running sum (w26).
//   7. Increments the iteration index (x25) by 1.
// - The loop repeats until all remaining iterations are processed.
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

// This section of the code performs the following:
// - Stores the result of a computation (stored in register w26) into memory at the address pointed to by x5.
// - Increments the loop counter for the inner loop (j) by 1, using register x24.
// - Branches back to the start of the inner loop labeled "inner_loop_j" to continue the iteration.
end_standard_loop:
    // Store result
    str     w26, [x5]

    add     x24, x24, #1           // j++
    b       inner_loop_j

// This section increments the loop counter `i` (stored in register x23) by 1
// and then branches back to the start of the outer loop labeled `outer_loop_i`.
// It effectively ends the inner loop and resumes execution of the outer loop.
end_inner_loop_j:
    add     x23, x23, #1           // i++
    b       outer_loop_i

// This section of the code restores the callee-saved registers and returns from the function.
// 
// - The stack pointer (sp) is restored to the value stored in x29.
// - The callee-saved registers (x19 to x30) are restored from the stack in reverse order of their storage.
// - Each `ldp` instruction loads two registers from the stack and increments the stack pointer by 16 bytes.
// - Finally, the `ret` instruction is executed to return to the caller.
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
