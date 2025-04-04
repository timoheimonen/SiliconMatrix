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
// ARM64 assembly implementation of matrix multiplication for 64-bit integers
// Function: void matrix_multiply_int64_asm(const int64_t* A, const int64_t* B, int64_t* C, int N)
// 
// Parameters:
// x0 - pointer to matrix A
// x1 - pointer to matrix B
// x2 - pointer to result matrix C
// w3 - size of matrices (N)

.global _matrix_multiply_int64_asm
.align 4


// _matrix_multiply_int64_asm:
// This function performs matrix multiplication for 64-bit integer matrices.
// Arguments:
// - x0 (A): Pointer to the first input matrix.
// - x1 (B): Pointer to the second input matrix.
// - x2 (C): Pointer to the output matrix.
// - w3 (N): Dimension of the square matrices (N x N).
// Functionality:
// - Sets up the stack and saves callee-saved registers.
// - Moves input arguments into callee-saved registers.
// - Computes the row stride (N * 8) for addressing 64-bit integers.
// - Checks if N is greater than 200 to branch into a large‑matrix path.
_matrix_multiply_int64_asm:
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

    // N*8 for faster addressing (64-bit integers = 8 bytes)
    lsl     x27, x22, #3           // row stride = N*8
    
    // For small matrices, fall back to C++ equivalent for 64-bit integers
    cmp     w22, #200
    b.gt    large_matrix_path
    
    // For small matrices, try to replicate what the C++ compiler likely does
    // Skip the matrix zeroing phase completely - initialize C[i][j] in the inner loop
    
    // Simple i-k-j loop order (known to be efficient for small matrices)
    mov     x23, xzr               // i = 0
    

// opt_i_loop:
// // Iterates over rows of matrix A. Calculates base addresses for A[i] and C[i].
opt_i_loop:
    cmp     x23, x22
    b.ge    opt_i_done
    
    // Calculate A[i][0] address (reuse for each k)
    mul     x28, x23, x27          // i * N * 8
    add     x9, x19, x28           // &A[i][0]
    
    // Calculate C[i][0] address (reuse for each j)
    add     x10, x21, x28          // &C[i][0]
    
    // Initialize C[i][j] values to zero first
    mov     x24, xzr               // j = 0
    

// opt_zero_j:
// // Iterates over all columns in row i to zero out the corresponding element in C.
opt_zero_j:
    cmp     x24, x22
    b.ge    opt_zero_done
    
    add     x11, x10, x24, lsl #3  // &C[i][j]
    str     xzr, [x11]             // C[i][j] = 0
    
    add     x24, x24, #1
    b       opt_zero_j
    

// opt_zero_done:
// This section of the code initializes the loop for processing all `k` values,
// optimizing for better cache locality when accessing matrix A. The `x25` 
// register is set to zero (`xzr`), which serves as the initial value for the 
// loop variable `k`.
opt_zero_done:
    // Process all k values first (better cache locality for A)
    mov     x25, xzr               // k = 0
    

// opt_k_loop:
// This loop iterates over the "k" dimension of the matrix multiplication.
// 
// - Compares the current "k" index (x25) with the upper bound (x22) to determine if the loop is done.
// - Loads the value of A[i][k] into register x0 for use in calculations across all "j" values.
// - Optimizes for the case where A[i][k] is zero, skipping unnecessary computations.
// - Calculates the base address of B[k][0] to reuse it for all "j" iterations.
// - Initializes the "j" index (x24) to zero for the inner loop that processes all columns.
opt_k_loop:
    cmp     x25, x22
    b.ge    opt_k_done
    
    // Load A[i][k] once for all j values
    add     x13, x9, x25, lsl #3   // &A[i][k]
    ldr     x0, [x13]              // A[i][k]
    
    // Optimize for the case where A[i][k] is zero (common in some matrices)
    cbz     x0, opt_k_next
    
    // Calculate B[k][0] address (reuse for each j)
    mul     x14, x25, x27          // k * N * 8
    add     x14, x20, x14          // &B[k][0]
    
    // Inner j loop processes all columns
    mov     x24, xzr               // j = 0
    

// opt_j_loop:
// This loop performs the matrix multiplication operation for a specific row and column.
// It iterates over the elements of the matrix B for a given column (j) and performs
// the multiply-accumulate operation to update the corresponding element in matrix C.
// Instructions:
// 1. Compare the current column index (x24) with the total number of columns (x22).
//    If x24 >= x22, the loop is terminated (branch to opt_j_done).
// 2. Calculate the address of the current element B[k][j] using the base address of B (x14)
//    and the column index (x24). Load the value of B[k][j] into register x2.
// 3. If B[k][j] is zero, skip the computation for this iteration (branch to opt_j_next).
// 4. Calculate the address of the current element C[i][j] using the base address of C (x10)
//    and the column index (x24). Load the current value of C[i][j] into register x4.
// 5. Perform the multiply-accumulate operation:
//    C[i][j] += A[i][k] * B[k][j]
//    - A[i][k] is stored in register x0.
//    - B[k][j] is stored in register x2.
//    - The result is accumulated in register x4.
// 6. Store the updated value of C[i][j] back to memory at its calculated address.
opt_j_loop:
    cmp     x24, x22
    b.ge    opt_j_done
    
    // Load B[k][j]
    add     x15, x14, x24, lsl #3  // &B[k][j]
    ldr     x2, [x15]              // B[k][j]
    
    // Skip computation if B[k][j] is zero
    cbz     x2, opt_j_next
    
    // Load current C[i][j] value
    add     x11, x10, x24, lsl #3  // &C[i][j]
    ldr     x4, [x11]              // Current C[i][j]
    
    // Multiply-accumulate
    madd    x4, x0, x2, x4         // C[i][j] += A[i][k] * B[k][j]
    
    // Store result back
    str     x4, [x11]
    

// opt_j_next:
// This code snippet increments the value in register x24 by 1 and then branches
// unconditionally to the label `opt_j_loop`. It is part of a loop structure,
// where `x24` likely serves as a loop counter for iterating over a specific range.
opt_j_next:
    add     x24, x24, #1
    b       opt_j_loop
    
// opt_j_done:
// Label: opt_j_done
// Description: This label marks the end of a specific operation or loop in the assembly code.
// It is likely used as a jump target to indicate the completion of a process or iteration.
opt_j_done:
    
// opt_k_next:
// This code snippet increments the value in register x25 by 1 and then branches
// unconditionally to the label `opt_k_loop`. It is likely part of a loop structure
// where x25 serves as a loop counter or an index variable.
opt_k_next:
    add     x25, x25, #1
    b       opt_k_loop
    
// opt_k_done:
// This section of the code increments the value in register x23 by 1,
// which is likely being used as a loop counter or index for the "k" loop.
// After incrementing, it branches back to the label "opt_i_loop" to continue
// execution of the outer loop or next iteration of the process.
opt_k_done:
    add     x23, x23, #1
    b       opt_i_loop
    
// opt_i_done:
// Branch to the label `end_outer_loop_i` to exit the outer loop for the `i` index.
// This is typically used to terminate the current iteration of the outer loop
// and proceed with the next steps in the program flow.
opt_i_done:
    b       end_outer_loop_i

// large_matrix_path:
// // For larger matrices, first zeroes out the entire C matrix, then proceeds.
large_matrix_path:
    // Original implementation for larger matrices
    // Use your existing implementation for larger matrices
    // Zero out the C matrix
    mov     x10, x21               // C pointer
    mul     x11, x27, x22          // Total size in bytes
    add     x11, x10, x11          // End of C
    
// zero_loop:
// This loop is used to zero out a memory region by writing zero values (xzr) 
// in 16-byte chunks. The loop continues until the address in x10 reaches or 
// exceeds the address in x11.
// Instructions:
// - `cmp x10, x11`: Compares the current address in x10 with the end address in x11.
// - `b.ge zero_done`: Branches to the `zero_done` label if x10 is greater than or equal to x11.
// - `stp xzr, xzr, [x10], #16`: Stores two zero registers (xzr) into memory at the address in x10 
//   and increments x10 by 16 bytes.
// - `b zero_loop`: Branches back to the start of the loop to continue zeroing the memory.
zero_loop:
    cmp     x10, x11
    b.ge    zero_done
    stp     xzr, xzr, [x10], #16   // Zero 16 bytes at a time when possible
    b       zero_loop
    
// zero_done:
// This section of the code checks if the matrix size (stored in w22) is less than 256.
// If the size is less than 256, it branches to the `small_matrix_path` label to handle
// small matrix multiplication. Otherwise, it continues to the `regular_implementation`
// label for the standard large matrix multiplication logic.
zero_done:
    // Continue with large matrix implementation
    cmp     w22, #256
    b.lt    small_matrix_path
    b       regular_implementation


// tiny_matrix_path:
// This section of the code is responsible for initializing the matrix multiplication process
// for small matrices using a classic i-k-j loop order. The goal is to minimize address
// calculation overhead. The steps include:
// 1. Setting up the pointer to the result matrix (C) in register x10.
// 2. Calculating the total size of the result matrix (C) in bytes by multiplying the
//    number of rows (x27) by the number of columns (x22), storing the result in x11.
// 3. Adding the base address of C (x10) to the total size (x11) to determine the end
//    address of the result matrix (C).
// This setup ensures that the result matrix (C) can be efficiently zeroed out before
// performing the matrix multiplication.
tiny_matrix_path:
    // For tiny matrices, use classic i-k-j loop order with minimized address calculation
    // Zero out C first
    mov     x10, x21               // C pointer
    mul     x11, x27, x22          // Total size in bytes
    add     x11, x10, x11          // End of C
    
// tiny_zero_loop:
// This loop initializes a block of memory to zero by storing pairs of zero registers (xzr) 
// at the address pointed to by x10, incrementing the pointer by 16 bytes after each iteration.
// The loop continues until x10 is greater than or equal to x11.
// - `cmp x10, x11`: Compares the current address in x10 with the end address in x11.
// - `b.ge tiny_zero_done`: Branches to `tiny_zero_done` if x10 >= x11, ending the loop.
// - `stp xzr, xzr, [x10], #16`: Stores two zero registers (xzr) at the address in x10 
//   and increments x10 by 16 bytes.
// - `b tiny_zero_loop`: Branches back to the start of the loop for the next iteration.
tiny_zero_loop:
    cmp     x10, x11
    b.ge    tiny_zero_done
    stp     xzr, xzr, [x10], #16
    b       tiny_zero_loop
    
// tiny_zero_done:
// This label `tiny_zero_done` marks the beginning of a section of code.
// The instruction `mov x23, xzr` initializes the register `x23` to zero.
// Here, `x23` is being used as a row counter (`i = 0`), and `xzr` is the zero register in ARM assembly.
tiny_zero_done:
    mov     x23, xzr               // i = 0 (row counter)
    
// tiny_loop_i:
// This section of the code implements a loop to process rows of matrices A and C.
// - `tiny_loop_i` is the label for the loop that iterates over rows of the matrices.
// - Compares the current row index `x23` with the total number of rows `x22`.
// - If `x23 >= x22`, the loop exits and branches to `tiny_done`.
// - Calculates the address of the first element in row `i` of matrix A (`&A[i][0]`):
//   - `x28` stores the offset for row `i` (i * N * 8, where N is the number of columns).
//   - `x9` stores the base address of `&A[i][0]`.
// - Calculates the address of the first element in row `i` of matrix C (`&C[i][0]`):
//   - `x10` stores the base address of `&C[i][0]`.
// - Initializes the column index `k` to 0 by setting `x25` to zero.
tiny_loop_i:
    cmp     x23, x22
    b.ge    tiny_done
    
    // Calculate A[i][0] address once per row
    mul     x28, x23, x27          // i * N * 8
    add     x9, x19, x28           // &A[i][0]
    
    // Calculate C[i][0] address once per row
    add     x10, x21, x28          // &C[i][0]
    
    mov     x25, xzr               // k = 0
    
// tiny_loop_k:
// This loop processes the k-dimension of a matrix multiplication for a fixed i and all j values.
// - Compares the current k index (x25) with the upper bound (x22). If k >= x22, the loop ends.
// - Loads the value of A[i][k] into x0. This value is used for all j values in the current iteration.
//   - The address of A[i][k] is calculated as &A[i][k] = x9 + (k << 3).
// - Computes the base address of B[k][0] for the current k iteration.
//   - The address is calculated as &B[k][0] = x20 + (k * N * 8), where N is the number of columns in B.
// - Initializes the j index (x24) to 0, preparing for processing all j values with the current A[i][k].
tiny_loop_k:
    cmp     x25, x22
    b.ge    tiny_loop_k_end
    
    // Load A[i][k] - only once for all j values
    add     x13, x9, x25, lsl #3   // &A[i][k]
    ldr     x0, [x13]              // A[i][k]
    
    // Calculate B[k][0] address once per k
    mul     x14, x25, x27          // k * N * 8
    add     x14, x20, x14          // &B[k][0]
    
    // Now process all j values with this A[i][k] and B[k][j]
    mov     x24, xzr               // j = 0
    
// tiny_loop_j:
// This loop performs the inner-most operation of a matrix multiplication algorithm,
// specifically iterating over the columns of the result matrix (indexed by `j`).
// Registers:
// - x24: Current column index `j`.
// - x22: Total number of columns (loop termination condition).
// - x10: Base address of the result matrix `C`.
// - x14: Base address of the matrix `B`.
// - x0: Current element of matrix `A` (A[i][k]).
// Operations:
// 1. Compare the current column index `j` (x24) with the total number of columns (x22).
//    If `j >= total_columns`, exit the loop.
// 2. Calculate the address of the current element `C[i][j]` in the result matrix.
// 3. Calculate the address of the current element `B[k][j]` in matrix `B`.
// 4. Load the value of `B[k][j]` into register x2.
// 5. Load the current value of `C[i][j]` into register x4.
// 6. Perform the multiply-accumulate operation: `C[i][j] += A[i][k] * B[k][j]`.
// 7. Store the updated value of `C[i][j]` back to memory.
// 8. Increment the column index `j` (x24) and repeat the loop.
tiny_loop_j:
    cmp     x24, x22
    b.ge    tiny_loop_j_end
    
    // Calculate addresses and load
    add     x11, x10, x24, lsl #3  // &C[i][j]
    add     x15, x14, x24, lsl #3  // &B[k][j]
    
    ldr     x2, [x15]              // B[k][j]
    ldr     x4, [x11]              // Current C[i][j] value
    
    // Multiply-accumulate
    madd    x4, x0, x2, x4         // C[i][j] += A[i][k] * B[k][j]
    
    // Store result
    str     x4, [x11]              // C[i][j]
    
    add     x24, x24, #1           // j++
    b       tiny_loop_j
    
// tiny_loop_j_end:
// This section of the code increments the loop counter `k` (stored in register x25)
// by 1 and then branches back to the label `tiny_loop_k` to continue the loop.
// The label `tiny_loop_j_end` marks the end of the current iteration of the loop.
tiny_loop_j_end:
    add     x25, x25, #1           // k++
    b       tiny_loop_k
    
// tiny_loop_k_end:
// This section of the code increments the loop counter `x23` by 1 (i++) 
// and then branches back to the label `tiny_loop_i` to continue the loop.
// It is likely part of a nested loop structure where `x23` serves as the 
// loop variable for the outer loop.
tiny_loop_k_end:
    add     x23, x23, #1           // i++
    b       tiny_loop_i
    
// tiny_done:
// Branch to the label `end_outer_loop_i` to exit the current loop.
// This is typically used to terminate the outer loop in a matrix multiplication routine.
// The label `tiny_done` serves as a marker for this branching operation.
tiny_done:
    b       end_outer_loop_i

// small_matrix_path:
// This is the entry point for the regular implementation of matrix multiplication.
// The implementation begins with a standard i-j-k nested loop structure, where:
// - `i` represents the row index of the first matrix.
// - `j` represents the column index of the second matrix.
// - `k` is the shared dimension index for the dot product computation.
// The inner loop is optimized for performance.
// The instruction `mov x23, xzr` initializes the row counter `i` to 0.
small_matrix_path:
    // For small matrices, use a different loop order (j-i-k) for better locality
    // This helps because entire matrices likely fit in L1/L2 cache
    
    // First zero out C (helps with alignment and simplifies the inner loop)
    mov     x10, x21
    mul     x11, x27, x22
    add     x11, x10, x11
    
// small_zero_loop:
// This loop initializes a block of memory to zero in pairs of 64-bit integers.
// Registers:
// - x10: Pointer to the current memory location being zeroed.
// - x11: Pointer to the end of the memory region to be zeroed.
// Instructions:
// 1. Compare the current pointer (x10) with the end pointer (x11).
// 2. If x10 >= x11, branch to `small_zero_done` to exit the loop.
// 3. Store two zeroed 64-bit integers (xzr, xzr) at the memory location pointed to by x10.
// 4. Increment x10 by 16 bytes (two 64-bit integers).
// 5. Repeat the loop until the entire memory region is zeroed.
small_zero_loop:
    cmp     x10, x11
    b.ge    small_zero_done
    stp     xzr, xzr, [x10], #16
    b       small_zero_loop
    
// small_zero_done:
// Label: small_zero_done
// This label marks the beginning of a section of code where the column counter is initialized.
// Instruction: mov x24, xzr
// Description: Sets the value of register x24 to zero by moving the zero register (xzr) into x24.
// Purpose: Initializes the column counter (j) to 0, likely as part of a loop or matrix operation.
small_zero_done:
    mov     x24, xzr               // j = 0 (column counter)
    
// small_loop_j:
// This label marks the beginning of a loop that iterates over columns (j).
// The loop compares the current column index (x24) with the total number of columns (x22).
// If the current column index is greater than or equal to the total, the loop exits (branch to small_done).
// Otherwise, it initializes the row counter (x23) to zero (i = 0) for processing rows in the current column.
small_loop_j:
    cmp     x24, x22
    b.ge    small_done
    
    mov     x23, xzr               // i = 0 (row counter)
    
// small_loop_i:
// This loop iterates over the rows of matrix C, indexed by `i`.
// Registers used:
// - x23: Current row index `i`.
// - x22: Total number of rows (loop termination condition).
// - x27: N * 8 (row size in bytes for matrix C).
// - x21: Base address of matrix C.
// - x24: Current column index `j`.
// - x10: Temporary register to hold the address of the start of row `i` in matrix C.
// - x11: Temporary register to hold the address of element C[i][j].
// - x4: Register to load the current value of C[i][j].
// - x25: Register to initialize the inner loop index `k` to 0.
// Instructions:
// - Compare the current row index `i` (x23) with the total number of rows (x22).
// - If `i` >= total rows, branch to `small_loop_i_end` to exit the loop.
// - Compute the address of the element C[i][j] in matrix C:
//   - Multiply `i` by the row size (N * 8) to get the byte offset for row `i`.
//   - Add the base address of matrix C to the row offset to get the start of row `i`.
//   - Add the column offset (`j` shifted left by 3 to multiply by 8) to get the address of C[i][j].
// - Load the current value of C[i][j] into x4 (assumes it has been initialized to zero).
// - Initialize the inner loop index `k` to 0 (stored in x25).
small_loop_i:
    cmp     x23, x22
    b.ge    small_loop_i_end
    
    // Compute C[i][j] position
    mul     x28, x23, x27          // i * N * 8
    add     x10, x21, x28          // &C[i][0]
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Load current C value (already zeroed)
    ldr     x4, [x11]
    
    // Process all k values 
    mov     x25, xzr               // k = 0
    
// small_loop_k:
// This assembly code performs a matrix multiplication operation for 64-bit integers
// (int64_t) in a nested loop structure. The code is part of an inner loop that
// calculates the value of a single element in the resulting matrix C[i][j].
// Registers used:
// - x19: Base address of matrix A.
// - x20: Base address of matrix B.
// - x11: Address to store the result in matrix C.
// - x23: Row index (i) for matrix A and matrix C.
// - x24: Column index (j) for matrix B and matrix C.
// - x25: Loop counter for the inner loop (k).
// - x22: Upper bound for the inner loop (number of columns in A / rows in B).
// - x27: Scaling factor (N * 8, where N is the number of columns in A / rows in B).
// - x28, x9, x13: Temporary registers for calculating addresses in matrix A.
// - x14, x15: Temporary registers for calculating addresses in matrix B.
// - x0: Temporary register to hold A[i][k].
// - x2: Temporary register to hold B[k][j].
// - x4: Accumulator register for the result of C[i][j].
// Code functionality:
// 1. Calculate the address of A[i][k] and load its value into x0.
// 2. Calculate the address of B[k][j] and load its value into x2.
// 3. Perform a multiply-accumulate operation: C[i][j] += A[i][k] * B[k][j].
// 4. Increment the loop counter (k) and check if the loop should continue.
// 5. Once the loop completes, store the result of C[i][j] in memory.
// 6. Increment the row index (i) and branch to the outer loop for the next iteration.
small_loop_k:
    // A[i][k]
    mul     x28, x23, x27          // i * N * 8 
    add     x9, x19, x28           // &A[i][0]
    add     x13, x9, x25, lsl #3   // &A[i][k]
    ldr     x0, [x13]              // A[i][k]
    
    // B[k][j]
    mul     x14, x25, x27          // k * N * 8
    add     x14, x20, x14          // &B[k][0]
    add     x15, x14, x24, lsl #3  // &B[k][j]
    ldr     x2, [x15]              // B[k][j]
    
    // Multiply-accumulate
    madd    x4, x0, x2, x4         // C[i][j] += A[i][k] * B[k][j]
    
    add     x25, x25, #1           // k++
    cmp     x25, x22
    b.lt    small_loop_k
    
    // Store result
    str     x4, [x11]              // C[i][j]
    
    add     x23, x23, #1           // i++
    b       small_loop_i
    
// small_loop_i_end:
// Increment the loop counter for the inner loop (j++).
// Branch back to the start of the inner loop (small_loop_j) to continue iteration.
small_loop_i_end:
    add     x24, x24, #1           // j++
    b       small_loop_j
    
// small_done:
// Branch to the label `end_outer_loop_i` to exit the current loop or 
// complete the operation. This marks the end of the `small_done` 
// section, which likely handles a specific case or condition in the 
// matrix multiplication process.
small_done:
    b       end_outer_loop_i

// regular_implementation:
// This is the entry point for the regular implementation of matrix multiplication.
// The implementation begins with a standard i-j-k nested loop structure, where:
// - `i` represents the row index of the first matrix.
// - `j` represents the column index of the second matrix.
// - `k` is the shared dimension index for the dot product computation.
// The inner loop is optimized for performance.
// The instruction `mov x23, xzr` initializes the row counter `i` to 0.
regular_implementation:
    // Start with i-j-k loops but optimize inner loop more aggressively
    mov     x23, xzr               // i = 0 (row counter)

// outer_loop_i:
// // Outer loop for the regular implementation (i-j-k). 
// // Calculates A[i] and C[i] addresses, prefetches A row into cache.
outer_loop_i:
    cmp     x23, x22
    b.ge    end_outer_loop_i       // if i >= N, end loop

    // i*N*8 for faster A and C row access
    mul     x28, x23, x27          // i * N * 8
    add     x9, x19, x28           // &A[i][0] 
    add     x10, x21, x28          // &C[i][0]
    
    // Prefetch A row
    prfm    pldl1keep, [x9]
    prfm    pldl1keep, [x9, #64]
    prfm    pldl1keep, [x9, #128]

    // Process the entire row with unrolled j-loop where possible
    mov     x24, xzr               // j = 0 (column counter)
    
    // Check if we can process at least 4 columns at once
    add     x16, x24, #3           // j+3
    cmp     x16, x22
    b.ge    check_unroll2          // Less than 4 columns, try unroll2
    
// inner_loop_j_unroll4:
// // Unrolled inner loop processing 4 columns at a time. Accumulators are initialized here.
inner_loop_j_unroll4:
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Initialize accumulators for this row
    mov     x4, xzr                // C[i][j] accumulator
    mov     x5, xzr                // C[i][j+1] accumulator
    mov     x6, xzr                // C[i][j+2] accumulator
    mov     x7, xzr                // C[i][j+3] accumulator
    
    // Process 4 columns at once through all k values
    mov     x25, xzr               // k = 0
    
// inner_loop_k_unroll4:
// // Processes 4 iterations of k in the unrolled inner loop and prefetches next B row.
inner_loop_k_unroll4:
    // Process A element
    add     x13, x9, x25, lsl #3   // &A[i][k]
    ldr     x0, [x13]              // A[i][k]
    
    // Set up B row
    mul     x14, x25, x27          // k * N * 8
    add     x14, x20, x14          // &B[k][0]
    add     x15, x14, x24, lsl #3  // &B[k][j]
    
    // Prefetch next B row if helpful
    add     x17, x25, #8
    cmp     x17, x22
    b.ge    skip_pf_b_unroll4
    mul     x17, x17, x27         
    add     x17, x20, x17         
    add     x17, x17, x24, lsl #3
    prfm    pldl1keep, [x17]


// skip_pf_b_unroll4:
// // Increments k after unrolled loop, then stores the accumulated results back to C.
skip_pf_b_unroll4:
    
    // Load and process B elements for 4 columns
    ldp     x1, x2, [x15]          // B[k][j], B[k][j+1]
    ldp     x3, x8, [x15, #16]     // B[k][j+2], B[k][j+3]
    
    // Multiply-accumulate for all 4 elements
    madd    x4, x0, x1, x4         // C[i][j] += A[i][k] * B[k][j]
    madd    x5, x0, x2, x5         // C[i][j+1] += A[i][k] * B[k][j+1]
    madd    x6, x0, x3, x6         // C[i][j+2] += A[i][k] * B[k][j+2]
    madd    x7, x0, x8, x7         // C[i][j+3] += A[i][k] * B[k][j+3]
    
    // Prefetch next A element if beneficial
    add     x17, x25, #16
    cmp     x17, x22
    b.ge    skip_pf_a_unroll4
    add     x17, x9, x17, lsl #3
    prfm    pldl1keep, [x17]

    
// skip_pf_a_unroll4:
// // Increments k after unrolled loop, then stores the accumulated results back to C.
skip_pf_a_unroll4:
    
    add     x25, x25, #1           // k++
    cmp     x25, x22
    b.lt    inner_loop_k_unroll4
    
    // Store results back to C
    stp     x4, x5, [x11]          // C[i][j], C[i][j+1]
    stp     x6, x7, [x11, #16]     // C[i][j+2], C[i][j+3]
    
    add     x24, x24, #4           // j += 4
    add     x16, x24, #3           // j+3
    cmp     x16, x22
    b.lt    inner_loop_j_unroll4
    


// check_unroll2:
// // Checks if at least 2 columns are available to process together.
check_unroll2:
    add     x16, x24, #1           // j+1
    cmp     x16, x22
    b.ge    inner_loop_j_single    // Less than 2 columns remaining
    
// inner_loop_j_unroll2:
// This label marks the start of an inner loop that processes two columns (j and j+1) of matrix C at once.
// - Calculate positions in C for j and j+1:
//   The address of C[i][j] is computed and stored in x11. This is done by adding the base address of row i 
//   (stored in x10) to the offset for column j (x24 shifted left by 3 to account for 64-bit integers).
// - Initialize accumulators:
//   Registers x4 and x5 are initialized to zero. These will accumulate the results for C[i][j] and C[i][j+1], respectively.
// - Process two columns at once through all k values:
//   Register x25 is initialized to zero, representing the starting index k = 0 for the loop that will iterate
//   through the shared dimension of the matrices being multiplied.
inner_loop_j_unroll2:
    // Calculate positions in C for j and j+1
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Initialize accumulators
    mov     x4, xzr                // C[i][j] accumulator
    mov     x5, xzr                // C[i][j+1] accumulator
    
    // Process two columns at once through all k values
    mov     x25, xzr               // k = 0
    
// inner_loop_k_unroll2:
// This loop performs matrix multiplication for two columns of the result matrix (C) at a time,
// iterating over the shared dimension (k) of the input matrices (A and B). The loop is unrolled
// to process two columns (j and j+1) simultaneously for optimization.
// Registers:
// x9: Base address of matrix A.
// x20: Base address of matrix B.
// x11: Base address of matrix C.
// x25: Current index k in the shared dimension.
// x24: Current column index j in matrix C.
// x22: Total number of columns in matrix C (N).
// x27: Number of columns in matrix B (N).
// x13, x14, x15: Temporary registers for address calculations.
// x0: Current element A[i][k].
// x2, x3: Current elements B[k][j] and B[k][j+1].
// x4, x5: Accumulators for C[i][j] and C[i][j+1].
// Steps:
// 1. Calculate the address of A[i][k] and load its value into x0.
// 2. Calculate the base address of B[k][0] and then the addresses of B[k][j] and B[k][j+1].
// 3. Load the values of B[k][j] and B[k][j+1] into x2 and x3.
// 4. Perform multiply-accumulate operations to update the accumulators x4 and x5:
// - C[i][j] += A[i][k] * B[k][j]
// - C[i][j+1] += A[i][k] * B[k][j+1]
// 5. Increment k and check if the loop should continue.
// 6. After the loop, store the accumulated results (x4 and x5) back to C[i][j] and C[i][j+1].
// 7. Increment j by 2 to process the next pair of columns.
// 8. Check if there are more columns to process; if so, repeat the loop.
// 9. Handle any remaining single column if the total number of columns is odd.
// Notes:
// - The loop is optimized for performance by unrolling to process two columns at a time.
// - The use of ldp and stp instructions reduces memory access overhead.
// - The madd instruction combines multiplication and addition in a single step for efficiency.
inner_loop_k_unroll2:
    add     x13, x9, x25, lsl #3   // &A[i][k]
    ldr     x0, [x13]              // A[i][k]
    
    mul     x14, x25, x27          // k * N * 8
    add     x14, x20, x14          // &B[k][0]
    add     x15, x14, x24, lsl #3  // &B[k][j]
    
    // Load B elements
    ldp     x2, x3, [x15]          // B[k][j], B[k][j+1]
    
    // Multiply-accumulate for both j values
    madd    x4, x0, x2, x4         // C[i][j] += A[i][k] * B[k][j]
    madd    x5, x0, x3, x5         // C[i][j+1] += A[i][k] * B[k][j+1]
    
    add     x25, x25, #1           // k++
    cmp     x25, x22
    b.lt    inner_loop_k_unroll2
    
    // Store results back to C
    stp     x4, x5, [x11]          // Store C[i][j], C[i][j+1]
    
    add     x24, x24, #2           // j += 2
    add     x16, x24, #1           // j+1
    cmp     x16, x22
    b.lt    inner_loop_j_unroll2
    
    // Handle remaining single column if needed
    cmp     x24, x22
    b.ge    end_inner_loop_j
    
// inner_loop_j_single:
// // Processes a single column j when columns left are fewer.
inner_loop_j_single:
    // Calculate position in C: C[i][j]
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Initialize accumulator
    mov     x4, xzr                // C[i][j] accumulator
    
    // Process one column through all k values
    mov     x25, xzr               // k = 0
    
// inner_loop_k_single:
// This function performs the inner loop of a matrix multiplication operation for a single element of the result matrix C[i][j].
// Steps:
// 1. Calculate the address of A[i][k] and load its value into x0.
// 2. Compute the address of B[k][j] using the base address of B and load its value into x2.
// 3. Perform the multiply-accumulate operation: C[i][j] += A[i][k] * B[k][j].
// 4. Increment the loop counter k and check if it has reached the limit (x22). If not, repeat the loop.
// 5. Once the loop is complete, store the computed value of C[i][j] into memory.
// 6. Increment the loop counter j and check if it has reached the limit (x22). If not, jump to the outer loop (inner_loop_j_single).
// Registers used:
// - x9: Base address of matrix A.
// - x20: Base address of matrix B.
// - x11: Address to store the result in matrix C.
// - x25: Loop counter for k.
// - x24: Loop counter for j.
// - x22: Limit for loop counters (N).
// - x27: Scaling factor for matrix B (N * 8).
// - x4: Accumulator for the result of C[i][j].
// - x0: Temporary register to hold A[i][k].
// - x2: Temporary register to hold B[k][j].
// - x13, x14, x15: Temporary registers for address calculations.
inner_loop_k_single:
    // Process one k for one j
    add     x13, x9, x25, lsl #3   // &A[i][k]
    ldr     x0, [x13]              // A[i][k]
    
    mul     x14, x25, x27          // k * N * 8
    add     x14, x20, x14          // &B[k][0]
    add     x15, x14, x24, lsl #3  // &B[k][j]
    ldr     x2, [x15]              // B[k][j]
    
    madd    x4, x0, x2, x4         // C[i][j] += A[i][k] * B[k][j]
    
    add     x25, x25, #1           // k++
    cmp     x25, x22
    b.lt    inner_loop_k_single
    
    // Store result
    str     x4, [x11]              // C[i][j]
    
    add     x24, x24, #1           // j++
    cmp     x24, x22
    b.lt    inner_loop_j_single
    
// end_inner_loop_j:
// // Ends the inner loop for i and prefetches the next row of A.
end_inner_loop_j:
    add     x23, x23, #1           // i++
    
    // Prefetch next A row
    cmp     x23, x22
    b.ge    skip_next_row_prefetch
    mul     x28, x23, x27          // next_i * N * 8
    add     x9, x19, x28           // &A[next_i][0]
    prfm    pldl1keep, [x9]
    prfm    pldl1keep, [x9, #64]

// skip_next_row_prefetch:
// This label `skip_next_row_prefetch` serves as a jump point in the assembly code.
// It is used to skip the prefetching of the next row of data and directly branch
// to the `outer_loop_i` label, which likely represents the start of an outer loop
// in the matrix multiplication process. This can be useful for optimizing performance
// by avoiding unnecessary memory operations under certain conditions.
skip_next_row_prefetch:
    
    b       outer_loop_i

// end_outer_loop_i:
// // Restores registers and returns.
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
