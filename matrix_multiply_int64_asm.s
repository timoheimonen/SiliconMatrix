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
    
opt_zero_j:
    cmp     x24, x22
    b.ge    opt_zero_done
    
    add     x11, x10, x24, lsl #3  // &C[i][j]
    str     xzr, [x11]             // C[i][j] = 0
    
    add     x24, x24, #1
    b       opt_zero_j
    
opt_zero_done:
    // Process all k values first (better cache locality for A)
    mov     x25, xzr               // k = 0
    
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
    
opt_j_next:
    add     x24, x24, #1
    b       opt_j_loop
    
opt_j_done:
    
opt_k_next:
    add     x25, x25, #1
    b       opt_k_loop
    
opt_k_done:
    add     x23, x23, #1
    b       opt_i_loop
    
opt_i_done:
    b       end_outer_loop_i

large_matrix_path:
    // Original implementation for larger matrices
    // Use your existing implementation for larger matrices
    // Zero out the C matrix
    mov     x10, x21               // C pointer
    mul     x11, x27, x22          // Total size in bytes
    add     x11, x10, x11          // End of C
    
zero_loop:
    cmp     x10, x11
    b.ge    zero_done
    stp     xzr, xzr, [x10], #16   // Zero 16 bytes at a time when possible
    b       zero_loop
    
zero_done:
    // Continue with large matrix implementation
    cmp     w22, #256
    b.lt    small_matrix_path
    b       regular_implementation

// NEW: Optimized path for very small matrices (≤ 200x200)
tiny_matrix_path:
    // For tiny matrices, use classic i-k-j loop order with minimized address calculation
    // Zero out C first
    mov     x10, x21               // C pointer
    mul     x11, x27, x22          // Total size in bytes
    add     x11, x10, x11          // End of C
    
tiny_zero_loop:
    cmp     x10, x11
    b.ge    tiny_zero_done
    stp     xzr, xzr, [x10], #16
    b       tiny_zero_loop
    
tiny_zero_done:
    mov     x23, xzr               // i = 0 (row counter)
    
tiny_loop_i:
    cmp     x23, x22
    b.ge    tiny_done
    
    // Calculate A[i][0] address once per row
    mul     x28, x23, x27          // i * N * 8
    add     x9, x19, x28           // &A[i][0]
    
    // Calculate C[i][0] address once per row
    add     x10, x21, x28          // &C[i][0]
    
    mov     x25, xzr               // k = 0
    
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
    
tiny_loop_j_end:
    add     x25, x25, #1           // k++
    b       tiny_loop_k
    
tiny_loop_k_end:
    add     x23, x23, #1           // i++
    b       tiny_loop_i
    
tiny_done:
    b       end_outer_loop_i

small_matrix_path:
    // For small matrices, use a different loop order (j-i-k) for better locality
    // This helps because entire matrices likely fit in L1/L2 cache
    
    // First zero out C (helps with alignment and simplifies the inner loop)
    mov     x10, x21
    mul     x11, x27, x22
    add     x11, x10, x11
    
small_zero_loop:
    cmp     x10, x11
    b.ge    small_zero_done
    stp     xzr, xzr, [x10], #16
    b       small_zero_loop
    
small_zero_done:
    mov     x24, xzr               // j = 0 (column counter)
    
small_loop_j:
    cmp     x24, x22
    b.ge    small_done
    
    mov     x23, xzr               // i = 0 (row counter)
    
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
    
small_loop_i_end:
    add     x24, x24, #1           // j++
    b       small_loop_j
    
small_done:
    b       end_outer_loop_i

regular_implementation:
    // Start with i-j-k loops but optimize inner loop more aggressively
    mov     x23, xzr               // i = 0 (row counter)

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
    
inner_loop_j_unroll4:
    // Calculate positions in C for j, j+1, j+2, j+3
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Initialize accumulators for this row
    mov     x4, xzr                // C[i][j] accumulator
    mov     x5, xzr                // C[i][j+1] accumulator
    mov     x6, xzr                // C[i][j+2] accumulator
    mov     x7, xzr                // C[i][j+3] accumulator
    
    // Process 4 columns at once through all k values
    mov     x25, xzr               // k = 0
    
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
    
    // Check for 2-column processing
check_unroll2:
    add     x16, x24, #1           // j+1
    cmp     x16, x22
    b.ge    inner_loop_j_single    // Less than 2 columns remaining
    
inner_loop_j_unroll2:
    // Calculate positions in C for j and j+1
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Initialize accumulators
    mov     x4, xzr                // C[i][j] accumulator
    mov     x5, xzr                // C[i][j+1] accumulator
    
    // Process two columns at once through all k values
    mov     x25, xzr               // k = 0
    
inner_loop_k_unroll2:
    // Process one k at a time but for two j values
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
    
inner_loop_j_single:
    // Calculate position in C: C[i][j]
    add     x11, x10, x24, lsl #3  // &C[i][j]
    
    // Initialize accumulator
    mov     x4, xzr                // C[i][j] accumulator
    
    // Process one column through all k values
    mov     x25, xzr               // k = 0
    
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
    
end_inner_loop_j:
    add     x23, x23, #1           // i++
    
    // Prefetch next A row
    cmp     x23, x22
    b.ge    skip_next_row_prefetch
    mul     x28, x23, x27          // next_i * N * 8
    add     x9, x19, x28           // &A[next_i][0]
    prfm    pldl1keep, [x9]
    prfm    pldl1keep, [x9, #64]
skip_next_row_prefetch:
    
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
