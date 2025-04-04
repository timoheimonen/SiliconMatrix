# Makefile for compiling and linking C++ and ARM64 assembly source files
# on macOS with Apple Silicon.

# Compiler and Linker
CXX = clang++
AS = clang # Use clang also for assembling .s files on macOS

# Optimization levels (can be adjusted as needed)
# Options: -O0 (no optimization), -O1, -O2, -O3, -Ofast
OPT_LEVEL ?= -O2

# Architecture-specific optimizations for Apple Silicon
ARCH_FLAGS = -mcpu=apple-m1 -mtune=apple-m1

# C++ compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra $(OPT_LEVEL) $(ARCH_FLAGS)

# Assembler flags with Apple Silicon optimizations
ASFLAGS = $(ARCH_FLAGS)

# Linker flags
LDFLAGS = 

# Source files
CPP_SRC = main.cpp
ASM_SRC = matrix_multiply_fp_asm.s matrix_multiply_int_asm.s matrix_multiply_int64_asm.s

# Object files (derived from source files)
CPP_OBJ = $(CPP_SRC:.cpp=.o)
ASM_OBJ = $(ASM_SRC:.s=.o)

# Target executable name
TARGET = SiliconMatrix

# Default target: Build the executable with O2 optimization
all: $(TARGET)

# Build with different optimization levels
.PHONY: o0 o1 o3 ofast
o0:
	$(MAKE) clean
	$(MAKE) OPT_LEVEL=-O0 all

o1:
	$(MAKE) clean
	$(MAKE) OPT_LEVEL=-O1 all

o3:
	$(MAKE) clean
	$(MAKE) OPT_LEVEL=-O3 all

ofast:
	$(MAKE) clean
	$(MAKE) OPT_LEVEL=-Ofast all

# Rule to link the executable from object files
$(TARGET): $(CPP_OBJ) $(ASM_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $(TARGET)
	@echo "Linked executable: $(TARGET) with optimization $(OPT_LEVEL)"

# Rule to compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled C++: $< -> $@ with $(OPT_LEVEL)"

# Rule to assemble assembly source files into object files
%.o: %.s
	$(AS) $(ASFLAGS) -c $< -o $@
	@echo "Assembled ASM: $< -> $@"

# Target to clean up build artifacts
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) $(CPP_OBJ) $(ASM_OBJ)
	@echo "Cleanup complete."

# Phony targets (targets that don't represent actual files)
.PHONY: all clean

