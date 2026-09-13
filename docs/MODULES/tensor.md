# Tensor Module

> **Multi-dimensional array with automatic differentiation support**

---

> **Per-file documentation:** For detailed implementation notes, see [tensor.h.md](../../wiki/files/tensor.h.md) and [tensor.cpp.md](../../wiki/files/tensor.cpp.md).

## 📚 Overview

The Tensor module provides the fundamental data structure for Transcender. It implements a multi-dimensional array that supports:

- Various data types (FP32, FP16, INT8, etc.)
- Automatic memory management
- Shape manipulation (reshape, permute, transpose)
- Slicing and indexing
- Automatic differentiation support
- Serialization

---

## 🎯 Key Features

### 1. Multi-Dimensional Support

Tensors can have any number of dimensions (rank), with each dimension having a specific size.

```cpp
Tensor scalar = Tensor({}, DType::F32);        // 0D tensor (scalar)
Tensor vector = Tensor({10}, DType::F32);       // 1D tensor (vector)
Tensor matrix = Tensor({3, 4}, DType::F32);     // 2D tensor (matrix)
Tensor cube = Tensor({2, 3, 4}, DType::F32);    // 3D tensor
Tensor hyper = Tensor({2, 3, 4, 5}, DType::F32); // 4D tensor
```

### 2. Data Types

Supported data types through the `DType` enumeration:

| DType | C++ Type | Size | Description |
|-------|----------|------|-------------|
| `F32` | `float` | 4 bytes | Single-precision floating point |
| `F16` | `uint16_t` | 2 bytes | Half-precision floating point |
| `I64` | `int64_t` | 8 bytes | 64-bit signed integer |
| `U8` | `uint8_t` | 1 byte | 8-bit unsigned integer |
| `U4` | Packed | 0.5 bytes | 4-bit unsigned (2 per byte) |
| `I2` | Packed | 0.5 bytes | 2-bit ternary (4 per byte) |
| `I1` | Packed | 0.125 bytes | 1-bit binary (8 per byte) |

### 3. Memory Management

- Automatic allocation and deallocation
- Support for custom allocators
- Arena allocation for efficient temporary tensors
- Contiguous and non-contiguous storage

### 4. Shape Operations

- Reshape without copying data
- Permute dimensions
- Transpose
- Slicing and indexing

### 5. Autograd Integration

- Track gradients for trainable parameters
- Support for automatic differentiation
- Integration with the AutogradEngine

---

## 🏗️ API Reference

### Class: Tensor

#### Construction

```cpp
// Default constructor
Tensor tensor1;

// Construct with shape and dtype
Tensor tensor2({2, 3, 4}, DType::F32);

// Construct with custom allocator
ArenaAllocator arena(1024 * 1024);
Tensor tensor3({100, 100}, DType::F32, &arena);

// Static constructors
Tensor zeros = Tensor::zeros({2, 3});
Tensor ones = Tensor::ones({2, 3});
Tensor empty = Tensor::empty({2, 3});
Tensor from_data = Tensor::from_data({2, 3}, DType::F32, data_ptr);
```

#### Shape Access

```cpp
Shape shape = tensor.shape();           // Get shape object
int rank = tensor.rank();               // Get number of dimensions
int64_t dim0 = tensor.dim(0);            // Get size of dimension 0
int64_t numel = tensor.numel();          // Get total number of elements
```

#### Data Access

```cpp
// Get raw data pointer
float* data = tensor.data<float>();
const float* const_data = tensor.data<float>();

// Element access (bounds-checked in debug builds)
float value = tensor.at<float>({i, j, k});
float& ref = tensor.at<float>({i, j, k});

// Unchecked element access (faster)
float value = tensor({i, j, k});
```

#### Shape Manipulation

```cpp
// Reshape (may copy if not contiguous)
Tensor reshaped = tensor.reshape({4, 3, 2});

// Permute dimensions
Tensor permuted = tensor.permute({2, 0, 1});  // Swap dimensions

// Transpose
Tensor transposed = tensor.transpose();      // 2D transpose
Tensor transposed = tensor.transpose(0, 2); // Swap dim 0 and 2

// Ensure contiguous storage
Tensor contiguous = tensor.contiguous();
```

#### Slicing

```cpp
// Slice along dimension
Tensor slice = tensor.slice(0, 1, 3);  // Dim 0, indices [1, 3)

// Index select
Tensor indices = Tensor({0, 2, 4}, DType::I64);
Tensor selected = tensor.index_select(0, indices);
```

#### Views

```cpp
// Create a view (no copy)
Tensor view = tensor.view({4, 3, 2});

// Narrow (reduce dimension size)
Tensor narrow = tensor.narrow(0, 1, 3);  // Dim 0, start=1, length=3
```

#### Properties

```cpp
bool is_contiguous = tensor.is_contiguous();
bool requires_grad = tensor.requires_grad();
size_t nbytes = tensor.nbytes();
DType dtype = tensor.dtype();
```

#### Autograd

```cpp
// Enable gradient tracking
tensor.requires_grad(true);

// Compute gradients (if in autograd context)
tensor.backward(gradient);
```

#### Serialization

```cpp
// Save to stream
tensor.save(output_stream);

// Load from stream
Tensor loaded = Tensor::load(input_stream);
```

#### Utility

```cpp
// Fill with value
tensor.fill_(42.0f);

// Zero out
tensor.zero_();
```

### Class: Shape

#### Construction

```cpp
Shape shape1;                          // Empty shape
Shape shape2({2, 3, 4});               // From initializer list
Shape shape3 = {2, 3, 4};              // From braced init
Shape shape4(std::vector<int64_t>{2, 3, 4}); // From vector
```

#### Access

```cpp
int rank = shape.rank();               // Number of dimensions
int64_t numel = shape.numel();          // Total number of elements
int64_t dim0 = shape.dims[0];           // Size of dimension 0
int64_t dim1 = shape.dims[1];           // Size of dimension 1
```

#### Comparison

```cpp
if (shape1 == shape2) {
    // Shapes are equal
}
```

---

## 🔍 Implementation Details

### Memory Layout

Tensors store data in a **row-major** (C-style) layout:

```
For a 2x3 matrix:
+-----+-----+-----+
| (0,0) | (0,1) | (0,2) |  -> Contiguous in memory
+-----+-----+-----+
| (1,0) | (1,1) | (1,2) |
+-----+-----+-----+
```

The element at position `{i, j, k, ...}` is located at:
```
index = i * (d1 * d2 * ...) + j * (d2 * d3 * ...) + k * d3 + ...
```

### Contiguity

A tensor is **contiguous** if its elements are stored sequentially in memory without gaps. Operations that preserve contiguity:

- `reshape()` - If the new shape is compatible
- `permute()` - Usually not contiguous
- `transpose()` - Usually not contiguous
- `contiguous()` - Always returns a contiguous tensor

### Views vs Copies

- **View:** Creates a new Tensor that references the same underlying data with a different shape. No memory is allocated.
- **Copy:** Creates a new Tensor with its own copy of the data. Memory is allocated.

Operations that return views:
- `view()`
- `narrow()`
- `slice()` (sometimes)
- `permute()` (sometimes)

Operations that return copies:
- `reshape()` (if shape incompatible)
- `contiguous()`
- Most arithmetic operations

### Custom Allocators

Transcender supports custom memory allocators for:

- Arena allocation (fast allocation/deallocation in bulk)
- Pool allocation (fixed-size blocks)
- GPU allocation (for GPU tensors)

```cpp
class CustomAllocator : public Allocator {
public:
    void* allocate(size_t size, size_t alignment) override;
    void deallocate(void* ptr, size_t size, size_t alignment) override;
};

CustomAllocator alloc;
Tensor t({100, 100}, DType::F32, &alloc);
```

### Data Type Conversions

The Tensor class provides methods for type conversion:

```cpp
Tensor f32 = Tensor::zeros({10}, DType::F32);
Tensor i64 = f32.to(DType::I64);
```

---

## 🎯 Best Practices

### 1. Memory Efficiency

- **Use views** when possible to avoid copies
- **Use arena allocators** for temporary tensors
- **Minimize intermediate tensors** in hot loops
- **Prefer in-place operations** (`add_()`, `mul_()`, etc.)

### 2. Performance

- **Prefer contiguous tensors** for performance
- **Avoid unnecessary copies**
- **Use appropriate data types** (FP32 for training, INT8 for inference)
- **Batch operations** when possible

### 3. Correctness

- **Check shapes** before operations
- **Use bounds-checked access** (`at()`) in development
- **Validate inputs** to functions
- **Handle edge cases** (empty tensors, zero dimensions)

### 4. Autograd

- **Only enable requires_grad** for trainable parameters
- **Clear gradients** between iterations
- **Use AutogradEngine** for complex operations

---

## 🐛 Common Pitfalls

### 1. Non-Contiguous Tensors

```cpp
Tensor a({2, 3});
Tensor b = a.transpose();  // Non-contiguous!

// This may fail or be slow
Tensor c = b.reshape({6});  // May need to copy

// Better: ensure contiguous first
Tensor c = b.contiguous().reshape({6});
```

### 2. Shape Mismatches

```cpp
Tensor a({2, 3});
Tensor b({3, 2});

// This will fail
Tensor c = a + b;  // Shape mismatch!

// Check shapes first
QUANT_CHECK(a.shape() == b.shape(), "Shape mismatch");
```

### 3. Data Type Mismatches

```cpp
Tensor a({10}, DType::F32);
Tensor b({10}, DType::I64);

// This may cause issues
Tensor c = a + b;  // Different dtypes!

// Convert to same dtype first
Tensor b_f32 = b.to(DType::F32);
Tensor c = a + b_f32;
```

### 4. Memory Leaks

```cpp
// Don't do this
Tensor* t = new Tensor({100, 100});
// Forgot to delete!

// Better: use stack allocation
Tensor t({100, 100});

// Or smart pointers if you need dynamic lifetime
std::unique_ptr<Tensor> t = std::make_unique<Tensor>({100, 100});
```

### 5. Autograd Context

```cpp
// Don't forget to enable autograd
AutogradEngine::set_enabled(true);

// And clear between iterations
AutogradEngine::instance().clear();
```

---

## 🔧 Debugging

### Debug Output

```cpp
// Print tensor info
std::cout << "Shape: " << tensor.shape().dims << std::endl;
std::cout << "DType: " << (int)tensor.dtype() << std::endl;
std::cout << "Contiguous: " << tensor.is_contiguous() << std::endl;

// Print tensor data
for (int64_t i = 0; i < tensor.numel(); i++) {
    std::cout << tensor.data<float>()[i] << " ";
}
```

### Debug Checks

```cpp
// Check shape
QUANT_CHECK(tensor.rank() == 2, "Expected 2D tensor");

// Check contiguity
QUANT_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

// Check dtype
QUANT_CHECK(tensor.dtype() == DType::F32, "Expected FP32 tensor");
```

### Memory Debugging

```bash
# Use address sanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQUANT_SANITIZE=ON
cmake --build build --parallel

# Run tests with sanitizer
./build/tests/test_tensor
```

---

## 📊 Performance Tips

### 1. Pre-Allocate Tensors

```cpp
// Bad: Allocate in hot loop
for (int i = 0; i < 1000; i++) {
    Tensor temp = Tensor::zeros({100, 100});
    // ...
}

// Good: Allocate once
Tensor temp = Tensor::zeros({100, 100});
for (int i = 0; i < 1000; i++) {
    temp.zero_();
    // ...
}
```

### 2. Use In-Place Operations

```cpp
// Bad: Creates temporary
a = a + b;

// Good: In-place
a.add_(b);
```

### 3. Batch Operations

```cpp
// Bad: Loop over individual elements
for (int i = 0; i < n; i++) {
    Tensor output_i = model.forward(input_i);
}

// Good: Batch processing
Tensor inputs = Tensor::stack(input_tensors);
Tensor outputs = model.forward(inputs);
```

### 4. Avoid Unnecessary Copies

```cpp
// Bad: Multiple copies
Tensor a = ...;
Tensor b = a;
Tensor c = b.reshape({10, 10});

// Good: Chain operations
Tensor c = a.reshape({10, 10});
```

---

## 📚 Related Modules

- **[Autograd](autograd.md)** - Automatic differentiation
- **[Math](math.md)** - Mathematical operations on tensors
- **[Model](model.md)** - Neural network models that use tensors
- **[Memory](memory.md)** - Memory management for tensors

---

## 🎓 Example: Complete Tensor Usage

```cpp
#include "quant/tensor.h"
#include "quant/math.h"
#include <iostream>

using namespace quant;

int main() {
    // Create tensors
    Tensor a = Tensor::ones({2, 3}, DType::F32);
    Tensor b = Tensor::zeros({2, 3}, DType::F32);
    
    // Perform operations
    Tensor c = a + b;
    
    // Print result
    for (int64_t i = 0; i < c.shape()[0]; i++) {
        for (int64_t j = 0; j < c.shape()[1]; j++) {
            std::cout << c({i, j}) << " ";
        }
        std::cout << std::endl;
    }
    
    // Matrix multiplication
    Tensor d({3, 4}, DType::F32);
    Tensor e({4, 5}, DType::F32);
    Tensor f = math::matmul(d, e);
    
    // Reshape
    Tensor g = f.reshape({3, 20});
    
    // Transpose
    Tensor h = f.transpose();
    
    return 0;
}
```

---

*Last updated: July 12, 2026*
