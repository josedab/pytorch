# Deep Dive: Tensor Operations and Kernel Dispatch

**Analysis based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Introduction

In the previous post, we explored PyTorch's overall architecture and core abstractions. Now, we'll dive deep into how PyTorch actually executes tensor operations—the foundation of all deep learning computations. We'll examine the sophisticated dispatch system, efficient iteration patterns, and how a single Python call like `torch.add()` becomes optimized assembly code running on your GPU.

Understanding tensor operations at this level is crucial for:
- **Performance optimization:** Knowing when operations are fused, how memory is accessed, and what triggers kernel launches
- **Debugging:** Understanding dispatch helps track down why an operation behaves differently on different devices
- **Extension:** Writing custom operators that integrate seamlessly with PyTorch's infrastructure

## Objectives

By the end of this post, you'll understand:

1. How TensorIterator enables efficient element-wise operations with broadcasting
2. The AT_DISPATCH macro system for compile-time type specialization
3. How operators are registered and dispatched to backend implementations
4. Real-world examples of CPU and CUDA kernel implementations
5. Memory layout, strides, and contiguity considerations

## TensorIterator: The Swiss Army Knife of Element-Wise Operations

One of PyTorch's most elegant abstractions is `TensorIterator`, which provides a unified interface for implementing element-wise operations across different devices, data types, and memory layouts.

### The Problem TensorIterator Solves

Consider implementing a simple element-wise operation like addition. You need to handle:

- **Broadcasting:** `[3, 1, 5] + [5]` should expand dimensions correctly
- **Multiple memory layouts:** Contiguous, strided, channels-last, etc.
- **Different devices:** CPU, CUDA, MPS, XPU
- **Different data types:** float32, float16, int64, complex128, etc.
- **Type promotion:** `int32 + float32` → `float32`
- **In-place operations:** `a.add_(b)` must modify `a` in place
- **Multi-dimensional iteration:** Efficiently traverse N-dimensional tensors

Without TensorIterator, every operation would need hundreds of lines of boilerplate to handle these cases.

### TensorIterator Architecture

**Source:** [`aten/src/ATen/native/TensorIterator.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/TensorIterator.h)

```cpp
class TensorIterator {
public:
  // Configuration builder pattern
  static TensorIteratorConfig()
    .add_output(output_tensor)
    .add_input(input1)
    .add_input(input2)
    .promote_inputs_to_common_dtype(true)
    .build();

  // Iteration
  void for_each(loop_fn, int64_t grain_size = at::internal::GRAIN_SIZE);

  // Properties
  int ntensors() const;
  int noutputs() const;
  IntArrayRef shape() const;

  // Type information
  ScalarType dtype(int arg = 0) const;
  Device device(int arg = 0) const;
};
```

### Example: Implementing `torch.add()`

Let's trace through a real addition operation:

**Python:**
```python
a = torch.randn(3, 4, 5, device='cuda')
b = torch.randn(5, device='cuda')
c = torch.add(a, b)  # Broadcasting: [3,4,5] + [5] → [3,4,5]
```

**Generated C++ API call** (from YAML):
```cpp
// aten/src/ATen/ops/add.h (generated)
at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha=1);
```

**Dispatcher routes to CUDA implementation:**
```cpp
// aten/src/ATen/native/BinaryOps.cpp
Tensor add_Tensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor out;
  // Build TensorIterator configuration
  auto iter = TensorIterator::binary_op(out, self, other);
  // Dispatch to device-specific kernel
  add_stub(iter.device_type(), iter, alpha);
  return iter.output();
}
```

**CUDA kernel implementation:**
```cpp
// aten/src/ATen/native/cuda/BinaryAddKernel.cu
void add_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kHalf, kBFloat16,
    iter.dtype(),
    "add_cuda",
    [&]() {
      // Type-specialized lambda
      using scalar_t = scalar_t;  // float, int64, complex64, etc.

      gpu_kernel_with_scalars(iter, [alpha]GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a + alpha.to<scalar_t>() * b;
      });
    }
  );
}
```

See the full implementation in [`aten/src/ATen/native/cuda/BinaryMiscOps.cu`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/cuda/BinaryMiscOps.cu).

### How TensorIterator Handles Broadcasting

Broadcasting is automatic and efficient:

```cpp
// Input shapes: a=[3,4,5], b=[5]
auto iter = TensorIterator::binary_op(out, a, b);

// TensorIterator internally:
// 1. Computes output shape: [3,4,5]
// 2. Determines each tensor's strides for efficient iteration
// 3. Identifies dimensions that need broadcasting (b's first two dims)
// 4. Configures iteration to minimize memory access
```

The iteration order is optimized based on:
- **Memory layout:** Iterates in the order that maximizes cache hits
- **Coalescing:** On CUDA, ensures adjacent threads access adjacent memory
- **Vectorization:** On CPU, enables SIMD instructions

## The AT_DISPATCH Macro System

PyTorch needs to generate specialized code for each supported data type (float32, int64, etc.) without writing the same code 15 times. The solution is C++ template metaprogramming through the `AT_DISPATCH_*` macro family.

### Why Type Specialization Matters

Consider a simple addition:
```cpp
// Generic version (slow)
template<typename T>
T add(T a, T b) { return a + b; }

// The compiler can't optimize this well because T is unknown
```

vs.

```cpp
// Specialized version (fast)
float add_float(float a, float b) { return a + b; }

// Compiler generates optimized assembly:
// - Uses SIMD instructions (addps on x86, fadd on ARM)
// - Eliminates function call overhead
// - Optimizes register allocation
```

AT_DISPATCH generates the specialized version at compile time while keeping the source code generic.

### AT_DISPATCH Macro Family

**Source:** [`aten/src/ATen/Dispatch.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/Dispatch.h)

```cpp
// Dispatch to floating-point types
AT_DISPATCH_FLOATING_TYPES(dtype, "op_name", [&]() {
  using scalar_t = scalar_t;  // Will be float, double, or float16
  // Implementation using scalar_t
});

// All types including half and bfloat16
AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "op_name", [&]() {
  // ...
});

// Integer types only
AT_DISPATCH_INTEGRAL_TYPES(dtype, "op_name", [&]() {
  // ...
});

// Complex types
AT_DISPATCH_COMPLEX_TYPES(dtype, "op_name", [&]() {
  // ...
});

// New unified API (recommended)
AT_DISPATCH_V2(dtype, "op_name", AT_WRAP([&]() {
  // Type-specialized code
}), AT_EXPAND(AT_ALL_TYPES_AND2(Half, BFloat16)));
```

### Example: ReLU Implementation

**CPU implementation:**

```cpp
// aten/src/ATen/native/Activation.cpp
Tensor relu_cpu(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "relu_cpu", [&]() {
    using scalar_t = scalar_t;
    cpu_kernel(iter, [](scalar_t a) -> scalar_t {
      return std::max(scalar_t(0), a);
    });
  });

  return iter.output();
}
```

**What happens at compile time:**

The macro expands to something like:
```cpp
switch (self.scalar_type()) {
  case ScalarType::Float: {
    using scalar_t = float;
    cpu_kernel(iter, [](float a) -> float {
      return std::max(0.0f, a);
    });
    break;
  }
  case ScalarType::Double: {
    using scalar_t = double;
    cpu_kernel(iter, [](double a) -> double {
      return std::max(0.0, a);
    });
    break;
  }
  // ... cases for int32, int64, int16, etc.
}
```

Each case gets fully optimized as if you wrote it by hand.

## Operator Registration and Dispatch

Now let's understand how PyTorch knows which implementation to call for a given operation.

### The Three-Level Dispatch System

**Level 1: Operator Schema Definition**

```yaml
# aten/src/ATen/native/native_functions.yaml
- func: relu(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: relu  # Fallback implementation
    CPU: relu_cpu                     # CPU-specific
    CUDA: relu_cuda                   # CUDA-specific
    QuantizedCPU: relu_quantized_cpu  # Quantized variant
```

**Level 2: Code Generation**

[`torchgen/gen.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torchgen/gen.py) reads the YAML and generates:

```cpp
// ATen/ops/relu.h (generated)
namespace at {
  Tensor relu(const Tensor& self);
}

// ATen/ops/relu_native.h (generated)
namespace at::native {
  Tensor relu_cpu(const Tensor& self);
  Tensor relu_cuda(const Tensor& self);
}

// Registration code (generated)
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("relu", TORCH_FN(at::native::relu_cpu));
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("relu", TORCH_FN(at::native::relu_cuda));
}
```

**Level 3: Runtime Dispatch**

```cpp
// User calls:
Tensor a = torch::randn({10}, torch::device(torch::kCUDA));
Tensor b = torch::relu(a);

// Dispatcher extracts DispatchKeySet from tensor 'a':
DispatchKeySet keys = a.key_set();
// keys includes: CUDA (from device), Dense (from layout)

// Find highest priority key:
DispatchKey key = keys.highestPriorityTypeId();
// key = DispatchKey::CUDA

// Lookup in dispatch table:
const auto& fn = dispatcher.findKernel("aten::relu", key);
// fn = &at::native::relu_cuda

// Call implementation:
return fn(a);
```

See the dispatcher implementation in [`aten/src/ATen/core/dispatch/Dispatcher.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/core/dispatch/Dispatcher.cpp).

### Autograd Integration

When a tensor requires gradients, an additional dispatch layer is inserted:

```
User calls: torch.relu(x)  # x.requires_grad = True
    ↓
DispatchKeySet includes AutogradCUDA
    ↓
Dispatch to autograd layer first
    ↓
Autograd wrapper:
  1. Record operation in computation graph
  2. Create ReluBackward node
  3. Redispatch to actual CUDA kernel
    ↓
CUDA relu kernel executes
    ↓
Return tensor with grad_fn = ReluBackward
```

The autograd wrappers are generated from [`tools/autograd/derivatives.yaml`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/tools/autograd/derivatives.yaml):

```yaml
# derivatives.yaml
- name: relu(Tensor self) -> Tensor
  self: grad * (self > 0).type_as(grad)
```

This generates:
```cpp
Tensor ReluBackward::apply(const Tensor& grad) {
  return grad * (self > 0).type_as(grad);
}
```

## Real-World Example: Matrix Multiplication

Let's trace a complete example from Python to CUDA kernel:

**Python:**
```python
A = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
B = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
C = torch.matmul(A, B)
```

**Step 1: Python Binding**
```cpp
// torch/csrc/Module.cpp (PyBind11)
m.def("matmul", &torch::matmul);
```

**Step 2: Dispatcher Routes to CUDA**
```cpp
// aten/src/ATen/native/LinearAlgebra.cpp
Tensor matmul(const Tensor& self, const Tensor& other) {
  // Dispatcher selects CUDA implementation
  return at::native::matmul_cuda(self, other);
}
```

**Step 3: CUDA Implementation**
```cpp
// aten/src/ATen/native/cuda/Blas.cpp
Tensor matmul_cuda(const Tensor& self, const Tensor& other) {
  // Validate inputs
  TORCH_CHECK(self.dim() >= 2 && other.dim() >= 2);

  // Allocate output
  auto out = at::empty({self.size(0), other.size(1)}, self.options());

  // Call cuBLAS
  at::cuda::blas::gemm<float>(
    /*transa=*/'N',
    /*transb=*/'N',
    /*m=*/self.size(0),
    /*n=*/other.size(1),
    /*k=*/self.size(1),
    /*alpha=*/1.0f,
    /*A=*/self.data_ptr<float>(),
    /*lda=*/self.stride(0),
    /*B=*/other.data_ptr<float>(),
    /*ldb=*/other.stride(0),
    /*beta=*/0.0f,
    /*C=*/out.data_ptr<float>(),
    /*ldc=*/out.stride(0)
  );

  return out;
}
```

See [`aten/src/ATen/native/cuda/Blas.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/cuda/Blas.cpp).

**Step 4: cuBLAS Kernel**

cuBLAS (NVIDIA's optimized BLAS library) launches highly optimized CUDA kernels that:
- Use tensor cores on supported GPUs (A100, H100)
- Tile the computation for cache efficiency
- Fuse memory operations to reduce bandwidth

## Memory Layout and Strides

Understanding memory layout is crucial for performance.

### Stride Computation

A tensor's strides determine how its logical indices map to memory offsets:

```python
# Contiguous tensor
a = torch.randn(3, 4, 5)
print(a.stride())  # (20, 5, 1)
# Element [i, j, k] is at memory offset: i*20 + j*5 + k*1

# Transposed tensor (non-contiguous)
b = a.transpose(1, 2)
print(b.shape)   # (3, 5, 4)
print(b.stride()) # (20, 1, 5)
# Element [i, j, k] is at memory offset: i*20 + j*1 + k*5
```

**Key insight:** Transpose doesn't copy data, it just changes the stride metadata!

### Contiguity Checks

Many operations require contiguous tensors for performance:

```cpp
// aten/src/ATen/native/cuda/SomeKernel.cu
void some_kernel(const Tensor& input) {
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

  // Can now assume memory is laid out sequentially
  const float* data = input.data_ptr<float>();
  // data[i*stride[0] + j*stride[1]] → data[i*m + j] when contiguous
}
```

If not contiguous, you can make it so:
```python
# Creates a contiguous copy if needed
b_contig = b.contiguous()
```

### Channels-Last Memory Format

For CNNs, channels-last format improves performance:

```python
# Default: NCHW (batch, channels, height, width)
x = torch.randn(32, 3, 224, 224)
print(x.stride())  # (150528, 50176, 224, 1)

# Channels-last: NHWC
x_cl = x.contiguous(memory_format=torch.channels_last)
print(x_cl.stride())  # (150528, 1, 672, 3)
# Better cache locality for convolution operations
```

See the implementation in [`aten/src/ATen/native/TensorProperties.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/TensorProperties.cpp).

## CPU Vectorization

PyTorch uses vectorization for efficient CPU operations:

```cpp
// aten/src/ATen/native/cpu/Loops.h
template <typename func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op) {
  using vec_t = Vectorized<scalar_t>;  // SSE/AVX/NEON vector type

  // Vectorized loop (process 8 floats at once on AVX2)
  for (int64_t i = 0; i < n; i += vec_t::size()) {
    vec_t a_vec = vec_t::loadu(a_ptr + i);
    vec_t b_vec = vec_t::loadu(b_ptr + i);
    vec_t result = op(a_vec, b_vec);
    result.store(out_ptr + i);
  }

  // Scalar tail loop (remaining elements)
  for (int64_t i = vec_n; i < n; i++) {
    out_ptr[i] = op(a_ptr[i], b_ptr[i]);
  }
}
```

This can provide 4-8x speedup on modern CPUs.

## Key Takeaways

1. **TensorIterator abstracts complexity:** Broadcasting, type promotion, memory layouts, and device differences are handled automatically.

2. **AT_DISPATCH enables type specialization:** Compile-time code generation produces optimized kernels for each data type without code duplication.

3. **Three-level dispatch:** Schema definition (YAML) → code generation → runtime dispatch table lookup provides flexibility and performance.

4. **Memory layout matters:** Understanding strides and contiguity is crucial for performance. Operations like transpose are free, but may require contiguous() before passing to some kernels.

5. **Autograd is transparent:** The dispatcher automatically inserts autograd tracking when needed without changing operation semantics.

6. **Vectorization is automatic:** CPU kernels use SIMD instructions, CUDA kernels use optimized libraries like cuBLAS.

## What's Next

In the next post, we'll explore PyTorch's patterns and practices:
- Testing infrastructure and device-type parametrization
- Code quality tools and CI/CD workflows
- Documentation generation and best practices
- Common design patterns throughout the codebase

## References

- **TensorIterator:** [`aten/src/ATen/native/TensorIterator.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/TensorIterator.h)
- **Dispatch Macros:** [`aten/src/ATen/Dispatch.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/Dispatch.h)
- **Binary Operations:** [`aten/src/ATen/native/BinaryOps.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/BinaryOps.cpp)
- **CUDA Kernels:** [`aten/src/ATen/native/cuda/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/cuda)
- **Dispatcher Implementation:** [`aten/src/ATen/core/dispatch/Dispatcher.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/core/dispatch/Dispatcher.cpp)
- **Native Functions:** [`aten/src/ATen/native/native_functions.yaml`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/native_functions.yaml)

---

*This analysis is based on PyTorch commit [`5d99a79`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984).*
