# Performance Analysis and Optimization in PyTorch

**Analysis based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Introduction

Performance is central to PyTorch's value proposition. Deep learning workloads demand massive computational throughput, and even small inefficiencies multiply across billions of operations. PyTorch achieves high performance through a multi-layered optimization strategy spanning JIT compilation, operator fusion, intelligent memory management, and kernel optimization.

In this post, we'll dissect PyTorch's performance optimization infrastructure, examine real bottlenecks, explore optimization techniques at every level of the stack, and provide actionable insights for improving your own PyTorch workloads.

## Objectives

By the end of this post, you'll understand:

1. PyTorch's JIT compilation system and graph optimizations
2. Operator fusion strategies and their impact
3. CUDA memory management and caching allocator design
4. Kernel-level optimizations (vectorization, cuBLAS integration)
5. Profiling infrastructure for identifying bottlenecks
6. Common performance pitfalls and how to avoid them

## The Performance Stack

PyTorch optimizes performance at multiple levels:

```
Level 1: Graph Optimization (TorchScript, Dynamo, Inductor)
├─ Dead code elimination
├─ Constant folding
├─ Common subexpression elimination
├─ Operator fusion
└─ Loop optimization

Level 2: Operator Fusion
├─ Add-ReLU fusion
├─ Linear layer fusion
├─ BatchNorm fusion
└─ Fused optimizers (FusedAdam, FusedSGD)

Level 3: Memory Management
├─ CUDA caching allocator
├─ Memory pooling
├─ Stream-based allocation
└─ Graph capture optimization

Level 4: Kernel Optimization
├─ Type specialization (AT_DISPATCH)
├─ CPU vectorization (AVX2, AVX512)
├─ CUDA kernel fusion
└─ cuBLAS/cuDNN integration

Level 5: Hardware Utilization
├─ TensorCore usage (mixed precision)
├─ Memory coalescing
├─ Kernel launch optimization
└─ PCIe transfer minimization
```

## JIT Compilation and Graph Optimization

### TorchScript: Ahead-of-Time Compilation

TorchScript converts eager Python code into an optimized intermediate representation.

**Example:**

```python
import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x

# Eager execution
module = MyModule()
result = module(torch.randn(5, 10))

# TorchScript compilation
scripted = torch.jit.script(module)
# or: scripted = torch.jit.trace(module, torch.randn(5, 10))

# Optimized execution
optimized_result = scripted(torch.randn(5, 10))
```

**What happens during compilation:**

**Source:** [`torch/csrc/jit/runtime/graph_executor.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/runtime/graph_executor.cpp)

```
Python → TorchScript AST
    ↓
Graph IR (intermediate representation)
    ↓
runProfilingInsensitiveOptimizations():
  - Constant propagation
  - Dead code elimination
  - Common subexpression elimination
  - Peephole optimization
    ↓
(Optional profiling run to gather type/shape info)
    ↓
runProfilingOptimizations():
  - Operator fusion (based on runtime shapes)
  - Specialize for observed types/shapes
  - Inline function calls
    ↓
runFinalOptimizations():
  - Remove dead code introduced by previous passes
  - Final fusion opportunities
    ↓
Compiled execution plan (cached)
```

### Graph Optimization Passes

**Key optimization passes:**

| Pass | Location | Impact |
|------|----------|--------|
| Constant Propagation | `torch/csrc/jit/passes/constant_propagation.cpp` | Evaluates constants at compile time |
| Dead Code Elimination | `torch/csrc/jit/passes/dead_code_elimination.cpp` | Removes unused computations |
| Common Subexpr. Elim. | `torch/csrc/jit/passes/common_subexpression_elimination.cpp` | Reuses repeated computations |
| Peephole | `torch/csrc/jit/passes/peephole.cpp` | Local algebraic simplifications |
| Add-ReLU Fusion | `torch/csrc/jit/passes/fuse_relu.cpp` | Fuses add + relu into single kernel |
| Linear Fusion | `torch/csrc/jit/passes/fuse_linear.cpp` | Fuses matmul + bias add |
| TensorExpr Fuser | `torch/csrc/jit/passes/tensorexpr_fuser.cpp` | Generates fused kernels for element-wise ops |

**Example: Constant propagation**

```python
# Before optimization
def forward(x):
    y = x + torch.tensor([1, 2, 3])
    z = y * 2
    return z

# After constant propagation
def forward(x):
    # torch.tensor([1, 2, 3]) * 2 computed at compile time
    return x + torch.tensor([2, 4, 6])
```

### TensorExpr Fuser

**Source:** [`torch/csrc/jit/passes/tensorexpr_fuser.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/passes/tensorexpr_fuser.cpp)

TensorExpr fuses multiple element-wise operations into single kernels:

**Unfused code:**
```python
def forward(x, y):
    a = x + y     # Kernel 1: read x, y; write a
    b = a * 2     # Kernel 2: read a; write b
    c = torch.sigmoid(b)  # Kernel 3: read b; write c
    return c

# 3 kernel launches, 6 memory operations
```

**Fused code (generated):**
```cuda
__global__ void fused_kernel(float* x, float* y, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = x[i] + y[i];
    float b = a * 2.0f;
    c[i] = 1.0f / (1.0f + expf(-b));
  }
}
// 1 kernel launch, 3 memory operations (read x, y; write c)
```

**Performance improvement:** 2-3x speedup by reducing memory bandwidth and kernel launch overhead.

**Supported operations:**
```cpp
// torch/csrc/jit/passes/tensorexpr_fuser.cpp
bool isSupported(Node* node) {
  static const OperatorSet supported_ops = {
    aten::add, aten::mul, aten::sub, aten::div,
    aten::relu, aten::sigmoid, aten::tanh,
    aten::exp, aten::log, aten::sqrt,
    aten::sin, aten::cos,
    // ... many more
  };
  return supported_ops.contains(node);
}
```

## Operator Fusion

### Add-ReLU Fusion

**Source:** [`torch/csrc/jit/passes/fuse_relu.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/passes/fuse_relu.cpp)

```python
# Pattern recognized:
x = a + b
y = torch.relu(x)

# Fused into:
y = torch._C._nn._add_relu(a, b)  # Single kernel
```

**CUDA kernel:**

```cpp
// aten/src/ATen/native/cuda/Activation.cu
template <typename scalar_t>
__global__ void add_relu_kernel(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    int64_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    scalar_t sum = a[idx] + b[idx];
    out[idx] = sum > scalar_t(0) ? sum : scalar_t(0);
  }
}
```

**Performance impact:**
- **Without fusion:** 2 kernel launches, 4 memory operations (read a, b, intermediate; write intermediate, output)
- **With fusion:** 1 kernel launch, 3 memory operations (read a, b; write output)
- **Speedup:** ~1.8x for memory-bound operations

### Linear Layer Fusion

**Source:** [`torch/csrc/jit/passes/fuse_linear.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/passes/fuse_linear.cpp)

```python
# Pattern 1: matmul + add
x = torch.matmul(input, weight.t())
y = x + bias

# Fused into:
y = torch.nn.functional.linear(input, weight, bias)

# Pattern 2: addmm (add matrix multiply)
y = torch.addmm(bias, input, weight.t())
# Already optimized, uses cuBLAS gemm with beta parameter
```

**Performance:** cuBLAS `gemm` with bias is 10-20% faster than separate matmul + add.

### Fused Optimizers

**Source:** [`aten/src/ATen/native/FusedAdam.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/FusedAdam.cpp)

Standard Adam optimizer:
```python
# Unfused: N kernel launches for N parameters
for param in model.parameters():
    grad = param.grad
    # Update momentum
    state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
    # Update variance
    state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad ** 2
    # Compute update
    param.data -= lr * state['exp_avg'] / (state['exp_avg_sq'].sqrt() + eps)
```

Fused Adam:
```python
# Single kernel for all parameters
torch._fused_adam_(
    params,      # All parameters (list of tensors)
    grads,       # All gradients
    exp_avgs,    # All momentum states
    exp_avg_sqs, # All variance states
    state_steps,
    lr=lr,
    beta1=beta1,
    beta2=beta2,
    eps=eps
)
```

**Performance improvement:** 2-3x faster on models with many small parameters (e.g., Transformers).

## CUDA Memory Management

### The Caching Allocator

**Source:** [`c10/cuda/CUDACachingAllocator.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/c10/cuda/CUDACachingAllocator.cpp)

**Problem:** `cudaMalloc` and `cudaFree` are extremely slow (~10-100µs each). Naive allocation would create huge overhead.

**Solution:** Three-tier memory pooling strategy:

```
Small allocations (< 1 MB):
  Pool: 2 MB blocks
  Strategy: Pack multiple allocations in same block
  Fragmentation: Minimal (small objects)

Medium allocations (1-10 MB):
  Pool: 20 MB blocks
  Strategy: Split blocks to exact size (with rounding)
  Fragmentation: Moderate

Large allocations (> 10 MB):
  Pool: Individual blocks (never split)
  Strategy: Reuse if size matches within 1 MB
  Fragmentation: None (dedicated blocks)
```

**Key design decisions:**

1. **Stream-based allocation:** Blocks are associated with CUDA streams for safe reuse
   ```cpp
   // Allocation tied to stream
   void* ptr = allocator.allocate(size, stream);
   // Only reused on same stream or after synchronization
   ```

2. **Lazy deallocation:** Free doesn't return memory to OS immediately
   ```cpp
   allocator.free(ptr);  // Returns to pool, doesn't call cudaFree
   ```

3. **Fallback strategy:** If allocation fails, try freeing cached blocks
   ```cpp
   // c10/cuda/CUDACachingAllocator.cpp
   // 1. Try cudaMalloc
   // 2. If fails, free one cached block and retry
   // 3. If fails, free ALL cached blocks and retry
   // 4. If still fails, throw OOM error
   ```

**Performance impact:**
- **Without caching:** cudaMalloc overhead: ~100µs × millions of allocations = seconds of wasted time
- **With caching:** Pool lookup: ~100ns × millions of allocations = milliseconds
- **Speedup:** 100-1000x reduction in allocation overhead

### Graph Capture Optimization

**CUDA Graphs** allow capturing a sequence of kernels and replaying them with minimal overhead.

**Problem:** Allocated tensors during graph capture may be freed and reallocated with different addresses during replay, causing crashes.

**Solution:** Graph-private memory pools

```cpp
// c10/cuda/CUDACachingAllocator.cpp

// During graph capture:
void* ptr = allocator.allocate(size);
// Address baked into CUDA graph

// Pool ensures this address remains valid across replays
// by maintaining separate pool for graph-captured allocations
```

**Performance:** Graph replay is 5-10x faster than individual kernel launches for sequences of operations.

## Kernel-Level Optimizations

### CPU Vectorization

**Source:** [`aten/src/ATen/cpu/vec/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/cpu/vec)

PyTorch uses SIMD (Single Instruction, Multiple Data) for CPU operations:

```cpp
// Scalar version (slow)
void add_cpu(float* out, const float* a, const float* b, int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];  // One operation per iteration
  }
}

// Vectorized version (fast)
void add_cpu_vec(float* out, const float* a, const float* b, int64_t n) {
  using Vec = Vectorized<float>;  // 8 floats on AVX2, 16 on AVX512
  int64_t vec_size = Vec::size();

  // Vector loop (process 8/16 at once)
  for (int64_t i = 0; i < n - vec_size; i += vec_size) {
    Vec a_vec = Vec::loadu(a + i);
    Vec b_vec = Vec::loadu(b + i);
    Vec result = a_vec + b_vec;  // Single instruction, 8/16 additions
    result.store(out + i);
  }

  // Scalar tail (remaining elements)
  for (int64_t i = (n / vec_size) * vec_size; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}
```

**Performance:** 4-8x speedup on modern CPUs (depending on AVX2/AVX512 support).

### CUDA Kernel Optimization

**cuBLAS Integration:**

**Source:** [`aten/src/ATen/native/cuda/Blas.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/cuda/Blas.cpp)

For matrix multiplication, PyTorch delegates to highly optimized cuBLAS:

```cpp
void gemm_cuda(
    const Tensor& a,
    const Tensor& b,
    Tensor& out,
    float alpha = 1.0,
    float beta = 0.0) {

  // Call cuBLAS
  cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      &alpha,
      a.data_ptr(), CUDA_R_32F, lda,
      b.data_ptr(), CUDA_R_32F, ldb,
      &beta,
      out.data_ptr(), CUDA_R_32F, ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use Tensor Cores if available
  );
}
```

**Tensor Core usage:** On A100/H100 GPUs, cuBLAS automatically uses Tensor Cores for mixed-precision operations, providing 10-20x speedup over regular CUDA cores.

## Profiling and Benchmarking

### PyTorch Profiler

**Source:** [`torch/profiler/profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/profiler.py)

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = MyModel().cuda()
inputs = torch.randn(32, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True
) as prof:
    output = model(inputs)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to TensorBoard
prof.export_chrome_trace("trace.json")
```

**Output example:**
```
---------------------------------  ------------  ------------
Name                               CPU time      CUDA time
---------------------------------  ------------  ------------
aten::conv2d                       5.123 ms      15.234 ms
aten::batch_norm                   2.341 ms      8.123 ms
aten::relu                         0.523 ms      1.234 ms
aten::linear                       3.234 ms      12.345 ms
cudaLaunchKernel                   8.234 ms      0.000 ms
---------------------------------  ------------  ------------
```

### Memory Profiler

**Source:** [`torch/profiler/_memory_profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/_memory_profiler.py)

```python
with torch.profiler.profile(
    profile_memory=True,
    record_shapes=True
) as prof:
    output = model(inputs)

# Memory timeline
print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))

# Peak memory
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Benchmark Utilities

**Source:** [`torch/utils/benchmark/utils/timer.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/utils/benchmark/utils/timer.py)

```python
import torch.utils.benchmark as benchmark

def benchmark_function():
    x = torch.randn(1000, 1000, device='cuda')
    return torch.matmul(x, x)

# Warm-up run
benchmark_function()

# Benchmark
timer = benchmark.Timer(
    stmt='benchmark_function()',
    setup='from __main__ import benchmark_function',
    globals={'torch': torch}
)

result = timer.blocked_autorange(min_run_time=1.0)
print(result)
# Output:
#   benchmark_function()
#   Median: 1.23 ms
#   IQR: 0.01 ms (1.22 to 1.24 ms)
```

## Performance Pitfalls and Solutions

### 1. Unnecessary CPU-GPU Synchronization

**Pitfall:**
```python
x = torch.randn(1000, device='cuda')
y = x + 1
print(y.sum().item())  # .item() forces synchronization!
# All CUDA operations must complete before print
```

**Solution:**
```python
# Batch synchronizations
results = []
for i in range(100):
    x = torch.randn(1000, device='cuda')
    y = x + 1
    results.append(y.sum())

# Single synchronization at end
final = torch.stack(results).cpu()  # One sync instead of 100
```

### 2. Inefficient Memory Access Patterns

**Pitfall:**
```python
# Non-contiguous tensor from transpose
x = torch.randn(1000, 1000).t()  # Transpose changes strides, not data
y = some_kernel(x)  # Kernel assumes contiguous layout → slow
```

**Solution:**
```python
x = torch.randn(1000, 1000).t().contiguous()  # Make contiguous
y = some_kernel(x)  # Fast access pattern
```

### 3. Too Many Small Kernels

**Pitfall:**
```python
# Unfused operations
def forward(x):
    x = x + 1     # Kernel 1
    x = x * 2     # Kernel 2
    x = x - 3     # Kernel 3
    return x
# 3 kernel launches, 6 memory operations
```

**Solution:**
```python
# Use TorchScript to enable fusion
@torch.jit.script
def forward(x):
    x = x + 1
    x = x * 2
    x = x - 3
    return x
# 1 fused kernel, 2 memory operations
```

### 4. Inefficient Optimizer Updates

**Pitfall:**
```python
# Standard optimizer (many kernel launches)
optimizer = torch.optim.Adam(model.parameters())
```

**Solution:**
```python
# Fused optimizer (single kernel)
optimizer = torch.optim.AdamW(model.parameters(), fused=True)
# Requires PyTorch 2.0+
```

## Key Takeaways

1. **Multi-level optimization:** Performance comes from optimization at every level—graph, operator, kernel, and hardware.

2. **JIT compilation is powerful:** TorchScript enables sophisticated graph optimizations like fusion and constant folding.

3. **Memory management matters:** The caching allocator eliminates allocation overhead, which would otherwise dominate small operations.

4. **Fusion reduces overhead:** Operator fusion dramatically improves performance by reducing kernel launches and memory bandwidth.

5. **Profile before optimizing:** Use PyTorch Profiler to identify actual bottlenecks rather than optimizing speculatively.

6. **Avoid synchronization:** CPU-GPU synchronization (`.item()`, `.numpy()`) should be minimized and batched when possible.

## Optimization Checklist

When optimizing PyTorch code:

- [ ] Use TorchScript (`torch.jit.script`) for production models
- [ ] Enable fused optimizers (`fused=True` in AdamW, etc.)
- [ ] Minimize `.item()`, `.numpy()` calls
- [ ] Use `.contiguous()` before passing to custom kernels
- [ ] Profile with `torch.profiler` to find bottlenecks
- [ ] Use channels-last memory format for CNNs
- [ ] Enable AMP (Automatic Mixed Precision) for Tensor Core usage
- [ ] Batch operations instead of loops when possible
- [ ] Use `torch.compile()` (PyTorch 2.0+) for automatic optimization

## What's Next

This concludes our blog series on PyTorch internals. You now understand:
- PyTorch's layered architecture and core abstractions
- How tensor operations are dispatched and executed
- Testing, documentation, and code quality practices
- Extension mechanisms for custom operators and backends
- Performance optimization strategies at every level

Armed with this knowledge, you're prepared to contribute to PyTorch, build custom extensions, optimize your own workloads, or design similar systems.

## References

- **Graph Executor:** [`torch/csrc/jit/runtime/graph_executor.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/runtime/graph_executor.cpp)
- **TensorExpr Fuser:** [`torch/csrc/jit/passes/tensorexpr_fuser.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/passes/tensorexpr_fuser.cpp)
- **Operator Fusion Passes:** [`torch/csrc/jit/passes/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/jit/passes)
- **CUDA Allocator:** [`c10/cuda/CUDACachingAllocator.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/c10/cuda/CUDACachingAllocator.cpp)
- **Fused Optimizers:** [`aten/src/ATen/native/FusedAdam.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/FusedAdam.cpp)
- **CPU Vectorization:** [`aten/src/ATen/cpu/vec/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/cpu/vec)
- **Profiler:** [`torch/profiler/profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/profiler.py)

---

*This analysis is based on PyTorch commit [`5d99a79`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984).*
