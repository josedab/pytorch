# Extending and Integrating PyTorch

**Analysis based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Introduction

One of PyTorch's greatest strengths is its extensibility. Whether you need a custom operator for a novel algorithm, want to integrate a specialized library, or need to add support for new hardware, PyTorch provides multiple extension points designed for these use cases.

In this post, we'll explore PyTorch's extension mechanisms from both Python and C++, examine real-world integration patterns, and walk through complete examples of custom operators with autograd support.

## Objectives

By the end of this post, you'll understand:

1. How to write custom autograd functions in Python
2. How to create custom C++ operators and register them with the dispatcher
3. How to add CUDA kernels for custom operations
4. PyTorch's extension build system (torch.utils.cpp_extension)
5. Integration patterns for external libraries
6. How to add support for new backends

## Python Extension: Custom Autograd Functions

The simplest extension mechanism is `torch.autograd.Function`, which allows you to implement custom forward and backward passes in Python.

### Basic Example: Custom ReLU

```python
import torch
from torch.autograd import Function

class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        # Save for backward
        ctx.save_for_backward(input)
        # Compute forward
        output = input.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        # Compute gradient
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage:
custom_relu = CustomReLU.apply

x = torch.randn(10, requires_grad=True)
y = custom_relu(x)
y.sum().backward()
print(x.grad)  # Gradient computed via custom backward
```

**Key Points:**
- **Static methods:** `forward` and `backward` must be `@staticmethod`
- **ctx object:** Context for saving information between forward and backward
- **Return values:** `backward` must return one gradient per input argument
- **.apply():** Creates a callable that integrates with autograd

### Advanced: Multiple Inputs and Outputs

```python
class CustomLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Save all inputs needed for backward
        ctx.save_for_backward(input, weight, bias)

        # Compute output
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors

        # Compute gradients
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

# Usage:
linear = CustomLinear.apply
```

**ctx.needs_input_grad:** Check if gradient is required before computing (optimization).

### Non-Differentiable Operations

Some operations don't have meaningful gradients:

```python
class Quantize(Function):
    @staticmethod
    def forward(ctx, input, num_bits=8):
        # Quantize to num_bits
        scale = (2 ** num_bits - 1) / (input.max() - input.min())
        zero_point = -input.min() * scale
        quantized = torch.round(input * scale + zero_point).clamp(0, 2**num_bits - 1)

        # Dequantize
        output = (quantized - zero_point) / scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient as-is
        return grad_output, None  # None for num_bits (not differentiable)
```

## C++ Extensions: Performance and Flexibility

For performance-critical operations, you can write custom operators in C++ and CUDA.

### Example: Custom CUDA Operator

Let's implement a custom element-wise operation in CUDA.

**custom_ops.cu:**
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void my_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const scalar_t alpha,
    const int64_t n) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + alpha * b[idx];
  }
}

torch::Tensor my_add_cuda(torch::Tensor a, torch::Tensor b, double alpha) {
  // Validate inputs
  TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

  // Allocate output
  auto out = torch::empty_like(a);

  const int64_t n = a.numel();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // Launch kernel with type dispatch
  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "my_add_cuda", ([&] {
    my_add_kernel<scalar_t><<<blocks, threads>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        static_cast<scalar_t>(alpha),
        n
    );
  }));

  return out;
}
```

**custom_ops.cpp (CPU fallback):**
```cpp
#include <torch/extension.h>

torch::Tensor my_add_cpu(torch::Tensor a, torch::Tensor b, double alpha) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return a + alpha * b;
}

// Dispatcher registration
TORCH_LIBRARY(my_ops, m) {
  m.def("my_add(Tensor a, Tensor b, float alpha=1.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("my_add", &my_add_cpu);
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("my_add", &my_add_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Python bindings (optional, for direct call)
  m.def("my_add", &my_add_cpu, "My custom add (CPU)");
  m.def("my_add_cuda", &my_add_cuda, "My custom add (CUDA)");
}
```

**setup.py:**
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_ops',
    ext_modules=[
        CUDAExtension(
            name='my_ops',
            sources=['custom_ops.cpp', 'custom_ops.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**Build and use:**
```bash
python setup.py install
```

```python
import torch
import my_ops

a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')

# Call via dispatcher (recommended)
result = torch.ops.my_ops.my_add(a, b, 2.0)

# Or call directly (bypasses dispatcher)
result = my_ops.my_add_cuda(a, b, 2.0)
```

See documentation: [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

## Operator Registration with Modern API

PyTorch's modern operator registration API (`TORCH_LIBRARY`) provides first-class dispatcher integration.

### Three-Part Registration

**1. Define operator schema:**
```cpp
TORCH_LIBRARY(myns, m) {
  // Schema defines signature
  m.def("my_op(Tensor input, int param) -> Tensor");

  // Can also specify default values
  m.def("my_op_v2(Tensor input, int param=10) -> Tensor");
}
```

**2. Implement for each backend:**
```cpp
// CPU implementation
TORCH_LIBRARY_IMPL(myns, CPU, m) {
  m.impl("my_op", [](const at::Tensor& input, int64_t param) {
    // CPU kernel
    return input * param;
  });
}

// CUDA implementation
TORCH_LIBRARY_IMPL(myns, CUDA, m) {
  m.impl("my_op", [](const at::Tensor& input, int64_t param) {
    // CUDA kernel
    return input.mul(param);  // Uses existing CUDA ops
  });
}

// Composite implementation (fallback for all devices)
TORCH_LIBRARY_IMPL(myns, CompositeImplicitAutograd, m) {
  m.impl("my_op", [](const at::Tensor& input, int64_t param) {
    // Compose from existing ops (works everywhere)
    return input * param;
  });
}
```

**3. Autograd support (optional):**
```cpp
TORCH_LIBRARY_IMPL(myns, Autograd, m) {
  m.impl("my_op", [](const at::Tensor& input, int64_t param) {
    // Forward pass
    auto result = at::redispatch::my_op(
        c10::DispatchKeySet(c10::DispatchKey::AutogradCPU).remove(c10::DispatchKey::Autograd),
        input,
        param
    );

    // Setup autograd
    auto grad_fn = std::make_shared<MyOpBackward>();
    grad_fn->set_next_edges(collect_next_edges(input));
    grad_fn->input_ = input;
    grad_fn->param_ = param;

    result.set_requires_grad(input.requires_grad());
    result.set_grad_fn(grad_fn);

    return result;
  });
}
```

See the registration guide: [`aten/src/ATen/core/op_registration/README.md`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/core/op_registration/README.md)

## Adding Custom Autograd in C++

For C++ operators, autograd support requires defining backward functions.

### Using torch::autograd::Function

```cpp
#include <torch/extension.h>

class MyMulFunction : public torch::autograd::Function<MyMulFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input,
      double alpha) {
    ctx->save_for_backward({input});
    ctx->saved_data["alpha"] = alpha;
    return input * alpha;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto alpha = ctx->saved_data["alpha"].toDouble();

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output * alpha;

    return {grad_input, torch::Tensor()};  // None for alpha
  }
};

// Python binding
torch::Tensor my_mul(torch::Tensor input, double alpha) {
  return MyMulFunction::apply(input, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_mul", &my_mul, "Multiply with custom autograd");
}
```

## Integration with External Libraries

PyTorch can integrate with external libraries through operator registration.

### Example: Integrating cuDNN

PyTorch's convolution wraps cuDNN:

**Source:** [`aten/src/ATen/native/cudnn/Conv.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/cudnn/Conv.cpp)

```cpp
#include <cudnn.h>

at::Tensor cudnn_convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  // Create cuDNN descriptors
  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensorDescriptor_t output_desc;

  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnCreateTensorDescriptor(&output_desc);

  // Set descriptor parameters
  // ... (configuration code)

  // Find best algorithm
  cudnnConvolutionFwdAlgo_t algo;
  cudnnGetConvolutionForwardAlgorithm_v7(
      cudnn_handle, input_desc, filter_desc, conv_desc, output_desc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

  // Allocate workspace
  size_t workspace_size;
  cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, input_desc, filter_desc, conv_desc, output_desc,
      algo, &workspace_size);

  auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, input.options().dtype(at::kByte));

  // Run convolution
  cudnnConvolutionForward(
      cudnn_handle,
      &alpha,
      input_desc, input.data_ptr(),
      filter_desc, weight.data_ptr(),
      conv_desc,
      algo,
      workspace.data_ptr(), workspace_size,
      &beta,
      output_desc, output.data_ptr()
  );

  // Cleanup descriptors
  // ... (cleanup code)

  return output;
}
```

**Key integration points:**
1. Wraps cuDNN API in PyTorch tensor interface
2. Manages cuDNN descriptors and workspace
3. Handles memory layout conversion (if needed)
4. Registers with dispatcher for automatic backend selection

### Example: Integrating Eigen

For CPU operations, PyTorch can use Eigen:

```cpp
#include <Eigen/Dense>
#include <torch/extension.h>

torch::Tensor eigen_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(0));

  // Map PyTorch tensors to Eigen matrices
  auto a_eigen = Eigen::Map<const Eigen::MatrixXf>(
      a.data_ptr<float>(),
      a.size(0),
      a.size(1)
  );

  auto b_eigen = Eigen::Map<const Eigen::MatrixXf>(
      b.data_ptr<float>(),
      b.size(0),
      b.size(1)
  );

  // Allocate output
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());

  // Map output tensor
  auto out_eigen = Eigen::Map<Eigen::MatrixXf>(
      out.data_ptr<float>(),
      out.size(0),
      out.size(1)
  );

  // Perform matrix multiplication with Eigen
  out_eigen = a_eigen * b_eigen;

  return out;
}
```

**Benefits:**
- Leverage optimized library implementations
- Maintain PyTorch tensor interface
- Automatic memory management

## Adding New Backend Support

PyTorch's dispatcher makes adding new backends straightforward.

### Example: Simplified Backend Registration

**1. Define backend dispatch key:**
```cpp
// c10/core/DispatchKey.h (add to enum)
enum class DispatchKey : uint16_t {
  // ... existing keys ...
  MyNewBackend,  // Your custom backend
};
```

**2. Implement operators for your backend:**
```cpp
// my_backend_ops.cpp
namespace at {
namespace native {

Tensor add_mybackend(const Tensor& a, const Tensor& b, const Scalar& alpha) {
  // Implementation using your backend's API
  return my_backend::add(a, b, alpha);
}

Tensor mul_mybackend(const Tensor& a, const Tensor& b) {
  return my_backend::mul(a, b);
}

// ... more operators ...

}}  // namespace at::native
```

**3. Register operators with dispatcher:**
```cpp
TORCH_LIBRARY_IMPL(aten, MyNewBackend, m) {
  m.impl("add.Tensor", TORCH_FN(add_mybackend));
  m.impl("mul.Tensor", TORCH_FN(mul_mybackend));
  // ... register all supported operators ...
}
```

**4. Create device type:**
```cpp
// c10/core/DeviceType.h (add to enum)
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
  // ... existing types ...
  MyBackend = 15,  // Your device type
};
```

**5. Python integration:**
```python
# torch/my_backend/__init__.py
import torch

# Register device type
torch._C._register_device_type('mybackend', 15)

# Create device
device = torch.device('mybackend')

# Now users can use:
x = torch.randn(10, device='mybackend')
y = torch.randn(10, device='mybackend')
z = x + y  # Dispatches to add_mybackend
```

**Real-world examples in PyTorch:**
- **MPS (Metal):** Apple Silicon GPU support
- **XPU:** Intel GPU support
- **HPU:** Habana Gaudi support
- **IPU:** Graphcore IPU support

See MPS implementation: [`aten/src/ATen/mps/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/mps)

## Common Integration Patterns

### 1. Wrapping External Libraries

```
PyTorch Tensor
    ↓
Convert to library format (if needed)
    ↓
Call library function
    ↓
Convert result back to Tensor
    ↓
Return
```

**Example:**
```cpp
torch::Tensor fftw_fft(torch::Tensor input) {
  // Convert to FFTW format
  auto* fftw_input = reinterpret_cast<fftw_complex*>(input.data_ptr<c10::complex<float>>());

  // Allocate output
  auto output = torch::empty_like(input);
  auto* fftw_output = reinterpret_cast<fftw_complex*>(output.data_ptr<c10::complex<float>>());

  // Plan and execute FFT
  fftw_plan plan = fftw_plan_dft_1d(input.size(0), fftw_input, fftw_output, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  return output;
}
```

### 2. Custom Memory Allocators

```cpp
// Custom allocator for special memory (e.g., pinned, shared)
struct MyAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = my_custom_malloc(nbytes);
    return {data, data, &my_custom_free, at::Device(at::kCPU)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &my_custom_free;
  }
};

// Use custom allocator
auto my_allocator = std::make_shared<MyAllocator>();
auto options = torch::TensorOptions().device(torch::kCPU).allocator(my_allocator);
auto tensor = torch::empty({100}, options);
```

### 3. Custom Tensor Subclasses (Python)

```python
class MyCustomTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, metadata):
        # Create underlying tensor
        tensor = torch.Tensor._make_subclass(cls, data)
        # Attach custom metadata
        tensor.metadata = metadata
        return tensor

    def custom_method(self):
        return self.metadata

# Usage:
x = MyCustomTensor(torch.randn(10), metadata={"name": "sensor_data"})
print(x.custom_method())  # Access custom metadata
y = x + 5  # Still works as a normal tensor
```

## Key Takeaways

1. **Multiple extension points:** Python autograd functions for prototyping, C++ for performance, full dispatcher integration for backends.

2. **torch.utils.cpp_extension:** Makes building C++/CUDA extensions easy with automatic compilation and integration.

3. **TORCH_LIBRARY API:** Modern, clean API for operator registration that integrates fully with the dispatcher.

4. **Backend integration is standardized:** Adding a new device type follows a well-defined pattern.

5. **External libraries integrate cleanly:** Wrap library calls in PyTorch operators while maintaining tensor interface and autograd support.

6. **Composition over duplication:** Use CompositeImplicitAutograd to define operations in terms of existing ops when possible.

## What's Next

In the next post, we'll explore performance optimization:
- JIT compilation and TorchScript
- Graph optimization passes
- Operator fusion strategies
- Memory optimization techniques
- Profiling and benchmarking tools

## References

- **Custom Autograd:** [`torch/autograd/function.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/autograd/function.py)
- **C++ Extension API:** [`torch/csrc/utils/python_arg_parser.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/csrc/utils/python_arg_parser.h)
- **Operator Registration:** [`aten/src/ATen/core/op_registration/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/core/op_registration)
- **TORCH_LIBRARY Docs:** [`aten/src/ATen/core/op_registration/README.md`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/core/op_registration/README.md)
- **MPS Backend:** [`aten/src/ATen/mps/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/mps)
- **cuDNN Integration:** [`aten/src/ATen/native/cudnn/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/cudnn)
- **Extension Tutorial:** https://pytorch.org/tutorials/advanced/cpp_extension.html

---

*This analysis is based on PyTorch commit [`5d99a79`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984).*
