# Additional High-Impact RFC Proposals for PyTorch

**Date:** 2025-11-16
**Status:** Concept Review

After comprehensive analysis of the PyTorch codebase, the following high-impact improvements have been identified beyond the initial 4 RFCs. These address critical pain points in developer experience, production deployment, and scalability.

---

## RFC-0005: Enhanced Error Messages and Stack Trace System

**Problem Severity:** 🔴 CRITICAL - Affects every developer daily

### Current Pain Points

1. **Cryptic Python/C++ Boundary Errors**
   ```python
   # Current error (unhelpful):
   RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

   # Where did this come from? Which tensor? Which operation?
   # Stack trace shows PyBind11 internals, not user code
   ```

2. **Lost Context in Autograd**
   ```python
   # Error during backward:
   RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

   # Which variable? Where was it modified? What operation?
   ```

3. **Dispatcher Errors Lack Context**
   ```python
   # Current:
   RuntimeError: Could not run 'aten::add' with arguments from the 'CUDA' backend

   # No indication of: dtype mismatch? shape issue? actual problem?
   ```

### Proposed Solution

**Component 1: Rich Error Context System**
```cpp
// New error framework
class TorchError {
    std::string message;
    std::vector<ContextFrame> context;
    SourceLocation source_location;
    std::vector<TensorDebugInfo> tensor_info;

    std::string format_for_user() {
        // Beautiful, actionable error messages
    }
};

struct TensorDebugInfo {
    std::string name;  // Variable name from Python
    Device device;
    ScalarType dtype;
    IntArrayRef sizes;
    IntArrayRef strides;
    std::string creation_stack;  // Where was this tensor created?
};
```

**Component 2: Operation Recording**
```python
# Enable operation recording (small overhead)
torch.set_debug_mode(True)

x = torch.randn(10, device='cuda', name='x')  # Track names
y = torch.randn(10, device='cpu', name='y')
z = x + y  # Error with full context!

# Enhanced error:
"""
RuntimeError: Device mismatch in binary operation

  Operation: torch.add
  Location: script.py:45 in forward()

  Operands:
    - x: Tensor(shape=[10], dtype=float32, device=cuda:0)
      Created at: script.py:32
    - y: Tensor(shape=[10], dtype=float32, device=cpu)
      Created at: script.py:33

  Suggestion: Move tensor 'y' to CUDA with: y = y.cuda()
"""
```

**Component 3: Autograd Graph Debugger**
```python
# Visualize autograd graph at error point
torch.autograd.set_detect_anomaly(True, graph_on_error=True)

loss.backward()
# On error, shows:
# 1. Full computation graph leading to error
# 2. Which tensor was modified
# 3. Where modification happened
# 4. Suggests fix
```

### Expected Impact

- **Developer Productivity:** 30-50% reduction in debugging time
- **Onboarding:** 40% faster time-to-productivity for new users
- **Support Burden:** 25% reduction in forum questions about errors
- **User Satisfaction:** Significant improvement (measured via surveys)

**Estimated Timeline:** 6 months (2 months design, 3 months implementation, 1 month rollout)

---

## RFC-0006: Operator Versioning and Model Compatibility System

**Problem Severity:** 🔴 CRITICAL - Production deployment blocker

### Current Pain Points

1. **Silent Behavior Changes**
   ```python
   # PyTorch 1.x:
   torch.div(5, 2)  # Returns 2 (integer division)

   # PyTorch 2.x:
   torch.div(5, 2)  # Returns 2.5 (true division)

   # Saved models break silently!
   ```

2. **No Forward Compatibility**
   ```python
   # Save model with PyTorch 2.2
   torch.save(model, 'model.pt')

   # Load with PyTorch 2.1 - crashes or wrong behavior
   # No version checking, no graceful degradation
   ```

3. **Operator Deprecation Breaks Models**
   ```python
   # Old model uses deprecated operator
   # New PyTorch removes it
   # Model won't load - no migration path
   ```

### Proposed Solution

**Component 1: Operator Versioning**
```yaml
# native_functions.yaml (enhanced)
- func: div.Tensor(Tensor self, Tensor other) -> Tensor
  versions:
    - version: 1
      behavior: floor_divide
      deprecated_in: 2.0
    - version: 2
      behavior: true_divide
      default_since: 2.0
  migration:
    from_v1: "Use torch.div(..., rounding_mode='floor') for old behavior"
```

**Component 2: Model Compatibility Manifest**
```python
# Save model with compatibility info
torch.save({
    'model_state_dict': model.state_dict(),
    'pytorch_version': '2.2.0',
    'operator_versions': {
        'aten::div': 2,
        'aten::conv2d': 1,
        # ... all operators used
    },
    'compatibility_mode': 'strict'  # or 'warn' or 'best_effort'
}, 'model.pt')

# Load with automatic compatibility
checkpoint = torch.load('model.pt', compat_mode='auto')
# Automatically uses v1 div if model expects it
```

**Component 3: Version Shims**
```cpp
// Dispatcher routes to correct version
Tensor div_v1(const Tensor& self, const Tensor& other) {
    return at::floor_divide(self, other);
}

Tensor div_v2(const Tensor& self, const Tensor& other) {
    return at::true_divide(self, other);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    // Register both versions
    m.impl("div.v1", &div_v1);
    m.impl("div.v2", &div_v2);

    // Default to v2, but compatibility loader can override
}
```

### Expected Impact

- **Model Reliability:** 99%+ forward/backward compatibility
- **Production Safety:** Zero silent behavior changes
- **Migration Time:** 80% reduction in time to upgrade PyTorch versions
- **Enterprise Adoption:** Removes major barrier to PyTorch adoption

**Estimated Timeline:** 8 months (3 months design, 4 months implementation, 1 month migration)

---

## RFC-0007: Unified Quantization Infrastructure

**Problem Severity:** 🟠 HIGH - Deployment bottleneck

### Current Pain Points

1. **Fragmented APIs**
   ```python
   # 4+ different quantization APIs!
   # 1. Eager mode quantization
   torch.quantization.quantize_dynamic(model, ...)

   # 2. FX graph mode
   torch.quantization.quantize_fx.prepare_fx(model, ...)

   # 3. PT2E (PyTorch 2 Export)
   torch.ao.quantization.quantize_pt2e.prepare_pt2e(model, ...)

   # 4. Custom backend quantization
   # Each backend (NNAPI, CoreML, etc.) has own API

   # Which one to use? All have different limitations!
   ```

2. **Poor Calibration UX**
   ```python
   # Current: Manual calibration
   model.eval()
   model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
   torch.quantization.prepare(model, inplace=True)

   # Run calibration data (how much? user doesn't know!)
   for data in calibration_loader:
       model(data)

   torch.quantization.convert(model, inplace=True)
   # Did we calibrate enough? No feedback!
   ```

3. **No Accuracy/Speed Tradeoff Exploration**
   ```python
   # Users don't know:
   # - Which layers benefit from quantization?
   # - What's the accuracy vs speed tradeoff?
   # - Which quantization scheme (per-tensor, per-channel, dynamic)?
   ```

### Proposed Solution

**Component 1: Declarative Quantization Config**
```python
from torch.quantization.v2 import QuantizationConfig, QuantizationStrategy

config = QuantizationConfig(
    # Declarative, not imperative
    strategy=QuantizationStrategy.PTQ_STATIC,

    # Automatic layer selection (or manual override)
    layers='auto',  # Profile and choose optimal layers

    # Calibration
    calibration_method='minmax',  # or 'histogram', 'percentile'
    num_calibration_steps='auto',  # Automatic sufficiency detection

    # Constraints
    max_accuracy_drop=0.01,  # 1% max accuracy loss
    target_speedup=2.0,      # Want 2x speedup

    # Backend
    backend='fbgemm',  # or 'qnnpack', 'tensorrt', etc.
)

# One API for all quantization
quantized_model = torch.quantization.quantize(
    model,
    config=config,
    calibration_data=calibration_loader,
    validation_data=val_loader,  # For accuracy checking
)

# Automatic report
print(quantized_model.quantization_report)
"""
Quantization Report:
  - Accuracy: 98.5% → 98.4% (0.1% drop ✓)
  - Speed: 2.3x faster (target: 2.0x ✓)
  - Size: 120MB → 32MB (73% reduction)

  Quantized layers (12/45):
    - conv1: int8 per-channel (2.1x speedup, 0.05% accuracy impact)
    - conv2: int8 per-channel (2.4x speedup, 0.02% accuracy impact)
    - fc1: int8 per-tensor (1.8x speedup, 0.03% accuracy impact)
    ...

  Skipped layers (33/45):
    - batch_norm1: quantization unhelpful (0.9x speedup)
    - attention: high accuracy impact (2% drop)
    ...
"""
```

**Component 2: Automatic Layer Selection**
```python
class QuantizationOptimizer:
    """Automatically select which layers to quantize"""

    def optimize(self, model, constraints):
        # Profile each layer
        layer_profiles = self.profile_layers(model)

        # For each layer: measure
        # 1. Quantization speedup
        # 2. Accuracy impact
        # 3. Size reduction

        # Solve optimization problem:
        # Maximize: speedup
        # Subject to: accuracy_drop <= max_accuracy_drop

        # Return optimal quantization config
```

**Component 3: Backend Abstraction**
```python
# Same API, different backends
config = QuantizationConfig(backend='fbgemm')  # CPU x86
config = QuantizationConfig(backend='qnnpack')  # ARM
config = QuantizationConfig(backend='tensorrt')  # NVIDIA GPU
config = QuantizationConfig(backend='coreml')  # Apple Silicon

# Backend-specific optimizations automatic
```

### Expected Impact

- **Developer Time:** 70% reduction in quantization effort
- **Model Quality:** 50% fewer accuracy regressions (auto layer selection)
- **Deployment Speed:** 30% faster (better quantization configs)
- **Adoption:** 3x increase in quantization usage

**Estimated Timeline:** 10 months (4 months design, 5 months implementation, 1 month migration)

---

## RFC-0008: Distributed Training Observability and Debugging

**Problem Severity:** 🟠 HIGH - Multi-GPU training pain point

### Current Pain Points

1. **Opaque Communication Patterns**
   ```python
   # DDP training hangs - where? why?
   model = DDP(model)
   for data in dataloader:
       loss = model(data).sum()
       loss.backward()  # Hangs here - which rank? which all-reduce?
       optimizer.step()
   ```

2. **No Straggler Detection**
   ```python
   # Some GPUs are slow - which ones? why?
   # User has no visibility into per-rank performance
   ```

3. **Memory Across Ranks**
   ```python
   # OOM on rank 3, but not others - why?
   # No per-rank memory profiling
   ```

### Proposed Solution

**Component 1: Distributed Profiler**
```python
from torch.profiler.distributed import DistributedProfiler

with DistributedProfiler(
    ranks='all',  # Profile all ranks
    sync_events=True,  # Correlate events across ranks
) as prof:
    for batch in dataloader:
        output = model(batch)
        loss = output.sum()
        loss.backward()
        optimizer.step()

# Generates report
prof.export_traces('distributed_trace/')
"""
Distributed Training Profile:

Communication:
  - Total time in collectives: 23% (too high!)
  - all_reduce: 18.2s across 100 steps
  - all_gather: 4.1s

Stragglers detected:
  - Rank 3: 15% slower than median
    - Reason: CPU bottleneck in data loading
    - Suggestion: Increase num_workers

  - Rank 7: 8% slower than median
    - Reason: Memory transfer overhead
    - Suggestion: Increase pin_memory buffer

Memory by rank:
  - Rank 0: 22.1 GB
  - Rank 1: 22.0 GB
  - Rank 2: 22.2 GB
  - Rank 3: 31.4 GB (HIGH! ⚠️)
    - Cause: Gradient accumulation mismatch

Recommendations:
  1. Reduce bucket size (currently 25MB → try 10MB)
  2. Enable gradient compression
  3. Fix data loading on rank 3
"""
```

**Component 2: Real-time Distributed Dashboard**
```python
# Web-based dashboard (localhost:8888)
torch.distributed.launch_dashboard(port=8888)

# Shows:
# - Per-rank GPU utilization (live)
# - Communication timeline
# - Memory usage across ranks
# - Straggler detection
# - Gradient norm tracking
```

**Component 3: Hang Detection**
```python
from torch.distributed.debugging import enable_hang_detection

enable_hang_detection(timeout=30)  # Detect hangs after 30s

# If training hangs, automatically:
# 1. Identifies which collective operation
# 2. Which ranks are waiting
# 3. Which rank is the straggler
# 4. Stack trace of all ranks at hang time
```

### Expected Impact

- **Debug Time:** 60% reduction in distributed debugging time
- **Training Efficiency:** 20% improvement (straggler detection)
- **Hang Resolution:** 90% of hangs auto-diagnosed
- **Adoption:** Removes barrier to multi-GPU training

**Estimated Timeline:** 6 months (2 months design, 3 months implementation, 1 month rollout)

---

## Summary: Additional RFC Priorities

| RFC | Title | Impact | Complexity | Timeline | Priority |
|-----|-------|--------|------------|----------|----------|
| **RFC-0005** | Enhanced Error Messages | 🔴 Critical | Medium | 6 mo | **P0** |
| **RFC-0006** | Operator Versioning | 🔴 Critical | High | 8 mo | **P0** |
| **RFC-0007** | Unified Quantization | 🟠 High | High | 10 mo | **P1** |
| **RFC-0008** | Distributed Observability | 🟠 High | Medium | 6 mo | **P1** |

### Recommended Prioritization

**Phase 1 (Immediate):**
- RFC-0005: Enhanced Error Messages
  - Universal pain point
  - Improves every developer's experience
  - Medium complexity, high ROI

**Phase 2 (Short-term):**
- RFC-0006: Operator Versioning
  - Critical for production users
  - Enables safe PyTorch upgrades
  - Blocks enterprise adoption

- RFC-0008: Distributed Observability
  - High impact for scaling
  - Relatively easier than quantization
  - Unlocks multi-GPU debugging

**Phase 3 (Medium-term):**
- RFC-0007: Unified Quantization
  - Deployment critical
  - Requires significant design work
  - Can build on versioning system (RFC-0006)

---

## Additional Ideas for Future Consideration

**Lower Priority but Still Valuable:**

1. **RFC-0009: Symbolic Shapes and Dynamic Compilation**
   - Better torch.compile support for dynamic shapes
   - Symbolic shape inference system
   - Impact: High | Complexity: Very High | Timeline: 12+ mo

2. **RFC-0010: Automatic Gradient Checkpointing**
   - Compiler-driven activation checkpointing
   - No manual annotation required
   - Impact: Medium | Complexity: High | Timeline: 8 mo

3. **RFC-0011: Build System Modernization**
   - Migrate from CMake to Bazel/Buck2
   - Hermetic builds, better caching
   - Impact: Medium | Complexity: Very High | Timeline: 12+ mo

4. **RFC-0012: First-Class Sparsity Infrastructure**
   - Unified sparse tensor support
   - Sparse-dense operators
   - Impact: Medium | Complexity: High | Timeline: 10 mo

5. **RFC-0013: Model Hub Integration**
   - First-party model zoo with versioning
   - Easy fine-tuning and deployment
   - Impact: Medium | Complexity: Medium | Timeline: 6 mo

---

**Recommendation:** Draft detailed RFCs for the P0 items (0005, 0006) first, as they address the most critical pain points with reasonable implementation complexity.
