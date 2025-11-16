# Enhanced Memory Profiler Examples

This directory contains examples for using the Enhanced Memory Profiler, implemented as part of RFC-0003: Enhanced Memory Profiling and Optimization.

## Features

The Enhanced Memory Profiler provides:

1. **Real-time Memory Tracking**: Track allocations with categorization by tensor type
2. **Fragmentation Analysis**: Analyze CUDA memory fragmentation
3. **Memory Leak Detection**: Automatically detect memory leaks in training loops
4. **Optimization Recommendations**: Get actionable suggestions for reducing memory usage
5. **Advanced Visualization**: Generate timeline plots, category breakdowns, and flamegraphs

## Quick Start

```python
from torch.profiler.memory import EnhancedMemoryProfiler

# Create profiler
profiler = EnhancedMemoryProfiler(
    track_stacks=True,
    categorize_allocations=True,
    detect_leaks=True
)

# Use with context manager
with profiler:
    # Your training code here
    pass

# Get memory breakdown
breakdown = profiler.get_memory_breakdown()
print(breakdown)

# Get optimization recommendations
recommendations = profiler.get_optimization_recommendations(
    model=your_model,
    batch_size=your_batch_size
)
```

## Examples

### Basic Memory Profiling

See `enhanced_memory_profiler_example.py` for detailed examples including:

- **Example 1**: Basic memory profiling during training
- **Example 2**: Memory fragmentation analysis
- **Example 3**: Memory leak detection
- **Example 4**: Getting optimization recommendations
- **Example 5**: Visualizing memory usage

### Running the Examples

```bash
python enhanced_memory_profiler_example.py
```

## API Reference

### EnhancedMemoryProfiler

Main profiler class for tracking memory usage.

**Parameters:**
- `track_stacks` (bool): Whether to capture stack traces for allocations
- `categorize_allocations` (bool): Whether to categorize allocations by type
- `detect_leaks` (bool): Whether to enable memory leak detection
- `device` (torch.device): Device to profile (defaults to current CUDA device)

**Methods:**
- `get_memory_breakdown()`: Get current memory usage by category
- `get_peak_allocation()`: Find allocation that caused peak memory
- `get_current_memory()`: Get current total memory allocated
- `get_optimization_recommendations(model, batch_size)`: Generate optimization recommendations

### FragmentationAnalyzer

Analyzes memory fragmentation in the CUDA caching allocator.

**Methods:**
- `analyze_fragmentation(device)`: Analyze fragmentation on a specific device

### MemoryLeakDetector

Detects memory leaks in training loops.

**Parameters:**
- `window_size` (int): Number of iterations to analyze for trends

**Methods:**
- `record_iteration(memory_used)`: Record memory usage after each iteration

### MemoryOptimizationEngine

Generates optimization recommendations based on profiling results.

**Parameters:**
- `model` (nn.Module): PyTorch model to analyze
- `batch_size` (int): Current batch size
- `device` (torch.device): Device being used

**Methods:**
- `analyze_and_recommend(profiler_results)`: Analyze and generate recommendations

### MemoryVisualizer

Visualizes memory profiling results.

**Methods:**
- `plot_memory_timeline(profiler)`: Create memory timeline plot
- `plot_category_breakdown(profiler)`: Create pie chart of memory by category
- `generate_flamegraph(profiler)`: Generate flamegraph data

## Tensor Categories

The profiler categorizes allocations into:

- `PARAMETER`: Model parameters
- `GRADIENT`: Gradients
- `ACTIVATION`: Forward pass activations
- `OPTIMIZER_STATE`: Optimizer state (momentum, etc.)
- `TEMPORARY`: Short-lived temporary tensors
- `PERSISTENT`: User-managed persistent tensors
- `UNKNOWN`: Uncategorized allocations

## Optimization Recommendations

The profiler can recommend:

1. **Activation Checkpointing**: When activations use significant memory
2. **Gradient Accumulation**: When total memory usage is high
3. **Mixed Precision Training**: When compatible hardware is available

Each recommendation includes:
- Priority level (high, medium, low)
- Detailed message explaining the issue
- Code example showing how to implement the optimization
- Expected memory savings

## Requirements

- PyTorch (with CUDA for GPU features)
- matplotlib (optional, for visualization)

## References

- RFC-0003: Enhanced Memory Profiling and Optimization
- PyTorch Profiler Documentation: https://pytorch.org/docs/stable/profiler.html
