# RFC-0003: Enhanced Memory Profiling and Optimization

**Status:** Draft
**Author:** Claude Code Analysis
**Created:** 2025-11-16
**Based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Summary

Enhance PyTorch's memory profiling infrastructure to provide actionable insights for memory optimization, including real-time memory tracking, automatic optimization recommendations, memory leak detection, and integration with the caching allocator for detailed fragmentation analysis.

## Motivation

### Current Memory Profiling Limitations

The existing memory profiler ([`torch/profiler/_memory_profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/_memory_profiler.py)) provides basic memory tracking but has significant gaps:

1. **Limited Visibility:**
   - Shows peak memory, but not *when* or *why* it occurred
   - No breakdown by tensor category (parameters, activations, gradients, optimizer states)
   - Doesn't track memory fragmentation in caching allocator

2. **Reactive, Not Proactive:**
   - Requires OOM to occur before investigation
   - No warnings before hitting memory limits
   - No automatic optimization suggestions

3. **Poor Integration:**
   - Separate from performance profiler
   - Doesn't correlate memory with specific operators
   - No visualization of memory timeline

4. **Missing Features:**
   - No memory leak detection
   - Can't identify unused but allocated memory
   - No tracking of cross-device transfers
   - Doesn't track memory in distributed training

### Real-World Impact

**User pain points:**
- "My model OOMs on GPU but I don't know why"
- "Memory usage keeps growing during training"
- "I can only fit batch size 8, but expected 32"
- "Memory fragmentation reduces effective capacity by 30%"

**Common causes:**
- Gradients not released (missing optimizer.zero_grad())
- Activation checkpointing not enabled
- Too many in-flight operations
- Excessive memory fragmentation
- Holding references to tensors

## Detailed Design

### Enhanced Memory Profiler Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Profiler V2                       │
├─────────────────────────────────────────────────────────────┤
│  Real-Time Tracking                                         │
│  ├─ Allocation/deallocation hooks                           │
│  ├─ Category tracking (param, grad, activation, etc.)      │
│  ├─ Stack trace capture                                     │
│  └─ Timeline tracking                                       │
├─────────────────────────────────────────────────────────────┤
│  Fragmentation Analysis                                     │
│  ├─ Integration with CUDACachingAllocator                   │
│  ├─ Free block analysis                                     │
│  ├─ Contiguous memory availability                          │
│  └─ Defragmentation recommendations                         │
├─────────────────────────────────────────────────────────────┤
│  Leak Detection                                             │
│  ├─ Reference counting analysis                             │
│  ├─ Unexpected retention detection                          │
│  └─ Growth pattern identification                           │
├─────────────────────────────────────────────────────────────┤
│  Optimization Engine                                        │
│  ├─ Activation checkpointing suggestions                    │
│  ├─ Gradient accumulation recommendations                   │
│  ├─ Batch size optimization                                 │
│  └─ Model partitioning suggestions                          │
├─────────────────────────────────────────────────────────────┤
│  Visualization                                              │
│  ├─ Memory timeline graphs                                  │
│  ├─ Allocation flamegraphs                                  │
│  ├─ Fragmentation heatmaps                                  │
│  └─ TensorBoard integration                                 │
└─────────────────────────────────────────────────────────────┘
```

### Component 1: Real-Time Memory Tracking

**Location:** Enhancement to [`torch/profiler/_memory_profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/_memory_profiler.py)

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import traceback

class TensorCategory(Enum):
    PARAMETER = "parameter"          # Model parameters
    GRADIENT = "gradient"            # Gradients
    ACTIVATION = "activation"        # Forward pass activations
    OPTIMIZER_STATE = "optimizer"    # Optimizer momentum, etc.
    TEMPORARY = "temporary"          # Short-lived temps
    PERSISTENT = "persistent"        # User-managed persistent tensors
    UNKNOWN = "unknown"

@dataclass
class AllocationEvent:
    """Detailed allocation event"""
    ptr: int                         # Memory address
    size: int                        # Bytes allocated
    device: torch.device
    category: TensorCategory
    timestamp: float                 # Monotonic time
    stack_trace: List[str]           # Allocation stack
    stream: int                      # CUDA stream ID
    tensor_id: Optional[int]         # Tensor object ID

class EnhancedMemoryProfiler:
    def __init__(self,
                 track_stacks: bool = True,
                 categorize_allocations: bool = True,
                 detect_leaks: bool = True):
        self.allocations = {}        # ptr -> AllocationEvent
        self.timeline = []           # List of (time, memory_used)
        self.peak_memory = 0
        self.peak_time = 0
        self.category_stats = defaultdict(int)

    def on_allocation(self, ptr: int, size: int, device: torch.device, stream: int):
        """Hook called on memory allocation"""
        # Categorize allocation
        category = self._categorize_allocation()

        # Capture stack trace
        stack = traceback.extract_stack() if self.track_stacks else []

        # Record event
        event = AllocationEvent(
            ptr=ptr,
            size=size,
            device=device,
            category=category,
            timestamp=time.monotonic(),
            stack_trace=[str(frame) for frame in stack],
            stream=stream,
            tensor_id=None  # Set by tensor creation hook
        )

        self.allocations[ptr] = event
        self.category_stats[category] += size

        # Update timeline
        current_memory = sum(e.size for e in self.allocations.values())
        self.timeline.append((event.timestamp, current_memory))

        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            self.peak_time = event.timestamp

    def on_deallocation(self, ptr: int):
        """Hook called on memory deallocation"""
        if ptr in self.allocations:
            event = self.allocations[ptr]
            self.category_stats[event.category] -= event.size
            del self.allocations[ptr]

    def _categorize_allocation(self) -> TensorCategory:
        """Infer allocation category from context"""
        # Analyze stack trace to determine category
        stack = traceback.extract_stack()

        # Check for gradient allocation
        if any('autograd' in frame.filename for frame in stack):
            return TensorCategory.GRADIENT

        # Check for optimizer state
        if any('optim' in frame.filename for frame in stack):
            return TensorCategory.OPTIMIZER_STATE

        # Check for parameter allocation
        if any('nn/modules' in frame.filename for frame in stack):
            return TensorCategory.PARAMETER

        # Check for forward pass
        if any('forward' in frame.name for frame in stack):
            return TensorCategory.ACTIVATION

        return TensorCategory.UNKNOWN

    def get_memory_breakdown(self) -> Dict[TensorCategory, int]:
        """Get current memory usage by category"""
        return dict(self.category_stats)

    def get_peak_allocation(self) -> Optional[AllocationEvent]:
        """Find allocation that caused peak memory"""
        # Find allocation event closest to peak time
        peak_allocations = [
            e for e in self.allocations.values()
            if abs(e.timestamp - self.peak_time) < 0.001
        ]
        return max(peak_allocations, key=lambda e: e.size) if peak_allocations else None
```

### Component 2: Fragmentation Analysis

**Location:** Integration with [`c10/cuda/CUDACachingAllocator.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/c10/cuda/CUDACachingAllocator.cpp)

```python
class FragmentationAnalyzer:
    """Analyze memory fragmentation in caching allocator"""

    def __init__(self):
        self.allocator_stats = {}

    def analyze_fragmentation(self, device: torch.device) -> FragmentationReport:
        """Analyze current fragmentation on device"""
        # Get allocator stats from C++
        stats = torch.cuda.memory_stats(device)

        # Calculate fragmentation metrics
        allocated = stats['allocated_bytes.all.current']
        reserved = stats['reserved_bytes.all.current']
        free_cached = reserved - allocated

        # Get block-level information
        blocks = self._get_block_info(device)

        # Calculate external fragmentation
        # (sum of free blocks / largest free block)
        free_blocks = [b for b in blocks if not b.allocated]
        if free_blocks:
            total_free = sum(b.size for b in free_blocks)
            largest_free = max(b.size for b in free_blocks)
            external_frag = 1 - (largest_free / total_free) if total_free > 0 else 0
        else:
            external_frag = 0

        return FragmentationReport(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            free_cached_bytes=free_cached,
            largest_free_block=largest_free if free_blocks else 0,
            external_fragmentation=external_frag,
            num_free_blocks=len(free_blocks),
            recommendations=self._generate_recommendations(external_frag, free_cached)
        )

    def _generate_recommendations(self, frag: float, free_cached: int) -> List[str]:
        """Generate defragmentation recommendations"""
        recommendations = []

        if frag > 0.5:
            recommendations.append(
                "High fragmentation detected (50%+). Consider torch.cuda.empty_cache() "
                "or reducing batch size."
            )

        if free_cached > 1e9:  # > 1 GB cached but free
            recommendations.append(
                f"{free_cached/1e9:.1f}GB cached but unused. Call torch.cuda.empty_cache() "
                "to release to OS."
            )

        return recommendations

    def _get_block_info(self, device: torch.device) -> List[BlockInfo]:
        """Get detailed block information from allocator"""
        # Call into C++ to get block-level stats
        return torch.cuda.memory._get_allocator_backend().get_block_info(device)
```

### Component 3: Memory Leak Detection

```python
class MemoryLeakDetector:
    """Detect memory leaks in training loops"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.memory_history = []
        self.leak_detected = False

    def record_iteration(self, memory_used: int):
        """Record memory usage after each iteration"""
        self.memory_history.append(memory_used)

        # Keep only recent history
        if len(self.memory_history) > self.window_size:
            self.memory_history.pop(0)

        # Detect leak
        if len(self.memory_history) == self.window_size:
            # Check for monotonic growth
            is_growing = all(
                self.memory_history[i] < self.memory_history[i+1]
                for i in range(len(self.memory_history) - 1)
            )

            if is_growing:
                growth_rate = (
                    (self.memory_history[-1] - self.memory_history[0]) /
                    self.memory_history[0]
                )

                if growth_rate > 0.1:  # 10% growth over window
                    self.leak_detected = True
                    return LeakWarning(
                        message=f"Memory leak detected: {growth_rate*100:.1f}% growth over "
                                f"{self.window_size} iterations",
                        growth_rate=growth_rate,
                        recommendations=self._diagnose_leak()
                    )

        return None

    def _diagnose_leak(self) -> List[str]:
        """Diagnose potential causes of leak"""
        recommendations = []

        # Check for common causes
        recommendations.append("Check if optimizer.zero_grad() is called each iteration")
        recommendations.append("Verify no tensors are being appended to lists without .detach()")
        recommendations.append("Ensure evaluation mode (model.eval()) is used during validation")
        recommendations.append("Check for circular references in custom modules")

        return recommendations
```

### Component 4: Optimization Engine

```python
class MemoryOptimizationEngine:
    """Generate optimization recommendations"""

    def __init__(self, model: nn.Module, batch_size: int, device: torch.device):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def analyze_and_recommend(self, profiler_results: EnhancedMemoryProfiler) -> List[Recommendation]:
        """Analyze profiling results and recommend optimizations"""
        recommendations = []

        # Analyze activation memory
        activation_memory = profiler_results.category_stats[TensorCategory.ACTIVATION]
        param_memory = profiler_results.category_stats[TensorCategory.PARAMETER]
        total_memory = sum(profiler_results.category_stats.values())

        # Recommendation 1: Activation checkpointing
        if activation_memory > 0.5 * total_memory:
            savings = self._estimate_checkpoint_savings(profiler_results)
            recommendations.append(Recommendation(
                type="activation_checkpointing",
                priority="high",
                message=f"Activations use {activation_memory/1e9:.1f}GB ({activation_memory/total_memory*100:.1f}%). "
                        f"Activation checkpointing could save {savings/1e9:.1f}GB.",
                code_example="""
# Add checkpointing to model
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
""",
                expected_savings_bytes=savings
            ))

        # Recommendation 2: Gradient accumulation
        if total_memory > torch.cuda.get_device_properties(self.device).total_memory * 0.9:
            recommendations.append(Recommendation(
                type="gradient_accumulation",
                priority="high",
                message=f"Using {total_memory/1e9:.1f}GB of {torch.cuda.get_device_properties(self.device).total_memory/1e9:.1f}GB. "
                        f"Consider gradient accumulation to reduce batch size.",
                code_example="""
# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
""",
                expected_savings_bytes=total_memory // 4
            ))

        # Recommendation 3: Mixed precision training
        if self._can_use_amp():
            recommendations.append(Recommendation(
                type="mixed_precision",
                priority="medium",
                message="Model appears compatible with mixed precision training. "
                        "Could reduce memory by ~50%.",
                code_example="""
# Use automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
""",
                expected_savings_bytes=param_memory // 2
            ))

        return sorted(recommendations, key=lambda r: r.priority, reverse=True)

    def _estimate_checkpoint_savings(self, profiler: EnhancedMemoryProfiler) -> int:
        """Estimate memory savings from activation checkpointing"""
        # Roughly 2/3 of activations can be recomputed
        return int(profiler.category_stats[TensorCategory.ACTIVATION] * 0.66)

    def _can_use_amp(self) -> bool:
        """Check if model is compatible with AMP"""
        # Check for CUDA capability and dtype
        return (
            self.device.type == 'cuda' and
            torch.cuda.get_device_capability(self.device)[0] >= 7  # Volta+ for Tensor Cores
        )
```

### Component 5: Visualization

```python
class MemoryVisualizer:
    """Visualize memory profiling results"""

    def plot_memory_timeline(self, profiler: EnhancedMemoryProfiler) -> Figure:
        """Create memory timeline plot"""
        import matplotlib.pyplot as plt

        times = [t for t, _ in profiler.timeline]
        memories = [m / 1e9 for _, m in profiler.timeline]  # GB

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, memories, linewidth=2)
        ax.axhline(profiler.peak_memory / 1e9, color='r', linestyle='--', label='Peak')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('GPU Memory Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_category_breakdown(self, profiler: EnhancedMemoryProfiler) -> Figure:
        """Create pie chart of memory by category"""
        import matplotlib.pyplot as plt

        categories = []
        sizes = []
        for cat, size in profiler.category_stats.items():
            if size > 0:
                categories.append(cat.value)
                sizes.append(size / 1e9)  # GB

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.set_title('Memory Usage by Category')

        return fig

    def generate_flamegraph(self, profiler: EnhancedMemoryProfiler) -> str:
        """Generate flamegraph of allocations"""
        # Export to flamegraph format
        lines = []
        for ptr, event in profiler.allocations.items():
            stack = ';'.join(reversed(event.stack_trace))
            lines.append(f"{stack} {event.size}")

        return '\n'.join(lines)
```

## Example Usage

### Before (Limited Visibility):

```python
import torch

model = MyLargeModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Manual tracking
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    # But no idea what's using the memory or how to optimize!
```

### After (Actionable Insights):

```python
from torch.profiler.memory import EnhancedMemoryProfiler, MemoryOptimizationEngine

profiler = EnhancedMemoryProfiler(
    track_stacks=True,
    categorize_allocations=True,
    detect_leaks=True
)

with profiler:
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Get detailed analysis
print(profiler.get_memory_breakdown())
# Output:
# {
#   TensorCategory.PARAMETER: 2.3 GB,
#   TensorCategory.GRADIENT: 2.3 GB,
#   TensorCategory.ACTIVATION: 8.1 GB,  # <-- Problem!
#   TensorCategory.OPTIMIZER_STATE: 4.6 GB,
# }

# Get optimization recommendations
optimizer_engine = MemoryOptimizationEngine(model, batch_size=32, device='cuda')
recommendations = optimizer_engine.analyze_and_recommend(profiler)

for rec in recommendations:
    print(f"[{rec.priority}] {rec.message}")
    print(f"Expected savings: {rec.expected_savings_bytes/1e9:.1f}GB")
    print(f"Example:\n{rec.code_example}\n")

# Output:
# [high] Activations use 8.1GB (50.3%). Activation checkpointing could save 5.4GB.
# Expected savings: 5.4GB
# Example:
# from torch.utils.checkpoint import checkpoint
# ...

# Visualize
profiler.plot_memory_timeline().savefig('memory_timeline.png')
profiler.plot_category_breakdown().savefig('memory_breakdown.png')
```

## Implementation Plan

### Phase 1: Core Infrastructure (2 months)

**Deliverables:**
1. Implement EnhancedMemoryProfiler with allocation tracking
2. Integration with CUDACachingAllocator for fragmentation analysis
3. Basic visualization (timeline, breakdown)

**Success Criteria:**
- Accurate categorization of 90%+ allocations
- Fragmentation analysis matches manual inspection
- Visualization renders correctly

### Phase 2: Leak Detection & Optimization Engine (2 months)

**Deliverables:**
1. Implement MemoryLeakDetector
2. Implement MemoryOptimizationEngine with 5+ recommendation types
3. TensorBoard integration

**Success Criteria:**
- Detects known leaks in test cases
- Recommendations reduce memory by 30%+ in benchmarks
- TensorBoard displays memory timeline

### Phase 3: Production Hardening (1 month)

**Deliverables:**
1. Performance optimization (< 5% overhead)
2. Distributed training support
3. Documentation and tutorials

**Success Criteria:**
- < 5% training slowdown when profiling enabled
- Works with DDP, FSDP, etc.
- Tutorial shows end-to-end workflow

## Backwards Compatibility

- New profiler is opt-in (must import explicitly)
- Existing `torch.profiler` unchanged
- No changes to tensor allocation APIs
- Compatible with existing memory APIs (memory_allocated, etc.)

## Alternatives Considered

### Alternative 1: External Tools (Nsight, CUDA Profilers)

**Pros:**
- No development cost for PyTorch
- Mature tools

**Cons:**
- Not PyTorch-aware (can't categorize allocations)
- Steeper learning curve
- No automatic recommendations

**Decision:** Complement, not replacement

### Alternative 2: Python Memory Profilers (memory_profiler, pympler)

**Pros:**
- General-purpose Python profiling

**Cons:**
- Don't understand CUDA memory
- Too coarse-grained for tensor operations

**Decision:** Not suitable for GPU memory profiling

## Open Questions

1. **Overhead acceptable for training?**
   - Stack trace capture is expensive (~10% overhead)
   - Proposal: Tiered profiling (lightweight by default, detailed on demand)

2. **Integration with distributed training?**
   - Need to aggregate memory stats across ranks
   - Proposal: Per-rank tracking with aggregation visualizations

3. **How to handle dynamic models (e.g., RNNs)?**
   - Memory usage varies by sequence length
   - Proposal: Profile multiple sequence lengths, report range

## Success Metrics

1. **Adoption:** 50%+ of users encountering OOM use memory profiler
2. **Effectiveness:** 70%+ of users reduce memory by following recommendations
3. **Leak Detection:** 90%+ detection rate on synthetic leak tests
4. **Performance:** < 5% overhead when profiling enabled

## References

- **Current Memory Profiler:** [`torch/profiler/_memory_profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/_memory_profiler.py)
- **CUDA Allocator:** [`c10/cuda/CUDACachingAllocator.cpp`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/c10/cuda/CUDACachingAllocator.cpp)
- **Profiler Infrastructure:** [`torch/profiler/profiler.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/profiler/profiler.py)
- **Related Work:** TensorFlow Memory Profiler, NVIDIA Nsight Systems

---

**Next Steps:**
1. Prototype allocation tracking hooks
2. Validate fragmentation analysis accuracy
3. Build recommendation engine prototype
4. User study with 10+ beta testers
5. Refine based on feedback
