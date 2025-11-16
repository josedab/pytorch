# mypy: allow-untyped-defs
"""
Enhanced Memory Profiling and Optimization

This module provides enhanced memory profiling capabilities for PyTorch, including:
- Real-time memory tracking with allocation categorization
- Memory fragmentation analysis
- Memory leak detection
- Optimization recommendations
- Advanced visualization

Example usage:
    from torch.profiler.memory import EnhancedMemoryProfiler

    profiler = EnhancedMemoryProfiler(
        track_stacks=True,
        categorize_allocations=True,
        detect_leaks=True
    )

    with profiler:
        # Your training code here
        pass

    # Get detailed analysis
    breakdown = profiler.get_memory_breakdown()
    recommendations = profiler.get_optimization_recommendations()
"""

import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TensorCategory(Enum):
    """Category of tensor allocation."""
    PARAMETER = "parameter"          # Model parameters
    GRADIENT = "gradient"            # Gradients
    ACTIVATION = "activation"        # Forward pass activations
    OPTIMIZER_STATE = "optimizer"    # Optimizer momentum, etc.
    TEMPORARY = "temporary"          # Short-lived temps
    PERSISTENT = "persistent"        # User-managed persistent tensors
    UNKNOWN = "unknown"


@dataclass
class AllocationEvent:
    """Detailed allocation event."""
    ptr: int                         # Memory address
    size: int                        # Bytes allocated
    device: torch.device
    category: TensorCategory
    timestamp: float                 # Monotonic time
    stack_trace: List[str]           # Allocation stack
    stream: int                      # CUDA stream ID
    tensor_id: Optional[int] = None  # Tensor object ID


@dataclass
class FragmentationReport:
    """Report on memory fragmentation."""
    allocated_bytes: int
    reserved_bytes: int
    free_cached_bytes: int
    largest_free_block: int
    external_fragmentation: float
    num_free_blocks: int
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Fragmentation Report:\n"
            f"  Allocated: {self.allocated_bytes / 1e9:.2f} GB\n"
            f"  Reserved: {self.reserved_bytes / 1e9:.2f} GB\n"
            f"  Free (cached): {self.free_cached_bytes / 1e9:.2f} GB\n"
            f"  Largest free block: {self.largest_free_block / 1e9:.2f} GB\n"
            f"  External fragmentation: {self.external_fragmentation * 100:.1f}%\n"
            f"  Number of free blocks: {self.num_free_blocks}\n"
        )


@dataclass
class LeakWarning:
    """Warning about a potential memory leak."""
    message: str
    growth_rate: float
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Memory Leak Warning:\n"
            f"  {self.message}\n"
            f"  Recommendations:\n" +
            "\n".join(f"    - {rec}" for rec in self.recommendations)
        )


@dataclass
class Recommendation:
    """Optimization recommendation."""
    type: str
    priority: str
    message: str
    code_example: str
    expected_savings_bytes: int


class EnhancedMemoryProfiler:
    """Enhanced memory profiler with real-time tracking and categorization.

    This profiler provides detailed insights into memory usage patterns,
    including allocation categorization, memory leak detection, and
    optimization recommendations.

    Args:
        track_stacks: Whether to capture stack traces for allocations
        categorize_allocations: Whether to categorize allocations by type
        detect_leaks: Whether to enable memory leak detection
        device: Device to profile (defaults to current CUDA device)
    """

    def __init__(
        self,
        track_stacks: bool = True,
        categorize_allocations: bool = True,
        detect_leaks: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.track_stacks = track_stacks
        self.categorize_allocations = categorize_allocations
        self.detect_leaks = detect_leaks
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.allocations: Dict[int, AllocationEvent] = {}
        self.timeline: List[Tuple[float, int]] = []
        self.peak_memory: int = 0
        self.peak_time: float = 0
        self.category_stats: Dict[TensorCategory, int] = defaultdict(int)
        self.start_time: Optional[float] = None
        self._active: bool = False

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.monotonic()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling."""
        self._active = False
        return False

    def on_allocation(
        self, ptr: int, size: int, device: torch.device, stream: int = 0
    ):
        """Hook called on memory allocation.

        Args:
            ptr: Memory pointer
            size: Allocation size in bytes
            device: Device where allocation occurred
            stream: CUDA stream ID
        """
        if not self._active:
            return

        # Categorize allocation
        category = self._categorize_allocation() if self.categorize_allocations else TensorCategory.UNKNOWN

        # Capture stack trace
        stack = []
        if self.track_stacks:
            stack_frames = traceback.extract_stack()
            stack = [str(frame) for frame in stack_frames]

        # Record event
        event = AllocationEvent(
            ptr=ptr,
            size=size,
            device=device,
            category=category,
            timestamp=time.monotonic(),
            stack_trace=stack,
            stream=stream,
            tensor_id=None,
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
        """Hook called on memory deallocation.

        Args:
            ptr: Memory pointer being freed
        """
        if not self._active:
            return

        if ptr in self.allocations:
            event = self.allocations[ptr]
            self.category_stats[event.category] -= event.size
            del self.allocations[ptr]

    def _categorize_allocation(self) -> TensorCategory:
        """Infer allocation category from context.

        Returns:
            TensorCategory indicating the type of allocation
        """
        # Analyze stack trace to determine category
        stack = traceback.extract_stack()

        # Check for gradient allocation
        if any('autograd' in frame.filename for frame in stack):
            return TensorCategory.GRADIENT

        # Check for optimizer state
        if any('optim' in frame.filename for frame in stack):
            return TensorCategory.OPTIMIZER_STATE

        # Check for parameter allocation
        if any('nn/modules' in frame.filename or 'nn\\modules' in frame.filename for frame in stack):
            return TensorCategory.PARAMETER

        # Check for forward pass
        if any('forward' in frame.name for frame in stack):
            return TensorCategory.ACTIVATION

        return TensorCategory.UNKNOWN

    def get_memory_breakdown(self) -> Dict[TensorCategory, int]:
        """Get current memory usage by category.

        Returns:
            Dictionary mapping TensorCategory to bytes used
        """
        return dict(self.category_stats)

    def get_peak_allocation(self) -> Optional[AllocationEvent]:
        """Find allocation that caused peak memory.

        Returns:
            AllocationEvent for the largest allocation near peak time
        """
        # Find allocation event closest to peak time
        peak_allocations = [
            e for e in self.allocations.values()
            if abs(e.timestamp - self.peak_time) < 0.001
        ]
        return max(peak_allocations, key=lambda e: e.size) if peak_allocations else None

    def get_current_memory(self) -> int:
        """Get current total memory allocated.

        Returns:
            Total bytes allocated
        """
        return sum(e.size for e in self.allocations.values())

    def get_optimization_recommendations(
        self, model: Optional[nn.Module] = None, batch_size: Optional[int] = None
    ) -> List[Recommendation]:
        """Generate optimization recommendations.

        Args:
            model: Optional model for context-aware recommendations
            batch_size: Optional batch size for recommendations

        Returns:
            List of optimization recommendations
        """
        if model is not None and batch_size is not None:
            engine = MemoryOptimizationEngine(model, batch_size, self.device)
            return engine.analyze_and_recommend(self)
        return []


class FragmentationAnalyzer:
    """Analyze memory fragmentation in caching allocator.

    This analyzer provides insights into how fragmented the CUDA memory
    allocator's cache has become.
    """

    def __init__(self):
        self.allocator_stats: Dict[str, Any] = {}

    def analyze_fragmentation(self, device: torch.device) -> FragmentationReport:
        """Analyze current fragmentation on device.

        Args:
            device: Device to analyze

        Returns:
            FragmentationReport with detailed metrics
        """
        if device.type != 'cuda':
            return FragmentationReport(
                allocated_bytes=0,
                reserved_bytes=0,
                free_cached_bytes=0,
                largest_free_block=0,
                external_fragmentation=0.0,
                num_free_blocks=0,
                recommendations=["Fragmentation analysis only available for CUDA devices"]
            )

        # Get allocator stats from CUDA
        stats = torch.cuda.memory_stats(device)

        # Calculate fragmentation metrics
        allocated = stats.get('allocated_bytes.all.current', 0)
        reserved = stats.get('reserved_bytes.all.current', 0)
        free_cached = reserved - allocated

        # Get number of free blocks
        num_alloc_retries = stats.get('num_alloc_retries', 0)
        num_ooms = stats.get('num_ooms', 0)

        # Estimate fragmentation based on memory snapshot if available
        try:
            snapshot = torch.cuda.memory_snapshot()
            free_blocks = [
                seg for seg in snapshot
                if seg.get('is_cached', False) and not seg.get('is_allocated', True)
            ]
            num_free_blocks = len(free_blocks)
            if free_blocks:
                total_free = sum(seg.get('size', 0) for seg in free_blocks)
                largest_free = max(seg.get('size', 0) for seg in free_blocks)
                external_frag = 1 - (largest_free / total_free) if total_free > 0 else 0
            else:
                largest_free = 0
                external_frag = 0
                num_free_blocks = 0
        except Exception:
            # Fallback if snapshot not available
            largest_free = free_cached
            external_frag = 0.0
            num_free_blocks = 1 if free_cached > 0 else 0

        recommendations = self._generate_recommendations(
            external_frag, free_cached, num_alloc_retries, num_ooms
        )

        return FragmentationReport(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            free_cached_bytes=free_cached,
            largest_free_block=largest_free,
            external_fragmentation=external_frag,
            num_free_blocks=num_free_blocks,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, frag: float, free_cached: int, num_retries: int, num_ooms: int
    ) -> List[str]:
        """Generate defragmentation recommendations.

        Args:
            frag: External fragmentation ratio
            free_cached: Amount of free cached memory
            num_retries: Number of allocation retries
            num_ooms: Number of out-of-memory errors

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if frag > 0.5:
            recommendations.append(
                f"High fragmentation detected ({frag*100:.1f}%). Consider "
                "torch.cuda.empty_cache() or reducing batch size."
            )

        if free_cached > 1e9:  # > 1 GB cached but free
            recommendations.append(
                f"{free_cached/1e9:.1f}GB cached but unused. Call "
                "torch.cuda.empty_cache() to release to OS."
            )

        if num_retries > 0:
            recommendations.append(
                f"{num_retries} allocation retries occurred. This indicates "
                "memory pressure and potential fragmentation."
            )

        if num_ooms > 0:
            recommendations.append(
                f"{num_ooms} out-of-memory errors occurred. Consider reducing "
                "batch size or enabling gradient checkpointing."
            )

        return recommendations


class MemoryLeakDetector:
    """Detect memory leaks in training loops.

    This detector monitors memory usage over iterations and identifies
    patterns consistent with memory leaks.
    """

    def __init__(self, window_size: int = 10):
        """Initialize leak detector.

        Args:
            window_size: Number of iterations to analyze for trends
        """
        self.window_size = window_size
        self.memory_history: List[int] = []
        self.leak_detected: bool = False

    def record_iteration(self, memory_used: int) -> Optional[LeakWarning]:
        """Record memory usage after each iteration.

        Args:
            memory_used: Memory usage in bytes

        Returns:
            LeakWarning if a leak is detected, None otherwise
        """
        self.memory_history.append(memory_used)

        # Keep only recent history
        if len(self.memory_history) > self.window_size:
            self.memory_history.pop(0)

        # Detect leak
        if len(self.memory_history) == self.window_size:
            # Check for monotonic growth
            is_growing = all(
                self.memory_history[i] < self.memory_history[i + 1]
                for i in range(len(self.memory_history) - 1)
            )

            if is_growing:
                growth_rate = (
                    (self.memory_history[-1] - self.memory_history[0]) /
                    self.memory_history[0]
                ) if self.memory_history[0] > 0 else 0

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
        """Diagnose potential causes of leak.

        Returns:
            List of potential causes and remedies
        """
        recommendations = [
            "Check if optimizer.zero_grad() is called each iteration",
            "Verify no tensors are being appended to lists without .detach()",
            "Ensure evaluation mode (model.eval()) is used during validation",
            "Check for circular references in custom modules",
            "Verify gradient accumulation is implemented correctly",
        ]

        return recommendations


class MemoryOptimizationEngine:
    """Generate optimization recommendations based on profiling results.

    This engine analyzes memory usage patterns and suggests optimizations
    to reduce memory consumption.
    """

    def __init__(self, model: nn.Module, batch_size: int, device: torch.device):
        """Initialize optimization engine.

        Args:
            model: PyTorch model to analyze
            batch_size: Current batch size
            device: Device being used
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def analyze_and_recommend(
        self, profiler_results: EnhancedMemoryProfiler
    ) -> List[Recommendation]:
        """Analyze profiling results and recommend optimizations.

        Args:
            profiler_results: Results from EnhancedMemoryProfiler

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Analyze activation memory
        activation_memory = profiler_results.category_stats.get(TensorCategory.ACTIVATION, 0)
        param_memory = profiler_results.category_stats.get(TensorCategory.PARAMETER, 0)
        total_memory = sum(profiler_results.category_stats.values())

        if total_memory == 0:
            return recommendations

        # Recommendation 1: Activation checkpointing
        if activation_memory > 0.4 * total_memory:
            savings = self._estimate_checkpoint_savings(profiler_results)
            recommendations.append(Recommendation(
                type="activation_checkpointing",
                priority="high",
                message=f"Activations use {activation_memory/1e9:.2f}GB "
                        f"({activation_memory/total_memory*100:.1f}%). "
                        f"Activation checkpointing could save {savings/1e9:.2f}GB.",
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
        if self.device.type == 'cuda':
            device_memory = torch.cuda.get_device_properties(self.device).total_memory
            if total_memory > device_memory * 0.85:
                recommendations.append(Recommendation(
                    type="gradient_accumulation",
                    priority="high",
                    message=f"Using {total_memory/1e9:.2f}GB of "
                            f"{device_memory/1e9:.2f}GB. Consider gradient "
                            "accumulation to reduce batch size.",
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
                        "Could reduce memory by ~40-50%.",
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
                expected_savings_bytes=int(param_memory * 0.45)
            ))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations

    def _estimate_checkpoint_savings(
        self, profiler: EnhancedMemoryProfiler
    ) -> int:
        """Estimate memory savings from activation checkpointing.

        Args:
            profiler: Memory profiler results

        Returns:
            Estimated bytes saved
        """
        # Roughly 60-70% of activations can be recomputed
        activation_memory = profiler.category_stats.get(TensorCategory.ACTIVATION, 0)
        return int(activation_memory * 0.65)

    def _can_use_amp(self) -> bool:
        """Check if model is compatible with AMP.

        Returns:
            True if AMP can be used
        """
        # Check for CUDA capability and dtype
        if self.device.type != 'cuda':
            return False

        # Check for Volta (SM 7.0) or newer for Tensor Cores
        capability = torch.cuda.get_device_capability(self.device)
        return capability[0] >= 7


class MemoryVisualizer:
    """Visualize memory profiling results.

    This visualizer creates plots and charts to help understand memory
    usage patterns.
    """

    def plot_memory_timeline(
        self, profiler: EnhancedMemoryProfiler, figsize: Tuple[int, int] = (12, 6)
    ):
        """Create memory timeline plot.

        Args:
            profiler: Memory profiler results
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        times = [t - (profiler.start_time or 0) for t, _ in profiler.timeline]
        memories = [m / 1e9 for _, m in profiler.timeline]  # GB

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, memories, linewidth=2)
        ax.axhline(
            profiler.peak_memory / 1e9,
            color='r',
            linestyle='--',
            label=f'Peak: {profiler.peak_memory / 1e9:.2f} GB'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('GPU Memory Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_category_breakdown(
        self, profiler: EnhancedMemoryProfiler, figsize: Tuple[int, int] = (10, 8)
    ):
        """Create pie chart of memory by category.

        Args:
            profiler: Memory profiler results
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        categories = []
        sizes = []
        for cat, size in profiler.category_stats.items():
            if size > 0:
                categories.append(cat.value)
                sizes.append(size / 1e9)  # GB

        if not sizes:
            raise ValueError("No memory data to visualize")

        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.set_title('Memory Usage by Category')

        return fig

    def generate_flamegraph(self, profiler: EnhancedMemoryProfiler) -> str:
        """Generate flamegraph data of allocations.

        Args:
            profiler: Memory profiler results

        Returns:
            Flamegraph data in standard format
        """
        lines = []
        for ptr, event in profiler.allocations.items():
            if event.stack_trace:
                # Format stack trace for flamegraph
                stack = ';'.join(reversed(event.stack_trace[:10]))  # Limit depth
                lines.append(f"{stack} {event.size}")

        return '\n'.join(lines)


__all__ = [
    'TensorCategory',
    'AllocationEvent',
    'FragmentationReport',
    'LeakWarning',
    'Recommendation',
    'EnhancedMemoryProfiler',
    'FragmentationAnalyzer',
    'MemoryLeakDetector',
    'MemoryOptimizationEngine',
    'MemoryVisualizer',
]
