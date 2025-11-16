"""
Enhanced Memory Profiler Usage Example

This example demonstrates how to use the Enhanced Memory Profiler
to analyze memory usage, detect leaks, and get optimization recommendations.

Based on RFC-0003: Enhanced Memory Profiling and Optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler.memory import (
    EnhancedMemoryProfiler,
    FragmentationAnalyzer,
    MemoryLeakDetector,
    MemoryOptimizationEngine,
    MemoryVisualizer,
)


# Define a simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self, input_size=1000, hidden_size=500, output_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x


def example_basic_profiling():
    """Example 1: Basic memory profiling."""
    print("=" * 80)
    print("Example 1: Basic Memory Profiling")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Create profiler
    profiler = EnhancedMemoryProfiler(
        track_stacks=True,
        categorize_allocations=True,
        detect_leaks=True,
        device=device,
    )

    # Training loop with profiling
    with profiler:
        for i in range(5):
            # Forward pass
            inputs = torch.randn(32, 1000, device=device)
            targets = torch.randint(0, 10, (32,), device=device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Get memory breakdown
    print("\nMemory Breakdown by Category:")
    print("-" * 80)
    breakdown = profiler.get_memory_breakdown()
    for category, size in breakdown.items():
        if size > 0:
            print(f"  {category.value:20s}: {size / 1e6:10.2f} MB")

    # Get peak memory info
    print(f"\nPeak Memory: {profiler.peak_memory / 1e6:.2f} MB")

    # Get current memory
    print(f"Current Memory: {profiler.get_current_memory() / 1e6:.2f} MB")


def example_fragmentation_analysis():
    """Example 2: Memory fragmentation analysis."""
    print("\n" + "=" * 80)
    print("Example 2: Memory Fragmentation Analysis")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("Fragmentation analysis requires CUDA")
        return

    device = torch.device('cuda:0')

    # Create some allocations to fragment memory
    tensors = []
    for i in range(20):
        size = (100 + i * 50, 100 + i * 50)
        tensors.append(torch.randn(*size, device=device))

    # Free every other tensor to create fragmentation
    for i in range(0, len(tensors), 2):
        del tensors[i]

    # Analyze fragmentation
    analyzer = FragmentationAnalyzer()
    report = analyzer.analyze_fragmentation(device)

    print(report)

    if report.recommendations:
        print("Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

    # Clean up
    del tensors
    torch.cuda.empty_cache()


def example_leak_detection():
    """Example 3: Memory leak detection."""
    print("\n" + "=" * 80)
    print("Example 3: Memory Leak Detection")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)

    # Create leak detector
    detector = MemoryLeakDetector(window_size=10)

    # Simulate training loop with a memory leak
    # (intentionally not calling zero_grad to create leak)
    print("\nSimulating training loop with memory leak...")

    leak_list = []  # This will accumulate tensors
    for i in range(15):
        inputs = torch.randn(32, 1000, device=device)
        outputs = model(inputs)

        # Intentionally create a leak by keeping references
        leak_list.append(outputs.detach())

        # Record memory usage
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device)
        else:
            # For CPU, approximate based on list length
            memory_used = len(leak_list) * outputs.element_size() * outputs.numel()

        warning = detector.record_iteration(memory_used)

        if warning:
            print(f"\nIteration {i + 1}:")
            print(warning)
            break

    # Clean up
    del leak_list


def example_optimization_recommendations():
    """Example 4: Get optimization recommendations."""
    print("\n" + "=" * 80)
    print("Example 4: Optimization Recommendations")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(input_size=2000, hidden_size=1000).to(device)
    batch_size = 64

    # Create profiler
    profiler = EnhancedMemoryProfiler(device=device)

    # Run a few iterations to build memory profile
    with profiler:
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        for _ in range(3):
            inputs = torch.randn(batch_size, 2000, device=device)
            targets = torch.randint(0, 10, (batch_size,), device=device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Get optimization recommendations
    recommendations = profiler.get_optimization_recommendations(
        model=model, batch_size=batch_size
    )

    print(f"\nFound {len(recommendations)} optimization recommendations:")
    print("-" * 80)

    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}: [{rec.priority.upper()}]")
        print(f"Type: {rec.type}")
        print(f"Message: {rec.message}")
        print(f"Expected Savings: {rec.expected_savings_bytes / 1e6:.2f} MB")
        print(f"Code Example:{rec.code_example}")


def example_visualization():
    """Example 5: Visualize memory usage."""
    print("\n" + "=" * 80)
    print("Example 5: Memory Visualization")
    print("=" * 80)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("matplotlib not available, skipping visualization example")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)

    # Create profiler
    profiler = EnhancedMemoryProfiler(device=device)

    # Run training to generate timeline
    with profiler:
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        for i in range(10):
            inputs = torch.randn(32, 1000, device=device)
            targets = torch.randint(0, 10, (32,), device=device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Create visualizer
    visualizer = MemoryVisualizer()

    # Generate timeline plot
    try:
        fig = visualizer.plot_memory_timeline(profiler)
        fig.savefig('/tmp/memory_timeline.png')
        print("\nMemory timeline saved to /tmp/memory_timeline.png")
    except Exception as e:
        print(f"Could not generate timeline plot: {e}")

    # Generate category breakdown
    try:
        fig = visualizer.plot_category_breakdown(profiler)
        fig.savefig('/tmp/memory_breakdown.png')
        print("Memory breakdown saved to /tmp/memory_breakdown.png")
    except Exception as e:
        print(f"Could not generate breakdown plot: {e}")

    # Generate flamegraph data
    flamegraph = visualizer.generate_flamegraph(profiler)
    if flamegraph:
        with open('/tmp/memory_flamegraph.txt', 'w') as f:
            f.write(flamegraph)
        print("Flamegraph data saved to /tmp/memory_flamegraph.txt")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Enhanced Memory Profiler Examples")
    print("Based on RFC-0003: Enhanced Memory Profiling and Optimization")
    print("=" * 80)

    # Run examples
    example_basic_profiling()
    example_fragmentation_analysis()
    example_leak_detection()
    example_optimization_recommendations()
    example_visualization()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
