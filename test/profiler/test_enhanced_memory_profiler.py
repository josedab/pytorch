# Owner(s): ["oncall: profiler"]
"""
Tests for Enhanced Memory Profiler (RFC-0003)

This test suite covers the new enhanced memory profiling capabilities:
- Real-time memory tracking with allocation categorization
- Memory fragmentation analysis
- Memory leak detection
- Optimization recommendations
- Advanced visualization
"""

import unittest

import torch
import torch.nn as nn
from torch.profiler.memory import (
    AllocationEvent,
    EnhancedMemoryProfiler,
    FragmentationAnalyzer,
    FragmentationReport,
    LeakWarning,
    MemoryLeakDetector,
    MemoryOptimizationEngine,
    MemoryVisualizer,
    Recommendation,
    TensorCategory,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestEnhancedMemoryProfiler(TestCase):
    """Test basic EnhancedMemoryProfiler functionality."""

    def test_profiler_context_manager(self):
        """Test that profiler can be used as a context manager."""
        profiler = EnhancedMemoryProfiler()
        self.assertFalse(profiler._active)

        with profiler:
            self.assertTrue(profiler._active)

        self.assertFalse(profiler._active)

    def test_allocation_tracking(self):
        """Test manual allocation tracking."""
        profiler = EnhancedMemoryProfiler()

        with profiler:
            # Simulate allocation
            profiler.on_allocation(
                ptr=0x1000, size=1024, device=torch.device('cpu'), stream=0
            )
            self.assertEqual(len(profiler.allocations), 1)
            self.assertEqual(profiler.get_current_memory(), 1024)

            # Simulate deallocation
            profiler.on_deallocation(ptr=0x1000)
            self.assertEqual(len(profiler.allocations), 0)
            self.assertEqual(profiler.get_current_memory(), 0)

    def test_memory_breakdown(self):
        """Test memory categorization."""
        profiler = EnhancedMemoryProfiler(categorize_allocations=False)

        with profiler:
            # Add allocations of different categories
            for i, category in enumerate([
                TensorCategory.PARAMETER,
                TensorCategory.GRADIENT,
                TensorCategory.ACTIVATION,
            ]):
                event = AllocationEvent(
                    ptr=0x1000 + i,
                    size=(i + 1) * 1024,
                    device=torch.device('cpu'),
                    category=category,
                    timestamp=0.0,
                    stack_trace=[],
                    stream=0,
                )
                profiler.allocations[event.ptr] = event
                profiler.category_stats[category] = event.size

        breakdown = profiler.get_memory_breakdown()
        self.assertEqual(breakdown[TensorCategory.PARAMETER], 1024)
        self.assertEqual(breakdown[TensorCategory.GRADIENT], 2048)
        self.assertEqual(breakdown[TensorCategory.ACTIVATION], 3072)

    def test_peak_memory_tracking(self):
        """Test peak memory tracking."""
        profiler = EnhancedMemoryProfiler()

        with profiler:
            # Simulate multiple allocations
            for i in range(5):
                profiler.on_allocation(
                    ptr=0x1000 + i, size=(i + 1) * 1024, device=torch.device('cpu')
                )

            # Peak should be sum of all allocations
            expected_peak = sum((i + 1) * 1024 for i in range(5))
            self.assertEqual(profiler.peak_memory, expected_peak)

    def test_timeline_tracking(self):
        """Test memory timeline recording."""
        profiler = EnhancedMemoryProfiler()

        with profiler:
            profiler.on_allocation(ptr=0x1000, size=1024, device=torch.device('cpu'))
            profiler.on_allocation(ptr=0x2000, size=2048, device=torch.device('cpu'))

        # Should have timeline entries
        self.assertGreater(len(profiler.timeline), 0)

        # Timeline should track memory over time
        _, final_memory = profiler.timeline[-1]
        self.assertEqual(final_memory, 3072)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        """Test profiler with CUDA device."""
        profiler = EnhancedMemoryProfiler(device=torch.device('cuda:0'))

        with profiler:
            profiler.on_allocation(
                ptr=0x1000, size=1024, device=torch.device('cuda:0')
            )

        self.assertEqual(profiler.device, torch.device('cuda:0'))


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestFragmentationAnalyzer(TestCase):
    """Test FragmentationAnalyzer functionality."""

    def test_analyzer_creation(self):
        """Test FragmentationAnalyzer can be created."""
        analyzer = FragmentationAnalyzer()
        self.assertIsNotNone(analyzer)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_fragmentation_analysis(self):
        """Test fragmentation analysis on CUDA."""
        analyzer = FragmentationAnalyzer()
        device = torch.device('cuda:0')

        # Allocate some memory
        tensors = [torch.randn(100, 100, device=device) for _ in range(10)]

        # Analyze fragmentation
        report = analyzer.analyze_fragmentation(device)

        self.assertIsInstance(report, FragmentationReport)
        self.assertGreaterEqual(report.allocated_bytes, 0)
        self.assertGreaterEqual(report.reserved_bytes, 0)
        self.assertGreaterEqual(report.external_fragmentation, 0.0)
        self.assertLessEqual(report.external_fragmentation, 1.0)

        # Clean up
        del tensors
        torch.cuda.empty_cache()

    def test_cpu_fragmentation_analysis(self):
        """Test fragmentation analysis on CPU (should return gracefully)."""
        analyzer = FragmentationAnalyzer()
        device = torch.device('cpu')

        report = analyzer.analyze_fragmentation(device)

        self.assertIsInstance(report, FragmentationReport)
        self.assertEqual(report.allocated_bytes, 0)
        self.assertIn("CUDA", report.recommendations[0])


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestMemoryLeakDetector(TestCase):
    """Test MemoryLeakDetector functionality."""

    def test_detector_creation(self):
        """Test MemoryLeakDetector can be created."""
        detector = MemoryLeakDetector(window_size=5)
        self.assertEqual(detector.window_size, 5)

    def test_no_leak_detection(self):
        """Test that constant memory usage doesn't trigger leak warning."""
        detector = MemoryLeakDetector(window_size=5)

        # Record constant memory usage
        for _ in range(10):
            warning = detector.record_iteration(1000)
            if warning is not None:
                self.fail("Leak detected when there should be none")

    def test_leak_detection(self):
        """Test that growing memory usage triggers leak warning."""
        detector = MemoryLeakDetector(window_size=5)

        # Record growing memory usage
        warning = None
        for i in range(10):
            memory = 1000 * (i + 1)  # Linear growth
            result = detector.record_iteration(memory)
            if result is not None:
                warning = result
                break

        self.assertIsNotNone(warning, "Leak should have been detected")
        self.assertIsInstance(warning, LeakWarning)
        self.assertGreater(warning.growth_rate, 0)
        self.assertGreater(len(warning.recommendations), 0)

    def test_window_size_limiting(self):
        """Test that history is limited to window size."""
        detector = MemoryLeakDetector(window_size=3)

        for i in range(10):
            detector.record_iteration(1000 * i)

        self.assertEqual(len(detector.memory_history), 3)


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestMemoryOptimizationEngine(TestCase):
    """Test MemoryOptimizationEngine functionality."""

    def test_engine_creation(self):
        """Test MemoryOptimizationEngine can be created."""
        model = nn.Linear(10, 10)
        engine = MemoryOptimizationEngine(
            model=model, batch_size=32, device=torch.device('cpu')
        )
        self.assertIsNotNone(engine)

    def test_activation_checkpointing_recommendation(self):
        """Test recommendation for activation checkpointing."""
        model = nn.Linear(10, 10)
        engine = MemoryOptimizationEngine(
            model=model, batch_size=32, device=torch.device('cpu')
        )

        # Create a mock profiler with high activation memory
        profiler = EnhancedMemoryProfiler()
        profiler.category_stats[TensorCategory.ACTIVATION] = 10 * 1024 ** 3  # 10 GB
        profiler.category_stats[TensorCategory.PARAMETER] = 1 * 1024 ** 3    # 1 GB

        recommendations = engine.analyze_and_recommend(profiler)

        # Should recommend activation checkpointing
        checkpoint_recs = [r for r in recommendations if r.type == "activation_checkpointing"]
        self.assertGreater(len(checkpoint_recs), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gradient_accumulation_recommendation(self):
        """Test recommendation for gradient accumulation."""
        model = nn.Linear(10, 10)
        device = torch.device('cuda:0')
        engine = MemoryOptimizationEngine(
            model=model, batch_size=32, device=device
        )

        # Create a mock profiler with very high memory usage
        profiler = EnhancedMemoryProfiler()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        profiler.category_stats[TensorCategory.ACTIVATION] = int(total_memory * 0.9)

        recommendations = engine.analyze_and_recommend(profiler)

        # Should recommend gradient accumulation
        grad_acc_recs = [r for r in recommendations if r.type == "gradient_accumulation"]
        self.assertGreater(len(grad_acc_recs), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_amp_recommendation(self):
        """Test recommendation for mixed precision training."""
        model = nn.Linear(10, 10)
        device = torch.device('cuda:0')

        # Only test if device supports AMP (Volta or newer)
        capability = torch.cuda.get_device_capability(device)
        if capability[0] < 7:
            self.skipTest("Device does not support AMP")

        engine = MemoryOptimizationEngine(
            model=model, batch_size=32, device=device
        )

        profiler = EnhancedMemoryProfiler()
        profiler.category_stats[TensorCategory.PARAMETER] = 1 * 1024 ** 3

        recommendations = engine.analyze_and_recommend(profiler)

        # Should recommend mixed precision
        amp_recs = [r for r in recommendations if r.type == "mixed_precision"]
        self.assertGreater(len(amp_recs), 0)

    def test_empty_profiler_recommendations(self):
        """Test that engine handles empty profiler gracefully."""
        model = nn.Linear(10, 10)
        engine = MemoryOptimizationEngine(
            model=model, batch_size=32, device=torch.device('cpu')
        )

        profiler = EnhancedMemoryProfiler()
        # Empty profiler

        recommendations = engine.analyze_and_recommend(profiler)
        self.assertEqual(len(recommendations), 0)


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestMemoryVisualizer(TestCase):
    """Test MemoryVisualizer functionality."""

    def test_visualizer_creation(self):
        """Test MemoryVisualizer can be created."""
        visualizer = MemoryVisualizer()
        self.assertIsNotNone(visualizer)

    def test_timeline_plot_requires_matplotlib(self):
        """Test that timeline plotting requires matplotlib."""
        try:
            import matplotlib  # noqa: F401
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        visualizer = MemoryVisualizer()
        profiler = EnhancedMemoryProfiler()

        with profiler:
            profiler.on_allocation(ptr=0x1000, size=1024, device=torch.device('cpu'))

        if matplotlib_available:
            # Should succeed
            fig = visualizer.plot_memory_timeline(profiler)
            self.assertIsNotNone(fig)
        else:
            # Should raise ImportError
            with self.assertRaises(ImportError):
                visualizer.plot_memory_timeline(profiler)

    def test_category_breakdown_requires_data(self):
        """Test that category breakdown requires data."""
        try:
            import matplotlib  # noqa: F401
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        if not matplotlib_available:
            self.skipTest("matplotlib not available")

        visualizer = MemoryVisualizer()
        profiler = EnhancedMemoryProfiler()

        # Empty profiler should raise ValueError
        with self.assertRaises(ValueError):
            visualizer.plot_category_breakdown(profiler)

    def test_flamegraph_generation(self):
        """Test flamegraph data generation."""
        visualizer = MemoryVisualizer()
        profiler = EnhancedMemoryProfiler(track_stacks=True)

        with profiler:
            # Create allocation with stack trace
            event = AllocationEvent(
                ptr=0x1000,
                size=1024,
                device=torch.device('cpu'),
                category=TensorCategory.PARAMETER,
                timestamp=0.0,
                stack_trace=['frame1', 'frame2', 'frame3'],
                stream=0,
            )
            profiler.allocations[event.ptr] = event

        flamegraph = visualizer.generate_flamegraph(profiler)
        self.assertIsInstance(flamegraph, str)
        self.assertGreater(len(flamegraph), 0)


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestDataclasses(TestCase):
    """Test dataclass structures."""

    def test_allocation_event(self):
        """Test AllocationEvent dataclass."""
        event = AllocationEvent(
            ptr=0x1000,
            size=1024,
            device=torch.device('cpu'),
            category=TensorCategory.PARAMETER,
            timestamp=0.0,
            stack_trace=[],
            stream=0,
        )
        self.assertEqual(event.ptr, 0x1000)
        self.assertEqual(event.size, 1024)
        self.assertEqual(event.category, TensorCategory.PARAMETER)

    def test_fragmentation_report(self):
        """Test FragmentationReport dataclass."""
        report = FragmentationReport(
            allocated_bytes=1000,
            reserved_bytes=2000,
            free_cached_bytes=1000,
            largest_free_block=500,
            external_fragmentation=0.5,
            num_free_blocks=2,
            recommendations=["Test recommendation"],
        )
        self.assertEqual(report.allocated_bytes, 1000)
        self.assertEqual(report.external_fragmentation, 0.5)

        # Test string representation
        report_str = str(report)
        self.assertIn("Fragmentation Report", report_str)

    def test_leak_warning(self):
        """Test LeakWarning dataclass."""
        warning = LeakWarning(
            message="Test warning",
            growth_rate=0.15,
            recommendations=["Rec 1", "Rec 2"],
        )
        self.assertEqual(warning.growth_rate, 0.15)
        self.assertEqual(len(warning.recommendations), 2)

        # Test string representation
        warning_str = str(warning)
        self.assertIn("Memory Leak Warning", warning_str)

    def test_recommendation(self):
        """Test Recommendation dataclass."""
        rec = Recommendation(
            type="test_type",
            priority="high",
            message="Test message",
            code_example="# Test code",
            expected_savings_bytes=1000,
        )
        self.assertEqual(rec.type, "test_type")
        self.assertEqual(rec.priority, "high")
        self.assertEqual(rec.expected_savings_bytes, 1000)


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestTensorCategory(TestCase):
    """Test TensorCategory enum."""

    def test_category_values(self):
        """Test that all expected categories exist."""
        expected_categories = [
            "parameter",
            "gradient",
            "activation",
            "optimizer",
            "temporary",
            "persistent",
            "unknown",
        ]

        for cat_value in expected_categories:
            # Should be able to access by value
            found = False
            for cat in TensorCategory:
                if cat.value == cat_value:
                    found = True
                    break
            self.assertTrue(found, f"Category {cat_value} not found")


if __name__ == "__main__":
    run_tests()
