"""
Example Test File Using OpInfo V2

This file demonstrates how to use OpInfo V2 to automatically generate
comprehensive test coverage for PyTorch operators.

To run these tests:
    python test/test_ops_v2_example.py
    python test/test_ops_v2_example.py TestAddOpsV2
    python test/test_ops_v2_example.py TestAddOpsV2.test_forward_add_cpu_float32
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.opinfo_v2_examples import op_db_v2
from torch.testing._internal.test_generator import generate_all_tests
from torch.testing._internal.test_optimizer import TestOptimizer


# Example 1: Generate all tests automatically
# This single line generates hundreds of test methods!
generate_all_tests(globals(), op_db_v2, base_class=TestCase)


# Example 2: Generate optimized tests (fewer redundant tests)
# Uncomment to see optimized test generation
"""
from torch.testing._internal.test_generator import TestGenerator

generator = TestGenerator()
optimizer = TestOptimizer(optimization_level='medium')

for opinfo in op_db_v2:
    # Generate tests
    tests = generator.generate_tests(opinfo)

    # Optimize tests
    optimized_tests = optimizer.filter_redundant_tests(opinfo, tests)

    print(f"{opinfo.name}: {len(tests)} tests -> {len(optimized_tests)} optimized tests")
"""


# Example 3: Custom test class with additional tests
class TestCustomOpsV2(TestCase):
    """Custom test class showing how to add additional tests alongside auto-generated ones."""

    def test_add_custom_edge_case(self):
        """Custom test for a specific edge case not covered by auto-generation."""
        # Test adding very large numbers
        a = torch.tensor([1e20], dtype=torch.float32)
        b = torch.tensor([1e20], dtype=torch.float32)
        result = torch.add(a, b)
        self.assertEqual(result, torch.tensor([2e20], dtype=torch.float32))

    def test_mul_custom_edge_case(self):
        """Custom test for multiplication edge case."""
        # Test multiplication with NaN
        a = torch.tensor([float('nan')])
        b = torch.tensor([2.0])
        result = torch.mul(a, b)
        self.assertTrue(torch.isnan(result).all())


# Example 4: Conditional test generation
class TestSelectiveOpsV2(TestCase):
    """Example of generating tests only for specific operators."""
    pass


# Only generate tests for 'add' and 'mul' operators
selective_op_db = [op for op in op_db_v2 if op.name in ['add', 'mul']]
generate_all_tests(
    {k: v for k, v in globals().items() if k == 'TestSelectiveOpsV2'},
    selective_op_db,
    base_class=TestCase
)


# Example 5: Manual test generation with custom configuration
class TestManualOpsV2(TestCase):
    """Example showing manual control over test generation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from torch.testing._internal.test_generator import TestGenerator

        cls.generator = TestGenerator()
        cls.add_opinfo = next(op for op in op_db_v2 if op.name == 'add')

    def test_generated_forward_tests_count(self):
        """Verify the number of forward tests generated."""
        tests = self.generator._generate_forward_tests(self.add_opinfo)
        # Should generate tests for multiple devices and dtypes
        self.assertGreater(len(tests), 0)
        print(f"Generated {len(tests)} forward tests for 'add'")

    def test_generated_backward_tests_count(self):
        """Verify the number of backward tests generated."""
        tests = self.generator._generate_backward_tests(self.add_opinfo)
        # Should generate tests only for floating-point dtypes
        self.assertGreater(len(tests), 0)
        print(f"Generated {len(tests)} backward tests for 'add'")


# Example 6: Performance comparison
class TestPerformanceV2(TestCase):
    """Example showing performance metrics."""

    def test_optimization_statistics(self):
        """Show optimization statistics."""
        from torch.testing._internal.test_generator import TestGenerator

        generator = TestGenerator()
        optimizer = TestOptimizer(optimization_level='medium')

        total_before = 0
        total_after = 0

        for opinfo in op_db_v2:
            tests = generator.generate_tests(opinfo)
            optimized = optimizer.filter_redundant_tests(opinfo, tests)

            total_before += len(tests)
            total_after += len(optimized)

        stats = optimizer.get_statistics()
        print(f"\nOptimization Statistics:")
        print(f"  Total generated: {stats['total_generated']}")
        print(f"  After optimization: {stats['kept']}")
        print(f"  Filtered out: {stats['filtered']}")
        print(f"  Reduction: {stats['reduction_percentage']:.1f}%")

        # Verify optimization actually reduced tests
        self.assertGreater(total_before, total_after)


# Example 7: Coverage analysis
class TestCoverageV2(TestCase):
    """Example showing coverage analysis."""

    def test_coverage_report(self):
        """Generate and verify coverage report."""
        from tools.test_dashboard.coverage_analyzer import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        report = analyzer.generate_report(op_db_v2)

        print(f"\nCoverage Report:")
        print(f"  Total operators: {report.total_operators}")
        print(f"  Average coverage: {report.statistics['average_coverage_percentage']:.1f}%")
        print(f"  Total gaps: {report.statistics['total_gaps']}")

        # Verify all operators have at least forward tests
        for op_coverage in report.operators:
            self.assertTrue(
                op_coverage.has_forward,
                f"Operator {op_coverage.name} missing forward tests"
            )


# Example 8: Test sharding
class TestShardingV2(TestCase):
    """Example showing test sharding for parallel execution."""

    def test_shard_tests(self):
        """Demonstrate test sharding."""
        from torch.testing._internal.test_generator import TestGenerator
        from torch.testing._internal.test_optimizer import TestSharding

        generator = TestGenerator()
        all_tests = []

        for opinfo in op_db_v2:
            all_tests.extend(generator.generate_tests(opinfo))

        # Shard into 4 groups
        sharding = TestSharding(num_shards=4)

        for shard_id in range(4):
            shard_tests = sharding.get_shard(all_tests, shard_id)
            print(f"Shard {shard_id}: {len(shard_tests)} tests")

        # Verify all shards together equal total tests
        total_sharded = sum(
            len(sharding.get_shard(all_tests, i))
            for i in range(4)
        )
        self.assertEqual(total_sharded, len(all_tests))


# Example 9: Comparing with reference implementation
class TestReferenceV2(TestCase):
    """Example showing reference implementation testing."""

    def test_add_against_numpy(self):
        """Verify PyTorch add matches NumPy."""
        import numpy as np

        # Get the add opinfo
        add_opinfo = next(op for op in op_db_v2 if op.name == 'add')

        # Generate sample
        for sample in add_opinfo.sample_inputs_fn(add_opinfo, 'cpu', torch.float32):
            # Compute with PyTorch
            pt_result = torch.add(sample.input, *sample.args, **sample.kwargs)

            # Compute with reference (NumPy)
            if add_opinfo.reference_fn:
                np_result = add_opinfo.reference_fn(
                    sample.input.numpy(),
                    *(arg.numpy() if isinstance(arg, torch.Tensor) else arg
                      for arg in sample.args),
                    **sample.kwargs
                )
                # Verify they match
                self.assertTrue(
                    np.allclose(pt_result.numpy(), np_result),
                    f"PyTorch and NumPy results differ for sample: {sample.name}"
                )


# Example 10: Test prioritization
class TestPrioritizationV2(TestCase):
    """Example showing test prioritization."""

    def test_prioritize_important_tests(self):
        """Demonstrate test prioritization."""
        from torch.testing._internal.test_generator import TestGenerator
        from torch.testing._internal.test_optimizer import TestOptimizer

        generator = TestGenerator()
        optimizer = TestOptimizer()

        all_tests = []
        for opinfo in op_db_v2:
            all_tests.extend(generator.generate_tests(opinfo))

        # Prioritize tests
        prioritized = optimizer.prioritize_tests(all_tests)

        # Verify forward and backward tests are prioritized
        print("\nTop 10 prioritized tests:")
        for i, test in enumerate(prioritized[:10]):
            print(f"  {i+1}. {test.test_name} ({test.category.name})")

        # Verify first test is high priority
        self.assertIn(
            prioritized[0].category.name,
            ['FORWARD', 'BACKWARD'],
            "First test should be high priority (forward or backward)"
        )


if __name__ == '__main__':
    # Run all tests
    run_tests()
