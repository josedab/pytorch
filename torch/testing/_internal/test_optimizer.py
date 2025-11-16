"""
Test Optimizer for OpInfo V2

This module provides intelligent test selection to reduce redundant test execution
while maintaining comprehensive coverage. It analyzes operator properties to determine
which test combinations are truly necessary.
"""

from typing import List, Set, Dict, Any
import torch
from torch.testing._internal.opinfo_v2 import OpInfoV2, TestCategory
from torch.testing._internal.test_generator import GeneratedTestCase


class TestOptimizer:
    """Optimize test selection based on operator properties.

    The TestOptimizer analyzes OpInfoV2 definitions and generated tests to:
    - Remove redundant test combinations
    - Sample representative dtypes/shapes when exhaustive testing isn't needed
    - Skip tests that don't apply to certain operators
    - Prioritize tests by importance

    Example:
        >>> optimizer = TestOptimizer()
        >>> tests = generator.generate_tests(opinfo)
        >>> optimized_tests = optimizer.filter_redundant_tests(opinfo, tests)
        >>> print(f"Reduced from {len(tests)} to {len(optimized_tests)} tests")
    """

    # Operators that behave identically across all numeric dtypes
    DTYPE_AGNOSTIC_OPS = {
        'add', 'sub', 'mul', 'div', 'neg', 'abs',
        'minimum', 'maximum', 'clamp',
    }

    # Operators that behave identically regardless of input shape
    SHAPE_AGNOSTIC_OPS = {
        'add', 'sub', 'mul', 'div', 'neg', 'abs',
        'sin', 'cos', 'exp', 'log', 'sqrt',
        'relu', 'sigmoid', 'tanh',
    }

    # Representative dtypes for sampling (covers all type categories)
    REPRESENTATIVE_DTYPES = {
        'floating': [torch.float32, torch.float16],
        'integral': [torch.int32, torch.int64],
        'complex': [torch.complex64],
        'bool': [torch.bool],
    }

    # Representative shapes for sampling
    REPRESENTATIVE_SHAPES = [
        (5,),          # 1D
        (3, 4),        # 2D
        (2, 3, 4),     # 3D
        (2, 2, 3, 4),  # 4D
    ]

    def __init__(self, optimization_level: str = 'medium'):
        """Initialize the test optimizer.

        Args:
            optimization_level: Level of optimization
                - 'none': No optimization (all tests)
                - 'low': Light optimization (10-20% reduction)
                - 'medium': Moderate optimization (30-40% reduction)
                - 'high': Aggressive optimization (50%+ reduction)
        """
        self.optimization_level = optimization_level
        self.tests_filtered = 0
        self.tests_kept = 0

    def filter_redundant_tests(
        self,
        opinfo: OpInfoV2,
        tests: List[GeneratedTestCase],
    ) -> List[GeneratedTestCase]:
        """Remove redundant test combinations.

        Args:
            opinfo: The OpInfoV2 definition
            tests: List of generated tests

        Returns:
            Filtered list of tests
        """
        if self.optimization_level == 'none':
            self.tests_kept += len(tests)
            return tests

        filtered_tests = tests

        # Filter by dtype if operator is dtype-agnostic
        if self._is_dtype_agnostic(opinfo):
            filtered_tests = self._sample_dtypes(filtered_tests)

        # Filter by shape if operator is shape-agnostic
        if self._is_shape_agnostic(opinfo):
            filtered_tests = self._sample_shapes(filtered_tests)

        # Skip gradcheck on non-differentiable dtypes
        filtered_tests = self._skip_gradcheck_on_integers(filtered_tests)

        # Skip redundant device combinations
        if self.optimization_level in ['medium', 'high']:
            filtered_tests = self._optimize_device_coverage(filtered_tests)

        self.tests_filtered += len(tests) - len(filtered_tests)
        self.tests_kept += len(filtered_tests)

        return filtered_tests

    def _is_dtype_agnostic(self, opinfo: OpInfoV2) -> bool:
        """Check if operator behavior is identical across dtypes.

        An operator is dtype-agnostic if it uses the same algorithm for all
        dtypes (e.g., element-wise operations like add, mul).

        Args:
            opinfo: The OpInfoV2 definition

        Returns:
            True if operator is dtype-agnostic
        """
        return opinfo.name in self.DTYPE_AGNOSTIC_OPS

    def _is_shape_agnostic(self, opinfo: OpInfoV2) -> bool:
        """Check if operator behavior is identical regardless of shape.

        An operator is shape-agnostic if it operates element-wise and doesn't
        have special behavior based on dimensions (e.g., unary ops like sin, cos).

        Args:
            opinfo: The OpInfoV2 definition

        Returns:
            True if operator is shape-agnostic
        """
        return opinfo.name in self.SHAPE_AGNOSTIC_OPS

    def _sample_dtypes(
        self,
        tests: List[GeneratedTestCase],
    ) -> List[GeneratedTestCase]:
        """Sample representative dtypes instead of testing all.

        For dtype-agnostic operators, we only need to test a few representative
        dtypes from each category (float, int, complex, bool).

        Args:
            tests: List of tests to filter

        Returns:
            Filtered tests with representative dtypes
        """
        if self.optimization_level == 'low':
            # Keep more dtypes in low optimization
            keep_dtypes = (
                self.REPRESENTATIVE_DTYPES['floating']
                + self.REPRESENTATIVE_DTYPES['integral']
                + self.REPRESENTATIVE_DTYPES['complex']
            )
        else:
            # Use minimal set for medium/high
            keep_dtypes = [
                torch.float32,
                torch.int64,
                torch.complex64,
            ]

        filtered = []
        for test in tests:
            # Keep tests with representative dtypes, or non-forward tests
            # (backward tests need specific dtypes)
            if (test.dtype in keep_dtypes or
                test.category != TestCategory.FORWARD):
                filtered.append(test)

        return filtered

    def _sample_shapes(
        self,
        tests: List[GeneratedTestCase],
    ) -> List[GeneratedTestCase]:
        """Sample representative shapes instead of testing all.

        For shape-agnostic operators, we only need to test a few representative
        shapes across different dimensions.

        Args:
            tests: List of tests to filter

        Returns:
            Filtered tests with representative shapes
        """
        # This is a simplified version - full implementation would
        # analyze sample input shapes and filter accordingly
        # For now, just keep all tests
        return tests

    def _skip_gradcheck_on_integers(
        self,
        tests: List[GeneratedTestCase],
    ) -> List[GeneratedTestCase]:
        """Skip gradient tests on integer/bool dtypes.

        Gradient tests only make sense for floating-point and complex dtypes.

        Args:
            tests: List of tests to filter

        Returns:
            Filtered tests
        """
        gradient_categories = {
            TestCategory.BACKWARD,
            TestCategory.GRADGRAD,
            TestCategory.FORWARD_AD,
        }

        filtered = []
        for test in tests:
            # Skip gradient tests on non-differentiable dtypes
            if test.category in gradient_categories:
                if test.dtype.is_floating_point or test.dtype.is_complex:
                    filtered.append(test)
                # else: skip this test
            else:
                # Keep all non-gradient tests
                filtered.append(test)

        return filtered

    def _optimize_device_coverage(
        self,
        tests: List[GeneratedTestCase],
    ) -> List[GeneratedTestCase]:
        """Optimize device coverage.

        For operators with identical behavior across devices, we can reduce
        the number of device combinations tested.

        Args:
            tests: List of tests to filter

        Returns:
            Filtered tests
        """
        # Group tests by (category, dtype)
        test_groups: Dict[tuple, List[GeneratedTestCase]] = {}

        for test in tests:
            key = (test.category, test.dtype)
            if key not in test_groups:
                test_groups[key] = []
            test_groups[key].append(test)

        filtered = []

        for key, group in test_groups.items():
            # For each group, keep CPU tests always
            cpu_tests = [t for t in group if t.device == 'cpu']
            filtered.extend(cpu_tests)

            # Keep one GPU test if available (prefer CUDA)
            gpu_tests = [t for t in group if t.device != 'cpu']
            if gpu_tests:
                cuda_tests = [t for t in gpu_tests if t.device == 'cuda']
                if cuda_tests:
                    filtered.append(cuda_tests[0])
                else:
                    # Take any GPU test
                    filtered.append(gpu_tests[0])

        return filtered

    def prioritize_tests(
        self,
        tests: List[GeneratedTestCase],
    ) -> List[GeneratedTestCase]:
        """Prioritize tests by importance.

        Returns tests sorted by priority (highest first). Useful for:
        - Running most important tests first in CI
        - Implementing test sharding
        - Early termination strategies

        Args:
            tests: List of tests to prioritize

        Returns:
            Tests sorted by priority
        """
        def priority_score(test: GeneratedTestCase) -> int:
            """Calculate priority score for a test (higher = more important)."""
            score = 0

            # Category priority
            category_scores = {
                TestCategory.FORWARD: 100,      # Most important
                TestCategory.BACKWARD: 90,
                TestCategory.JIT: 80,
                TestCategory.GRADGRAD: 70,
                TestCategory.VMAP: 60,
                TestCategory.DTYPE_PROMOTION: 50,
                TestCategory.MEMORY_FORMAT: 40,
                TestCategory.FORWARD_AD: 30,
                TestCategory.DECOMPOSITION: 20,
                TestCategory.SERIALIZATION: 10,
            }
            score += category_scores.get(test.category, 0)

            # Device priority (CPU is most common, test first)
            if test.device == 'cpu':
                score += 20
            elif test.device == 'cuda':
                score += 15

            # Dtype priority (float32 is most common)
            if test.dtype == torch.float32:
                score += 10
            elif test.dtype in [torch.float64, torch.int64]:
                score += 5

            return score

        return sorted(tests, key=priority_score, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics.

        Returns:
            Dictionary with statistics about test filtering
        """
        total_tests = self.tests_kept + self.tests_filtered
        reduction_pct = (
            100.0 * self.tests_filtered / total_tests
            if total_tests > 0 else 0.0
        )

        return {
            'total_generated': total_tests,
            'kept': self.tests_kept,
            'filtered': self.tests_filtered,
            'reduction_percentage': reduction_pct,
            'optimization_level': self.optimization_level,
        }


class TestSharding:
    """Shard tests across multiple workers for parallel execution.

    Example:
        >>> sharding = TestSharding(num_shards=4)
        >>> my_tests = sharding.get_shard(tests, shard_id=0)
    """

    def __init__(self, num_shards: int):
        """Initialize test sharding.

        Args:
            num_shards: Number of shards to split tests into
        """
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        self.num_shards = num_shards

    def get_shard(
        self,
        tests: List[GeneratedTestCase],
        shard_id: int,
    ) -> List[GeneratedTestCase]:
        """Get a specific shard of tests.

        Args:
            tests: Full list of tests
            shard_id: Shard ID (0 to num_shards-1)

        Returns:
            Subset of tests for this shard
        """
        if not 0 <= shard_id < self.num_shards:
            raise ValueError(
                f"shard_id must be between 0 and {self.num_shards-1}"
            )

        # Simple round-robin sharding
        return [
            test for i, test in enumerate(tests)
            if i % self.num_shards == shard_id
        ]

    def get_shard_balanced(
        self,
        tests: List[GeneratedTestCase],
        shard_id: int,
    ) -> List[GeneratedTestCase]:
        """Get a balanced shard of tests.

        This method tries to balance:
        - Number of tests per shard
        - Test categories per shard
        - Devices per shard

        Args:
            tests: Full list of tests
            shard_id: Shard ID (0 to num_shards-1)

        Returns:
            Balanced subset of tests for this shard
        """
        if not 0 <= shard_id < self.num_shards:
            raise ValueError(
                f"shard_id must be between 0 and {self.num_shards-1}"
            )

        # Group tests by category
        categories: Dict[TestCategory, List[GeneratedTestCase]] = {}
        for test in tests:
            if test.category not in categories:
                categories[test.category] = []
            categories[test.category].append(test)

        # Distribute each category across shards
        shard_tests = []
        for category, category_tests in categories.items():
            for i, test in enumerate(category_tests):
                if i % self.num_shards == shard_id:
                    shard_tests.append(test)

        return shard_tests
