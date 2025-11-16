# RFC-0002: Unified Testing Infrastructure with OpInfo V2

**Status:** Draft
**Author:** Claude Code Analysis
**Created:** 2025-11-16
**Based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Summary

Create a unified, declarative testing infrastructure (OpInfo V2) that automatically generates comprehensive test coverage for all PyTorch operators across devices, dtypes, and edge cases. This reduces test maintenance burden, increases coverage, and provides consistent testing standards.

## Motivation

### Current Testing Challenges

The existing test infrastructure (located in [`torch/testing/_internal/common_methods_invocations.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_methods_invocations.py)) has grown organically to over 1.2 MB with 1000+ OpInfo definitions, leading to:

1. **Inconsistent Coverage:**
   - Some operators have comprehensive tests (grad, gradgrad, JIT, etc.)
   - Others have minimal tests (only forward pass)
   - No systematic way to ensure all operators meet quality bar

2. **Test Duplication:**
   - Similar test patterns repeated across test files
   - Device-type tests, dtype tests, shape tests duplicated
   - Each new test type requires updates across hundreds of operators

3. **Maintenance Burden:**
   - Adding a new test dimension (e.g., channels_last support) requires updating 1000+ test cases
   - No automated way to identify coverage gaps
   - Test failures are hard to debug due to complex parametrization

4. **Slow Test Execution:**
   - Redundant test combinations (testing all dtypes when only float matters)
   - No intelligent test selection based on operator properties
   - CI runtime has grown from 30 minutes to 2+ hours

### Opportunity

A unified, declarative testing system can:
- Automatically generate tests based on operator metadata
- Eliminate test duplication
- Ensure consistent coverage
- Reduce CI time by 30-40% through intelligent test selection

## Detailed Design

### OpInfo V2 Schema

**Location:** New file `torch/testing/_internal/opinfo_v2.py`

```python
from dataclasses import dataclass
from typing import List, Callable, Optional
from enum import Enum, auto

class TestCategory(Enum):
    """Test categories to run for an operator"""
    FORWARD = auto()           # Basic forward execution
    BACKWARD = auto()          # First-order gradients
    GRADGRAD = auto()          # Second-order gradients
    FORWARD_AD = auto()        # Forward-mode autodiff
    JIT = auto()               # TorchScript compilation
    VMAP = auto()              # functorch vmap
    DECOMPOSITION = auto()     # Decomposition into primitives
    SERIALIZATION = auto()     # Pickle/TorchScript save/load
    DTYPE_PROMOTION = auto()   # Type promotion rules
    MEMORY_FORMAT = auto()     # Contiguous/channels_last support
    SPARSE = auto()            # Sparse tensor support
    COMPLEX = auto()           # Complex number support

@dataclass
class OpInfoV2:
    """Declarative operator test specification"""

    # Basic metadata
    name: str                              # Operator name (e.g., "add")
    variant: str = ""                      # Variant name (e.g., "Tensor", "Scalar")

    # Supported test categories (automatic test generation)
    test_categories: List[TestCategory] = field(default_factory=lambda: [
        TestCategory.FORWARD,
        TestCategory.BACKWARD,
        TestCategory.JIT,
    ])

    # Type support
    dtypes: Optional[List[torch.dtype]] = None  # Supported dtypes (None = all)
    complex_dtypes: bool = True                  # Supports complex numbers
    integral_dtypes: bool = True                 # Supports integers
    floating_dtypes: bool = True                 # Supports floats

    # Device support
    supports_cpu: bool = True
    supports_cuda: bool = True
    supports_mps: bool = True
    supports_xpu: bool = True

    # Shape/memory constraints
    min_ndim: int = 0                     # Minimum dimensions
    max_ndim: Optional[int] = None        # Maximum dimensions
    supports_channels_last: bool = False  # Channels-last memory format
    supports_sparse: bool = False         # Sparse tensors

    # Test input generation
    sample_inputs_fn: Callable            # Generate test inputs
    reference_fn: Optional[Callable] = None  # Reference implementation (NumPy, etc.)

    # Gradient testing
    supports_autograd: bool = True
    supports_gradgrad: bool = False
    supports_forward_ad: bool = False
    gradcheck_kwargs: dict = field(default_factory=dict)  # Custom gradcheck args

    # Performance/optimization
    supports_fusion: bool = False         # Can be fused with other ops
    supports_out: bool = False            # Supports out= parameter
    inplace_variant: Optional[str] = None # Inplace version (e.g., "add_" for "add")

    # Test exclusions/known issues
    skips: List[SkipInfo] = field(default_factory=list)  # Skip certain combinations
    expected_failures: List[FailureInfo] = field(default_factory=list)

    # Documentation
    references: List[str] = field(default_factory=list)  # Links to docs, papers

# Example usage:
op_db_v2 = [
    OpInfoV2(
        name='add',
        variant='Tensor',
        test_categories=[
            TestCategory.FORWARD,
            TestCategory.BACKWARD,
            TestCategory.GRADGRAD,
            TestCategory.JIT,
            TestCategory.VMAP,
            TestCategory.DTYPE_PROMOTION,
        ],
        dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
        sample_inputs_fn=sample_inputs_add_tensor,
        reference_fn=lambda x, y, alpha=1: x + alpha * y,  # NumPy reference
        supports_fusion=True,
        supports_out=True,
        inplace_variant='add_',
        supports_channels_last=True,
        skips=[
            SkipInfo('TestGradients', 'test_fn_grad', dtypes=[torch.bool]),
        ],
    ),
]
```

### Automatic Test Generation

**Location:** New file `torch/testing/_internal/test_generator.py`

```python
class TestGenerator:
    """Generates tests from OpInfoV2 definitions"""

    def generate_tests(self, opinfo: OpInfoV2) -> List[TestCase]:
        """Generate all test cases for an operator"""
        tests = []

        # Generate tests for each enabled category
        if TestCategory.FORWARD in opinfo.test_categories:
            tests.extend(self._generate_forward_tests(opinfo))

        if TestCategory.BACKWARD in opinfo.test_categories:
            tests.extend(self._generate_backward_tests(opinfo))

        if TestCategory.JIT in opinfo.test_categories:
            tests.extend(self._generate_jit_tests(opinfo))

        # ... more categories

        return tests

    def _generate_forward_tests(self, opinfo: OpInfoV2) -> List[TestCase]:
        """Generate forward execution tests"""
        tests = []

        # Test on each supported device
        for device in self._get_supported_devices(opinfo):
            # Test on each supported dtype
            for dtype in self._get_supported_dtypes(opinfo):
                # Generate sample inputs
                for sample in opinfo.sample_inputs_fn(opinfo, device, dtype):
                    # Create test case
                    test = self._create_forward_test(
                        opinfo, device, dtype, sample
                    )
                    tests.append(test)

        return tests

    def _create_forward_test(self, opinfo, device, dtype, sample):
        """Create a single forward test case"""
        def test_fn(self):
            # Execute operator
            result = getattr(torch, opinfo.name)(
                sample.input, *sample.args, **sample.kwargs
            )

            # Verify against reference (if provided)
            if opinfo.reference_fn:
                expected = opinfo.reference_fn(
                    sample.input.cpu().numpy(),
                    *(arg.cpu().numpy() for arg in sample.args)
                )
                self.assertEqual(result.cpu().numpy(), expected)

            # Verify properties
            self.assertEqual(result.device, sample.input.device)
            self.assertEqual(result.dtype, sample.input.dtype)

        return test_fn
```

### Intelligent Test Selection

**Goal:** Only run tests that matter for each operator

```python
class TestOptimizer:
    """Optimize test selection based on operator properties"""

    def filter_redundant_tests(self, opinfo: OpInfoV2, tests: List[TestCase]) -> List[TestCase]:
        """Remove redundant test combinations"""

        # If operator is dtype-agnostic, only test representative dtypes
        if self._is_dtype_agnostic(opinfo):
            tests = self._sample_dtypes(tests, [torch.float32, torch.int64, torch.complex64])

        # If operator is shape-agnostic, reduce shape combinations
        if self._is_shape_agnostic(opinfo):
            tests = self._sample_shapes(tests, [(5,), (3, 4), (2, 3, 4)])

        # Skip gradcheck on non-differentiable dtypes
        if TestCategory.BACKWARD in opinfo.test_categories:
            tests = self._skip_gradcheck_on_integers(tests)

        return tests

    def _is_dtype_agnostic(self, opinfo: OpInfoV2) -> bool:
        """Check if operator behavior is identical across dtypes"""
        # Example: add, mul are dtype-agnostic (same algorithm for all types)
        # Counter-example: sigmoid is NOT dtype-agnostic (numerical differences)
        return opinfo.name in ['add', 'sub', 'mul', 'div', 'neg', 'abs']
```

### Test Coverage Dashboard

**Location:** New tool `tools/test_dashboard/coverage_analyzer.py`

```python
class CoverageAnalyzer:
    """Analyze test coverage across all operators"""

    def generate_report(self, op_db: List[OpInfoV2]) -> CoverageReport:
        """Generate coverage report"""

        report = CoverageReport()

        for opinfo in op_db:
            # Check coverage for each category
            coverage = {
                'forward': TestCategory.FORWARD in opinfo.test_categories,
                'backward': TestCategory.BACKWARD in opinfo.test_categories,
                'gradgrad': TestCategory.GRADGRAD in opinfo.test_categories,
                'jit': TestCategory.JIT in opinfo.test_categories,
                'vmap': TestCategory.VMAP in opinfo.test_categories,
            }

            report.add_operator(opinfo.name, coverage)

        return report

    def identify_gaps(self, report: CoverageReport) -> List[CoverageGap]:
        """Identify operators with insufficient coverage"""
        gaps = []

        for op, coverage in report.items():
            # All operators should have backward tests if supports_autograd
            if not coverage['backward'] and op.supports_autograd:
                gaps.append(CoverageGap(op, 'backward', 'Missing gradient tests'))

            # All operators should have JIT tests
            if not coverage['jit']:
                gaps.append(CoverageGap(op, 'jit', 'Missing TorchScript tests'))

        return gaps
```

## Example Usage

### Before (Current System):

**test/test_ops.py** (scattered tests, manual maintenance)

```python
class TestBinaryUfuncs(TestCase):
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_add_cpu(self, device, dtype):
        # Manual test implementation
        a = torch.randn(10, device=device, dtype=dtype)
        b = torch.randn(10, device=device, dtype=dtype)
        c = torch.add(a, b)
        # ... assertions

    @dtypes(*floating_types_and(torch.half))
    def test_add_grad(self, device, dtype):
        # Manual gradient test
        a = torch.randn(10, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(10, device=device, dtype=dtype, requires_grad=True)
        # ... gradcheck

    # ... 50 more manual test methods for 'add'
```

### After (OpInfo V2):

**torch/testing/_internal/opinfo_v2.py** (declarative, automatic)

```python
op_db_v2 = [
    OpInfoV2(
        name='add',
        variant='Tensor',
        test_categories=ALL_TEST_CATEGORIES,  # Run all applicable tests
        sample_inputs_fn=sample_inputs_add,
        reference_fn=lambda x, y, alpha=1: np.add(x, alpha * y),
    ),
    # That's it! All tests generated automatically
]
```

**test/test_ops_v2.py** (auto-generated)

```python
from torch.testing._internal.test_generator import generate_all_tests

# Automatically generates 1000+ test methods from op_db_v2
generate_all_tests(globals(), op_db_v2)
```

## Implementation Plan

### Phase 1: Infrastructure (2 months)

**Deliverables:**
1. Implement OpInfoV2 schema and dataclasses
2. Implement TestGenerator for automatic test generation
3. Implement TestOptimizer for intelligent test selection
4. Port 10 representative operators to OpInfoV2 (add, mul, relu, conv2d, etc.)

**Success Criteria:**
- OpInfoV2 operators have equivalent coverage to existing tests
- Generated tests run in CI without failures
- Test execution time reduced by 20% for ported operators

### Phase 2: Migration (4 months)

**Deliverables:**
1. Port top 100 operators (by usage frequency)
2. Create automated migration tool
3. Implement coverage dashboard

**Success Criteria:**
- 80% of operators migrated
- Coverage dashboard shows gaps
- CI time reduced by 30%

### Phase 3: Coverage Expansion (3 months)

**Deliverables:**
1. Add missing test categories (VMAP, DECOMPOSITION, etc.)
2. Fill coverage gaps identified by dashboard
3. Migrate remaining operators

**Success Criteria:**
- 100% operators migrated
- 95%+ coverage across all test categories
- CI time reduced by 40%

## Backwards Compatibility

### Migration Strategy

1. **Parallel systems:** OpInfo V1 and V2 coexist during migration
2. **Gradual migration:** Operators migrated incrementally
3. **No API changes:** Python/C++ operator APIs unchanged

### Third-Party Impact

**External test writers:**
- Continue using device-type parametrization
- Optional: Adopt OpInfoV2 for their operators
- Provide migration guide

## Alternatives Considered

### Alternative 1: Improve Existing OpInfo

**Pros:**
- Lower migration cost
- Familiar to contributors

**Cons:**
- Can't address fundamental design issues
- Technical debt persists

**Decision:** Rejected - incremental improvement insufficient

### Alternative 2: Property-Based Testing (Hypothesis)

Use property-based testing instead of example-based:

```python
from hypothesis import given, strategies as st

@given(st.tensors())
def test_add_properties(x, y):
    # Test properties (commutativity, associativity, etc.)
    assert torch.add(x, y) == torch.add(y, x)
```

**Pros:**
- Excellent for finding edge cases
- Less manual test writing

**Cons:**
- Slower test execution
- Harder to debug failures
- Requires property definitions for each operator

**Decision:** Consider as complement, not replacement

### Alternative 3: Fuzzing-Based Testing

Continuous fuzzing to find crashes:

```python
while True:
    inputs = generate_random_inputs()
    try:
        result = torch.add(*inputs)
    except Exception as e:
        report_crash(e, inputs)
```

**Pros:**
- Finds obscure bugs
- No manual test writing

**Cons:**
- No gradient/JIT coverage
- Hard to ensure consistent coverage

**Decision:** Complement to OpInfo V2, not replacement

## Open Questions

1. **Test prioritization in CI?**
   - Run fast smoke tests first
   - Run comprehensive tests nightly
   - Need cost model for test execution time

2. **Dynamic test generation in CI vs. ahead-of-time?**
   - AOT: Faster test collection, but large test file
   - Dynamic: Smaller codebase, but slower collection
   - Proposal: Dynamic for developers, AOT for CI

3. **Backwards compatibility with PyTest fixtures?**
   - Current system uses pytest parametrize
   - OpInfo V2 generates test methods programmatically
   - Can we support both?

4. **Test result caching?**
   - Cache test results for unchanged operators
   - Invalidate cache on operator code changes
   - Proposal: Integrate with pytest-cache

## Success Metrics

1. **Coverage:** 95%+ operators have backward, gradgrad, and JIT tests
2. **CI Time:** 40% reduction in total test execution time
3. **Maintenance:** 50% reduction in time to add new test dimension
4. **Quality:** 20% increase in bugs caught before merge (from better coverage)
5. **Developer Experience:** 80%+ positive feedback on OpInfo V2 from contributors

## References

- **Current OpInfo:** [`torch/testing/_internal/common_methods_invocations.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_methods_invocations.py)
- **Device-Type Testing:** [`torch/testing/_internal/common_device_type.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_device_type.py)
- **Test Utilities:** [`torch/testing/_internal/common_utils.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_utils.py)
- **Related Work:** TensorFlow's tf.test infrastructure, JAX's jax.test_util

---

**Next Steps:**
1. Gather feedback from PyTorch testing team
2. Create prototype with 10 operators
3. Measure coverage and performance improvements
4. Refine design based on results
5. Submit for formal review
