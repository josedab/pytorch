# OpInfo V2 - Unified Testing Infrastructure

OpInfo V2 is a declarative testing system that automatically generates comprehensive test coverage for PyTorch operators.

## Quick Example

```python
from torch.testing._internal.opinfo_v2 import OpInfoV2, SampleInput

def sample_inputs_add(opinfo, device, dtype):
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        args=(torch.randn(10, device=device, dtype=dtype),)
    )

op = OpInfoV2(
    name='add',
    sample_inputs_fn=sample_inputs_add,
    supports_autograd=True,
)

# Generate hundreds of tests automatically!
from torch.testing._internal.test_generator import generate_all_tests
generate_all_tests(globals(), [op])
```

## Key Benefits

- **Automatic Test Generation**: Define operator metadata once, get comprehensive tests
- **Consistent Coverage**: Ensures all operators meet the same quality bar
- **Reduced Maintenance**: No need to update hundreds of tests when adding new test dimensions
- **Intelligent Optimization**: Automatically reduces redundant test combinations
- **Coverage Analysis**: Built-in tools to identify and track coverage gaps

## Architecture

```
torch/testing/_internal/
├── opinfo_v2.py              # Core OpInfoV2 schema and data classes
├── test_generator.py         # Automatic test generation from OpInfoV2
├── test_optimizer.py         # Intelligent test selection and sharding
├── opinfo_v2_examples.py     # Example operator definitions
└── OPINFO_V2_README.md       # This file

test/
└── test_ops_v2_example.py    # Example test file with 10+ usage patterns

tools/test_dashboard/
└── coverage_analyzer.py      # Coverage analysis and reporting tool

docs/source/testing/
└── opinfo_v2_guide.md        # Comprehensive user guide
```

## Core Components

### 1. OpInfoV2 Schema (`opinfo_v2.py`)

Defines operator metadata:
- Test categories to generate (forward, backward, JIT, etc.)
- Supported dtypes and devices
- Gradient testing capabilities
- Sample input generator
- Reference implementation (optional)

### 2. Test Generator (`test_generator.py`)

Automatically generates test methods from OpInfoV2 definitions:
- Forward execution tests
- Backward/gradient tests
- Second-order gradient tests
- TorchScript/JIT tests
- And more...

### 3. Test Optimizer (`test_optimizer.py`)

Reduces redundant test combinations:
- Dtype sampling for dtype-agnostic operators
- Shape sampling for shape-agnostic operators
- Skips gradcheck on integer dtypes
- Test prioritization and sharding

### 4. Coverage Analyzer (`coverage_analyzer.py`)

Analyzes test coverage:
- Identifies missing test categories
- Generates JSON and HTML reports
- Tracks coverage over time

## Usage Patterns

### Pattern 1: Simple Auto-Generation

```python
# Define operators
op_db_v2 = [OpInfoV2(...), OpInfoV2(...)]

# Generate all tests
generate_all_tests(globals(), op_db_v2)
```

### Pattern 2: Optimized Tests

```python
generator = TestGenerator()
optimizer = TestOptimizer(optimization_level='medium')

for opinfo in op_db_v2:
    tests = generator.generate_tests(opinfo)
    optimized = optimizer.filter_redundant_tests(opinfo, tests)
    # 30-40% fewer tests!
```

### Pattern 3: Coverage Analysis

```python
from tools.test_dashboard.coverage_analyzer import CoverageAnalyzer

analyzer = CoverageAnalyzer()
report = analyzer.generate_report(op_db_v2)
analyzer.print_report(report)
```

### Pattern 4: Test Sharding

```python
from torch.testing._internal.test_optimizer import TestSharding

sharding = TestSharding(num_shards=4)
my_tests = sharding.get_shard(all_tests, shard_id=0)
```

## File Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `opinfo_v2.py` | Core schema | Import OpInfoV2, TestCategory, etc. |
| `test_generator.py` | Test generation | Generate tests from OpInfoV2 |
| `test_optimizer.py` | Test optimization | Reduce redundant tests |
| `opinfo_v2_examples.py` | Examples | Learn how to define operators |
| `test_ops_v2_example.py` | Example tests | Learn test file patterns |
| `coverage_analyzer.py` | Coverage tools | Analyze coverage gaps |
| `opinfo_v2_guide.md` | Documentation | Full usage guide |

## Getting Started

1. **Read the examples**: Check `opinfo_v2_examples.py` for operator definitions
2. **Try the demo**: Run `python test/test_ops_v2_example.py`
3. **Read the guide**: See `docs/source/testing/opinfo_v2_guide.md`
4. **Define your operator**: Create an OpInfoV2 definition
5. **Generate tests**: Use `generate_all_tests()` in your test file

## Design Philosophy

### Before (Manual Testing)

```python
# 100+ lines of manual test code per operator
class TestAdd(TestCase):
    @dtypes(...)
    def test_add_cpu_float32(self): ...

    @dtypes(...)
    def test_add_cpu_float64(self): ...

    @dtypes(...)
    def test_add_cuda_float32(self): ...

    # ... 50 more test methods
```

### After (OpInfo V2)

```python
# 20 lines of declarative metadata
OpInfoV2(
    name='add',
    sample_inputs_fn=sample_inputs_add,
    test_categories=BASIC_TEST_CATEGORIES,
    supports_autograd=True,
)

# Tests auto-generated!
```

## Performance Impact

Based on RFC-0002 projections:

- **Coverage**: 95%+ operators with comprehensive tests
- **CI Time**: 40% reduction through intelligent test selection
- **Maintenance**: 50% reduction in time to add new test dimensions
- **Quality**: 20% increase in bugs caught before merge

## Implementation Status

**Phase 1 - Infrastructure** (Completed):
- ✅ OpInfoV2 schema
- ✅ TestGenerator
- ✅ TestOptimizer
- ✅ Coverage analysis tools
- ✅ Example operators and tests
- ✅ Documentation

**Phase 2 - Migration** (TODO):
- ⏳ Port top 100 operators
- ⏳ Create automated migration tool
- ⏳ Integrate into CI

**Phase 3 - Coverage Expansion** (TODO):
- ⏳ Add missing test categories (VMAP, DECOMPOSITION)
- ⏳ Fill coverage gaps
- ⏳ Migrate remaining operators

## Related Work

- **Current OpInfo**: `torch/testing/_internal/common_methods_invocations.py`
- **Device-Type Testing**: `torch/testing/_internal/common_device_type.py`
- **Test Utilities**: `torch/testing/_internal/common_utils.py`
- **RFC-0002**: See `rfcs/RFC-0002-unified-testing-infrastructure.md`

## Contributing

When adding new operators:

1. Create OpInfoV2 definition with comprehensive sample inputs
2. Set appropriate test categories and capabilities
3. Run coverage analysis to verify coverage
4. Add skips for known issues with documentation

When extending the framework:

1. Add new TestCategory if needed
2. Implement generator method in `test_generator.py`
3. Add optimization logic in `test_optimizer.py` if applicable
4. Update documentation

## Support

- **Documentation**: `docs/source/testing/opinfo_v2_guide.md`
- **Examples**: `torch/testing/_internal/opinfo_v2_examples.py`
- **Issues**: File GitHub issues with label `module: testing`
- **Discussions**: PyTorch Forums - Testing category

## License

PyTorch is BSD-licensed. See LICENSE file.
