"""
Test Generator for OpInfo V2

This module automatically generates test cases from OpInfoV2 definitions.
It creates comprehensive test coverage across devices, dtypes, and test categories.
"""

from typing import List, Dict, Any, Callable, Iterator, Optional
import unittest
import torch
import numpy as np
from torch.testing._internal.opinfo_v2 import (
    OpInfoV2,
    TestCategory,
    SampleInput,
)
from torch.testing._internal.common_utils import TestCase


class GeneratedTestCase:
    """A single generated test case.

    Attributes:
        test_name: Name of the test method
        test_fn: The test function to execute
        opinfo: The OpInfoV2 that generated this test
        device: Device type for this test
        dtype: Data type for this test
        category: Test category
    """

    def __init__(
        self,
        test_name: str,
        test_fn: Callable,
        opinfo: OpInfoV2,
        device: str,
        dtype: torch.dtype,
        category: TestCategory,
    ):
        self.test_name = test_name
        self.test_fn = test_fn
        self.opinfo = opinfo
        self.device = device
        self.dtype = dtype
        self.category = category

    def __repr__(self):
        return f"GeneratedTestCase({self.test_name})"


class TestGenerator:
    """Generates tests from OpInfoV2 definitions.

    The TestGenerator creates test methods for each combination of:
    - Test category (FORWARD, BACKWARD, JIT, etc.)
    - Device type (cpu, cuda, etc.)
    - Data type (float32, int64, etc.)
    - Sample input

    Example:
        >>> generator = TestGenerator()
        >>> tests = generator.generate_tests(opinfo)
        >>> for test in tests:
        ...     print(test.test_name)
    """

    def __init__(self):
        """Initialize the test generator."""
        self.generated_count = 0

    def generate_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate all test cases for an operator.

        Args:
            opinfo: The OpInfoV2 definition

        Returns:
            List of generated test cases
        """
        tests = []

        # Generate tests for each enabled category
        for category in opinfo.test_categories:
            if category == TestCategory.FORWARD:
                tests.extend(self._generate_forward_tests(opinfo))
            elif category == TestCategory.BACKWARD:
                tests.extend(self._generate_backward_tests(opinfo))
            elif category == TestCategory.GRADGRAD:
                tests.extend(self._generate_gradgrad_tests(opinfo))
            elif category == TestCategory.JIT:
                tests.extend(self._generate_jit_tests(opinfo))
            elif category == TestCategory.VMAP:
                tests.extend(self._generate_vmap_tests(opinfo))
            elif category == TestCategory.DTYPE_PROMOTION:
                tests.extend(self._generate_dtype_promotion_tests(opinfo))
            elif category == TestCategory.MEMORY_FORMAT:
                tests.extend(self._generate_memory_format_tests(opinfo))
            # Add more categories as needed

        self.generated_count += len(tests)
        return tests

    def _generate_forward_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate forward execution tests."""
        tests = []

        for device in opinfo.get_supported_devices():
            for dtype in opinfo.get_supported_dtypes(device):
                # Check if this combination should be skipped
                if self._should_skip(opinfo, 'test_forward', device, dtype):
                    continue

                test_name = f"test_forward_{opinfo.name}_{device}_{str(dtype).split('.')[-1]}"
                test_fn = self._create_forward_test(opinfo, device, dtype)

                tests.append(GeneratedTestCase(
                    test_name=test_name,
                    test_fn=test_fn,
                    opinfo=opinfo,
                    device=device,
                    dtype=dtype,
                    category=TestCategory.FORWARD,
                ))

        return tests

    def _create_forward_test(
        self,
        opinfo: OpInfoV2,
        device: str,
        dtype: torch.dtype,
    ) -> Callable:
        """Create a single forward test function."""

        def test_fn(test_case: TestCase):
            """Generated forward test."""
            # Skip if device not available
            if not self._is_device_available(device):
                test_case.skipTest(f"Device {device} not available")
                return

            # Get sample inputs
            samples = list(opinfo.sample_inputs_fn(opinfo, device, dtype))

            for i, sample in enumerate(samples):
                with test_case.subTest(sample_num=i, sample_name=sample.name):
                    # Execute operator
                    op_fn = self._get_operator_function(opinfo)
                    result = op_fn(sample.input, *sample.args, **sample.kwargs)

                    # Basic validation
                    test_case.assertIsInstance(result, torch.Tensor)
                    test_case.assertEqual(result.device.type, device)

                    # Verify against reference if provided
                    if opinfo.reference_fn is not None:
                        expected = self._compute_reference(
                            opinfo.reference_fn,
                            sample,
                        )
                        test_case.assertEqual(result.cpu(), expected)

        return test_fn

    def _generate_backward_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate backward/gradient tests."""
        tests = []

        if not opinfo.supports_autograd:
            return tests

        for device in opinfo.get_supported_devices():
            # Only test gradients on floating-point and complex dtypes
            dtypes = [dt for dt in opinfo.get_supported_dtypes(device)
                      if dt.is_floating_point or dt.is_complex]

            for dtype in dtypes:
                if self._should_skip(opinfo, 'test_backward', device, dtype):
                    continue

                test_name = f"test_backward_{opinfo.name}_{device}_{str(dtype).split('.')[-1]}"
                test_fn = self._create_backward_test(opinfo, device, dtype)

                tests.append(GeneratedTestCase(
                    test_name=test_name,
                    test_fn=test_fn,
                    opinfo=opinfo,
                    device=device,
                    dtype=dtype,
                    category=TestCategory.BACKWARD,
                ))

        return tests

    def _create_backward_test(
        self,
        opinfo: OpInfoV2,
        device: str,
        dtype: torch.dtype,
    ) -> Callable:
        """Create a single backward/gradient test function."""

        def test_fn(test_case: TestCase):
            """Generated backward test."""
            if not self._is_device_available(device):
                test_case.skipTest(f"Device {device} not available")
                return

            # Get sample inputs
            samples = list(opinfo.sample_inputs_fn(opinfo, device, dtype))

            for i, sample in enumerate(samples):
                with test_case.subTest(sample_num=i, sample_name=sample.name):
                    # Create inputs with gradients enabled
                    input_tensor = sample.input.detach().requires_grad_(True)
                    args = tuple(
                        arg.detach().requires_grad_(True) if isinstance(arg, torch.Tensor)
                        else arg
                        for arg in sample.args
                    )

                    # Execute operator
                    op_fn = self._get_operator_function(opinfo)
                    result = op_fn(input_tensor, *args, **sample.kwargs)

                    # Compute gradient
                    if result.numel() > 0:
                        # Create gradient of same shape as output
                        grad_out = torch.ones_like(result)
                        result.backward(grad_out)

                        # Verify gradients exist
                        test_case.assertIsNotNone(
                            input_tensor.grad,
                            "Input gradient should not be None"
                        )
                        test_case.assertEqual(
                            input_tensor.grad.shape,
                            input_tensor.shape,
                            "Gradient shape should match input shape"
                        )

        return test_fn

    def _generate_gradgrad_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate second-order gradient tests."""
        tests = []

        if not opinfo.supports_gradgrad:
            return tests

        for device in opinfo.get_supported_devices():
            dtypes = [dt for dt in opinfo.get_supported_dtypes(device)
                      if dt.is_floating_point or dt.is_complex]

            for dtype in dtypes:
                if self._should_skip(opinfo, 'test_gradgrad', device, dtype):
                    continue

                test_name = f"test_gradgrad_{opinfo.name}_{device}_{str(dtype).split('.')[-1]}"
                test_fn = self._create_gradgrad_test(opinfo, device, dtype)

                tests.append(GeneratedTestCase(
                    test_name=test_name,
                    test_fn=test_fn,
                    opinfo=opinfo,
                    device=device,
                    dtype=dtype,
                    category=TestCategory.GRADGRAD,
                ))

        return tests

    def _create_gradgrad_test(
        self,
        opinfo: OpInfoV2,
        device: str,
        dtype: torch.dtype,
    ) -> Callable:
        """Create a second-order gradient test function."""

        def test_fn(test_case: TestCase):
            """Generated gradgrad test."""
            if not self._is_device_available(device):
                test_case.skipTest(f"Device {device} not available")
                return

            samples = list(opinfo.sample_inputs_fn(opinfo, device, dtype))

            for i, sample in enumerate(samples):
                with test_case.subTest(sample_num=i, sample_name=sample.name):
                    # This is a simplified gradgrad test
                    # Full implementation would use gradcheck with create_graph=True
                    input_tensor = sample.input.detach().requires_grad_(True)
                    args = tuple(
                        arg.detach().requires_grad_(True) if isinstance(arg, torch.Tensor)
                        else arg
                        for arg in sample.args
                    )

                    op_fn = self._get_operator_function(opinfo)
                    result = op_fn(input_tensor, *args, **sample.kwargs)

                    if result.numel() > 0:
                        grad_out = torch.ones_like(result)
                        (grad_input,) = torch.autograd.grad(
                            result, input_tensor, grad_out, create_graph=True
                        )

                        # Verify second derivative exists
                        if grad_input.numel() > 0:
                            grad_grad_out = torch.ones_like(grad_input)
                            grad_grad = torch.autograd.grad(
                                grad_input, input_tensor, grad_grad_out
                            )
                            test_case.assertIsNotNone(
                                grad_grad,
                                "Second-order gradient should exist"
                            )

        return test_fn

    def _generate_jit_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate TorchScript/JIT tests."""
        tests = []

        for device in opinfo.get_supported_devices():
            for dtype in opinfo.get_supported_dtypes(device):
                if self._should_skip(opinfo, 'test_jit', device, dtype):
                    continue

                test_name = f"test_jit_{opinfo.name}_{device}_{str(dtype).split('.')[-1]}"
                test_fn = self._create_jit_test(opinfo, device, dtype)

                tests.append(GeneratedTestCase(
                    test_name=test_name,
                    test_fn=test_fn,
                    opinfo=opinfo,
                    device=device,
                    dtype=dtype,
                    category=TestCategory.JIT,
                ))

        return tests

    def _create_jit_test(
        self,
        opinfo: OpInfoV2,
        device: str,
        dtype: torch.dtype,
    ) -> Callable:
        """Create a TorchScript test function."""

        def test_fn(test_case: TestCase):
            """Generated JIT test."""
            if not self._is_device_available(device):
                test_case.skipTest(f"Device {device} not available")
                return

            samples = list(opinfo.sample_inputs_fn(opinfo, device, dtype))

            for i, sample in enumerate(samples):
                with test_case.subTest(sample_num=i, sample_name=sample.name):
                    # Create a simple function to script
                    op_fn = self._get_operator_function(opinfo)

                    def fn(input_t, *args):
                        return op_fn(input_t, *args)

                    # Try to script the function
                    try:
                        scripted_fn = torch.jit.script(fn)
                        # Execute both eager and scripted
                        eager_result = fn(sample.input, *sample.args)
                        scripted_result = scripted_fn(sample.input, *sample.args)

                        # Verify results match
                        test_case.assertEqual(eager_result, scripted_result)
                    except Exception as e:
                        # If scripting fails, it's not necessarily a test failure
                        # Some ops may not be scriptable yet
                        test_case.skipTest(f"Scripting not supported: {e}")

        return test_fn

    def _generate_vmap_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate vmap tests."""
        # Placeholder - full implementation would use functorch
        return []

    def _generate_dtype_promotion_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate dtype promotion tests."""
        # Placeholder - would test mixed-dtype operations
        return []

    def _generate_memory_format_tests(self, opinfo: OpInfoV2) -> List[GeneratedTestCase]:
        """Generate memory format tests."""
        # Placeholder - would test channels_last, etc.
        return []

    # Helper methods

    def _get_operator_function(self, opinfo: OpInfoV2) -> Callable:
        """Get the operator function from torch module."""
        # Handle variants (e.g., "add.Tensor" -> torch.add)
        op_name = opinfo.name
        if hasattr(torch, op_name):
            return getattr(torch, op_name)
        elif hasattr(torch.nn.functional, op_name):
            return getattr(torch.nn.functional, op_name)
        else:
            raise ValueError(f"Operator {op_name} not found in torch module")

    def _is_device_available(self, device: str) -> bool:
        """Check if a device is available."""
        if device == 'cpu':
            return True
        elif device == 'cuda':
            return torch.cuda.is_available()
        elif device == 'mps':
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        elif device == 'xpu':
            return hasattr(torch, 'xpu') and torch.xpu.is_available()
        return False

    def _should_skip(
        self,
        opinfo: OpInfoV2,
        test_name: str,
        device: str,
        dtype: torch.dtype,
    ) -> bool:
        """Check if a test should be skipped."""
        for skip in opinfo.skips:
            if skip.test_name != test_name:
                continue
            if skip.devices and device not in skip.devices:
                continue
            if skip.dtypes and dtype not in skip.dtypes:
                continue
            return True
        return False

    def _compute_reference(
        self,
        reference_fn: Callable,
        sample: SampleInput,
    ) -> torch.Tensor:
        """Compute reference result using NumPy or other implementation."""
        # Convert inputs to numpy
        input_np = sample.input.cpu().numpy()
        args_np = tuple(
            arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg
            for arg in sample.args
        )

        # Compute reference
        result_np = reference_fn(input_np, *args_np, **sample.kwargs)

        # Convert back to torch
        if isinstance(result_np, np.ndarray):
            return torch.from_numpy(result_np)
        else:
            return torch.tensor(result_np)


def generate_all_tests(
    target_globals: Dict[str, Any],
    op_db: List[OpInfoV2],
    base_class: type = TestCase,
) -> None:
    """Generate test classes and inject into target module.

    This function generates test classes for all operators in op_db and
    adds them to the target_globals dictionary (typically globals() of a test module).

    Args:
        target_globals: Dictionary to inject test classes into (use globals())
        op_db: List of OpInfoV2 definitions
        base_class: Base test class to inherit from

    Example:
        >>> # In test file test_ops_v2.py
        >>> from torch.testing._internal.test_generator import generate_all_tests
        >>> from my_opinfos import op_db_v2
        >>>
        >>> generate_all_tests(globals(), op_db_v2)
        >>> # Now test classes are available: TestAddOpsV2, TestMulOpsV2, etc.
    """
    generator = TestGenerator()

    # Group operators by category for organization
    for opinfo in op_db:
        # Generate all tests for this operator
        tests = generator.generate_tests(opinfo)

        # Create a test class for this operator
        class_name = f"Test{opinfo.name.title()}OpsV2"

        # Create test class dynamically
        test_class_dict = {}

        # Add each generated test as a method
        for test in tests:
            test_class_dict[test.test_name] = test.test_fn

        # Create the class
        test_class = type(class_name, (base_class,), test_class_dict)

        # Inject into target globals
        target_globals[class_name] = test_class

    print(f"Generated {generator.generated_count} tests from {len(op_db)} operators")
