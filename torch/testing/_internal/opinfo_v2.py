"""
OpInfo V2: Unified Testing Infrastructure

This module provides a declarative way to specify test coverage for PyTorch operators.
OpInfo V2 automatically generates comprehensive test coverage across devices, dtypes,
and test categories, reducing maintenance burden and ensuring consistent testing standards.

See RFC-0002 for full design details.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any, Tuple
from enum import Enum, auto
import torch


class TestCategory(Enum):
    """Test categories to run for an operator.

    Each category represents a different aspect of operator behavior that should be tested.
    """
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
class SampleInput:
    """A single test input for an operator.

    Attributes:
        input: The primary input tensor
        args: Positional arguments to the operator
        kwargs: Keyword arguments to the operator
        name: Optional name for this sample (for debugging)
    """
    input: torch.Tensor
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __repr__(self):
        name_str = f"'{self.name}' " if self.name else ""
        return f"SampleInput({name_str}shape={tuple(self.input.shape)}, dtype={self.input.dtype})"


@dataclass
class SkipInfo:
    """Information about a test skip.

    Attributes:
        test_class: Name of the test class to skip (e.g., 'TestGradients')
        test_name: Name of the test method to skip (e.g., 'test_fn_grad')
        dtypes: List of dtypes to skip (None means skip for all dtypes)
        devices: List of devices to skip (None means skip for all devices)
        reason: Human-readable reason for the skip
    """
    test_class: str
    test_name: str
    dtypes: Optional[List[torch.dtype]] = None
    devices: Optional[List[str]] = None
    reason: str = ""


@dataclass
class FailureInfo:
    """Information about an expected test failure.

    Similar to SkipInfo but represents tests that run but are expected to fail.
    """
    test_class: str
    test_name: str
    dtypes: Optional[List[torch.dtype]] = None
    devices: Optional[List[str]] = None
    reason: str = ""


@dataclass
class OpInfoV2:
    """Declarative operator test specification.

    This class defines all the metadata needed to automatically generate comprehensive
    tests for a PyTorch operator. Tests are generated based on the specified test
    categories, supported dtypes/devices, and provided sample inputs.

    Example:
        >>> def sample_inputs_add(opinfo, device, dtype):
        ...     yield SampleInput(
        ...         torch.randn(10, device=device, dtype=dtype),
        ...         args=(torch.randn(10, device=device, dtype=dtype),)
        ...     )
        ...
        >>> add_opinfo = OpInfoV2(
        ...     name='add',
        ...     variant='Tensor',
        ...     test_categories=[
        ...         TestCategory.FORWARD,
        ...         TestCategory.BACKWARD,
        ...         TestCategory.JIT,
        ...     ],
        ...     sample_inputs_fn=sample_inputs_add,
        ... )

    Attributes:
        name: Operator name (e.g., "add", "relu", "conv2d")
        variant: Variant name for overloads (e.g., "Tensor", "Scalar")
        test_categories: List of test categories to generate tests for
        dtypes: Supported dtypes (None means all dtypes)
        complex_dtypes: Whether to test complex dtypes
        integral_dtypes: Whether to test integral dtypes
        floating_dtypes: Whether to test floating point dtypes
        supports_cpu: Whether operator works on CPU
        supports_cuda: Whether operator works on CUDA
        supports_mps: Whether operator works on MPS (Apple Silicon)
        supports_xpu: Whether operator works on XPU (Intel)
        min_ndim: Minimum number of dimensions required
        max_ndim: Maximum number of dimensions supported
        supports_channels_last: Whether channels-last memory format is supported
        supports_sparse: Whether sparse tensors are supported
        sample_inputs_fn: Function that generates test inputs
        reference_fn: Optional reference implementation (e.g., NumPy)
        supports_autograd: Whether operator supports autograd
        supports_gradgrad: Whether operator supports second-order gradients
        supports_forward_ad: Whether operator supports forward-mode AD
        gradcheck_kwargs: Custom arguments for gradcheck
        supports_fusion: Whether operator can be fused
        supports_out: Whether operator supports out= parameter
        inplace_variant: Name of inplace variant (e.g., "add_" for "add")
        skips: List of test skips
        expected_failures: List of expected failures
        references: Links to documentation, papers, etc.
    """

    # Basic metadata
    name: str
    variant: str = ""

    # Supported test categories (automatic test generation)
    test_categories: List[TestCategory] = field(default_factory=lambda: [
        TestCategory.FORWARD,
        TestCategory.BACKWARD,
        TestCategory.JIT,
    ])

    # Type support
    dtypes: Optional[List[torch.dtype]] = None  # None = all dtypes
    complex_dtypes: bool = True
    integral_dtypes: bool = True
    floating_dtypes: bool = True

    # Device support
    supports_cpu: bool = True
    supports_cuda: bool = True
    supports_mps: bool = True
    supports_xpu: bool = True

    # Shape/memory constraints
    min_ndim: int = 0
    max_ndim: Optional[int] = None
    supports_channels_last: bool = False
    supports_sparse: bool = False

    # Test input generation
    sample_inputs_fn: Optional[Callable] = None
    reference_fn: Optional[Callable] = None

    # Gradient testing
    supports_autograd: bool = True
    supports_gradgrad: bool = False
    supports_forward_ad: bool = False
    gradcheck_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Performance/optimization
    supports_fusion: bool = False
    supports_out: bool = False
    inplace_variant: Optional[str] = None

    # Test exclusions/known issues
    skips: List[SkipInfo] = field(default_factory=list)
    expected_failures: List[FailureInfo] = field(default_factory=list)

    # Documentation
    references: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate OpInfo configuration."""
        if self.sample_inputs_fn is None:
            raise ValueError(f"OpInfoV2 for '{self.name}' must provide sample_inputs_fn")

        # Validate test categories
        if TestCategory.BACKWARD in self.test_categories and not self.supports_autograd:
            raise ValueError(
                f"OpInfoV2 for '{self.name}' includes BACKWARD tests but "
                f"supports_autograd=False"
            )

        if TestCategory.GRADGRAD in self.test_categories and not self.supports_gradgrad:
            raise ValueError(
                f"OpInfoV2 for '{self.name}' includes GRADGRAD tests but "
                f"supports_gradgrad=False"
            )

    @property
    def full_name(self) -> str:
        """Return full operator name including variant."""
        if self.variant:
            return f"{self.name}.{self.variant}"
        return self.name

    def get_supported_devices(self) -> List[str]:
        """Get list of supported device types."""
        devices = []
        if self.supports_cpu:
            devices.append('cpu')
        if self.supports_cuda:
            devices.append('cuda')
        if self.supports_mps:
            devices.append('mps')
        if self.supports_xpu:
            devices.append('xpu')
        return devices

    def get_supported_dtypes(self, device: str = 'cpu') -> List[torch.dtype]:
        """Get list of supported dtypes.

        Args:
            device: Device type (used to filter dtypes not available on device)

        Returns:
            List of supported dtypes for the given device
        """
        if self.dtypes is not None:
            return self.dtypes

        # Build dtype list based on flags
        dtypes = []

        if self.floating_dtypes:
            dtypes.extend([torch.float32, torch.float64])
            if device != 'cpu':  # half precision mainly on GPU
                dtypes.extend([torch.float16, torch.bfloat16])

        if self.integral_dtypes:
            dtypes.extend([torch.int32, torch.int64, torch.int16, torch.int8])
            dtypes.extend([torch.uint8])

        if self.complex_dtypes:
            dtypes.extend([torch.complex64, torch.complex128])

        return dtypes


# Utility functions for creating common dtype lists
def all_types():
    """All numeric types (int and float, no complex)."""
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.float32, torch.float64, torch.bfloat16]


def all_types_and_complex():
    """All numeric types including complex."""
    return all_types() + [torch.complex64, torch.complex128]


def all_types_and(*extra_dtypes):
    """All numeric types plus additional types."""
    return all_types() + list(extra_dtypes)


def all_types_and_complex_and(*extra_dtypes):
    """All numeric types including complex plus additional types."""
    return all_types_and_complex() + list(extra_dtypes)


def floating_types():
    """All floating point types."""
    return [torch.float16, torch.float32, torch.float64, torch.bfloat16]


def floating_and_complex_types():
    """All floating point and complex types."""
    return floating_types() + [torch.complex64, torch.complex128]


def integral_types():
    """All integer types."""
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


# Predefined test category groups
ALL_TEST_CATEGORIES = [
    TestCategory.FORWARD,
    TestCategory.BACKWARD,
    TestCategory.GRADGRAD,
    TestCategory.FORWARD_AD,
    TestCategory.JIT,
    TestCategory.VMAP,
    TestCategory.DECOMPOSITION,
    TestCategory.SERIALIZATION,
    TestCategory.DTYPE_PROMOTION,
    TestCategory.MEMORY_FORMAT,
]

BASIC_TEST_CATEGORIES = [
    TestCategory.FORWARD,
    TestCategory.BACKWARD,
    TestCategory.JIT,
]

GRADIENT_TEST_CATEGORIES = [
    TestCategory.FORWARD,
    TestCategory.BACKWARD,
    TestCategory.GRADGRAD,
    TestCategory.FORWARD_AD,
]
