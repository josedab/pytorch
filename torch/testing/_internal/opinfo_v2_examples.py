"""
Example OpInfo V2 Definitions

This module demonstrates how to define operators using OpInfo V2.
It includes examples for common operators like add, mul, relu, etc.
"""

import torch
import numpy as np
from torch.testing._internal.opinfo_v2 import (
    OpInfoV2,
    SampleInput,
    TestCategory,
    SkipInfo,
    all_types_and_complex_and,
    floating_types,
    BASIC_TEST_CATEGORIES,
    GRADIENT_TEST_CATEGORIES,
)


# Sample input generators

def sample_inputs_add(opinfo, device, dtype):
    """Generate sample inputs for torch.add."""
    # Simple tensor addition
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        args=(torch.randn(10, device=device, dtype=dtype),),
        name="1d_same_shape"
    )

    # Broadcast addition
    yield SampleInput(
        torch.randn(3, 4, device=device, dtype=dtype),
        args=(torch.randn(4, device=device, dtype=dtype),),
        name="2d_broadcast"
    )

    # With alpha parameter
    yield SampleInput(
        torch.randn(5, device=device, dtype=dtype),
        args=(torch.randn(5, device=device, dtype=dtype),),
        kwargs={'alpha': 2.0},
        name="with_alpha"
    )

    # Zero-dimensional tensors
    yield SampleInput(
        torch.tensor(3.14, device=device, dtype=dtype),
        args=(torch.tensor(2.71, device=device, dtype=dtype),),
        name="0d_tensors"
    )


def sample_inputs_mul(opinfo, device, dtype):
    """Generate sample inputs for torch.mul."""
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        args=(torch.randn(10, device=device, dtype=dtype),),
        name="1d_same_shape"
    )

    yield SampleInput(
        torch.randn(3, 4, 5, device=device, dtype=dtype),
        args=(torch.randn(5, device=device, dtype=dtype),),
        name="3d_broadcast"
    )


def sample_inputs_relu(opinfo, device, dtype):
    """Generate sample inputs for torch.relu."""
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        name="1d"
    )

    yield SampleInput(
        torch.randn(3, 4, 5, device=device, dtype=dtype),
        name="3d"
    )

    # Test with negative values to ensure they become zero
    yield SampleInput(
        torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device, dtype=dtype),
        name="mixed_signs"
    )


def sample_inputs_abs(opinfo, device, dtype):
    """Generate sample inputs for torch.abs."""
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        name="1d"
    )

    yield SampleInput(
        torch.randn(3, 4, device=device, dtype=dtype),
        name="2d"
    )

    # Negative values
    if dtype.is_floating_point or dtype.is_complex:
        yield SampleInput(
            torch.tensor([-1.5, -2.5, 3.5], device=device, dtype=dtype),
            name="negative_values"
        )


def sample_inputs_matmul(opinfo, device, dtype):
    """Generate sample inputs for torch.matmul."""
    # 1D @ 1D (dot product)
    yield SampleInput(
        torch.randn(5, device=device, dtype=dtype),
        args=(torch.randn(5, device=device, dtype=dtype),),
        name="1d_dot_product"
    )

    # 2D @ 2D (matrix multiplication)
    yield SampleInput(
        torch.randn(3, 4, device=device, dtype=dtype),
        args=(torch.randn(4, 5, device=device, dtype=dtype),),
        name="2d_matmul"
    )

    # Batch matrix multiplication
    yield SampleInput(
        torch.randn(10, 3, 4, device=device, dtype=dtype),
        args=(torch.randn(10, 4, 5, device=device, dtype=dtype),),
        name="batch_matmul"
    )


def sample_inputs_sum(opinfo, device, dtype):
    """Generate sample inputs for torch.sum."""
    # Sum all elements
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        name="sum_all"
    )

    # Sum along dimension
    yield SampleInput(
        torch.randn(3, 4, 5, device=device, dtype=dtype),
        kwargs={'dim': 1},
        name="sum_dim1"
    )

    # Sum with keepdim
    yield SampleInput(
        torch.randn(3, 4, device=device, dtype=dtype),
        kwargs={'dim': 0, 'keepdim': True},
        name="sum_keepdim"
    )


# OpInfo V2 Database

op_db_v2 = [
    # Binary arithmetic operators
    OpInfoV2(
        name='add',
        variant='Tensor',
        test_categories=GRADIENT_TEST_CATEGORIES + [TestCategory.DTYPE_PROMOTION],
        dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
        sample_inputs_fn=sample_inputs_add,
        reference_fn=lambda x, y, alpha=1: np.add(x, alpha * y),
        supports_autograd=True,
        supports_gradgrad=True,
        supports_out=True,
        inplace_variant='add_',
        supports_fusion=True,
        supports_channels_last=True,
        references=[
            'https://pytorch.org/docs/stable/generated/torch.add.html',
        ],
    ),

    OpInfoV2(
        name='mul',
        variant='Tensor',
        test_categories=GRADIENT_TEST_CATEGORIES,
        dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
        sample_inputs_fn=sample_inputs_mul,
        reference_fn=np.multiply,
        supports_autograd=True,
        supports_gradgrad=True,
        supports_out=True,
        inplace_variant='mul_',
        supports_fusion=True,
        supports_channels_last=True,
        references=[
            'https://pytorch.org/docs/stable/generated/torch.mul.html',
        ],
    ),

    # Activation functions
    OpInfoV2(
        name='relu',
        test_categories=BASIC_TEST_CATEGORIES,
        dtypes=floating_types(),
        sample_inputs_fn=sample_inputs_relu,
        reference_fn=lambda x: np.maximum(x, 0),
        supports_autograd=True,
        supports_gradgrad=False,  # ReLU has discontinuous second derivative
        supports_out=True,
        inplace_variant='relu_',
        supports_fusion=True,
        supports_channels_last=True,
        references=[
            'https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html',
        ],
    ),

    # Unary operators
    OpInfoV2(
        name='abs',
        test_categories=BASIC_TEST_CATEGORIES,
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        sample_inputs_fn=sample_inputs_abs,
        reference_fn=np.abs,
        supports_autograd=True,
        supports_gradgrad=False,  # abs has discontinuous derivative at 0
        supports_out=True,
        supports_fusion=True,
        supports_channels_last=True,
        # Skip gradient tests on integer dtypes
        skips=[
            SkipInfo(
                'TestGradients',
                'test_backward',
                dtypes=[torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
                reason='Gradient not defined for integer types',
            ),
        ],
        references=[
            'https://pytorch.org/docs/stable/generated/torch.abs.html',
        ],
    ),

    # Linear algebra
    OpInfoV2(
        name='matmul',
        test_categories=GRADIENT_TEST_CATEGORIES,
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        sample_inputs_fn=sample_inputs_matmul,
        # NumPy reference would use np.matmul but with shape handling
        reference_fn=None,  # Complex shape handling, skip reference for now
        supports_autograd=True,
        supports_gradgrad=True,
        supports_out=False,
        supports_fusion=False,  # matmul is already optimized
        min_ndim=1,
        # MPS has some known issues with matmul
        skips=[
            SkipInfo(
                'TestForward',
                'test_forward',
                devices=['mps'],
                dtypes=[torch.float16],
                reason='Known MPS numerical precision issue with float16',
            ),
        ],
        references=[
            'https://pytorch.org/docs/stable/generated/torch.matmul.html',
        ],
    ),

    # Reduction operators
    OpInfoV2(
        name='sum',
        test_categories=GRADIENT_TEST_CATEGORIES,
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        sample_inputs_fn=sample_inputs_sum,
        # NumPy sum with axis handling
        reference_fn=None,  # Complex kwargs handling
        supports_autograd=True,
        supports_gradgrad=True,
        supports_out=True,
        supports_fusion=True,
        references=[
            'https://pytorch.org/docs/stable/generated/torch.sum.html',
        ],
    ),
]


# Utility function to get operator by name
def get_opinfo_by_name(name: str, variant: str = ""):
    """Get OpInfoV2 by name and optional variant.

    Args:
        name: Operator name (e.g., "add")
        variant: Optional variant (e.g., "Tensor")

    Returns:
        OpInfoV2 if found, None otherwise
    """
    for opinfo in op_db_v2:
        if opinfo.name == name and opinfo.variant == variant:
            return opinfo
    return None


# Print summary when module is run
if __name__ == '__main__':
    print("OpInfo V2 Example Database")
    print("=" * 60)
    print(f"Total operators: {len(op_db_v2)}")
    print()

    for opinfo in op_db_v2:
        print(f"{opinfo.full_name}:")
        print(f"  Test categories: {len(opinfo.test_categories)}")
        print(f"  Supports autograd: {opinfo.supports_autograd}")
        print(f"  Supports gradgrad: {opinfo.supports_gradgrad}")
        print(f"  Inplace variant: {opinfo.inplace_variant or 'None'}")
        print()
