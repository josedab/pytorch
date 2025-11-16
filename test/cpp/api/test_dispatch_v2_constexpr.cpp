#include <gtest/gtest.h>

#include <ATen/Dispatch_v2_constexpr.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/complex.h>

/**
 * Tests for AT_DISPATCH_V2_CONSTEXPR dispatch system
 *
 * This test suite validates the template-based constexpr dispatch implementation
 * proposed in RFC-0001-dispatch-v2-performance.md
 */

// Test helper: returns the size of the scalar type in bytes
template <typename scalar_t>
size_t get_scalar_size() {
  return sizeof(scalar_t);
}

// Test helper: returns a string representation of the type
template <typename scalar_t>
std::string get_type_name() {
  if (std::is_same_v<scalar_t, float>)
    return "float";
  if (std::is_same_v<scalar_t, double>)
    return "double";
  if (std::is_same_v<scalar_t, int>)
    return "int";
  if (std::is_same_v<scalar_t, int64_t>)
    return "int64_t";
  if (std::is_same_v<scalar_t, int8_t>)
    return "int8_t";
  if (std::is_same_v<scalar_t, uint8_t>)
    return "uint8_t";
  if (std::is_same_v<scalar_t, int16_t>)
    return "int16_t";
  if (std::is_same_v<scalar_t, c10::Half>)
    return "Half";
  if (std::is_same_v<scalar_t, c10::BFloat16>)
    return "BFloat16";
  if (std::is_same_v<scalar_t, c10::complex<float>>)
    return "ComplexFloat";
  if (std::is_same_v<scalar_t, c10::complex<double>>)
    return "ComplexDouble";
  if (std::is_same_v<scalar_t, bool>)
    return "bool";
  return "unknown";
}

TEST(DispatchV2ConstexprTest, FloatingTypes) {
  // Test dispatching to float
  auto result_float = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Float,
      "test_float",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_float, sizeof(float));

  // Test dispatching to double
  auto result_double = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Double,
      "test_double",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_double, sizeof(double));
}

TEST(DispatchV2ConstexprTest, FloatingTypesAndHalf) {
  // Test dispatching to Half
  auto result_half = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES_AND_HALF(
      at::ScalarType::Half,
      "test_half",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_half, sizeof(c10::Half));

  // Test that float still works
  auto result_float = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES_AND_HALF(
      at::ScalarType::Float,
      "test_float",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_float, sizeof(float));
}

TEST(DispatchV2ConstexprTest, IntegralTypes) {
  // Test various integral types
  auto result_int8 = AT_DISPATCH_V2_CONSTEXPR_INTEGRAL_TYPES(
      at::ScalarType::Char,
      "test_int8",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_int8, sizeof(int8_t));

  auto result_int16 = AT_DISPATCH_V2_CONSTEXPR_INTEGRAL_TYPES(
      at::ScalarType::Short,
      "test_int16",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_int16, sizeof(int16_t));

  auto result_int32 = AT_DISPATCH_V2_CONSTEXPR_INTEGRAL_TYPES(
      at::ScalarType::Int,
      "test_int32",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_int32, sizeof(int));

  auto result_int64 = AT_DISPATCH_V2_CONSTEXPR_INTEGRAL_TYPES(
      at::ScalarType::Long,
      "test_int64",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_int64, sizeof(int64_t));

  auto result_uint8 = AT_DISPATCH_V2_CONSTEXPR_INTEGRAL_TYPES(
      at::ScalarType::Byte,
      "test_uint8",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_uint8, sizeof(uint8_t));
}

TEST(DispatchV2ConstexprTest, AllTypes) {
  // Test that all types work with AT_DISPATCH_V2_CONSTEXPR_ALL_TYPES
  std::vector<at::ScalarType> types_to_test = {
      at::ScalarType::Byte,
      at::ScalarType::Char,
      at::ScalarType::Short,
      at::ScalarType::Int,
      at::ScalarType::Long,
      at::ScalarType::Float,
      at::ScalarType::Double,
  };

  for (auto dtype : types_to_test) {
    auto result = AT_DISPATCH_V2_CONSTEXPR_ALL_TYPES(
        dtype, "test_all_types", [&]<typename scalar_t>() {
          return get_scalar_size<scalar_t>();
        });

    // Verify the size matches expectations
    EXPECT_GT(result, 0);
  }
}

TEST(DispatchV2ConstexprTest, ComplexTypes) {
  // Test complex float
  auto result_complex_float = AT_DISPATCH_V2_CONSTEXPR_COMPLEX_TYPES(
      at::ScalarType::ComplexFloat,
      "test_complex_float",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_complex_float, sizeof(c10::complex<float>));

  // Test complex double
  auto result_complex_double = AT_DISPATCH_V2_CONSTEXPR_COMPLEX_TYPES(
      at::ScalarType::ComplexDouble,
      "test_complex_double",
      [&]<typename scalar_t>() { return get_scalar_size<scalar_t>(); });

  EXPECT_EQ(result_complex_double, sizeof(c10::complex<double>));
}

TEST(DispatchV2ConstexprTest, FloatingAndComplexTypes) {
  // Test that both floating and complex types work
  std::vector<at::ScalarType> types_to_test = {
      at::ScalarType::Float,
      at::ScalarType::Double,
      at::ScalarType::ComplexFloat,
      at::ScalarType::ComplexDouble,
  };

  for (auto dtype : types_to_test) {
    auto result = AT_DISPATCH_V2_CONSTEXPR_FLOATING_AND_COMPLEX_TYPES(
        dtype, "test_floating_and_complex", [&]<typename scalar_t>() {
          return get_type_name<scalar_t>();
        });

    // Verify we got a valid type name
    EXPECT_NE(result, "unknown");
  }
}

TEST(DispatchV2ConstexprTest, AllTypesAndComplex) {
  // Test all types including complex
  std::vector<at::ScalarType> types_to_test = {
      at::ScalarType::Byte,
      at::ScalarType::Char,
      at::ScalarType::Short,
      at::ScalarType::Int,
      at::ScalarType::Long,
      at::ScalarType::Float,
      at::ScalarType::Double,
      at::ScalarType::ComplexFloat,
      at::ScalarType::ComplexDouble,
  };

  for (auto dtype : types_to_test) {
    auto result = AT_DISPATCH_V2_CONSTEXPR_ALL_TYPES_AND_COMPLEX(
        dtype, "test_all_types_and_complex", [&]<typename scalar_t>() {
          return get_type_name<scalar_t>();
        });

    // Verify we got a valid type name
    EXPECT_NE(result, "unknown");
  }
}

TEST(DispatchV2ConstexprTest, TypeNameRetrieval) {
  // Test that we can retrieve type information correctly
  auto float_name = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Float,
      "test_type_name",
      [&]<typename scalar_t>() { return get_type_name<scalar_t>(); });

  EXPECT_EQ(float_name, "float");

  auto double_name = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Double,
      "test_type_name",
      [&]<typename scalar_t>() { return get_type_name<scalar_t>(); });

  EXPECT_EQ(double_name, "double");

  auto int_name = AT_DISPATCH_V2_CONSTEXPR_INTEGRAL_TYPES(
      at::ScalarType::Int,
      "test_type_name",
      [&]<typename scalar_t>() { return get_type_name<scalar_t>(); });

  EXPECT_EQ(int_name, "int");
}

TEST(DispatchV2ConstexprTest, CaptureVariables) {
  // Test that we can capture variables in the lambda
  int counter = 0;

  AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Float,
      "test_capture",
      [&]<typename scalar_t>() {
        counter++;
        return counter;
      });

  EXPECT_EQ(counter, 1);

  // Test with multiple dispatches
  AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Double,
      "test_capture",
      [&]<typename scalar_t>() {
        counter++;
        return counter;
      });

  EXPECT_EQ(counter, 2);
}

TEST(DispatchV2ConstexprTest, ReturnDifferentTypes) {
  // Test that we can return different types from the lambda

  // Return int
  auto int_result = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Float,
      "test_return_int",
      [&]<typename scalar_t>() { return 42; });

  EXPECT_EQ(int_result, 42);

  // Return string
  auto string_result = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Double,
      "test_return_string",
      [&]<typename scalar_t>() { return std::string("hello"); });

  EXPECT_EQ(string_result, "hello");

  // Return size_t
  auto size_result = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
      at::ScalarType::Float,
      "test_return_size",
      [&]<typename scalar_t>() { return sizeof(scalar_t); });

  EXPECT_EQ(size_result, sizeof(float));
}

TEST(DispatchV2ConstexprTest, ReducedFloatingTypes) {
  // Test reduced precision types
  auto half_size = AT_DISPATCH_V2_CONSTEXPR_REDUCED_FLOATING_TYPES(
      at::ScalarType::Half,
      "test_half",
      [&]<typename scalar_t>() { return sizeof(scalar_t); });

  EXPECT_EQ(half_size, sizeof(c10::Half));

  auto bfloat16_size = AT_DISPATCH_V2_CONSTEXPR_REDUCED_FLOATING_TYPES(
      at::ScalarType::BFloat16,
      "test_bfloat16",
      [&]<typename scalar_t>() { return sizeof(scalar_t); });

  EXPECT_EQ(bfloat16_size, sizeof(c10::BFloat16));
}

TEST(DispatchV2ConstexprTest, FloatingTypesAnd2) {
  // Test AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES_AND2 with Half and BFloat16
  std::vector<at::ScalarType> types_to_test = {
      at::ScalarType::Float,
      at::ScalarType::Double,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
  };

  for (auto dtype : types_to_test) {
    auto result = AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES_AND2(
        c10::Half,
        c10::BFloat16,
        dtype,
        "test_floating_and2",
        [&]<typename scalar_t>() { return sizeof(scalar_t); });

    EXPECT_GT(result, 0);
  }
}

TEST(DispatchV2ConstexprTest, AllTypesAndComplexAnd2) {
  // Test adding additional types to ALL_TYPES_AND_COMPLEX
  std::vector<at::ScalarType> types_to_test = {
      at::ScalarType::Float,
      at::ScalarType::Double,
      at::ScalarType::Int,
      at::ScalarType::Long,
      at::ScalarType::ComplexFloat,
      at::ScalarType::ComplexDouble,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
  };

  for (auto dtype : types_to_test) {
    auto result = AT_DISPATCH_V2_CONSTEXPR_ALL_TYPES_AND_COMPLEX_AND2(
        c10::Half,
        c10::BFloat16,
        dtype,
        "test_all_and2",
        [&]<typename scalar_t>() { return get_type_name<scalar_t>(); });

    EXPECT_NE(result, "unknown");
  }
}

TEST(DispatchV2ConstexprTest, IndexTypes) {
  // Test index types (int32 and int64)
  auto int32_result = AT_DISPATCH_V2_CONSTEXPR_INDEX_TYPES(
      at::ScalarType::Int,
      "test_index_int32",
      [&]<typename scalar_t>() { return sizeof(scalar_t); });

  EXPECT_EQ(int32_result, sizeof(int));

  auto int64_result = AT_DISPATCH_V2_CONSTEXPR_INDEX_TYPES(
      at::ScalarType::Long,
      "test_index_int64",
      [&]<typename scalar_t>() { return sizeof(scalar_t); });

  EXPECT_EQ(int64_result, sizeof(int64_t));
}

TEST(DispatchV2ConstexprTest, ErrorOnUnsupportedType) {
  // Test that dispatching to an unsupported type throws an error
  EXPECT_THROW(
      {
        AT_DISPATCH_V2_CONSTEXPR_FLOATING_TYPES(
            at::ScalarType::Int, // Int is not in FLOATING_TYPES
            "test_unsupported",
            [&]<typename scalar_t>() { return sizeof(scalar_t); });
      },
      c10::Error);
}

// Test type mapping
TEST(DispatchV2ConstexprTest, TypeMapping) {
  // Verify CppTypeToScalarType mapping
  using namespace at::detail;

  EXPECT_EQ(CppTypeToScalarType<float>::value, at::ScalarType::Float);
  EXPECT_EQ(CppTypeToScalarType<double>::value, at::ScalarType::Double);
  EXPECT_EQ(CppTypeToScalarType<int>::value, at::ScalarType::Int);
  EXPECT_EQ(CppTypeToScalarType<int64_t>::value, at::ScalarType::Long);
  EXPECT_EQ(CppTypeToScalarType<int8_t>::value, at::ScalarType::Char);
  EXPECT_EQ(CppTypeToScalarType<uint8_t>::value, at::ScalarType::Byte);
  EXPECT_EQ(CppTypeToScalarType<int16_t>::value, at::ScalarType::Short);
  EXPECT_EQ(CppTypeToScalarType<c10::Half>::value, at::ScalarType::Half);
  EXPECT_EQ(CppTypeToScalarType<c10::BFloat16>::value, at::ScalarType::BFloat16);
  EXPECT_EQ(
      CppTypeToScalarType<c10::complex<float>>::value,
      at::ScalarType::ComplexFloat);
  EXPECT_EQ(
      CppTypeToScalarType<c10::complex<double>>::value,
      at::ScalarType::ComplexDouble);
  EXPECT_EQ(CppTypeToScalarType<bool>::value, at::ScalarType::Bool);
}
