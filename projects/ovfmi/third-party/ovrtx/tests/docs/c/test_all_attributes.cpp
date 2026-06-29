// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Round-trip coverage for supported authored USD attribute types via the C API.

#include <gtest/gtest.h>
#include "helpers.h"

#include <ovrtx/ovrtx_attributes.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr char kWorld[] = "/World";
constexpr char kExtentLeaf[] = "/World/ExtentTranslate/ExtentScale/ExtentLeaf";

DLDataType dl_type(uint8_t code, uint8_t bits, uint16_t lanes = 1) { return DLDataType{code, bits, lanes}; }

uint16_t half_from_float(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mantissa + 0x1000u) >> 13));
}

float half_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<uint16_t> half_values(std::initializer_list<float> values) {
    std::vector<uint16_t> out;
    out.reserve(values.size());
    for (float value : values) {
        out.push_back(half_from_float(value));
    }
    return out;
}

std::vector<uint8_t> bytes_of(char const *text) { return std::vector<uint8_t>(text, text + std::strlen(text)); }

uint64_t create_token(ovrtx_renderer_t *renderer, char const *text) {
    path_dictionary_instance_t pd{};
    EXPECT_API_SUCCESS(ovrtx_get_path_dictionary(renderer, &pd).status);
    ovx_string_t source = ovx_str(text);
    ovx_token_t token = 0;
    EXPECT_EQ(path_dictionary_create_tokens_from_strings(&pd, &source, 1, &token).status, OVX_API_SUCCESS);
    return token;
}

std::string token_to_string(ovrtx_renderer_t *renderer, uint64_t token_value) {
    path_dictionary_instance_t pd{};
    EXPECT_API_SUCCESS(ovrtx_get_path_dictionary(renderer, &pd).status);
    ovx_token_t token = token_value;
    ovx_string_t text{};
    EXPECT_EQ(path_dictionary_get_strings_from_tokens(&pd, &token, 1, &text).status, OVX_API_SUCCESS);
    return std::string(text.ptr, text.length);
}

template <typename T> std::vector<T> updated_values(std::vector<T> initial, DLDataType dtype) {
    for (T &value : initial) {
        if (dtype.code == kDLBool) {
            value = value ? T(0) : T(1);
        } else if (dtype.code == kDLFloat && dtype.bits == 16) {
            value = static_cast<T>(half_from_float(half_to_float(static_cast<uint16_t>(value)) + 0.75f));
        } else if (dtype.code == kDLFloat) {
            value = static_cast<T>(value + T(0.75));
        } else {
            value = static_cast<T>(value + T(7));
        }
    }
    return initial;
}

template <typename T>
void expect_values_near(char const *label, std::vector<T> const &actual, std::vector<T> const &expected,
                        DLDataType dtype) {
    ASSERT_EQ(actual.size(), expected.size()) << label;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (dtype.code == kDLFloat && dtype.bits == 16) {
            EXPECT_NEAR(half_to_float(static_cast<uint16_t>(actual[i])),
                        half_to_float(static_cast<uint16_t>(expected[i])), 1e-3f)
                << label << " element " << i;
        } else if constexpr (std::is_floating_point_v<T>) {
            const double tolerance = std::is_same_v<T, float> ? 1e-5 : 1e-8;
            EXPECT_NEAR(static_cast<double>(actual[i]), static_cast<double>(expected[i]), tolerance)
                << label << " element " << i;
        } else {
            EXPECT_EQ(actual[i], expected[i]) << label << " element " << i;
        }
    }
}

template <typename T>
void read_values(ovrtx_renderer_t *renderer, char const *attribute, DLDataType dtype, bool is_array,
                 size_t expected_elements, std::vector<T> &out, char const *prim = kWorld) {
    ovx_string_t path = ovx_str(prim);
    ovrtx_binding_desc_or_handle_t binding =
        ovrtx_make_binding_desc(&path, 1, ovx_str(attribute), OVRTX_SEMANTIC_NONE, dtype);
    binding.binding_desc.attribute_type.is_array = is_array;

    ovrtx_read_handle_t read_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_read_attribute(renderer, &binding, nullptr, &read_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer, eq.op_index);

    ovrtx_read_output_t output{};
    ASSERT_API_SUCCESS(ovrtx_fetch_read_result(renderer, read_handle, ovrtx_timeout_infinite, &output).status);
    ASSERT_EQ(output.is_array, is_array) << attribute;
    ASSERT_EQ(output.prim_count, 1u) << attribute;
    ASSERT_EQ(output.buffer_count, 1u) << attribute;

    DLTensor const &tensor = output.buffers[0].dl;
    ASSERT_EQ(tensor.ndim, 1) << attribute;
    ASSERT_EQ(tensor.shape[0], static_cast<int64_t>(expected_elements)) << attribute;
    ASSERT_EQ(tensor.dtype.code, dtype.code) << attribute;
    ASSERT_EQ(tensor.dtype.bits, dtype.bits) << attribute;
    ASSERT_EQ(tensor.dtype.lanes, dtype.lanes) << attribute;

    const size_t value_count = expected_elements * dtype.lanes;
    const auto *data = static_cast<const uint8_t *>(tensor.data) + tensor.byte_offset;
    const T *begin = reinterpret_cast<const T *>(data);
    out.assign(begin, begin + value_count);

    ovrtx_cuda_sync_t no_sync{};
    ASSERT_API_SUCCESS(ovrtx_release_read_result(renderer, output.map_handle, no_sync).status);
}

template <typename T>
void write_values(ovrtx_renderer_t *renderer, char const *attribute, DLDataType dtype, bool is_array,
                  std::vector<T> const &values, char const *prim = kWorld,
                  ovrtx_attribute_semantic_t semantic = OVRTX_SEMANTIC_NONE) {
    ASSERT_EQ(values.size() % dtype.lanes, 0u) << attribute;
    size_t element_count = values.size() / dtype.lanes;
    ovx_string_t path = ovx_str(prim);
    ovrtx_binding_desc_or_handle_t binding = ovrtx_make_binding_desc(&path, 1, ovx_str(attribute), semantic, dtype);
    binding.binding_desc.attribute_type.is_array = is_array;
    DLTensor tensor = ovrtx_make_write_cpu_tensor(values.data(), &element_count, dtype);
    ovrtx_input_buffer_t buffer{};
    buffer.tensors = &tensor;
    buffer.tensor_count = 1;

    ovrtx_enqueue_result_t eq = ovrtx_write_attribute(renderer, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer, eq.op_index);
}

template <typename T>
void check_numeric_case(ovrtx_renderer_t *renderer, char const *attribute, DLDataType dtype, bool is_array,
                        std::vector<T> const &initial, bool skip_initial_usd_read = false) {
    const size_t expected_elements = initial.size() / dtype.lanes;
    std::vector<T> actual;
    if (!skip_initial_usd_read) {
        read_values(renderer, attribute, dtype, is_array, expected_elements, actual);
        const std::string label = std::string(attribute) + " initial";
        expect_values_near(label.c_str(), actual, initial, dtype);
    }

    std::vector<T> updated = updated_values(initial, dtype);
    write_values(renderer, attribute, dtype, is_array, updated);
    read_values(renderer, attribute, dtype, is_array, expected_elements, actual);
    const std::string label = std::string(attribute) + " updated";
    expect_values_near(label.c_str(), actual, updated, dtype);
}

void check_token_case(ovrtx_renderer_t *renderer, char const *attribute, char const *initial, char const *updated,
                      bool is_array = false, char const *initial_b = nullptr, char const *updated_b = nullptr) {
    DLDataType dtype = dl_type(kDLUInt, 64, 1);
    const size_t expected_elements = is_array ? 2 : 1;

    std::vector<uint64_t> actual;
    read_values(renderer, attribute, dtype, is_array, expected_elements, actual);
    EXPECT_EQ(token_to_string(renderer, actual[0]), initial);
    if (is_array) {
        EXPECT_EQ(token_to_string(renderer, actual[1]), initial_b);
    }

    std::vector<uint64_t> updated_tokens = {create_token(renderer, updated)};
    if (is_array) {
        updated_tokens.push_back(create_token(renderer, updated_b));
    }
    write_values(renderer, attribute, dtype, is_array, updated_tokens, kWorld, OVRTX_SEMANTIC_TOKEN_ID);

    read_values(renderer, attribute, dtype, is_array, expected_elements, actual);
    EXPECT_EQ(token_to_string(renderer, actual[0]), updated);
    if (is_array) {
        EXPECT_EQ(token_to_string(renderer, actual[1]), updated_b);
    }
}

void check_asset_case(ovrtx_renderer_t *renderer) {
    DLDataType dtype = dl_type(kDLUInt, 64, 2);
    std::vector<uint64_t> actual;
    read_values(renderer, "test:asset", dtype, false, 1, actual);
    ASSERT_EQ(actual.size(), 2u);
    EXPECT_EQ(token_to_string(renderer, actual[0]), "initial_asset.usd");
    EXPECT_EQ(actual[1], 0u);

    std::vector<uint64_t> updated = {create_token(renderer, "updated_asset.usd"), 0};
    write_values(renderer, "test:asset", dtype, false, updated);

    read_values(renderer, "test:asset", dtype, false, 1, actual);
    EXPECT_EQ(token_to_string(renderer, actual[0]), "updated_asset.usd");
    EXPECT_EQ(actual[1], 0u);
}

void check_string_case(ovrtx_renderer_t *renderer) {
    DLDataType dtype = dl_type(kDLUInt, 8, 1);
    std::vector<uint8_t> actual;
    std::vector<uint8_t> initial = bytes_of("initial string");
    std::vector<uint8_t> updated = bytes_of("updated longer string");

    read_values(renderer, "test:string", dtype, true, initial.size(), actual);
    EXPECT_EQ(std::string(actual.begin(), actual.end()), "initial string");

    write_values(renderer, "test:string", dtype, true, updated);
    read_values(renderer, "test:string", dtype, true, updated.size(), actual);
    EXPECT_EQ(std::string(actual.begin(), actual.end()), "updated longer string");
}

void expect_attribute_not_populated(ovrtx_renderer_t *renderer, char const *attribute) {
    ovx_string_or_token_t attr_name{};
    attr_name.string = ovx_str(attribute);

    ovrtx_filter_t filter{};
    filter.kind = OVRTX_FILTER_HAS_ATTRIBUTE;
    filter.name = attr_name;

    ovrtx_query_desc_t desc{};
    desc.require_all = &filter;
    desc.require_all_count = 1;
    desc.attribute_filter.mode = OVRTX_ATTRIBUTE_FILTER_SPECIFIC;
    desc.attribute_filter.attribute_names = &attr_name;
    desc.attribute_filter.attribute_name_count = 1;

    ovrtx_query_handle_t query_handle = 0;
    ovrtx_enqueue_result_t eq = ovrtx_query_prims(renderer, &desc, &query_handle);
    ASSERT_API_SUCCESS(eq.status);
    docs_wait_no_errors(renderer, eq.op_index);

    ovrtx_query_result_t qr{};
    ASSERT_API_SUCCESS(ovrtx_fetch_query_results(renderer, query_handle, ovrtx_timeout_infinite, &qr).status);
    EXPECT_EQ(qr.total_prim_count, 0u) << attribute;
    EXPECT_EQ(qr.group_count, 0u) << attribute;
    ASSERT_API_SUCCESS(ovrtx_release_query_results(renderer, query_handle).status);
}

} // namespace

class AllAttributesTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        TestConfig tc("AllAttributesTest");
        ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
    }

    static void load_all_attributes() {
        ovrtx_enqueue_result_t eq = ovrtx_reset_stage(renderer_);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        std::string scene = get_docs_test_data_dir() + "/all-attributes.usda";
        eq = ovrtx_open_usd_from_file(renderer_, {scene.c_str(), scene.size()});
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);

        eq = ovrtx_reset(renderer_, 0.0);
        ASSERT_API_SUCCESS(eq.status);
        docs_wait_no_errors(renderer_, eq.op_index);
    }

    static ovrtx_renderer_t *renderer_;
};

ovrtx_renderer_t *AllAttributesTest::renderer_ = nullptr;

TEST_F(AllAttributesTest, SupportedAuthoredAttributesRoundTrip) {
    load_all_attributes();

    check_asset_case(renderer_);
    check_numeric_case<uint8_t>(renderer_, "test:bool", dl_type(kDLBool, 8), false, {1});
    check_numeric_case<uint8_t>(renderer_, "test:boolArray", dl_type(kDLBool, 8), true, {1, 0});
    check_numeric_case<double>(renderer_, "test:color3d", dl_type(kDLFloat, 64, 3), false, {1.1, 1.2, 1.3});
    check_numeric_case<double>(renderer_, "test:color3dArray", dl_type(kDLFloat, 64, 3), true,
                               {1.1, 1.2, 1.3, 2.1, 2.2, 2.3});
    check_numeric_case<float>(renderer_, "test:color3f", dl_type(kDLFloat, 32, 3), false, {3.1f, 3.2f, 3.3f});
    check_numeric_case<float>(renderer_, "test:color3fArray", dl_type(kDLFloat, 32, 3), true,
                              {3.1f, 3.2f, 3.3f, 4.1f, 4.2f, 4.3f});
    check_numeric_case<uint16_t>(renderer_, "test:color3h", dl_type(kDLFloat, 16, 3), false,
                                 half_values({5.0f, 5.5f, 6.0f}));
    check_numeric_case<uint16_t>(renderer_, "test:color3hArray", dl_type(kDLFloat, 16, 3), true,
                                 half_values({5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f}));
    check_numeric_case<double>(renderer_, "test:color4d", dl_type(kDLFloat, 64, 4), false, {7.1, 7.2, 7.3, 7.4});
    check_numeric_case<double>(renderer_, "test:color4dArray", dl_type(kDLFloat, 64, 4), true,
                               {7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4});
    check_numeric_case<float>(renderer_, "test:color4f", dl_type(kDLFloat, 32, 4), false, {9.1f, 9.2f, 9.3f, 9.4f});
    check_numeric_case<float>(renderer_, "test:color4fArray", dl_type(kDLFloat, 32, 4), true,
                              {9.1f, 9.2f, 9.3f, 9.4f, 10.1f, 10.2f, 10.3f, 10.4f});
    check_numeric_case<uint16_t>(renderer_, "test:color4h", dl_type(kDLFloat, 16, 4), false,
                                 half_values({11.0f, 11.5f, 12.0f, 12.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:color4hArray", dl_type(kDLFloat, 16, 4), true,
                                 half_values({11.0f, 11.5f, 12.0f, 12.5f, 13.0f, 13.5f, 14.0f, 14.5f}));
    check_numeric_case<double>(renderer_, "test:double", dl_type(kDLFloat, 64), false, {15.25});
    check_numeric_case<double>(renderer_, "test:double2", dl_type(kDLFloat, 64, 2), false, {16.1, 16.2});
    check_numeric_case<double>(renderer_, "test:double2Array", dl_type(kDLFloat, 64, 2), true,
                               {16.1, 16.2, 17.1, 17.2});
    check_numeric_case<double>(renderer_, "test:double3", dl_type(kDLFloat, 64, 3), false, {18.1, 18.2, 18.3});
    check_numeric_case<double>(renderer_, "test:double3Array", dl_type(kDLFloat, 64, 3), true,
                               {18.1, 18.2, 18.3, 19.1, 19.2, 19.3});
    check_numeric_case<double>(renderer_, "test:double4", dl_type(kDLFloat, 64, 4), false, {20.1, 20.2, 20.3, 20.4});
    check_numeric_case<double>(renderer_, "test:double4Array", dl_type(kDLFloat, 64, 4), true,
                               {20.1, 20.2, 20.3, 20.4, 21.1, 21.2, 21.3, 21.4});
    check_numeric_case<double>(renderer_, "test:doubleArray", dl_type(kDLFloat, 64), true, {22.1, 22.2});
    check_numeric_case<float>(renderer_, "test:float", dl_type(kDLFloat, 32), false, {23.5f});
    check_numeric_case<float>(renderer_, "test:float2", dl_type(kDLFloat, 32, 2), false, {24.1f, 24.2f});
    check_numeric_case<float>(renderer_, "test:float2Array", dl_type(kDLFloat, 32, 2), true,
                              {24.1f, 24.2f, 25.1f, 25.2f});
    check_numeric_case<float>(renderer_, "test:float3Array", dl_type(kDLFloat, 32, 3), true,
                              {26.1f, 26.2f, 26.3f, 27.1f, 27.2f, 27.3f});
    check_numeric_case<float>(renderer_, "test:float4", dl_type(kDLFloat, 32, 4), false, {28.1f, 28.2f, 28.3f, 28.4f});
    check_numeric_case<float>(renderer_, "test:float4Array", dl_type(kDLFloat, 32, 4), true,
                              {28.1f, 28.2f, 28.3f, 28.4f, 29.1f, 29.2f, 29.3f, 29.4f});
    check_numeric_case<float>(renderer_, "test:floatArray", dl_type(kDLFloat, 32), true, {30.1f, 30.2f});
    check_numeric_case<double>(renderer_, "test:frame4d", dl_type(kDLFloat, 64, 16), false,
                               {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 4, 5, 6, 1});
    check_numeric_case<double>(
        renderer_, "test:frame4dArray", dl_type(kDLFloat, 64, 16), true,
        {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 4, 5, 6, 1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 5, 6, 7, 1});
    check_numeric_case<uint16_t>(renderer_, "test:half", dl_type(kDLFloat, 16), false, half_values({31.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:half2", dl_type(kDLFloat, 16, 2), false, half_values({32.0f, 32.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:half2Array", dl_type(kDLFloat, 16, 2), true,
                                 half_values({32.0f, 32.5f, 33.0f, 33.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:half3", dl_type(kDLFloat, 16, 3), false,
                                 half_values({34.0f, 34.5f, 35.0f}));
    check_numeric_case<uint16_t>(renderer_, "test:half3Array", dl_type(kDLFloat, 16, 3), true,
                                 half_values({34.0f, 34.5f, 35.0f, 35.5f, 36.0f, 36.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:half4", dl_type(kDLFloat, 16, 4), false,
                                 half_values({37.0f, 37.5f, 38.0f, 38.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:half4Array", dl_type(kDLFloat, 16, 4), true,
                                 half_values({37.0f, 37.5f, 38.0f, 38.5f, 39.0f, 39.5f, 40.0f, 40.5f}));
    check_numeric_case<uint16_t>(renderer_, "test:halfArray", dl_type(kDLFloat, 16), true, half_values({41.0f, 41.5f}));
    check_numeric_case<int32_t>(renderer_, "test:int", dl_type(kDLInt, 32), false, {-42});
    check_numeric_case<int32_t>(renderer_, "test:int2", dl_type(kDLInt, 32, 2), false, {-43, 44});
    check_numeric_case<int32_t>(renderer_, "test:int2Array", dl_type(kDLInt, 32, 2), true, {-43, 44, 45, -46});
    check_numeric_case<int32_t>(renderer_, "test:int3", dl_type(kDLInt, 32, 3), false, {-47, 48, -49});
    check_numeric_case<int32_t>(renderer_, "test:int3Array", dl_type(kDLInt, 32, 3), true, {-47, 48, -49, 50, -51, 52});
    check_numeric_case<int32_t>(renderer_, "test:int4", dl_type(kDLInt, 32, 4), false, {-53, 54, -55, 56});
    check_numeric_case<int32_t>(renderer_, "test:int4Array", dl_type(kDLInt, 32, 4), true,
                                {-53, 54, -55, 56, 57, -58, 59, -60});
    check_numeric_case<int64_t>(renderer_, "test:int64", dl_type(kDLInt, 64), false, {-6100000000LL});
    check_numeric_case<int64_t>(renderer_, "test:int64Array", dl_type(kDLInt, 64), true, {-6200000000LL, 6300000000LL});
    check_numeric_case<int32_t>(renderer_, "test:intArray", dl_type(kDLInt, 32), true, {-64, 65});
    check_numeric_case<double>(renderer_, "test:matrix2d", dl_type(kDLFloat, 64, 4), false, {1, 2, 3, 4});
    check_numeric_case<double>(renderer_, "test:matrix2dArray", dl_type(kDLFloat, 64, 4), true,
                               {1, 2, 3, 4, 5, 6, 7, 8});
    check_numeric_case<double>(renderer_, "test:matrix3d", dl_type(kDLFloat, 64, 9), false,
                               {1, 2, 3, 4, 5, 6, 7, 8, 9});
    check_numeric_case<double>(renderer_, "test:matrix3dArray", dl_type(kDLFloat, 64, 9), true,
                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    check_numeric_case<double>(renderer_, "test:matrix4d", dl_type(kDLFloat, 64, 16), false,
                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    check_numeric_case<double>(renderer_, "test:matrix4dArray", dl_type(kDLFloat, 64, 16), true,
                               {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
    check_numeric_case<double>(renderer_, "test:normal3d", dl_type(kDLFloat, 64, 3), false, {66.1, 66.2, 66.3});
    check_numeric_case<double>(renderer_, "test:normal3dArray", dl_type(kDLFloat, 64, 3), true,
                               {66.1, 66.2, 66.3, 67.1, 67.2, 67.3});
    check_numeric_case<float>(renderer_, "test:normal3f", dl_type(kDLFloat, 32, 3), false, {68.1f, 68.2f, 68.3f});
    check_numeric_case<float>(renderer_, "test:normal3fArray", dl_type(kDLFloat, 32, 3), true,
                              {68.1f, 68.2f, 68.3f, 69.1f, 69.2f, 69.3f});
    check_numeric_case<uint16_t>(renderer_, "test:normal3h", dl_type(kDLFloat, 16, 3), false,
                                 half_values({70.0f, 70.5f, 71.0f}));
    check_numeric_case<uint16_t>(renderer_, "test:normal3hArray", dl_type(kDLFloat, 16, 3), true,
                                 half_values({70.0f, 70.5f, 71.0f, 71.5f, 72.0f, 72.5f}));
    check_numeric_case<double>(renderer_, "test:point3d", dl_type(kDLFloat, 64, 3), false, {73.1, 73.2, 73.3});
    check_numeric_case<double>(renderer_, "test:point3dArray", dl_type(kDLFloat, 64, 3), true,
                               {73.1, 73.2, 73.3, 74.1, 74.2, 74.3});
    check_numeric_case<float>(renderer_, "test:point3f", dl_type(kDLFloat, 32, 3), false, {75.1f, 75.2f, 75.3f});
    check_numeric_case<float>(renderer_, "test:point3fArray", dl_type(kDLFloat, 32, 3), true,
                              {75.1f, 75.2f, 75.3f, 76.1f, 76.2f, 76.3f});
    check_numeric_case<uint16_t>(renderer_, "test:point3h", dl_type(kDLFloat, 16, 3), false,
                                 half_values({77.0f, 77.5f, 78.0f}));
    check_numeric_case<uint16_t>(renderer_, "test:point3hArray", dl_type(kDLFloat, 16, 3), true,
                                 half_values({77.0f, 77.5f, 78.0f, 78.5f, 79.0f, 79.5f}));
    check_numeric_case<double>(renderer_, "test:quatd", dl_type(kDLFloat, 64, 4), false, {80.1, 80.2, 80.3, 1});
    check_numeric_case<double>(renderer_, "test:quatdArray", dl_type(kDLFloat, 64, 4), true,
                               {80.1, 80.2, 80.3, 1, 81.1, 81.2, 81.3, 1});
    check_numeric_case<float>(renderer_, "test:quatf", dl_type(kDLFloat, 32, 4), false, {82.1f, 82.2f, 82.3f, 1.0f});
    check_numeric_case<float>(renderer_, "test:quatfArray", dl_type(kDLFloat, 32, 4), true,
                              {82.1f, 82.2f, 82.3f, 1.0f, 83.1f, 83.2f, 83.3f, 1.0f});
    check_numeric_case<uint16_t>(renderer_, "test:quath", dl_type(kDLFloat, 16, 4), false,
                                 half_values({84.0f, 84.5f, 85.0f, 1.0f}));
    check_numeric_case<uint16_t>(renderer_, "test:quathArray", dl_type(kDLFloat, 16, 4), true,
                                 half_values({84.0f, 84.5f, 85.0f, 1.0f, 85.5f, 86.0f, 86.5f, 1.0f}));
    check_string_case(renderer_);
    check_numeric_case<float>(renderer_, "test:texCoord2f", dl_type(kDLFloat, 32, 2), false, {87.1f, 87.2f});
    check_numeric_case<float>(renderer_, "test:texCoord2fArray", dl_type(kDLFloat, 32, 2), true,
                              {87.1f, 87.2f, 88.1f, 88.2f});
    check_token_case(renderer_, "test:token", "initialToken", "updatedToken");
    check_token_case(renderer_, "test:tokenArray", "initialTokenA", "updatedTokenA", true, "initialTokenB",
                     "updatedTokenB");
    check_numeric_case<uint8_t>(renderer_, "test:uchar", dl_type(kDLUInt, 8), false, {91});
    check_numeric_case<uint8_t>(renderer_, "test:ucharArray", dl_type(kDLUInt, 8), true, {92, 93});
    check_numeric_case<uint32_t>(renderer_, "test:uint", dl_type(kDLUInt, 32), false, {94});
    check_numeric_case<uint64_t>(renderer_, "test:uint64", dl_type(kDLUInt, 64), false, {9500000000ULL});
    check_numeric_case<uint64_t>(renderer_, "test:uint64Array", dl_type(kDLUInt, 64), true,
                                 {9600000000ULL, 9700000000ULL});
    check_numeric_case<uint32_t>(renderer_, "test:uintArray", dl_type(kDLUInt, 32), true, {98, 99});
    check_numeric_case<double>(renderer_, "test:vector3d", dl_type(kDLFloat, 64, 3), false, {100.1, 100.2, 100.3});
    check_numeric_case<double>(renderer_, "test:vector3dArray", dl_type(kDLFloat, 64, 3), true,
                               {100.1, 100.2, 100.3, 101.1, 101.2, 101.3});
    check_numeric_case<float>(renderer_, "test:vector3f", dl_type(kDLFloat, 32, 3), false, {102.1f, 102.2f, 102.3f});
    check_numeric_case<float>(renderer_, "test:vector3fArray", dl_type(kDLFloat, 32, 3), true,
                              {102.1f, 102.2f, 102.3f, 103.1f, 103.2f, 103.3f});
    check_numeric_case<uint16_t>(renderer_, "test:vector3h", dl_type(kDLFloat, 16, 3), false,
                                 half_values({104.0f, 104.5f, 105.0f}));
    check_numeric_case<uint16_t>(renderer_, "test:vector3hArray", dl_type(kDLFloat, 16, 3), true,
                                 half_values({104.0f, 104.5f, 105.0f, 105.5f, 106.0f, 106.5f}));
}

TEST_F(AllAttributesTest, ScalarFloat3PopulationBugIsExplicit) {
    load_all_attributes();

    DLDataType dtype = dl_type(kDLFloat, 32, 3);
    std::vector<float> values;
    read_values(renderer_, "test:float3", dtype, false, 1, values);
    expect_values_near("test:float3 populated bug value", values, std::vector<float>{0.0f, 0.0f, 0.0f}, dtype);

    write_values(renderer_, "test:float3", dtype, false, std::vector<float>{26.85f, 26.95f, 27.05f});
    read_values(renderer_, "test:float3", dtype, false, 1, values);
    expect_values_near("test:float3 updated", values, std::vector<float>{26.85f, 26.95f, 27.05f}, dtype);
}

TEST_F(AllAttributesTest, UnsupportedAuthoredAttributesAreNotPopulated) {
    load_all_attributes();

    expect_attribute_not_populated(renderer_, "test:assetArray");
    expect_attribute_not_populated(renderer_, "test:rel");
    expect_attribute_not_populated(renderer_, "test:relArray");
    expect_attribute_not_populated(renderer_, "test:stringArray");
    expect_attribute_not_populated(renderer_, "test:timecode");
    expect_attribute_not_populated(renderer_, "test:timecodeArray");
}

TEST_F(AllAttributesTest, RawReadWriteSnippets) {
    load_all_attributes();

    // [snippet:doc-read-usd-float-c]
    ovx_string_t float_path = ovx_str(kWorld);
    DLDataType float_type = {kDLFloat, 32, 1};
    ovrtx_binding_desc_or_handle_t float_binding =
        ovrtx_make_binding_desc(&float_path, 1, ovx_str("test:float"), OVRTX_SEMANTIC_NONE, float_type);
    ovrtx_read_handle_t float_read_handle = 0;
    ovrtx_enqueue_result_t float_read_enqueue =
        ovrtx_read_attribute(renderer_, &float_binding, nullptr, &float_read_handle);
    if (float_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:float read");
    }
    ovrtx_op_wait_result_t float_read_wait{};
    ovrtx_result_t float_read_wait_status =
        ovrtx_wait_op(renderer_, float_read_enqueue.op_index, ovrtx_timeout_infinite, &float_read_wait);
    if (float_read_wait_status.status != OVRTX_API_SUCCESS || float_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:float read operation failed");
    }
    ovrtx_read_output_t float_output{};
    ovrtx_result_t float_fetch =
        ovrtx_fetch_read_result(renderer_, float_read_handle, ovrtx_timeout_infinite, &float_output);
    if (float_fetch.status != OVRTX_API_SUCCESS || float_output.buffer_count != 1 ||
        float_output.buffers[0].dl.dtype.code != float_type.code ||
        float_output.buffers[0].dl.dtype.bits != float_type.bits ||
        float_output.buffers[0].dl.dtype.lanes != float_type.lanes) {
        throw std::runtime_error("test:float read returned an unexpected tensor");
    }
    auto const *float_bytes =
        static_cast<uint8_t const *>(float_output.buffers[0].dl.data) + float_output.buffers[0].dl.byte_offset;
    float const *float_data = reinterpret_cast<float const *>(float_bytes);
    std::vector<float> float_values(float_data, float_data + float_output.buffers[0].dl.shape[0]);
    ovrtx_cuda_sync_t float_no_sync{};
    ovrtx_result_t float_release = ovrtx_release_read_result(renderer_, float_output.map_handle, float_no_sync);
    if (float_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:float read result");
    }
    // [/snippet:doc-read-usd-float-c]
    ASSERT_API_SUCCESS(float_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(float_read_wait);
    ASSERT_API_SUCCESS(float_fetch.status);
    ASSERT_API_SUCCESS(float_release.status);
    expect_values_near("test:float", float_values, std::vector<float>{23.5f}, float_type);

    // [snippet:doc-write-usd-float-c]
    ovx_string_t float_write_path = ovx_str(kWorld);
    DLDataType float_write_type = {kDLFloat, 32, 1};
    ovrtx_binding_desc_or_handle_t float_write_binding =
        ovrtx_make_binding_desc(&float_write_path, 1, ovx_str("test:float"), OVRTX_SEMANTIC_NONE, float_write_type);
    float updated_float[] = {24.25f};
    size_t updated_float_count = 1;
    DLTensor updated_float_tensor = ovrtx_make_write_cpu_tensor(updated_float, &updated_float_count, float_write_type);
    ovrtx_input_buffer_t updated_float_buffer{};
    updated_float_buffer.tensors = &updated_float_tensor;
    updated_float_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t float_write_enqueue =
        ovrtx_write_attribute(renderer_, &float_write_binding, &updated_float_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (float_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:float write");
    }
    ovrtx_op_wait_result_t float_write_wait{};
    ovrtx_result_t float_write_wait_status =
        ovrtx_wait_op(renderer_, float_write_enqueue.op_index, ovrtx_timeout_infinite, &float_write_wait);
    if (float_write_wait_status.status != OVRTX_API_SUCCESS || float_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:float write operation failed");
    }
    // [/snippet:doc-write-usd-float-c]
    ASSERT_API_SUCCESS(float_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(float_write_wait);
    read_values(renderer_, "test:float", float_type, false, 1, float_values);
    expect_values_near("test:float updated", float_values, std::vector<float>{24.25f}, float_type);

    // [snippet:doc-read-usd-point3f-c]
    ovx_string_t point3f_path = ovx_str(kWorld);
    DLDataType point3f_type = {kDLFloat, 32, 3};
    ovrtx_binding_desc_or_handle_t point3f_binding =
        ovrtx_make_binding_desc(&point3f_path, 1, ovx_str("test:point3f"), OVRTX_SEMANTIC_NONE, point3f_type);
    ovrtx_read_handle_t point3f_read_handle = 0;
    ovrtx_enqueue_result_t point3f_read_enqueue =
        ovrtx_read_attribute(renderer_, &point3f_binding, nullptr, &point3f_read_handle);
    if (point3f_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:point3f read");
    }
    ovrtx_op_wait_result_t point3f_read_wait{};
    ovrtx_result_t point3f_read_wait_status =
        ovrtx_wait_op(renderer_, point3f_read_enqueue.op_index, ovrtx_timeout_infinite, &point3f_read_wait);
    if (point3f_read_wait_status.status != OVRTX_API_SUCCESS || point3f_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:point3f read operation failed");
    }
    ovrtx_read_output_t point3f_output{};
    ovrtx_result_t point3f_fetch =
        ovrtx_fetch_read_result(renderer_, point3f_read_handle, ovrtx_timeout_infinite, &point3f_output);
    if (point3f_fetch.status != OVRTX_API_SUCCESS || point3f_output.buffer_count != 1 ||
        point3f_output.buffers[0].dl.dtype.code != point3f_type.code ||
        point3f_output.buffers[0].dl.dtype.bits != point3f_type.bits ||
        point3f_output.buffers[0].dl.dtype.lanes != point3f_type.lanes) {
        throw std::runtime_error("test:point3f read returned an unexpected tensor");
    }
    auto const *point3f_bytes =
        static_cast<uint8_t const *>(point3f_output.buffers[0].dl.data) + point3f_output.buffers[0].dl.byte_offset;
    float const *point3f_data = reinterpret_cast<float const *>(point3f_bytes);
    std::vector<float> point3f_values(point3f_data, point3f_data + point3f_output.buffers[0].dl.shape[0] *
                                                                       point3f_output.buffers[0].dl.dtype.lanes);
    ovrtx_cuda_sync_t point3f_no_sync{};
    ovrtx_result_t point3f_release = ovrtx_release_read_result(renderer_, point3f_output.map_handle, point3f_no_sync);
    if (point3f_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:point3f read result");
    }
    // [/snippet:doc-read-usd-point3f-c]
    ASSERT_API_SUCCESS(point3f_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(point3f_read_wait);
    ASSERT_API_SUCCESS(point3f_fetch.status);
    ASSERT_API_SUCCESS(point3f_release.status);
    expect_values_near("test:point3f", point3f_values, std::vector<float>{75.1f, 75.2f, 75.3f}, point3f_type);

    // [snippet:doc-write-usd-point3f-c]
    ovx_string_t point3f_write_path = ovx_str(kWorld);
    DLDataType point3f_write_type = {kDLFloat, 32, 3};
    ovrtx_binding_desc_or_handle_t point3f_write_binding = ovrtx_make_binding_desc(
        &point3f_write_path, 1, ovx_str("test:point3f"), OVRTX_SEMANTIC_NONE, point3f_write_type);
    float updated_point3f[] = {75.85f, 75.95f, 76.05f};
    size_t updated_point3f_count = 1;
    DLTensor updated_point3f_tensor =
        ovrtx_make_write_cpu_tensor(updated_point3f, &updated_point3f_count, point3f_write_type);
    ovrtx_input_buffer_t updated_point3f_buffer{};
    updated_point3f_buffer.tensors = &updated_point3f_tensor;
    updated_point3f_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t point3f_write_enqueue =
        ovrtx_write_attribute(renderer_, &point3f_write_binding, &updated_point3f_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (point3f_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:point3f write");
    }
    ovrtx_op_wait_result_t point3f_write_wait{};
    ovrtx_result_t point3f_write_wait_status =
        ovrtx_wait_op(renderer_, point3f_write_enqueue.op_index, ovrtx_timeout_infinite, &point3f_write_wait);
    if (point3f_write_wait_status.status != OVRTX_API_SUCCESS || point3f_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:point3f write operation failed");
    }
    // [/snippet:doc-write-usd-point3f-c]
    ASSERT_API_SUCCESS(point3f_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(point3f_write_wait);
    read_values(renderer_, "test:point3f", point3f_type, false, 1, point3f_values);
    expect_values_near("test:point3f updated", point3f_values, std::vector<float>{75.85f, 75.95f, 76.05f},
                       point3f_type);

    // [snippet:doc-read-usd-point3f-array-c]
    ovx_string_t point3f_array_path = ovx_str(kWorld);
    DLDataType point3f_array_type = {kDLFloat, 32, 3};
    ovrtx_binding_desc_or_handle_t point3f_array_binding = ovrtx_make_binding_desc(
        &point3f_array_path, 1, ovx_str("test:point3fArray"), OVRTX_SEMANTIC_NONE, point3f_array_type);
    point3f_array_binding.binding_desc.attribute_type.is_array = true;
    ovrtx_read_handle_t point3f_array_read_handle = 0;
    ovrtx_enqueue_result_t point3f_array_read_enqueue =
        ovrtx_read_attribute(renderer_, &point3f_array_binding, nullptr, &point3f_array_read_handle);
    if (point3f_array_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:point3fArray read");
    }
    ovrtx_op_wait_result_t point3f_array_read_wait{};
    ovrtx_result_t point3f_array_read_wait_status =
        ovrtx_wait_op(renderer_, point3f_array_read_enqueue.op_index, ovrtx_timeout_infinite, &point3f_array_read_wait);
    if (point3f_array_read_wait_status.status != OVRTX_API_SUCCESS || point3f_array_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:point3fArray read operation failed");
    }
    ovrtx_read_output_t point3f_array_output{};
    ovrtx_result_t point3f_array_fetch =
        ovrtx_fetch_read_result(renderer_, point3f_array_read_handle, ovrtx_timeout_infinite, &point3f_array_output);
    if (point3f_array_fetch.status != OVRTX_API_SUCCESS || point3f_array_output.buffer_count != 1 ||
        point3f_array_output.buffers[0].dl.dtype.code != point3f_array_type.code ||
        point3f_array_output.buffers[0].dl.dtype.bits != point3f_array_type.bits ||
        point3f_array_output.buffers[0].dl.dtype.lanes != point3f_array_type.lanes) {
        throw std::runtime_error("test:point3fArray read returned an unexpected tensor");
    }
    auto const *point3f_array_bytes = static_cast<uint8_t const *>(point3f_array_output.buffers[0].dl.data) +
                                      point3f_array_output.buffers[0].dl.byte_offset;
    float const *point3f_array_data = reinterpret_cast<float const *>(point3f_array_bytes);
    std::vector<float> point3f_array_values(point3f_array_data,
                                            point3f_array_data + point3f_array_output.buffers[0].dl.shape[0] *
                                                                     point3f_array_output.buffers[0].dl.dtype.lanes);
    ovrtx_cuda_sync_t point3f_array_no_sync{};
    ovrtx_result_t point3f_array_release =
        ovrtx_release_read_result(renderer_, point3f_array_output.map_handle, point3f_array_no_sync);
    if (point3f_array_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:point3fArray read result");
    }
    // [/snippet:doc-read-usd-point3f-array-c]
    ASSERT_API_SUCCESS(point3f_array_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(point3f_array_read_wait);
    ASSERT_API_SUCCESS(point3f_array_fetch.status);
    ASSERT_API_SUCCESS(point3f_array_release.status);
    expect_values_near("test:point3fArray", point3f_array_values,
                       std::vector<float>{75.1f, 75.2f, 75.3f, 76.1f, 76.2f, 76.3f}, point3f_type);

    // [snippet:doc-write-usd-point3f-array-c]
    ovx_string_t point3f_array_write_path = ovx_str(kWorld);
    DLDataType point3f_array_write_type = {kDLFloat, 32, 3};
    ovrtx_binding_desc_or_handle_t point3f_array_write_binding = ovrtx_make_binding_desc(
        &point3f_array_write_path, 1, ovx_str("test:point3fArray"), OVRTX_SEMANTIC_NONE, point3f_array_write_type);
    point3f_array_write_binding.binding_desc.attribute_type.is_array = true;
    float updated_point3f_array[] = {75.85f, 75.95f, 76.05f, 76.85f, 76.95f, 77.05f};
    size_t updated_point3f_array_count = 2;
    DLTensor updated_point3f_array_tensor =
        ovrtx_make_write_cpu_tensor(updated_point3f_array, &updated_point3f_array_count, point3f_array_write_type);
    ovrtx_input_buffer_t updated_point3f_array_buffer{};
    updated_point3f_array_buffer.tensors = &updated_point3f_array_tensor;
    updated_point3f_array_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t point3f_array_write_enqueue = ovrtx_write_attribute(
        renderer_, &point3f_array_write_binding, &updated_point3f_array_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (point3f_array_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:point3fArray write");
    }
    ovrtx_op_wait_result_t point3f_array_write_wait{};
    ovrtx_result_t point3f_array_write_wait_status = ovrtx_wait_op(renderer_, point3f_array_write_enqueue.op_index,
                                                                   ovrtx_timeout_infinite, &point3f_array_write_wait);
    if (point3f_array_write_wait_status.status != OVRTX_API_SUCCESS || point3f_array_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:point3fArray write operation failed");
    }
    // [/snippet:doc-write-usd-point3f-array-c]
    ASSERT_API_SUCCESS(point3f_array_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(point3f_array_write_wait);
    read_values(renderer_, "test:point3fArray", point3f_type, true, 2, point3f_array_values);
    expect_values_near("test:point3fArray updated", point3f_array_values,
                       std::vector<float>{75.85f, 75.95f, 76.05f, 76.85f, 76.95f, 77.05f}, point3f_type);

    // [snippet:doc-read-usd-matrix4d-c]
    ovx_string_t matrix4d_path = ovx_str(kWorld);
    DLDataType matrix4d_type = {kDLFloat, 64, 16};
    ovrtx_binding_desc_or_handle_t matrix4d_binding =
        ovrtx_make_binding_desc(&matrix4d_path, 1, ovx_str("test:matrix4d"), OVRTX_SEMANTIC_NONE, matrix4d_type);
    ovrtx_read_handle_t matrix4d_read_handle = 0;
    ovrtx_enqueue_result_t matrix4d_read_enqueue =
        ovrtx_read_attribute(renderer_, &matrix4d_binding, nullptr, &matrix4d_read_handle);
    if (matrix4d_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:matrix4d read");
    }
    ovrtx_op_wait_result_t matrix4d_read_wait{};
    ovrtx_result_t matrix4d_read_wait_status =
        ovrtx_wait_op(renderer_, matrix4d_read_enqueue.op_index, ovrtx_timeout_infinite, &matrix4d_read_wait);
    if (matrix4d_read_wait_status.status != OVRTX_API_SUCCESS || matrix4d_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:matrix4d read operation failed");
    }
    ovrtx_read_output_t matrix4d_output{};
    ovrtx_result_t matrix4d_fetch =
        ovrtx_fetch_read_result(renderer_, matrix4d_read_handle, ovrtx_timeout_infinite, &matrix4d_output);
    if (matrix4d_fetch.status != OVRTX_API_SUCCESS || matrix4d_output.buffer_count != 1 ||
        matrix4d_output.buffers[0].dl.dtype.code != matrix4d_type.code ||
        matrix4d_output.buffers[0].dl.dtype.bits != matrix4d_type.bits ||
        matrix4d_output.buffers[0].dl.dtype.lanes != matrix4d_type.lanes) {
        throw std::runtime_error("test:matrix4d read returned an unexpected tensor");
    }
    auto const *matrix4d_bytes =
        static_cast<uint8_t const *>(matrix4d_output.buffers[0].dl.data) + matrix4d_output.buffers[0].dl.byte_offset;
    double const *matrix4d_data = reinterpret_cast<double const *>(matrix4d_bytes);
    std::vector<double> matrix4d_values(matrix4d_data, matrix4d_data + matrix4d_output.buffers[0].dl.shape[0] *
                                                                           matrix4d_output.buffers[0].dl.dtype.lanes);
    ovrtx_cuda_sync_t matrix4d_no_sync{};
    ovrtx_result_t matrix4d_release =
        ovrtx_release_read_result(renderer_, matrix4d_output.map_handle, matrix4d_no_sync);
    if (matrix4d_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:matrix4d read result");
    }
    // [/snippet:doc-read-usd-matrix4d-c]
    ASSERT_API_SUCCESS(matrix4d_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(matrix4d_read_wait);
    ASSERT_API_SUCCESS(matrix4d_fetch.status);
    ASSERT_API_SUCCESS(matrix4d_release.status);
    expect_values_near("test:matrix4d", matrix4d_values,
                       std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, matrix4d_type);

    // [snippet:doc-write-usd-matrix4d-c]
    ovx_string_t matrix4d_write_path = ovx_str(kWorld);
    DLDataType matrix4d_write_type = {kDLFloat, 64, 16};
    ovrtx_binding_desc_or_handle_t matrix4d_write_binding = ovrtx_make_binding_desc(
        &matrix4d_write_path, 1, ovx_str("test:matrix4d"), OVRTX_SEMANTIC_NONE, matrix4d_write_type);
    double updated_matrix4d[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    size_t updated_matrix4d_count = 1;
    DLTensor updated_matrix4d_tensor =
        ovrtx_make_write_cpu_tensor(updated_matrix4d, &updated_matrix4d_count, matrix4d_write_type);
    ovrtx_input_buffer_t updated_matrix4d_buffer{};
    updated_matrix4d_buffer.tensors = &updated_matrix4d_tensor;
    updated_matrix4d_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t matrix4d_write_enqueue =
        ovrtx_write_attribute(renderer_, &matrix4d_write_binding, &updated_matrix4d_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (matrix4d_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:matrix4d write");
    }
    ovrtx_op_wait_result_t matrix4d_write_wait{};
    ovrtx_result_t matrix4d_write_wait_status =
        ovrtx_wait_op(renderer_, matrix4d_write_enqueue.op_index, ovrtx_timeout_infinite, &matrix4d_write_wait);
    if (matrix4d_write_wait_status.status != OVRTX_API_SUCCESS || matrix4d_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:matrix4d write operation failed");
    }
    // [/snippet:doc-write-usd-matrix4d-c]
    ASSERT_API_SUCCESS(matrix4d_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(matrix4d_write_wait);
    read_values(renderer_, "test:matrix4d", matrix4d_type, false, 1, matrix4d_values);
    expect_values_near("test:matrix4d updated", matrix4d_values,
                       std::vector<double>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, matrix4d_type);

    // [snippet:doc-read-usd-quatf-c]
    ovx_string_t quatf_path = ovx_str(kWorld);
    DLDataType quatf_type = {kDLFloat, 32, 4};
    ovrtx_binding_desc_or_handle_t quatf_binding =
        ovrtx_make_binding_desc(&quatf_path, 1, ovx_str("test:quatf"), OVRTX_SEMANTIC_NONE, quatf_type);
    ovrtx_read_handle_t quatf_read_handle = 0;
    ovrtx_enqueue_result_t quatf_read_enqueue =
        ovrtx_read_attribute(renderer_, &quatf_binding, nullptr, &quatf_read_handle);
    if (quatf_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:quatf read");
    }
    ovrtx_op_wait_result_t quatf_read_wait{};
    ovrtx_result_t quatf_read_wait_status =
        ovrtx_wait_op(renderer_, quatf_read_enqueue.op_index, ovrtx_timeout_infinite, &quatf_read_wait);
    if (quatf_read_wait_status.status != OVRTX_API_SUCCESS || quatf_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:quatf read operation failed");
    }
    ovrtx_read_output_t quatf_output{};
    ovrtx_result_t quatf_fetch =
        ovrtx_fetch_read_result(renderer_, quatf_read_handle, ovrtx_timeout_infinite, &quatf_output);
    if (quatf_fetch.status != OVRTX_API_SUCCESS || quatf_output.buffer_count != 1 ||
        quatf_output.buffers[0].dl.dtype.code != quatf_type.code ||
        quatf_output.buffers[0].dl.dtype.bits != quatf_type.bits ||
        quatf_output.buffers[0].dl.dtype.lanes != quatf_type.lanes) {
        throw std::runtime_error("test:quatf read returned an unexpected tensor");
    }
    auto const *quatf_bytes =
        static_cast<uint8_t const *>(quatf_output.buffers[0].dl.data) + quatf_output.buffers[0].dl.byte_offset;
    float const *quatf_data = reinterpret_cast<float const *>(quatf_bytes);
    std::vector<float> quatf_values(quatf_data, quatf_data + quatf_output.buffers[0].dl.shape[0] *
                                                                 quatf_output.buffers[0].dl.dtype.lanes);
    ovrtx_cuda_sync_t quatf_no_sync{};
    ovrtx_result_t quatf_release = ovrtx_release_read_result(renderer_, quatf_output.map_handle, quatf_no_sync);
    if (quatf_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:quatf read result");
    }
    // [/snippet:doc-read-usd-quatf-c]
    ASSERT_API_SUCCESS(quatf_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(quatf_read_wait);
    ASSERT_API_SUCCESS(quatf_fetch.status);
    ASSERT_API_SUCCESS(quatf_release.status);
    expect_values_near("test:quatf", quatf_values, std::vector<float>{82.1f, 82.2f, 82.3f, 1.0f}, quatf_type);

    // [snippet:doc-write-usd-quatf-c]
    ovx_string_t quatf_write_path = ovx_str(kWorld);
    DLDataType quatf_write_type = {kDLFloat, 32, 4};
    ovrtx_binding_desc_or_handle_t quatf_write_binding =
        ovrtx_make_binding_desc(&quatf_write_path, 1, ovx_str("test:quatf"), OVRTX_SEMANTIC_NONE, quatf_write_type);
    float updated_quatf[] = {82.85f, 82.95f, 83.05f, 1.75f};
    size_t updated_quatf_count = 1;
    DLTensor updated_quatf_tensor = ovrtx_make_write_cpu_tensor(updated_quatf, &updated_quatf_count, quatf_write_type);
    ovrtx_input_buffer_t updated_quatf_buffer{};
    updated_quatf_buffer.tensors = &updated_quatf_tensor;
    updated_quatf_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t quatf_write_enqueue =
        ovrtx_write_attribute(renderer_, &quatf_write_binding, &updated_quatf_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (quatf_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:quatf write");
    }
    ovrtx_op_wait_result_t quatf_write_wait{};
    ovrtx_result_t quatf_write_wait_status =
        ovrtx_wait_op(renderer_, quatf_write_enqueue.op_index, ovrtx_timeout_infinite, &quatf_write_wait);
    if (quatf_write_wait_status.status != OVRTX_API_SUCCESS || quatf_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:quatf write operation failed");
    }
    // [/snippet:doc-write-usd-quatf-c]
    ASSERT_API_SUCCESS(quatf_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(quatf_write_wait);
    read_values(renderer_, "test:quatf", quatf_type, false, 1, quatf_values);
    expect_values_near("test:quatf updated", quatf_values, std::vector<float>{82.85f, 82.95f, 83.05f, 1.75f},
                       quatf_type);

    // [snippet:doc-read-usd-token-c]
    ovx_string_t token_path = ovx_str(kWorld);
    DLDataType token_type = {kDLUInt, 64, 1};
    ovrtx_binding_desc_or_handle_t token_binding =
        ovrtx_make_binding_desc(&token_path, 1, ovx_str("test:token"), OVRTX_SEMANTIC_NONE, token_type);
    ovrtx_read_handle_t token_read_handle = 0;
    ovrtx_enqueue_result_t token_read_enqueue =
        ovrtx_read_attribute(renderer_, &token_binding, nullptr, &token_read_handle);
    if (token_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:token read");
    }
    ovrtx_op_wait_result_t token_read_wait{};
    ovrtx_result_t token_read_wait_status =
        ovrtx_wait_op(renderer_, token_read_enqueue.op_index, ovrtx_timeout_infinite, &token_read_wait);
    if (token_read_wait_status.status != OVRTX_API_SUCCESS || token_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:token read operation failed");
    }
    ovrtx_read_output_t token_output{};
    ovrtx_result_t token_fetch =
        ovrtx_fetch_read_result(renderer_, token_read_handle, ovrtx_timeout_infinite, &token_output);
    if (token_fetch.status != OVRTX_API_SUCCESS || token_output.buffer_count != 1 ||
        token_output.buffers[0].dl.dtype.code != token_type.code ||
        token_output.buffers[0].dl.dtype.bits != token_type.bits ||
        token_output.buffers[0].dl.dtype.lanes != token_type.lanes) {
        throw std::runtime_error("test:token read returned an unexpected tensor");
    }
    auto const *token_bytes =
        static_cast<uint8_t const *>(token_output.buffers[0].dl.data) + token_output.buffers[0].dl.byte_offset;
    uint64_t const *token_ids = reinterpret_cast<uint64_t const *>(token_bytes);
    path_dictionary_instance_t token_pd{};
    if (ovrtx_get_path_dictionary(renderer_, &token_pd).status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to get path dictionary");
    }
    ovx_token_t token_id = token_ids[0];
    ovx_string_t token_string{};
    if (path_dictionary_get_strings_from_tokens(&token_pd, &token_id, 1, &token_string).status != OVX_API_SUCCESS) {
        throw std::runtime_error("Failed to resolve test:token");
    }
    ovrtx_cuda_sync_t token_no_sync{};
    ovrtx_result_t token_release = ovrtx_release_read_result(renderer_, token_output.map_handle, token_no_sync);
    if (token_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:token read result");
    }
    // [/snippet:doc-read-usd-token-c]
    ASSERT_API_SUCCESS(token_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(token_read_wait);
    ASSERT_API_SUCCESS(token_fetch.status);
    ASSERT_API_SUCCESS(token_release.status);
    EXPECT_EQ(std::string(token_string.ptr, token_string.length), "initialToken");

    // [snippet:doc-write-usd-token-c]
    ovx_string_t token_write_path = ovx_str(kWorld);
    ovx_string_t updated_token = ovx_str("updatedToken");
    ovrtx_enqueue_result_t token_write_enqueue =
        ovrtx_set_token_attributes(renderer_, &token_write_path, 1, ovx_str("test:token"), &updated_token);
    if (token_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:token write");
    }
    ovrtx_op_wait_result_t token_write_wait{};
    ovrtx_result_t token_write_wait_status =
        ovrtx_wait_op(renderer_, token_write_enqueue.op_index, ovrtx_timeout_infinite, &token_write_wait);
    if (token_write_wait_status.status != OVRTX_API_SUCCESS || token_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:token write operation failed");
    }
    // [/snippet:doc-write-usd-token-c]
    ASSERT_API_SUCCESS(token_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(token_write_wait);
    std::vector<uint64_t> token_values;
    read_values(renderer_, "test:token", token_type, false, 1, token_values);
    EXPECT_EQ(token_to_string(renderer_, token_values[0]), "updatedToken");

    // [snippet:doc-read-usd-token-array-c]
    ovx_string_t token_array_path = ovx_str(kWorld);
    DLDataType token_array_type = {kDLUInt, 64, 1};
    ovrtx_binding_desc_or_handle_t token_array_binding = ovrtx_make_binding_desc(
        &token_array_path, 1, ovx_str("test:tokenArray"), OVRTX_SEMANTIC_NONE, token_array_type);
    token_array_binding.binding_desc.attribute_type.is_array = true;
    ovrtx_read_handle_t token_array_read_handle = 0;
    ovrtx_enqueue_result_t token_array_read_enqueue =
        ovrtx_read_attribute(renderer_, &token_array_binding, nullptr, &token_array_read_handle);
    if (token_array_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:tokenArray read");
    }
    ovrtx_op_wait_result_t token_array_read_wait{};
    ovrtx_result_t token_array_read_wait_status =
        ovrtx_wait_op(renderer_, token_array_read_enqueue.op_index, ovrtx_timeout_infinite, &token_array_read_wait);
    if (token_array_read_wait_status.status != OVRTX_API_SUCCESS || token_array_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:tokenArray read operation failed");
    }
    ovrtx_read_output_t token_array_output{};
    ovrtx_result_t token_array_fetch =
        ovrtx_fetch_read_result(renderer_, token_array_read_handle, ovrtx_timeout_infinite, &token_array_output);
    if (token_array_fetch.status != OVRTX_API_SUCCESS || token_array_output.buffer_count != 1 ||
        token_array_output.buffers[0].dl.dtype.code != token_array_type.code ||
        token_array_output.buffers[0].dl.dtype.bits != token_array_type.bits ||
        token_array_output.buffers[0].dl.dtype.lanes != token_array_type.lanes) {
        throw std::runtime_error("test:tokenArray read returned an unexpected tensor");
    }
    auto const *token_array_bytes = static_cast<uint8_t const *>(token_array_output.buffers[0].dl.data) +
                                    token_array_output.buffers[0].dl.byte_offset;
    uint64_t const *token_array_ids = reinterpret_cast<uint64_t const *>(token_array_bytes);
    path_dictionary_instance_t token_array_pd{};
    if (ovrtx_get_path_dictionary(renderer_, &token_array_pd).status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to get path dictionary");
    }
    std::vector<std::string> token_array_values;
    for (int64_t i = 0; i < token_array_output.buffers[0].dl.shape[0]; ++i) {
        ovx_token_t id = token_array_ids[i];
        ovx_string_t text{};
        if (path_dictionary_get_strings_from_tokens(&token_array_pd, &id, 1, &text).status != OVX_API_SUCCESS) {
            throw std::runtime_error("Failed to resolve test:tokenArray");
        }
        token_array_values.emplace_back(text.ptr, text.length);
    }
    ovrtx_cuda_sync_t token_array_no_sync{};
    ovrtx_result_t token_array_release =
        ovrtx_release_read_result(renderer_, token_array_output.map_handle, token_array_no_sync);
    if (token_array_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:tokenArray read result");
    }
    // [/snippet:doc-read-usd-token-array-c]
    ASSERT_API_SUCCESS(token_array_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(token_array_read_wait);
    ASSERT_API_SUCCESS(token_array_fetch.status);
    ASSERT_API_SUCCESS(token_array_release.status);
    EXPECT_EQ(token_array_values, (std::vector<std::string>{"initialTokenA", "initialTokenB"}));

    // [snippet:doc-write-usd-token-array-c]
    ovx_string_t token_array_write_path = ovx_str(kWorld);
    DLDataType token_array_write_type = {kDLUInt, 64, 1};
    path_dictionary_instance_t token_array_write_pd{};
    if (ovrtx_get_path_dictionary(renderer_, &token_array_write_pd).status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to get path dictionary");
    }
    ovx_string_t updated_token_array_strings[] = {ovx_str("updatedTokenA"), ovx_str("updatedTokenB")};
    ovx_token_t updated_token_array_ids[2] = {};
    if (path_dictionary_create_tokens_from_strings(&token_array_write_pd, updated_token_array_strings, 2,
                                                   updated_token_array_ids)
            .status != OVX_API_SUCCESS) {
        throw std::runtime_error("Failed to create test:tokenArray tokens");
    }
    size_t updated_token_array_count = 2;
    DLTensor updated_token_array_tensor =
        ovrtx_make_write_cpu_tensor(updated_token_array_ids, &updated_token_array_count, token_array_write_type);
    ovrtx_input_buffer_t updated_token_array_buffer{};
    updated_token_array_buffer.tensors = &updated_token_array_tensor;
    updated_token_array_buffer.tensor_count = 1;
    ovrtx_binding_desc_or_handle_t token_array_write_binding = ovrtx_make_binding_desc(
        &token_array_write_path, 1, ovx_str("test:tokenArray"), OVRTX_SEMANTIC_TOKEN_ID, token_array_write_type);
    token_array_write_binding.binding_desc.attribute_type.is_array = true;
    ovrtx_enqueue_result_t token_array_write_enqueue = ovrtx_write_attribute(
        renderer_, &token_array_write_binding, &updated_token_array_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (token_array_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:tokenArray write");
    }
    ovrtx_op_wait_result_t token_array_write_wait{};
    ovrtx_result_t token_array_write_wait_status =
        ovrtx_wait_op(renderer_, token_array_write_enqueue.op_index, ovrtx_timeout_infinite, &token_array_write_wait);
    if (token_array_write_wait_status.status != OVRTX_API_SUCCESS || token_array_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:tokenArray write operation failed");
    }
    // [/snippet:doc-write-usd-token-array-c]
    ASSERT_API_SUCCESS(token_array_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(token_array_write_wait);
    read_values(renderer_, "test:tokenArray", token_type, true, 2, token_values);
    EXPECT_EQ(token_to_string(renderer_, token_values[0]), "updatedTokenA");
    EXPECT_EQ(token_to_string(renderer_, token_values[1]), "updatedTokenB");

    // [snippet:doc-read-usd-string-c]
    ovx_string_t string_path = ovx_str(kWorld);
    DLDataType string_type = {kDLUInt, 8, 1};
    ovrtx_binding_desc_or_handle_t string_binding =
        ovrtx_make_binding_desc(&string_path, 1, ovx_str("test:string"), OVRTX_SEMANTIC_NONE, string_type);
    string_binding.binding_desc.attribute_type.is_array = true;
    ovrtx_read_handle_t string_read_handle = 0;
    ovrtx_enqueue_result_t string_read_enqueue =
        ovrtx_read_attribute(renderer_, &string_binding, nullptr, &string_read_handle);
    if (string_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:string read");
    }
    ovrtx_op_wait_result_t string_read_wait{};
    ovrtx_result_t string_read_wait_status =
        ovrtx_wait_op(renderer_, string_read_enqueue.op_index, ovrtx_timeout_infinite, &string_read_wait);
    if (string_read_wait_status.status != OVRTX_API_SUCCESS || string_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:string read operation failed");
    }
    ovrtx_read_output_t string_output{};
    ovrtx_result_t string_fetch =
        ovrtx_fetch_read_result(renderer_, string_read_handle, ovrtx_timeout_infinite, &string_output);
    if (string_fetch.status != OVRTX_API_SUCCESS || string_output.buffer_count != 1 ||
        string_output.buffers[0].dl.dtype.code != string_type.code ||
        string_output.buffers[0].dl.dtype.bits != string_type.bits ||
        string_output.buffers[0].dl.dtype.lanes != string_type.lanes) {
        throw std::runtime_error("test:string read returned an unexpected tensor");
    }
    auto const *string_data =
        static_cast<char const *>(string_output.buffers[0].dl.data) + string_output.buffers[0].dl.byte_offset;
    std::string string_value(string_data, string_data + string_output.buffers[0].dl.shape[0]);
    ovrtx_cuda_sync_t string_no_sync{};
    ovrtx_result_t string_release = ovrtx_release_read_result(renderer_, string_output.map_handle, string_no_sync);
    if (string_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:string read result");
    }
    // [/snippet:doc-read-usd-string-c]
    ASSERT_API_SUCCESS(string_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(string_read_wait);
    ASSERT_API_SUCCESS(string_fetch.status);
    ASSERT_API_SUCCESS(string_release.status);
    EXPECT_EQ(string_value, "initial string");

    // [snippet:doc-write-usd-string-c]
    ovx_string_t string_write_path = ovx_str(kWorld);
    DLDataType string_write_type = {kDLUInt, 8, 1};
    ovrtx_binding_desc_or_handle_t string_write_binding =
        ovrtx_make_binding_desc(&string_write_path, 1, ovx_str("test:string"), OVRTX_SEMANTIC_NONE, string_write_type);
    string_write_binding.binding_desc.attribute_type.is_array = true;
    char const updated_string[] = "updated longer string";
    size_t updated_string_count = std::strlen(updated_string);
    DLTensor updated_string_tensor =
        ovrtx_make_write_cpu_tensor(updated_string, &updated_string_count, string_write_type);
    ovrtx_input_buffer_t updated_string_buffer{};
    updated_string_buffer.tensors = &updated_string_tensor;
    updated_string_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t string_write_enqueue =
        ovrtx_write_attribute(renderer_, &string_write_binding, &updated_string_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (string_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:string write");
    }
    ovrtx_op_wait_result_t string_write_wait{};
    ovrtx_result_t string_write_wait_status =
        ovrtx_wait_op(renderer_, string_write_enqueue.op_index, ovrtx_timeout_infinite, &string_write_wait);
    if (string_write_wait_status.status != OVRTX_API_SUCCESS || string_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:string write operation failed");
    }
    // [/snippet:doc-write-usd-string-c]
    ASSERT_API_SUCCESS(string_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(string_write_wait);
    std::vector<uint8_t> string_values;
    read_values(renderer_, "test:string", string_type, true, std::strlen(updated_string), string_values);
    EXPECT_EQ(std::string(string_values.begin(), string_values.end()), "updated longer string");

    // [snippet:doc-read-usd-asset-c]
    ovx_string_t asset_path = ovx_str(kWorld);
    DLDataType asset_type = {kDLUInt, 64, 2};
    ovrtx_binding_desc_or_handle_t asset_binding =
        ovrtx_make_binding_desc(&asset_path, 1, ovx_str("test:asset"), OVRTX_SEMANTIC_NONE, asset_type);
    ovrtx_read_handle_t asset_read_handle = 0;
    ovrtx_enqueue_result_t asset_read_enqueue =
        ovrtx_read_attribute(renderer_, &asset_binding, nullptr, &asset_read_handle);
    if (asset_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:asset read");
    }
    ovrtx_op_wait_result_t asset_read_wait{};
    ovrtx_result_t asset_read_wait_status =
        ovrtx_wait_op(renderer_, asset_read_enqueue.op_index, ovrtx_timeout_infinite, &asset_read_wait);
    if (asset_read_wait_status.status != OVRTX_API_SUCCESS || asset_read_wait.num_error_ops != 0) {
        throw std::runtime_error("test:asset read operation failed");
    }
    ovrtx_read_output_t asset_output{};
    ovrtx_result_t asset_fetch =
        ovrtx_fetch_read_result(renderer_, asset_read_handle, ovrtx_timeout_infinite, &asset_output);
    if (asset_fetch.status != OVRTX_API_SUCCESS || asset_output.buffer_count != 1 ||
        asset_output.buffers[0].dl.dtype.code != asset_type.code ||
        asset_output.buffers[0].dl.dtype.bits != asset_type.bits ||
        asset_output.buffers[0].dl.dtype.lanes != asset_type.lanes) {
        throw std::runtime_error("test:asset read returned an unexpected tensor");
    }
    auto const *asset_bytes =
        static_cast<uint8_t const *>(asset_output.buffers[0].dl.data) + asset_output.buffers[0].dl.byte_offset;
    uint64_t const *asset_data = reinterpret_cast<uint64_t const *>(asset_bytes);
    ovx_token_t asset_token = asset_data[0];
    ovx_string_t asset_string{};
    path_dictionary_instance_t asset_pd{};
    if (ovrtx_get_path_dictionary(renderer_, &asset_pd).status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to get path dictionary");
    }
    if (path_dictionary_get_strings_from_tokens(&asset_pd, &asset_token, 1, &asset_string).status != OVX_API_SUCCESS) {
        throw std::runtime_error("Failed to resolve test:asset");
    }
    ovrtx_cuda_sync_t asset_no_sync{};
    ovrtx_result_t asset_release = ovrtx_release_read_result(renderer_, asset_output.map_handle, asset_no_sync);
    if (asset_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release test:asset read result");
    }
    // [/snippet:doc-read-usd-asset-c]
    ASSERT_API_SUCCESS(asset_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(asset_read_wait);
    ASSERT_API_SUCCESS(asset_fetch.status);
    ASSERT_API_SUCCESS(asset_release.status);
    EXPECT_EQ(std::string(asset_string.ptr, asset_string.length), "initial_asset.usd");

    // [snippet:doc-write-usd-asset-c]
    ovx_string_t asset_write_path = ovx_str(kWorld);
    DLDataType asset_write_type = {kDLUInt, 64, 2};
    ovrtx_binding_desc_or_handle_t asset_write_binding =
        ovrtx_make_binding_desc(&asset_write_path, 1, ovx_str("test:asset"), OVRTX_SEMANTIC_NONE, asset_write_type);
    path_dictionary_instance_t asset_write_pd{};
    if (ovrtx_get_path_dictionary(renderer_, &asset_write_pd).status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to get path dictionary");
    }
    ovx_string_t updated_asset_string = ovx_str("updated_asset.usd");
    ovx_token_t updated_asset_token = 0;
    if (path_dictionary_create_tokens_from_strings(&asset_write_pd, &updated_asset_string, 1, &updated_asset_token)
            .status != OVX_API_SUCCESS) {
        throw std::runtime_error("Failed to create test:asset token");
    }
    uint64_t updated_asset[] = {updated_asset_token, 0};
    size_t updated_asset_count = 1;
    DLTensor updated_asset_tensor = ovrtx_make_write_cpu_tensor(updated_asset, &updated_asset_count, asset_write_type);
    ovrtx_input_buffer_t updated_asset_buffer{};
    updated_asset_buffer.tensors = &updated_asset_tensor;
    updated_asset_buffer.tensor_count = 1;
    ovrtx_enqueue_result_t asset_write_enqueue =
        ovrtx_write_attribute(renderer_, &asset_write_binding, &updated_asset_buffer, OVRTX_DATA_ACCESS_SYNC);
    if (asset_write_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue test:asset write");
    }
    ovrtx_op_wait_result_t asset_write_wait{};
    ovrtx_result_t asset_write_wait_status =
        ovrtx_wait_op(renderer_, asset_write_enqueue.op_index, ovrtx_timeout_infinite, &asset_write_wait);
    if (asset_write_wait_status.status != OVRTX_API_SUCCESS || asset_write_wait.num_error_ops != 0) {
        throw std::runtime_error("test:asset write operation failed");
    }
    // [/snippet:doc-write-usd-asset-c]
    ASSERT_API_SUCCESS(asset_write_enqueue.status);
    ASSERT_NO_OP_ERRORS(asset_write_wait);
    std::vector<uint64_t> asset_values;
    read_values(renderer_, "test:asset", asset_type, false, 1, asset_values);
    EXPECT_EQ(token_to_string(renderer_, asset_values[0]), "updated_asset.usd");
}

TEST_F(AllAttributesTest, ExtentAndWorldExtentAreReadable) {
    load_all_attributes();

    // [snippet:doc-extent-world-extent-c]
    ovx_string_t extent_path = ovx_str(kExtentLeaf);
    DLDataType extent_type = {kDLFloat, 64, 6};

    ovrtx_binding_desc_or_handle_t extent_binding =
        ovrtx_make_binding_desc(&extent_path, 1, ovx_str("extent"), OVRTX_SEMANTIC_NONE, extent_type);
    ovrtx_read_handle_t extent_read_handle = 0;
    ovrtx_enqueue_result_t extent_read_enqueue =
        ovrtx_read_attribute(renderer_, &extent_binding, nullptr, &extent_read_handle);
    if (extent_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue extent read");
    }
    ovrtx_op_wait_result_t extent_read_wait{};
    ovrtx_result_t extent_read_wait_status =
        ovrtx_wait_op(renderer_, extent_read_enqueue.op_index, ovrtx_timeout_infinite, &extent_read_wait);
    if (extent_read_wait_status.status != OVRTX_API_SUCCESS || extent_read_wait.num_error_ops != 0) {
        throw std::runtime_error("extent read operation failed");
    }
    ovrtx_read_output_t extent_output{};
    ovrtx_result_t extent_fetch =
        ovrtx_fetch_read_result(renderer_, extent_read_handle, ovrtx_timeout_infinite, &extent_output);
    if (extent_fetch.status != OVRTX_API_SUCCESS || extent_output.buffer_count != 1 ||
        extent_output.buffers[0].dl.dtype.code != extent_type.code ||
        extent_output.buffers[0].dl.dtype.bits != extent_type.bits ||
        extent_output.buffers[0].dl.dtype.lanes != extent_type.lanes) {
        throw std::runtime_error("extent read returned an unexpected tensor");
    }
    auto const *extent_bytes =
        static_cast<uint8_t const *>(extent_output.buffers[0].dl.data) + extent_output.buffers[0].dl.byte_offset;
    double const *extent_data = reinterpret_cast<double const *>(extent_bytes);
    std::vector<double> local_extent(extent_data, extent_data + extent_output.buffers[0].dl.shape[0] *
                                                                    extent_output.buffers[0].dl.dtype.lanes);
    ovrtx_cuda_sync_t extent_no_sync{};
    ovrtx_result_t extent_release = ovrtx_release_read_result(renderer_, extent_output.map_handle, extent_no_sync);
    if (extent_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release extent read result");
    }

    ovrtx_binding_desc_or_handle_t world_extent_binding =
        ovrtx_make_binding_desc(&extent_path, 1, ovx_str("_worldExtent"), OVRTX_SEMANTIC_NONE, extent_type);
    ovrtx_read_handle_t world_extent_read_handle = 0;
    ovrtx_enqueue_result_t world_extent_read_enqueue =
        ovrtx_read_attribute(renderer_, &world_extent_binding, nullptr, &world_extent_read_handle);
    if (world_extent_read_enqueue.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to enqueue _worldExtent read");
    }
    ovrtx_op_wait_result_t world_extent_read_wait{};
    ovrtx_result_t world_extent_read_wait_status =
        ovrtx_wait_op(renderer_, world_extent_read_enqueue.op_index, ovrtx_timeout_infinite, &world_extent_read_wait);
    if (world_extent_read_wait_status.status != OVRTX_API_SUCCESS || world_extent_read_wait.num_error_ops != 0) {
        throw std::runtime_error("_worldExtent read operation failed");
    }
    ovrtx_read_output_t world_extent_output{};
    ovrtx_result_t world_extent_fetch =
        ovrtx_fetch_read_result(renderer_, world_extent_read_handle, ovrtx_timeout_infinite, &world_extent_output);
    if (world_extent_fetch.status != OVRTX_API_SUCCESS || world_extent_output.buffer_count != 1 ||
        world_extent_output.buffers[0].dl.dtype.code != extent_type.code ||
        world_extent_output.buffers[0].dl.dtype.bits != extent_type.bits ||
        world_extent_output.buffers[0].dl.dtype.lanes != extent_type.lanes) {
        throw std::runtime_error("_worldExtent read returned an unexpected tensor");
    }
    auto const *world_extent_bytes = static_cast<uint8_t const *>(world_extent_output.buffers[0].dl.data) +
                                     world_extent_output.buffers[0].dl.byte_offset;
    double const *world_extent_data = reinterpret_cast<double const *>(world_extent_bytes);
    std::vector<double> world_extent(world_extent_data,
                                     world_extent_data + world_extent_output.buffers[0].dl.shape[0] *
                                                             world_extent_output.buffers[0].dl.dtype.lanes);
    ovrtx_cuda_sync_t world_extent_no_sync{};
    ovrtx_result_t world_extent_release =
        ovrtx_release_read_result(renderer_, world_extent_output.map_handle, world_extent_no_sync);
    if (world_extent_release.status != OVRTX_API_SUCCESS) {
        throw std::runtime_error("Failed to release _worldExtent read result");
    }
    // [/snippet:doc-extent-world-extent-c]

    ASSERT_API_SUCCESS(extent_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(extent_read_wait);
    ASSERT_API_SUCCESS(extent_fetch.status);
    ASSERT_API_SUCCESS(extent_release.status);
    ASSERT_API_SUCCESS(world_extent_read_enqueue.status);
    ASSERT_NO_OP_ERRORS(world_extent_read_wait);
    ASSERT_API_SUCCESS(world_extent_fetch.status);
    ASSERT_API_SUCCESS(world_extent_release.status);
    expect_values_near("extent", local_extent, std::vector<double>{-1, -2, -3, 1, 2, 3}, extent_type);
    expect_values_near("_worldExtent", world_extent, std::vector<double>{8, 14, 18, 12, 26, 42}, extent_type);
}
