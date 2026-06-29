// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Tests for C support APIs that do not naturally fit a workflow-specific test.

#include <gtest/gtest.h>
#include "helpers.h"

#include <cstdint>
#include <string>

class SupportApiTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        TestConfig tc("SupportApiTest");
        ovrtx_result_t result = ovrtx_create_renderer(&tc.config, &renderer_);
        ASSERT_API_SUCCESS(result.status);
    }

    static void TearDownTestSuite() {
        if (renderer_) {
            ovrtx_destroy_renderer(renderer_);
            renderer_ = nullptr;
        }
    }

    static ovrtx_renderer_t* renderer_;
};

ovrtx_renderer_t* SupportApiTest::renderer_ = nullptr;

TEST_F(SupportApiTest, VersionAndConfigConstruction) {
    // [snippet:doc-version-and-config-c]
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;
    ovrtx_get_version(&major, &minor, &patch);

    ovrtx_config_entry_t entries[] = {
        ovrtx_config_entry_log_level(ovx_str("info")),
        ovrtx_config_entry_sync_mode(true),
    };
    ovrtx_config_t config{entries, 2};

    ovrtx_renderer_t* configured_renderer = nullptr;
    ovrtx_result_t create_result = ovrtx_create_renderer(&config, &configured_renderer);
    ASSERT_API_SUCCESS(create_result.status);
    ASSERT_NE(configured_renderer, nullptr);
    ovrtx_destroy_renderer(configured_renderer);
    // [/snippet:doc-version-and-config-c]

    EXPECT_TRUE(major != 0u || minor != 0u || patch != 0u);
    EXPECT_EQ(config.entry_count, 2u);
}

TEST_F(SupportApiTest, QueryOperationStatus) {
    std::string scene = get_docs_test_data_dir() + "/ovrtx-test-base.usda";
    ovrtx_enqueue_result_t eq =
        ovrtx_open_usd_from_file(renderer_, {scene.c_str(), scene.size()});
    ASSERT_API_SUCCESS(eq.status);

    // [snippet:doc-query-op-status-c]
    ovrtx_op_status_t status{};
    ASSERT_API_SUCCESS(ovrtx_query_op_status(renderer_, eq.op_index, &status).status);
    ASSERT_EQ(status.op_id, eq.op_index);
    ASSERT_TRUE(status.state == OVRTX_EVENT_PENDING ||
                status.state == OVRTX_EVENT_COMPLETED);
    ASSERT_TRUE(status.progress < 0.0 || (status.progress >= 0.0 && status.progress <= 1.0));
    for (size_t i = 0; i < status.counter_count; ++i) {
        ASSERT_NE(status.counters[i].name.ptr, nullptr);
        ASSERT_GT(status.counters[i].name.length, 0u);
        if (status.counters[i].total != 0u) {
            ASSERT_LE(status.counters[i].current, status.counters[i].total);
        }
    }
    ASSERT_API_SUCCESS(ovrtx_release_op_status(renderer_, &status).status);
    // [/snippet:doc-query-op-status-c]

    docs_wait_no_errors(renderer_, eq.op_index);
}

TEST_F(SupportApiTest, QueryMissingExtensionError) {
    // [snippet:doc-get-last-error-c]
    const void* vtable = nullptr;
    ovrtx_result_t result = ovrtx_query_extension("ovrtx.docs.missing_extension", &vtable);
    ASSERT_EQ(result.status, OVRTX_API_ERROR);

    ovx_string_t error = ovrtx_get_last_error();
    std::string message(error.ptr, error.length);
    // [/snippet:doc-get-last-error-c]

    EXPECT_EQ(vtable, nullptr);
    EXPECT_FALSE(message.empty());
}
