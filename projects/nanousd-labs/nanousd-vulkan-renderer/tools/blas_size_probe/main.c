// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * blas_size_probe — measure BLAS size for fp32 vs fp16 vertex inputs.
 *
 * Calls vkGetAccelerationStructureBuildSizesKHR twice on the same logical
 * geometry (1000 indexed triangles, 3000 unique vertices), once with
 * VK_FORMAT_R32G32B32_SFLOAT and once with VK_FORMAT_R16G16B16A16_SFLOAT,
 * and reports the resulting accelerationStructureSize / buildScratchSize /
 * updateScratchSize values plus the GPU name.
 *
 * The size query does NOT dereference vertex/index device addresses — it only
 * reads format / stride / maxVertex / primitive count. No buffers are
 * allocated; no commands are submitted.
 *
 * If the driver does not advertise
 * VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR for a given
 * format, that format is skipped with a clear error message.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#define TRI_COUNT     1000u
#define VERTEX_COUNT  3000u   /* 1000 triangles, indexed, 3 unique verts each */

#define VK_CHECK(expr) do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "%s:%d: " #expr " failed (VkResult=%d)\n", \
                __FILE__, __LINE__, (int)_r); \
        return 1; \
    } \
} while (0)

static int format_supported_for_as(VkPhysicalDevice pd, VkFormat fmt)
{
    VkFormatProperties props = {0};
    vkGetPhysicalDeviceFormatProperties(pd, fmt, &props);
    /* The relevant flag lives in bufferFeatures for AS vertex input. */
    return (props.bufferFeatures &
            VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR) != 0;
}

static const char* fmt_name(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_R32G32B32_SFLOAT:    return "R32G32B32_SFLOAT";
        case VK_FORMAT_R16G16B16A16_SFLOAT: return "R16G16B16A16_SFLOAT";
        default: return "?";
    }
}

static uint32_t fmt_bytes(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_R32G32B32_SFLOAT:    return 12u;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8u;
        default: return 0u;
    }
}

/* Query BLAS sizes for the same logical mesh under a given vertex format.
 * deviceAddress is left at 0 — the size query does not dereference it. */
static int query_blas_sizes(VkDevice device,
                            PFN_vkGetAccelerationStructureBuildSizesKHR pfn,
                            VkFormat vertex_format,
                            VkAccelerationStructureBuildSizesInfoKHR* out)
{
    VkAccelerationStructureGeometryTrianglesDataKHR tri = {0};
    tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat             = vertex_format;
    tri.vertexData.deviceAddress = 0;          /* not read during size query */
    tri.vertexStride             = fmt_bytes(vertex_format);
    tri.maxVertex                = VERTEX_COUNT - 1u;
    tri.indexType                = VK_INDEX_TYPE_UINT32;
    tri.indexData.deviceAddress  = 0;          /* not read during size query */
    tri.transformData.deviceAddress = 0;

    VkAccelerationStructureGeometryKHR geom = {0};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles = tri;

    VkAccelerationStructureBuildGeometryInfoKHR build = {0};
    build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    /* Match production renderer flags so the measurement is comparable. */
    build.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    build.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries   = &geom;

    uint32_t prim_count = TRI_COUNT;

    out->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, &prim_count, out);
    return 0;
}

int main(void)
{
    /* --- Instance (no surface, no swapchain) --- */
    VkApplicationInfo ai = {0};
    ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "blas_size_probe";
    ai.apiVersion       = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {0};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;

    VkInstance instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ici, NULL, &instance));

    /* --- Pick a physical device (prefer discrete) --- */
    uint32_t pd_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, NULL));
    if (pd_count == 0) {
        fprintf(stderr, "No Vulkan physical devices found.\n");
        return 1;
    }
    VkPhysicalDevice* pds = (VkPhysicalDevice*)malloc(pd_count * sizeof(*pds));
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, pds));

    VkPhysicalDevice pd = pds[0];
    for (uint32_t i = 0; i < pd_count; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pd = pds[i];
            break;
        }
    }
    free(pds);

    VkPhysicalDeviceProperties props = {0};
    vkGetPhysicalDeviceProperties(pd, &props);

    /* --- Confirm required device extensions are present --- */
    const char* required_exts[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    };
    const uint32_t required_count = (uint32_t)(sizeof(required_exts) / sizeof(required_exts[0]));

    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ext_count, NULL);
    VkExtensionProperties* avail = (VkExtensionProperties*)malloc(ext_count * sizeof(*avail));
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ext_count, avail);

    for (uint32_t r = 0; r < required_count; r++) {
        int found = 0;
        for (uint32_t i = 0; i < ext_count; i++) {
            if (strcmp(avail[i].extensionName, required_exts[r]) == 0) { found = 1; break; }
        }
        if (!found) {
            fprintf(stderr, "Missing required device extension: %s\n", required_exts[r]);
            free(avail);
            vkDestroyInstance(instance, NULL);
            return 1;
        }
    }
    free(avail);

    /* --- Pick a queue family (any with COMPUTE or GRAPHICS bits — we won't submit work) --- */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, NULL);
    VkQueueFamilyProperties* qfs = (VkQueueFamilyProperties*)malloc(qf_count * sizeof(*qfs));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qfs);
    uint32_t qf_index = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qfs[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) {
            qf_index = i;
            break;
        }
    }
    free(qfs);
    if (qf_index == UINT32_MAX) {
        fprintf(stderr, "No graphics/compute queue family found.\n");
        return 1;
    }

    /* --- Logical device with RT extensions + bufferDeviceAddress feature --- */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qf_index;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_feat = {0};
    as_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_feat.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_feat = {0};
    rt_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rt_feat.rayTracingPipeline = VK_TRUE;
    rt_feat.pNext = &as_feat;

    VkPhysicalDeviceVulkan12Features v12 = {0};
    v12.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.pNext               = &rt_feat;

    VkPhysicalDeviceFeatures2 feats2 = {0};
    feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats2.pNext = &v12;

    VkDeviceCreateInfo dci = {0};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &feats2;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = required_count;
    dci.ppEnabledExtensionNames = required_exts;

    VkDevice device = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDevice(pd, &dci, NULL, &device));

    /* Load the AS size-query function pointer (extension entry point). */
    PFN_vkGetAccelerationStructureBuildSizesKHR pfn_get_sizes =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    if (!pfn_get_sizes) {
        fprintf(stderr, "vkGetDeviceProcAddr(vkGetAccelerationStructureBuildSizesKHR) returned NULL\n");
        return 1;
    }

    /* --- Format support probe (AS vertex buffer bit) --- */
    VkFormat fmt_fp32 = VK_FORMAT_R32G32B32_SFLOAT;
    VkFormat fmt_fp16 = VK_FORMAT_R16G16B16A16_SFLOAT;
    int sup_fp32 = format_supported_for_as(pd, fmt_fp32);
    int sup_fp16 = format_supported_for_as(pd, fmt_fp16);

    /* --- Header --- */
    printf("device       = %s\n", props.deviceName);
    printf("api_version  = %u.%u.%u\n",
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
    printf("driver_ver   = 0x%08x\n", props.driverVersion);
    printf("vendor_id    = 0x%04x\n", props.vendorID);
    printf("tri_count    = %u\n", TRI_COUNT);
    printf("vertex_count = %u (indexed, uint32)\n", VERTEX_COUNT);
    printf("flags        = PREFER_FAST_TRACE | ALLOW_COMPACTION\n");
    printf("fp32 fmt     = %s  stride=%u  as_supported=%d\n",
           fmt_name(fmt_fp32), fmt_bytes(fmt_fp32), sup_fp32);
    printf("fp16 fmt     = %s  stride=%u  as_supported=%d\n",
           fmt_name(fmt_fp16), fmt_bytes(fmt_fp16), sup_fp16);
    printf("\n");

    if (!sup_fp32) {
        fprintf(stderr, "ERROR: VK_FORMAT_R32G32B32_SFLOAT is not advertised with "
                        "ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR on this device. "
                        "This is unexpected — R32G32B32_SFLOAT is the standard BLAS vertex format.\n");
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 2;
    }
    if (!sup_fp16) {
        fprintf(stderr, "ERROR: VK_FORMAT_R16G16B16A16_SFLOAT is not advertised with "
                        "ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR on this device. "
                        "The fp16 BLAS path is not supported here; the fp32 vs fp16 "
                        "measurement cannot be performed.\n");
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 3;
    }

    /* --- Run both queries --- */
    VkAccelerationStructureBuildSizesInfoKHR sz32 = {0};
    VkAccelerationStructureBuildSizesInfoKHR sz16 = {0};
    query_blas_sizes(device, pfn_get_sizes, fmt_fp32, &sz32);
    query_blas_sizes(device, pfn_get_sizes, fmt_fp16, &sz16);

    long long s32 = (long long)sz32.accelerationStructureSize;
    long long s16 = (long long)sz16.accelerationStructureSize;
    long long delta = s16 - s32;   /* negative = fp16 smaller */
    double pct = (s32 > 0) ? (100.0 * (double)delta / (double)s32) : 0.0;

    printf("fp32_size=%lld  fp16_size=%lld  delta=%lld  delta_pct=%.2f%%\n",
           s32, s16, delta, pct);
    printf("fp32  buildScratchSize=%llu  updateScratchSize=%llu\n",
           (unsigned long long)sz32.buildScratchSize,
           (unsigned long long)sz32.updateScratchSize);
    printf("fp16  buildScratchSize=%llu  updateScratchSize=%llu\n",
           (unsigned long long)sz16.buildScratchSize,
           (unsigned long long)sz16.updateScratchSize);

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
