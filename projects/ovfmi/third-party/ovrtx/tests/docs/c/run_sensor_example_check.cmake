# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

if(NOT DEFINED EXAMPLE_NAME)
    message(FATAL_ERROR "EXAMPLE_NAME is required")
endif()

if(NOT DEFINED EXAMPLE_EXECUTABLE)
    message(FATAL_ERROR "EXAMPLE_EXECUTABLE is required")
endif()

execute_process(
    COMMAND "${EXAMPLE_EXECUTABLE}"
    RESULT_VARIABLE example_result
    OUTPUT_VARIABLE example_stdout
    ERROR_VARIABLE example_stderr
)

set(example_output "${example_stdout}\n${example_stderr}")

if(NOT example_result EQUAL 0)
    message(FATAL_ERROR
        "${EXAMPLE_NAME} example failed with exit code ${example_result}\n"
        "${example_output}"
    )
endif()

if(EXAMPLE_NAME STREQUAL "lidar")
    string(REGEX MATCH
        "valid points=([0-9]+), mean intensity=([-+0-9.eE]+), max [^=]+=([0-9]+) ns"
        lidar_match
        "${example_output}"
    )
    if(NOT lidar_match)
        message(FATAL_ERROR "Could not parse lidar example output:\n${example_output}")
    endif()

    set(valid_points "${CMAKE_MATCH_1}")
    set(mean_intensity "${CMAKE_MATCH_2}")
    set(max_time_offset_ns "${CMAKE_MATCH_3}")

    if(valid_points LESS_EQUAL 1000)
        message(FATAL_ERROR "Expected more than 1000 lidar points, got ${valid_points}")
    endif()
    if(mean_intensity LESS 0.0)
        message(FATAL_ERROR "Expected non-negative mean lidar intensity, got ${mean_intensity}")
    endif()
    if(max_time_offset_ns LESS_EQUAL 0)
        message(FATAL_ERROR "Expected positive lidar max time offset, got ${max_time_offset_ns}")
    endif()
elseif(EXAMPLE_NAME STREQUAL "radar")
    string(REGEX MATCHALL
        "step [0-9]+: valid points=[0-9]+, RCS min/max=\\[[-+0-9.eE]+, [-+0-9.eE]+\\], radial velocity min/max=\\[[-+0-9.eE]+, [-+0-9.eE]+\\] m/s"
        radar_steps
        "${example_output}"
    )
    list(LENGTH radar_steps radar_step_count)
    if(NOT radar_step_count EQUAL 10)
        message(FATAL_ERROR
            "Expected 10 radar step summaries, got ${radar_step_count}\n"
            "${example_output}"
        )
    endif()

    string(REGEX MATCH
        "Observed ([0-9]+) detections with \\|radial velocity\\| > 0\\.1 m/s; max \\|radial velocity\\| = ([-+0-9.eE]+) m/s"
        radar_observed
        "${example_output}"
    )
    if(NOT radar_observed)
        message(FATAL_ERROR "Could not parse radar observation summary:\n${example_output}")
    endif()

    set(moving_detections "${CMAKE_MATCH_1}")
    set(max_abs_radial_velocity "${CMAKE_MATCH_2}")

    if(moving_detections LESS_EQUAL 0)
        message(FATAL_ERROR "Expected moving radar detections, got ${moving_detections}")
    endif()
    if(max_abs_radial_velocity LESS_EQUAL 0.1)
        message(FATAL_ERROR
            "Expected max |radial velocity| > 0.1 m/s, got ${max_abs_radial_velocity}"
        )
    endif()
else()
    message(FATAL_ERROR "Unknown EXAMPLE_NAME: ${EXAMPLE_NAME}")
endif()
