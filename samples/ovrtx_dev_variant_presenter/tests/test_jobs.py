# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dev_variant_presenter.batch.jobs import BatchJob, MatrixMode, permutation_name
from dev_variant_presenter.models import QualitySpec, VariantChoice


def test_matrix_modes_exist():
    assert MatrixMode.FULL_CARTESIAN.value == "full_cartesian"
    assert MatrixMode.ONE_AT_A_TIME.value == "one_at_a_time"
    assert MatrixMode.CURATED.value == "curated"


def test_batch_job_defaults_and_fields():
    job = BatchJob(
        mode=MatrixMode.ONE_AT_A_TIME,
        base_selection=(VariantChoice("/W/Looks", "Doors", "Open"),),
        included={"Carpaint": ("Noir", "Sakura")},
        cameras=["/World/Cameras/Cameras/Cameras_ALL/Main_Cam_01"],
        quality=QualitySpec(),
        frame_mode="single",
        out_dir="C:/out",
    )
    assert job.curated == ()
    assert job.frame_mode == "single"
    assert job.included["Carpaint"] == ("Noir", "Sakura")


def test_permutation_name_set_variant_convention():
    sel = (
        VariantChoice("/W/Looks", "Carpaint", "Sakura"),
        VariantChoice("/W/Body", "Doors", "Open"),
    )
    assert permutation_name(sel) == "Carpaint-Sakura_Doors-Open"


def test_permutation_name_is_folder_safe():
    sel = (VariantChoice("/W/L", "Wheel/Colors", "Matte Black"),)
    name = permutation_name(sel)
    assert "/" not in name and "\\" not in name and " " not in name
    assert name == "Wheel_Colors-Matte_Black"


def test_permutation_name_empty_selection():
    assert permutation_name(()) == "default"
