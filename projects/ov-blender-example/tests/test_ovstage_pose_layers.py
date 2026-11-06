# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "runtime"))
from ovstage_pose_layers import PoseLayerPopulation, pose_layer_text


def matrix(x):
    return [[1,0,0,0], [0,1,0,0], [0,0,1,0], [x,0,0,1]]


class Owner:
    def __init__(self, path): self.path = path; self.failure = None
    def replace(self, path, *, ordinal):
        if self.failure: raise self.failure
        self.path = path
        return {"ordinal": ordinal}


def test_pose_composition_retains_other_owners_and_bounds_only_owned_files(tmp_path):
    base = tmp_path / "accepted.usda"; base.write_text("#usda 1.0")
    foreign = tmp_path / "user.usda"; foreign.write_text("keep")
    owner = Owner(base); helper = PoseLayerPopulation(owner, tmp_path)
    helper.apply({"/World/A": matrix(1)}, ordinal=3)
    predecessor = helper.files
    helper.apply({"/World/B": matrix(2)}, ordinal=4)
    text = helper.files[0].read_text()
    assert text.count('over "World"') == 1
    assert 'over "A"' in text and 'over "B"' in text
    wrapper = helper.wrapper.read_text()
    assert wrapper.index(helper.files[0].as_posix()) < wrapper.index(base.as_posix())
    assert all(not path.exists() for path in predecessor)
    assert foreign.read_text() == "keep"
    assert len(list(tmp_path.glob("native-poses*.usda"))) == 2


def test_failed_candidate_preserves_accepted_cache_and_rollback_files(tmp_path):
    owner = Owner(tmp_path / "accepted.usda"); helper = PoseLayerPopulation(owner, tmp_path)
    helper.apply({"/World/A": matrix(1)}, ordinal=3)
    files = helper.files; accepted = owner.path
    failure = RuntimeError("projection failed; restored")
    failure.rollback_succeeded = True
    owner.failure = failure
    with pytest.raises(RuntimeError): helper.apply({"/World/A": matrix(2)}, ordinal=4)
    assert owner.path == accepted and helper.poses["/World/A"] == matrix(1)
    assert helper.files == files and all(path.exists() for path in files)
    assert not (tmp_path / "native-poses-4.usda").exists()


def test_new_accepted_generation_drops_previous_pose_owners(tmp_path):
    owner = Owner(tmp_path / "old.usda"); helper = PoseLayerPopulation(owner, tmp_path)
    helper.apply({"/World/Deleted": matrix(1)}, ordinal=3)
    previous = helper.files
    owner.path = tmp_path / "new.usda"
    helper.accepted_generation_replaced()
    assert not helper.poses and helper.base == owner.path
    assert all(not path.exists() for path in previous)
    helper.apply({"/World/New": matrix(2)}, ordinal=5)
    assert "Deleted" not in helper.files[0].read_text()
    assert helper.base.as_posix() in helper.wrapper.read_text()


@pytest.mark.parametrize("path,value", [("/World/bad-name",matrix(1)),("World/A",matrix(1)),("/World/A",matrix(float("nan")))])
def test_invalid_pose_is_rejected_before_any_source_replacement(path,value):
    with pytest.raises(ValueError): pose_layer_text({path:value})


def test_scoped_generation_retains_backing_pose_files_until_replacement(tmp_path):
    owner = Owner(tmp_path / "old.usda"); helper = PoseLayerPopulation(owner, tmp_path)
    helper.apply({"/Parent": matrix(0)}, ordinal=2)
    files = helper.files
    owner.path = tmp_path / "new.usda"
    helper.accepted_generation_replaced(retain_sources=True)
    assert helper.wrapper is None and helper.poses == {}
    assert helper.retained_files == files and all(p.exists() for p in files)
    helper.accepted_generation_replaced()
    assert helper.retained_files == () and not any(p.exists() for p in files)
