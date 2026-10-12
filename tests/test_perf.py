"""Perf platform: hardware binding, store, runner, workloads (offline, fake GPUs/workloads)."""

import json

import pytest

from datascience.perf import hw, store

T4 = {"name": "Tesla T4", "compute_capability": "7.5", "vram_gb": 15}
A100 = {"name": "NVIDIA A100-SXM4-80GB", "compute_capability": "8.0", "vram_gb": 79}
TARGETS = {
    "train": {"gpu_name": "A100", "compute_capability": "8.0", "gpu_count": 1, "min_vram_gb": 40},
    "infer": {"gpu_name": "T4", "compute_capability": "7.5", "gpu_count": 1, "min_vram_gb": 14},
}


@pytest.fixture()
def box(monkeypatch):
    """Pretend this box has the given GPUs: box(T4) / box(A100, A100) / box() for CPU."""

    def _set(*gpus):
        monkeypatch.setattr(hw, "_gpus", lambda: (list(gpus), {"driver": "550"}))
        return hw.fingerprint()

    monkeypatch.setattr(hw, "load_targets", lambda: TARGETS)
    return _set


# --- hardware binding ---------------------------------------------------------------------

def test_hw_id_stable_and_gpu_specific(box):
    t4 = box(T4)["hw_id"]
    assert t4 == box(T4)["hw_id"]
    assert t4.startswith("tesla-t4-1x-")
    assert box(A100)["hw_id"] != t4
    assert box(T4, T4)["hw_id"] != t4          # count is part of the key


def test_sw_versions_do_not_change_hw_id(box, monkeypatch):
    a = box(T4)["hw_id"]
    monkeypatch.setattr(hw, "_version", lambda *d: "999")
    assert box(T4)["hw_id"] == a


def test_assert_role_matches_target_only(box):
    assert hw.assert_role("infer", box(T4))
    with pytest.raises(hw.HardwareMismatch, match="not the train target"):
        hw.assert_role("train", box(T4))
    with pytest.raises(hw.HardwareMismatch, match="no GPU visible"):
        hw.assert_role("infer", box())
    with pytest.raises(hw.HardwareMismatch, match="gpu count"):
        hw.assert_role("train", box(A100, A100))


def test_unfilled_target_is_a_mismatch(box):
    targets = {**TARGETS, "train": {"gpu_name": None, "compute_capability": None}}
    assert "not filled" in hw.mismatches("train", box(A100), targets)[0]


def test_supports_reads_the_target_not_the_box():
    assert not hw.supports("bf16", "infer", TARGETS)   # T4 endpoint: no bf16
    assert hw.supports("fp16", "infer", TARGETS)
    assert hw.supports("bf16", "train", TARGETS)


# --- store --------------------------------------------------------------------------------

def test_store_roundtrip_on_same_box(tmp_artifacts, box):
    fp = box(T4)
    store.save("infer", "infer", fp, {"grid": {}, "trials": [{}], "chosen": {"batch_size": 16}})
    assert store.load_tuned("infer", "infer", fp) == {"batch_size": 16}
    rows = store.list_results(role="infer")
    assert rows[0]["hw_id"] == fp["hw_id"] and rows[0]["trials"] == 1


def test_store_refuses_result_tuned_on_other_gpu(tmp_artifacts, box):
    store.save("train", "train", box(A100), {"trials": [], "chosen": {"batch_size": 64}})
    with pytest.raises(hw.HardwareMismatch, match="re-tune here"):
        store.load_tuned("train", "train", box(T4))
    with pytest.raises(FileNotFoundError):
        store.load_tuned("train", "kernel-x", box(T4))


def test_register_records_train_hw(tmp_artifacts, box):
    from datascience.entity.config_entity import RegistryConfig
    from datascience.registry import base

    box(A100)
    model = tmp_artifacts / "m"
    model.mkdir()
    (model / "w.bin").write_bytes(b"0")
    base.register(model, {"f1": 1.0}, {}, RegistryConfig(name="M", remote_backend="none"))

    meta = json.loads((tmp_artifacts / "registry" / "M" / "v1" / "registry_meta.json").read_text())
    assert meta["metadata"]["train_hw"]["gpus"][0]["name"] == A100["name"]
