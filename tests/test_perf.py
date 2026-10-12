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


# --- runner -------------------------------------------------------------------------------

class _FakeWorkload:
    """Throughput grows with batch; batch 64 OOMs; batch 32 busts the latency budget."""

    name, role = "fake", "infer"

    def space(self, grid):
        from datascience.perf.runner import grid_points

        return grid_points(grid)

    def trial(self, cfg):
        bs = cfg["batch_size"]
        if bs == 64:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"throughput": bs * 100.0, "p99_ms": bs * 2.0}


def test_sweep_picks_best_feasible_and_stores_it(tmp_artifacts, box):
    from datascience.perf.runner import sweep

    fp = box(T4)
    res = sweep(_FakeWorkload(), {"batch_size": [1, 16, 32, 64]}, "throughput",
                constraints={"p99_ms": {"max": 40}}, fp=fp, track=False)

    assert res["chosen"] == {"batch_size": 16}
    reasons = {t["config"]["batch_size"]: t["reason"] for t in res["trials"]}
    assert reasons[64] == "oom" and "p99_ms" in reasons[32]
    assert store.load_tuned("infer", "fake", fp) == {"batch_size": 16}


def test_sweep_refuses_wrong_role_box(tmp_artifacts, box):
    from datascience.perf.runner import sweep

    with pytest.raises(hw.HardwareMismatch):
        sweep(_FakeWorkload(), {"batch_size": [1]}, "throughput", fp=box(A100), track=False)


def test_sweep_non_oom_errors_propagate(tmp_artifacts, box):
    from datascience.perf.runner import sweep

    class Broken(_FakeWorkload):
        def trial(self, cfg):
            raise KeyError("bug")

    with pytest.raises(KeyError):
        sweep(Broken(), {"batch_size": [1]}, "throughput", fp=box(T4), track=False)


def test_resolve_entry(tmp_path, monkeypatch):
    from datascience.perf.runner import resolve_entry

    (tmp_path / "my_entry.py").write_text("def make():\n    return 'target'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert resolve_entry("my_entry:make") == "target"
    with pytest.raises(ValueError):
        resolve_entry("my_entry")


# --- infer workload -----------------------------------------------------------------------

class _FakeTarget:
    """fp16 is slightly worse than fp32; int8 loses too much quality; onnx has no fp16."""

    _quality = {"fp32": 0.90, "fp16": 0.895, "int8": 0.80}

    def predictor(self, backend, precision):
        from datascience.perf.runner import Unsupported

        if backend == "onnx" and precision == "fp16":
            raise Unsupported("onnx fp16")
        fn = lambda batch: [0] * len(batch)  # noqa: E731
        fn.quality = self._quality[precision]
        return fn

    def inputs(self, length, batch_size):
        return ["x"] * batch_size

    def score(self, predict):
        return predict.quality


def test_infer_drops_bf16_for_t4_and_guards_quality(tmp_artifacts, box):
    from datascience.perf.runner import sweep
    from datascience.perf.workloads.infer import InferWorkload

    grid = {"backend": ["torch", "onnx"], "precision": ["fp32", "fp16", "bf16", "int8"],
            "batch_size": [4]}
    wl = InferWorkload(_FakeTarget(), ["short", "long"], iters=3, warmup=1, targets=TARGETS)
    res = sweep(wl, grid, "throughput", constraints={"quality_drop": {"max": 0.01}},
                fp=box(T4), track=False)

    precisions = {t["config"]["precision"] for t in res["trials"]}
    assert "bf16" not in precisions                                  # T4 target: never tried
    by_cfg = {(t["config"]["backend"], t["config"]["precision"]): t for t in res["trials"]}
    assert not by_cfg[("torch", "int8")]["feasible"]                 # quality guard
    assert by_cfg[("onnx", "fp16")]["reason"].startswith("unsupported")
    assert by_cfg[("torch", "fp16")]["feasible"]
    assert res["chosen"]["precision"] in ("fp32", "fp16")
    assert set(res["chosen_metrics"]["by_length"]) == {"short", "long"}  # length measured, not chosen
    assert "length" not in res["chosen"]


# --- train workload -----------------------------------------------------------------------

def test_effective_batch_is_preserved():
    from datascience.perf.workloads.train import batch_candidates, grad_accum

    assert batch_candidates(32, 2) == [1, 2, 4, 8, 16]
    for b in batch_candidates(48, 4):
        assert b * grad_accum(48, b, 4) * 4 == 48
    with pytest.raises(ValueError):
        grad_accum(32, 3, 1)


def test_max_fit_bisects_to_largest_fitting():
    from datascience.perf.workloads.train import max_fit

    probed = []

    def fits(b):
        probed.append(b)
        return b <= 12

    assert max_fit([1, 2, 3, 4, 6, 8, 12, 16, 24, 48], fits) == 12
    assert len(probed) <= 4                                   # log2(10) probes, not 10
    with pytest.raises(RuntimeError, match="out of memory"):
        max_fit([1, 2], lambda b: False)


def test_train_space_respects_train_target():
    from datascience.perf.workloads.train import TrainWorkload

    t4_train = {**TARGETS, "train": TARGETS["infer"]}
    wl = TrainWorkload(target=None, effective_batch=32, targets=t4_train)
    cfgs = wl.space({"precision": ["fp32", "fp16", "bf16"], "tf32": [False, True]})
    assert {c["precision"] for c in cfgs} == {"fp32", "fp16"}      # no bf16 on a T4 trainer
    assert {c["tf32"] for c in cfgs} == {False}


def test_training_args_mapping():
    from datascience.perf.workloads.train import training_args

    args = training_args({"precision": "bf16", "tf32": True, "per_device_batch": 8, "grad_accum": 4})
    assert args["bf16"] and not args["fp16"] and args["tf32"]
    assert args["per_device_train_batch_size"] == 8 and args["gradient_accumulation_steps"] == 4
