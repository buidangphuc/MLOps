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


# --- profile ------------------------------------------------------------------------------

def test_busy_split_separates_memcpy_and_idle():
    from types import SimpleNamespace

    from datascience.perf.profile import busy_split

    ev = [SimpleNamespace(key="ampere_sgemm_128x64", self_device_time_total=6000),
          SimpleNamespace(key="Memcpy HtoD (Pageable -> Device)", self_device_time_total=2000),
          SimpleNamespace(key="aten::linear", self_device_time_total=0)]
    s = busy_split(ev, wall_ms=10.0)
    assert (s["compute_ms"], s["memcpy_ms"], s["idle_ms"]) == (6.0, 2.0, 2.0)
    assert s["gpu_busy_frac"] == 0.8


# --- kernel workload ----------------------------------------------------------------------

def test_kernel_sweep_rejects_fast_but_wrong_config(tmp_artifacts, box, monkeypatch):
    """Config 'fast' is quickest but numerically wrong -> the correct, slower one wins."""
    import sys
    from types import SimpleNamespace

    from datascience.perf.runner import sweep
    from datascience.perf.workloads.kernel import KernelWorkload

    class _Out:  # the slice of the tensor API the harness uses for the error check
        def __init__(self, v):
            self.v = v

        def float(self):
            return self

        def __sub__(self, o):
            return _Out(self.v - o.v)

        def abs(self):
            return _Out(abs(self.v))

        def max(self):
            return self.v

    # no GPU here: stub torch/triton; do_bench "times" by which config produced the output
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="f16", cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "triton.testing", SimpleNamespace(
        do_bench=lambda fn, **kw: 0.01 if fn().v == 1.5 else 0.02))
    kernel = SimpleNamespace(
        space=lambda shape, grid: [{"cfg": "fast"}, {"cfg": "ok"}],
        make_inputs=lambda shape, dtype: None,
        reference=lambda inputs: _Out(1.0),
        run=lambda cfg, inputs: _Out(1.5 if cfg["cfg"] == "fast" else 1.0),
        bytes_moved=lambda shape, dtype: 10**6,
    )
    wl = KernelWorkload("fake", [8, 16], "fp16", role="infer", module=kernel)

    res = sweep(wl, {}, "ms", "min", wl.constraints(), fp=box(T4), track=False)

    assert res["chosen"] == {"cfg": "ok"}
    wrong = next(t for t in res["trials"] if t["config"]["cfg"] == "fast")
    assert not wrong["feasible"] and "max_abs_err" in wrong["reason"]
    assert store.load_tuned("infer", "kernel-fake-8x16-fp16", box(T4)) == {"cfg": "ok"}


# --- PTQ schemes --------------------------------------------------------------------------

T4_ONLY = {"fp32", "fp16", "int8-wo", "int8-dyn", "nf4", "bnb-int8"}   # cc 7.5


def _targets(cc):
    return {**TARGETS, "infer": {**TARGETS["infer"], "compute_capability": cc}}


def test_schemes_on_t4_infer_target():
    from datascience.perf import quant

    assert set(quant.supported("infer", TARGETS)) == T4_ONLY           # no bf16 / int4-wo / fp8


def test_schemes_on_h100_infer_target():
    from datascience.perf import quant

    assert set(quant.supported("infer", _targets("9.0"))) == set(quant.SCHEMES)


def test_unsupported_scheme_is_never_measured(tmp_artifacts, box):
    from datascience.perf.runner import sweep
    from datascience.perf.workloads.infer import InferWorkload

    class Target(_FakeTarget):
        _quality = {"fp32": 0.9, "fp16": 0.9, "fp8": 0.9}

    wl = InferWorkload(Target(), ["short"], iters=2, warmup=1, targets=TARGETS)
    res = sweep(wl, {"backend": ["torch"], "precision": ["fp32", "fp8", "fp16"], "batch_size": [2]},
                "throughput", fp=box(T4), track=False)
    assert {t["config"]["precision"] for t in res["trials"]} == {"fp32", "fp16"}


def test_ort_int8_is_not_a_scheme_so_it_survives_filtering(tmp_artifacts, box):
    from datascience.perf.workloads.infer import InferWorkload

    wl = InferWorkload(_FakeTarget(), ["short"], targets=TARGETS)
    pts = wl.space({"backend": ["onnx"], "precision": ["fp32", "int8", "bf16"], "batch_size": [1]})
    assert [p["precision"] for p in pts] == ["fp32", "int8"]


def test_backend_that_cannot_build_a_scheme_is_infeasible_and_sweep_continues(tmp_artifacts, box):
    from datascience.perf.runner import Unsupported, sweep
    from datascience.perf.workloads.infer import InferWorkload

    class Target(_FakeTarget):
        _quality = {"fp32": 0.9, "int8-wo": 0.9}

        def predictor(self, backend, precision):
            if precision == "int8-wo":
                raise Unsupported("torchao is not installed")
            return super().predictor(backend, precision)

    wl = InferWorkload(Target(), ["short"], iters=2, warmup=1, targets=TARGETS)
    res = sweep(wl, {"backend": ["torch"], "precision": ["fp32", "int8-wo"], "batch_size": [2]},
                "throughput", fp=box(T4), track=False)
    by = {t["config"]["precision"]: t for t in res["trials"]}
    assert not by["int8-wo"]["feasible"] and by["int8-wo"]["reason"].startswith("unsupported")
    assert res["chosen"]["precision"] == "fp32"


def test_fastest_scheme_over_the_quality_drop_is_not_chosen(tmp_artifacts, box, monkeypatch):
    from datascience.perf.runner import sweep
    from datascience.perf.workloads import infer
    from datascience.perf.workloads.infer import InferWorkload

    class Target(_FakeTarget):
        _quality = {"fp32": 0.90, "fp16": 0.895, "int8-dyn": 0.70}

    speed = {"fp32": 3.0, "fp16": 2.0, "int8-dyn": 0.5}   # int8-dyn: fastest, but loses 0.2

    def fake_time(fn, iters, warmup):
        return [speed[Target.current]] * iters

    class Probe(Target):
        def predictor(self, backend, precision):
            Target.current = precision
            return super().predictor(backend, precision)

    monkeypatch.setattr(infer, "time_ms", fake_time)
    wl = InferWorkload(Probe(), ["short"], iters=3, warmup=1, targets=TARGETS)
    res = sweep(wl, {"backend": ["torch"], "precision": ["fp32", "fp16", "int8-dyn"], "batch_size": [4]},
                "throughput", constraints={"quality_drop": {"max": 0.01}}, fp=box(T4), track=False)
    by = {t["config"]["precision"]: t for t in res["trials"]}
    assert not by["int8-dyn"]["feasible"] and "quality_drop" in by["int8-dyn"]["reason"]
    assert by["int8-dyn"]["metrics"]["throughput"] > by["fp16"]["metrics"]["throughput"]
    assert res["chosen"]["precision"] == "fp16"


def test_tokens_per_s_reported_when_target_generates(monkeypatch):
    from datascience.perf.workloads import infer
    from datascience.perf.workloads.infer import InferWorkload

    monkeypatch.setattr(infer, "time_ms", lambda fn, iters, warmup: [100.0] * iters)
    target = _FakeTarget()
    target.new_tokens = 8
    wl = InferWorkload(target, ["short"], iters=2, warmup=1, targets=TARGETS)
    wl.space({"backend": ["torch"], "precision": ["fp32"], "batch_size": [4]})
    m = wl.trial({"backend": "torch", "precision": "fp32", "batch_size": 4})
    assert m["throughput"] == 40.0 and m["tokens_per_s"] == 320.0   # 4 items x 8 tokens / 0.1 s


@pytest.mark.parametrize("scheme,lib", [("int8-wo", "torchao"), ("int8-dyn", "torchao"),
                                        ("nf4", "bitsandbytes"), ("bnb-int8", "bitsandbytes")])
def test_apply_without_the_library_is_unsupported(scheme, lib, monkeypatch):
    import sys

    from datascience.perf import quant
    from datascience.perf.runner import Unsupported

    monkeypatch.setitem(sys.modules, lib, None)                 # import raises ImportError
    monkeypatch.setitem(sys.modules, f"{lib}.quantization", None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)   # past the GPU check, down to the import
    with pytest.raises(Unsupported, match=lib):
        quant.apply(object() if scheme.startswith("int8-") else "model_dir", scheme)


def test_apply_unknown_scheme_and_missing_gpu_are_unsupported(monkeypatch):
    from datascience.perf import quant
    from datascience.perf.runner import Unsupported

    with pytest.raises(Unsupported, match="unknown scheme"):
        quant.apply(object(), "int3")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    for scheme in ("fp8", "int4-wo", "nf4"):
        with pytest.raises(Unsupported, match="CUDA"):
            quant.apply("model_dir", scheme)


def test_apply_dtype_casts_and_load_trained_uses_it():
    import torch

    from datascience.perf import quant

    assert quant.apply(torch.nn.Linear(2, 2), "fp16").weight.dtype == torch.float16
    assert quant.apply(torch.nn.Linear(2, 2).half(), "fp32").weight.dtype == torch.float32


def test_perf_entry_resolves(tmp_artifacts, monkeypatch):
    from datascience.infer import perf_entry
    from datascience.perf.runner import Unsupported, resolve_entry

    ev = type("Ev", (), {"text_field": "text", "label_field": "label"})()
    monkeypatch.setattr(perf_entry.ConfigurationManager, "evaluation", lambda self: ev)
    monkeypatch.setattr(perf_entry.ConfigurationManager, "__init__", lambda self, *a, **k: None)

    def write(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    write(tmp_artifacts / "data_transformation" / "test.jsonl",
          [{"text": t, "label": "a"} for t in ("mid text", "tiny", "the longest text of them all")])
    write(tmp_artifacts / "data_transformation" / "val.jsonl",
          [{"text": "x", "label": "a"}, {"text": "y", "label": "b"}, {"text": "z", "label": "a"}])

    class FakePredictor:
        labels = ["a", "b"]
        schemes = []

        def __init__(self, scheme=None):
            FakePredictor.schemes.append(scheme)

        def predict_batch(self, texts, batch_size=32):
            return [{"label": "a", "score": 1.0} for _ in texts]

    monkeypatch.setattr(perf_entry, "Predictor", FakePredictor)
    monkeypatch.setattr(perf_entry, "VAL_SLICE", 2)

    t = resolve_entry("datascience.infer.perf_entry:target")
    assert t.lengths == ["short", "long"]
    assert t.inputs("short", 2) == ["tiny", "mid text"]
    assert t.inputs("long", 4)[0] == "the longest text of them all" and len(t.inputs("long", 4)) == 4
    predict = t.predictor("torch", "int8-wo")
    assert FakePredictor.schemes == ["int8-wo"] and len(predict(["q", "r"])) == 2
    t.predictor("torch", "fp32")
    assert FakePredictor.schemes[-1] is None
    assert t.score(predict) == pytest.approx(1 / 3)   # slice = a,b vs all-"a": F1 2/3 and 0
    with pytest.raises(Unsupported):
        t.predictor("onnx", "fp32")

    seen = {}
    tt = perf_entry.TrainTarget.__new__(perf_entry.TrainTarget)
    tt.trainer = type("T", (), {"build_trainer": lambda self, overrides: seen.update(overrides) or "trainer"})()
    assert tt.build_trainer({"bf16": True}) == "trainer"
    assert seen["bf16"] and seen["output_dir"].startswith(str(tmp_artifacts))   # scratch dir, not CKPT_DIR
