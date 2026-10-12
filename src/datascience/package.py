"""Package the trained GLiNER model into a Triton python-backend Multi-Model Endpoint
(MME) repository — one model dir per real model, one tar each.

This replaces the template's ONNX-ensemble text-classification packager. It ports the
working structure of `gliner_multimodel_triton/` and fixes the deployment-report bugs:

  - deps installed ONCE into a shared layer, not re-pip'd per model (#15)
  - deps from requirements.txt (pinned → reproducible build) (#16)
  - build must run on Python 3.10 so .so ABI = cpython-310 (matches Triton 23.07) (#1);
    the interpreter is asserted and any bundled .so ABI is checked
  - model version = the real data/lineage version, NOT the container-start date (#3)
  - NO `_1`/`_2` duplicate models — one dir per real model (#10)
  - per-entity threshold comes from eval's threshold.json, not a hardcoded table

`package_triton()` is the GPU/prod path. The CPU/test path does NOT use Triton — see
serve/local.py (FastAPI) + optimize.py (ONNX+int8).
"""

import json
import shutil
import sys
import tarfile

from pathlib import Path

from datascience import log
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.utils.io import read_data_version

# The Triton SageMaker DLC 23.07 ships CPython 3.10. Bundled compiled deps (.so) must match.
_TRITON_ABI = "cpython-310"


# --- the python-backend model.py served INSIDE Triton (one per model dir) -----------------
# Template string (runs inside the Triton container, not importable here): loads the bundled
# GLiNER weights + label list + per-entity thresholds, answers text -> entities.
_MODEL_PY = '''\
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, "deps")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import triton_python_backend_utils as pb_utils


def _merge_spans(texts):
    merged = []
    for t in (x.strip() for x in texts):
        if t and not any(t in m for m in merged):
            merged.append(t)
    return " ".join(merged)


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from gliner import GLiNER

        model_dir = os.path.join(_HERE, "model")
        meta = json.load(open(os.path.join(_HERE, "serving_meta.json")))
        self.version = meta["version"]                 # real model version, not a date
        self.labels = meta["labels"]
        self.flat_ner = meta.get("flat_ner", True)
        self.max_width = meta.get("max_width")
        thr = meta.get("threshold", 0.5)               # per-entity dict (from eval) or float
        self.entity_thresholds = thr if isinstance(thr, dict) else None
        self.default_threshold = min(thr.values()) if isinstance(thr, dict) else thr

        want_gpu = os.environ.get("SERVE_DEVICE", "gpu") == "gpu"
        self.device = "cuda" if (want_gpu and torch.cuda.is_available()) else "cpu"
        kw = {"load_tokenizer": True}
        if self.max_width is not None:
            kw["max_width"] = self.max_width
        self.model = GLiNER.from_pretrained(model_dir, **kw).to(self.device).eval()

    def execute(self, requests):
        # Output shape the bds-genai-sse GLiNER clients expect: a JSON string per text of
        # {"predictions": {<Label>: <merged span text>}} (PascalCase labels; the client
        # lowercases the first letter and wraps as parsed["output"]). Same parser dict the
        # FastAPI path returns, so the endpoint is swappable without client changes.
        responses = []
        for req in requests:
            arr = pb_utils.get_input_tensor_by_name(req, "text_input").as_numpy()
            texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in arr.reshape(-1)]
            batch = self.model.batch_predict_entities(
                texts, self.labels, threshold=self.default_threshold, flat_ner=self.flat_ner
            )
            out = []
            for ents in batch:
                if self.entity_thresholds:
                    ents = [e for e in ents
                            if e["score"] >= self.entity_thresholds.get(e["label"], self.default_threshold)]
                by_label = {}
                for e in ents:
                    by_label.setdefault(e["label"], []).append(e["text"])
                predictions = {lbl: _merge_spans(v) for lbl, v in by_label.items()}
                out.append(json.dumps({"predictions": predictions, "version": self.version},
                                      ensure_ascii=False))
            t = pb_utils.Tensor("text_output", np.array(out, dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        self.model = None
'''


def _config_pbtxt(name: str, t, kind: str) -> str:
    return f'''name: "{name}"
backend: "python"
max_batch_size: {t.max_batch_size}

input [
  {{ name: "text_input", data_type: TYPE_STRING, dims: [ -1 ] }}
]
output [
  {{ name: "text_output", data_type: TYPE_STRING, dims: [ -1 ] }}
]

instance_group [
  {{ kind: {kind}, count: {t.instance_group_count} }}
]

dynamic_batching {{
  max_queue_delay_microseconds: {t.max_queue_delay_us}
}}
'''


def _serving_version() -> str:
    """Real model version for lineage — from the data fingerprint written at transform time,
    NOT the container-start date (bug #3). 'unversioned' if absent."""
    dv = read_data_version()
    return f"data-{dv['sha256']}" if dv else "unversioned"


def _assert_py310():
    if not sys.version.startswith("3.10"):
        raise RuntimeError(
            f"package must run on Python 3.10 for Triton-container ABI ({_TRITON_ABI}); "
            f"got {sys.version.split()[0]}. Use the tri310 env."
        )


def _bundle_deps(shared_deps: Path) -> None:
    """Install pinned serving deps ONCE into a shared layer (#15, #16), then check ABI (#1)."""
    if shared_deps.exists():
        log.info(f"reusing bundled deps at {shared_deps}")
        return
    import subprocess

    shared_deps.mkdir(parents=True, exist_ok=True)
    req = paths.PROJECT_ROOT / "requirements.txt"
    log.info(f"bundling deps -> {shared_deps} (from {req.name}, Python 3.10)")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(req),
         "gliner==0.2.21", "--target", str(shared_deps)]
    )
    _assert_abi(shared_deps)


def _assert_abi(deps: Path) -> None:
    bad = [p.name for p in deps.rglob("*.so") if "cpython-3" in p.name and _TRITON_ABI not in p.name]
    if bad:
        raise RuntimeError(f"bundled .so ABI != {_TRITON_ABI} (won't load in Triton 23.07): {bad[:3]}")


def package_triton():
    """Build a Triton python-backend MME repo (one dir per real model) + per-model tars."""
    _assert_py310()
    cm = ConfigurationManager()
    sv = cm.serving()
    t, sm = sv.triton, sv.sagemaker
    version = _serving_version()
    kind = "KIND_GPU" if sv.get("device", "gpu") == "gpu" else "KIND_CPU"

    names = [m["name"] for m in sm.models]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate model names in serving.sagemaker.models: {names}")

    if paths.TRITON_REPO.exists():
        shutil.rmtree(paths.TRITON_REPO)          # regenerable artifact — safe to rebuild
    paths.MME_TARS.mkdir(parents=True, exist_ok=True)

    shared_deps = paths.ARTIFACTS / "triton_deps"
    _bundle_deps(shared_deps)

    labels = _labels()
    threshold, flat_ner, max_width = _serving_meta(cm)

    tars = {}
    for m in sm.models:
        name, ckpt = m["name"], Path(m["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint for '{name}' not found: {ckpt}")

        base = paths.TRITON_REPO / name / "1"
        base.mkdir(parents=True, exist_ok=True)
        (base / "model.py").write_text(_MODEL_PY)
        shutil.copytree(ckpt, base / "model")
        (base / "serving_meta.json").write_text(json.dumps(
            {"version": version, "labels": labels, "threshold": threshold,
             "flat_ner": flat_ner, "max_width": max_width}, ensure_ascii=False, indent=2))
        shutil.copytree(shared_deps, base / "deps")     # shared layer, built once (#15)
        (paths.TRITON_REPO / name / "config.pbtxt").write_text(_config_pbtxt(name, t, kind))

        tar_path = paths.MME_TARS / f"{name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(paths.TRITON_REPO / name, arcname=name)
        tars[name] = tar_path
        log.info(f"packaged {name} -> {tar_path} (version={version}, {kind})")

    log.info(f"Triton MME: {len(tars)} model(s), no duplicates -> {paths.MME_TARS}")
    return tars


def _labels() -> list:
    from datascience.model import load_labels

    return load_labels(paths.LABEL_MAP)


def _serving_meta(cm):
    """(threshold [per-entity dict from eval, else 0.5], flat_ner, max_width)."""
    ev = cm.evaluation()
    threshold = 0.5
    if paths.THRESHOLD_FILE.exists():
        threshold = json.loads(paths.THRESHOLD_FILE.read_text())["threshold"]
    return threshold, ev.flat_ner, ev.max_width


def package_bundle():
    raise NotImplementedError("bundle packaging not migrated for GLiNER; use package_triton")
