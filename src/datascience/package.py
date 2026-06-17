"""PACKAGE — assemble deployable artifacts from the optimized ONNX model.

Two targets:
  • bundle  → one self-contained dir (onnx + tokenizer + label_map + temperature + model_card.json)
  • triton  → a Triton ensemble model_repository (text in → {label, score}) + docker-compose

Lean: straightforward file assembly. Triton/SageMaker bits run on the GPU box.
"""

import json
import shutil
import subprocess

from datetime import datetime, timezone
from pathlib import Path

from src.datascience import log
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.constants import paths
from src.datascience.model import load_label_map


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _temperature():
    tf = paths.MODEL_DIR / "temperature.json"
    return json.loads(tf.read_text())["temperature"] if tf.exists() else 1.0


def _model_card(cm, label2id, id2label):
    metrics = json.loads(paths.METRICS_FILE.read_text()) if paths.METRICS_FILE.exists() else {}
    return {
        "name": cm.params.registry.name,
        "base_model": cm.params.model.base_model,
        "max_length": cm.params.model.max_length,
        "task": "text-classification",
        "classes": [id2label[i] for i in sorted(id2label)],
        "num_labels": len(label2id),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "optimize": dict(cm.serving().optimize),
        "temperature": _temperature(),
        "io_schema": {"input": {"text": "string"}, "output": {"label": "string", "score": "float"}},
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def package_bundle():
    cm = ConfigurationManager()
    label2id, id2label = load_label_map(paths.LABEL_MAP)

    out = paths.BUNDLE_DIR
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(paths.OPTIMIZED_DIR, out)  # onnx + tokenizer
    shutil.copy(paths.LABEL_MAP, out / "label_map.json")
    (out / "temperature.json").write_text(json.dumps({"temperature": _temperature()}))
    (out / "model_card.json").write_text(
        json.dumps(_model_card(cm, label2id, id2label), indent=2, ensure_ascii=False)
    )
    log.info(f"bundle ready -> {out}")


# --- Triton ensemble: preprocess(py) -> classifier(onnx) -> postprocess(py) -----

_PREPROCESS_PY = '''import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        here = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(here, "tokenizer"))
        self.max_length = json.load(open(os.path.join(here, "settings.json")))["max_length"]

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            texts = [(t[0].decode("utf-8") if isinstance(t[0], bytes) else str(t[0])) for t in raw]
            enc = self.tokenizer(
                texts, truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="np",
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("input_ids", enc["input_ids"].astype(np.int64)),
                pb_utils.Tensor("attention_mask", enc["attention_mask"].astype(np.int64)),
            ]))
        return responses
'''

_POSTPROCESS_PY = '''import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        here = os.path.join(args["model_repository"], args["model_version"])
        lm = json.load(open(os.path.join(here, "label_map.json")))
        self.id2label = {int(k): v for k, v in lm["id2label"].items()}
        self.temperature = json.load(open(os.path.join(here, "temperature.json"))).get("temperature", 1.0)

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy().astype(np.float32)
            z = logits / self.temperature
            e = np.exp(z - z.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            idx = probs.argmax(axis=1)
            labels = np.array([[self.id2label[int(i)].encode("utf-8")] for i in idx], dtype=object)
            scores = probs.max(axis=1).reshape(-1, 1).astype(np.float32)
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("label", labels),
                pb_utils.Tensor("score", scores),
            ]))
        return responses
'''


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def package_triton():
    cm = ConfigurationManager()
    t = cm.serving().triton
    label2id, _ = load_label_map(paths.LABEL_MAP)
    n = len(label2id)
    mbs = t.max_batch_size
    kind = "KIND_GPU" if t.device == "gpu" else "KIND_CPU"
    pref = ", ".join(str(b) for b in t.preferred_batch_sizes)

    repo = paths.TRITON_REPO
    if repo.exists():
        shutil.rmtree(repo)

    # preprocess (python)
    from transformers import AutoTokenizer

    _write(repo / "preprocess" / "1" / "model.py", _PREPROCESS_PY)
    AutoTokenizer.from_pretrained(paths.OPTIMIZED_DIR).save_pretrained(repo / "preprocess" / "1" / "tokenizer")
    _write(repo / "preprocess" / "1" / "settings.json", json.dumps({"max_length": cm.params.model.max_length}))
    _write(repo / "preprocess" / "config.pbtxt", f'''name: "preprocess"
backend: "python"
max_batch_size: {mbs}
input [{{ name: "TEXT" data_type: TYPE_STRING dims: [1] }}]
output [
  {{ name: "input_ids" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "attention_mask" data_type: TYPE_INT64 dims: [-1] }}
]
instance_group [{{ kind: KIND_CPU count: 1 }}]
''')

    # classifier (onnxruntime)
    _write(repo / "classifier" / "config.pbtxt", f'''name: "classifier"
backend: "onnxruntime"
max_batch_size: {mbs}
input [
  {{ name: "input_ids" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "attention_mask" data_type: TYPE_INT64 dims: [-1] }}
]
output [{{ name: "logits" data_type: TYPE_FP32 dims: [{n}] }}]
dynamic_batching {{ preferred_batch_size: [{pref}] max_queue_delay_microseconds: {t.max_queue_delay_us} }}
instance_group [{{ kind: {kind} count: {t.instance_count} }}]
# GPU fp16/TensorRT (tune on the GPU box):
# optimization {{ execution_accelerators {{ gpu_execution_accelerator [
#   {{ name: "tensorrt" parameters {{ key: "precision_mode" value: "FP16" }} }} ] }} }}
''')
    (repo / "classifier" / "1").mkdir(parents=True, exist_ok=True)
    shutil.copy(paths.OPTIMIZED_DIR / "model.onnx", repo / "classifier" / "1" / "model.onnx")

    # postprocess (python)
    _write(repo / "postprocess" / "1" / "model.py", _POSTPROCESS_PY)
    shutil.copy(paths.LABEL_MAP, repo / "postprocess" / "1" / "label_map.json")
    (repo / "postprocess" / "1" / "temperature.json").write_text(json.dumps({"temperature": _temperature()}))
    _write(repo / "postprocess" / "config.pbtxt", f'''name: "postprocess"
backend: "python"
max_batch_size: {mbs}
input [{{ name: "logits" data_type: TYPE_FP32 dims: [{n}] }}]
output [
  {{ name: "label" data_type: TYPE_STRING dims: [1] }},
  {{ name: "score" data_type: TYPE_FP32 dims: [1] }}
]
instance_group [{{ kind: KIND_CPU count: 1 }}]
''')

    # ensemble
    (repo / "text_classifier" / "1").mkdir(parents=True, exist_ok=True)
    _write(repo / "text_classifier" / "config.pbtxt", f'''name: "text_classifier"
platform: "ensemble"
max_batch_size: {mbs}
input [{{ name: "TEXT" data_type: TYPE_STRING dims: [1] }}]
output [
  {{ name: "label" data_type: TYPE_STRING dims: [1] }},
  {{ name: "score" data_type: TYPE_FP32 dims: [1] }}
]
ensemble_scheduling {{ step [
  {{ model_name: "preprocess" model_version: -1
     input_map {{ key: "TEXT" value: "TEXT" }}
     output_map {{ key: "input_ids" value: "input_ids" }}
     output_map {{ key: "attention_mask" value: "attention_mask" }} }},
  {{ model_name: "classifier" model_version: -1
     input_map {{ key: "input_ids" value: "input_ids" }}
     input_map {{ key: "attention_mask" value: "attention_mask" }}
     output_map {{ key: "logits" value: "logits" }} }},
  {{ model_name: "postprocess" model_version: -1
     input_map {{ key: "logits" value: "logits" }}
     output_map {{ key: "label" value: "label" }}
     output_map {{ key: "score" value: "score" }} }}
] }}
''')

    _write_compose()
    log.info(f"Triton model_repository ready -> {repo}")


def _write_compose():
    # The python backends need `transformers`; install it into the Triton image at start.
    compose = paths.PROJECT_ROOT / "infra" / "triton" / "docker-compose.yml"
    _write(compose, '''services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.08-py3
    # GPU box: uncomment ->
    # deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] } } }
    ports: ["8000:8000", "8001:8001", "8002:8002"]
    volumes:
      - ../../artifacts/model_repository:/models
    command: >
      bash -lc "pip install -q transformers && tritonserver --model-repository=/models"
''')
    log.info(f"docker-compose -> {compose}")
