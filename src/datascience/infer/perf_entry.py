"""Perf entries backed by the pipeline's own predictor and trainer (kept out of `perf/`, which
stays free of pipeline imports):

    datascience perf tune infer --entry datascience.infer.perf_entry:target
    datascience perf tune train --entry datascience.infer.perf_entry:train_target

Traffic is the TEST split (shortest / longest texts); quality is macro-F1 on a fixed VAL slice.
"""

from itertools import cycle, islice

from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.infer.predict import Predictor
from datascience.metrics import compute_scalar_metrics
from datascience.perf.runner import Unsupported
from datascience.utils.io import read_jsonl

VAL_SLICE = 200    # VAL rows scored per precision for quality / quality_drop
SCORE_BATCH = 16   # texts per call while scoring that slice


class InferTarget:
    lengths = ["short", "long"]

    def __init__(self):
        ev = ConfigurationManager().evaluation()
        self.text_field, self.label_field = ev.text_field, ev.label_field
        self._by_len = None

    def predictor(self, backend: str, precision: str):
        if backend != "torch":
            raise Unsupported(f"perf entry serves the torch backend only, not {backend}")
        p = Predictor(scheme=None if precision == "fp32" else precision)
        return lambda texts: p.predict_batch(texts, batch_size=len(texts) * len(p.labels))   # one forward

    def inputs(self, length: str, batch_size: int) -> list:
        if self._by_len is None:
            rows = read_jsonl(paths.TEST_FILE, required_fields=[self.text_field])
            self._by_len = sorted((r[self.text_field] for r in rows), key=len)   # stable -> deterministic
        texts = self._by_len if length == "short" else self._by_len[::-1]
        return list(islice(cycle(texts), batch_size))

    def score(self, predict) -> float:
        rows = read_jsonl(paths.VAL_FILE, required_fields=[self.text_field, self.label_field])[:VAL_SLICE]
        pred = []
        for i in range(0, len(rows), SCORE_BATCH):
            pred += [p["label"] for p in predict([r[self.text_field] for r in rows[i : i + SCORE_BATCH]])]
        return compute_scalar_metrics(["macro_f1"], [r[self.label_field] for r in rows], pred)["macro_f1"]


class TrainTarget:
    """Sweeps run in a scratch dir with no tracking: never touch the real checkpoints or resume."""

    def __init__(self):
        from datascience.components.model_trainer import ModelTrainer

        self.trainer = ModelTrainer(ConfigurationManager().trainer())

    def build_trainer(self, overrides: dict):
        scratch = {"output_dir": str(paths.PERF_DIR / "train_scratch"), "report_to": []}
        return self.trainer.build_trainer(overrides={**scratch, **overrides})


def target():
    return InferTarget()


def train_target():
    return TrainTarget()
