import numpy as np

from datascience import metrics as M
from datascience import log
from datascience.constants import paths
from datascience.entity.config_entity import ModelEvaluationConfig
from datascience.model import load_label_map, load_trained
from datascience.utils.io import atomic_write_json, read_jsonl, write_jsonl
from datascience.utils.runtime import pick_device, set_run_status


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _predict_logits(self, model, tokenizer, rows, device):
        import torch

        chunks = []
        bs = self.config.eval_batch_size
        for i in range(0, len(rows), bs):
            texts = [r[self.config.text_field] for r in rows[i : i + bs]]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                chunks.append(model(**enc).logits.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def evaluate(self):
        set_run_status("evaluate", "running")
        try:
            self._run()
        except BaseException as e:
            set_run_status("evaluate", "failed", error=str(e))
            raise
        set_run_status("evaluate", "done")

    def _load_rows(self, path, label2id):
        cfg = self.config
        rows = [
            r
            for r in read_jsonl(path, required_fields=[cfg.text_field, cfg.label_field])
            if r[cfg.label_field] in label2id
        ]
        labels = np.array([label2id[r[cfg.label_field]] for r in rows])
        return rows, labels

    def _run(self):
        cfg = self.config
        device = pick_device()
        label2id, id2label = load_label_map(paths.LABEL_MAP)
        model, tokenizer = load_trained(paths.MODEL_DIR, device)

        test_rows, y_true = self._load_rows(paths.TEST_FILE, label2id)
        test_logits = self._predict_logits(model, tokenizer, test_rows, device)
        probs = M.softmax(test_logits)
        y_pred = probs.argmax(axis=1)

        # Scalar metrics (top-level — what the gate reads by name)
        result = M.compute_scalar_metrics(cfg.compute, y_true, y_pred, probs)
        log.info(f"test metrics: { {k: round(v, 4) for k, v in result.items()} }")

        if cfg.per_class:
            report, cm, classes = M.per_class_report(y_true, y_pred, id2label)
            result["per_class"] = report
            result["confusion_matrix"] = {"labels": classes, "matrix": cm}

        if cfg.bootstrap_ci:
            from sklearn.metrics import accuracy_score, f1_score

            fns = {
                "accuracy": lambda a, b: accuracy_score(a, b),
                "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
                "micro_f1": lambda a, b: f1_score(a, b, average="micro", zero_division=0),
            }
            fn = fns.get(cfg.objective)
            if fn:
                lo, hi = M.bootstrap_ci(fn, y_true, y_pred)
                result[f"{cfg.objective}_ci95"] = [lo, hi]
                log.info(f"{cfg.objective} 95% CI: [{lo:.4f}, {hi:.4f}]")

        if cfg.calibration:
            ece = M.expected_calibration_error(probs, y_true)
            val_rows, val_labels = self._load_rows(paths.VAL_FILE, label2id)
            val_logits = self._predict_logits(model, tokenizer, val_rows, device)
            temperature = M.fit_temperature(val_logits, val_labels)
            ece_after = M.expected_calibration_error(M.softmax(test_logits / temperature), y_true)
            result["calibration"] = {"ece": ece, "ece_after_temp": ece_after, "temperature": temperature}
            atomic_write_json(paths.MODEL_DIR / "temperature.json", {"temperature": temperature})
            log.info(f"calibration ECE {ece:.4f} -> {ece_after:.4f} (T={temperature:.3f})")

        if cfg.error_dump:
            errors = [
                {
                    "text": r[cfg.text_field],
                    "true": id2label[int(y_true[i])],
                    "pred": id2label[int(y_pred[i])],
                    "score": float(probs[i].max()),
                }
                for i, r in enumerate(test_rows)
                if y_true[i] != y_pred[i]
            ]
            write_jsonl(paths.EVAL_DIR / "errors.jsonl", errors)
            log.info(f"error dump: {len(errors)} misclassified -> errors.jsonl")

        if cfg.slice_field:
            result["slices"] = self._slices(test_rows, y_true, y_pred, cfg.slice_field)

        atomic_write_json(paths.METRICS_FILE, result)
        log.info(f"metrics saved -> {paths.METRICS_FILE}")

    def _slices(self, rows, y_true, y_pred, field):
        from sklearn.metrics import f1_score

        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r.get(field)), []).append(i)
        out = {}
        for key, idx in groups.items():
            idx = np.array(idx)
            out[key] = {
                "n": int(len(idx)),
                "macro_f1": float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)),
            }
        return out
