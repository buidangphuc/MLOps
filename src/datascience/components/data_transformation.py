from sklearn.model_selection import train_test_split

from datascience import log
from datascience.constants import paths
from datascience.entity.config_entity import DataTransformationConfig
from datascience.utils.io import atomic_write_json, data_fingerprint, read_jsonl, write_jsonl

# Shared split helpers live in utils.split; keep the historical underscore
# names as aliases so reference code (and tests) read the same as before.
from datascience.utils.split import (
    can_stratify as _can_stratify,
    drop_cross_split_dups as _drop_cross_split_dups,
    group_split as _group_split,
)


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):
        """Stratified (or group-aware) train/val/test split + label map.

        - `val` drives early-stop/model selection; `test` is touched once at eval (no leakage).
        - Leakage guard drops text duplicated across splits.
        - label->id map is built from TRAIN only and saved (shared by train/eval/infer).
        """
        cfg = self.config
        rows = read_jsonl(paths.RAW_DATA, required_fields=[cfg.text_field, cfg.label_field])

        holdout = cfg.val_size + cfg.test_size
        rel_test = cfg.test_size / holdout

        if cfg.group_field:
            log.info(f"group-aware split by '{cfg.group_field}' (no leakage across groups)")
            groups = [r[cfg.group_field] for r in rows]
            train, remainder = _group_split(rows, groups, holdout, cfg.seed)
            rem_groups = [r[cfg.group_field] for r in remainder]
            val, test = _group_split(remainder, rem_groups, rel_test, cfg.seed)
        else:
            labels = [r[cfg.label_field] for r in rows]
            strat = labels if _can_stratify(labels) else None
            if strat is None:
                log.warning("a class is too small to stratify -> random split")
            train, remainder = train_test_split(rows, test_size=holdout, random_state=cfg.seed, stratify=strat)
            rem_labels = [r[cfg.label_field] for r in remainder]
            strat2 = rem_labels if _can_stratify(rem_labels) else None
            val, test = train_test_split(remainder, test_size=rel_test, random_state=cfg.seed, stratify=strat2)

        if cfg.leakage_check:
            train, val, test = _drop_cross_split_dups(train, val, test, cfg.text_field)

        # Label map from TRAIN only (sorted for determinism).
        labels = sorted({r[cfg.label_field] for r in train}, key=str)
        label2id = {lbl: i for i, lbl in enumerate(labels)}
        atomic_write_json(
            paths.LABEL_MAP,
            {"label2id": label2id, "id2label": {i: lbl for lbl, i in label2id.items()}},
        )

        for name, part, path in [
            ("train", train, paths.TRAIN_FILE),
            ("val", val, paths.VAL_FILE),
            ("test", test, paths.TEST_FILE),
        ]:
            write_jsonl(path, part)
            log.info(f"{name} split: {len(part)} rows")

        # data 'version' for lineage (hash of the ingested data); DVC-remote push = seam
        atomic_write_json(paths.DATA_VERSION, data_fingerprint(paths.RAW_DATA))
        log.info(f"label map ({len(labels)} classes) saved: {label2id}")
