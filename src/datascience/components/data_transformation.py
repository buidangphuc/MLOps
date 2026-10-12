from collections import Counter

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from datascience import log
from datascience.constants import paths
from datascience.entity.config_entity import DataTransformationConfig
from datascience.utils.io import atomic_write_json, data_fingerprint, read_jsonl, write_jsonl


def _can_stratify(labels) -> bool:
    counts = Counter(labels)
    return len(counts) >= 2 and min(counts.values()) >= 2


def _group_split(rows, groups, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(rows, groups=groups))
    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]


def _drop_cross_split_dups(train, val, test, text_field):
    """Remove rows whose text already appears in an earlier split (keep in train)."""
    seen = {r[text_field] for r in train}
    val2 = [r for r in val if r[text_field] not in seen]
    seen |= {r[text_field] for r in val2}
    test2 = [r for r in test if r[text_field] not in seen]
    dropped = (len(val) - len(val2)) + (len(test) - len(test2))
    if dropped:
        log.warning(f"leakage guard: dropped {dropped} rows with text duplicated across splits")
    return train, val2, test2


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
