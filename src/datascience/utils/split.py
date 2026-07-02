"""Shared train/val/test split helpers.

Pure, task-agnostic logic used by any pipeline's transformation stage
(classification today; copy-and-edit pipelines reuse these instead of
re-copying them). Trainer/evaluator code stays reference-style — only small
stable utilities live here.
"""

from collections import Counter

from datascience import log


def can_stratify(labels) -> bool:
    """Stratified splitting needs >=2 classes with >=2 members each."""
    counts = Counter(labels)
    return len(counts) >= 2 and min(counts.values()) >= 2


def group_split(rows, groups, test_size, seed):
    """Group-aware split: no group appears on both sides (no leakage)."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(rows, groups=groups))
    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]


def drop_cross_split_dups(train, val, test, text_field):
    """Remove rows whose text already appears in an earlier split (keep in train)."""
    seen = {r[text_field] for r in train}
    val2 = [r for r in val if r[text_field] not in seen]
    seen |= {r[text_field] for r in val2}
    test2 = [r for r in test if r[text_field] not in seen]
    dropped = (len(val) - len(val2)) + (len(test) - len(test2))
    if dropped:
        log.warning(
            f"leakage guard: dropped {dropped} rows with text duplicated across splits"
        )
    return train, val2, test2
