from collections import Counter

from datascience import log
from datascience.constants import paths
from datascience.entity.config_entity import DataValidationConfig
from datascience.utils.io import atomic_write_text, read_jsonl


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self) -> bool:
        """Gate: data must have the text/label fields, non-empty text, non-null label.
        Writes STATUS_FILE so downstream can refuse to proceed on failure."""
        rows = read_jsonl(paths.RAW_DATA)
        tf, lf = self.config.text_field, self.config.label_field

        missing = sum(1 for r in rows if tf not in r or lf not in r)
        empty_text = sum(1 for r in rows if not str(r.get(tf, "")).strip())
        null_label = sum(1 for r in rows if r.get(lf) in (None, ""))
        status = missing == 0 and empty_text == 0 and null_label == 0

        atomic_write_text(paths.STATUS_FILE, f"Validation status: {status}")

        if status:
            counts = Counter(r[lf] for r in rows)
            log.info(f"validation passed | {len(rows)} rows | class counts: {dict(counts)}")
        else:
            log.warning(
                f"validation FAILED | missing_field={missing} empty_text={empty_text} null_label={null_label}"
            )
        return status
