"""Quality gate: read metrics and decide whether a model may be promoted.

Task-agnostic — it only reads metrics by NAME from metrics.json and compares to
thresholds. `on_fail=error` makes a failing gate exit non-zero (CI can block on it).
"""

from src.datascience import log
from src.datascience.entity.config_entity import GateConfig


class GateFailed(Exception):
    pass


class QualityGate:

    def __init__(self, config: GateConfig):
        self.config = config

    def check(self, metrics: dict) -> bool:
        """Returns True if the model passes (or gate disabled). Raises GateFailed
        when a rule fails and on_fail=error; warns and returns False otherwise."""
        cfg = self.config
        if not cfg.enabled:
            log.info("quality gate disabled — skipping")
            return True

        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        failures = []
        for rule in cfg.rules:
            name, minimum = rule["metric"], rule["min"]
            if name not in scalar:
                raise ValueError(
                    f"gate rule metric '{name}' not found in metrics.json (have {sorted(scalar)})"
                )
            value = scalar[name]
            ok = value >= minimum
            log.info(f"gate: {name}={value:.4f} >= {minimum} -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{name}={value:.4f} < {minimum}")

        if failures:
            message = "quality gate FAILED: " + "; ".join(failures)
            if cfg.on_fail == "error":
                raise GateFailed(message)
            log.warning(message + " (on_fail=warn -> not promoting)")
            return False

        log.info("quality gate PASSED")
        return True
