"""Shared fail-fast validators for pipeline config files.

Any pipeline's ConfigurationManager (classification today, copy-and-edit
pipelines later) reuses these instead of re-copying the checks.
"""


def validate_split_sizes(val_size, test_size, *, where: str = "data") -> None:
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError(
            f"{where}.val_size/test_size must each be in (0,1) and sum < 1"
        )


def validate_gate(gate) -> None:
    """Gate block shape: on_fail in {error,warn}; every rule has metric+min."""
    if gate.on_fail not in ("error", "warn"):
        raise ValueError("gate.on_fail must be 'error' or 'warn'")
    for rule in gate.rules:
        if "metric" not in rule or "min" not in rule:
            raise ValueError(f"each gate rule needs 'metric' and 'min': {dict(rule)}")
