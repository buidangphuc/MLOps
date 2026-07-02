"""Offline tests for the drift-monitor metric (PSI). The full monitor() path needs a model."""

from datascience.metrics import population_stability_index


def test_psi_zero_when_identical():
    d = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert population_stability_index(d, d) < 1e-6


def test_psi_grows_with_shift():
    ref = {"a": 0.5, "b": 0.5}
    small = population_stability_index(ref, {"a": 0.45, "b": 0.55})
    large = population_stability_index(ref, {"a": 0.10, "b": 0.90})
    assert 0 < small < large


def test_psi_handles_missing_label():
    # a label present in reference but absent in current must not blow up (eps-floored)
    assert population_stability_index({"a": 0.5, "b": 0.5}, {"a": 1.0}) > 0
