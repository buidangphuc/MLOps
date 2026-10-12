"""Triton serving-config tuning — SEPARATE from model tuning (tune/).

This tunes serving PARAMS (instance_group, dynamic_batching, ...) against REAL traffic, so
it runs AFTER the first deploy: capture real requests from the live endpoint, then replay
them here so the optimal config reflects the real input + load — not synthetic inputs.
`bench` measures one config; this sweeps and picks (NVIDIA Model Analyzer). Run on a
prod-class GPU host (matches serving.sagemaker.instance_type).
"""

import json
import subprocess

from pathlib import Path

import yaml

from datascience import log
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.utils.io import read_jsonl


def _ma_config(a, repo, perf_input) -> dict:
    """Build the NVIDIA Model Analyzer config: auto-search instance × batching for the
    ensemble under the p99 latency budget, driven by the real-traffic input file."""
    return {
        "model_repository": str(repo),
        "output_model_repository_path": str(paths.ARTIFACTS / "analyze_out"),
        "override_output_model_repository": True,
        "triton_launch_mode": "docker",
        "run_config_search_max_instance_count": a.max_instance_count,
        "run_config_search_max_model_batch_size": a.max_batch_size,
        "profile_models": {
            "text_classifier": {                 # the ensemble — driven by REAL text traffic
                "objectives": ["perf_throughput"],
                "constraints": {"perf_latency_p99": {"max": a.latency_budget_ms}},
                "perf_analyzer_flags": {"input-data": str(perf_input)},
            }
        },
    }


def analyze():
    cm = ConfigurationManager()
    a = cm.serving().analyze
    repo = paths.TRITON_REPO
    if not repo.exists():
        raise SystemExit("no model_repository — run `python main.py package --target triton` first")
    if not a.get("traffic_file"):
        raise SystemExit(
            "set serving.analyze.traffic_file — real requests captured from the first deploy "
            '({"text": ...} per line). Tune AFTER deploying, with real traffic, not synthetic.'
        )

    # replay the captured real requests through the ensemble (TEXT in) as perf_analyzer input
    texts = [r["text"] for r in read_jsonl(Path(a.traffic_file), required_fields=["text"])]
    perf_input = paths.ARTIFACTS / "traffic_perf.json"
    perf_input.write_text(json.dumps({"data": [{"TEXT": [t]} for t in texts]}))
    log.info(f"replaying {len(texts)} real requests from {a.traffic_file}")

    ma_config = _ma_config(a, repo, perf_input)
    cfg = paths.ARTIFACTS / "model_analyzer.yaml"
    cfg.write_text(yaml.safe_dump(ma_config, sort_keys=False))
    log.info(f"Model Analyzer config -> {cfg}")

    subprocess.run(
        ["model-analyzer", "profile", "-f", str(cfg),
         "--checkpoint-directory", str(paths.ARTIFACTS / "analyze_ckpt")],
        check=True,
    )
    log.info(
        f"sweep done — top configs ranked under {paths.ARTIFACTS / 'analyze_out'}. "
        f"Bake the winning config.pbtxt into {repo}, then `bench` to confirm and re-`deploy`."
    )
