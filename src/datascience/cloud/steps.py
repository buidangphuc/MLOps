"""What runs INSIDE a SageMaker job: `python main.py --model <id> sm-step <step>`.

Maps SageMaker's container dirs onto the model's artifact tree, so the pack's stages run
unchanged in a job:

  all       (Training Job)    every stage, gate + register              -> /opt/ml/model
  train     (pipeline Train)  stages before `evaluate`                  -> /opt/ml/model
  evaluate  (pipeline Eval)   restore model.tar.gz, `evaluate`          -> evaluation.json + model
  register  (pipeline Reg.)   restore the evaluated model, gate + register

SM_ROOT (default /opt/ml) relocates the tree for local runs and tests.
"""

import json
import os
import shutil
import tarfile
from pathlib import Path

from datascience import log
from datascience.constants import paths

STEPS = ("all", "train", "evaluate", "register")


def _root() -> Path:
    return Path(os.environ.get("SM_ROOT", "/opt/ml"))


def _load_mlflow_secret() -> None:
    arn = os.environ.get("MLFLOW_SECRET_ARN")
    if not arn or os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        return
    import boto3

    secret = json.loads(boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"])
    os.environ["MLFLOW_TRACKING_USERNAME"] = secret["username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secret["password"]


def _export(p, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(p.MODEL_DIR, dest / "model")
    shutil.copytree(p.SPLIT_DIR, dest / "split")
    if p.TRAIN_RUN_FILE.exists():
        shutil.copy2(p.TRAIN_RUN_FILE, dest / "mlflow_run.json")


def _restore(p, src: Path) -> None:
    archives = list(src.glob("*.tar.gz"))
    if archives:
        unpacked = src / "_unpacked"
        with tarfile.open(archives[0]) as tar:
            tar.extractall(unpacked, filter="data")
        src = unpacked
    for d in (p.MODEL_DIR, p.SPLIT_DIR):
        if d.exists():
            shutil.rmtree(d)
    shutil.copytree(src / "model", p.MODEL_DIR)
    shutil.copytree(src / "split", p.SPLIT_DIR)
    if (src / "mlflow_run.json").exists():
        p.TRAIN_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "mlflow_run.json", p.TRAIN_RUN_FILE)


def run(step: str) -> None:
    from datascience.config.configuration import ConfigurationManager
    from datascience.pipeline import stages

    _load_mlflow_secret()
    p = paths.active()
    root = _root()
    names = stages.names()
    before_eval = names[: names.index("evaluate")]

    if step == "all":
        stages.run_all()
        _export(p, root / "model")
    elif step == "train":
        for name in before_eval:
            stages.run_stage(name)
        _export(p, root / "model")
    elif step == "evaluate":
        _restore(p, root / "processing" / "input" / "model")
        stages.run_stage("evaluate", register=False)
        out = root / "processing" / "evaluation"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p.METRICS_FILE, out / "evaluation.json")
        _export(p, root / "processing" / "model")
    elif step == "register":
        _restore(p, root / "processing" / "input" / "model")
        p.METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "processing" / "input" / "evaluation" / "evaluation.json", p.METRICS_FILE)
        result = stages.gate_and_register(ConfigurationManager())
        log.info(f"[sm-step] register -> {result}")
    else:
        raise ValueError(f"unknown sm-step '{step}' (choices: {STEPS})")
