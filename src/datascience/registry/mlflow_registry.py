import json
import os

from datascience import log

ARTIFACT_PATH = "model"


class MLflowRegistry:
    def __init__(self, tracking_uri: str, name: str, alias: str, experiment: str = None):
        self.tracking_uri = tracking_uri
        self.name = name
        self.alias = alias
        self.experiment = experiment

    def _client(self):
        from mlflow import MlflowClient

        return MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)

    def _new_run(self, c) -> str:
        exp = c.get_experiment_by_name(self.experiment)
        exp_id = exp.experiment_id if exp else c.create_experiment(self.experiment)
        return c.create_run(exp_id, tags={"mlflow.runName": f"register-{self.name}"}).info.run_id

    def push(self, model_dir, metrics: dict, metadata: dict) -> str:
        from mlflow.exceptions import MlflowException

        c = self._client()
        own_run = not metadata.get("run_id")
        run_id = self._new_run(c) if own_run else metadata["run_id"]

        c.log_artifacts(run_id, str(model_dir), ARTIFACT_PATH)
        for k, v in metrics.items():
            c.log_metric(run_id, f"test_{k}", v)
        if "config" in metadata:
            c.log_dict(run_id, metadata["config"], "config.yaml")
        if own_run:
            c.set_terminated(run_id)

        try:
            c.create_registered_model(self.name)
        except MlflowException as e:
            if e.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        tags = {
            k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            for k, v in metadata.items()
            if k != "config"
        }
        source = f"{c.get_run(run_id).info.artifact_uri}/{ARTIFACT_PATH}"
        mv = c.create_model_version(self.name, source, run_id=run_id, tags=tags)
        c.set_registered_model_alias(self.name, self.alias, mv.version)

        uri = f"models:/{self.name}/{mv.version}"
        log.info(f"[mlflow registry] registered {uri} (@{self.alias}, run_id={run_id})")
        return uri

    def resolve_version(self, version: str) -> str:
        version = str(version or self.alias)
        if version.isdigit():
            return version
        return str(self._client().get_model_version_by_alias(self.name, version).version)

    def pull(self, version: str, dest) -> str:
        import mlflow.artifacts

        version = self.resolve_version(version)
        dest = os.path.join(str(dest), self.name, version)
        if os.path.isdir(dest) and os.listdir(dest):
            log.info(f"[mlflow registry] {self.name} v{version} already cached at {dest}")
            return dest

        os.makedirs(dest, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{self.name}/{version}", dst_path=dest,
            tracking_uri=self.tracking_uri,
        )
        log.info(f"[mlflow registry] pulled {self.name} v{version} -> {dest}")
        return dest


def from_config(cfg) -> MLflowRegistry:
    return MLflowRegistry(tracking_uri=cfg.tracking_uri, name=cfg.name, alias=cfg.alias,
                          experiment=cfg.experiment_name)
