import json
import os
import shutil

from src.datascience import log


class LocalRegistry:
    """Versioned model catalog on local disk — no external service.

    Layout: <root>/<name>/v1, v2, ... each holding a copy of the model dir plus
    a registry_meta.json; <root>/<name>/latest.json points at the newest version.
    """

    def __init__(self, root: str, name: str):
        self.dir = os.path.join(root, name)
        self.name = name

    def register(self, model_dir: str, metrics: dict, metadata: dict) -> str:
        os.makedirs(self.dir, exist_ok=True)
        existing = [
            int(d[1:])
            for d in os.listdir(self.dir)
            if d.startswith("v") and d[1:].isdigit()
        ]
        version = (max(existing) + 1) if existing else 1
        dest = os.path.join(self.dir, f"v{version}")

        shutil.copytree(model_dir, dest)
        with open(os.path.join(dest, "registry_meta.json"), "w") as f:
            json.dump(
                {"name": self.name, "version": version, "metrics": metrics, "metadata": metadata},
                f,
                indent=2,
                ensure_ascii=False,
            )
        with open(os.path.join(self.dir, "latest.json"), "w") as f:
            json.dump({"version": version, "path": dest, "metrics": metrics}, f, indent=2)

        log.info(f"[local registry] registered {self.name} v{version} -> {dest}")
        return str(version)
