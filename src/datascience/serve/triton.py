"""Run Triton locally via docker-compose, serving the model_repository.

On a GPU/Linux box this is the real serving path; on a non-GPU box Docker runs
the CPU profile (slower, functional). Same repo deploys to SageMaker.
"""

import subprocess

from datascience import log
from datascience.constants import paths


def serve_triton():
    compose = paths.PROJECT_ROOT / "infra" / "triton" / "docker-compose.yml"
    if not compose.exists():
        raise FileNotFoundError(f"{compose} missing — run `package --target triton` first")
    log.info(f"starting Triton via docker compose ({compose}); Ctrl-C to stop")
    subprocess.run(["docker", "compose", "-f", str(compose), "up"], check=True)
