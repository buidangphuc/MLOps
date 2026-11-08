import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from datascience import log, pack
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.serve.release import resolve_version, tar_name

_TRITON_ABI = "cpython-310"


def _assert_py310():
    if not sys.version.startswith("3.10"):
        raise RuntimeError(
            f"package must run on Python 3.10 for Triton-container ABI ({_TRITON_ABI}); "
            f"got {sys.version.split()[0]}. Use the tri310 env."
        )


def _assert_abi(deps: Path) -> None:
    bad = [p.name for p in deps.rglob("*.so") if "cpython-3" in p.name and _TRITON_ABI not in p.name]
    if bad:
        raise RuntimeError(f"bundled .so ABI != {_TRITON_ABI} (won't load in Triton 23.07): {bad[:3]}")


def _deps_dir(t) -> Path:
    if not t.pip_no_deps:
        return paths.ARTIFACTS / "triton_deps"
    tag = hashlib.sha256(" ".join(t.pip_no_deps).encode()).hexdigest()[:8]
    return paths.ARTIFACTS / f"triton_deps_{tag}"


def _bundle_deps(t) -> Path:
    shared = _deps_dir(t)
    if shared.exists():
        log.info(f"reusing bundled deps at {shared}")
        return shared
    shared.mkdir(parents=True, exist_ok=True)
    req = paths.PROJECT_ROOT / "requirements.txt"
    log.info(f"bundling deps -> {shared} (from {req.name}, Python 3.10)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req), *t.pip, "--target", str(shared)])
    if t.pip_no_deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-deps",
                               *t.pip_no_deps, "--target", str(shared)])
    _assert_abi(shared)
    return shared


def _materialize(files: dict, root: Path) -> None:
    for arc, src in files.items():
        dest = root / arc
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, bytes):
            dest.write_bytes(src)
        elif isinstance(src, str):
            dest.write_text(src)
        elif Path(src).is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


def _triton_python(t, key: str, bundle: bool) -> Path:
    root = paths.TRITON_REPO / key
    _materialize(t.files, root)
    if bundle:
        shutil.copytree(_bundle_deps(t), root / "1" / "deps")
    tar_path = paths.MME_TARS / f"{key}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(root, arcname=key)
    return root


def _pytorch_dlc(t, key: str, bundle: bool) -> Path:
    root = paths.TRITON_REPO / key
    _materialize(t.files, root)
    names = sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file())
    tar_path = paths.MME_TARS / f"{key}.tar.gz"
    with open(tar_path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for name in names:
            data = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            info.uid, info.gid, info.uname, info.gname = 0, 0, "", ""
            tar.addfile(info, io.BytesIO(data))
    return root


_WRITERS = {"triton-python": _triton_python, "pytorch-dlc": _pytorch_dlc}


def package(version=None) -> dict:
    cm = ConfigurationManager()
    p = paths.active()
    version = resolve_version(version)
    bundle = bool(cm.serving().get("triton", {}).get("bundle_deps", True))
    targets = pack.load().targets(cm, p, version)

    names = [t.name for t in targets]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate target names: {names}")
    if bundle and any(t.kind == "triton-python" for t in targets):
        _assert_py310()

    if paths.TRITON_REPO.exists():
        shutil.rmtree(paths.TRITON_REPO)
    paths.MME_TARS.mkdir(parents=True, exist_ok=True)

    smoke = getattr(pack.load(), "smoke", None)
    built = {}
    for t in targets:
        key = tar_name(t.name, version)[: -len(".tar.gz")]
        target_dir = _WRITERS[t.kind](t, key, bundle)
        if smoke:
            smoke(target_dir, t, cm)
        built[t.name] = {"kind": t.kind, **t.meta}
        log.info(f"packaged {key} -> {key}.tar.gz ({t.kind})")

    run_id = json.loads(paths.TRAIN_RUN_FILE.read_text())["run_id"] if paths.TRAIN_RUN_FILE.exists() else None
    (paths.MME_TARS / f"build-{version}.json").write_text(json.dumps(
        {"model_id": p.MODEL_ID, "thin": not bundle, "mlflow_run_id": run_id, "targets": built}, indent=2))
    log.info(f"{len(targets)} target(s) -> {paths.MME_TARS}. Publish with: python main.py release --version {version}")
    return built
