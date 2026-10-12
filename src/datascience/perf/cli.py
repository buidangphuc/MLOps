"""`datascience perf …` subcommands (wired from cli.py)."""

import json

from datascience.perf.hw import ROLES


def add_parser(sub) -> None:
    pf = sub.add_parser("perf", help="GPU perf platform: fingerprint / show / tune / profile")
    ps = pf.add_subparsers(dest="perf_cmd", required=True)
    ps.add_parser("fingerprint", help="print this box + whether it matches the train/infer target")
    sh = ps.add_parser("show", help="list stored tuned results")
    sh.add_argument("--role", choices=ROLES)
    sh.add_argument("--hw", help="filter by hw_id")
    tn = ps.add_parser("tune", help="sweep a workload on this box, store the best (hardware-bound)")
    tn.add_argument("workload", choices=["infer", "train"])
    tn.add_argument("--entry", required=True, help="module:fn returning the model/trainer target")
    tn.add_argument("--no-track", action="store_true", help="skip MLflow logging")


def _cfg(workload: str) -> dict:
    import yaml

    from datascience.constants import paths

    return yaml.safe_load(paths.PERF_FILE.read_text())[workload]


def _build(workload: str, entry: str, cfg: dict):
    from datascience.perf.runner import resolve_entry

    target = resolve_entry(entry)
    if workload == "infer":
        from datascience.perf.workloads.infer import InferWorkload

        return InferWorkload(target, cfg["lengths"], iters=cfg["iters"], warmup=cfg["warmup"])
    if workload == "train":
        from datascience.perf.workloads.train import TrainWorkload

        return TrainWorkload(target, cfg["effective_batch"], cfg["steps"], cfg["warmup_steps"])
    raise ValueError(workload)


def dispatch(args) -> None:
    if args.perf_cmd == "fingerprint":
        from datascience.perf.hw import fingerprint, mismatches

        fp = fingerprint()
        print(json.dumps(fp, indent=2))
        for role in ROLES:
            bad = mismatches(role, fp)
            print(f"{role:>5}: " + ("MATCH" if not bad else "no match — " + "; ".join(bad)))
    elif args.perf_cmd == "show":
        from datascience.perf.store import list_results

        for r in list_results(args.role, args.hw):
            print(f"{r['role']:>5}  {r['hw_id']}  {r['workload']:<12} trials={r['trials']:<3} "
                  f"chosen={json.dumps(r['chosen'])}")
    elif args.perf_cmd == "tune":
        from datascience.perf.hw import assert_role
        from datascience.perf.runner import sweep

        cfg = _cfg(args.workload)
        fp = assert_role(args.workload)          # fail before loading any model on the wrong box
        sweep(_build(args.workload, args.entry, cfg), cfg["grid"], cfg["objective"],
              cfg.get("mode", "max"), cfg.get("constraints"), fp=fp, track=not args.no_track)
