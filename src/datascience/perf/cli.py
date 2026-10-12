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
    tn.add_argument("workload", choices=["infer", "train", "kernel"])
    tn.add_argument("kernel", nargs="?", help="kernel module under perf/kernels (perf tune kernel layernorm)")
    tn.add_argument("--entry", help="module:fn returning the model/trainer target (infer/train)")
    tn.add_argument("--role", choices=ROLES, default="train",
                    help="kernel only: which target the result binds to")
    tn.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    rp = ps.add_parser("report", help="quant-vs-quality table from this box's stored infer sweep")
    rp.add_argument("--role", choices=["infer"], default="infer")
    rp.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    pr = ps.add_parser("profile", help="torch.profiler + NVML over N steady-state steps")
    pr.add_argument("workload", choices=["infer", "train"])
    pr.add_argument("--entry", required=True, help="module:fn returning the model/trainer target")
    pr.add_argument("--steps", type=int, default=10)
    pr.add_argument("--top-k", type=int, default=25)
    pr.add_argument("--no-track", action="store_true", help="skip MLflow logging")


def _profile_cfg(workload: str, cfg: dict) -> dict:
    """This box's tuned config if there is one, else the grid's first values."""
    from datascience.perf.hw import HardwareMismatch
    from datascience.perf.store import load_tuned

    try:
        chosen = load_tuned(workload, workload)
    except (FileNotFoundError, HardwareMismatch):
        chosen = {k: v[0] for k, v in cfg["grid"].items()}
    if workload == "infer":
        chosen = {**chosen, "length": cfg["lengths"][-1]}   # profile the heaviest bucket
    return chosen


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


def _tune_kernels(args) -> None:
    from datascience.perf.hw import assert_role
    from datascience.perf.runner import sweep
    from datascience.perf.workloads.kernel import KernelWorkload

    if not args.kernel:
        raise SystemExit("perf tune kernel needs a kernel name, e.g. `perf tune kernel layernorm`")
    cfg = _cfg("kernel")
    fp = assert_role(args.role)
    kcfg = cfg[args.kernel]
    for shape in kcfg["shapes"]:
        wl = KernelWorkload(args.kernel, shape, kcfg["dtype"], role=args.role)
        sweep(wl, cfg["grid"], cfg["objective"], cfg["mode"], wl.constraints(), fp=fp,
              track=not args.no_track)


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
    elif args.perf_cmd == "tune" and args.workload == "kernel":
        _tune_kernels(args)
    elif args.perf_cmd == "tune":
        from datascience.perf.hw import assert_role
        from datascience.perf.runner import sweep

        if not args.entry:
            raise SystemExit(f"perf tune {args.workload} needs --entry module:fn")
        cfg = _cfg(args.workload)
        fp = assert_role(args.workload)          # fail before loading any model on the wrong box
        sweep(_build(args.workload, args.entry, cfg), cfg["grid"], cfg["objective"],
              cfg.get("mode", "max"), cfg.get("constraints"), fp=fp, track=not args.no_track)
    elif args.perf_cmd == "report":
        from datascience.perf.report import build_report

        print(build_report(track=not args.no_track))
    elif args.perf_cmd == "profile":
        from datascience.perf.profile import profile_workload
        from datascience.perf.runner import resolve_entry

        cfg = _cfg(args.workload)
        out, summary = profile_workload(args.workload, resolve_entry(args.entry),
                                        _profile_cfg(args.workload, cfg), args.steps, args.top_k,
                                        track=not args.no_track)
        print(json.dumps(summary, indent=2))
        print((out / "top_kernels.txt").read_text())
