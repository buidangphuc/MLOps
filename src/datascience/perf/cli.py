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
