"""Unified CLI. Server-friendly: each stage runs independently, `--gpu` selects
the device, exit codes propagate (a failing gate exits non-zero).

  python main.py [--model <id>] run [--only train | --from evaluate] [--gpu 0]
  python main.py <stage> [--variant <name>]      # the active pack's stages, e.g. train / eval
  python main.py variants                        # list the named configs in experiments.yaml
  python main.py tune                            # run+rank every variant in experiments.yaml
  python main.py promote <variant>               # version the chosen variant into the registry
  python main.py infer --text "..."
  python main.py infer --input-file in.jsonl --output-file out.jsonl --batch-size 32
"""

import argparse
import json
import os
import sys

_ALIASES = {"eval": "evaluate"}


def _model_arg(argv):
    it = iter(argv)
    for a in it:
        if a == "--model":
            return next(it, None)
        if a.startswith("--model="):
            return a.split("=", 1)[1]
        if a == "--gpu":
            next(it, None)
        elif not a.startswith("-"):
            return None
    return None


def build_parser(stage_names=(), commands=None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="datascience", description="multi-model MLOps pipeline (one model pack per run)")
    p.add_argument("--gpu", help="GPU id(s) -> sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--model", dest="model_id",
                   help="model id = config/<id>/ + artifacts/<id>/ (default: DS_MODEL env, else text_classification)")
    sub = p.add_subparsers(dest="command", required=True)

    variant_help = "run a named config from experiments.yaml instead of params.yaml"

    runp = sub.add_parser("run", help="run the full pipeline (or a slice)")
    runp.add_argument("--only", help="run just this stage")
    runp.add_argument("--from", dest="frm", help="start from this stage")
    runp.add_argument("--to", help="stop after this stage")
    runp.add_argument("--variant", help=variant_help)

    for stage in [*stage_names, *[a for a, n in _ALIASES.items() if n in stage_names]]:
        sp = sub.add_parser(stage, help=f"run the {_ALIASES.get(stage, stage)} stage")
        sp.add_argument("--variant", help=variant_help)
    for name, (help_text, _) in (commands or {}).items():
        sub.add_parser(name, help=help_text)

    sub.add_parser("variants", help="list the named configs in experiments.yaml")
    sub.add_parser("tune", help="run+rank the variants in experiments.yaml")
    pr = sub.add_parser("promote", help="version a tuning variant into the registry")
    pr.add_argument("variant", help="variant name from experiments.yaml (see `tune` ranking)")

    ip = sub.add_parser("infer", help="predict on text or a JSONL file")
    ip.add_argument("--text")
    ip.add_argument("--input-file")
    ip.add_argument("--output-file")
    ip.add_argument("--batch-size", type=int, default=32)
    ip.add_argument("--model-path", help="a specific model dir (default: the trained model)")

    sub.add_parser("serve-triton", help="run the packaged Triton repo locally (docker compose)")
    ps = sub.add_parser("promote-stage", help="point staging/production at a registered version")
    ps.add_argument("name", help="registry model name (registry.name in params.yaml)")
    ps.add_argument("version", help="registered version, e.g. v3 (or 3)")
    ps.add_argument("--to", dest="stage", choices=["staging", "production"], required=True)
    rb = sub.add_parser("rollback", help="restore the previous version for a stage")
    rb.add_argument("name")
    rb.add_argument("--stage", choices=["staging", "production"], default="production")
    sp = sub.add_parser("stage-path", help="print the model dir a stage points at")
    sp.add_argument("name")
    sp.add_argument("stage", choices=["staging", "production"])

    from datascience.perf.cli import add_parser as add_perf_parser

    add_perf_parser(sub)
    return p


def main(argv=None):
    from datascience import pack
    from datascience.constants import paths

    argv = sys.argv[1:] if argv is None else argv
    model_id = _model_arg(argv) or os.getenv("DS_MODEL") or paths.DEFAULT_MODEL
    os.environ["DS_MODEL"] = model_id
    paths.use(model_id)
    the_pack = pack.load()
    stage_names = [n for n, _ in the_pack.STAGES]
    commands = getattr(the_pack, "COMMANDS", {})
    args = build_parser(stage_names, commands).parse_args(argv)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = args.command
    stage = _ALIASES.get(cmd, cmd)

    if cmd == "variants":
        from datascience.tune.variants import list_variants

        for name in list_variants():
            print(name)
    elif cmd == "run" or stage in stage_names:
        from datascience.pipeline import stages

        # --variant resolves experiments.yaml onto params.yaml through the SAME code `tune`
        # uses. Stage commands only: `promote` takes a positional also called `variant`.
        cfg_path = None
        if getattr(args, "variant", None):
            from datascience.tune.variants import resolve_variant

            cfg_path = resolve_variant(args.variant)
        if cmd == "run":
            stages.run_all(only=args.only, frm=args.frm, to=args.to, params_filepath=cfg_path)
        else:
            stages.run_stage(stage, cfg_path)
    elif cmd in commands:
        from datascience.config.configuration import ConfigurationManager

        commands[cmd][1](ConfigurationManager(), paths.active())
    elif cmd == "tune":
        from datascience.tune.loop import run_tuning

        run_tuning()
    elif cmd == "promote":
        from datascience.tune.promote import promote

        promote(args.variant)
    elif cmd == "infer":
        predictor = the_pack.load_predictor(model_dir=args.model_path)
        if args.text:
            print(json.dumps(predictor.predict(args.text), ensure_ascii=False))
        elif args.input_file:
            predictor.predict_file(args.input_file, args.output_file or "predictions.jsonl", args.batch_size)
        else:
            raise SystemExit("infer needs --text or --input-file")
    elif cmd == "serve-triton":
        from datascience.serve.triton import serve_triton

        serve_triton()
    elif cmd == "perf":
        from datascience.perf.cli import dispatch

        dispatch(args)
    elif cmd == "promote-stage":
        from datascience.registry.stages import assign_stage

        assign_stage(args.name, args.version, args.stage)
    elif cmd == "rollback":
        from datascience.registry.stages import rollback

        rollback(args.name, args.stage)
    elif cmd == "stage-path":
        from datascience.registry.stages import stage_path

        print(stage_path(args.name, args.stage))


if __name__ == "__main__":
    main()
