"""Unified CLI. Server-friendly: each stage runs independently, `--gpu` selects
the device, exit codes propagate (a failing gate exits non-zero).

  python main.py run [--only train | --from eval] [--gpu 0]
  python main.py ingest | validate | transform | train | eval | sanity [--gpu 0]
  python main.py tune                  # train+eval+rank the variants in experiments.yaml
  python main.py promote <variant>     # version the chosen variant into the registry
  python main.py infer --text "..."
  python main.py infer --input-file in.jsonl --output-file out.jsonl --batch-size 32
"""

import argparse
import json
import os


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="datascience", description="Text-classification fine-tune pipeline")
    p.add_argument("--gpu", help="GPU id(s) -> sets CUDA_VISIBLE_DEVICES")
    sub = p.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="run the full pipeline (or a slice)")
    runp.add_argument("--only", help="run just this stage")
    runp.add_argument("--from", dest="frm", help="start from this stage")
    runp.add_argument("--to", help="stop after this stage")

    for stage in ("ingest", "validate", "transform", "train", "eval", "sanity"):
        sub.add_parser(stage, help=f"run the {stage} stage")

    # tuning loop (Phase 2 / Layer 2)
    sub.add_parser("tune", help="train+eval+rank the variants in experiments.yaml")
    pr = sub.add_parser("promote", help="version a tuning variant into the registry")
    pr.add_argument("variant", help="variant name from experiments.yaml (see `tune` ranking)")
    sub.add_parser("dapt", help="R4 domain-adaptive pretraining -> adapted backbone (artifacts/dapt)")

    ip = sub.add_parser("infer", help="predict on text or a JSONL file")
    ip.add_argument("--text")
    ip.add_argument("--input-file")
    ip.add_argument("--output-file")
    ip.add_argument("--batch-size", type=int, default=32)

    # serving / inference (Phase 2)
    sub.add_parser("optimize", help="export trained model -> ONNX (+ optional quantize)")
    pk = sub.add_parser("package", help="build a deployable artifact")
    pk.add_argument("--target", choices=["bundle", "triton"], default="bundle")
    sv = sub.add_parser("serve", help="run an inference endpoint")
    sv.add_argument("--backend", choices=["local", "triton"], default="local")
    sv.add_argument("--port", type=int, default=8080)
    dp = sub.add_parser("deploy", help="deploy to a managed endpoint")
    dp.add_argument("--target", choices=["sagemaker-triton"], default="sagemaker-triton")
    bn = sub.add_parser("bench", help="benchmark inference latency/throughput (one config)")
    bn.add_argument("--target", choices=["local", "triton"], default="local")
    sub.add_parser("analyze", help="sweep Triton configs with Model Analyzer (prod-class GPU host)")
    sub.add_parser("monitor", help="drift check on captured prod traffic vs the eval reference (warn-only)")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from datascience.pipeline import stages

    cmd = args.command
    if cmd == "run":
        stages.run_all(only=args.only, frm=args.frm, to=args.to)
    elif cmd == "ingest":
        stages.run_ingest()
    elif cmd == "validate":
        stages.run_validate()
    elif cmd == "transform":
        stages.run_transform()
    elif cmd == "train":
        stages.run_train()
    elif cmd == "eval":
        stages.run_evaluate()
    elif cmd == "sanity":
        stages.run_sanity()
    elif cmd == "tune":
        from datascience.tune.loop import run_tuning

        run_tuning()
    elif cmd == "promote":
        from datascience.tune.promote import promote

        promote(args.variant)
    elif cmd == "dapt":
        from datascience.tune.dapt import run_dapt

        run_dapt()
    elif cmd == "infer":
        from datascience.infer.predict import Predictor

        predictor = Predictor()
        if args.text:
            print(json.dumps(predictor.predict(args.text), ensure_ascii=False))
        elif args.input_file:
            predictor.predict_file(
                args.input_file, args.output_file or "predictions.jsonl", args.batch_size
            )
        else:
            raise SystemExit("infer needs --text or --input-file")
    elif cmd == "optimize":
        from datascience.optimize import optimize

        optimize()
    elif cmd == "package":
        from datascience.package import package_bundle, package_triton

        (package_triton if args.target == "triton" else package_bundle)()
    elif cmd == "serve":
        if args.backend == "triton":
            from datascience.serve.triton import serve_triton

            serve_triton()
        else:
            from datascience.serve.local import serve

            serve(port=args.port)
    elif cmd == "deploy":
        from datascience.serve.deploy_sagemaker import deploy_sagemaker

        deploy_sagemaker()
    elif cmd == "bench":
        from datascience.bench import bench_local, bench_triton

        (bench_triton if args.target == "triton" else bench_local)()
    elif cmd == "analyze":
        from datascience.serve_analyzer import analyze

        analyze()
    elif cmd == "monitor":
        from datascience.monitor import monitor

        monitor()


if __name__ == "__main__":
    main()
