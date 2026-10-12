"""Kernel autotune targets — copy-and-edit reference modules, one file per kernel.

A kernel module exposes:
    space(shape, grid) -> list[dict]        # launch configs worth trying for this shape
    make_inputs(shape, dtype) -> tuple      # on cuda
    run(cfg, inputs) -> out                 # launch the kernel with one config
    reference(inputs) -> out                # the torch op it must match
    bytes_moved(shape, dtype) -> int        # for effective GB/s
Tune with: datascience perf tune kernel <module> [--role train|infer]
"""
