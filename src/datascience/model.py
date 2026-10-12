"""HF model + tokenizer factory.

One place that builds/loads the causal LM, shared by trainer / eval / infer so the
architecture can never drift. To use a different model, change `model.base_model` in
params.yaml — the code only assumes a HF AutoModelForCausalLM. The saved directory is always a
plain causal-LM directory (adapters merged), loadable without peft.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from pathlib import Path


def load_label_map(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    label2id = obj["label2id"]
    id2label = {int(k): v for k, v in obj["id2label"].items()}
    return label2id, id2label


def bf16_ok() -> bool:
    """bf16 training: the train TARGET must support it (hardware.yaml) and this box must have CUDA."""
    import torch

    from datascience.perf.hw import supports

    return torch.cuda.is_available() and supports("bf16", "train")


def _train_dtype():
    import torch

    return torch.bfloat16 if bf16_ok() else torch.float32


def build_for_training(cfg):
    """(model, tokenizer) for `cfg.finetune.method`. full trains every weight; lora trains
    adapters only; qlora does the same over a 4-bit nf4 base (needs the `quant` extra, GPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ft, dtype = cfg.finetune, _train_dtype()
    if ft.method == "qlora":
        import torch
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, quantization_config=quant, device_map={"": torch.cuda.current_device()}
        )
        model = prepare_model_for_kbit_training(model)
    else:
        import torch

        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype if ft.method == "lora" else torch.float32)

    if ft.method in ("lora", "qlora"):
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=ft.lora_r,
            lora_alpha=ft.lora_alpha,
            lora_dropout=ft.lora_dropout,
            target_modules=ft.lora_target_modules,   # None -> PEFT's default for the architecture
        ))
        model.print_trainable_parameters()
    return model, tokenizer


def merge_adapters(model, cfg):
    """A plain causal LM from the trained model. lora merges in place; qlora reloads the base in
    the train dtype first (merging into 4-bit weights is lossy); full/qat pass through."""
    method = cfg.finetune.method
    if method == "lora":
        return model.merge_and_unload()
    if method == "qlora":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as adapters:
            model.save_pretrained(adapters)
            base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=_train_dtype())
            return PeftModel.from_pretrained(base, adapters).merge_and_unload()
    return model


def load_trained(model_dir, device: str = "cpu", scheme=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if scheme is not None:
        raise NotImplementedError("quantization schemes are applied by perf/quant (group 2)")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    return model, tokenizer


def save_pretrained_atomic(model, tokenizer, model_dir) -> None:
    """Save model+tokenizer to a temp dir then swap in — never a half-written model dir."""
    model_dir = Path(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=model_dir.parent, prefix=".model_tmp_"))
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    os.replace(tmp, model_dir)
