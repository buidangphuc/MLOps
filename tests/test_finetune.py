"""Causal-LM SFT pipeline: config, prompt masking, label scoring, fine-tune methods, eval, perf hook.

Offline: a tiny Qwen2 built from a config plus a word-level tokenizer, no downloads.
"""

import dataclasses
import json

import numpy as np
import pytest
import torch
import yaml

from datascience.components.model_evaluation import ModelEvaluation
from datascience.components.model_trainer import ModelTrainer, training_kwargs
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.infer.scoring import IGNORE, build_prompt, encode_example, pad_batch, score_labels
from datascience.model import bf16_ok, build_for_training, load_trained, merge_adapters, save_pretrained_atomic
from datascience.perf import store
from datascience.perf.hw import HardwareMismatch
from datascience.perf.runner import Unsupported

LABELS = ["Business", "Sci", "Sports", "World"]
WORDS = ["<pad>", "<eos>", "<unk>", "You", "classify", "news", "Topics", "Article", "Topic", ":", ".", ",",
         "stocks", "rally", "team", "wins", "match", "the", "of", *LABELS]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """A saved tiny Qwen2 + tokenizer, loadable by build_for_training like a hub model."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, eos_token_id=1, pad_token_id=0)
    path = tmp_path_factory.mktemp("tiny")
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.fixture(autouse=True)
def fast_tokenizer(monkeypatch):
    """config.json says qwen2, so AutoTokenizer would build the real Qwen2 tokenizer and ignore ours."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", PreTrainedTokenizerFast.from_pretrained)


@pytest.fixture()
def tiny(tiny_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM.from_pretrained(tiny_dir), AutoTokenizer.from_pretrained(tiny_dir)


@pytest.fixture()
def no_dirs(monkeypatch):
    monkeypatch.setattr(paths, "ensure_dirs", lambda: None)   # don't create the real artifact tree


@pytest.fixture()
def cm(no_dirs):
    return ConfigurationManager()


def _trainer_cfg(cm, base_model, method="full", **over):
    cfg = cm.trainer()
    return dataclasses.replace(cfg, base_model=str(base_model),
                               finetune=dataclasses.replace(cfg.finetune, method=method), **over)


# --- config ---------------------------------------------------------------------------

def test_shipped_params_loads(cm):
    assert cm.ingestion().dataset_name == "ag_news"
    assert cm.validation().label_field == "label"
    assert cm.transformation().val_size > 0
    t = cm.trainer()
    assert t.finetune.method == "full" and not t.apply_tuned
    ev = cm.evaluation()
    assert ev.objective in ev.compute
    assert cm.gate().rules and cm.registry().name


def test_unknown_finetune_method_rejected(no_dirs, tmp_path):
    params = yaml.safe_load(paths.PARAMS_FILE.read_text())
    params["finetune"]["method"] = "dora"
    f = tmp_path / "params.yaml"
    f.write_text(yaml.safe_dump(params))
    with pytest.raises(ValueError, match=r"full.*lora.*qlora.*qat"):
        ConfigurationManager(params_filepath=f)


# --- prompt + example encoding --------------------------------------------------------

def test_prompt_tokens_are_ignored_by_the_loss(tiny, cm):
    _, tok = tiny
    task = cm.task()
    ex = encode_example(tok, task, "stocks rally", "Business", LABELS)
    n_prompt = len(tok(build_prompt(tok, task, "stocks rally", LABELS), add_special_tokens=False)["input_ids"])
    ids, labels = ex["input_ids"], ex["labels"]
    assert all(x == IGNORE for x in labels[:n_prompt])
    assert labels[n_prompt:] == ids[n_prompt:]
    assert ids[n_prompt:] == tok("Business", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    short = encode_example(tok, task, "match", "Sci", LABELS)
    longer = encode_example(tok, task, "stocks rally the team wins the match", "Business", LABELS)
    batch = pad_batch([longer, short], tok.pad_token_id)
    pad = batch["attention_mask"] == 0
    assert pad.any() and (batch["labels"][pad] == IGNORE).all()


def test_text_is_trimmed_to_max_length(tiny, cm):
    _, tok = tiny
    ex = encode_example(tok, cm.task(), "stocks rally " * 50, "Sports", LABELS, max_length=40)
    assert len(ex["input_ids"]) <= 40 and ex["labels"][-1] == tok.eos_token_id


# --- scoring --------------------------------------------------------------------------

class Rigged(torch.nn.Module):
    """Always predicts `favourite` as the next token."""

    def __init__(self, favourite, vocab=64):
        super().__init__()
        self.anchor, self.favourite, self.vocab = torch.nn.Parameter(torch.zeros(1)), favourite, vocab

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[..., self.favourite] = 10.0
        return type("Out", (), {"logits": logits})


def test_highest_likelihood_label_wins(tiny, cm):
    _, tok = tiny
    b = tok("Sports", add_special_tokens=False)["input_ids"][0]
    scores = score_labels(Rigged(b), tok, cm.task(), ["stocks rally", "team wins", "match"], LABELS, batch_size=4)
    assert scores.shape == (3, len(LABELS))
    assert (scores.argmax(axis=1) == LABELS.index("Sports")).all()


def test_scoring_a_real_model_gives_one_finite_score_per_label(tiny, cm):
    model, tok = tiny
    scores = score_labels(model, tok, cm.task(), ["stocks rally", "team wins"], LABELS, max_length=64)
    assert scores.shape == (2, 4) and np.isfinite(scores).all() and (scores <= 0).all()


# --- fine-tune methods ----------------------------------------------------------------

def test_lora_trains_only_adapters_and_saves_a_plain_model(tiny_dir, cm, tmp_path):
    cfg = _trainer_cfg(cm, tiny_dir, "lora")
    model, tok = build_for_training(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert len(trainable) < len(list(model.named_parameters()))

    ex = encode_example(tok, cfg.task, "stocks rally", "Business", LABELS)
    model(**pad_batch([ex], tok.pad_token_id)).loss.backward()
    assert all((p.grad is not None) == ("lora_" in n) for n, p in model.named_parameters())

    out = tmp_path / "model"
    save_pretrained_atomic(merge_adapters(model, cfg), tok, out)
    from transformers import AutoModelForCausalLM

    plain = AutoModelForCausalLM.from_pretrained(out)   # no peft needed to load it
    assert not any("lora" in n for n, _ in plain.named_parameters())
    assert load_trained(out, "cpu")[0].config.model_type == "qwen2"


def test_full_finetune_trains_every_weight_and_pads_with_eos(tiny_dir, cm):
    model, tok = build_for_training(_trainer_cfg(cm, tiny_dir, "full"))
    assert all(p.requires_grad for p in model.parameters())
    assert tok.pad_token_id is not None


def test_qat_scheme_the_serving_gpu_cannot_run_stops_before_any_step(tmp_artifacts, tiny_dir, cm, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a training step ran")

    monkeypatch.setattr(ModelTrainer, "build_trainer", boom)
    cfg = _trainer_cfg(cm, tiny_dir, "qat")
    cfg = dataclasses.replace(cfg, finetune=dataclasses.replace(cfg.finetune, qat_scheme="int4-wo"))
    with pytest.raises(Unsupported, match=r"int4-wo.*T4.*7\.5"):
        ModelTrainer(cfg).train()


def test_qat_scheme_must_be_qat_capable(tmp_artifacts, tiny_dir, cm, tmp_path):
    cfg = _trainer_cfg(cm, tiny_dir, "qat")
    cfg = dataclasses.replace(cfg, finetune=dataclasses.replace(cfg.finetune, qat_scheme="nf4"))
    with pytest.raises(ValueError, match=r"int8-dyn.*int4-wo"):
        ModelTrainer(cfg).train()

    params = yaml.safe_load(paths.PARAMS_FILE.read_text())
    params["finetune"].update(method="qat", qat={"scheme": "nf4"})
    f = tmp_path / "params.yaml"
    f.write_text(yaml.safe_dump(params))
    with pytest.raises(ValueError, match=r"int8-dyn.*int4-wo"):
        ConfigurationManager(params_filepath=f)


def test_load_trained_reuses_the_recorded_scheme(tiny_dir, tmp_path, monkeypatch):
    import shutil

    from datascience.perf import quant

    model_dir = tmp_path / "qat_model"
    shutil.copytree(tiny_dir, model_dir)
    seen = []
    monkeypatch.setattr(quant, "apply", lambda m, scheme: seen.append(scheme) or m)

    load_trained(model_dir, "cpu")
    assert seen == []                                    # nothing recorded -> nothing applied
    (model_dir / "quant_scheme.json").write_text(json.dumps({"scheme": "int8-dyn"}))
    load_trained(model_dir, "cpu")
    load_trained(model_dir, "cpu", scheme="bf16")        # an explicit scheme wins
    assert seen == ["int8-dyn", "bf16"]


def test_load_trained_applies_a_scheme(tiny_dir):
    from datascience.perf.runner import Unsupported

    assert next(load_trained(tiny_dir, "cpu", scheme="bf16")[0].parameters()).dtype == torch.bfloat16
    with pytest.raises(Unsupported, match="unknown scheme"):
        load_trained(tiny_dir, "cpu", scheme="int3")


def test_trainer_runs_and_saves_a_plain_model(tmp_artifacts, tiny_dir, cm):
    rows = [{"text": x, "label": lb} for lb in LABELS[:2] for x in ("stocks rally", "team wins", "match") * 2]
    for f in (paths.TRAIN_FILE, paths.VAL_FILE):
        f.write_text("\n".join(json.dumps(r) for r in rows))
    paths.LABEL_MAP.write_text(json.dumps({"label2id": {"Business": 0, "Sci": 1}, "id2label": {"0": "Business", "1": "Sci"}}))
    cfg = _trainer_cfg(cm, tiny_dir, "lora", epochs=1, tracking_backend="none", max_length=64)
    ModelTrainer(cfg).train()

    from transformers import AutoModelForCausalLM

    assert not any("lora" in n for n, _ in AutoModelForCausalLM.from_pretrained(paths.MODEL_DIR).named_parameters())


# --- evaluation -----------------------------------------------------------------------

def test_metrics_file_keeps_gate_keys_and_ignores_labels_absent_from_train(tmp_artifacts, tiny, cm, monkeypatch):
    model, tok = tiny
    rows = [{"text": t, "label": lb} for lb in LABELS[:3] for t in ("stocks rally", "team wins", "match", "rally")]
    unseen = {"text": "stocks", "label": "World"}   # only in test: not a train label
    for path, part in ((paths.TEST_FILE, rows + [unseen]), (paths.VAL_FILE, rows)):
        path.write_text("\n".join(json.dumps(r) for r in part))
    labels = LABELS[:3]
    paths.LABEL_MAP.write_text(json.dumps({"label2id": {lb: i for i, lb in enumerate(labels)},
                                           "id2label": {str(i): lb for i, lb in enumerate(labels)}}))
    monkeypatch.setattr("datascience.components.model_evaluation.load_trained", lambda *a, **k: (model, tok))
    monkeypatch.setattr("datascience.components.model_evaluation.pick_device", lambda: "cpu")

    ev = cm.evaluation()
    ModelEvaluation(ev)._run()

    result = json.loads(paths.METRICS_FILE.read_text())
    assert all(m in result for m in ev.compute) and ev.objective in result
    assert set(result["per_class"]) == set(labels)
    assert sum(v["support"] for v in result["per_class"].values()) == len(rows)   # unseen row excluded
    assert "calibration" in result and (paths.MODEL_DIR / "temperature.json").exists()


# --- opt-in tuned perf hook -----------------------------------------------------------

def test_tuned_perf_results_are_not_read_when_opt_in_is_off(cm, tiny_dir, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("perf results were read")

    monkeypatch.setattr(store, "load_tuned", boom)
    kwargs = training_kwargs(_trainer_cfg(cm, tiny_dir), 8, [])
    assert kwargs["per_device_train_batch_size"] == 8
    assert "gradient_checkpointing" not in kwargs and "torch_compile" not in kwargs


def test_tuned_perf_config_is_merged_when_opted_in(cm, tiny_dir, monkeypatch):
    tuned = {"precision": "fp32", "per_device_batch": 2, "grad_accum": 4, "gradient_checkpointing": True}
    monkeypatch.setattr(store, "load_tuned", lambda role, workload: tuned)
    kwargs = training_kwargs(_trainer_cfg(cm, tiny_dir, apply_tuned=True), 8, [])
    assert kwargs["per_device_train_batch_size"] == 2 and kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["gradient_checkpointing"] is True


def test_result_tuned_on_another_gpu_stops_training(tmp_artifacts, cm, tiny_dir):
    store.save("train", "train", {"hw_id": "a100-1x-deadbeef"}, {"chosen": {"precision": "bf16"}, "trials": []})
    with pytest.raises(HardwareMismatch):
        training_kwargs(_trainer_cfg(cm, tiny_dir, apply_tuned=True), 8, [])


def test_bf16_needs_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bf16_ok() is False
