"""The one place a classification row becomes a causal-LM prompt, a training example, or a score.

Training (model_trainer), evaluation, prediction and the perf entry all go through here, so the
prompt a model is trained on can never drift from the prompt it is scored with.

  build_prompt     chat prompt asking for the label (article text trimmed to fit max_length)
  encode_example   prompt + label completion + EOS, loss labels = IGNORE on the prompt
  pad_batch        right-pad a list of encoded examples (labels padded with IGNORE)
  score_labels     length-normalised log-likelihood of every candidate label -> [n, n_labels]

Scoring runs one forward per (text, label) pair: O(n_labels) cost, fine for a handful of labels.
"""

import numpy as np

IGNORE = -100   # cross-entropy ignore index: positions the loss must not learn


def _render(tokenizer, task, text, labels) -> str:
    user = task.prompt_template.format(text=text, labels=", ".join(labels))
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": task.system_prompt}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{task.system_prompt}\n\n{user}\n"


def _ids(tokenizer, text) -> list:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _completion(tokenizer, label) -> list:
    return _ids(tokenizer, str(label)) + [tokenizer.eos_token_id]


def build_prompt(tokenizer, task, text, labels, max_length=None) -> str:
    """The prompt for `text`. With max_length, the article is cut so that the prompt plus the
    longest label completion fits."""
    if max_length:
        overhead = len(_ids(tokenizer, _render(tokenizer, task, "", labels)))
        budget = max_length - overhead - max(len(_completion(tokenizer, lb)) for lb in labels)
        text_ids = _ids(tokenizer, text)
        if len(text_ids) > budget:
            text = tokenizer.decode(text_ids[: max(budget, 0)])
    return _render(tokenizer, task, text, labels)


def encode_example(tokenizer, task, text, label, labels, max_length=None) -> dict:
    """Training example: prompt ids, then label ids + EOS. Loss labels are IGNORE on the prompt."""
    prompt = _ids(tokenizer, build_prompt(tokenizer, task, text, labels, max_length))
    completion = _completion(tokenizer, label)
    return {
        "input_ids": prompt + completion,
        "attention_mask": [1] * (len(prompt) + len(completion)),
        "labels": [IGNORE] * len(prompt) + completion,
    }


def pad_batch(features, pad_id) -> dict:
    """Right-pad encoded examples into tensors; padding gets attention 0 and label IGNORE."""
    import torch

    width = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        gap = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_id] * gap)
        batch["attention_mask"].append(f["attention_mask"] + [0] * gap)
        batch["labels"].append(f["labels"] + [IGNORE] * gap)
    return {k: torch.tensor(v) for k, v in batch.items()}


def score_labels(model, tokenizer, task, texts, labels, max_length=None, batch_size=16) -> np.ndarray:
    """[len(texts), len(labels)] mean per-token log-likelihood of each label completion (label
    tokens + EOS) given the prompt. Length-normalised so short label names are not favoured."""
    import torch

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pairs = [(t, lb) for t in texts for lb in labels]
    out = []
    model.eval()
    device = next(model.parameters()).device
    for i in range(0, len(pairs), batch_size):
        feats = [encode_example(tokenizer, task, t, lb, labels, max_length) for t, lb in pairs[i : i + batch_size]]
        batch = {k: v.to(device) for k, v in pad_batch(feats, pad_id).items()}
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        target = batch["labels"][:, 1:]
        mask = target != IGNORE
        token_lp = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        out.append(((token_lp * mask).sum(-1) / mask.sum(-1)).cpu().numpy())
    return np.concatenate(out).reshape(len(texts), len(labels))
