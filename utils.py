"""Basic helpers for evaluation: tasks, models, data and input domains.

A task is one (model, dataset) pair from `configs/benchmarks.yaml`. The main entry point is

    model, loader, domain = load_data_model(task_id, data_num)

Conventions shared by every task:

* `model` is an `nn.Module` in eval mode, float32, whose `forward(*inputs)` returns one tensor.
  Image models take pixels in [0, 1]; the normalization is inside the model, so the input domain
  is in pre-normalization space as `configs/domains.yaml` requires.
* `loader` holds exactly `data_num` samples. Each batch is `(inputs, labels)`: `inputs` is a tuple
  of tensors passed as `model(*inputs)`, and `labels` is a LongTensor (-1 where a task has no
  class label).
* `domain` is an `InputDomain` (see below).

Samples are drawn with a fixed seed. Datasets are streamed, so only the selected samples are
downloaded. Gated sources need a Hugging Face token (`huggingface-cli login`):
ILSVRC/imagenet-1k and the Llama models. ImageNet can instead be read from a local folder in
torchvision ImageFolder layout: set `DLCLV_IMAGENET_DIR=/path/to/imagenet/val`.

Not supported yet: `vnncomp_small_nets` (needs the pinned VNN-COMP ONNX files) and
`text_classification_v2` (TFLite only).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 20261031            # configs/datasets.yaml sampling seed
SHUFFLE_BUFFER = 10_000    # streaming shuffle buffer
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    task_id: int
    model: str          # model id in configs/benchmarks.yaml
    dataset: str        # dataset id in configs/datasets.yaml
    tier: str           # core | ambitious
    seq_len: int = 0    # fixed sequence length for token inputs
    device: str = ""    # forced device; empty means the caller's choice


TASKS = [
    Task(0, "squeezenet_1_1", "imagenet_val", "core"),
    Task(1, "mobilenet_v2", "imagenet_val", "core"),
    Task(2, "resnet18", "imagenet_val", "core"),
    Task(3, "all_minilm_l6_v2", "sst2", "core", seq_len=128),
    Task(4, "resnet50", "imagenet_val", "ambitious"),
    Task(5, "whisper_tiny_encoder", "librispeech", "ambitious"),
    Task(6, "distilbert_sst2", "sst2", "ambitious", seq_len=128),
    Task(7, "vit_b_16", "imagenet_val", "ambitious"),
    Task(8, "smollm2_135m", "tinystories", "ambitious", seq_len=128),
    Task(9, "smollm2_135m", "open_orca", "ambitious", seq_len=128),
    Task(10, "gpt2", "tinystories", "ambitious", seq_len=128),
    Task(11, "gpt2", "open_orca", "ambitious", seq_len=128),
    Task(12, "bert_large_squad", "squad_1_1", "ambitious", seq_len=384),
    Task(13, "qwen3_0_6b", "open_orca", "ambitious", seq_len=128),
    Task(14, "llama_3_2_1b", "open_orca", "ambitious", seq_len=128),
    Task(15, "llama_3_1_8b", "cnn_dailymail", "ambitious", seq_len=128),
    # Quantized kernels exist only on CPU in PyTorch.
    Task(16, "quantized_mobilenet_v2_int8", "imagenet_val", "ambitious", device="cpu"),
]


def get_task(task_id: int) -> Task:
    if not 0 <= task_id < len(TASKS):
        raise ValueError(f"task_id {task_id} not in [0, {len(TASKS) - 1}]")
    return TASKS[task_id]


def list_tasks():
    return list(TASKS)


def set_random_seed(seed: int = SEED, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------------
# Input domains (configs/domains.yaml)
# --------------------------------------------------------------------------------------------

@dataclass
class InputDomain:
    """A box over each model input, without the batch dimension.

    `lower[i]` and `upper[i]` bound input i elementwise. They are Python numbers or tensors that
    broadcast to `shapes[i]`. Integer inputs (token ids, masks) take every integer in the box.
    """
    domain_id: str
    kind: str                       # box | finite_integer_vector
    names: list
    shapes: list
    dtypes: list
    lower: list
    upper: list
    note: str = ""
    meta: dict = field(default_factory=dict)

    def contains(self, inputs) -> bool:
        """True if every sample in a batched input tuple lies in the domain."""
        if len(inputs) != len(self.names):
            return False
        for x, shape, dt, lo, hi in zip(inputs, self.shapes, self.dtypes, self.lower, self.upper):
            if tuple(x.shape[1:]) != tuple(shape) or x.dtype != dt:
                return False
            lo = torch.as_tensor(lo, dtype=torch.float64, device=x.device)
            hi = torch.as_tensor(hi, dtype=torch.float64, device=x.device)
            xd = x.to(torch.float64)
            if not (torch.isfinite(xd).all() and (xd >= lo).all() and (xd <= hi).all()):
                return False
        return True

    def sample(self, n: int = 1, generator: Optional[torch.Generator] = None):
        """Draw n points uniformly from the box (for smoke tests, never for verification)."""
        out = []
        for shape, dt, lo, hi in zip(self.shapes, self.dtypes, self.lower, self.upper):
            lo = torch.as_tensor(lo, dtype=torch.float64).expand(shape)
            hi = torch.as_tensor(hi, dtype=torch.float64).expand(shape)
            u = torch.rand((n, *shape), dtype=torch.float64, generator=generator)
            if dt.is_floating_point:
                out.append((lo + u * (hi - lo)).to(dt))
            else:
                out.append(torch.floor(lo + u * (hi - lo + 1)).clamp(max=hi).to(dt))
        return tuple(out)


def image_domain(size: int = 224) -> InputDomain:
    return InputDomain(
        "D4_global_valid", "box", ["pixels"], [(3, size, size)], [torch.float32], [0.0], [1.0],
        note="RGB pixels in [0, 1] before normalization; local balls D1/D2 clip to this box")


def token_domain(vocab_size: int, seq_len: int, token_type: bool = False) -> InputDomain:
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if token_type else [])
    n = len(names)
    return InputDomain(
        "D5_token_ids", "finite_integer_vector", names, [(seq_len,)] * n, [torch.long] * n,
        [0] * n, [vocab_size - 1, 1] + ([1] if token_type else []),
        note=f"token ids in [0, {vocab_size - 1}] at fixed sequence length {seq_len}; "
             "padding positions are masked",
        meta={"vocab_size": vocab_size, "seq_len": seq_len})


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------

class Normalized(nn.Module):
    """Applies (x - mean) / std before the wrapped network, so inputs stay in [0, 1]."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class HFLogits(nn.Module):
    """Returns `.logits` from a Hugging Face model called with token inputs."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask):
        return self.net(input_ids=input_ids, attention_mask=attention_mask).logits


class HFPixelLogits(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        return self.net(pixel_values=pixel_values).logits


class MeanPooledEmbedding(nn.Module):
    """sentence-transformers pipeline: encoder, mean pooling over unmasked tokens, L2 norm."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask):
        h = self.net(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return nn.functional.normalize(pooled, p=2, dim=1)


class SpanLogits(nn.Module):
    """Question answering: stacks start and end logits into shape [B, T, 2]."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask, token_type_ids):
        o = self.net(input_ids=input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids)
        return torch.stack([o.start_logits, o.end_logits], dim=-1)


class EncoderHidden(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features=input_features).last_hidden_state


def _torchvision(name):
    import torchvision.models as tvm
    net = getattr(tvm, name)(weights="IMAGENET1K_V1")
    return Normalized(net, IMAGENET_MEAN, IMAGENET_STD)


def _quantized_mobilenet_v2():
    from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights
    net = mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True)
    return Normalized(net, IMAGENET_MEAN, IMAGENET_STD)


HF_SOURCES = {
    "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert_sst2": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "vit_b_16": "google/vit-base-patch16-224",
    "whisper_tiny_encoder": "openai/whisper-tiny",
    "bert_large_squad": "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
    "gpt2": "openai-community/gpt2",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "llama_3_2_1b": "meta-llama/Llama-3.2-1B",
    "llama_3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}
LLMS = {"smollm2_135m", "gpt2", "qwen3_0_6b", "llama_3_2_1b", "llama_3_1_8b"}


def _hf_model(name):
    import transformers as tf
    src = HF_SOURCES[name]
    kw = dict(dtype=torch.float32, attn_implementation="eager")
    if name == "all_minilm_l6_v2":
        return MeanPooledEmbedding(tf.AutoModel.from_pretrained(src, **kw))
    if name == "distilbert_sst2":
        return HFLogits(tf.AutoModelForSequenceClassification.from_pretrained(src, **kw))
    if name == "bert_large_squad":
        return SpanLogits(tf.AutoModelForQuestionAnswering.from_pretrained(src, **kw))
    if name == "vit_b_16":
        net = tf.ViTForImageClassification.from_pretrained(src, **kw)
        proc = tf.AutoImageProcessor.from_pretrained(src)
        return Normalized(HFPixelLogits(net), proc.image_mean, proc.image_std)
    if name == "whisper_tiny_encoder":
        return EncoderHidden(tf.WhisperModel.from_pretrained(src, **kw).get_encoder())
    if name in LLMS:
        net = tf.AutoModelForCausalLM.from_pretrained(src, **kw)
        net.config.use_cache = False
        return HFLogits(net)
    raise KeyError(name)


def load_model(task_id: int, device: Optional[str] = None) -> nn.Module:
    """Pretrained model for a task, in eval mode with gradients off."""
    task = get_task(task_id)
    device = task.device or device or default_device()
    name = task.model
    if name in ("squeezenet_1_1", "mobilenet_v2", "resnet18", "resnet50"):
        model = _torchvision({"squeezenet_1_1": "squeezenet1_1"}.get(name, name))
    elif name == "quantized_mobilenet_v2_int8":
        model = _quantized_mobilenet_v2()
    else:
        model = _hf_model(name)
    model.eval().requires_grad_(False)
    model.task = task
    return model.to(device)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

class SampleDataset(Dataset):
    """In-memory list of (inputs_tuple, label) pairs."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    inputs = tuple(torch.stack(xs) for xs in zip(*(b[0] for b in batch)))
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return inputs, labels


def _take(stream, n, seed, label_key=None):
    """Seeded selection of n rows from a streamed split, stratified by label when given."""
    rows = stream.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    if label_key is None:
        out = list(rows.take(n))
    else:
        pool = list(rows.take(max(10 * n, 200)))
        by_label = {}
        for r in pool:
            by_label.setdefault(r[label_key], []).append(r)
        groups = [by_label[k] for k in sorted(by_label)]
        out, i = [], 0
        while len(out) < n and any(i < len(g) for g in groups):
            out += [g[i] for g in groups if i < len(g)][: n - len(out)]
            i += 1
    if len(out) < n:
        raise ValueError(f"asked for {n} samples, source has {len(out)}")
    return out


def _stream(path, split, name=None):
    from datasets import load_dataset
    return load_dataset(path, name, split=split, streaming=True)


def _image_transform(model_name):
    import torchvision.transforms as T
    if model_name == "vit_b_16":   # ViTImageProcessor: resize to 224x224, no crop
        return T.Compose([T.Resize((224, 224)), T.ToTensor()])
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def _imagenet_rows(n, seed):
    root = os.environ.get("DLCLV_IMAGENET_DIR")
    if root:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(root)
        idx = np.random.RandomState(seed).permutation(len(ds))[:n]
        if len(idx) < n:
            raise ValueError(f"asked for {n} samples, {root} has {len(ds)}")
        return [{"image": ds.loader(ds.samples[i][0]), "label": ds.samples[i][1]} for i in idx]
    return _take(_stream("ILSVRC/imagenet-1k", "validation"), n, seed)


def _tokenizer(model_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_SOURCES[model_name])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _encode(tok, seq_len, *texts, token_type=False):
    enc = tok(*texts, padding="max_length", truncation="only_second" if len(texts) == 2 else True,
              max_length=seq_len, return_tensors="pt", return_token_type_ids=token_type)
    keys = ["input_ids", "attention_mask"] + (["token_type_ids"] if token_type else [])
    return tuple(enc[k][0].long() for k in keys)


def _vocab_size(tok, model):
    emb = model.modules()
    for m in emb:
        if isinstance(m, nn.Embedding):
            return m.num_embeddings     # first embedding is the token table in all HF models
    return len(tok)


def load_data(task_id: int, data_num: int, model: nn.Module, seed: int = SEED):
    """Return (samples, domain) for a task. `model` is used only to read the vocabulary size."""
    task = get_task(task_id)
    name, ds = task.model, task.dataset

    if ds == "imagenet_val":
        tfm = _image_transform(name)
        rows = _imagenet_rows(data_num, seed)
        samples = [((tfm(r["image"].convert("RGB")).float(),), int(r["label"])) for r in rows]
        return samples, image_domain()

    if ds == "librispeech":
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(HF_SOURCES[name])
        rows = _take(_stream("openslr/librispeech_asr", "validation", "clean"), data_num, seed)
        samples = []
        for r in rows:
            a = r["audio"]
            feats = fe(a["array"], sampling_rate=a["sampling_rate"], return_tensors="pt")
            samples.append(((feats.input_features[0].float(),), -1))
        x = torch.stack([s[0][0] for s in samples])
        dom = InputDomain(
            "D3_data_hull", "box", ["input_features"], [tuple(x.shape[1:])], [torch.float32],
            [x.min(0).values], [x.max(0).values],
            note="per-feature min/max of log-mel features over the selected samples only")
        return samples, dom

    tok = _tokenizer(name)
    vocab = _vocab_size(tok, model)

    if ds == "sst2":
        rows = _take(_stream("stanfordnlp/sst2", "validation"), data_num, seed, label_key="label")
        samples = [(_encode(tok, task.seq_len, r["sentence"]), int(r["label"])) for r in rows]
        return samples, token_domain(vocab, task.seq_len)

    if ds == "squad_1_1":
        rows = _take(_stream("rajpurkar/squad", "validation"), data_num, seed)
        samples = [(_encode(tok, task.seq_len, r["question"], r["context"], token_type=True), -1)
                   for r in rows]
        return samples, token_domain(vocab, task.seq_len, token_type=True)

    text_of: Callable
    if ds == "tinystories":
        stream, text_of = _stream("roneneldan/TinyStories", "validation"), (lambda r: r["text"])
    elif ds == "open_orca":
        stream = _stream("Open-Orca/OpenOrca", "train")
        text_of = lambda r: (r["system_prompt"] + "\n\n" + r["question"]).strip()
    elif ds == "cnn_dailymail":
        stream = _stream("abisee/cnn_dailymail", "test", "3.0.0")
        text_of = lambda r: "Summarize the following news article:\n\n" + r["article"]
    else:
        raise NotImplementedError(f"dataset {ds}")
    rows = _take(stream, data_num, seed)
    samples = [(_encode(tok, task.seq_len, text_of(r)), -1) for r in rows]
    return samples, token_domain(vocab, task.seq_len)


def load_data_model(task_id: int, data_num: int, batch_size: int = 8,
                    device: Optional[str] = None, seed: int = SEED):
    """Load the model, a DataLoader with exactly `data_num` samples, and the input domain.

    Returns `(model, dataloader, input_domain)`. See the module docstring for conventions.
    """
    if data_num < 1:
        raise ValueError("data_num must be positive")
    set_random_seed(seed)
    model = load_model(task_id, device)
    samples, domain = load_data(task_id, data_num, model, seed)
    loader = DataLoader(SampleDataset(samples), batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    return model, loader, domain
