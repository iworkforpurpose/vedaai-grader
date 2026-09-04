"""The cross-encoder that answers the checks, on this machine.

Kept apart from `verifier.py` so the decision logic — thresholds, windows, the
deferral band — is testable without downloading a model or importing torch. The
verifier takes an `Entailment` and does not care what implements it.

**Why an encoder rather than a small generative model.** The task is a fixed
binary question about two given texts, which is what a cross-encoder is: both
texts go through the network together, so the attention is over the pair rather
than over two independently-compressed summaries of it. That is also why a
bi-encoder — the embedding scorer already in this codebase — cannot do this job:
cosine between an answer and a claim measures topical similarity, and "the lines
cross at the poles" is maximally similar to "the lines never cross" while being
its negation.

**Why this size.** A 0.4B ModernBERT cross-encoder matches a DeBERTa-v3-large
teacher on NLI while using half the memory and running faster, at hundreds of
pairs a second on a CPU. One script is a few thousand pairs, so marking becomes
a fraction of a second and costs nothing.

The model is downloaded once and cached by `transformers`. It is an optional
extra: without it the graders that need a provider are unaffected, and this one
reports itself unavailable rather than failing a submission.
"""

from __future__ import annotations

import os

#: The cross-encoder. Overridable, because this is exactly the kind of choice that
#: should be settled by `score_scientsbank.py` rather than by taste.
#:
#: The default is an NLI model rather than a general reranker: it is trained on
#: entailment, which is the question being asked, and its label set distinguishes
#: contradiction from neutral — a difference this product needs, since a
#: contradicted check is a confident "no" and a neutral one is a deferral.
MODEL = os.getenv("NLI_MODEL") or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

#: Pairs scored per forward pass. Large enough to keep a CPU busy, small enough
#: that a long answer does not allocate a batch the size of the page.
BATCH = int(os.getenv("NLI_BATCH") or 32)


def available() -> bool:
    """Whether local entailment can run at all."""
    if os.environ.get("NLI", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class CrossEncoderEntailment:
    """Scores entailment for (premise, hypothesis) pairs, batched, on CPU.

    Loaded lazily and once. The first call pays for the download and the graph
    build; every call after it is a matrix multiply.
    """

    def __init__(self, model: str | None = None, batch: int | None = None) -> None:
        self.model_name = model or MODEL
        self.batch = batch or BATCH
        self._tokenizer = None
        self._model = None
        self._entail_index: int | None = None

    @property
    def name(self) -> str:
        return f"nli:{self.model_name}"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        torch.set_num_threads(int(os.getenv("NLI_THREADS") or os.cpu_count() or 4))

        # Which output column means "entailment", read from the model's own label
        # map rather than assumed. NLI checkpoints disagree about the order —
        # some are (contradiction, neutral, entailment), others the reverse — and
        # guessing produces a marker that is confidently backwards.
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._entail_index = labels.get("entailment", len(labels) - 1)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment probability for each (premise, hypothesis) pair."""
        if not pairs:
            return []
        self._load()
        import torch

        out: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch):
                chunk = pairs[start : start + self.batch]
                encoded = self._tokenizer(
                    [p for p, _h in chunk],
                    [h for _p, h in chunk],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, self._entail_index]
                out.extend(float(x) for x in probs)
        return out


_shared: CrossEncoderEntailment | None = None


def shared() -> CrossEncoderEntailment:
    """One loaded model per process. It is ~400MB; two would be careless."""
    global _shared
    if _shared is None:
        _shared = CrossEncoderEntailment()
    return _shared
