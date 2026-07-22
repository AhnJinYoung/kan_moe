from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import load_model_from_checkpoint

try:
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.model import TemplateLM
except ImportError:  # Optional dependency; fail with a useful message at construction.
    lm_eval_utils = None
    TemplateLM = object  # type: ignore[assignment,misc]


class CheckpointHarnessLM(TemplateLM):  # type: ignore[misc]
    """lm-evaluation-harness adapter for a native dmoe checkpoint."""

    backend = "causal"

    def __init__(
        self,
        checkpoint: str,
        tokenizer_path: str,
        device: str = "cuda:0",
        batch_size: int = 8,
        max_gen_toks: int = 256,
        top_k: int | None = None,
        precision: str = "bf16",
    ) -> None:
        if lm_eval_utils is None:
            raise RuntimeError(
                "lm-evaluation-harness is not installed; install the 'eval' extra"
            )
        super().__init__()
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("transformers is required for benchmark evaluation") from error

        self._device = torch.device(device)
        self.model, self.checkpoint_state = load_model_from_checkpoint(
            checkpoint, self._device, top_k=top_k
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, trust_remote_code=False
        )
        self._tokenizer_path = str(Path(tokenizer_path))
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._precision = precision
        tokenizer_size = len(self.tokenizer)
        if tokenizer_size > self.model.config.vocab_size:
            raise ValueError(
                f"tokenizer has {tokenizer_size} tokens but model vocabulary has "
                f"{self.model.config.vocab_size}"
            )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.checkpoint_state["config"]["data"]["eos_token_id"]
        self._eot_token_id = int(eos_id)

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return self.model.config.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_path

    def get_model_info(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_state.get("step"),
            "tokens_seen": self.checkpoint_state.get("tokens_seen"),
            "model_type": self.model.config.model_type,
            "top_k": self.model.config.top_k,
            "aggregation": self.model.config.aggregation,
            **self.model.parameter_report(),
        }

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs: Any
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(
            string, add_special_tokens=add_special_tokens, **kwargs
        )

    def tok_decode(self, tokens: int | list[int], **kwargs: Any) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, **kwargs)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self._precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self._precision == "bf16" else torch.float16
        return torch.autocast(self._device.type, dtype=dtype)

    @torch.inference_mode()
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        del disable_tqdm, kwargs
        indexed = list(enumerate(requests))
        indexed.sort(key=lambda item: len(item[1][1]) + len(item[1][2]), reverse=True)
        results: list[tuple[float, bool] | None] = [None] * len(requests)

        for batch_start in range(0, len(indexed), self._batch_size):
            chunk = indexed[batch_start : batch_start + self._batch_size]
            encoded_inputs: list[list[int]] = []
            continuations: list[list[int]] = []
            input_lengths: list[int] = []
            for _, (_, context_tokens, continuation_tokens) in chunk:
                if not context_tokens or not continuation_tokens:
                    raise ValueError("context and continuation token lists must be non-empty")
                if len(continuation_tokens) > self.max_length:
                    raise ValueError("continuation exceeds model maximum sequence length")
                combined = (context_tokens + continuation_tokens)[
                    -(self.max_length + 1) :
                ]
                model_input = combined[:-1]
                encoded_inputs.append(model_input)
                continuations.append(continuation_tokens)
                input_lengths.append(len(model_input))

            padded_length = max(input_lengths)
            input_tensor = torch.full(
                (len(chunk), padded_length),
                self.eot_token_id,
                dtype=torch.long,
                device=self._device,
            )
            for row, tokens in enumerate(encoded_inputs):
                input_tensor[row, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=self._device
                )

            with self._autocast():
                logits = self.model(input_tensor)["logits"].float()
            log_probabilities = F.log_softmax(logits, dim=-1)
            for row, (original_index, request) in enumerate(chunk):
                key, _, _ = request
                continuation = continuations[row]
                continuation_length = len(continuation)
                input_length = input_lengths[row]
                selected = log_probabilities[
                    row, input_length - continuation_length : input_length, :
                ]
                target = torch.tensor(
                    continuation, dtype=torch.long, device=self._device
                )
                token_scores = selected.gather(-1, target[:, None]).squeeze(-1)
                score = float(token_scores.sum().item())
                greedy = bool(torch.equal(selected.argmax(dim=-1), target))
                results[original_index] = (score, greedy)
                if key is not None:
                    self.cache_hook.add_partial("loglikelihood", key, (score, greedy))

        return [result for result in results if result is not None]

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        del disable_tqdm
        results: list[float] = []
        for request in requests:
            (text,) = request.args
            windows = list(
                map(
                    lm_eval_utils.make_disjoint_window,
                    lm_eval_utils.get_rolling_token_windows(
                        token_list=self.tok_encode(text),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            token_requests = [(None, context, continuation) for context, continuation in windows]
            scores = self._loglikelihood_tokens(token_requests, disable_tqdm=True)
            total = sum(score for score, _ in scores)
            results.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
        return results

    @torch.inference_mode()
    def generate_until(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[str]:
        del disable_tqdm
        results: list[str] = []
        for request in requests:
            context, generation_kwargs = request.args
            generation_kwargs = dict(generation_kwargs)
            until = generation_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = int(
                generation_kwargs.get("max_gen_toks", self.max_gen_toks)
            )
            maximum_context = max(1, self.max_length - max_new_tokens)
            context_tokens = self.tok_encode(context)[-maximum_context:]
            if not context_tokens:
                context_tokens = [self.prefix_token_id]
            input_tensor = torch.tensor(
                context_tokens, dtype=torch.long, device=self._device
            )[None, :]
            with self._autocast():
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.eot_token_id,
                )
            continuation_tokens = generated[0, len(context_tokens) :].tolist()
            text = self.tok_decode(continuation_tokens, skip_special_tokens=False)
            stop_positions = [text.find(stop) for stop in until if stop and stop in text]
            if stop_positions:
                text = text[: min(stop_positions)]
            results.append(text)
            self.cache_hook.add_partial(
                "generate_until", (context, generation_kwargs), text
            )
        return results

