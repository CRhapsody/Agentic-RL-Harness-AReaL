from __future__ import annotations

import hashlib
from typing import Any

from .base import ModelResponse


class HuggingFaceChatModel:
    """Small-model smoke backend with exact generated token log-prob capture."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The hf backend requires torch and transformers; run scripts/bootstrap_remote.sh"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=False,
        )
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        self.model.to(device)
        self.model.eval()
        config_fingerprint = hashlib.sha256(
            str(self.model.config.to_dict()).encode("utf-8")
        ).hexdigest()[:12]
        resolved_commit = getattr(self.model.config, "_commit_hash", None)
        if not isinstance(resolved_commit, str) or len(resolved_commit) != 40:
            raise RuntimeError("Hugging Face model revision did not resolve to a 40-character commit")
        self.policy_version = f"hf:{model_name}@{resolved_commit}:{config_fingerprint}"
        tokenizer_payload = str(self.tokenizer.init_kwargs)
        tokenizer_fingerprint = hashlib.sha256(tokenizer_payload.encode("utf-8")).hexdigest()[:12]
        self.tokenizer_version = f"hf:{model_name}@{resolved_commit}:{tokenizer_fingerprint}"

    def _encode(self, messages: list[dict[str, str]]) -> Any:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        rendered = "\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
        rendered += "\nASSISTANT:"
        return self.tokenizer(rendered, return_tensors="pt").input_ids

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> ModelResponse:
        torch = self._torch
        input_ids = self._encode(messages).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids_tensor = generated.sequences[0, input_ids.shape[1] :]
        output_ids = output_ids_tensor.tolist()
        output_logprobs: list[float] = []
        for score, token_id in zip(generated.scores, output_ids):
            logprob = torch.log_softmax(score[0].float(), dim=-1)[token_id]
            output_logprobs.append(float(logprob.cpu()))
        return ModelResponse(
            text=self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            input_token_ids=input_ids[0].tolist(),
            output_token_ids=output_ids,
            output_token_logprobs=output_logprobs,
            output_versions=[0] * len(output_ids),
            completion_loss_mask=[1] * len(output_ids),
            policy_version=self.policy_version,
            tokenizer_version=self.tokenizer_version,
            policy_kind="causal_lm",
            token_metadata_status="available",
        )
