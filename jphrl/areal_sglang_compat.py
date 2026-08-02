from __future__ import annotations

from typing import Any

from areal.api.cli_args import InferenceEngineConfig
from areal.engine.sglang_remote import RemoteSGLangEngine, SGLangBackend
from areal.infra import RemoteInfEngine

from jphrl.compat.sglang_score import SGLangScoreTailBinding


class JPHSGLangBackend(SGLangBackend):
    """AReaL v2.0.0 backend with a fail-closed SGLang score-tail parser."""

    def __init__(self) -> None:
        self._score_tail_binding = SGLangScoreTailBinding()

    def build_score_request(
        self,
        input_ids: list[int],
        target_len: int,
        with_lora: bool,
        version: int,
    ):
        request = super().build_score_request(
            input_ids=input_ids,
            target_len=target_len,
            with_lora=with_lora,
            version=version,
        )
        self._score_tail_binding.begin(input_ids, target_len)
        return request

    def parse_score_response(
        self,
        response: dict[str, Any],
        target_len: int,
    ) -> list[float]:
        return self._score_tail_binding.consume(response, target_len)


class JPHRemoteSGLangEngine(RemoteSGLangEngine):
    """Thin engine adapter that changes only the score response parser."""

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._engine = RemoteInfEngine(config, JPHSGLangBackend())
