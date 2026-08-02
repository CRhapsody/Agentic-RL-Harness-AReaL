from __future__ import annotations

from typing import Any

from areal.api.cli_args import InferenceEngineConfig
from areal.engine.sglang_remote import RemoteSGLangEngine, SGLangBackend
from areal.infra import RemoteInfEngine

from jphrl.compat.sglang_score import parse_sglang_score_tail


class JPHSGLangBackend(SGLangBackend):
    """AReaL v2.0.0 backend with a fail-closed SGLang score-tail parser."""

    def parse_score_response(
        self,
        response: dict[str, Any],
        target_len: int,
    ) -> list[float]:
        return parse_sglang_score_tail(response, target_len)


class JPHRemoteSGLangEngine(RemoteSGLangEngine):
    """Thin engine adapter that changes only the score response parser."""

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._engine = RemoteInfEngine(config, JPHSGLangBackend())
