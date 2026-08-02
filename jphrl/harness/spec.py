from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json


class HarnessAction(str, Enum):
    """The bounded action space from the JPH-RL project plan."""

    DIRECT = "DIRECT"
    RETRIEVE_SKILL = "RETRIEVE_SKILL"
    VERIFY = "VERIFY"
    REPLAN = "REPLAN"
    COMPRESS = "COMPRESS"


@dataclass(frozen=True)
class HarnessSpec:
    version: str = "calculator-harness-v1"
    prompt_version: str = "calculator-json-v1"
    parser_version: str = "strict-json-v1"
    tool_schema_version: str = "calculator-fraction-v1"
    evaluator_version: str = "calculator-exact-v1"
    max_model_retries: int = 2
    max_tool_calls: int = 1

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
