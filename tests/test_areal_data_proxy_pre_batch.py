from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from jphrl.trajectory.areal_data_proxy_pre_batch import (
    HOOK_STAGE,
    ArealDataProxyPreBatchHookError,
    VerifiedDataProxyPreBatchHook,
    export_session_trajectory_with_pre_batch_hook,
    require_data_proxy_pre_batch_export,
    validate_pre_batch_trajectory_export,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREAL_ROOT = PROJECT_ROOT.parent / "AReaL"
PINNED_AREAL_COMMIT = "fee938eada49208a5aabdbc1095730a13076a349"
PATCH_PATH = (
    PROJECT_ROOT / "patches" / "areal-v2.0.0-data-proxy-pre-batch-hook.patch"
)
_ASSERTIONS = unittest.TestCase()


def _load_pinned_areal() -> dict[str, Any]:
    areal_root = Path(
        os.environ.get("JPH_AREAL_SOURCE", str(DEFAULT_AREAL_ROOT))
    ).resolve()
    if not (areal_root / "areal").is_dir():
        raise unittest.SkipTest(f"pinned AReaL source is unavailable: {areal_root}")
    if str(areal_root) not in sys.path:
        sys.path.insert(0, str(areal_root))
    try:
        import torch
        from areal.api import ModelResponse
        from areal.experimental.openai.types import InteractionWithTokenLogpReward
        from areal.utils.data import concat_padded_tensors
        from areal.v2.inference_service.data_proxy.session import SessionData
    except ImportError as exc:
        raise unittest.SkipTest(
            f"pinned AReaL CPU test dependencies are unavailable: {exc}"
        ) from exc
    return {
        "torch": torch,
        "ModelResponse": ModelResponse,
        "Interaction": InteractionWithTokenLogpReward,
        "concat_padded_tensors": concat_padded_tensors,
        "SessionData": SessionData,
    }


def _ready_session(*, session_id: str, style: str) -> tuple[Any, int]:
    areal = _load_pinned_areal()
    ModelResponse = areal["ModelResponse"]
    Interaction = areal["Interaction"]
    session = areal["SessionData"](session_id=session_id)
    chat_template_type = "concat" if style == "concat" else "hf"

    first = Interaction(
        messages=[{"role": "user", "content": "calculate 17 + 25"}],
        output_message_list=[
            {
                "role": "assistant",
                "content": '{"tool":"calculator","expression":"17 + 25"}',
            }
        ],
        model_response=ModelResponse(
            input_tokens=[10, 11],
            output_tokens=[20, 21],
            output_logprobs=[-0.2, -0.3],
            output_versions=[7, 7],
            stop_reason="tool_calls",
        ),
        chat_template_type=chat_template_type,
    )
    first._interaction_id = "interaction-1"
    session.active_completions[first.interaction_id] = first

    second = Interaction(
        messages=[
            {"role": "user", "content": "calculate 17 + 25"},
            {
                "role": "assistant",
                "content": '{"tool":"calculator","expression":"17 + 25"}',
            },
            {"role": "tool", "content": "42"},
        ],
        output_message_list=[
            {"role": "assistant", "content": '{"answer":"42"}'}
        ],
        model_response=ModelResponse(
            input_tokens=[10, 11, 20, 21, 30],
            output_tokens=[40],
            output_logprobs=[-0.4],
            output_versions=[7],
            stop_reason="stop",
        ),
        chat_template_type=chat_template_type,
    )
    second._interaction_id = "interaction-2"
    session.active_completions[second.interaction_id] = second
    assert second.parent is first

    reward_result = session.set_reward(interaction_id=None, reward=1.0)
    assert reward_result.trajectory_id is not None
    return session, reward_result.trajectory_id


def test_pre_batch_hook_individual_session_preserves_each_interaction_identity() -> None:
    """Individual export exposes both real interactions before any batch merge."""

    # Arrange
    areal = _load_pinned_areal()
    session, trajectory_id = _ready_session(
        session_id="individual-session", style="individual"
    )
    observed = []

    def consume(export):
        observed.append(export)
        return {"ignored": True}

    hook = VerifiedDataProxyPreBatchHook(consume)

    # Act
    exported_trajectory_id, interactions = asyncio.run(
        export_session_trajectory_with_pre_batch_hook(
            session,
            discount=1.0,
            style="individual",
            trajectory_id=trajectory_id,
            hook=hook,
        )
    )
    audit = validate_pre_batch_trajectory_export(observed[0])

    # Assert
    assert exported_trajectory_id == trajectory_id
    assert list(interactions) == ["interaction-1", "interaction-2"]
    assert len(observed) == 1
    assert observed[0].session_id == "individual-session"
    assert observed[0].trajectory_id == trajectory_id
    assert observed[0].hook_stage == HOOK_STAGE
    assert audit.exported_interaction_ids == (
        "interaction-1",
        "interaction-2",
    )
    assert audit.all_interaction_ids == ("interaction-1", "interaction-2")
    with _ASSERTIONS.assertRaises(TypeError):
        observed[0].exported_interactions["invented"] = interactions["interaction-1"]
    areal["torch"].testing.assert_close(
        interactions["interaction-2"].to_tensor_dict()["loss_mask"],
        areal["torch"].tensor([[0, 0, 0, 0, 0, 1]]),
        rtol=0,
        atol=0,
    )


def test_pre_batch_hook_concat_session_preserves_ancestor_identity() -> None:
    """Concat export exposes its real leaf and recoverable parent chain to the hook."""

    # Arrange
    areal = _load_pinned_areal()
    session, trajectory_id = _ready_session(
        session_id="concat-session", style="concat"
    )
    observed = []

    async def consume(export) -> None:
        observed.append(export)

    hook = VerifiedDataProxyPreBatchHook(consume)

    # Act
    exported_trajectory_id, interactions = asyncio.run(
        export_session_trajectory_with_pre_batch_hook(
            session,
            discount=1.0,
            style="concat",
            trajectory_id=trajectory_id,
            hook=hook,
        )
    )
    audit = validate_pre_batch_trajectory_export(observed[0])

    # Assert
    assert exported_trajectory_id == trajectory_id
    assert list(interactions) == ["interaction-2"]
    assert len(observed) == 1
    assert observed[0].exported_interactions["interaction-2"].parent.interaction_id == (
        "interaction-1"
    )
    assert audit.exported_interaction_ids == ("interaction-2",)
    assert audit.all_interaction_ids == ("interaction-1", "interaction-2")
    areal["torch"].testing.assert_close(
        interactions["interaction-2"].to_tensor_dict()["loss_mask"],
        areal["torch"].tensor([[0, 0, 1, 1, 0, 1]]),
        rtol=0,
        atol=0,
    )


def test_pre_batch_hook_padded_tensor_batch_rejects_post_batch_binding() -> None:
    """A real padded tensor batch cannot masquerade as interaction objects."""

    # Arrange
    areal = _load_pinned_areal()
    session, trajectory_id = _ready_session(
        session_id="post-batch-session", style="individual"
    )
    _, interactions = asyncio.run(
        export_session_trajectory_with_pre_batch_hook(
            session,
            discount=1.0,
            style="individual",
            trajectory_id=trajectory_id,
            hook=VerifiedDataProxyPreBatchHook(lambda export: None),
        )
    )
    padded_batch = areal["concat_padded_tensors"](
        [interaction.to_tensor_dict() for interaction in interactions.values()]
    )

    # Act / Assert
    with _ASSERTIONS.assertRaisesRegex(
        ArealDataProxyPreBatchHookError,
        "post-batch trajectory data cannot be bound",
    ):
        require_data_proxy_pre_batch_export(
            session_id="post-batch-session",
            trajectory_id=trajectory_id,
            interactions=padded_batch,
            discount=1.0,
            style="individual",
        )


def test_pre_batch_hook_consumer_exception_aborts_export() -> None:
    """Consumer failures cross the callback boundary instead of releasing a batch."""

    # Arrange
    session, trajectory_id = _ready_session(
        session_id="failure-session", style="individual"
    )

    def fail_consumer(export) -> None:
        del export
        raise RuntimeError("staged record write failed")

    hook = VerifiedDataProxyPreBatchHook(fail_consumer)

    # Act / Assert
    with _ASSERTIONS.assertRaisesRegex(RuntimeError, "staged record write failed"):
        asyncio.run(
            export_session_trajectory_with_pre_batch_hook(
                session,
                discount=1.0,
                style="individual",
                trajectory_id=trajectory_id,
                hook=hook,
            )
        )


def test_pre_batch_hook_consumer_key_error_is_not_treated_as_missing_trajectory() -> None:
    """The upstream patch catches only export KeyError, never callback KeyError."""

    # Arrange
    session, trajectory_id = _ready_session(
        session_id="key-error-session", style="individual"
    )

    def fail_lookup(export) -> None:
        del export
        raise KeyError("staged binding is missing")

    hook = VerifiedDataProxyPreBatchHook(fail_lookup)

    # Act / Assert
    with _ASSERTIONS.assertRaisesRegex(KeyError, "staged binding is missing"):
        asyncio.run(
            export_session_trajectory_with_pre_batch_hook(
                session,
                discount=1.0,
                style="individual",
                trajectory_id=trajectory_id,
                hook=hook,
            )
        )


def test_upstream_patch_pinned_tree_applies_and_exposes_deployment_injection() -> None:
    """The pinned tree accepts the patch and its real CLI can import the hook."""

    # Arrange
    areal_root = Path(
        os.environ.get("JPH_AREAL_SOURCE", str(DEFAULT_AREAL_ROOT))
    ).resolve()
    if not (areal_root / ".git").is_dir():
        raise unittest.SkipTest(
            f"pinned AReaL git checkout is unavailable: {areal_root}"
        )
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=areal_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Act
    apply_check = subprocess.run(
        ["git", "apply", "--check", str(PATCH_PATH)],
        cwd=areal_root,
        check=False,
        capture_output=True,
        text=True,
    )
    patch_text = PATCH_PATH.read_text(encoding="utf-8")

    # Assert
    assert actual_commit == PINNED_AREAL_COMMIT
    assert apply_check.returncode == 0, apply_check.stderr
    assert "AREAL_PRE_BATCH_HOOK" in patch_text
    assert "pre_batch_export_hook=pre_batch_export_hook" in patch_text
    assert "exported_trajectory_id, interactions = session.export_trajectory" in patch_text
    callback_position = patch_text.index("+                hook_result = pre_batch_export_hook(")
    assert patch_text.index("except KeyError:") < callback_position
    assert callback_position < patch_text.index(
        "+            merged.update(interactions)"
    )
    main_patch = patch_text.split(
        "diff --git a/areal/v2/inference_service/data_proxy/__main__.py", 1
    )[1].split("diff --git ", 1)[0]
    assert "DataProxyConfig" not in main_patch


def load_tests(loader, tests, pattern):
    """Register dependency-light functions with unittest discovery."""

    del loader, tests, pattern
    suite = unittest.TestSuite()
    for test_function in (
        test_pre_batch_hook_individual_session_preserves_each_interaction_identity,
        test_pre_batch_hook_concat_session_preserves_ancestor_identity,
        test_pre_batch_hook_padded_tensor_batch_rejects_post_batch_binding,
        test_pre_batch_hook_consumer_exception_aborts_export,
        test_pre_batch_hook_consumer_key_error_is_not_treated_as_missing_trajectory,
        test_upstream_patch_pinned_tree_applies_and_exposes_deployment_injection,
    ):
        suite.addTest(
            unittest.FunctionTestCase(
                test_function,
                description=test_function.__doc__,
            )
        )
    return suite
