from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import uuid

from .envs.calculator import CalculatorTask, evaluate_expression, verify_answer
from .harness.controller import HarnessController, HarnessDecision, HarnessState
from .harness.spec import HarnessAction, HarnessSpec
from .models.base import ModelResponse, StructuredChatModel
from .trajectory.schema import EpisodeTrace, JointVersion
from .trajectory.token_contract import validate_token_metadata


SYSTEM_PROMPT = """你是一个受严格 Harness 控制的计算 Agent。
你每次只能输出一个 JSON 对象，不要输出 Markdown、解释或额外文字。
需要调用工具时输出：{"tool":"calculator","expression":"算术表达式"}
收到工具结果后输出：{"answer":"结果"}
禁止在未调用工具时直接给最终答案。"""


@dataclass(frozen=True)
class SmokeResult:
    success: bool
    reward: float | None
    final_answer: str | None
    tool_result: str | None
    trace: EpisodeTrace


def _content_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("model output must be exactly one valid JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def _response_contract_error(response: ModelResponse, trace: EpisodeTrace) -> str | None:
    if response.policy_version != trace.joint_version.policy:
        return "model response policy version differs from pinned episode version"
    if response.tokenizer_version != trace.joint_version.tokenizer:
        return "model response tokenizer version differs from pinned episode version"
    try:
        validate_token_metadata(
            input_token_ids=response.input_token_ids,
            output_token_ids=response.output_token_ids,
            output_token_logprobs=response.output_token_logprobs,
            completion_loss_mask=response.completion_loss_mask,
            policy_kind=response.policy_kind,
            token_metadata_status=response.token_metadata_status,
        )
    except ValueError as exc:
        return str(exc)
    if response.token_metadata_status == "available":
        if len(response.output_versions) != len(response.output_token_ids):
            return "output inference versions do not align with output tokens"
        if not all(
            type(version) is int and version >= 0
            for version in response.output_versions
        ):
            return "output inference versions must be non-negative integers"
    elif response.output_versions:
        return "scripted policy must not fabricate output inference versions"
    return None


def _record_decision(
    trace: EpisodeTrace,
    decision: HarnessDecision,
    state: HarnessState,
) -> None:
    payload = asdict(decision)
    payload["action"] = decision.action.value
    payload["state"] = asdict(state)
    trace.append("harness_decision", "harness", payload)


def _request_model(
    trace: EpisodeTrace,
    model: StructuredChatModel,
    phase: str,
    attempt: int,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, ModelResponse | None, str | None, str | None]:
    call_id = f"{trace.episode_id}:model:{phase}:{attempt}"
    trace.append(
        "model_request",
        "core",
        {
            "model_call_id": call_id,
            "phase": phase,
            "attempt": attempt,
            "effective_prompt_hash": _content_hash(messages),
            "max_new_tokens": max_new_tokens,
        },
    )
    try:
        response = model.generate(messages, max_new_tokens=max_new_tokens)
    except Exception as exc:  # external model implementations define their own errors
        error = f"{type(exc).__name__}: {exc}"
        trace.append(
            "model_error",
            "policy",
            {
                "model_call_id": call_id,
                "phase": phase,
                "attempt": attempt,
                "exception_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return call_id, None, "infrastructure_invalid", error

    contract_error = _response_contract_error(response, trace)
    if contract_error is not None:
        trace.append(
            "model_contract_error",
            "adapter",
            {
                "model_call_id": call_id,
                "phase": phase,
                "attempt": attempt,
                "error": contract_error,
            },
        )
        return call_id, None, "trace_contract_invalid", contract_error

    trace.append(
        "model_response",
        "policy",
        {
            "model_call_id": call_id,
            "phase": phase,
            "attempt": attempt,
            "output_text": response.text,
            "output_text_hash": _text_hash(response.text),
            "input_token_ids": response.input_token_ids,
            "output_token_ids": response.output_token_ids,
            "output_token_logprobs": response.output_token_logprobs,
            "output_versions": response.output_versions,
            "completion_loss_mask": response.completion_loss_mask,
            "policy_release_id": response.policy_version,
            "policy_kind": response.policy_kind,
            "token_metadata_status": response.token_metadata_status,
        },
    )
    return call_id, response, None, None


def _finish(
    trace: EpisodeTrace,
    *,
    success: bool,
    reward: float | None,
    validity_class: str,
    failure_category: str | None,
    termination_reason: str,
    final_answer: str | None,
    tool_result: str | None,
    model_call_ids: list[str],
    harness_decision_ids: list[str],
) -> SmokeResult:
    trace.success = success
    trace.reward = reward
    trace.validity_class = validity_class
    trace.failure_category = failure_category
    trace.append(
        "reward_assigned",
        "evaluator",
        {
            "reward": reward,
            "task_success": int(success),
            "validity_class": validity_class,
            "failure_category": failure_category,
            "target_model_call_ids": model_call_ids,
            "target_harness_decision_ids": harness_decision_ids,
        },
    )
    trace.append(
        "episode_ended",
        "core",
        {
            "success": success,
            "termination_reason": termination_reason,
            "model_calls": len(model_call_ids),
            "harness_decisions": len(harness_decision_ids),
            "tool_calls": sum(event.kind == "tool_result" for event in trace.events),
        },
    )
    trace.validate()
    return SmokeResult(success, reward, final_answer, tool_result, trace)


def run_calculator_smoke(
    model: StructuredChatModel,
    task: CalculatorTask,
    controller: HarnessController,
    harness_spec: HarnessSpec,
    seed: int = 0,
    max_new_tokens: int = 96,
) -> SmokeResult:
    joint_version = JointVersion(
        policy=model.policy_version,
        harness_controller=controller.version,
        harness_artifact=harness_spec.version,
        tool_schema=harness_spec.tool_schema_version,
        parser=harness_spec.parser_version,
        environment="calculator-env-v2",
        evaluator=harness_spec.evaluator_version,
        tokenizer=model.tokenizer_version,
        context_builder=harness_spec.prompt_version,
    )
    trace = EpisodeTrace(
        episode_id=str(uuid.uuid4()),
        task_id=task.task_id,
        seed=seed,
        joint_version=joint_version,
        harness_spec_hash=harness_spec.fingerprint(),
    )
    model_call_ids: list[str] = []
    harness_decision_ids: list[str] = []
    trace.append(
        "episode_started",
        "core",
        {
            "task_id": task.task_id,
            "seed": seed,
            "joint_version_id": joint_version.version_id,
            "harness_spec_hash": trace.harness_spec_hash,
        },
    )

    initial_state = HarnessState(
        turn=0,
        remaining_tool_calls=harness_spec.max_tool_calls,
        remaining_model_retries=harness_spec.max_model_retries,
        context_chars=0,
        last_error=None,
        retrieval_hit=False,
        verifier_status="not-run",
        task_domain="calculator",
    )
    try:
        first_decision = controller.choose(initial_state)
        _record_decision(trace, first_decision, initial_state)
    except Exception as exc:  # controller implementations define their own errors
        trace.append(
            "controller_error",
            "harness",
            {"turn": 0, "exception_type": type(exc).__name__, "error": str(exc)},
        )
        return _finish(
            trace,
            success=False,
            reward=None,
            validity_class="infrastructure_invalid",
            failure_category="controller_infra",
            termination_reason=f"{type(exc).__name__}: {exc}",
            final_answer=None,
            tool_result=None,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )
    harness_decision_ids.append(first_decision.decision_id)
    if first_decision.controller_version != joint_version.harness_controller:
        return _finish(
            trace,
            success=False,
            reward=None,
            validity_class="trace_contract_invalid",
            failure_category="controller_version",
            termination_reason="controller version mismatch",
            final_answer=None,
            tool_result=None,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )
    if first_decision.action is not HarnessAction.DIRECT:
        return _finish(
            trace,
            success=False,
            reward=0.0,
            validity_class="policy_failure",
            failure_category="harness",
            termination_reason="DIRECT action required before tool request",
            final_answer=None,
            tool_result=None,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.question},
    ]
    tool_request: dict[str, object] | None = None
    for attempt in range(harness_spec.max_model_retries + 1):
        call_id, response, invalidity, error = _request_model(
            trace, model, "tool_request", attempt, messages, max_new_tokens
        )
        model_call_ids.append(call_id)
        if response is None:
            return _finish(
                trace,
                success=False,
                reward=None,
                validity_class=invalidity or "infrastructure_invalid",
                failure_category="model_contract" if invalidity == "trace_contract_invalid" else "infra",
                termination_reason=error or "model request failed",
                final_answer=None,
                tool_result=None,
                model_call_ids=model_call_ids,
                harness_decision_ids=harness_decision_ids,
            )
        try:
            candidate = _extract_json_object(response.text)
            if candidate.get("tool") != "calculator":
                raise ValueError("first action must call calculator")
            expression = candidate.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError("calculator expression must be a non-empty string")
            tool_request = candidate
            trace.append(
                "parse_result",
                "parser",
                {
                    "model_call_id": call_id,
                    "phase": "tool_request",
                    "valid": True,
                    "typed_action": "calculator",
                    "arguments": {"expression": expression},
                },
            )
            break
        except ValueError as exc:
            trace.append(
                "parse_result",
                "parser",
                {
                    "model_call_id": call_id,
                    "phase": "tool_request",
                    "valid": False,
                    "error_code": "INVALID_TOOL_ACTION",
                    "error": str(exc),
                },
            )
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {
                        "role": "user",
                        "content": "格式或动作不合法。只输出 calculator 调用 JSON，不要直接回答。",
                    },
                ]
            )
    if tool_request is None:
        return _finish(
            trace,
            success=False,
            reward=0.0,
            validity_class="policy_failure",
            failure_category="parser",
            termination_reason="no valid tool request within retry budget",
            final_answer=None,
            tool_result=None,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )

    expression = str(tool_request["expression"])
    try:
        tool_result = evaluate_expression(expression)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        trace.append(
            "tool_result",
            "tool",
            {
                "tool": "calculator",
                "expression": expression,
                "valid": False,
                "error_code": "CALCULATOR_REJECTED",
                "error": str(exc),
            },
        )
        return _finish(
            trace,
            success=False,
            reward=0.0,
            validity_class="policy_failure",
            failure_category="tool",
            termination_reason="calculator rejected model action",
            final_answer=None,
            tool_result=None,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )
    trace.append(
        "tool_result",
        "tool",
        {
            "tool": "calculator",
            "expression": expression,
            "result": tool_result,
            "result_hash": _text_hash(tool_result),
            "valid": True,
        },
    )

    verify_state = HarnessState(
        turn=1,
        remaining_tool_calls=harness_spec.max_tool_calls - 1,
        remaining_model_retries=harness_spec.max_model_retries,
        context_chars=sum(len(message["content"]) for message in messages),
        last_error=None,
        retrieval_hit=False,
        verifier_status="tool-ready",
        task_domain="calculator",
    )
    try:
        verify_decision = controller.choose(verify_state)
        _record_decision(trace, verify_decision, verify_state)
    except Exception as exc:  # controller implementations define their own errors
        trace.append(
            "controller_error",
            "harness",
            {"turn": 1, "exception_type": type(exc).__name__, "error": str(exc)},
        )
        return _finish(
            trace,
            success=False,
            reward=None,
            validity_class="infrastructure_invalid",
            failure_category="controller_infra",
            termination_reason=f"{type(exc).__name__}: {exc}",
            final_answer=None,
            tool_result=tool_result,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )
    harness_decision_ids.append(verify_decision.decision_id)
    if verify_decision.controller_version != joint_version.harness_controller:
        return _finish(
            trace,
            success=False,
            reward=None,
            validity_class="trace_contract_invalid",
            failure_category="controller_version",
            termination_reason="controller version mismatch",
            final_answer=None,
            tool_result=tool_result,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )
    if verify_decision.action is not HarnessAction.VERIFY:
        return _finish(
            trace,
            success=False,
            reward=0.0,
            validity_class="policy_failure",
            failure_category="harness",
            termination_reason="VERIFY action required after tool result",
            final_answer=None,
            tool_result=tool_result,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )

    verifier_passed = tool_result.strip() == task.expected_answer
    trace.append(
        "verifier_result",
        "verifier",
        {
            "candidate_hash": _text_hash(tool_result),
            "passed": verifier_passed,
            "evaluator_version": harness_spec.evaluator_version,
        },
    )
    if not verifier_passed:
        return _finish(
            trace,
            success=False,
            reward=0.0,
            validity_class="policy_failure",
            failure_category="verifier",
            termination_reason="tool result did not pass hidden verifier",
            final_answer=None,
            tool_result=tool_result,
            model_call_ids=model_call_ids,
            harness_decision_ids=harness_decision_ids,
        )

    messages.extend(
        [
            {"role": "assistant", "content": json.dumps(tool_request, ensure_ascii=False)},
            {
                "role": "user",
                "content": f"calculator 返回：{tool_result}\n现在只输出 final answer JSON。",
            },
        ]
    )
    final_answer: str | None = None
    for attempt in range(harness_spec.max_model_retries + 1):
        call_id, response, invalidity, error = _request_model(
            trace, model, "final_answer", attempt, messages, max_new_tokens
        )
        model_call_ids.append(call_id)
        if response is None:
            return _finish(
                trace,
                success=False,
                reward=None,
                validity_class=invalidity or "infrastructure_invalid",
                failure_category="model_contract" if invalidity == "trace_contract_invalid" else "infra",
                termination_reason=error or "model request failed",
                final_answer=None,
                tool_result=tool_result,
                model_call_ids=model_call_ids,
                harness_decision_ids=harness_decision_ids,
            )
        try:
            candidate = _extract_json_object(response.text)
            answer = candidate.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                raise ValueError("final response must contain a scalar answer")
            final_answer = str(answer).strip()
            trace.append(
                "parse_result",
                "parser",
                {
                    "model_call_id": call_id,
                    "phase": "final_answer",
                    "valid": True,
                    "typed_action": "final_answer",
                    "answer": final_answer,
                },
            )
            break
        except ValueError as exc:
            trace.append(
                "parse_result",
                "parser",
                {
                    "model_call_id": call_id,
                    "phase": "final_answer",
                    "valid": False,
                    "error_code": "INVALID_FINAL_ANSWER",
                    "error": str(exc),
                },
            )
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": "格式不合法。只输出 {\"answer\":\"工具结果\"}。"},
                ]
            )

    answer_is_correct = final_answer is not None and verify_answer(task, tool_result, final_answer)
    within_frozen_budget = len(model_call_ids) == 2
    success = answer_is_correct and within_frozen_budget
    if answer_is_correct and not within_frozen_budget:
        failure_category = "budget"
        termination_reason = "answer was correct but exceeded the frozen two-call budget"
    elif success:
        failure_category = None
        termination_reason = "success"
    elif final_answer is None:
        failure_category = "parser"
        termination_reason = "no valid final answer within retry budget"
    else:
        failure_category = "evaluator"
        termination_reason = "final answer failed exact evaluation"
    return _finish(
        trace,
        success=success,
        reward=1.0 if success else 0.0,
        validity_class="valid" if success else "policy_failure",
        failure_category=failure_category,
        termination_reason=termination_reason,
        final_answer=final_answer,
        tool_result=tool_result,
        model_call_ids=model_call_ids,
        harness_decision_ids=harness_decision_ids,
    )
