from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import operator


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPRESSION_CHARS = 128
_MAX_AST_DEPTH = 16
_MAX_LITERAL_DIGITS = 32
_MAX_RESULT_BITS = 256


@dataclass(frozen=True)
class CalculatorTask:
    task_id: str
    question: str
    expected_answer: str


TASKS = {
    "add-17-25": CalculatorTask(
        task_id="add-17-25",
        question="计算 17 与 25 的和。必须先调用 calculator，再根据工具结果回答。",
        expected_answer="42",
    ),
    "multiply-13-7": CalculatorTask(
        task_id="multiply-13-7",
        question="计算 13 与 7 的乘积。必须先调用 calculator，再根据工具结果回答。",
        expected_answer="91",
    ),
}


def _bounded(value: Fraction) -> Fraction:
    if (
        abs(value.numerator).bit_length() > _MAX_RESULT_BITS
        or value.denominator.bit_length() > _MAX_RESULT_BITS
    ):
        raise OverflowError("calculator result exceeds the safety bound")
    return value


def _evaluate_node(node: ast.AST, depth: int = 0) -> Fraction:
    if depth > _MAX_AST_DEPTH:
        raise ValueError("calculator expression is too deeply nested")
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("calculator accepts integer literals only")
        if len(str(abs(node.value))) > _MAX_LITERAL_DIGITS:
            raise ValueError("integer literal exceeds the safety bound")
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, depth + 1)
        right = _evaluate_node(node.right, depth + 1)
        return _bounded(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _bounded(_UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, depth + 1)))
    raise ValueError(f"forbidden calculator syntax: {type(node).__name__}")


def evaluate_expression(expression: str) -> str:
    if not expression.strip():
        raise ValueError("calculator expression cannot be empty")
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ValueError("calculator expression is too long")
    value = _evaluate_node(ast.parse(expression, mode="eval"))
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def verify_answer(task: CalculatorTask, tool_result: str, final_answer: str) -> bool:
    return tool_result.strip() == task.expected_answer and final_answer.strip() == tool_result.strip()
