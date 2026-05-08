"""安全算术工具：仅允许数字与 + - * / 括号，禁止任意代码执行。"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_num(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return float(_ALLOWED_UNARY[type(node.op)](_eval_num(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return float(
            _ALLOWED_BINOPS[type(node.op)](_eval_num(node.left), _eval_num(node.right))
        )
    raise ValueError("表达式包含不允许的语法（仅支持数字与 + - * / ** % 与括号）")


@tool
def safe_calculator(expression: str) -> str:
    """对纯算术表达式求值，例如 (1+2)*3、2**10。禁止变量名与函数调用。"""
    expr = (expression or "").strip()
    if not expr:
        return "（空表达式）"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"语法错误: {e}"
    if not isinstance(tree.body, ast.Expression):
        return "（不支持的表达式结构）"
    try:
        val = _eval_num(tree.body)
    except Exception as e:  # noqa: BLE001 — 向模型返回可读错误
        return f"无法求值: {e}"
    return str(val)
