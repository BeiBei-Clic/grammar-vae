"""评估指标模块

计算符号回归模型的性能指标
"""

import numpy as np
import sympy as sp
from typing import Dict, Callable, Optional, Tuple
import warnings


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差"""
    mask = np.isfinite(y_pred)
    if not mask.any():
        return float('inf')
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    return np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 R² 分数"""
    mask = np.isfinite(y_pred)
    if not mask.any():
        return -float('inf')
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -float('inf')

    return 1 - (ss_res / ss_tot)


def compile_expression(expr_str: str, var_names: list) -> Optional[Callable]:
    """将表达式字符串编译为可执行函数

    Args:
        expr_str: 表达式字符串
        var_names: 变量名列表，如 ['x0', 'x1']

    Returns:
        可执行函数，如果编译失败则返回 None
    """
    try:
        # 规范化表达式
        expr_str = expr_str.replace('^', '**')

        # 创建符号变量
        symbols = [sp.Symbol(v) for v in var_names]

        # 解析表达式
        expr = sp.sympify(expr_str)

        # 编译为 lambda 函数
        func = sp.lambdify(symbols, expr, modules='numpy')
        return func
    except Exception as e:
        warnings.warn(f"Failed to compile expression '{expr_str}': {e}")
        return None


def evaluate_expression(expr_func: Callable, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """评估表达式在数据上的性能

    Args:
        expr_func: 可执行函数
        X: 输入数据，shape [n_samples, n_features]
        y_true: 真实标签

    Returns:
        包含 rmse 和 r2 的字典
    """
    try:
        # 准备参数
        n_features = X.shape[1]

        if n_features == 1:
            y_pred = expr_func(X[:, 0])
        else:
            # 将每列作为单独的参数传递
            args = [X[:, i] for i in range(n_features)]
            y_pred = expr_func(*args)

        # 确保返回值是 numpy 数组
        y_pred = np.array(y_pred, dtype=np.float64)

        # 处理标量情况
        if y_pred.shape == ():
            y_pred = np.full_like(y_true, y_pred)

        rmse = compute_rmse(y_true, y_pred)
        r2 = compute_r2(y_true, y_pred)

        return {
            'rmse': rmse,
            'r2': r2,
            'valid': not np.isinf(rmse) and not np.isnan(rmse)
        }
    except Exception as e:
        warnings.warn(f"Error evaluating expression: {e}")
        return {
            'rmse': float('inf'),
            'r2': -float('inf'),
            'valid': False
        }


def evaluate_expression_string(expr_str: str, X: np.ndarray, y_true: np.ndarray,
                                var_names: Optional[list] = None) -> Dict[str, float]:
    """评估表达式字符串在数据上的性能

    Args:
        expr_str: 表达式字符串
        X: 输入数据
        y_true: 真实标签
        var_names: 变量名列表，如果为 None 则自动生成为 ['x0', 'x1', ...]

    Returns:
        包含 rmse 和 r2 的字典
    """
    if var_names is None:
        var_names = [f'x{i}' for i in range(X.shape[1])]

    func = compile_expression(expr_str, var_names)
    if func is None:
        return {
            'rmse': float('inf'),
            'r2': -float('inf'),
            'valid': False
        }

    return evaluate_expression(func, X, y_true)


def normalize_expression(expr_str: str) -> str:
    """规范化表达式字符串

    Args:
        expr_str: 原始表达式字符串

    Returns:
        规范化后的表达式字符串
    """
    expr = sp.sympify(expr_str.replace('^', '**'))
    return str(expr.simplify())


if __name__ == '__main__':
    # 测试评估指标
    X_test = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_true = np.array([3.0, 5.0, 7.0])

    # 测试 1: 简单加法 x0 + x1
    result = evaluate_expression_string("x0 + x1", X_test, y_true)
    print(f"x0 + x1: {result}")

    # 测试 2: 带常数 x0 * x1 + 1
    y_true2 = np.array([3.0, 7.0, 13.0])
    result2 = evaluate_expression_string("x0 * x1 + 1", X_test, y_true2)
    print(f"x0 * x1 + 1: {result2}")

    # 测试 3: 三角函数 sin(x0)
    y_true3 = np.sin(X_test[:, 0])
    result3 = evaluate_expression_string("sin(x0)", X_test, y_true3)
    print(f"sin(x0): {result3}")
