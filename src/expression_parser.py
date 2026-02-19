"""表达式与语法规则序列的双向转换器"""

import sympy as sp
from typing import List, Tuple, Optional
from nltk import Nonterminal

from src.grammar_sr import GCFG, Expr, Term, Factor, Atom, Unary, get_productions_by_nonterminal


# 运算符优先级映射
SYMPY_OP_TO_GRAMMAR = {
    sp.Add: ('Expr', 'Expr + Term'),
    sp.Mul: ('Term', 'Term * Factor'),
    sp.Pow: ('Factor', 'pow ( Expr , Expr )'),
    sp.sin: ('Unary', 'sin'),
    sp.cos: ('Unary', 'cos'),
    sp.tan: ('Unary', 'tan'),
    sp.exp: ('Unary', 'exp'),
    sp.log: ('Unary', 'log'),
    sp.sqrt: ('Unary', 'sqrt'),
    sp.asin: ('Unary', 'asin'),
    sp.acos: ('Unary', 'acos'),
    sp.atan: ('Unary', 'atan'),
    sp.sinh: ('Unary', 'sinh'),
    sp.cosh: ('Unary', 'cosh'),
    sp.tanh: ('Unary', 'tanh'),
    sp.Abs: ('Unary', 'abs'),
}

GRAMMAR_UNARY_TO_SYMPY = {
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'asin': sp.asin,
    'acos': sp.acos,
    'atan': sp.atan,
    'sinh': sp.sinh,
    'cosh': sp.cosh,
    'tanh': sp.tanh,
    'abs': sp.Abs,
}


class ExpressionParser:
    """表达式与语法规则序列的双向转换器"""

    def __init__(self):
        self.productions = list(GCFG.productions())

    def expression_to_rules(self, expr: str) -> List[int]:
        """将表达式字符串转换为规则序列索引

        Args:
            expr: 表达式字符串，如 "x0 + sin(x1)"

        Returns:
            规则索引列表
        """
        # 规范化表达式
        expr = self._normalize_expression(expr)

        # 解析为 sympy 表达式
        sympy_expr = sp.sympify(expr)

        # 转换为规则序列
        rules = []
        self._expr_to_rules(sympy_expr, 'Expr', rules)
        return rules

    def _expr_to_rules(self, sympy_expr, nonterminal: str, rules: List[int]):
        """递归将 sympy 表达式转换为规则序列"""
        if nonterminal == 'Expr':
            self._expr_to_rules_expr(sympy_expr, rules)
        elif nonterminal == 'Term':
            self._expr_to_rules_term(sympy_expr, rules)
        elif nonterminal == 'Factor':
            self._expr_to_rules_factor(sympy_expr, rules)
        elif nonterminal == 'Atom':
            self._expr_to_rules_atom(sympy_expr, rules)
        else:
            raise ValueError(f"Unknown nonterminal: {nonterminal}")

    def _expr_to_rules_expr(self, sympy_expr, rules: List[int]):
        """处理 Expr 非终结符"""
        # 检查是否是加法或减法
        if isinstance(sympy_expr, sp.Add):
            args = sympy_expr.as_ordered_terms()
            if len(args) >= 2:
                # 递归处理左操作数
                left = args[0]
                right = sp.Add(*args[1:]) if len(args) > 2 else args[1]

                self._expr_to_rules_expr(left, rules)
                # 添加 Expr → Expr + Term 规则
                rule_idx = self._get_rule_index("Expr -> Expr '+' Term")
                rules.append(rule_idx)
                self._expr_to_rules_term(right, rules)
                return

        # 检查是否是减法（通过 Add 中包含负数）
        if isinstance(sympy_expr, sp.Add) and any(
            isinstance(arg, sp.Mul) and arg.as_coeff_mul()[0] < 0
            for arg in sympy_expr.args
        ):
            args = sympy_expr.as_ordered_terms()
            if len(args) >= 2:
                left = args[0]
                right = -args[1]

                self._expr_to_rules_expr(left, rules)
                rule_idx = self._get_rule_index("Expr -> Expr '-' Term")
                rules.append(rule_idx)
                self._expr_to_rules_term(right, rules)
                return

        # 不是加减法，使用 Expr → Term
        rule_idx = self._get_rule_index("Expr -> Term")
        rules.append(rule_idx)
        self._expr_to_rules_term(sympy_expr, rules)

    def _expr_to_rules_term(self, sympy_expr, rules: List[int]):
        """处理 Term 非终结符"""
        if isinstance(sympy_expr, sp.Mul) and not any(
            isinstance(arg, sp.Pow) and arg.exp == sp.Rational(1, 2)
            for arg in sympy_expr.args
        ):
            args = sympy_expr.as_ordered_factors()
            if len(args) >= 2:
                left = args[0]
                right = sp.Mul(*args[1:]) if len(args) > 2 else args[1]

                self._expr_to_rules_term(left, rules)

                # 检查是否是除法
                if right.is_Number or (
                    hasattr(right, 'is_Pow') and
                    isinstance(right, sp.Pow) and
                    right.exp == -1
                ):
                    rule_idx = self._get_rule_index("Term -> Term '/' Factor")
                    rules.append(rule_idx)
                    if isinstance(right, sp.Pow) and right.exp == -1:
                        self._expr_to_rules_factor(right.base, rules)
                    else:
                        self._expr_to_rules_factor(right, rules)
                else:
                    rule_idx = self._get_rule_index("Term -> Term '*' Factor")
                    rules.append(rule_idx)
                    self._expr_to_rules_factor(right, rules)
                return

        # 检查是否是除法
        if isinstance(sympy_expr, sp.Pow) and sympy_expr.exp == -1:
            rule_idx = self._get_rule_index("Term -> Term '/' Factor")
            rules.append(rule_idx)
            self._expr_to_rules_factor(sympy_expr.base, rules)
            return

        # 不是乘除法，使用 Term → Factor
        rule_idx = self._get_rule_index("Term -> Factor")
        rules.append(rule_idx)
        self._expr_to_rules_factor(sympy_expr, rules)

    def _expr_to_rules_factor(self, sympy_expr, rules: List[int]):
        """处理 Factor 非终结符"""
        # 检查是否是幂运算
        if isinstance(sympy_expr, sp.Pow) and sympy_expr.exp != -1:
            rule_idx = self._get_rule_index("Factor -> pow '(' Expr ',' Expr ')'")
            rules.append(rule_idx)
            self._expr_to_rules(sympy_expr.base, 'Expr', rules)
            self._expr_to_rules(sympy_expr.exp, 'Expr', rules)
            return

        # 检查是否是一元函数
        if sympy_expr.func in SYMPY_OP_TO_GRAMMAR:
            nt, rule_str = SYMPY_OP_TO_GRAMMAR[sympy_expr.func]
            if nt == 'Unary':
                rule_idx = self._get_rule_index(f"Factor -> Unary Expr")
                rules.append(rule_idx)
                unary_rule_idx = self._get_rule_index(f"Unary -> '{sympy_expr.func.__name__}'")
                rules.append(unary_rule_idx)
                self._expr_to_rules(sympy_expr.args[0], 'Expr', rules)
                return

        # 检查是否是括号表达式
        if sympy_expr.is_Atom or isinstance(sympy_expr, sp.Symbol):
            rule_idx = self._get_rule_index("Factor -> Atom")
            rules.append(rule_idx)
            self._expr_to_rules_atom(sympy_expr, rules)
            return

        # 默认使用括号
        rule_idx = self._get_rule_index("Factor -> '(' Expr ')'")
        rules.append(rule_idx)
        self._expr_to_rules(sympy_expr, 'Expr', rules)

    def _expr_to_rules_atom(self, sympy_expr, rules: List[int]):
        """处理 Atom 非终结符"""
        if isinstance(sympy_expr, sp.Symbol):
            var_name = str(sympy_expr)
            if var_name.startswith('x') and var_name[1:].isdigit():
                rule_idx = self._get_rule_index(f"Atom -> '{var_name}'")
                rules.append(rule_idx)
                return

        # 处理常数
        if sympy_expr.is_Number:
            val = float(sympy_expr)
            if val == 0:
                rule_idx = self._get_rule_index("Atom -> '0'")
            elif val == 1:
                rule_idx = self._get_rule_index("Atom -> '1'")
            elif val == 2:
                rule_idx = self._get_rule_index("Atom -> '2'")
            elif val == 3:
                rule_idx = self._get_rule_index("Atom -> '3'")
            elif val == 4:
                rule_idx = self._get_rule_index("Atom -> '4'")
            elif val == 5:
                rule_idx = self._get_rule_index("Atom -> '5'")
            elif val == 6:
                rule_idx = self._get_rule_index("Atom -> '6'")
            elif val == 7:
                rule_idx = self._get_rule_index("Atom -> '7'")
            elif val == 8:
                rule_idx = self._get_rule_index("Atom -> '8'")
            elif val == 9:
                rule_idx = self._get_rule_index("Atom -> '9'")
            elif abs(val - 0.1) < 1e-10:
                rule_idx = self._get_rule_index("Atom -> '0.1'")
            elif abs(val - 0.5) < 1e-10:
                rule_idx = self._get_rule_index("Atom -> '0.5'")
            elif abs(val - 3.14159) < 1e-5:
                rule_idx = self._get_rule_index("Atom -> 'pi'")
            elif abs(val - 2.71828) < 1e-5:
                rule_idx = self._get_rule_index("Atom -> 'e'")
            else:
                # 对于其他数值，尝试转换为简单形式
                if abs(val - int(val)) < 1e-10:
                    rule_idx = self._get_rule_index(f"Atom -> '{int(val)}'")
                else:
                    rule_idx = self._get_rule_index(f"Atom -> '{val}'")
            rules.append(rule_idx)
            return

        # 处理特殊常数
        if sympy_expr == sp.pi:
            rule_idx = self._get_rule_index("Atom -> 'pi'")
            rules.append(rule_idx)
            return
        if sympy_expr == sp.E:
            rule_idx = self._get_rule_index("Atom -> 'e'")
            rules.append(rule_idx)
            return

        raise ValueError(f"Cannot convert expression to atom: {sympy_expr}")

    def _get_rule_index(self, rule_str: str) -> int:
        """获取规则的索引"""
        for i, p in enumerate(self.productions):
            if str(p) == rule_str.replace("'", "").replace("(", "").replace(")", "").replace(",", ", "):
                return i
        # 尝试更宽松的匹配
        for i, p in enumerate(self.productions):
            if rule_str.split(" -> ")[1].strip() in str(p.rhs()):
                return i
        raise ValueError(f"Rule not found: {rule_str}")

    def _normalize_expression(self, expr: str) -> str:
        """规范化表达式字符串"""
        expr = expr.strip()
        # 替换 ** 为 pow
        expr = expr.replace('**', '^')
        # 处理 sin(x) 等
        # 将 exp(x) 转换为 e^x
        # 处理除法
        return expr

    def rules_to_expression(self, rules: List[int]) -> str:
        """将规则序列转换为表达式字符串

        Args:
            rules: 规则索引列表

        Returns:
            表达式字符串
        """
        stack = [Expr]
        productions = list(GCFG.productions())

        for rule_idx in rules:
            if not stack:
                break
            nt = stack.pop(0)
            rule = productions[rule_idx]
            # 将 RHS 的非终结符按相反顺序添加到栈中
            for symbol in reversed(rule.rhs()):
                if isinstance(symbol, Nonterminal):
                    stack.insert(0, symbol)

        # 从规则序列构建表达式（简化版本）
        # 实际实现需要更复杂的解析逻辑
        return self._build_from_rules(rules)

    def _build_from_rules(self, rules: List[int]) -> str:
        """从规则序列构建表达式字符串"""
        productions = list(GCFG.productions())
        stack = []

        for rule_idx in rules:
            rule = productions[rule_idx]
            rhs = str(rule.rhs())[1:-1]  # 去掉括号

            if rhs in ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       '0.1', '0.5', 'pi', 'e']:
                stack.append(rhs)
            elif rhs == '+' and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(f'({a}+{b})')
            elif rhs == '-' and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(f'({a}-{b})')
            elif rhs == '*' and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(f'({a}*{b})')
            elif rhs == '/' and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(f'({a}/{b})')
            elif rhs in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'abs']:
                if stack:
                    a = stack.pop()
                    stack.append(f'{rhs}({a})')
            elif rhs == 'pow ( Expr , Expr )':
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(f'({a}**{b})')

        if stack:
            result = stack[-1]
            # 尝试用 sympy 简化
            try:
                simplified = str(sp.simplify(result))
                return simplified
            except:
                return result
        return ""


# 预定义规则掩码，用于快速查找
EXPR_RULES = [0, 1, 2]  # Expr → Expr + Term, Expr → Expr - Term, Expr → Term
TERM_RULES = [3, 4, 5]  # Term → Term * Factor, Term → Term / Factor, Term → Factor


def get_rule_masks():
    """获取各非终结符的规则掩码"""
    productions = list(GCFG.productions())

    masks = {
        'Expr': [i for i, p in enumerate(productions) if str(p.lhs()) == 'Expr'],
        'Term': [i for i, p in enumerate(productions) if str(p.lhs()) == 'Term'],
        'Factor': [i for i, p in enumerate(productions) if str(p.lhs()) == 'Factor'],
        'Atom': [i for i, p in enumerate(productions) if str(p.lhs()) == 'Atom'],
        'Unary': [i for i, p in enumerate(productions) if str(p.lhs()) == 'Unary'],
    }
    return masks


if __name__ == '__main__':
    parser = ExpressionParser()

    # 测试表达式转规则
    test_exprs = [
        "x0 + x1",
        "sin(x0)",
        "x0 * x1 + 1",
        "x0 / (x1 + 1)",
        "sqrt(x0**2 + x1**2)",
    ]

    for expr in test_exprs:
        try:
            rules = parser.expression_to_rules(expr)
            print(f"{expr} -> {rules}")
        except Exception as e:
            print(f"Error parsing {expr}: {e}")
