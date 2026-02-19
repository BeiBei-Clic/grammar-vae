"""扩展的符号回归语法规则系统

支持:
- 多变量 x0-x9
- 运算符: +, -, *, /, pow
- 函数: sin, cos, tan, exp, log, sqrt, asin, acos, atan, sinh, cosh, tanh, abs
- 常数: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0.1, 0.5, pi, e
"""

import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal


# 语法规则定义
grammar_rules = [
    # Expr → Expr + Term (0)
    "Expr -> Expr '+' Term",
    # Expr → Expr - Term (1)
    "Expr -> Expr '-' Term",
    # Expr → Term (2)
    "Expr -> Term",
    # Term → Term * Factor (3)
    "Term -> Term '*' Factor",
    # Term → Term / Factor (4)
    "Term -> Term '/' Factor",
    # Term → Factor (5)
    "Term -> Factor",
    # Factor → pow (6)
    "Factor -> pow '(' Expr ',' Expr ')'",
    # Factor → ( Expr ) (7)
    "Factor -> '(' Expr ')'",
    # Factor → unary Expr (8)
    "Factor -> Unary Expr",
    # Unary → sin (9)
    "Unary -> 'sin'",
    # Unary → cos (10)
    "Unary -> 'cos'",
    # Unary → tan (11)
    "Unary -> 'tan'",
    # Unary → exp (12)
    "Unary -> 'exp'",
    # Unary → log (13)
    "Unary -> 'log'",
    # Unary → sqrt (14)
    "Unary -> 'sqrt'",
    # Unary → asin (15)
    "Unary -> 'asin'",
    # Unary → acos (16)
    "Unary -> 'acos'",
    # Unary → atan (17)
    "Unary -> 'atan'",
    # Unary → sinh (18)
    "Unary -> 'sinh'",
    # Unary → cosh (19)
    "Unary -> 'cosh'",
    # Unary → tanh (20)
    "Unary -> 'tanh'",
    # Unary → abs (21)
    "Unary -> 'abs'",
    # Factor → Atom (22)
    "Factor -> Atom",
    # Atom → x0 (23)
    "Atom -> 'x0'",
    # Atom → x1 (24)
    "Atom -> 'x1'",
    # Atom → x2 (25)
    "Atom -> 'x2'",
    # Atom → x3 (26)
    "Atom -> 'x3'",
    # Atom → x4 (27)
    "Atom -> 'x4'",
    # Atom → x5 (28)
    "Atom -> 'x5'",
    # Atom → x6 (29)
    "Atom -> 'x6'",
    # Atom → x7 (30)
    "Atom -> 'x7'",
    # Atom → x8 (31)
    "Atom -> 'x8'",
    # Atom → x9 (32)
    "Atom -> 'x9'",
    # Atom → 0 (33)
    "Atom -> '0'",
    # Atom → 1 (34)
    "Atom -> '1'",
    # Atom → 2 (35)
    "Atom -> '2'",
    # Atom → 3 (36)
    "Atom -> '3'",
    # Atom → 4 (37)
    "Atom -> '4'",
    # Atom → 5 (38)
    "Atom -> '5'",
    # Atom → 6 (39)
    "Atom -> '6'",
    # Atom → 7 (40)
    "Atom -> '7'",
    # Atom → 8 (41)
    "Atom -> '8'",
    # Atom → 9 (42)
    "Atom -> '9'",
    # Atom → 0.1 (43)
    "Atom -> '0.1'",
    # Atom → 0.5 (44)
    "Atom -> '0.5'",
    # Atom → pi (45)
    "Atom -> 'pi'",
    # Atom → e (46)
    "Atom -> 'e'",
]

NUM_RULES = len(grammar_rules)

grammar_str = "\n".join(grammar_rules)
GCFG = CFG.fromstring(grammar_str)

# 非终结符
Expr = Nonterminal('Expr')
Term = Nonterminal('Term')
Factor = Nonterminal('Factor')
Atom = Nonterminal('Atom')
Unary = Nonterminal('Unary')


def get_mask(nonterminal, grammar=GCFG, as_variable=False):
    """获取给定非终结符的掩码，指示哪些规则可以扩展该非终结符"""
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    elif isinstance(nonterminal, str):
        nt = Nonterminal(nonterminal)
        mask = [rule.lhs() == nt for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal or string')


def get_productions_by_nonterminal(nonterminal, grammar=GCFG):
    """获取可以扩展给定非终结符的所有产生式规则"""
    nt = Nonterminal(nonterminal) if isinstance(nonterminal, str) else nonterminal
    return [p for p in grammar.productions() if p.lhs() == nt]


def get_rule_index(rule):
    """获取规则的索引"""
    productions = list(GCFG.productions())
    return productions.index(rule)


# 规则到字符串的映射（用于调试）
RULE_STRINGS = grammar_rules


if __name__ == '__main__':
    print(f"Total rules: {NUM_RULES}")
    print(f"Expr mask: {get_mask(Expr)}")
    print(f"Term mask: {get_mask(Term)}")
    print(f"Factor mask: {get_mask(Factor)}")
    print(f"Atom mask: {get_mask(Atom)}")
