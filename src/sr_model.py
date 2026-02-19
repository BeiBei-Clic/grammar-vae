"""GrammarVAE 符号回归模型

整合编码器、解码器、表达式解析器等组件，实现符号回归功能
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import List, Tuple, Dict, Optional, Callable
from nltk import Nonterminal
import logging

from src.encoder_sr import Encoder
from src.decoder_sr import Decoder
from src.grammar_sr import GCFG, Expr, get_mask, NUM_RULES
from src.expression_parser import ExpressionParser, get_rule_masks
from src.metrics import evaluate_expression_string, compute_rmse, compute_r2
from src.stack import Stack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrammarVAEModel(nn.Module):
    """GrammarVAE 符号回归模型"""

    def __init__(self, z_dim: int = 32, hidden_encoder_dim: int = 50,
                 hidden_decoder_dim: int = 100, max_length: int = 50):
        super(GrammarVAEModel, self).__init__()
        self.z_dim = z_dim
        self.max_length = max_length
        self.encoder = Encoder(hidden_dim=hidden_encoder_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, hidden_size=hidden_decoder_dim)
        self.parser = ExpressionParser()
        self.rule_masks = get_rule_masks()

        # 用于生成
        self.grammar = GCFG
        self.start_symbol = Expr

    def encode(self, rules_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码规则序列

        Args:
            rules_onehot: shape [batch, NUM_RULES, max_length]

        Returns:
            mu, sigma: 潜在分布的参数
        """
        return self.encoder(rules_onehot)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在向量

        Args:
            z: shape [batch, z_dim]

        Returns:
            logits: shape [batch, max_length, NUM_RULES]
        """
        return self.decoder(z, max_length=self.max_length)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: shape [batch, NUM_RULES, max_length]

        Returns:
            logits, mu, sigma
        """
        mu, sigma = self.encode(x)
        z = self.sample(mu, sigma)
        logits = self.decode(z)
        return logits, mu, sigma

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """重参数化采样"""
        eps = torch.randn_like(sigma)
        return mu + eps * torch.sqrt(sigma)

    def kl_loss(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """计算 KL 散度"""
        return -0.5 * torch.sum(1 + torch.log(sigma + 1e-10) - mu.pow(2) - sigma, dim=1).mean()

    def generate_from_z(self, z: torch.Tensor, sample: bool = False) -> str:
        """从潜在向量 z 生成表达式字符串

        Args:
            z: shape [1, z_dim] 或 [z_dim]
            sample: 是否使用采样，否则使用贪心解码

        Returns:
            表达式字符串
        """
        self.eval()
        with torch.no_grad():
            if z.dim() == 1:
                z = z.unsqueeze(0)

            logits = self.decode(z).squeeze(0)  # [max_length, NUM_RULES]

            # 使用栈式解析生成表达式
            stack = Stack(grammar=self.grammar, start_symbol=self.start_symbol)
            rules = []
            t = 0

            while stack.nonempty and t < self.max_length:
                alpha = stack.pop()
                mask = get_mask(alpha, self.grammar, as_variable=False)
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=z.device)

                # 应用掩码
                logit = logits[t] + (1 - mask_tensor) * -1e9
                probs = torch.softmax(logit, dim=-1)

                if sample:
                    m = Categorical(probs)
                    i = m.sample()
                else:
                    i = torch.argmax(probs)

                i = i.item()
                rule = self.grammar.productions()[i]
                rules.append(i)

                # 将 RHS 非终结符加入栈
                for symbol in reversed(rule.rhs()):
                    if isinstance(symbol, Nonterminal):
                        stack.push(symbol)

                t += 1

            # 从规则序列构建表达式
            return self._rules_to_expression(rules)

    def _rules_to_expression(self, rules: List[int]) -> str:
        """从规则序列构建表达式字符串"""
        # 使用更精确的解析方法
        productions = list(self.grammar.productions())

        # 收集所有终结符
        tokens = []
        for rule_idx in rules:
            rule = productions[rule_idx]
            for symbol in rule.rhs():
                if not isinstance(symbol, Nonterminal):
                    # 是终结符
                    if isinstance(symbol, str):
                        tokens.append(symbol)
                    else:
                        tokens.append(str(symbol))

        # 构建表达式字符串
        expr_str = ''.join(tokens)

        # 清理表达式（去除多余的括号）
        try:
            import sympy as sp
            expr = sp.sympify(expr_str)
            return str(expr)
        except:
            return "x0"

    def expression_to_onehot(self, expr: str) -> torch.Tensor:
        """将表达式字符串转换为 one-hot 编码的规则序列

        Args:
            expr: 表达式字符串

        Returns:
            onehot: shape [1, NUM_RULES, max_length]
        """
        rules = self.parser.expression_to_rules(expr)
        max_len = min(len(rules), self.max_length)

        onehot = torch.zeros(1, NUM_RULES, self.max_length)
        for i, rule_idx in enumerate(rules[:max_len]):
            onehot[0, rule_idx, i] = 1.0

        return onehot

    def search_expression(self, X: np.ndarray, y: np.ndarray,
                          n_iterations: int = 1000,
                          method: str = 'random',
                          seed: int = 42) -> Tuple[str, Dict]:
        """在潜在空间中搜索最优表达式

        Args:
            X: 训练数据
            y: 目标值
            n_iterations: 迭代次数
            method: 搜索方法 ('random', 'cmaes', 'genetic')
            seed: 随机种子

        Returns:
            最佳表达式字符串和结果字典
        """
        rng = np.random.default_rng(seed)
        best_expr = "x0"
        best_rmse = float('inf')
        best_r2 = -float('inf')
        evolution_rmse = []

        var_names = [f'x{i}' for i in range(X.shape[1])]

        if method == 'random':
            for i in range(n_iterations):
                # 随机采样潜在向量
                z = torch.randn(1, self.z_dim)
                expr = self.generate_from_z(z, sample=True)

                result = evaluate_expression_string(expr, X, y, var_names)
                rmse = result['rmse']
                r2 = result['r2']

                evolution_rmse.append(rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_r2 = r2
                    best_expr = expr

                if (i + 1) % 100 == 0:
                    logger.info(f"Iteration {i+1}/{n_iterations}: Best RMSE = {best_rmse:.6f}")

        elif method == 'cmaes':
            try:
                import cma
            except ImportError:
                logger.warning("cma package not installed, falling back to random search")
                return self.search_expression(X, y, n_iterations, method='random', seed=seed)

            def fitness_func(z_values):
                z = torch.tensor(z_values.reshape(1, -1), dtype=torch.float32)
                expr = self.generate_from_z(z, sample=False)
                result = evaluate_expression_string(expr, X, y, var_names)
                return result['rmse'] if result['valid'] else 1e6

            # CMA-ES 优化
            x0 = rng.standard_normal(self.z_dim)
            sigma0 = 1.0

            try:
                es = cma.CMAEvolutionStrategy(x0, sigma0,
                                            {'popsize': min(32, 4 + int(3 * np.log(self.z_dim))),
                                             'maxiter': n_iterations})

                for i in range(n_iterations):
                    solutions = es.ask()
                    fitnesses = [fitness_func(x) for x in solutions]
                    es.tell(solutions, fitnesses)

                    best_rmse = es.result.fbest
                    evolution_rmse.append(best_rmse)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Iteration {i+1}/{n_iterations}: Best RMSE = {best_rmse:.6f}")

                    if es.stop():
                        break

                # 获取最佳解
                best_z = es.result.xbest
                best_expr = self.generate_from_z(torch.tensor(best_z).unsqueeze(0), sample=False)
                best_result = evaluate_expression_string(best_expr, X, y, var_names)
                best_rmse = best_result['rmse']
                best_r2 = best_result['r2']

            except Exception as e:
                logger.warning(f"CMA-ES failed: {e}, falling back to random search")
                return self.search_expression(X, y, n_iterations, method='random', seed=seed)

        return best_expr, {
            'train_rmse': best_rmse,
            'train_r2': best_r2,
            'evolution_rmse': evolution_rmse
        }

    def train_vae(self, expressions: List[str], n_epochs: int = 100,
                  batch_size: int = 32, lr: float = 1e-3):
        """训练 VAE

        Args:
            expressions: 训练表达式列表
            n_epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 准备数据
        onehot_list = []
        for expr in expressions:
            try:
                onehot = self.expression_to_onehot(expr)
                onehot_list.append(onehot)
            except:
                pass

        if not onehot_list:
            logger.warning("No valid expressions for training")
            return

        dataset = torch.cat(onehot_list, dim=0)

        for epoch in range(n_epochs):
            indices = torch.randperm(len(dataset))
            epoch_loss = 0

            for i in range(0, len(dataset), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = dataset[batch_indices]

                logits, mu, sigma = self.forward(batch)

                # 重构损失
                target = batch.argmax(dim=1)
                logits_flat = logits.view(-1, NUM_RULES)
                target_flat = target.view(-1)

                recon_loss = nn.CrossEntropyLoss()(logits_flat, target_flat)
                kl = self.kl_loss(mu, sigma)
                loss = recon_loss + 0.1 * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss = {epoch_loss/(len(dataset)//batch_size):.4f}")


if __name__ == '__main__':
    # 测试模型
    model = GrammarVAEModel(z_dim=16, max_length=30)

    # 测试生成
    z = torch.randn(1, 16)
    expr = model.generate_from_z(z)
    print(f"Generated expression: {expr}")

    # 测试表达式编码
    test_expr = "x0 + sin(x1)"
    onehot = model.expression_to_onehot(test_expr)
    print(f"Onehot shape: {onehot.shape}")

    # 测试搜索
    X = np.random.randn(100, 2)
    y = X[:, 0] + np.sin(X[:, 1])

    best_expr, result = model.search_expression(X, y, n_iterations=50, method='random')
    print(f"Best expression: {best_expr}")
    print(f"Result: {result}")
