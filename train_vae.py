"""GrammarVAE 训练脚本

生成随机表达式并训练 VAE 模型
"""

import argparse
import random
import logging
from pathlib import Path

import torch
import numpy as np
import sympy as sp
from nltk import Nonterminal

from src.sr_model import GrammarVAEModel
from src.grammar_sr import GCFG, Expr, get_productions_by_nonterminal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 预定义的有效表达式模板
EXPRESSION_TEMPLATES = [
    # 简单表达式
    "x0", "x1", "x2",
    "x0 + x1", "x0 * x1", "x0 / x1", "x0 - x1",
    "x0 + 1", "x0 * 2", "x0 / 2",
    # 三角函数
    "sin(x0)", "cos(x0)", "tan(x0)",
    "sin(x0 + x1)", "cos(x0 * x1)",
    # 指数和对数
    "exp(x0)", "log(x0)", "sqrt(x0)",
    "exp(x0 + x1)", "sqrt(x0**2 + x1**2)",
    # 复杂表达式
    "x0**2", "x0**2 + x1**2",
    "x0 / (x1 + 1)", "x0 * x1 / x2",
    "sin(x0) + cos(x1)",
    "x0 * sin(x1)", "x0 / sqrt(x1**2 + 1)",
    "x0 + x1 * x2", "x0 * x1 + x2",
    # 幂运算
    "x0**2 + x1**2", "x0**2 - x1**2",
    "sqrt(x0**2 + x1**2)",
    # 双曲函数
    "sinh(x0)", "cosh(x0)", "tanh(x0)",
    # 反三角函数
    "asin(x0)", "acos(x0)", "atan(x0)",
    # 绝对值
    "abs(x0)", "abs(x0 - x1)",
    # 组合
    "sin(x0) * cos(x1)", "x0 * exp(-x1)",
    "x0 / (1 + x1**2)", "sqrt(1 - x0**2)",
    "x0 * x1 + x0 * x2",
    # 更多复杂组合
    "(x0 + x1) / x2", "x0 / (x1 + x2)",
    "sin(x0 + x1)", "cos(x0 * x1)",
    "x0**2 + x1", "x0 + x1**2",
    "1 / (x0 + 1)", "x0 / (x0 + x1)",
]


def generate_random_expression():
    """随机生成一个表达式"""
    template = random.choice(EXPRESSION_TEMPLATES)
    # 随机替换变量名
    expr = template
    for i in range(3):
        old_var = f"x{i}"
        new_var = f"x{random.randint(0, 9)}"
        expr = expr.replace(old_var, new_var)
    return expr


def generate_training_data(n_samples):
    """生成训练数据"""
    expressions = []
    for i in range(n_samples):
        expr = generate_random_expression()
        expressions.append(expr)
        if (i + 1) % 1000 == 0:
            logger.info(f"Generated {i+1}/{n_samples} expressions")
    return expressions


def train_vae(model, expressions, n_epochs=100, batch_size=32, lr=1e-3, kl_weight=0.1):
    """训练 VAE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 准备数据
    onehot_list = []
    valid_exprs = []
    for expr in expressions:
        try:
            onehot = model.expression_to_onehot(expr)
            onehot_list.append(onehot.squeeze(0))
            valid_exprs.append(expr)
        except Exception as e:
            pass

    if not onehot_list:
        logger.warning("No valid expressions for training")
        return

    dataset = torch.stack(onehot_list, dim=0)
    logger.info(f"Training with {len(dataset)} valid expressions")

    n_batches = max(len(dataset) // batch_size, 1)
    best_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        indices = torch.randperm(len(dataset))

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = dataset[batch_indices]

            # 前向传播
            logits, mu, sigma = model.forward(batch)

            # 计算损失
            target = batch.argmax(dim=1)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target.view(-1)

            recon_loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
            kl = model.kl_loss(mu, sigma)
            loss = recon_loss + kl_weight * kl

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl.item()

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_kl = epoch_kl_loss / n_batches

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/vae_best.pt')
            logger.info(f"Saved best model with loss {best_loss:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/vae_final.pt')
    logger.info("Training complete. Model saved.")


def parse_args():
    parser = argparse.ArgumentParser(description='Train GrammarVAE')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of training expressions to generate')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='KL divergence weight')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing model checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()

    Path('checkpoints').mkdir(exist_ok=True)

    model = GrammarVAEModel(z_dim=args.z_dim)

    if args.load:
        logger.info(f"Loading model from {args.load}")
        model.load_state_dict(torch.load(args.load))

    logger.info(f"Generating {args.n_samples} training expressions...")
    expressions = generate_training_data(args.n_samples)

    logger.info("Sample expressions:")
    for expr in random.sample(expressions, min(10, len(expressions))):
        logger.info(f"  {expr}")

    logger.info("Starting training...")
    train_vae(model, expressions,
              n_epochs=args.n_epochs,
              batch_size=args.batch_size,
              lr=args.lr,
              kl_weight=args.kl_weight)


if __name__ == '__main__':
    main()
