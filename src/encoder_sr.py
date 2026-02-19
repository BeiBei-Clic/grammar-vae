"""编码器模块 - 将规则序列编码为潜在向量 z"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.grammar_sr import NUM_RULES


class Encoder(nn.Module):
    """卷积编码器，将规则序列编码为潜在分布"""

    def __init__(self, hidden_dim=50, z_dim=32):
        super(Encoder, self).__init__()

        # 调整卷积层以适应更大的规则集
        self.conv1 = nn.Conv1d(NUM_RULES, 32, kernel_size=2)
        self.conv2 = nn.Conv1d(32, 48, kernel_size=3)
        self.conv3 = nn.Conv1d(48, 64, kernel_size=4)
        self.conv4 = nn.Conv1d(64, 80, kernel_size=5)

        # 使用自适应池化处理不同长度的输入
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)

        # 固定的展平维度
        self.flatten_dim = 80 * 10
        self.linear = nn.Linear(self.flatten_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """编码 x 为正态分布的均值和方差

        Args:
            x: shape [batch, NUM_RULES, max_length] 的 one-hot 编码

        Returns:
            mu, sigma: 潜在分布的均值和方差
        """
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.relu(h)
        h = self.conv4(h)
        h = self.relu(h)

        # 自适应池化到固定长度
        h = self.adaptive_pool(h)

        h = h.view(x.size(0), -1)
        h = self.linear(h)
        h = self.relu(h)
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma

    def sample(self, mu, sigma):
        """重参数化采样"""
        eps = torch.randn_like(sigma)
        return mu + eps * torch.sqrt(sigma)

    def kl(self, mu, sigma):
        """KL 散度"""
        return -0.5 * torch.sum(1 + torch.log(sigma + 1e-10) - mu.pow(2) - sigma, dim=1).mean()


if __name__ == '__main__':
    encoder = Encoder(hidden_dim=50, z_dim=32)
    x = torch.randn(10, NUM_RULES, 50)
    mu, sigma = encoder(x)
    print(f"mu shape: {mu.shape}, sigma shape: {sigma.shape}")
