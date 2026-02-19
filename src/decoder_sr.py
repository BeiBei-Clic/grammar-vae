"""解码器模块 - 从潜在向量 z 生成规则序列"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.grammar_sr import NUM_RULES


class Decoder(nn.Module):
    """RNN 解码器，从潜在向量 z 重构规则序列"""

    def __init__(self, z_dim=32, hidden_size=100):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.linear_in = nn.Linear(z_dim, hidden_size)
        self.linear_out = nn.Linear(hidden_size, NUM_RULES)

        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.relu = nn.ReLU()

    def forward(self, z, max_length=50):
        """前向传播

        Args:
            z: shape [batch, z_dim] 潜在向量
            max_length: 最大序列长度

        Returns:
            logits: shape [batch, max_length, NUM_RULES]
        """
        x = self.linear_in(z)
        x = self.relu(x)

        # 在每个时间步输入相同的 z
        x = x.unsqueeze(1).expand(-1, max_length, -1)

        # 初始化隐藏状态 (num_layers, batch, hidden_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)

        x, _ = self.rnn(x, (h0, c0))

        x = self.relu(x)
        x = self.linear_out(x)
        return x


if __name__ == '__main__':
    decoder = Decoder(z_dim=32, hidden_size=100)
    z = torch.randn(10, 32)
    logits = decoder(z, max_length=50)
    print(f"logits shape: {logits.shape}")
