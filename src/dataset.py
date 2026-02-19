"""标准符号回归数据集加载器

支持 SRBench 标准格式的 CSV 文件：
- 第一行: 真实表达式字符串
- 第二行起: 数值数据（空格分隔）
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandardSRDataset:
    """标准符号回归数据集加载器"""

    def __init__(self, csv_path: str, test_ratio: float = 0.2, seed: int = 42,
                 max_input_points: Optional[int] = None):
        """初始化数据集

        Args:
            csv_path: CSV 文件路径
            test_ratio: 测试集比例
            seed: 随机种子
            max_input_points: 最大输入点数，None 表示使用全部数据
        """
        self.csv_path = Path(csv_path)
        self.test_ratio = test_ratio
        self.seed = seed
        self.max_input_points = max_input_points

        self.ground_truth = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.n_features = None

        self._load_data()

    def _load_data(self):
        """加载数据"""
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()

        # 第一行是真实表达式
        self.ground_truth = lines[0].strip()

        # 解析数值数据
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                data.append(values)

        data = np.array(data)

        # 限制数据点数
        if self.max_input_points is not None and len(data) > self.max_input_points:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(data), self.max_input_points, replace=False)
            data = data[indices]

        # 分离特征和标签
        self.X = data[:, :-1]
        self.y = data[:, -1]
        self.n_features = self.X.shape[1]

        # 划分训练集和测试集
        self._split_data()

        logger.info(f"Loaded dataset: {self.csv_path.name}")
        logger.info(f"Ground truth: {self.ground_truth}")
        logger.info(f"Features: {self.n_features}, Samples: {len(self.X)}")
        logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")

    def _split_data(self):
        """划分训练集和测试集"""
        rng = np.random.default_rng(self.seed)
        n_samples = len(self.X)
        n_test = int(n_samples * self.test_ratio)

        indices = rng.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        self.X_test = self.X[test_indices]
        self.y_test = self.y[test_indices]
        self.X_train = self.X[train_indices]
        self.y_train = self.y[train_indices]

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试数据"""
        return self.X_test, self.y_test

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取全部数据"""
        return self.X, self.y


def find_dataset_files(dataset_path: str, pattern: str = "*.csv") -> List[Path]:
    """查找数据集文件

    Args:
        dataset_path: 数据集目录或文件路径
        pattern: 文件匹配模式

    Returns:
        文件路径列表
    """
    path = Path(dataset_path)
    if path.is_file():
        return [path]
    return sorted(path.rglob(pattern))


if __name__ == '__main__':
    # 测试数据集加载器
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 使用示例路径
        csv_path = "dataset/feynman/Feynman_with_units/I.10.7.csv"

    dataset = StandardSRDataset(csv_path, max_input_points=100)
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()

    print(f"Ground truth: {dataset.ground_truth}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
