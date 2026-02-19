"""GrammarVAE 符号回归主实验脚本

支持单个 CSV 文件或文件夹批量实验，支持多 GPU 并行。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.dataset import StandardSRDataset, find_dataset_files
from src.sr_model import GrammarVAEModel
from src.metrics import evaluate_expression_string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='GrammarVAE Symbolic Regression')

    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集文件或文件夹路径')
    parser.add_argument('--max_input_points', type=int, default=500,
                        help='每个数据集最大输入点数')

    # 模型参数
    parser.add_argument('--z_dim', type=int, default=32,
                        help='潜在向量维度')
    parser.add_argument('--hidden_encoder_dim', type=int, default=50,
                        help='编码器隐藏层维度')
    parser.add_argument('--hidden_decoder_dim', type=int, default=100,
                        help='解码器隐藏层维度')
    parser.add_argument('--max_length', type=int, default=50,
                        help='最大序列长度')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='预训练模型检查点路径')

    # 搜索参数
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='最大搜索迭代次数')
    parser.add_argument('--search_method', type=str, default='random',
                        choices=['random', 'cmaes'],
                        help='搜索方法')

    # 实验参数
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='随机种子数量')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='测试集比例')

    # 多 GPU 参数
    parser.add_argument('--gpus', type=str, default=None,
                        help='使用的 GPU 列表，如 "0,1,2,3"')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='基础随机种子')

    return parser.parse_args()


def run_single_dataset(dataset_path: Path, args: argparse.Namespace, gpu_id: int = None) -> Dict[str, Any]:
    """运行单个数据集的实验

    Args:
        dataset_path: 数据集文件路径
        args: 命令行参数
        gpu_id: GPU ID，如果为 None 则使用 CPU

    Returns:
        实验结果字典
    """
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    logger.info(f"Running experiment on {dataset_path.name} with device: {device}")

    # 加载数据集
    dataset = StandardSRDataset(
        str(dataset_path),
        test_ratio=args.test_ratio,
        max_input_points=args.max_input_points
    )

    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()

    var_names = [f'x{i}' for i in range(dataset.n_features)]

    # 初始化模型
    model = GrammarVAEModel(
        z_dim=args.z_dim,
        hidden_encoder_dim=args.hidden_encoder_dim,
        hidden_decoder_dim=args.hidden_decoder_dim,
        max_length=args.max_length
    ).to(device)

    # 加载预训练检查点
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

    runs = []

    for seed_idx in range(args.num_seeds):
        seed = args.base_seed + seed_idx
        logger.info(f"Dataset: {dataset_path.name}, Seed: {seed_idx + 1}/{args.num_seeds}")

        # 在训练集上搜索
        best_expr, train_result = model.search_expression(
            X_train, y_train,
            n_iterations=args.max_iterations,
            method=args.search_method,
            seed=seed
        )

        # 在测试集上评估
        test_result = evaluate_expression_string(best_expr, X_test, y_test, var_names)

        runs.append({
            'seed': seed,
            'final_expression': best_expr,
            'train_rmse': float(train_result['train_rmse']),
            'train_r2': float(train_result['train_r2']),
            'test_rmse': float(test_result['rmse']),
            'test_r2': float(test_result['r2']),
            'evolution_rmse': [float(x) for x in train_result['evolution_rmse']]
        })

        logger.info(f"  Final expression: {best_expr}")
        logger.info(f"  Train RMSE: {runs[-1]['train_rmse']:.6f}, R²: {runs[-1]['train_r2']:.6f}")
        logger.info(f"  Test RMSE: {runs[-1]['test_rmse']:.6f}, R²: {runs[-1]['test_r2']:.6f}")

    return {
        'dataset': dataset_path.stem,
        'ground_truth': dataset.ground_truth,
        'n_features': dataset.n_features,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'runs': runs
    }


def save_results(results: Dict[str, Any], args: argparse.Namespace):
    """保存实验结果"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存完整结果
    result_file = output_dir / f'results_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {result_file}")

    # 保存摘要
    summary = []
    for dataset_name, dataset_results in results.items():
        best_test_rmse = min(r['test_rmse'] for r in dataset_results['runs'])
        best_test_r2 = max(r['test_r2'] for r in dataset_results['runs'])
        summary.append({
            'dataset': dataset_name,
            'ground_truth': dataset_results['ground_truth'],
            'best_test_rmse': best_test_rmse,
            'best_test_r2': best_test_r2
        })

    summary_file = output_dir / f'summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")


def run_worker(dataset_files: List[Path], args: argparse.Namespace, gpu_id: int = None) -> Dict[str, Any]:
    """工作进程，运行一组数据集的实验

    Args:
        dataset_files: 数据集文件列表
        args: 命令行参数
        gpu_id: GPU ID

    Returns:
        实验结果字典
    """
    results = {}

    for dataset_file in dataset_files:
        try:
            dataset_results = run_single_dataset(dataset_file, args, gpu_id)
            results[dataset_file.stem] = dataset_results
        except Exception as e:
            logger.error(f"Error processing {dataset_file}: {e}")
            results[dataset_file.stem] = {'error': str(e)}

    return results


def main():
    args = parse_args()

    # 查找数据集文件
    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        dataset_files = [dataset_path]
    else:
        dataset_files = find_dataset_files(str(dataset_path), '*.csv')

    if not dataset_files:
        logger.error(f"No dataset files found in {args.dataset}")
        return

    logger.info(f"Found {len(dataset_files)} dataset files")

    # 处理 GPU 分配
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_ids = None

    all_results = {}

    if gpu_ids and len(dataset_files) > 1:
        # 多 GPU 并行处理
        from concurrent.futures import ProcessPoolExecutor

        # 将数据集分配给不同的 GPU
        files_per_gpu = (len(dataset_files) + len(gpu_ids) - 1) // len(gpu_ids)

        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for i, gpu_id in enumerate(gpu_ids):
                start_idx = i * files_per_gpu
                end_idx = min(start_idx + files_per_gpu, len(dataset_files))
                files = dataset_files[start_idx:end_idx]

                if files:
                    future = executor.submit(run_worker, files, args, gpu_id)
                    futures.append(future)

            # 收集结果
            for future in futures:
                results = future.result()
                all_results.update(results)
    else:
        # 单进程处理
        all_results = run_worker(dataset_files, args, gpu_ids[0] if gpu_ids else None)

    # 保存结果
    save_results(all_results, args)


if __name__ == '__main__':
    main()
