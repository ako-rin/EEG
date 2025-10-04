"""
通用的 Subject-wise 性能对比绘图工具

可以比较:
1. 不同模型 (PGCN, DGCNN, EEGNet, etc.)
2. 同一模型的不同超参数配置
3. 不同数据集划分策略

Usage:
    # 自动扫描所有日志
    python utils/plot_comparison_auto.py
    
    # 指定日志文件
    python utils/plot_comparison_auto.py \
        --logs result/PGCN/b32_3sess_frontback.log:PGCN \
               result/DGCNN/b32_lr0.0015.log:DGCNN
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from pathlib import Path
from typing import Dict, List


def extract_subject_results(log_file: str, metric: str = 'acc') -> List[float]:
    """
    从训练日志中提取每个 subject 的测试指标
    
    Args:
        log_file: 日志文件路径
        metric: 指标名称 ('acc' 或 'macro-f1')
    
    Returns:
        list: 每个 subject 的测试指标列表 (百分比形式)
    """
    subject_metrics = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading {log_file}: {e}")
        return []
    
    # 根据 metric 选择正则表达式
    if metric == 'acc':
        pattern = r'best_test_acc:\s*([\d.]+)'
    elif 'macro' in metric or 'f1' in metric:
        pattern = r'best_test_macro-f1:\s*([\d.]+)'
    else:
        pattern = rf'best_test_{metric}:\s*([\d.]+)'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        value = float(match)
        subject_metrics.append(value * 100)  # 转换为百分比
    
    return subject_metrics


def auto_discover_logs(base_dir: str = 'result') -> Dict[str, str]:
    """
    自动发现训练日志文件
    
    Args:
        base_dir: 结果目录
    
    Returns:
        dict: {model_name: log_file_path}
    """
    discovered = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Directory not found: {base_dir}")
        return discovered
    
    # 扫描所有模型目录
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # 查找日志文件
        log_files = list(model_dir.glob('*.log'))
        
        for log_file in log_files:
            # 构建友好的名称
            model_name = model_dir.name
            config = log_file.stem  # 文件名(不含扩展名)
            
            if config.startswith(model_name.lower()):
                # 如果文件名包含模型名,去掉重复
                label = f"{model_name}"
            else:
                label = f"{model_name}({config})"
            
            discovered[label] = str(log_file)
    
    return discovered


def plot_subject_comparison(
    results_dict: Dict[str, List[float]], 
    save_path: str = 'subject_comparison.png',
    title: str = 'Subject-wise Performance Comparison',
    ylabel: str = 'Accuracy (%)',
    figsize: tuple = (12, 7),
    show_mean: bool = True
):
    """
    绘制多模型在不同 subjects 上的性能对比图
    
    Args:
        results_dict: {model_name: [value1, value2, ...]}
        save_path: 保存路径
        title: 图表标题
        ylabel: Y轴标签
        figsize: 图像大小
        show_mean: 是否显示均值线
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 预定义颜色方案(配色参考论文常用方案)
    colors = [
        '#E74C3C',  # Red
        '#F39C12',  # Orange  
        '#3498DB',  # Blue
        '#F1C40F',  # Yellow
        '#2ECC71',  # Green
        '#9B59B6',  # Purple
        '#1ABC9C',  # Turquoise
        '#E67E22',  # Carrot
    ]
    
    # 标记样式
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    # 绘制每个模型的曲线
    for idx, (model_name, values) in enumerate(results_dict.items()):
        subjects = list(range(len(values)))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # 绘制主曲线
        line = ax.plot(subjects, values, 
                       marker=marker, 
                       color=color, 
                       linewidth=2.5, 
                       markersize=7,
                       label=model_name,
                       alpha=0.85,
                       markerfacecolor=color,
                       markeredgecolor='white',
                       markeredgewidth=0.5)
        
        # 可选:绘制均值水平线
        if show_mean:
            mean_val = np.mean(values)
            ax.axhline(mean_val, color=color, linestyle='--', 
                      linewidth=1, alpha=0.3)
    
    # 设置图表样式
    ax.set_xlabel('Subject', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='best', fontsize=11, framealpha=0.95, 
             edgecolor='gray', fancybox=True, shadow=True)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # 网格在图形下方
    
    # 设置 Y 轴范围
    all_values = [v for values in results_dict.values() for v in values]
    if all_values:
        y_min = max(0, min(all_values) - 3)
        y_max = min(100, max(all_values) + 2)
        ax.set_ylim(y_min, y_max)
    
    # 设置 X 轴
    max_subjects = max(len(values) for values in results_dict.values())
    ax.set_xlim(-0.5, max_subjects - 0.5)
    
    # 美化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Plot saved to: {save_path}")
    plt.close()


def print_statistics(results_dict: Dict[str, List[float]]):
    """
    打印详细的统计信息
    """
    print("\n" + "="*85)
    print("Model Performance Statistics")
    print("="*85)
    print(f"{'Model':<30} {'Mean±Std':<18} {'Min':<10} {'Max':<10} {'Subjects':<8}")
    print("-"*85)
    
    for model_name, values in results_dict.items():
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        n_subjects = len(values)
        
        print(f"{model_name:<30} {mean:>5.2f}±{std:<5.2f}%       "
              f"{min_val:>6.2f}%    {max_val:>6.2f}%    {n_subjects:<8}")
    
    print("="*85)


def main():
    parser = argparse.ArgumentParser(description='Plot subject-wise performance comparison')
    parser.add_argument('--logs', nargs='+', 
                       help='Log files in format: path:label (e.g., result/PGCN/run1.log:PGCN-v1)')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-discover all logs in result/ directory')
    parser.add_argument('--result_dir', default='result',
                       help='Base directory for auto-discovery (default: result)')
    parser.add_argument('--metric', default='acc', choices=['acc', 'macro-f1'],
                       help='Which metric to plot (default: acc)')
    parser.add_argument('--output', default='subject_comparison.png',
                       help='Output file path (default: subject_comparison.png)')
    parser.add_argument('--title', default='Subject-wise Performance Comparison',
                       help='Plot title')
    parser.add_argument('--max_models', type=int, default=6,
                       help='Maximum number of models to plot (default: 6)')
    
    args = parser.parse_args()
    
    results = {}
    
    # 模式1: 自动发现
    if args.auto or not args.logs:
        print(f"🔍 Auto-discovering logs in {args.result_dir}/...")
        discovered = auto_discover_logs(args.result_dir)
        
        if not discovered:
            print("❌ No log files found!")
            return
        
        print(f"✅ Found {len(discovered)} log files:")
        for label, path in list(discovered.items())[:args.max_models]:
            print(f"   - {label}: {path}")
            metrics = extract_subject_results(path, args.metric)
            if metrics:
                results[label] = metrics
    
    # 模式2: 手动指定
    else:
        print(f"📂 Loading specified logs...")
        for log_spec in args.logs:
            if ':' in log_spec:
                path, label = log_spec.split(':', 1)
            else:
                path = log_spec
                label = Path(path).stem
            
            if not Path(path).exists():
                print(f"⚠️  File not found: {path}")
                continue
            
            metrics = extract_subject_results(path, args.metric)
            if metrics:
                results[label] = metrics
                print(f"✅ Loaded {len(metrics)} subjects from {label}")
    
    if not results:
        print("\n❌ No valid results extracted!")
        return
    
    # 打印统计数据
    print_statistics(results)
    
    # 绘图
    ylabel = 'Accuracy (%)' if args.metric == 'acc' else 'Macro-F1 (%)'
    plot_subject_comparison(
        results,
        save_path=args.output,
        title=args.title,
        ylabel=ylabel
    )
    
    print(f"\n🎉 Done! Check {args.output}")


if __name__ == '__main__':
    main()
