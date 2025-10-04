"""
Plot subject-wise performance comparison across different models.

Usage:
    python utils/plot_subject_performance.py
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path


def extract_subject_results(log_file):
    """
    从训练日志中提取每个 subject 的测试准确率
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        list: 每个 subject 的测试准确率列表
    """
    subject_accs = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 匹配 best_test_acc: 0.85 格式
    matches = re.findall(r'best_test_acc:\s*([\d.]+)', content)
    
    for match in matches:
        acc = float(match)
        subject_accs.append(acc * 100)  # 转换为百分比
    
    return subject_accs


def plot_subject_comparison(results_dict, save_path='subject_comparison.png', 
                           title='Subject-wise Performance Comparison',
                           ylabel='Accuracy(%)',
                           figsize=(10, 6)):
    """
    绘制多模型在不同 subjects 上的性能对比图
    
    Args:
        results_dict: 字典 {model_name: [acc1, acc2, ...]}
        save_path: 保存路径
        title: 图表标题
        ylabel: Y轴标签
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 定义颜色和标记样式
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#FFE66D', '#95E1D3', '#4A90E2']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 绘制每个模型的曲线
    for idx, (model_name, accs) in enumerate(results_dict.items()):
        subjects = list(range(len(accs)))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(subjects, accs, 
                marker=marker, 
                color=color, 
                linewidth=2, 
                markersize=8,
                label=model_name,
                alpha=0.8)
    
    # 设置图表样式
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 Y 轴范围
    all_values = [v for accs in results_dict.values() for v in accs]
    y_min = max(0, min(all_values) - 5)
    y_max = min(100, max(all_values) + 2)
    plt.ylim(y_min, y_max)
    
    # 设置 X 轴刻度
    max_subjects = max(len(accs) for accs in results_dict.values())
    plt.xticks(range(0, max_subjects, 5))  # 每5个显示一个刻度
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.show()


def calculate_statistics(results_dict):
    """
    计算并打印每个模型的统计数据
    
    Args:
        results_dict: 字典 {model_name: [acc1, acc2, ...]}
    """
    print("\n" + "="*70)
    print("Model Performance Statistics")
    print("="*70)
    print(f"{'Model':<25} {'Mean±Std':<20} {'Min':<10} {'Max':<10}")
    print("-"*70)
    
    for model_name, accs in results_dict.items():
        mean = np.mean(accs)
        std = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        
        print(f"{model_name:<25} {mean:>5.2f}±{std:<5.2f}        "
              f"{min_acc:>6.2f}     {max_acc:>6.2f}")
    
    print("="*70 + "\n")


def main():
    """
    主函数:读取日志并绘图
    """
    # 定义日志文件路径
    result_dir = Path(__file__).parent.parent / 'result' / 'PGCN'
    
    # 示例:比较不同配置的 PGCN
    results = {}
    
    # 配置1: 当前训练 (lr=0.01, 3 sessions)
    log_file_1 = result_dir / 'b32_3sess_frontback.log'
    if log_file_1.exists():
        accs = extract_subject_results(log_file_1)
        if accs:
            results['PGCN (lr=0.01)'] = accs
            print(f"✅ Loaded {len(accs)} subjects from {log_file_1.name}")
    
    # 配置2: 单 session 训练
    log_file_2 = result_dir / 'b32lr0.01.log'
    if log_file_2.exists():
        accs = extract_subject_results(log_file_2)
        if accs:
            results['PGCN (1 session)'] = accs[:15]  # 只取前15个
            print(f"✅ Loaded {len(accs[:15])} subjects from {log_file_2.name}")
    
    # 如果有其他模型的结果,可以继续添加
    # 例如:
    # dgcnn_log = result_dir.parent / 'DGCNN' / 'b32_lr0.0015.log'
    # if dgcnn_log.exists():
    #     accs = extract_subject_results(dgcnn_log)
    #     if accs:
    #         results['DGCNN'] = accs
    
    if not results:
        print("❌ No valid log files found!")
        print(f"   Searched in: {result_dir}")
        return
    
    # 计算统计数据
    calculate_statistics(results)
    
    # 绘制对比图
    save_path = result_dir / 'subject_performance_comparison.png'
    plot_subject_comparison(
        results, 
        save_path=str(save_path),
        title='PGCN: Subject-wise Performance on SEED Dataset',
        ylabel='Test Accuracy (%)'
    )


if __name__ == '__main__':
    main()
