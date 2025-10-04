#!/bin/bash
# ============================================================
# PGCN 优化训练脚本 - 基于论文配置
# ============================================================
# 
# 本脚本根据性能分析报告的建议，使用最接近论文的配置
# 
# 主要改进:
# 1. 使用 seed_sub_dependent_front_back_setting (前9后6，无验证集)
# 2. 只使用 session 1 (降低数据复杂度)
# 3. 使用论文的超参数 (lr=0.01, beta=5e-5, weight_decay=5e-4, label_smoothing=0.05)
# 4. 集成 warmup scheduler
# 5. 使用论文的 CE_Label_Smooth_Loss
# 6. DataLoader 添加 drop_last=True
# 
# 预期性能: 75-80% (接近论文的 80%+)
# ============================================================

echo "=========================================="
echo "PGCN Training - Paper Configuration"
echo "=========================================="
echo ""
echo "配置概要:"
echo "  - Setting: seed_sub_dependent_front_back_setting (前9后6)"
echo "  - Sessions: 1 only"
echo "  - Batch size: 32"
echo "  - Learning rate: 0.01"
echo "  - Beta (adj LR): 5e-5"
echo "  - Weight decay: 5e-4"
echo "  - Label smoothing: 0.05"
echo "  - Epochs: 150"
echo ""
echo "=========================================="

# 实验 1: 最接近论文的配置 (推荐首先运行)
python ./src/PGCN_train.py \
  -metrics acc macro-f1 \
  -metric_choose macro-f1 \
  -setting seed_sub_dependent_front_back_setting \
  -dataset_path ../../datasets/SEED \
  -dataset seed_de_lds \
  -batch_size 32 \
  -epochs 150 \
  -lr 0.01 \
  -beta 5e-5 \
  -weight_decay 5e-4 \
  -label_smoothing 0.05 \
  -model PGCN \
  -sessions 1 \
  -onehot \
  >result/PGCN/paper_config_session1_frontback.log 2>&1

echo ""
echo "=========================================="
echo "训练完成！"
echo "日志文件: result/PGCN/paper_config_session1_frontback.log"
echo ""
echo "查看结果: tail -n 50 result/PGCN/paper_config_session1_frontback.log"
echo "=========================================="
