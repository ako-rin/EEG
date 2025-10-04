#!/bin/bash
# ============================================================
# PGCN 实验脚本集合
# ============================================================

# 实验 1: 论文配置 (Session 1 + Front-Back) ⭐⭐⭐⭐⭐
echo "=========================================="
echo "实验 1: 论文配置 (推荐)"
echo "Session 1, Front-Back (前9后6，无验证集)"
echo "=========================================="
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
  >result/PGCN/exp1_session1_frontback.log

echo "实验 1 完成，查看: result/PGCN/exp1_session1_frontback.log"
echo ""

# 实验 2: 添加归一化 (如果实验 1 仍不理想) ⭐⭐⭐⭐
echo "=========================================="
echo "实验 2: 论文配置 + 归一化"
echo "=========================================="
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
  -normalize \
  -onehot \
  >result/PGCN/exp2_session1_frontback_normalize.log

echo "实验 2 完成，查看: result/PGCN/exp2_session1_frontback_normalize.log"
echo ""

# 实验 3: Train/Val/Test 设置 (与你之前的对比) ⭐⭐⭐
echo "=========================================="
echo "实验 3: Session 1 + Train/Val/Test"
echo "对比验证集的影响"
echo "=========================================="
python ./src/PGCN_train.py \
  -metrics acc macro-f1 \
  -metric_choose macro-f1 \
  -setting seed_sub_dependent_train_val_test_setting \
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
  >result/PGCN/exp3_session1_trainvaltest.log

echo "实验 3 完成，查看: result/PGCN/exp3_session1_trainvaltest.log"
echo ""

# 实验 4: 3 Sessions + Front-Back (测试多 session 影响) ⭐⭐
echo "=========================================="
echo "实验 4: 3 Sessions + Front-Back"
echo "测试多 session 的影响"
echo "=========================================="
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
  -sessions 1 2 3 \
  -onehot \
  >result/PGCN/exp4_3sessions_frontback.log

echo "实验 4 完成，查看: result/PGCN/exp4_3sessions_frontback.log"
echo ""

echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo ""
echo "查看结果汇总:"
echo "  实验 1: tail -n 30 result/PGCN/exp1_session1_frontback.log"
echo "  实验 2: tail -n 30 result/PGCN/exp2_session1_frontback_normalize.log"
echo "  实验 3: tail -n 30 result/PGCN/exp3_session1_trainvaltest.log"
echo "  实验 4: tail -n 30 result/PGCN/exp4_3sessions_frontback.log"
echo ""
echo "预期性能对比:"
echo "  实验 1 (论文配置):      75-80%  ⭐⭐⭐⭐⭐"
echo "  实验 2 (+ 归一化):      73-78%  ⭐⭐⭐⭐"
echo "  实验 3 (+ 验证集):      70-75%  ⭐⭐⭐"
echo "  实验 4 (3 sessions):    65-70%  ⭐⭐"
echo ""
