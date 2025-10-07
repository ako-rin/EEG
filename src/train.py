import sys
from pathlib import Path

# Bootstrap sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]  # /home/ako/Project/work
_libeer_root = _proj_root / 'lib' / 'libeer-ako'
for p in (str(_proj_root), str(_libeer_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

import time
import numpy as np
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from types import SimpleNamespace
import pytorch_warmup as warmup

# Import PGCN model and utilities
from EEG.models.PGCN import PGCN, convert_dis_m, get_ini_dis_m, return_coordinates, CE_Label_Smooth_Loss

# Import LibEER framework
from LibEER.config.setting import preset_setting, set_setting_by_args, Setting
from LibEER.data_utils.load_data import get_data
from LibEER.data_utils.split import index_to_data, get_split_index, merge_to_part
from LibEER.utils.args import get_args_parser
from LibEER.utils.store import make_output_dir
from LibEER.utils.utils import state_log, result_log, setup_seed, sub_result_log
from LibEER.Trainer.training import train

def main():
    # Build setting
    setting = Setting(dataset='seed_de',  # Select the dataset
                      dataset_path='/home/ako/Project/datasets/SEED/',  # Specify the path to the corresponding dataset.
                      # 使用的DE特征也要保留下面这两行需
                      sample_length=1,
                      stride=1,
                      seed=2024,  # 供数据管线读取用，如早停划分时的随机打乱、抽样等
                      feature_type='de_lds',  # set the feature type to extract (using DE features: LDS only; no re-extraction)
                      experiment_mode="subject-dependent",
                      split_type='front-back',  # 数据分割成 train test val
                      test_size=0.2,
                      val_size=0.2,
                      # 无关设置
                      pass_band=[0.3, 50],  # use a band-pass filter with a range of 0.3 to 50 Hz,
                      extract_bands=[[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]],
                      time_window=1,  # Set the time window for feature extraction to 1 second.
                      overlap=0,  # The overlap length of the time window for feature extraction.
                      )

    #设置运行期的随机状态：random、numpy、torch（CPU/CUDA）、以及部分 cudnn 选项。
    # 影响权重初始化、Dropout、DataLoader 打乱等。
    setup_seed(2024)

    # Load data
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting) # 数据划分成part
    
    # 设备放在使用前初始化，避免未定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PGCN config
    pgcn_dict = dict(
        in_feature=int(feature_dim),
        out_feature=20,
        n_class=num_classes,
        dropout=0.4,
        epsilon=0.05,
        device=str(device),
        lr=0.1,  # LeakyReLU slope parameter
        module=""
    )
    pgcn_config = SimpleNamespace(**pgcn_dict)
    
    best_metrics = []
    subjects_metrics = [[]for _ in range(len(data))]
    
    # zip(data, label)： 将 data 和 label 按顺序进行配对成组，并从 1 开始分配编号
    # (data_i, label_i)： zip(data, label) 中的一个组
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        
        # 从划分的数据每个 part 中划分训练的部分以及验证的部分
        tts = get_split_index(data_i, label_i, setting)
        
        # 获取到训练、测试、验证的索引
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            
            setup_seed(2024)

            if val_indexes[0] == -1:    # 没有划分给val，只有 train 和 test
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # 通过索引加载数据
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes)
    
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), delta=9))).to(device)
            coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)
            model = PGCN(pgcn_config, adj_matrix, coordinate_matrix).to(device)
            
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            """
            模型训练参数配置
            """
            lap_params, local_params, weight_params = [], [], []
            for pname, p in model.named_parameters():
                lname = pname.lower()
                if 'adj' in lname or 'lap' in lname:
                    lap_params.append(p)
                elif 'local' in lname:
                    local_params.append(p)
                else:
                    weight_params.append(p)
            optimizer = optim.AdamW([
                {'params': lap_params, 'lr': 5e-5},
                {'params': local_params, 'lr': 0.01},
                {'params': weight_params, 'lr': 0.01},
            ], betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100 // 3], gamma=0.1)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
            warmup_scheduler.last_step = -1
            criterion = CE_Label_Smooth_Loss(classes=pgcn_config.n_class, epsilon=pgcn_config.epsilon).to(device)

            # 辅助损失：对图相关参数做 L1 稀疏正则（鼓励稀疏的邻接/拉普拉斯）
            def aux_l1_sparse(cfg):
                lam = float(cfg.get('lam', 0.0))
                params = cfg.get('params') or []  # e.g., lap_params / adj 参数
                dev = cfg.get('device', None)
                if not params:
                    # 无可正则化参数时返回 0（放在正确设备上）
                    return torch.tensor(0.0, device=dev if dev is not None else next(model.parameters()).device)
                total = None
                for p in params:
                    if p is None or (not p.requires_grad):
                        continue
                    term = p.abs().sum()
                    total = term if total is None else (total + term)
                if total is None:
                    return torch.tensor(0.0, device=dev if dev is not None else next(model.parameters()).device)
                return lam * total

            loss_func = aux_l1_sparse
            loss_param = {'lam': 1e-4, 'params': lap_params, 'device': device}

            round_metric = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir="result/PGCN/TEST",
                metrics=["acc","macro-f1"],
                metric_choose="macro-f1",
                optimizer=optimizer,
                scheduler=scheduler,
                warmup_scheduler=warmup_scheduler,
                batch_size=32,
                epochs=100,
                criterion=criterion,
                loss_func=loss_func,
                loss_param=loss_param,
            )
            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx-1].append(round_metric)
    metrics = dict (
        metric = ["acc","macro-f1"]
    )
    metrics = SimpleNamespace(**metrics)
    result_log(metrics, best_metrics)

if __name__ == "__main__" :
    main()