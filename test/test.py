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
    
    adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), delta=9))).to(device)
    coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)
    model = PGCN(pgcn_config, adj_matrix, coordinate_matrix).to(device)
    
    # 直接在 main 中打印每一层权重（名称、形状与统计信息）
    for name, param in model.named_parameters():
        if param is None or not param.requires_grad:
            continue
        t = param.detach().float().cpu()
        print(f"model.{name}: shape={list(t.shape)}, numel={t.numel()}, "
              f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
              f"min={t.min().item():.6f}, max={t.max().item():.6f}")

    # 外部的图参数（不在 model.named_parameters 里）也一并打印
    t_adj = adj_matrix.detach().float().cpu()
    print(f"adj_matrix: shape={list(t_adj.shape)}, numel={t_adj.numel()}, "
          f"mean={t_adj.mean().item():.6f}, std={t_adj.std().item():.6f}, "
          f"min={t_adj.min().item():.6f}, max={t_adj.max().item():.6f}")
    
    
    
    lap_params, local_params, weight_params = [], [], []
    for pname, p in model.named_parameters():
        lname = pname.lower()
        print(lname)
        if 'adj' in lname or 'lap' in lname:
            lap_params.append(p)
        elif 'local' in lname:
            local_params.append(p)
        else:
            weight_params.append(p)
            
    # === 分组完成后：打印每组里“参数名 + 模块类型 + 形状” ===
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    named_modules = dict(model.named_modules())
    
    groups = [('lap_params', lap_params),
              ('local_params', local_params),
              ('weight_params', weight_params)]

    for gname, plist in groups:
        print(f"\n{gname} (count={len(plist)}):")
        for p in plist:
            name = param_to_name.get(id(p), '(unknown)')
            # 通过参数名推断所属模块名（去掉 .weight/.bias 等尾巴）
            module_name = name.rsplit('.', 1)[0] if '.' in name else ''
            module = named_modules.get(module_name, model if module_name == '' else None)
            module_type = type(module).__name__ if module is not None else 'UnknownModule'
            print(f"  {name:<40} | module={module_type:<20} | shape={tuple(p.shape)}")
    
    optimizer = optim.AdamW([
        {'params': lap_params, 'lr': 5e-5},
        {'params': local_params, 'lr': 0.01},
        {'params': weight_params, 'lr': 0.01},
    ], betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-4)
    

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100 // 3], gamma=0.1)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    warmup_scheduler.last_step = -1


if __name__ == "__main__" :
    main()