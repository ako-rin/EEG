"""
Train PGCN with LibEER framework.

Adapted from Reference/PGCN/PGCN/main_PGCN.py to work with LibEER's data loading.
Uses original PGCN interface: PGCN(args, adj_matrix, coordinates)

Usage:
  python PGCN_train.py -metrics acc macro-f1 \
    -metric_choose macro-f1 -setting seed_sub_dependent_front_back_setting \
    -dataset seed_de_lds -batch_size 32 -epochs 80 -lr 0.0015
"""

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

try:
    import pytorch_warmup as warmup
    WARMUP_AVAILABLE = True
except ImportError:
    print("[WARNING] pytorch_warmup not installed. Install with: pip install pytorch-warmup")
    WARMUP_AVAILABLE = False

# Import PGCN model and utilities
from EEG.models.PGCN import PGCN, convert_dis_m, get_ini_dis_m, return_coordinates

# Import LibEER framework
from LibEER.config.setting import preset_setting, set_setting_by_args
from LibEER.data_utils.load_data import get_data
from LibEER.data_utils.split import index_to_data, get_split_index, merge_to_part
from LibEER.utils.args import get_args_parser
from LibEER.utils.store import make_output_dir
from LibEER.utils.utils import state_log, result_log, setup_seed, sub_result_log
from LibEER.Trainer.training import train


class CE_Label_Smooth_Loss(torch.nn.Module):
    """Label smoothing loss from original PGCN paper."""
    def __init__(self, classes, epsilon=0.1):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: (batch_size, num_classes) - logits
            target: (batch_size,) - class indices (already converted from one-hot in Trainer)
        """
        # Ensure target is 1D long tensor
        target = target.long()
        if target.dim() > 1:
            target = target.squeeze(-1)
        
        # Compute log probabilities
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        
        # Create weight matrix for label smoothing
        weight = input.new_ones(input.size()) * (self.epsilon / (self.classes - 1.))
        
        # Scatter the correct class weight (1 - epsilon)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        
        # Compute loss
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def train_one_subject(args, data, label, train_indexes, test_indexes, val_indexes,
                      model, optimizer, scheduler, warmup_scheduler, criterion, device, output_dir):
    """Train PGCN for one subject using LibEER's train() function."""
    
    # Prepare data
    train_data, train_label, val_data, val_label, test_data, test_label = \
        index_to_data(data, label, train_indexes, test_indexes, val_indexes, args.keep_dim)
    
    # Fallback: if no val provided, use test as val
    if len(val_data) == 0:
        val_data = test_data
        val_label = test_label
    
    # Create TensorDatasets
    dataset_train = torch.utils.data.TensorDataset(
        torch.Tensor(train_data), 
        torch.Tensor(train_label)
    )
    dataset_val = torch.utils.data.TensorDataset(
        torch.Tensor(val_data), 
        torch.Tensor(val_label)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.Tensor(test_data), 
        torch.Tensor(test_label)
    )
    
    # Use LibEER's train() function
    round_metric = train(
        model=model,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        device=device,
        output_dir=output_dir,
        metrics=args.metrics,
        metric_choose=args.metric_choose,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        batch_size=args.batch_size,
        epochs=args.epochs,
        criterion=criterion,
        loss_func=None,
        loss_param=None,
    )
    
    return round_metric

def main(args):
    # Build setting
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    # Load data
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    # Use feature_dim returned by get_data as in_feature
    in_feature = int(feature_dim)

    # PGCN args template (we will instantiate a fresh pgcn_args per round to be safe)
    pgcn_template = dict(
        in_feature=in_feature,
        out_feature=20,
        n_class=num_classes,
        dropout=args.dropout,
        epsilon=args.epsilon,
        dataset=args.dataset,
        device=str(device),
        lr=0.1,  # LeakyReLU slope parameter
        module=""
    )

    # Loss settings (we will create criterion per round if desired)
    # Training summary
    best_metrics = []
    subjects_metrics = [[] for _ in range(len(data))] if setting.experiment_mode == 'subject-dependent' else []

    output_dir = make_output_dir(args, "PGCN")

    # Loop subjects / rounds — create model + optimizer per round to match original behavior
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        # rridx: 受试者编号
        # data_i，label_i：当前受试者的数据和标签
        tts = get_split_index(data_i, label_i, setting) # 数据分割器
        # 
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)

            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # Prepare pgcn_args and create fresh model for this round
            pgcn_args = SimpleNamespace(**pgcn_template)
            # init adj and coordinates on device
            adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), delta=9))).to(device)
            coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)
            model = PGCN(pgcn_args, adj_matrix, coordinate_matrix).to(device)

            # build optimizer param groups robustly
            lap_params, local_params, weight_params = [], [], []
            for pname, p in model.named_parameters():
                lname = pname.lower()
                if 'adj' in lname or 'lap' in lname:
                    lap_params.append(p)
                elif 'local' in lname:
                    local_params.append(p)
                else:
                    weight_params.append(p)

            beta = float(getattr(args, 'beta', 5e-5))
            weight_decay = float(getattr(args, 'weight_decay', 5e-4))
            optimizer = optim.AdamW([
                {'params': lap_params, 'lr': beta},
                {'params': local_params, 'lr': args.lr},
                {'params': weight_params, 'lr': args.lr},
            ], betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-4)

            # scheduler + warmup
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 3], gamma=0.1)
            if WARMUP_AVAILABLE:
                warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
                warmup_scheduler.last_step = -1
            else:
                warmup_scheduler = None

            # criterion (use CE_Label_Smooth_Loss you defined)
            criterion = CE_Label_Smooth_Loss(classes=pgcn_args.n_class, epsilon=pgcn_args.epsilon).to(device)

            # Now call train for this round
            round_metric = train_one_subject(
                args, data_i, label_i,
                train_indexes, test_indexes, val_indexes,
                model, optimizer, scheduler, warmup_scheduler, criterion, device, output_dir
            )

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    # Log results
    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)

if __name__ == '__main__':
    parser = get_args_parser()
    
    # PGCN-specific arguments (use single dash to match LibEER style)
    parser.add_argument('-beta', type=float, default=5e-5, help='Learning rate for laplacian matrix (adj params)')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
    parser.add_argument('-epsilon', type=float, default=0.05, help='Label smoothing epsilon')
    parser.add_argument('-dropout', type=float, default=0.4, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Log train state
    state_log(args)
    main(args)

