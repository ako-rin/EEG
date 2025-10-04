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
from LibEER.data_utils.split import index_to_data, get_split_index
from LibEER.utils.args import get_args_parser
from LibEER.utils.store import make_output_dir
from LibEER.utils.utils import state_log, result_log, setup_seed, sub_result_log


class CE_Label_Smooth_Loss(torch.nn.Module):
    """Label smoothing loss from original PGCN paper."""
    def __init__(self, classes, epsilon=0.1):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.classes
        loss = (-targets * log_probs).sum(dim=1).mean()
        return loss


def train_one_subject(args, setting, data, label, train_indexes, test_indexes, output_dir, i_subject):
    """Train PGCN for one subject using original training logic."""
    
    # Prepare data
    x_train, y_train = index_to_data(data, label, train_indexes)
    x_test, y_test = index_to_data(data, label, test_indexes)
    
    # Create DataLoaders
    train_set = TensorDataset(
        torch.from_numpy(x_train).type(torch.FloatTensor),
        torch.from_numpy(y_train).type(torch.FloatTensor)
    )
    val_set = TensorDataset(
        torch.from_numpy(x_test).type(torch.FloatTensor),
        torch.from_numpy(y_test).type(torch.FloatTensor)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    # Initialize adjacency matrix and coordinates (PGCN-specific)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), delta=9))).to(device)
    coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)
    
    # Build args namespace for PGCN
    pgcn_args = SimpleNamespace(
        in_feature=data.shape[2],
        out_feature=20,
        n_class=len(np.unique(label)),
        dropout=0.4,
        epsilon=0.05,
        dataset=args.dataset,
        device=str(device),
        lr=0.1,  # LeakyReLU slope
        module=""
    )
    
    # Initialize model
    model = PGCN(pgcn_args, adj_matrix, coordinate_matrix).to(device)
    
    # Setup optimizer with parameter groups
    lap_params, local_params, weight_params = [], [], []
    for pname, p in model.named_parameters():
        if str(pname) == "adj":
            lap_params += [p]
        elif "local" in str(pname):
            local_params += [p]
        else:
            weight_params += [p]
    
    optimizer = optim.AdamW([
        {'params': lap_params, 'lr': 5e-5},
        {'params': local_params, 'lr': args.lr},
        {'params': weight_params, 'lr': args.lr},
    ], betas=(0.9, 0.999), weight_decay=0.01)
    
    criterion = CE_Label_Smooth_Loss(classes=pgcn_args.n_class, epsilon=0.05).to(device)
    
    # Scheduler with warmup
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.epochs // 3], gamma=0.1
    )
    warmup_scheduler = None
    if WARMUP_AVAILABLE:
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        warmup_scheduler.last_step = -1
    
    '''
    训练逻辑：
    '''
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device=device, dtype=torch.int64)
            
            optimizer.zero_grad()
            output, _, _ = model(x)  # output, lap_matrix, fused_features
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if warmup_scheduler and i < len(train_loader) - 1:
                with warmup_scheduler.dampening():
                    pass
            
            train_loss += loss.item() * y.size(0)
            train_correct += (torch.argmax(output, dim=1) == y).sum().item()
            train_total += y.size(0)
        
        if warmup_scheduler:
            with warmup_scheduler.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device=device, dtype=torch.int64)
                output, _, _ = model(x)  # output, lap_matrix, fused_features
                loss = criterion(output, y)
                
                val_loss += loss.item() * y.size(0)
                val_correct += (torch.argmax(output, dim=1) == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}@{best_epoch}")
        
        if best_val_acc >= 0.9999:
            break
    
    print(f"Subject {i_subject} finished. Best: {best_val_acc:.4f}@{best_epoch}")
    return best_val_acc


def main(args):
    # Build setting
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    # Output directory
    output_dir = make_output_dir(args, setting)
    state_log(args)
    
    # Load data
    data, label = get_data(args)
    print(f"Data shape: {data.shape}, Label shape: {label.shape}")
    
    # Get split configuration
    split_indexes = get_split_index(data, label, args, setting)
    
    # Training loop
    acc_list = []
    
    for i_subject in range(setting['subjects']):
        print(f"\n{'='*80}")
        print(f"Training Subject {i_subject + 1}/{setting['subjects']}")
        print(f"{'='*80}")
        
        train_indexes = split_indexes['train'][i_subject]
        test_indexes = split_indexes['test'][i_subject]
        
        val_acc = train_one_subject(
            args, setting, data, label,
            train_indexes, test_indexes,
            output_dir, i_subject
        )
        
        acc_list.append(val_acc)
        sub_result_log(i_subject, val_acc, acc_list)
    
    # Final results
    result_log(acc_list, args, setting, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Average accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = get_args_parser()
    
    # PGCN-specific arguments
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.0015, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    main(args)
    print(f'Batch size: {getattr(args, "batch_size", 32)}')
    print(f'Learning rate: {getattr(args, "lr", 0.001)}')
    print(f'Beta (adj LR): {getattr(args, "beta", 5e-5)}')
    print(f'Weight decay: {getattr(args, "weight_decay", 5e-4)}')
    print(f'Label smoothing: {getattr(args, "label_smoothing", 0.0)}')
    print(f'Epochs: {getattr(args, "epochs", 40)}')
    print('='*60)
    
    # log out train state
    state_log(args)
    main(args)
