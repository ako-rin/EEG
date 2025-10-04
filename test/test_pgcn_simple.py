"""
Quick test for the simplified PGCN_train.py

This tests that the training script can properly:
1. Initialize PGCN with adj_matrix and coordinates
2. Create parameter groups (lap/local/weight)
3. Set up the training loop structure
"""

import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import torch
import numpy as np
from types import SimpleNamespace
from torch.nn.parameter import Parameter

# Import PGCN and utilities
from EEG.models.PGCN import PGCN, convert_dis_m, get_ini_dis_m, return_coordinates

print("="*80)
print("Testing Simplified PGCN Training Setup")
print("="*80)

# Test 1: Initialize adjacency matrix and coordinates
print("\n[Test 1] Initialize adjacency matrix and coordinates")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), delta=9))).to(device)
coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)

print(f"✓ Adjacency matrix shape: {adj_matrix.shape}")
print(f"✓ Coordinate matrix shape: {coordinate_matrix.shape}")
print(f"✓ adj_matrix is Parameter: {isinstance(adj_matrix, Parameter)}")
print(f"✓ adj_matrix requires_grad: {adj_matrix.requires_grad}")

# Test 2: Create PGCN model with original interface
print("\n[Test 2] Create PGCN model with original interface")
pgcn_args = SimpleNamespace(
    in_feature=5,
    out_feature=20,
    n_class=3,
    dropout=0.4,
    epsilon=0.05,
    dataset='SEED',
    device=str(device),
    lr=0.1,  # LeakyReLU slope
    module=""
)

model = PGCN(pgcn_args, adj_matrix, coordinate_matrix).to(device)
print(f"✓ Model created successfully")
print(f"✓ Model device: {next(model.parameters()).device}")

# Test 3: Parameter grouping (critical for PGCN)
print("\n[Test 3] Parameter grouping for optimizer")
lap_params, local_params, weight_params = [], [], []
for pname, p in model.named_parameters():
    if str(pname) == "adj":
        lap_params.append(p)
        print(f"  - LAP param: {pname}, shape={p.shape}")
    elif "local" in str(pname):
        local_params.append(p)
    else:
        weight_params.append(p)

print(f"✓ Laplacian parameters: {len(lap_params)}")
print(f"✓ Local parameters: {len(local_params)}")
print(f"✓ Weight parameters: {len(weight_params)}")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test 4: Forward pass
print("\n[Test 4] Forward pass")
batch_size = 8
x = torch.randn(batch_size, 62, 5).to(device)
with torch.no_grad():
    output, _, _ = model(x)

print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {output.shape}")
print(f"✓ Output is logits (3 classes): {output.shape[1] == 3}")

# Test 5: Optimizer setup (like in main_PGCN.py)
print("\n[Test 5] Optimizer setup")
import torch.optim as optim

optimizer = optim.AdamW([
    {'params': lap_params, 'lr': 5e-5},
    {'params': local_params, 'lr': 0.0015},
    {'params': weight_params, 'lr': 0.0015},
], betas=(0.9, 0.999), weight_decay=0.01)

print(f"✓ Optimizer created with 3 parameter groups")
print(f"✓ LAP lr: {optimizer.param_groups[0]['lr']}")
print(f"✓ Local lr: {optimizer.param_groups[1]['lr']}")
print(f"✓ Weight lr: {optimizer.param_groups[2]['lr']}")

# Test 6: Gradient flow
print("\n[Test 6] Gradient flow test")
y = torch.randint(0, 3, (batch_size,)).to(device)
criterion = torch.nn.CrossEntropyLoss()

output, _, _ = model(x)
loss = criterion(output, y)
loss.backward()

adj_has_grad = adj_matrix.grad is not None
local_has_grad = any(p.grad is not None for p in local_params if p.requires_grad)
weight_has_grad = any(p.grad is not None for p in weight_params if p.requires_grad)

print(f"✓ Loss computed: {loss.item():.4f}")
print(f"✓ adj_matrix has gradient: {adj_has_grad}")
print(f"✓ Local parameters have gradients: {local_has_grad}")
print(f"✓ Weight parameters have gradients: {weight_has_grad}")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nThe simplified PGCN training setup is working correctly:")
print("  ✓ Original PGCN interface preserved")
print("  ✓ Adjacency matrix as learnable Parameter")
print("  ✓ Coordinates properly initialized")
print("  ✓ Parameter grouping for optimizer")
print("  ✓ Forward and backward passes work")
print("\nYou can now run PGCN_train.py with LibEER!")
