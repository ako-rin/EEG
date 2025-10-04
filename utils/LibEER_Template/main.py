from LibEER.models.Models import Model
from LibEER.config.setting import Setting, preset_setting
from LibEER.data_utils.load_data import get_data
from LibEER.data_utils.split import merge_to_part, index_to_data, get_split_index
from LibEER.utils.args import get_args_parser
from LibEER.utils.store import make_output_dir
from LibEER.utils.utils import result_log, setup_seed
from LibEER.Trainer.training import train
from LibEER.models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score
import os
import inspect

# --- Runtime patch: extend LibEER Metric to support precision/recall ---
try:
    from LibEER.utils import metric as _metric_mod

    def _macro_precision(self):
        self.values['macro-precision'] = precision_score(self.targets, self.outputs, average='macro', zero_division=0)
        return self.values['macro-precision']

    def _micro_precision(self):
        self.values['micro-precision'] = precision_score(self.targets, self.outputs, average='micro', zero_division=0)
        return self.values['micro-precision']

    def _weighted_precision(self):
        self.values['weighted-precision'] = precision_score(self.targets, self.outputs, average='weighted', zero_division=0)
        return self.values['weighted-precision']

    def _macro_recall(self):
        self.values['macro-recall'] = recall_score(self.targets, self.outputs, average='macro', zero_division=0)
        return self.values['macro-recall']

    def _micro_recall(self):
        self.values['micro-recall'] = recall_score(self.targets, self.outputs, average='micro', zero_division=0)
        return self.values['micro-recall']

    def _weighted_recall(self):
        self.values['weighted-recall'] = recall_score(self.targets, self.outputs, average='weighted', zero_division=0)
        return self.values['weighted-recall']

    # Attach new methods to Metric
    _Metric = _metric_mod.Metric
    if not hasattr(_Metric, 'macro_precision'):
        _Metric.macro_precision = _macro_precision
        _Metric.micro_precision = _micro_precision
        _Metric.weighted_precision = _weighted_precision
        _Metric.macro_recall = _macro_recall
        _Metric.micro_recall = _micro_recall
        _Metric.weighted_recall = _weighted_recall

        # Override value() to support new metric keys while保持向后兼容
        _orig_value = _Metric.value
        def _value(self):
            # one-hot to index if needed
            if len(self.targets) > 0 and isinstance(self.targets[0], list):
                try:
                    self.targets = [self.targets[i].index(1) for i in range(len(self.targets))]
                except ValueError:
                    return "unavailable"
            out_parts = []
            for m in self.metrics:
                if m == 'acc':
                    v = self.accuracy()
                elif m in ('macro-f1', 'micro-f1', 'weighted-f1'):
                    v = getattr(self, m.replace('-', '_'))()
                elif m == 'ck':
                    v = self.ck_coe()
                elif m in (
                    'macro-precision','micro-precision','weighted-precision',
                    'macro-recall','micro-recall','weighted-recall',
                ):
                    v = getattr(self, m.replace('-', '_'))()
                else:
                    # fallback to original value() behavior for unknown keys
                    # temporarily set only this metric and delegate
                    prev = self.metrics
                    try:
                        self.metrics = [m]
                        return _orig_value(self)
                    finally:
                        self.metrics = prev
                out_parts.append(f"{m}: {v:.3f}")
            if len(self.losses) != 0:
                return "   ".join(out_parts) + f"   loss: {sum(self.losses)/len(self.losses):.4f}"
            return "   ".join(out_parts)
        _Metric.value = _value
except Exception:
    # If patch fails (e.g., during static analysis), skip silently; runtime will still work for built-ins
    pass

# --- Runtime patch: verbose preprocess logger (prints pipeline steps) ---
try:
    from LibEER.data_utils import preprocess as _pp
    if not hasattr(_pp, '_orig_preprocess'):
        _pp._orig_preprocess = _pp.preprocess

        def _preprocess_verbose_wrapper(*args, **kwargs):
            # allow toggling by env: export LIBEER_PREPROCESS_VERBOSE=0 to silence
            verbose_env = os.getenv('LIBEER_PREPROCESS_VERBOSE', '1')
            verbose = kwargs.pop('verbose', verbose_env != '0')
            try:
                sig = inspect.signature(_pp._orig_preprocess)
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
            except Exception:
                bound = None
            if verbose:
                print("[LibEER][preprocess] start")
                if bound is not None:
                    b = bound.arguments
                    # Key switches
                    print(f"  baseline: {'on' if b.get('baseline', None) is not None else 'off'}")
                    print(f"  pass_band: {b.get('pass_band', None)}  eog_clean: {b.get('eog_clean', None)}")
                    print(f"  feature_type: {b.get('feature_type', None)}  only_seg: {b.get('only_seg', None)}")
                    print(f"  time_window: {b.get('time_window', None)}  overlap: {b.get('overlap', None)}")
                    print(f"  sample_length: {b.get('sample_length', None)}  stride: {b.get('stride', None)}")
            out = _pp._orig_preprocess(*args, **kwargs)
            if verbose:
                # try to infer feature_dim
                try:
                    data_out, feature_dim = out
                    print(f"[LibEER][preprocess] done, feature_dim={feature_dim}")
                except Exception:
                    print("[LibEER][preprocess] done")
            return out

        _pp.preprocess = _preprocess_verbose_wrapper
except Exception:
    pass


def main(args):
    setting = Setting(dataset='seed_de',  # Select the dataset
                      dataset_path='/home/ako/Project/datasets/SEED/',  # Specify the path to the corresponding dataset.
                      pass_band=[0.3, 50],  # use a band-pass filter with a range of 0.3 to 50 Hz,
                      extract_bands=[[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]],
                      # Set the frequency bands for extracting frequency features.
                      time_window=1,  # Set the time window for feature extraction to 1 second.
                      overlap=0,  # The overlap length of the time window for feature extraction.
                      sample_length=1,
                      # Use a sliding window to extract the features with a window size of sample_length set to 1 and a step size of 1.
                      stride=1,
                      seed=2024,  # set up the random seed
                      feature_type='de_lds',  # set the feature type to extract (using DE features: LDS only; no re-extraction)
                      label_used=['valence'], # specify the label used (ignored for SEED/SEED-V discrete labels)
                      bounds=[5,5], # For DEAP/DREAMER continuous labels; ignored for SEED/SEED-V
                      experiment_mode="subject-dependent",
                      split_type='early-stop',
                      test_size=0.2,
                      val_size=0.2)
    setup_seed(2024) # Set the random seed for the experiment to ensure reproducibility.
    data, label, channels, feature_dim, num_classes = get_data(setting) # Get the corresponding data and information based on the setting class.
    # The organization of data and label is [session(1), subject(32), trial(40), sample(XXX)].
    data, label = merge_to_part(data, label, setting) # Merge the data based on the experiment task specified in the setting class.
    # After the merge_to_part() function and the specified subject-independent method, the organization of data and label will be [[subject(32), trail(40), sample(xxx)]].
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 优先用 GPU
    best_metrics = [] # Prepare to record the experimental results.
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1): # This loop will only execute 32 times; it will be enabled only under the subject-dependent task.
        tts = get_split_index(data_i, label_i, setting) # Get the task indexes for the experiment based on the setting class. The leave-one-out splitting method was chosen.
        # Here, in tts:
        # train indexes:[2, 15, 4, 17, 5, 22, 39, 20, 23, 7, 18, 14, 35, 28, 12, 3, 33, 31, 36, 11, 32, 13, 9, 24], val indexes:[1, 19, 25, 16, 27, 29, 8, 6], test indexes:[0, 21, 26, 30, 10, 38, 37, 34]
        # train indexes:[0, 19, 1, 23, 8, 13, 10, 17, 18, 3, 11, 2, 24, 22, 29, 38, 26, 33, 28, 37, 34, 36, 5, 20], val indexes:[35, 39, 14, 15, 6, 21, 32, 4], test indexes:[25, 7, 16, 12, 27, 9, 31, 30]
        # ...
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed) # Set the random seed again to ensure reproducibility.
            # some splitters may provide empty val_indexes; guard against it
            if not val_indexes or (len(val_indexes) > 0 and val_indexes[0] == -1):
                # normalize empty to [-1] so downstream code can check
                if not val_indexes:
                    val_indexes = [-1]
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # Retrieve the corresponding data based on the indexes. train_data contains data from 24 trails, val_data contains data from 8 trails, and test_data contains data from other 8 trails.
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes)
            # If using raw_tc (time x channel), transpose to (channel x time) for CDCN and similar models
            if getattr(setting, 'feature_type', None) == 'raw_tc':
                if len(train_data) > 0 and train_data[0].ndim == 3:
                    train_data = np.transpose(train_data, (0, 2, 1))
                if len(val_data) > 0 and val_data[0].ndim == 3:
                    val_data = np.transpose(val_data, (0, 2, 1))
                if len(test_data) > 0 and test_data[0].ndim == 3:
                    test_data = np.transpose(test_data, (0, 2, 1))
            # model to train
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            # Choose a model. Alternatively, you can use the method below to import the DGCNN model:
            # model = DGCNN(channels, feature_dim, num_classes)
            # You can configure the model parameters in model_param/DGCNN.yaml
            model = Model['DGCNN'](channels, feature_dim, num_classes).to(device)

            # Prepare the corresponding dataloader.
            X_train = torch.as_tensor(train_data, dtype=torch.float32)
            X_val   = torch.as_tensor(val_data,   dtype=torch.float32)
            X_test  = torch.as_tensor(test_data,  dtype=torch.float32)

            # CrossEntropyLoss 需要 Long 类型且为 1D 类别索引
            y_train = torch.as_tensor(train_label, dtype=torch.long).squeeze(-1)
            y_val   = torch.as_tensor(val_label,   dtype=torch.long).squeeze(-1)
            y_test  = torch.as_tensor(test_label,  dtype=torch.long).squeeze(-1)

            dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
            dataset_val   = torch.utils.data.TensorDataset(X_val,   y_val)
            dataset_test  = torch.utils.data.TensorDataset(X_test,  y_test)

            # Select an appropriate optimizer.
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            # Select appropriate loss functions. The first is a classification loss function, and the second is the L2 regularization loss in DGCNN.
            criterion = nn.CrossEntropyLoss()
            loss_func = NewSparseL2Regularization(0.01).to(device)
            # Specify the output_dir, mainly for saving intermediate results during model training. It is set based on args but may show errors currently.
            output_dir = make_output_dir(args, "CDCN")
            # Call the training function to train. Batch size, epochs, etc., can be set via command-line parameters, or manually if desired.
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, loss_func=loss_func, loss_param=model)
            best_metrics.append(round_metric)
    # best metrics: every round metrics dict
    result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
