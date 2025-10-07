#-------------------------------------
# 主程序入口
# Date: 2024.4.24
# Author: Ming Jin
# All Rights Reserved
#-------------------------------------
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import torch
import datetime
import shutil
import sys
import os

# from chord import Chord

from torch.autograd import Variable
import torch.optim as optim
# import torch.optim.RAd
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter
import pytorch_warmup as warmup

from torch.nn import functional as F

# write_path = SummaryWriter('/home/ming/workspace/test_GAAT/GAAT/tensorboard')
from pathlib import Path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]  # /home/ako/Project/work
_libeer_root = _proj_root / 'lib' / 'libeer-ako'
for p in (str(_proj_root), str(_libeer_root)):
    if p not in sys.path:
        sys.path.insert(0, p)
from EEG.utils.args import *
from EEG.models.PGCN import PGCN,convert_dis_m,get_ini_dis_m,return_coordinates,normalize_adj

# Dependent Mode
def load_data_de(path, subject):
    dict_load = np.load(os.path.join(path, (str(subject))), allow_pickle=True)
    data = dict_load[()]['sample']
    label = dict_load[()]['label']
    split_index = dict_load[()]["clip"]
    x_tr = data[:split_index]
    x_ts = data[split_index:]
    y_tr = label[:split_index]
    y_ts = label[split_index:]
    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }
    return data_and_label

# Independent Mode
def load_data_inde(path, subject):
    x_tr = np.array([])
    y_tr = np.array([])
    x_ts = np.array([])
    y_ts = np.array([])
    for i_subject in os.listdir(path):
        dict_load = np.load(os.path.join(path, (str(i_subject))), allow_pickle=True)
        data = dict_load[()]['sample']
        label = dict_load[()]['label']
        if i_subject == subject:
            x_ts = data
            y_ts = label
        else:
            if x_tr.shape[0] == 0:
                x_tr = data
                y_tr = label
            else:
                x_tr = np.append(x_tr, data, axis=0)
                y_tr = np.append(y_tr, label, axis=0)
    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }
    return data_and_label

# Loss Function
class CE_Label_Smooth_Loss(torch.nn.Module):
    def __init__(self, classes=4, epsilon=0.14, ):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

# Log Config
def set_logging_config(logdir):
    """
    logging configuration
    :param logdir:
    :return:
    """
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.Formatter.converter = beijing
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])

# CheckPoint Mode
def save_checkpoint(state, is_best, dir, subject_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'),
                        os.path.join(f'{dir}', f'{subject_name}_model_best.pth.tar'))


# from torchsummary import summary
np.set_printoptions(threshold=np.inf)

from thop import profile
from ptflops import get_model_complexity_info


# 一次一个Session
class Trainer(object):

    def __init__(self, args, subject_name):
        self.args = args
        self.subject_name = subject_name

    def train(self, data_and_label):
        logger = logging.getLogger("train")
        laplacian_array = []  # 存放该subject优化后的laplacian matrix 列表
        # data_and_label中将数据划分出来，一部分给train一部分给val
        # x_tr：训练集
        # y_tr：训练集label
        train_set = TensorDataset((torch.from_numpy(data_and_label["x_tr"])).type(torch.FloatTensor),
                                  (torch.from_numpy(data_and_label["y_tr"])).type(torch.FloatTensor))
        val_set = TensorDataset((torch.from_numpy(data_and_label["x_ts"])).type(torch.FloatTensor),
                                (torch.from_numpy(data_and_label["y_ts"])).type(torch.FloatTensor))

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        # 局部视野的邻接矩阵
        adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9))).to(self.args.device)

        # 返回节点的绝对坐标
        coordinate_matrix = torch.FloatTensor(return_coordinates()).to(self.args.device)


        #####################################################################################
        # 2.define model
        #####################################################################################
        model = PGCN(self.args, adj_matrix, coordinate_matrix)

        lap_params, local_params, weight_params = [], [], []
        for pname, p in model.named_parameters():
        #     print(pname)
            if str(pname) == "adj":
                lap_params += [p]
            elif "local" in str(pname):
                local_params += [p]
            else :
                weight_params += [p]
        for n, p in model.named_parameters():
            print(n, p.shape, p.requires_grad)
        optimizer = optim.AdamW([
            {'params': lap_params, 'lr': self.args.beta},
            {'params': local_params, 'lr': self.args.lr},
            {'params': weight_params, 'lr': self.args.lr},
        ], betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        _loss = CE_Label_Smooth_Loss(classes=self.args.n_class, epsilon=self.args.epsilon).to(self.args.device)
        model = model.to(self.args.device)


        #############################################################################
        # 3.start train
        #############################################################################
        train_epoch = self.args.epochs

        # warm up
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[train_epoch // 3],
                                                            gamma=0.1)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        warmup_scheduler.last_step = -1

        best_val_acc = 0

        for epoch in range(train_epoch):
            epoch_start_time = time.time()
            train_acc = 0
            train_loss = 0
            val_loss = 0
            val_acc = 0

            model.train()
            for i, (x, y) in enumerate(train_loader):
                model.zero_grad()  # 清空上一步残余更新参数值

                x, y = x.to(self.args.device), y.to(device=self.args.device, dtype=torch.int64)
                output, lap_1, _ = model(x)
                loss = _loss(output, y)
                loss.backward()  # 误差反向传播，计算参数更新值

                optimizer.step()  # 将参数更新值施加到net的parmeters上

                if i < len(train_loader) - 1:
                    with warmup_scheduler.dampening():
                        pass

                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
                train_loss += loss.item() * y.size(0)

            train_acc = train_acc / train_set.__len__()
            train_loss = train_loss / train_set.__len__()

            with warmup_scheduler.dampening():
                lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for j, (a, b) in enumerate(val_loader):
                    a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                    output, lap, fused_feature = model(a)

                    val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())
                    batch_loss = _loss(output, b)
                    val_loss += batch_loss.item() * b.size(0)

            val_acc = round(float(val_acc / val_set.__len__()), 4)
            val_loss = round(float(val_loss / val_set.__len__()), 4)

            is_best_acc = 0
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                is_best_acc = 1

            if epoch == 0:
                logger.info(self.args)

            if epoch % 5 == 0:
                logger.info("val acc and loss on epoch_{} are: {} and {}".format(epoch, val_acc, val_loss))

            save_checkpoint({
                'iteration': epoch,
                'enc_module_state_dict': model.state_dict(),
                'test_acc': val_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best_acc, self.args.log_dir, self.subject_name)

            if best_val_acc == 1:
                break
        # self.writer.close()
        return best_val_acc, laplacian_array




def main():
    args = parse_args()
    print("")
    print(f"Current device is {args.device}.")

    # 固定随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 将当前的实验结果存储到按秒组织的文件夹中
    datatime_path = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d-%H:%M:%S')
    args.log_dir = os.path.join(args.log_dir, args.dataset, datatime_path)
    set_logging_config(args.log_dir)
    logger = logging.getLogger("main")
    logger.info("Logs and checkpoints will be saved to：{}".format(args.log_dir))

    # summary(model, input_size=(62, 5))

    acc_list = []
    acc_dic = {}
    count = 0
    true_path = os.path.join('/home/ako/Project/work/Reference/PGCN/npy_data/seed/', str(1))
    for subject in os.listdir(true_path):
        
        # 15个受试者分别轮询，并在每个受试者内划分数据来训练
        
        count += 1

        # load data of every single subject, 对于de和inde的设定，所读取的数据有所区别
        data_and_label = None
        
        # 移除后缀.npy，保留受试者编号
        subject_name = str(subject).strip('.npy')
        
        
        if args.mode == "dependent":
            
            # 依赖受试实验就是在单个受试者内划分数据
            logger.info(f"Dependent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_de(true_path, subject)
        elif args.mode == "independent":
            logger.info(f"Independent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_inde(true_path, subject)
        else:
            raise ValueError("Wrong mode selected.")

        trainer = Trainer(args, subject_name) # 传递受试者编号仅传给log
        
        # 将划分好的数据集传入给训练函数
        valAcc, lap_array = trainer.train(data_and_label)

        acc_list.append(valAcc)
        lap_array = np.array(lap_array)
        acc_dic[subject_name] = valAcc
        logger.info("Current best acc is : {}".format(acc_dic))
        logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof=1)))


if __name__ == "__main__":
    main()