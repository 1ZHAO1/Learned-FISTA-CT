# -*- coding: utf-8 -*-
# [MODIFIED] - This file has been adapted for our data pipeline.

# from loader import CTDataset # [REMOVED] - This import is no longer needed here, main.py or train.py will handle it.
import torch.optim as optim
# from M1LapReg import callLapReg # [REMOVED] - We no longer use this on-the-fly calculation.
import torch
import torch.nn as nn
from os.path import dirname, join as pjoin
from collections import OrderedDict
import time
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import os

# [KEPT] - Loss functions remain unchanged.
def l1_loss(pred, target, l1_weight):
    """
    Compute L1 loss;
    l1_weigh default: 0.1
    """
    err = torch.mean(torch.abs(pred - target))
    err = l1_weight * err
    return err

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class Solver(object):
    def __init__(self, model, data_loader, args, test_data, test_images):
        # [MODIFIED] - Updated the list of supported models.
        # [NOTE] - Added M5FISTANetPlus based on our main.py
        assert args.model_name in ['FBPConv', 'ISTANet', 'FISTANet', 'M5FISTANet', 'M5FISTANetPlus']

        self.model_name = args.model_name
        self.model = model
        self.data_loader = data_loader
        
        # [FIXED] - 将 args.data_dir 改为 args.data_path，与 main.py 保持一致
        self.data_dir = args.data_path
        
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        
        # [KEPT] - Optimizer setup is complex and model-specific, kept as is.
        # It assumes FISTANet, M5FISTANet, etc., have similar parameter naming conventions.
        if 'FISTANet' in self.model_name:
            # set different lr for regularization weights and network weights
            # 注意: 这里的参数名可能需要根据 M5FISTANetPlus.py 的实际定义微调
            self.optimizer = optim.Adam([
            {'params': self.model.fcs.parameters()}, 
            {'params': self.model.w_theta, 'lr': 0.001},
            {'params': self.model.b_theta, 'lr': 0.001},
            {'params': self.model.w_mu, 'lr': 0.001},
            {'params': self.model.b_mu, 'lr': 0.001},
            {'params': self.model.w_rho, 'lr': 0.001},
            {'params': self.model.b_rho, 'lr': 0.001}], 
            lr=self.lr, weight_decay=0.001)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9) # step-wise


        self.save_path = pjoin(args.result_path, args.model_name) # [MODIFIED] - 使用新的路径参数
        self.multi_gpu = args.multi_gpu
        self.device = args.device
        self.log_interval = args.log_interval
        self.test_epoch = args.test_epoch
        self.test_data = test_data
        self.test_images = test_images
        self.train_loss = nn.MSELoss()

    def save_model(self, iter_):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
        torch.save(self.model.state_dict(), f)
    
    def load_model(self, iter_):
        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.model.load_state_dict(state_d)
        else:
            self.model.load_state_dict(torch.load(f))
            
    
    def train(self):
        os.makedirs('./figures', exist_ok=True)
        train_losses = []
        start_time = time.time()
        # set up Tensorboard
        #writer = SummaryWriter('runs/'+self.model_name)
        
        # ========================================================================================== #
        # [CORE MODIFICATION] The entire training loop logic is adapted to our new data pipeline.
        # ========================================================================================== #

        for epoch in range(1 + self.start_epoch, self.num_epochs + 1 + self.start_epoch):
            self.model.train(True)
                  
            for batch_idx, data_batch in enumerate(self.data_loader):
                
                x_img = data_batch['fbp'].to(self.device)
                y_target = data_batch['gt'].to(self.device)
                x_in = data_batch['sino'].to(self.device)

                # --- [MODIFIED] The Failsafe Logic (保险丝逻辑) ---
                
                # 1. 计算前向传播和损失
                if 'FISTANet' in self.model_name:
                    [pred, loss_layers_sym, loss_st] = self.model(x_img, x_in)
                    loss_discrepancy = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_layers_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
                    sparsity_constraint = 0
                    for k, _ in enumerate(loss_st, 0):
                        sparsity_constraint += torch.mean(torch.abs(loss_st[k]))
                    loss = loss_discrepancy +  0.01 * loss_constraint + 0.001 * sparsity_constraint
                else: # For other models like FBPConv, ISTANet
                    # You can add logic for other models here if needed
                    # For now, let's assume we only run FISTANetPlus
                    pred = self.model(x_img) # This might need adjustment for other models
                    loss = self.train_loss(pred, y_target)

                # 2. 检查损失值是否有效，如果无效则跳过此batch
                if torch.isinf(loss) or torch.isnan(loss):
                    print(f"!! Epoch {epoch}, Batch {batch_idx}: Invalid loss detected (inf or nan). Skipping update.")
                    continue # 跳过这个batch，不进行反向传播和优化

                # 3. 如果损失有效，则执行反向传播和优化
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                train_losses.append(loss.item())
                
                # --- End of Failsafe Logic ---

                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\t TIME:{:.1f}s'
                          ''.format(epoch, batch_idx * self.data_loader.batch_size,
                                    len(self.data_loader.dataset),
                                    100. * batch_idx / len(self.data_loader),
                                    loss.item(), # 使用 .item() 获取数值
                                    time.time() - start_time))

                    if 'FISTANet' in self.model_name:
                        # 使用 .item() 或 .data.mean() 来安全地打印参数值
                        print("Threshold value w: {}".format(self.model.w_theta.data.mean().item()))
                        print("Gradient step w: {}".format(self.model.w_mu.data.mean().item()))

            if epoch % 1 == 0:
                self.save_model(epoch)
                np.save(pjoin(self.save_path, 'loss_{}_epoch.npy'.format(epoch)), np.array(train_losses))    