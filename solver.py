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
import torch
from os.path import join as pjoin
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
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


    def test(self):
        """
        对整个测试集进行完整的评估，并保存结果。
        """
        # 1. 加载最佳模型
        try:
            # self.test_epoch 由 --test_epoch 命令行参数设置
            self.load_model(self.test_epoch)
            print(f"从 epoch {self.test_epoch} 加载模型成功。")
        except Exception as e:
            print(f"从 epoch {self.test_epoch} 加载模型时出错: {e}")
            return

        self.model.eval()  # 设置为评估模式

        # 准备一个文件夹来保存可视化结果
        vis_path = pjoin(self.save_path, 'test_visualizations')
        os.makedirs(vis_path, exist_ok=True)
        
        # 用于存储所有样本分数的列表
        all_psnr_scores = []
        all_ssim_scores = []
        
        # 使用 torch.no_grad() 来节约显存并加速
        with torch.no_grad():
            # 遍历测试数据加载器
            for i, data_batch in enumerate(self.data_loader):
                
                # 从数据加载器获取数据
                x_fbp = data_batch['fbp'].to(self.device)
                y_sino = data_batch['sino'].to(self.device)
                gt_image_batch = data_batch['gt'] # GT（真实图像）保留在CPU上，格式为Numpy，方便计算

                # 模型进行前向传播（重建）
                # 注意：模型的输出是 PyTorch Tensor，在GPU上
                reconstructed_batch, _, _ = self.model(x_fbp, y_sino)

                # 将重建结果和输入FBP图从GPU转到CPU，并转为Numpy数组
                reconstructed_batch_np = reconstructed_batch.cpu().numpy()
                x_fbp_np = x_fbp.cpu().numpy()
                
                # 对一个batch里的每张图片进行评估
                for j in range(reconstructed_batch_np.shape[0]):
                    gt_img = gt_image_batch[j].squeeze()
                    recon_img = reconstructed_batch_np[j].squeeze()

                    # 计算 PSNR 和 SSIM
                    current_psnr = psnr(gt_img, recon_img, data_range=gt_img.max() - gt_img.min())
                    current_ssim = ssim(gt_img, recon_img, data_range=gt_img.max() - gt_img.min())
                    
                    all_psnr_scores.append(current_psnr)
                    all_ssim_scores.append(current_ssim)

                    # 只保存前5个batch的可视化结果，避免文件过多
                    if i < 5:
                        fbp_img = x_fbp_np[j].squeeze()
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        
                        axes[0].imshow(gt_img, cmap='gray')
                        axes[0].set_title('Ground Truth (真实图像)')
                        axes[0].axis('off')
                        
                        axes[1].imshow(fbp_img, cmap='gray')
                        axes[1].set_title('FBP (输入)')
                        axes[1].axis('off')
                        
                        axes[2].imshow(recon_img, cmap='gray')
                        axes[2].set_title(f'Model Output (模型输出)\nPSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}')
                        axes[2].axis('off')

                        plt.savefig(pjoin(vis_path, f'comparison_batch_{i}_img_{j}.png'))
                        plt.close(fig)

        # 计算并打印平均分
        avg_psnr = np.mean(all_psnr_scores)
        avg_ssim = np.mean(all_ssim_scores)
        
        print("\n--- 评估完成 ---")
        print(f"在 {len(all_psnr_scores)} 张测试图像上进行了评估。")
        print(f"平均峰值信噪比 (Average PSNR): {avg_psnr:.2f} dB")
        print(f"平均结构相似性 (Average SSIM): {avg_ssim:.4f}")
        print("---------------------------\n")