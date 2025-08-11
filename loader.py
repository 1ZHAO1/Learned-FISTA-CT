# -*- coding: utf-8 -*-

# =============================================================================
#
# 文件名: loader.py (新版本)
#
# 目标:
#   定义一个自定义的PyTorch Dataset类，名为CTDataset。
#   这个类负责从硬盘读取由generate_data.py创建的数据对
#   （FBP重建图和金标准图），并将它们转换成PyTorch张量，
#   以供训练时使用。
#
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class CTDataset(Dataset):
    """自定义CT重建数据集"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含 .npy 文件的目录 (例如 './dataset/train')。
            transform (callable, optional): 可选的转换操作，应用在样本上。
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 查找所有fbp文件，并以此为基础构建文件列表
        # 我们假设每个fbp文件都有一个对应的gt文件
        self.fbp_files = sorted(glob(os.path.join(self.root_dir, '*_fbp.npy')))
        if not self.fbp_files:
            raise RuntimeError(f"No FBP .npy files found in {self.root_dir}")

    def __len__(self):
        """返回数据集的总大小"""
        return len(self.fbp_files)

    # [UPDATED] - loader.py中的__getitem__方法
# loader.py -> __getitem__() method [FINAL ROBUST VERSION]

    def __getitem__(self, idx):
        """
        根据索引idx，加载并返回一个数据样本。
        """
        fbp_path = self.fbp_files[idx]
        gt_path = fbp_path.replace('_fbp.npy', '_gt.npy')
        sino_path = fbp_path.replace('_fbp.npy', '_sino.npy')

        fbp_image = np.load(fbp_path)
        gt_image = np.load(gt_path)
        sino_data = np.load(sino_path)

        fbp_tensor = torch.from_numpy(fbp_image).float()
        gt_tensor = torch.from_numpy(gt_image).float()
        sino_tensor = torch.from_numpy(sino_data).float()

        # [NEW ROBUST NORMALIZATION] - Normalize and then clamp the values.
        # This is a much more stable way to prepare data for the network.
        # 1. Normalize by max value (or a global max if known).
        fbp_tensor = fbp_tensor / (fbp_tensor.max() + 1e-8)
        gt_tensor = gt_tensor / (gt_tensor.max() + 1e-8)
        # Sinogram data has a different scale, simple max normalization is fine for it.
        sino_tensor = sino_tensor / (sino_tensor.max() + 1e-8)

        # 2. Clamp image tensors to [0, 1] to remove any extreme values from FBP.
        fbp_tensor = torch.clamp(fbp_tensor, 0, 1)
        gt_tensor = torch.clamp(gt_tensor, 0, 1)

        # 3. Add channel dimension
        fbp_tensor = fbp_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
        
        sample = {'fbp': fbp_tensor, 'gt': gt_tensor, 'sino': sino_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample

# --- 你可以在这里添加一个小的测试代码块来验证loader是否工作正常 ---
if __name__ == '__main__':
    # 这个测试只有在你先运行了generate_data.py之后才能成功
    print("Running a quick test for CTDataset...")
    
    # 假设数据集在 './dataset/train'
    train_dataset_path = os.path.join('./dataset', 'train')

    if not os.path.exists(train_dataset_path):
        print(f"Error: Test failed. Dataset path does not exist: {train_dataset_path}")
        print("Please run generate_data.py first.")
    else:
        dataset = CTDataset(root_dir=train_dataset_path)
        
        print(f"Dataset size: {len(dataset)}")
        
        # 取出第一个样本进行检查
        first_sample = dataset[0]
        fbp_tensor = first_sample['fbp']
        gt_tensor = first_sample['gt']
        
        print(f"Sample 0 FBP tensor shape: {fbp_tensor.shape}") # 应该输出 torch.Size([1, 128, 128])
        print(f"Sample 0 GT tensor shape: {gt_tensor.shape}")   # 应该输出 torch.Size([1, 128, 128])
        print("Test successful!")