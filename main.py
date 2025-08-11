# -*- coding: utf-8 -*-
# [MODIFIED] - This file has been completely refactored for our CT reconstruction project.
import numpy as np
import astra
import torch
import argparse
import os
from os.path import join as pjoin
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




# [MODIFIED] 1. 导入我们自己的模块和PyTorch的核心组件
from torch.utils.data import DataLoader
from loader import CTDataset # 我们新的数据加载器
from solver import Solver     # 我们适配好的训练器

# [MODIFIED] 2. 动态导入模型，而不是写死
# 这样可以轻松地添加或切换模型
from fista_net_plus import FISTANetPlus as M5FISTANetPlus # 以这个最复杂的模型为例
# from M5FISTANet import FISTANet as M5FISTANet
# from M4ISTANet import ISTANet as M4ISTANet

def main(config):
    # =========================================================================
    # [MODIFIED] 3. 创建保存结果和模型的文件夹
    # =========================================================================
    # 基于模型名称创建独立的文件夹，方便管理
    save_path = pjoin(config.result_path, config.model_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results and models will be saved in: {os.path.abspath(save_path)}")

    # =========================================================================
    # [MODIFIED] 4. 设置设备 (CPU or GPU)
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # =========================================================================
    # [MODIFIED] 5. 加载我们的CT数据集
    # 我们不再使用原作者的DataSplit，而是用我们自己的CTDataset和DataLoader
    # =========================================================================
    print("Loading dataset...")
    train_path = pjoin(config.data_path, 'train')
    test_path = pjoin(config.data_path, 'test')

    train_dataset = CTDataset(root_dir=train_path)
    test_dataset = CTDataset(root_dir=test_path)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size, # 使用与训练相同的batch_size
                             shuffle=False,
                             num_workers=config.num_workers)
    print("Dataset loaded successfully.")
    
    # [KEPT] 6. 为solver准备一个固定的测试集
    # 原作者的solver设计期望一个固定的测试集张量，我们遵循这个设计。
    # 我们从test_loader中取出一批数据作为代表性的测试集。
    print("Preparing fixed test batch for solver...")
    fixed_test_batch = next(iter(test_loader))
    test_images = fixed_test_batch['gt']
    # test_fbp_images = fixed_test_batch['fbp']
    test_sinograms = fixed_test_batch['sino']


    # =========================================================================
    # [NEW] 7a. 为CT模型创建几何对象
    # =========================================================================
    # 我们需要和generate_data.py中完全一致的几何参数
    IMAGE_SIZE = 128
    DETECTOR_COUNT = 192
    DETECTOR_SPACING = 1.0
    PROJECTION_ANGLES = np.linspace(0, np.pi * 2/3, 180, False)

    vol_geom = astra.create_vol_geom(IMAGE_SIZE, IMAGE_SIZE)
    proj_geom = astra.create_proj_geom('parallel', DETECTOR_SPACING, DETECTOR_COUNT, PROJECTION_ANGLES)


    # =========================================================================
    # [MODIFIED] 7b. 根据命令行参数选择并初始化模型
    # =========================================================================
    print(f"Initializing model: {config.model_name}")
    model = None
    if config.model_name == 'M5FISTANetPlus':
        # [MODIFIED] - 现在我们用正确的参数来初始化模型
        from fista_net_plus import FISTANetPlus # 确保导入的是新版
        model = FISTANetPlus(LayerNo=9, proj_geom=proj_geom, vol_geom=vol_geom)

    # elif config.model_name == 'M5FISTANet':
    #     # model = M5FISTANet(...) # 也可以用同样的方式改造其他模型
    else:
        raise NotImplementedError(f"Model {config.model_name} is not implemented in main.py")

    model = model.to(device)
    # =========================================================================
    # [MODIFIED] 8. 初始化我们修改过的Solver
    # =========================================================================
    # 注意：我们将config对象直接传递给solver，让solver自己去取需要的参数。
    # 这比传递一长串参数要整洁得多。
    # 我们需要对solver的__init__做一点小小的修改来接收config。
    # 为了减少修改，我们还是按原样传递参数。
    solver = Solver(model=model, 
                    data_loader=train_loader, 
                    args=config, 
                    test_data=test_sinograms, # solver期望的test_data是sinogram
                    test_images=test_images) # solver期望的test_images是ground truth


    # =========================================================================
    # [MODIFIED] 9. 根据模式启动训练或测试
    # =========================================================================
    if config.mode == 'train':
        print("Starting training...")
        solver.train()
    elif config.mode == 'test':
        print("Starting testing...")
        # 注意：solver.test()也需要修改才能正确工作。
        # 我们主要关注训练流程。
        solver.test()


if __name__ == '__main__':
    # =========================================================================
    # [MODIFIED] 10. 创建一个统一的、强大的命令行参数解析器
    # 这是最核心的修改，它让我们的脚本变得专业和可配置。
    # =========================================================================
    parser = argparse.ArgumentParser()

    # --- 模式和路径参数 ---
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test mode')
    parser.add_argument('--model_name', type=str, default='M5FISTANetPlus', help='The name of the model to use')
    parser.add_argument('--data_path', type=str, default='./dataset', help='Path to the dataset directory')
    parser.add_argument('--result_path', type=str, default='./results', help='Path to save results and models')
    
    # --- 训练超参数 ---
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=147, help='Epoch to start training from (for resuming)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # --- 日志和测试参数 ---
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')
    parser.add_argument('--test_epoch', type=int, default=100, help='The epoch of the model to load for testing')
    
    # --- GPU设置 ---
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Use multiple GPUs if available')
    
    # 解析参数
    config = parser.parse_args()
    
    # 将设备信息添加到config中，方便solver使用
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印所有配置信息
    print("Configuration:")
    print(config)

    # 启动主函数
    main(config)