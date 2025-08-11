import numpy as np
import astra
import torch
import argparse
import os
from os.path import join as pjoin
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from torch.utils.data import DataLoader
from loader import CTDataset
from solver import Solver
from fista_net_plus import FISTANetPlus as M5FISTANetPlus
from simple_unet import SimpleUNet

# (如果需要) 从config.py导入几何参数
# from config import *

def main(config):
    save_path = pjoin(config.result_path, config.model_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results and models will be saved in: {os.path.abspath(save_path)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_path = pjoin(config.data_path, 'train')
    test_path = pjoin(config.data_path, 'test')
    train_dataset = CTDataset(root_dir=train_path)
    test_dataset = CTDataset(root_dir=test_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print("Dataset loaded successfully.")
    
    print("Preparing fixed test batch for solver...")
    fixed_test_batch = next(iter(test_loader))
    test_images = fixed_test_batch['gt']
    test_sinograms = fixed_test_batch['sino']

    IMAGE_SIZE = 128
    DETECTOR_COUNT = 192
    DETECTOR_SPACING = 1.0
    PROJECTION_ANGLES = np.linspace(0, np.pi * 2/3, 180, False)
    vol_geom = astra.create_vol_geom(IMAGE_SIZE, IMAGE_SIZE)
    proj_geom = astra.create_proj_geom('parallel', DETECTOR_SPACING, DETECTOR_COUNT, PROJECTION_ANGLES)

    print(f"Initializing model: {config.model_name}")
    model = None
    if config.model_name == 'M5FISTANetPlus':
        model = M5FISTANetPlus(LayerNo=9, proj_geom=proj_geom, vol_geom=vol_geom)
    elif config.model_name == 'SimpleUNet':
        model = SimpleUNet(in_channels=1, out_channels=1)
    else:
        raise NotImplementedError(f"Model {config.model_name} is not implemented in main.py")
    model = model.to(device)

    if config.mode == 'train':
        active_loader = train_loader
    else:
        active_loader = test_loader
        
    solver = Solver(model=model, 
                    data_loader=active_loader, 
                    args=config, 
                    test_data=test_sinograms,
                    test_images=test_images)

    if config.mode == 'train':
        print("Starting training...")
        solver.train()
    elif config.mode == 'test':
        print("Starting testing...")
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test mode')
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='SimpleUNet',
        choices=['M5FISTANetPlus', 'SimpleUNet'],
        help='The name of the model to use'
    )
    parser.add_argument('--data_path', type=str, default='./dataset', help='Path to the dataset directory')
    parser.add_argument('--result_path', type=str, default='./results', help='Path to save results and models')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from (for resuming)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')
    parser.add_argument('--test_epoch', type=int, default=100, help='The epoch of the model to load for testing')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Use multiple GPUs if available')
    config = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Configuration:")
    print(config)
    main(config)