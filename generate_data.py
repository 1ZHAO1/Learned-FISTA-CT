# -*- coding: utf-8 -*-

# =============================================================================
#
# 文件名: generate_data.py
#
# 目标:
#   为我们的深度学习模型批量生成并保存训练和测试数据。
#   每一对数据包含：
#     1. ground_truth: 随机生成的、清晰的原始体模图像。
#     2. fbp_recon: 对该体模进行有限角扫描后，用FBP算法重建出的、带伪影的低质量图像。
#   这些数据将被保存为.npy文件，以便后续的loader.py读取。
#
# =============================================================================

import os
import numpy as np
import astra
from skimage.draw import ellipse
from tqdm import tqdm # 用于显示一个漂亮的进度条

# -----------------------------------------------------------------------------
# 步骤 0: 配置生成参数
# -----------------------------------------------------------------------------
print(">> Step 0: Configuring data generation parameters...")

# 数据集参数
NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 200
OUTPUT_DIR = './dataset' # 数据集保存的总目录

# 图像和几何设置
IMAGE_SIZE = 128  # 为了加快训练速度，我们先使用128x128的图像尺寸
DETECTOR_COUNT = 192
DETECTOR_SPACING = 1.0

# 有限角扫描设置 (120度, 180个投影角度)
PROJECTION_ANGLES = np.linspace(0, np.pi * 2/3, 180, False)

# -----------------------------------------------------------------------------
# 核心函数 1: 创建随机体模
# 这是至关重要的一步。为了让模型学会通用的伪影去除能力，
# 我们不能只用同一个Shepp-Logan体模。我们需要生成大量随机的、
# 结构各不相同的体模。这里我们用随机椭圆来模拟。
# -----------------------------------------------------------------------------
def generate_random_phantom(size):
    """
    生成一个包含随机椭圆的体模图像
    """
    image = np.zeros((size, size), dtype=np.float32)
    num_ellipses = np.random.randint(5, 15) # 每个体模包含5到15个椭圆

    for _ in range(num_ellipses):
        # 随机位置
        center_x = np.random.randint(size * 0.1, size * 0.9)
        center_y = np.random.randint(size * 0.1, size * 0.9)
        # 随机大小 (半径)
        radius_x = np.random.randint(size * 0.05, size * 0.25)
        radius_y = np.random.randint(size * 0.05, size * 0.25)
        # 随机旋转
        orientation = np.random.rand() * np.pi
        # 随机灰度值
        intensity = np.random.rand() * 0.5 + 0.5

        rr, cc = ellipse(center_y, center_x, radius_y, radius_x, rotation=orientation, shape=image.shape)
        image[rr, cc] = intensity

    return image

# -----------------------------------------------------------------------------
# 核心函数 2: 生成并保存一个数据对
# -----------------------------------------------------------------------------
# [UPDATED] - 核心函数 2: 生成并保存一个数据对（增加保存sinogram）
def generate_and_save_sample(sample_idx, output_folder, proj_geom, vol_geom):
    """
    生成单个样本（ground_truth, fbp_recon, 和 sinogram）并保存到指定文件夹
    """
    # 1. 生成随机体模作为金标准
    ground_truth = generate_random_phantom(IMAGE_SIZE)

    # 2. 模拟CT扫描（前向投影）
    try:
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        sinogram_id, sinogram = astra.create_sino(ground_truth, projector_id)
    except Exception as e:
        print(f"Error creating CUDA projector, falling back to CPU. Error: {e}")
        projector_id = astra.create_projector('line', proj_geom, vol_geom)
        sinogram_id, sinogram = astra.create_sino(ground_truth, projector_id)

    # 3. 添加少量噪声，让数据更真实
    noise_level = 0.01 * np.max(sinogram)
    noisy_sinogram = sinogram + np.random.normal(0, noise_level, sinogram.shape)
    astra.data2d.delete(sinogram_id)
    sinogram_id = astra.data2d.create('-sino', proj_geom, noisy_sinogram)

    # 4. 使用FBP重建低质量图像
    recon_id = astra.data2d.create('-vol', vol_geom)
    cfg_fbp = astra.astra_dict('FBP_CUDA')
    cfg_fbp['ReconstructionDataId'] = recon_id
    cfg_fbp['ProjectionDataId'] = sinogram_id
    alg_fbp_id = astra.algorithm.create(cfg_fbp)
    astra.algorithm.run(alg_fbp_id)
    fbp_recon = astra.data2d.get(recon_id)

    # 5. 保存成对的 .npy 文件
    gt_path = os.path.join(output_folder, f'sample_{sample_idx:04d}_gt.npy')
    fbp_path = os.path.join(output_folder, f'sample_{sample_idx:04d}_fbp.npy')
    sino_path = os.path.join(output_folder, f'sample_{sample_idx:04d}_sino.npy') # 新增
    np.save(gt_path, ground_truth)
    np.save(fbp_path, fbp_recon)
    np.save(sino_path, noisy_sinogram) # 新增

    # 6. 清理ASTRA内存
    astra.algorithm.delete(alg_fbp_id)
    astra.projector.delete(projector_id)
    astra.data2d.delete([sinogram_id, recon_id])


# -----------------------------------------------------------------------------
# 主执行流程
# -----------------------------------------------------------------------------
def main():
    print(">> Starting data generation...")

    # 创建输出目录
    train_dir = os.path.join(OUTPUT_DIR, 'train')
    test_dir = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 配置ASTRA几何（只需配置一次）
    vol_geom = astra.create_vol_geom(IMAGE_SIZE, IMAGE_SIZE)
    proj_geom = astra.create_proj_geom('parallel', DETECTOR_SPACING, DETECTOR_COUNT, PROJECTION_ANGLES)

    # 生成训练数据
    print(f">> Generating {NUM_TRAIN_SAMPLES} training samples...")
    for i in tqdm(range(NUM_TRAIN_SAMPLES)):
        generate_and_save_sample(i, train_dir, proj_geom, vol_geom)

    # 生成测试数据
    print(f">> Generating {NUM_TEST_SAMPLES} testing samples...")
    for i in tqdm(range(NUM_TEST_SAMPLES)):
        generate_and_save_sample(i, test_dir, proj_geom, vol_geom)

    print("\nData generation complete!")
    print(f"Dataset saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()