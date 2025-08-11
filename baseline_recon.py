# -*- coding: utf-8 -*-

# =============================================================================
#
# 文件名: baseline_recon.py
#
# 目标:
#   1. 创建一个标准的Shepp-Logan数字体模作为“真实”图像 (Ground Truth)。
#   2. 模拟有限角度下的CT扫描过程（前向投影），生成投影数据（Sinogram）。
#   3. 使用两种经典的算法进行图像重建：
#      - 滤波反投影 (FBP): 速度快，但对有限角问题效果差。
#      - 带总变分正则化的同步代数重建技术 (SART-TV): 迭代算法，效果好，但速度慢。
#   4. 计算重建图像的质量（PSNR和SSIM），并进行可视化对比。
#
# =============================================================================

import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------------------------
# 步骤 0: 配置实验参数
# 在这里我们可以方便地修改所有实验设置，而无需改动核心代码。
# -----------------------------------------------------------------------------
print(">> Step 0: Configuring experiment parameters...")

# 图像和几何设置
PHANTOM_SIZE = 256  # 图像的像素尺寸 (N x N)
DETECTOR_COUNT = 384  # 探测器的数量
DETECTOR_SPACING = 1.0  # 探测器单元的物理尺寸

# 角度设置
# 我们将模拟一个120度的有限角扫描
PROJECTION_ANGLES_LIMITED = np.linspace(0, np.pi * 2/3, 180, False) # 从0到120度，采集180个角度

# 迭代算法设置
SART_ITERATIONS = 3000  # SART算法的迭代次数
TV_REGULARIZATION_PARAM = 0.03 # TV正则化项的权重 (可以调整以观察效果)

# -----------------------------------------------------------------------------
# 步骤 1: 创建 Ground Truth (真实图像)
# 我们使用scikit-image库中内置的Shepp-Logan体模。
# 这是一个在医学成像领域广泛用于测试重建算法的标准模型。
# -----------------------------------------------------------------------------
print(">> Step 1: Creating the Ground Truth phantom...")

# 创建原始体模并将其尺寸调整为我们设定的PHANTOM_SIZE
ground_truth = resize(shepp_logan_phantom(), (PHANTOM_SIZE, PHANTOM_SIZE), anti_aliasing=True)

# -----------------------------------------------------------------------------
# 步骤 2: 配置 ASTRA 工具箱的几何参数
# 这是告诉ASTRA我们的CT扫描仪长什么样，以及被扫描的物体有多大。
# -----------------------------------------------------------------------------
print(">> Step 2: Configuring ASTRA geometry...")

# 定义被扫描物体（体模）的几何信息
# 这是一个2D的方形区域
vol_geom = astra.create_vol_geom(PHANTOM_SIZE, PHANTOM_SIZE)

# 定义投影（扫描）的几何信息
# 我们使用平行光束（'parallel'），这对应于最简单的CT扫描模式
proj_geom_limited = astra.create_proj_geom('parallel', DETECTOR_SPACING, DETECTOR_COUNT, PROJECTION_ANGLES_LIMITED)

# -----------------------------------------------------------------------------
# 步骤 3: 模拟扫描过程 (前向投影)
# 使用我们定义的几何参数和体模，计算出在有限角度下应该采集到的投影数据。
# 这个过程在物理上是X光穿过物体被探测器接收。在数学上，是执行雷登变换。
# -----------------------------------------------------------------------------
print(">> Step 3: Simulating the limited-angle scanning process...")

# 创建一个前向投影算子(A)。我们使用CUDA（如果可用）来加速计算。
# ASTRA会自动检测GPU，如果没有，则会报错。请确保安装了正确的CUDA版本。
try:
    projector_id = astra.create_projector('cuda', proj_geom_limited, vol_geom)
except Exception:
    # 如果没有CUDA GPU，则使用CPU版本（非常慢）
    projector_id = astra.create_projector('line', proj_geom_limited, vol_geom)
    print("!! CUDA GPU not detected. Falling back to CPU, which will be very slow.")


# 执行前向投影，得到投影数据(sinogram) y = Ax
# sinogram_id是ASTRA内存中数据的ID，sinogram是其numpy数组表示
sinogram_id, sinogram = astra.create_sino(ground_truth, projector_id)

# -----------------------------------------------------------------------------
# 步骤 4: 使用 FBP 算法进行重建
# FBP是解析重建算法，速度快，但对数据噪声和不完整性（如有限角）非常敏感。
# 我们用它来展示一个“糟糕”的基线结果。
# -----------------------------------------------------------------------------
print(">> Step 4: Reconstructing with the FBP algorithm...")

# 为重建结果创建一个数据对象
recon_fbp_id = astra.data2d.create('-vol', vol_geom)

# 创建FBP重建算法的配置
# 我们使用'FBP_CUDA'表示使用GPU加速的FBP算法
cfg_fbp = astra.astra_dict('FBP_CUDA')
cfg_fbp['ReconstructionDataId'] = recon_fbp_id
cfg_fbp['ProjectionDataId'] = sinogram_id
cfg_fbp['ProjectorId'] = projector_id

# 创建并运行算法
alg_fbp_id = astra.algorithm.create(cfg_fbp)
astra.algorithm.run(alg_fbp_id)

# 从ASTRA内存中获取重建结果
recon_fbp = astra.data2d.get(recon_fbp_id)

# -----------------------------------------------------------------------------
# 步骤 5: 使用 SART-TV 算法进行重建
# SART是迭代算法，它会一点点地修正图像，使其投影结果逼近真实的投影数据。
# TV正则化则会惩罚图像中过多的噪声和细节，使结果更平滑，有助于抑制伪影。
# -----------------------------------------------------------------------------
print(">> Step 5: Reconstructing with the SART-TV algorithm...")

# 为重建结果创建数据对象
recon_sart_tv_id = astra.data2d.create('-vol', vol_geom)

# 创建SART重建算法的配置
cfg_sart = astra.astra_dict('SART_CUDA')
cfg_sart['ReconstructionDataId'] = recon_sart_tv_id
cfg_sart['ProjectionDataId'] = sinogram_id
cfg_sart['ProjectorId'] = projector_id

# 创建SART算法对象
alg_sart_id = astra.algorithm.create(cfg_sart)

# --- TV 正则化部分 ---
# 加载TV正则化插件
# ASTRA通过插件机制来为迭代算法增加额外的功能
try:
    # 查找并创建TV正则化插件
    tv_plugin_id = astra.plugin.get('Regularizer_TV_CUDA')
    # 将插件与SART算法关联起来
    astra.plugin.set_params(tv_plugin_id, {'nonneg': True, 'lambda': TV_REGULARIZATION_PARAM})
    astra.algorithm.set_plugin(alg_sart_id, tv_plugin_id)
    print(f"   TV regularizer plugin loaded with lambda = {TV_REGULARIZATION_PARAM}")
except Exception:
    print("!! TV plugin failed to load or is not supported in this ASTRA version. Running SART only.")


# 运行SART迭代。与FBP不同，这里需要指定迭代次数。
astra.algorithm.run(alg_sart_id, SART_ITERATIONS)

# 获取重建结果
recon_sart_tv = astra.data2d.get(recon_sart_tv_id)


# -----------------------------------------------------------------------------
# 步骤 6: 评估与可视化
# 我们计算PSNR(峰值信噪比)和SSIM(结构相似性)来定量评估重建质量。
# PSNR越高越好，SSIM越接近1越好。
# -----------------------------------------------------------------------------
print(">> Step 6: Evaluating and visualizing the results...")

# 计算指标
psnr_fbp = psnr(ground_truth, recon_fbp, data_range=ground_truth.max() - ground_truth.min())
ssim_fbp = ssim(ground_truth, recon_fbp, data_range=ground_truth.max() - ground_truth.min())

psnr_sart_tv = psnr(ground_truth, recon_sart_tv, data_range=ground_truth.max() - ground_truth.min())
ssim_sart_tv = ssim(ground_truth, recon_sart_tv, data_range=ground_truth.max() - ground_truth.min())

print(f"\n--- Evaluation Results ---")
print(f"FBP      : PSNR = {psnr_fbp:.2f} dB, SSIM = {ssim_fbp:.4f}")
print(f"SART-TV  : PSNR = {psnr_sart_tv:.2f} dB, SSIM = {ssim_sart_tv:.4f}")
print("------------------\n")


# 可视化结果
# 创建一个1行3列的图床
plt.figure(figsize=(18, 6))
plt.gray() # 使用灰度色彩图

# 图1: Ground Truth
plt.subplot(1, 3, 1)
plt.imshow(ground_truth)
plt.title('Ground Truth', fontsize=16)
plt.axis('off')

# 图2: FBP 重建
plt.subplot(1, 3, 2)
plt.imshow(recon_fbp)
plt.title(f'FBP Reconstruction ({len(PROJECTION_ANGLES_LIMITED)} Projections)\nPSNR: {psnr_fbp:.2f}, SSIM: {ssim_fbp:.4f}', fontsize=16)
plt.axis('off')

# 图3: SART-TV 重建
plt.subplot(1, 3, 3)
plt.imshow(recon_sart_tv)
plt.title(f'SART-TV Reconstruction ({SART_ITERATIONS} iters)\nPSNR: {psnr_sart_tv:.2f}, SSIM: {ssim_sart_tv:.4f}', fontsize=16)
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 步骤 7: 清理ASTRA内存
# ASTRA在GPU/CPU内存中创建了许多对象，在程序结束时最好手动清理。
# -----------------------------------------------------------------------------
print(">> Step 7: Cleaning up ASTRA memory...")
astra.algorithm.delete([alg_fbp_id, alg_sart_id])
astra.data2d.delete([recon_fbp_id, recon_sart_tv_id, sinogram_id])
astra.projector.delete(projector_id)
if 'tv_plugin_id' in locals():
    astra.plugin.delete(tv_plugin_id)

print("\nStep 1 successfully completed!")