# -*- coding: utf-8 -*-
# [FINAL VERSION 3] - This version corrects the ASTRA API calls.

import torch
import torch.nn as nn
import astra
import numpy as np

# =============================================================================
# [FINAL] Part 1: The ASTRA-PyTorch Bridge (API Corrected)
# 作用:
#   定义两个自定义的PyTorch自动求导函数 (torch.autograd.Function)。
#   这两个函数作为桥梁，使得我们可以在PyTorch的计算图中调用ASTRA工具箱
#   中的物理运算（前向投影和反投影），并正确地进行梯度反向传播。
# =============================================================================

class FpOp(torch.autograd.Function):
    """
    自定义前向投影算子 (Custom Forward Projection Operator, A)

    这个类将ASTRA的前向投影操作封装成一个PyTorch可以识别的层。
    它负责将输入的图像(x)通过CT物理模型，计算出对应的投影数据(sinogram, y)。
    即，它实现了 y = Ax 的运算。
    """
    @staticmethod
    def forward(ctx, image_batch, projector_id):
        """
        定义算子的前向传播行为。
        
        Args:
            ctx: 一个上下文对象，用于存储信息以便在反向传播时使用。
            image_batch: 输入的图像批次，形状为 (batch_size, 1, height, width)。
            projector_id: ASTRA中已经配置好的投影仪ID，包含了所有的几何信息。
        
        Returns:
            一个包含投影数据的PyTorch张量。
        """
        sino_list = []
        batch_size = image_batch.shape[0]

        # 由于ASTRA本身不直接支持批处理，我们需要遍历batch中的每一张图像
        for i in range(batch_size):
            # 从PyTorch张量中取出单张图像，并转换为ASTRA需要的NumPy数组格式
            single_image = image_batch[i, 0, :, :]
            image_np = single_image.detach().cpu().numpy()
            
            # 调用ASTRA API执行前向投影
            sino_id, sinogram = astra.create_sino(image_np, projector_id)
            # 立即从ASTRA的内存中删除创建的数据对象，防止内存泄漏
            astra.data2d.delete(sino_id)
            
            sino_list.append(sinogram)
            
        # 将所有样本的结果堆叠成一个新的批次，并转换回PyTorch张量
        output_sinograms = np.stack(sino_list, axis=0)
        output_tensor = torch.from_numpy(output_sinograms).to(image_batch.device)
        
        # 将投影仪ID保存在上下文中，反向传播时需要用它来执行反投影
        ctx.projector_id = projector_id
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output_batch):
        """
        定义算子的反向传播行为（即梯度计算）。
        对于前向投影算子 A，其梯度（伴随算子）是反投影算子 A^T。
        
        Args:
            ctx: 前向传播时保存的上下文对象。
            grad_output_batch: 来自上一层的、关于本层输出的梯度。
        
        Returns:
            一个包含图像梯度的PyTorch张量。
        """
        grad_list = []
        batch_size = grad_output_batch.shape[0]

        # 同样需要遍历批次
        for i in range(batch_size):
            single_grad_sino = grad_output_batch[i, :, :]
            grad_output_np = single_grad_sino.detach().cpu().numpy()

            # 调用ASTRA API执行反投影来计算梯度
            recon_id, gradient_image = astra.create_backprojection(grad_output_np, ctx.projector_id)
            # 及时清理内存
            astra.data2d.delete(recon_id)
            
            grad_list.append(gradient_image)
            
        # 将所有梯度图像堆叠并转回PyTorch张量
        output_gradients = np.stack(grad_list, axis=0)
        # 使用unsqueeze(1)来增加通道维度，以匹配输入的格式 (batch_size, 1, height, width)
        output_tensor = torch.from_numpy(output_gradients).unsqueeze(1).to(grad_output_batch.device)
        
        # 返回的梯度必须与forward的输入参数一一对应。
        # projector_id不是一个需要梯度的张量，所以它的梯度是None。
        return output_tensor, None


class BpOp(torch.autograd.Function):
    """
    自定义反投影算子 (Custom Backward Projection Operator, A^T)

    这个类将ASTRA的反投影操作封装成一个PyTorch可以识别的层。
    它负责将输入的投影数据(sinogram, y)通过反投影，计算出对应的图像(x)。
    即，它实现了 x = A^T*y 的运算。
    """
    @staticmethod
    def forward(ctx, sino_batch, projector_id):
        """
        定义算子的前向传播行为。
        """
        recon_list = []
        batch_size = sino_batch.shape[0]

        for i in range(batch_size):
            single_sino = sino_batch[i, :, :]
            sinogram_np = single_sino.detach().cpu().numpy()
            
            # 调用ASTRA API执行反投影
            recon_id, image = astra.create_backprojection(sinogram_np, projector_id)
            astra.data2d.delete(recon_id)

            recon_list.append(image)

        output_images = np.stack(recon_list, axis=0)
        # 增加通道维度
        output_tensor = torch.from_numpy(output_images).unsqueeze(1).to(sino_batch.device)
        
        ctx.projector_id = projector_id
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output_batch):
        """
        定义算子的反向传播行为。
        对于反投影算子 A^T，其梯度（伴随算子）是前向投影算子 A。
        """
        grad_list = []
        batch_size = grad_output_batch.shape[0]

        for i in range(batch_size):
            single_grad_img = grad_output_batch[i, 0, :, :]
            grad_output_np = single_grad_img.detach().cpu().numpy()

            # 调用ASTRA API执行前向投影来计算梯度
            sino_id, gradient_sino = astra.create_sino(grad_output_np, ctx.projector_id)
            astra.data2d.delete(sino_id)
            
            grad_list.append(gradient_sino)
        
        output_gradients = np.stack(grad_list, axis=0)
        output_tensor = torch.from_numpy(output_gradients).to(grad_output_batch.device)
        
        return output_tensor, None
    
# =============================================================================
# The FISTANetPlus Model Itself
# =============================================================================

class FISTANetPlus(nn.Module):
    """
    FISTANetPlus模型的主体。
    它是一个“算法展开”(Algorithm Unrolling)网络，将FISTA优化算法的
    迭代步骤映射为神经网络的层。
    """
    def __init__(self, LayerNo, proj_geom, vol_geom):
        """
        初始化模型。
        
        Args:
            LayerNo (int): 网络的层数，对应FISTA算法的迭代次数。
            proj_geom: ASTRA的投影几何对象。
            vol_geom: ASTRA的体模几何对象。
        """
        super(FISTANetPlus, self).__init__()
        
        self.LayerNo = LayerNo
        
        # 创建一个会被所有层共享的ASTRA投影仪，避免重复创建开销。
        try:
            self.projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        except Exception:
            # 如果没有CUDA GPU，则回退到CPU版本（会非常慢）。
            self.projector_id = astra.create_projector('line', proj_geom, vol_geom)
            print("!! M5FISTANetPlus: CUDA GPU not detected. Falling back to CPU.")

        # 将我们自定义的算子函数赋值给类成员，方便在forward方法中调用。
        self.fp = FpOp.apply
        self.bp = BpOp.apply

        # 初始化每一层（即每一次迭代）所需的神经网络模块。
        onelayer = []
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            # 这里的CNN模块用于替代FISTA中的近端映射步骤，作为一个可学习的正则项/去噪器。
            self.fcs.append(nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, 1, 1)
            ))
            
        # 将FISTA算法中的固定超参数定义为可学习的 nn.Parameter。
        # 这样网络就可以在训练中自动学习到最优的参数。
        
        # w_theta, b_theta: 用于可学习的软阈值操作 (对应近端映射的阈值 lambda)
        self.w_theta = nn.Parameter(torch.ones(LayerNo, 1, 1, 1))
        self.b_theta = nn.Parameter(torch.zeros(LayerNo, 1, 1, 1))
        # w_mu, b_mu: 用于可学习的梯度下降步长 (对应 mu)
        self.w_mu = nn.Parameter(torch.ones(LayerNo, 1, 1, 1))
        self.b_mu = nn.Parameter(torch.zeros(LayerNo, 1, 1, 1))
        # w_rho, b_rho: 用于可学习的动量项 (对应FISTA中的加速步骤)
        self.w_rho = nn.Parameter(torch.ones(LayerNo, 1, 1, 1))
        self.b_rho = nn.Parameter(torch.zeros(LayerNo, 1, 1, 1))

    def forward(self, x_fbp, y_sino_batch):
        """
        定义模型的前向传播逻辑，完整模拟了FISTA的迭代过程。
        
        Args:
            x_fbp: FBP重建的低质量图像，作为迭代的初始点 x_0。
            y_sino_batch: 真实的投影数据 y，用于计算数据保真项。
        
        Returns:
            - 最终重建的高质量图像。
            - loss_layers_sym: 用于计算对称损失的中间结果。
            - loss_st: 用于计算稀疏损失的中间结果。
        """
        # 初始化迭代变量
        x = x_fbp  # 当前迭代的解
        z = x      # 加速步骤的辅助变量
        loss_layers_sym = [] # 存储用于计算约束损失的中间变量
        loss_st = []         # 存储用于计算稀疏损失的中间变量
        
        # 循环展开FISTA算法
        for i in range(self.LayerNo):
            # --- 1. 更新x (等同于FISTA的梯度下降步) ---
            
            # 计算 A*x_k
            Ax = self.fp(x, self.projector_id)
            # 计算 A^T(A*x_k - y)，即数据保真项的梯度
            x_grad = self.bp(Ax - y_sino_batch, self.projector_id)

            # [--- 最终稳定性修正 ---]
            # 这是解决顽固性nan/inf问题的关键一步。
            # 由于反投影算子可能会产生数值范围非常大的结果，直接使用会导致数值爆炸。
            # 因此，在这里对梯度进行强制归一化，使其尺度稳定。
            # 我们计算每个样本在batch中的最大绝对值，并用它来归一化。
            # keepdim=True 保持维度以便进行广播除法 (例如, [B,1,1,1])。
            abs_max = torch.amax(torch.abs(x_grad), dim=(-3, -2, -1), keepdim=True)
            x_grad = x_grad / (abs_max + 1e-8) # 使用1e-8防止除以0
            # [--- 修正结束 ---]

            # 执行梯度下降更新: x_k' = x_k - mu * grad
            # 这里的步长 mu 是可学习的参数 w_mu[i]
            x = x - self.w_mu[i] * x_grad + self.b_mu[i]
            
            # --- 2. 更新z (等同于FISTA的近端映射和加速步) ---
            
            # FISTA的加速步: z_tilde 类似于一个带有动量的更新
            z_tilde = x + self.w_rho[i] * (x - z) + self.b_rho[i]
            
            # 将z_tilde输入到该层的CNN模块中，进行特征提取/去噪
            z_tilde_Fe = self.fcs[i](z_tilde)
            
            # 执行可学习的软阈值操作，这是近端映射的核心
            # z = sign(z_tilde) * ReLU(|z_tilde| - theta)
            z = torch.mul(torch.sign(z_tilde), torch.relu(torch.abs(z_tilde) - self.w_theta[i])) + self.b_theta[i]
            
            # 保存中间结果，用于在solver中计算额外的约束损失
            loss_layers_sym.append(z_tilde_Fe - z)
            loss_st.append(z)

        return z, loss_layers_sym, loss_st

    def __del__(self):
        """
        定义对象的析构函数。
        当模型的实例被销毁时，这个函数会被调用，以确保ASTRA的内存被正确释放。
        """
        if hasattr(self, 'projector_id'):
            astra.projector.delete(self.projector_id)