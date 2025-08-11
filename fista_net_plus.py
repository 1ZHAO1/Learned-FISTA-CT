import torch
import torch.nn as nn
import astra
import numpy as np

class FpOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_batch, projector_id):
        sino_list = []
        batch_size = image_batch.shape[0]
        for i in range(batch_size):
            single_image = image_batch[i, 0, :, :]
            image_np = single_image.detach().cpu().numpy()
            sino_id, sinogram = astra.create_sino(image_np, projector_id)
            astra.data2d.delete(sino_id)
            sino_list.append(sinogram)
        output_sinograms = np.stack(sino_list, axis=0)
        output_tensor = torch.from_numpy(output_sinograms).to(image_batch.device)
        ctx.projector_id = projector_id
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output_batch):
        grad_list = []
        batch_size = grad_output_batch.shape[0]
        for i in range(batch_size):
            single_grad_sino = grad_output_batch[i, :, :]
            grad_output_np = single_grad_sino.detach().cpu().numpy()
            recon_id, gradient_image = astra.create_backprojection(grad_output_np, ctx.projector_id)
            astra.data2d.delete(recon_id)
            grad_list.append(gradient_image)
        output_gradients = np.stack(grad_list, axis=0)
        output_tensor = torch.from_numpy(output_gradients).unsqueeze(1).to(grad_output_batch.device)
        return output_tensor, None

class BpOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sino_batch, projector_id):
        recon_list = []
        batch_size = sino_batch.shape[0]
        for i in range(batch_size):
            single_sino = sino_batch[i, :, :]
            sinogram_np = single_sino.detach().cpu().numpy()
            recon_id, image = astra.create_backprojection(sinogram_np, projector_id)
            astra.data2d.delete(recon_id)
            recon_list.append(image)
        output_images = np.stack(recon_list, axis=0)
        output_tensor = torch.from_numpy(output_images).unsqueeze(1).to(sino_batch.device)
        ctx.projector_id = projector_id
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output_batch):
        grad_list = []
        batch_size = grad_output_batch.shape[0]
        for i in range(batch_size):
            single_grad_img = grad_output_batch[i, 0, :, :]
            grad_output_np = single_grad_img.detach().cpu().numpy()
            sino_id, gradient_sino = astra.create_sino(grad_output_np, ctx.projector_id)
            astra.data2d.delete(sino_id)
            grad_list.append(gradient_sino)
        output_gradients = np.stack(grad_list, axis=0)
        output_tensor = torch.from_numpy(output_gradients).to(grad_output_batch.device)
        return output_tensor, None

class FISTANetPlus(nn.Module):
    def __init__(self, LayerNo, proj_geom, vol_geom):
        super(FISTANetPlus, self).__init__()
        self.LayerNo = LayerNo
        try:
            self.projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        except Exception:
            self.projector_id = astra.create_projector('line', proj_geom, vol_geom)
            print("!! M5FISTANetPlus: CUDA GPU not detected. Falling back to CPU.")
        self.fp = FpOp.apply
        self.bp = BpOp.apply
        onelayer = []
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            res_block = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1)
            )
            self.fcs.append(res_block)
        self.w_theta = nn.Parameter(torch.ones(LayerNo, 1, 1, 1))
        self.b_theta = nn.Parameter(torch.zeros(LayerNo, 1, 1, 1))
        self.w_mu = nn.Parameter(torch.ones(LayerNo, 1, 1, 1))
        self.b_mu = nn.Parameter(torch.zeros(LayerNo, 1, 1, 1))
        self.w_rho = nn.Parameter(torch.ones(LayerNo, 1, 1, 1))
        self.b_rho = nn.Parameter(torch.zeros(LayerNo, 1, 1, 1))

    def forward(self, x_fbp, y_sino_batch):
        x = x_fbp
        z = x
        loss_layers_sym = []
        loss_st = []
        for i in range(self.LayerNo):
            Ax = self.fp(x, self.projector_id)
            x_grad = self.bp(Ax - y_sino_batch, self.projector_id)
            # 使用梯度裁剪，将梯度的每个元素值限制在 [-1, 1] 区间内
            x_grad = torch.clamp(x_grad, -1.0, 1.0)
            x = x - self.w_mu[i] * x_grad + self.b_mu[i]
            z_tilde = x + self.w_rho[i] * (x - z) + self.b_rho[i]
            residual = self.fcs[i](z_tilde)
            z_tilde_Fe = z_tilde + residual
            z = torch.mul(torch.sign(z_tilde), torch.relu(torch.abs(z_tilde) - self.w_theta[i])) + self.b_theta[i]
            loss_layers_sym.append(z_tilde_Fe - z)
            loss_st.append(z)
        return z, loss_layers_sym, loss_st

    def __del__(self):
        if hasattr(self, 'projector_id'):
            astra.projector.delete(self.projector_id)