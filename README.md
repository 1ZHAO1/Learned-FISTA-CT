# Physics-Informed Deep Unrolling for Limited-Angle CT Reconstruction

This project provides a complete framework for training and evaluating a physics-informed deep learning model for limited-angle Computerized Tomography (CT) reconstruction. The core of the project is `FISTANetPlus`, a model that unrolls the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) into a neural network, embedding the physical forward and back-projection operations directly within its layers.

The primary goal is to overcome the severe artifacts produced by conventional reconstruction methods like Filtered Back-Projection (FBP) when projection data is only available from a limited angular range.

## 核心特性 (Core Features)

* **物理驱动的深度学习 (Physics-Informed Deep Learning)**: 模型的核心不是一个通用的黑盒CNN，而是将CT成像的物理过程（通过ASTRA Toolbox实现的前向和反向投影）作为网络的可微操作层。
* **端到端的完整流程 (End-to-End Workflow)**: 提供从合成数据生成、基准模型测试、数据加载、模型训练到结果评估的全套脚本。
* **可复现的数据生成 (Reproducible Data Generation)**: 内置脚本可生成大量的随机椭圆体模（phantom）用于训练，确保模型学习到的是通用的伪影去除能力，而非针对特定结构。
* **鲁棒的训练策略 (Robust Training Strategy)**: 训练器（Solver）中包含了梯度裁剪、学习率调度以及针对`inf`/`nan`损失的“保险丝”逻辑，确保复杂模型训练过程的稳定性。
* **模块化代码结构 (Modular Code Structure)**: 项目被清晰地划分为数据、模型、训练器和主函数等模块，具有高可读性和可扩展性。
* **强大的可配置性 (Highly Configurable)**: 通过主函数的命令行参数，可以轻松调整学习率、批大小、训练周期、模型名称等关键超参数，方便进行科学实验。

## 文件结构 (File Structure)

```
ct-reconstruction-project/
|
|-- dataset/              # 存放由generate_data.py生成的数据集
|   |-- train/
|   |-- test/
|
|-- results/              # 存放训练好的模型和日志
|   |-- M5FISTANetPlus/
|
|-- baseline_recon.py     # 脚本：运行传统FBP和SART-TV算法作为基准
|-- generate_data.py      # 脚本：为模型生成训练和测试数据集
|-- loader.py             # 模块：定义PyTorch的CTDataset数据加载器
|-- fista_net_plus.py     # 模块：定义FISTANetPlus核心模型架构
|-- solver.py             # 模块：定义包含训练和测试逻辑的Solver类
|-- main.py               # 主入口：组装所有模块并启动程序
|-- README.md             # 本文件
```

## 方法论 (Methodology)

### 1. 问题的数学描述

CT重建本质上是一个求解线性逆问题的过程。其离散化的数学模型可以表示为：

$$
y = Ax + \epsilon
$$

* $x \in \mathbb{R}^{N \times 1}$: 这是我们想要重建的**真实图像**，被向量化为一个列向量。$N$ 是图像的总像素数（例如，对于128x128的图像，$N = 128 \times 128 = 16384$）。
* $y \in \mathbb{R}^{M \times 1}$: 这是CT扫描仪实际测量到的**投影数据（Sinogram）**，同样被向量化。$M$ 是探测器数量与投影角度数的乘积。
* $A \in \mathbb{R}^{M \times N}$: 这是**系统矩阵（System Matrix）**，也称为**前向投影算子**。它是一个巨大的稀疏矩阵，描述了X光如何穿过物体$x$并最终形成投影$y$的物理过程。矩阵的每一行对应一次X射线路径积分。
* $\epsilon$: 代表测量过程中不可避免的**噪声**。

我们的目标是已知 $y$ 和 $A$ 的情况下，求解出最接近真实的 $x$。

### 2. 基于优化的重建方法

直接对 $y = Ax$ 求逆（即 $x = A^{-1}y$）是不可行的，因为在有限角情况下，矩阵 $A$ 是病态的（ill-conditioned），微小的噪声 $\epsilon$ 都会导致解的巨大偏差。因此，我们通常通过求解一个优化问题来找到一个稳定的解。这个优化问题通常包含两个部分：

$$
\hat{x} = \arg\min_{x} \left( \frac{1}{2} \|Ax - y\|_2^2 + \lambda \mathcal{R}(x) \right)
$$

* **数据保真项 (Data Fidelity Term)**: $\frac{1}{2} \|Ax - y\|_2^2$
    * 这个部分确保我们重建的图像 $\hat{x}$ 在经过模拟的CT扫描后（即 $A\hat{x}$），应该与我们实际测量到的数据 $y$ 尽可能一致。
    * $\| \cdot \|_2^2$ 表示欧几里得L2范数的平方，计算的是两者差异的平方和，这在物理上对应于假设噪声是高斯分布的。我们称这一项为 $f(x)$。
* **正则项 (Regularization Term)**: $\mathcal{R}(x)$
    * 这一项加入了我们对真实图像 $x$ 的**先验知识**。例如，我们可能认为医学图像在大部分区域是平滑的，或者在某些变换域（如小波域）是稀疏的。
    * 常见的正则项有总变分（Total Variation, TV）或L1范数（$\|x\|_1$），它们都能促进图像的稀疏性或边缘保持。我们称这一项为 $g(x)$。
* **正则化参数 (Regularization Parameter)**: $\lambda$
    * 这是一个标量超参数，用于平衡**数据保真度**和**正则化约束**之间的重要性。$\lambda$ 越大，解就越符合先验知识（例如更平滑或更稀疏），但可能会偏离真实测量数据。

### 3. FISTA 算法原理

FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) 是一种高效求解上述优化问题的迭代算法。它的核心思想是将问题分解为两个可以交替执行的简单步骤：

**第 $k$ 次迭代:**

1.  **梯度下降步 (Gradient Descent Step)**: 首先，对光滑的数据保真项 $f(x)$ 进行一步梯度下降。
    $$
    z_k = x_k - \mu \nabla f(x_k)
    $$
    * $x_k$: 第 $k$ 次迭代的图像解。
    * $\mu$: 梯度下降的**步长**（learning rate）。
    * $\nabla f(x_k)$: $f(x)$ 在 $x_k$ 处的梯度。根据 $f(x)=\frac{1}{2} \|Ax - y\|_2^2$，其梯度为：
        $$
        \nabla f(x_k) = A^T(Ax_k - y)
        $$
        这里的 $A^T$ 是系统矩阵 $A$ 的转置，在物理上它代表了**反投影（Back-Projection）**操作。

2.  **近端映射步 (Proximal Mapping Step)**: 然后，对上一步的结果 $z_k$ 应用一个与正则项 $g(x)$ 相关的“修正”操作，称为近端映射。
    $$
    x_{k+1} = \text{prox}_{\mu\lambda g}(z_k)
    $$
    * $\text{prox}_{\alpha g}(z) = \arg\min_{u} (\frac{1}{2}\|u-z\|_2^2 + \alpha g(u))$: 近端算子的定义。它寻找一个点 $u$，既要离 $z_k$ 近，又要使正则项 $g(u)$ 的值小。
    * 对于L1正则项 $g(x)=\|x\|_1$，其近端算子是一个非常简单的**软阈值函数 (Soft-Thresholding)**：
        $$
        \text{prox}_{\alpha L_1}(z)_i = \text{sign}(z_i) \cdot \max(|z_i| - \alpha, 0)
        $$
        这个公式意味着将 $z_k$ 的每个元素 $z_i$ 向零收缩一个阈值 $\alpha$，小于该阈值的直接置为零。

### 4. FISTANetPlus: 算法展开

`FISTANetPlus` 的核心思想是将FISTA的迭代过程“展开”成一个深度神经网络。FISTA的每一次迭代，都变成网络的一“层”。

* **梯度下降步的展开**: FISTA中的计算 $A^T(Ax_k - y)$ 被代码 `self.bp(self.fp(x, ...) - y_sino_batch, projector_id)` 精确地实现。而固定的步长 $\mu$ 被一个**可学习的参数** `self.w_mu[k]` 替代。
* **近端映射步的展开**: 传统的近端映射步被一个更强大、更灵活的**神经网络模块**所替代。这是模型最核心的创新。

#### **核心创新：作为可学习正则项的CNN模块**

在`FISTANetPlus`中，传统FISTA算法中基于固定先验（如L1范数）的近端映射（Proximal Mapping）步骤，被一个**可学习的卷积神经网络（CNN）模块**所取代。这个模块在代码中由`self.fcs`定义。这一设计的核心价值在于：

1.  **从固定先验到数据驱动先验**: 传统的L1正则项假设图像在某个域是稀疏的，这是一个固定的、手工设计的先验知识。而CNN模块则可以**从大规模的训练数据中自动学习**一个最优的、针对CT图像特性的先验模型。它学习的不是一个简单的稀疏假设，而是伪影、噪声与真实图像结构之间的复杂非线性关系。

2.  **从简单去噪到智能修复**: 传统的软阈值操作本质上是一个像素级的、简单的“去噪器”。而`self.fcs`模块 是一个更强大的“图像修复器”。凭借其多层卷积结构和感受野，它能够分析每个像素的**局部上下文信息**，从而能够区分伪影、噪声和真实的解剖结构细节，实现更智能、更精细的图像恢复。

3.  **模型的核心“智能”**: 可以说，`FISTANetPlus`的“智能”主要就体现在这个CNN模块中。它将一个纯粹的数学优化步骤，升华为一个具有强大特征提取和非线性映射能力的深度学习模块，是区分“Learned FISTA”与“Classic FISTA”的根本所在。

通过这种方式，`FISTANetPlus`不仅遵循了迭代优化的物理框架，还利用深度学习的强大能力学习到了一个最优的、数据驱动的正则项，从而在有限角CT重建任务上取得了远超传统方法的效果。

## 环境安装 (Setup and Installation)

建议使用Conda管理环境。请确保你有一块支持CUDA的NVIDIA GPU，因为ASTRA Toolbox的CPU模式非常慢。

```bash
# 1. 创建并激活Conda环境
conda create -n ct-recon python=3.8 -y
conda activate ct-recon

# 2. 安装PyTorch (请根据你的CUDA版本去PyTorch官网获取命令)
# 例如 (CUDA 11.8):
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. 安装ASTRA Toolbox
conda install -c astra-toolbox astra-toolbox

# 4. 安装其他依赖
pip install numpy scikit-image matplotlib tqdm
```

## 使用流程 (Step-by-Step Workflow)

#### 第1步: 运行基准测试 (可选，但推荐)

首先运行基准脚本，以直观了解问题的挑战性以及传统方法的性能上限。

```bash
python baseline_recon.py
```
这会显示Shepp-Logan体模、FBP重建和SART-TV重建的对比图。

#### 第2步: 生成数据集

运行数据生成脚本。这将创建用于训练和测试的全部数据，并保存在 `./dataset` 目录下。

```bash
python generate_data.py
```
该过程会生成 `NUM_TRAIN_SAMPLES` (默认1000) 组训练数据和 `NUM_TEST_SAMPLES` (默认200) 组测试数据。

#### 第3步: 训练模型

所有配置都通过 `main.py` 的命令行参数控制。以下是一个开始训练的示例命令：

```bash
python main.py --model_name M5FISTANetPlus --mode train --num_epochs 200 --batch_size 16 --lr 1e-6
```
训练过程中的模型权重（checkpoints）和损失日志将保存在 `./results/M5FISTANetPlus/` 目录下。

#### 第4步: 测试模型

要测试一个已经训练好的模型，可以修改 `--mode` 为 `test`，并使用 `--test_epoch` 指定要加载的模型是在哪个epoch保存的。

```bash
python main.py --model_name M5FISTANetPlus --mode test --test_epoch 200
```
*注意: `solver.py` 中的 `test()` 函数可能需要根据具体评估需求进行适配。*

## 模块功能详解

#### `solver.py`
* **作用**: 封装所有训练和测试的逻辑。
* **功能**:
    * **复合损失函数**: 在训练`FISTANetPlus`时，优化的不仅仅是最终输出，还包括对中间迭代步骤的约束。总损失函数 $L(\Theta)$ 由三部分构成，其中 $\Theta$ 代表网络所有可学习的参数：
        $$
        L(\Theta) = L_{\text{discrepancy}} + \beta \cdot L_{\text{constraint}} + \gamma \cdot L_{\text{sparsity}}
        $$
        1.  **$L_{\text{discrepancy}}$ (差异损失)**: 这是主要的监督信号，用于衡量最终重建结果 $\hat{x}$ 与金标准图像 $x_{\text{gt}}$ 之间的差距。它由MSE和L1损失两部分组成。
            $$
            L_{\text{discrepancy}} = \| \hat{x} - x_{\text{gt}} \|_2^2 + \eta \cdot \| \hat{x} - x_{\text{gt}} \|_1
            $$
        2.  **$L_{\text{constraint}}$ (约束损失)**: 这是代码中的`loss_layers_sym`。它对网络中间层的行为施加约束，鼓励由CNN提取的特征与软阈值操作的结果之间存在某种对称关系。
        3.  **$L_{\text{sparsity}}$ (稀疏损失)**: 这是代码中的`loss_st`。它直接对每次迭代后的图像解 $z^{(k)}$ 的L1范数进行惩罚，鼓励中间解是稀疏的。
    * **鲁棒性设计**: 实现“保险丝”逻辑，在反向传播前检查损失是否为 `inf` 或 `nan`。同时使用梯度裁剪 `clip_grad_norm_` 来防止梯度爆炸。

#### `fista_net_plus.py`
* **作用**: 定义项目的核心——`FISTANetPlus`模型。
* **功能**:
    * 通过继承 `torch.autograd.Function`，实现了 `FpOp` (前向投影, $A$) 和 `BpOp` (反向投影, $A^T$) 两个自定义可微算子。
    * **`FpOp.forward`** 执行 `astra.create_sino` 运算。
    * **`FpOp.backward`** 执行 `astra.create_backprojection` 运算。
    * **`BpOp.forward`** 执行 `astra.create_backprojection` 运算。
    * **`BpOp.backward`** 执行 `astra.create_sino` 运算。
    * 在 `forward` 方法中，实现了对反投影梯度 $A^T(Ax-y)$ 的强制归一化，这是解决数值不稳定问题的关键修正。

（其他文件的功能描述与前一版相同，此处省略以保持简洁）

## 未来可改进方向

* **引入验证集**: 在训练过程中引入验证集，用于监控模型性能、防止过拟合，并实现早期停止（Early Stopping）。
* **配置文件**: 将CT几何参数等硬编码在多个文件中的设置提取到一个独立的配置文件中（如 `config.yaml`），实现统一管理。
* **性能优化**: 研究ASTRA算子的批处理方法，或寻找其他支持批处理的投影库，以减少`FpOp`和`BpOp`中Python循环带来的开销。
* **更完善的测试逻辑**: 扩展`solver.py`中的`test()`方法，使其能够遍历整个测试集，计算平均指标，并保存可视化结果。

## 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源。