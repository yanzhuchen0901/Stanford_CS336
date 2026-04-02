# Assignment 1 - Transformer fron scratch
## Overview
This project involves building a Transformer-based Language Model from scratch, trained on the TinyStories dataset. The implementation includes a custom BPE (Byte Pair Encoding) tokenizer and a full Transformer architecture. The training was conducted on an AMD-based hardware stack (Ryzen 7 8745H & Radeon RX 7700 XT), achieving a significant loss reduction from 1.8 to 1.5 through three major training iterations.

## Challenges & Troubleshooting
Version Incompatibility: Initial attempts using Python 3.14 led to a total failure in installing core libraries like torch-directml and jaxtyping. The solution was downgrading to the stable Python 3.11.9 environment.

### AMD GPU Integration:
Standard PyTorch defaults to CUDA (NVIDIA). On AMD hardware, this resulted in an AssertionError. I resolved this by utilizing Microsoft DirectML, allowing PyTorch to interface with the Radeon GPU.

### Environment PATH Conflicts:
Residual files from multiple Python versions caused pip to install packages into incorrect directories. I fixed this by cleaning the system Environment Variables and forcing execution via the py -3.11 launcher.

### Hardware Throttling:
I discovered that disconnecting the HDMI cable or allowing the system to sleep caused the GPU to downclock. Maintaining a physical display connection and disabling sleep mode was essential for sustained performance.

## Optimization & Results
Hardware Acceleration: By migrating from CPU-only training to AMD GPU parallel computing, the training speed per step improved by 53% compared to the first run.

### Parallelism Strategy:
Increased the batch_size to 128, effectively saturating the GPU VRAM bandwidth and reducing the total training time for 300M+ tokens.

### Model Refinement:
Through BPE vocabulary optimization and learning rate (LR) cosine scheduling, the model's convergence improved significantly, reaching a final loss of 1.5.

# 作业一 - 从零开始的Transformer
## 项目概述
本项目实现了从零开始的 Transformer 语言模型构建，并在 TinyStories 数据集上进行了训练。项目包含完整的 BPE 分词器实现与模型架构开发。依托 AMD 硬件平台（Ryzen 7 8745H 与 Radeon RX 7700 XT），通过三次重大的训练迭代，成功将模型 Loss 从 1.8 优化至 1.5。

## 踩坑经历与解决方案
版本兼容性问题：最初尝试使用 Python 3.14 预览版，导致 torch-directml、jaxtyping 等关键库无法安装。最终通过回退至 Python 3.11.9 稳定版解决了环境问题。

### AMD GPU 驱动适配
A 卡无法直接使用 CUDA，导致初期报错。通过引入 DirectML 插件，成功打通了 PyTorch 与 AMD 显卡之间的硬件加速通道。

### 环境变量冲突：
多版本 Python 并存导致 pip 安装路径混乱。通过手动清理 PATH 环境变量并强制使用 py -3.11 -m pip 指令，确保了依赖库的精准安装。

### 硬件性能限制：
实验发现拔掉 HDMI 线或系统休眠会导致显卡进入节能模式，大幅降低算力。通过保持物理连接和关闭自动休眠，确保了 GPU 的全力输出。

## 优化策略与成果
硬件加速优化：从 CPU 串行训练切换至 AMD GPU 并行训练。利用显卡数千个核心处理矩阵运算，使平均每步训练速度比初期提升了 53%。

### 并行度提升：
将 batch_size 调整至 128。较大的 Batch Size 充分利用了显存带宽，显著提高了数据吞吐量。

### 超参数调优：
结合 BPE 词表优化与余弦退火学习率策略，模型生成的文本逻辑性显著增强，Loss 最终稳定在 1.5。