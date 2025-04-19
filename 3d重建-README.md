# 3D场景重建项目

这是一个基于多视图图像的3D场景重建项目。项目使用COLMAP作为主要的重建引擎，结合Open3D进行点云处理和可视化。通过GPU加速和多线程并行处理，实现了高效的3D场景重建流程。

## 项目概述

本项目实现了从多视角图像到3D模型的完整重建流程，包括图像预处理、特征提取与匹配、相机位姿估计、稀疏重建、稠密重建以及可视化等步骤。项目采用模块化设计，各个组件可以独立工作，也可以组合成完整的重建管线。

## 技术栈

- **COLMAP**: 用于Structure from Motion (SfM)和Multi-View Stereo (MVS)的开源重建引擎
- **Open3D**: 用于点云处理、网格重建和3D可视化
- **OpenCV**: 用于图像处理和特征提取
- **PyTorch**: 用于GPU加速的图像处理和深度学习任务
- **NumPy**: 用于高效的数值计算
- **CUDA**: 用于GPU加速（如果可用）

## 项目结构

```
.
├── configs/              # 配置文件目录
│   └── config.yaml      # 主配置文件，包含所有参数设置
├── data/                 # 数据目录
│   ├── raw/             # 原始图像数据
│   ├── output/          # 输出结果目录
│   │   ├── preprocessed/ # 预处理后的图像
│   │   ├── reconstruction/ # 重建结果
│   │   └── screenshots/ # 可视化截图
│   └── colmap_workspace/ # COLMAP工作目录
├── src/                  # 源代码目录
│   ├── preprocessing/   # 数据预处理模块
│   │   └── preprocess.py # 图像预处理脚本
│   ├── reconstruction/  # 3D重建核心模块
│   │   └── reconstruct.py # 3D重建脚本
│   ├── visualization/   # 可视化模块
│   │   └── visualize.py # 可视化脚本
│   └── utils/           # 工具函数
│       └── config.py    # 配置加载工具
└── requirements.txt      # 项目依赖
```

## 技术原理

### 1. 图像预处理

预处理模块对输入图像进行一系列优化，提高重建质量：

- **图像缩放**: 将图像调整到合适的分辨率，平衡计算效率和重建质量
- **噪声消除**: 使用高斯滤波或双边滤波减少图像噪声
- **对比度增强**: 通过直方图均衡化提高图像对比度
- **色彩平衡**: 校正图像色彩，确保一致性
- **GPU加速**: 支持使用CUDA加速图像处理

### 2. 3D重建流程

重建过程分为以下几个主要步骤：

1. **特征提取与匹配**:
   - 使用SIFT算法提取图像特征点
   - 通过特征匹配建立图像间的对应关系

2. **稀疏重建**:
   - 相机位姿估计（Structure from Motion）
   - 三角测量生成稀疏点云
   - 捆绑调整优化相机参数和3D点位置

3. **稠密重建**:
   - 多视图立体匹配生成深度图
   - 深度图融合生成稠密点云
   - 泊松表面重建生成网格模型

4. **纹理映射**:
   - 将原始图像投影到网格模型上
   - 生成具有真实纹理的3D模型

### 3. 可视化功能

可视化模块提供多种查看和交互方式：

- **点云可视化**: 支持不同渲染模式（默认、法线、深度）
- **网格模型显示**: 支持线框、平面和纹理显示模式
- **交互式控制**: 支持旋转、缩放、平移等操作
- **截图功能**: 可保存当前视图为图像文件
- **自定义设置**: 可调整点大小、背景颜色、光照效果等

## 安装依赖

项目依赖以下主要库和工具：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- numpy, opencv-python, scipy, matplotlib
- open3d (>=0.15.0)
- colmap, pycolmap
- torch, torchvision (GPU加速)
- tqdm, pillow, scikit-image
- pyyaml, logging, psutil

## 使用方法

### 基本使用流程

1. 将原始图像放置在 `data/raw` 目录下
2. 运行预处理脚本：
   ```bash
   python src/preprocessing/preprocess.py
   ```
3. 运行重建脚本：
   ```bash
   python src/reconstruction/reconstruct.py
   ```
4. 查看结果：
   ```bash
   python src/visualization/visualize.py
   ```

### 配置参数说明

所有配置参数都在 `configs/config.yaml` 文件中，可以根据需要进行调整：

#### 预处理参数

- `image_size`: 图像缩放尺寸，如 [1920, 1080]
- `max_features`: 每张图像提取的最大特征点数，默认10000
- `use_gpu`: 是否使用GPU加速，默认true
- `batch_size`: 批处理大小，默认4
- `noise_reduction`: 是否进行噪声消除，默认true
- `contrast_enhancement`: 是否进行对比度增强，默认true

#### 重建参数

- `use_gpu`: 是否使用GPU加速，默认true
- `num_threads`: 使用的线程数，默认8
- 稀疏重建参数：
  - `min_num_matches`: 最小匹配点数，默认15
  - `max_reproj_error`: 最大重投影误差，默认4.0
- 稠密重建参数：
  - `patch_size`: 补丁大小，默认11
  - `window_radius`: 窗口半径，默认5

#### 可视化参数

- `point_size`: 点云显示点大小，默认1.0
- `background_color`: 背景颜色，默认[0, 0, 0]（黑色）
- `mesh_simplification`: 网格简化比例，默认0.5
- `interactive`: 是否启用交互式可视化，默认true

## 性能优化

项目实现了多项性能优化措施：

1. **GPU加速**：
   - 预处理阶段使用PyTorch进行GPU加速
   - 重建阶段使用COLMAP的GPU加速功能
   - 可视化阶段使用Open3D的GPU渲染

2. **多线程并行**：
   - 图像批处理并行化
   - 特征提取与匹配的多线程处理
   - 稠密重建的并行计算

3. **内存优化**：
   - 批处理机制减少内存占用
   - 点云下采样减少数据量
   - 网格简化控制模型复杂度

## 注意事项

- 确保原始图像质量良好，有足够的重叠区域（建议30%-60%）
- 建议使用高分辨率图像以获得更好的重建效果
- 重建过程可能需要较长时间，取决于图像数量和分辨率
- GPU加速需要CUDA支持，请确保安装了兼容的CUDA版本
- 对于大规模场景，可能需要调整内存相关参数

## 高级功能

- **自动相机校准**：自动估计相机内参和畸变参数
- **增量式重建**：支持逐步添加新图像到现有重建中
- **多分辨率处理**：根据计算资源自动调整处理分辨率
- **异常检测与处理**：自动识别并处理异常图像和匹配
- **交互式编辑**：支持手动调整重建结果

## 输出结果

成功运行后，可在以下位置找到结果：

- 预处理图像：`data/output/preprocessed/`
- 稀疏点云：`data/output/reconstruction/sparse.ply`
- 稠密点云：`data/output/reconstruction/dense.ply`
- 网格模型：`data/output/reconstruction/mesh.ply`
- 纹理模型：`data/output/reconstruction/textured_mesh.obj`
- 可视化截图：`data/output/screenshots/`